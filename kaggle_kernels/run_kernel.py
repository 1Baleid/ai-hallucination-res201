"""Push a bundled kernel, poll its status, download its outputs and print a short summary.

Usage (from the repository root, after ``build_kernels.py``)::

    python kaggle_kernels/run_kernel.py hash  --username <kaggle-user>
    python kaggle_kernels/run_kernel.py match --username <kaggle-user> --timeout-min 240
    python kaggle_kernels/run_kernel.py export --username <kaggle-user> --no-push --version 3

Steps: (1) check the build folder ``kaggle_kernels/build/<slug>/`` (``kernel.py`` and a
``kernel-metadata.json`` that is private with internet off; anything else is refused); (2) check that
the download folder ``data/kaggle/<slug>/`` is git-ignored and holds no tracked file (otherwise
refuse: Kaggle outputs such as the mapping table must never reach git); (3) ``kaggle kernels push``;
(4) ``kaggle kernels status`` every ``--poll-s`` seconds until complete, error or cancelled, or until
``--timeout-min``; (5) ``kaggle kernels output`` into the download folder (also on error: the log is
useful); (6) print a summary of the downloaded files. The Kaggle CLI is always called through
``subprocess`` (importing the ``kaggle`` package would start authentication); credentials come from
the usual ``kaggle.json`` or environment variables.

Exit codes: 0 complete, 1 the kernel ended in error, was cancelled or timed out, 2 refused or failed
before or while pushing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cxrdev.constants import KERNEL_SLUGS  # noqa: E402

DEFAULT_BUILD = REPO_ROOT / "kaggle_kernels" / "build"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "kaggle"
TERMINAL_STATUSES = ("complete", "error", "cancel_acknowledged")
Runner = Callable[[Sequence[str]], tuple]


class RefusedError(RuntimeError):
    """An action was refused because it could leak data or push the wrong thing."""


# --------------------------------------------------------------------------- CLI plumbing
def kaggle_exe() -> str:
    """Path of the Kaggle CLI next to this interpreter (the project venv), else ``kaggle`` on PATH."""
    scripts = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        cand = scripts / name
        if cand.is_file():
            return str(cand)
    return shutil.which("kaggle") or "kaggle"


def default_runner(cmd: Sequence[str], timeout: float = 3600.0) -> tuple[int, str]:
    """Run a command; return (exit code, stdout + stderr). A missing executable gives code 127."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, env=env)
    except FileNotFoundError as exc:
        return 127, f"{exc}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout} s: {' '.join(cmd)}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def kernel_ref(username: str, slug: str, version: int | None = None) -> str:
    """``<user>/<slug>`` or ``<user>/<slug>/<version>``."""
    return f"{username}/{slug}" + (f"/{int(version)}" if version else "")


def push_cmd(folder: Path, exe: str | None = None) -> list[str]:
    return [exe or kaggle_exe(), "kernels", "push", "-p", str(folder)]


def status_cmd(ref: str, exe: str | None = None) -> list[str]:
    return [exe or kaggle_exe(), "kernels", "status", ref]


def output_cmd(ref: str, dest: Path, exe: str | None = None) -> list[str]:
    return [exe or kaggle_exe(), "kernels", "output", ref, "-p", str(dest), "-o"]


def parse_status(text: str) -> str | None:
    """Normalised status from ``kaggle kernels status`` output.

    Kaggle CLI 2.x prints ``<ref> has status "KernelWorkerStatus.COMPLETE"``; older versions print
    ``"complete"``. Returns e.g. "complete", "running", "error", "queued", "cancel_acknowledged".
    """
    m = re.search(r'has status "([^"]+)"', text or "")
    if m:
        return m.group(1).strip().split(".")[-1].strip().lower()
    m = re.search(r"\b(queued|running|complete|error|cancel_requested|cancel_acknowledged|new_script)\b",
                  text or "", re.I)
    return m.group(1).lower() if m else None


def parse_push_version(text: str) -> int | None:
    """Version number from ``Kernel version N successfully pushed``."""
    m = re.search(r"Kernel version (\d+) successfully", text or "")
    return int(m.group(1)) if m else None


def push_failed(code: int, text: str) -> bool:
    """True when the push command failed (non-zero code, an error message, or no success line)."""
    low = (text or "").lower()
    if code != 0 or "kernel push error" in low:
        return True
    return "error" in low and "successfully" not in low


# --------------------------------------------------------------------------- safety checks
def _git_runner(cmd: Sequence[str]) -> tuple[int, str]:
    return default_runner(cmd, timeout=60)


def check_local_target(path: Path, repo_root: Path = REPO_ROOT, git: Runner | None = None) -> None:
    """Refuse a local folder inside the repository unless it is git-ignored and holds no tracked file.

    Folders outside the repository are allowed. Raises RefusedError (also when git cannot be run).
    """
    target = Path(path).resolve()
    root = Path(repo_root).resolve()
    try:
        rel = target.relative_to(root)
    except ValueError:
        return
    rel_s = rel.as_posix() if str(rel) != "." else "."
    git = git or _git_runner
    code, out = git(["git", "-C", str(root), "ls-files", "--", rel_s])
    if code != 0:
        raise RefusedError(f"cannot check {path} with git (code {code}): {out.strip()[:200]}")
    if out.strip():
        raise RefusedError(f"refusing {path}: it holds git-tracked files")
    code, out = git(["git", "-C", str(root), "check-ignore", "-q", "--", rel_s])
    if code == 1:
        raise RefusedError(f"refusing {path}: the folder is not git-ignored (use a folder under data/)")
    if code != 0:
        raise RefusedError(f"cannot check {path} with git check-ignore (code {code}): {out.strip()[:200]}")


def check_build_folder(folder: Path, username: str, slug: str) -> dict:
    """Validate ``kernel-metadata.json`` before a push (private, internet off, right id, code present)."""
    meta_path = Path(folder) / "kernel-metadata.json"
    if not meta_path.is_file():
        raise RefusedError(f"{meta_path} not found: run kaggle_kernels/build_kernels.py first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("is_private") is not True:
        raise RefusedError(f"{meta_path}: is_private must be true")
    if meta.get("enable_internet") is not False:
        raise RefusedError(f"{meta_path}: enable_internet must be false")
    if meta.get("id") != f"{username}/{slug}":
        raise RefusedError(f"{meta_path}: id {meta.get('id')!r} is not {username}/{slug}")
    code = Path(folder) / str(meta.get("code_file", ""))
    if not code.is_file():
        raise RefusedError(f"code file {code} not found")
    return meta


# --------------------------------------------------------------------------- steps
def poll_status(runner: Runner, ref: str, timeout_s: float, poll_s: float, initial_delay_s: float = 0.0,
                sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                log: Callable[[str], None] = print, exe: str | None = None, max_failures: int = 5) -> str:
    """Poll until a terminal status; returns it, "timeout", or "status_failed" after repeated CLI errors."""
    if initial_delay_s > 0:
        sleep(initial_delay_s)
    start = clock()
    last = None
    failures = 0
    while True:
        code, out = runner(status_cmd(ref, exe))
        status = parse_status(out) if code == 0 else None
        if status is None:
            failures += 1
            log(f"{ref}: status unavailable (code {code}): {out.strip()[:200]}")
            if failures >= max_failures:
                return "status_failed"
        else:
            failures = 0
            if status != last:
                log(f"{ref}: {status}")
                last = status
            if status in TERMINAL_STATUSES:
                return status
        if clock() - start >= timeout_s:
            return "timeout"
        sleep(poll_s)


def summarize(kind: str, dest: Path) -> str:
    """Short text summary of a download folder (aggregates only; no UID is printed)."""
    dest = Path(dest)
    lines = [f"downloaded into {dest}:"]
    files = sorted(p for p in dest.iterdir() if p.is_file()) if dest.is_dir() else []
    for p in files:
        lines.append(f"  {p.name:32s} {p.stat().st_size:>12,d} bytes")
    try:
        if kind == "hash":
            rs = dest / "run_summary.json"
            if rs.is_file():
                s = json.loads(rs.read_text(encoding="utf-8"))
                folders = s.get("folders", {})
                done = sorted(k for k, v in folders.items() if v.get("status") == "done")
                failed = sorted(k for k, v in folders.items() if v.get("status") == "failed")
                lines.append(f"folders done {len(done)}, failed {failed or 'none'}, "
                             f"images hashed {s.get('n_hashed_total')}")
                lines.append("steps: " + ", ".join(f"{k}={v.get('status')}" for k, v in s.get("steps", {}).items()))
            yt = dest / "yield_table.csv"
            if yt.is_file():
                with open(yt, newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                lines.append("yield: " + ", ".join(f"{r['metric']}={r['value']}" for r in rows[:12]))
            dk = dest / "dimkey_test.json"
            if dk.is_file():
                d = json.loads(dk.read_text(encoding="utf-8"))
                lines.append(f"dimension key: share found {d.get('share_found')} of {d.get('n_read')} JPEGs")
        elif kind == "match":
            ms = dest / "match_stats.json"
            if ms.is_file():
                s = json.loads(ms.read_text(encoding="utf-8"))
                lines.append(f"status counts {s.get('status_counts')}; match rate {s.get('match_rate')}; "
                             f"cut {s.get('cut')} ({s.get('cut_rule')}); ambiguous within patient "
                             f"{s.get('ambiguous_share_within_patient')}")
            n_grids = len([p for p in files if p.name.startswith("spotcheck_")])
            lines.append(f"spot-check grids: {n_grids} (review all of them before using the mapping)")
        elif kind == "export":
            man = dest / "manifest.csv"
            zp = dest / "selected_nih_png.zip"
            if man.is_file():
                with open(man, newline="", encoding="utf-8") as fh:
                    n_rows = sum(1 for _ in csv.DictReader(fh))
                lines.append(f"manifest rows {n_rows}")
                if zp.is_file():
                    with zipfile.ZipFile(zp) as zf:
                        n_entries = len(zf.namelist())
                    lines.append(f"zip entries {n_entries}" + ("" if n_entries == n_rows else " (MISMATCH)"))
    except Exception as exc:  # the summary must never hide the download
        lines.append(f"(summary incomplete: {type(exc).__name__}: {exc})")
    return "\n".join(lines)


def run(kind: str, username: str, build_dir: Path = DEFAULT_BUILD, out_root: Path = DEFAULT_OUT_ROOT,
        repo_root: Path = REPO_ROOT, runner: Runner = default_runner, push: bool = True, wait: bool = True,
        download: bool = True, version: int | None = None, timeout_s: float = 6 * 3600, poll_s: float = 60,
        initial_delay_s: float = 30, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = print,
        exe: str | None = None, git: Runner | None = None) -> int:
    """Push, poll and download one kernel; returns the exit code described in the module docstring."""
    if kind not in KERNEL_SLUGS:
        raise ValueError(f"unknown kernel {kind!r}")
    slug = KERNEL_SLUGS[kind]
    folder = Path(build_dir) / slug
    dest = Path(out_root) / slug
    try:
        if push:
            check_build_folder(folder, username, slug)
        if download:
            check_local_target(dest, repo_root, git)
    except RefusedError as exc:
        log(f"refused: {exc}")
        return 2
    if push:
        code, out = runner(push_cmd(folder, exe))
        log(out.strip())
        if push_failed(code, out):
            log(f"push failed (code {code})")
            return 2
        version = parse_push_version(out) or version
    ref = kernel_ref(username, slug, version)
    status = "not_polled"
    if wait:
        status = poll_status(runner, ref, timeout_s, poll_s, initial_delay_s if push else 0.0, sleep, clock, log, exe)
        log(f"{ref}: final status {status}")
    if download and status not in ("timeout", "status_failed"):
        dest.mkdir(parents=True, exist_ok=True)
        code, out = runner(output_cmd(ref, dest, exe))
        if out.strip():
            log(out.strip()[-2000:])
        if code != 0:
            log(f"download failed (code {code})")
            return 1
        log(summarize(kind, dest))
    if status in ("complete", "not_polled"):
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Push, poll and download a private RES201 Kaggle kernel.")
    ap.add_argument("kernel", choices=sorted(KERNEL_SLUGS))
    ap.add_argument("--username", required=True)
    ap.add_argument("--build-dir", default=str(DEFAULT_BUILD))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="downloads go to <out-root>/<slug>/")
    ap.add_argument("--no-push", action="store_true", help="do not push; poll/download the latest or --version")
    ap.add_argument("--no-wait", action="store_true", help="do not poll the status")
    ap.add_argument("--no-download", action="store_true", help="do not download the outputs")
    ap.add_argument("--version", type=int, default=None, help="kernel version to poll/download")
    ap.add_argument("--timeout-min", type=float, default=360.0)
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--initial-delay-s", type=float, default=30.0)
    args = ap.parse_args(argv)
    return run(args.kernel, args.username, Path(args.build_dir), Path(args.out_root), push=not args.no_push,
               wait=not args.no_wait, download=not args.no_download, version=args.version,
               timeout_s=args.timeout_min * 60.0, poll_s=args.poll_s, initial_delay_s=args.initial_delay_s)


if __name__ == "__main__":
    sys.exit(main())
