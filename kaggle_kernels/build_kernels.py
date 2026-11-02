"""Bundle the three private Kaggle kernels into self-contained scripts with their metadata.

Usage (from the repository root)::

    python kaggle_kernels/build_kernels.py --username <kaggle-user> [--out kaggle_kernels/build]
        [--kernels hash,match,export] [--hash-folders images_005,images_009] [--match-config '{"cut": 30}']

For each kernel the bundler reads ``kaggle_kernels/<name>_kernel/kernel_main.py``, follows its
single-line ``from cxrdev.<module> import <names>`` statements (and those of the modules it pulls in),
and writes ``<out>/<slug>/kernel.py``: one ``from __future__ import annotations`` line, then every
needed ``cxrdev`` module in dependency order with its ``cxrdev`` import lines removed, then the kernel
main (its ``# BUNDLE-STRIP-START`` .. ``# BUNDLE-STRIP-END`` block, which only fixes ``sys.path`` for
local runs, is removed). Next to it, ``kernel-metadata.json`` makes the kernel a private CPU script
with internet off and the right competition, dataset and kernel sources.

Checks on every bundle: it compiles; it parses with Python 3.10 grammar (``ast`` feature_version, plus
a token check for f-strings that only Python 3.12+ accepts); no ``cxrdev`` import or name is left;
no top-level name is defined by two different source files (the files share one namespace).
The output is deterministic (no timestamps; the header lists the sha256 of every source file).
Slugs come from ``cxrdev.constants.KERNEL_SLUGS``; the Kaggle CLI 2.2.5 reads the metadata fields
written here (``kaggle kernels push -p <out>/<slug>``; see run_kernel.py).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import tokenize
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cxrdev.constants import FILELIST_DATASET_SLUG, KAGGLE_COMPETITION, KAGGLE_NIH_DATASET, KERNEL_SLUGS  # noqa: E402

KERNEL_DIRS = {"hash": "hash_kernel", "match": "match_kernel", "export": "export_kernel"}
TITLES = {"hash": "RES201 hash", "match": "RES201 match", "export": "RES201 export"}
DEFAULT_OUT = REPO_ROOT / "kaggle_kernels" / "build"
MIN_PYTHON = (3, 10)
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_CXRDEV_IMPORT = re.compile(r"^from cxrdev\.(\w+) import ([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*(?:#.*)?$")
_ANY_CXRDEV_IMPORT = re.compile(r"^\s*(?:from|import)\s+cxrdev\b")
_FUTURE = re.compile(r"^from __future__ import .+$")
_STRIP_START = "# BUNDLE-STRIP-START"
_STRIP_END = "# BUNDLE-STRIP-END"
_HASH_FOLDERS_LINE = re.compile(r"^FOLDERS_OVERRIDE = None\b.*$", re.M)
_MATCH_CONFIG_LINE = re.compile(r"^CONFIG_OVERRIDES = \{\}.*$", re.M)


class BundleError(ValueError):
    """A source file cannot be bundled safely."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_source(source: str, origin: str) -> tuple[str, list[str]]:
    """Remove ``__future__`` lines, strip blocks and ``from cxrdev.<m> import ...`` lines.

    Returns the stripped source and the ``cxrdev`` modules it imported. Raises BundleError for any
    other form of ``cxrdev`` import (aliases, ``import cxrdev``, parenthesised or indented imports).
    """
    out: list[str] = []
    deps: list[str] = []
    skipping = False
    for lineno, line in enumerate(source.splitlines(), 1):
        s = line.strip()
        if s == _STRIP_START:
            if skipping:
                raise BundleError(f"{origin}:{lineno}: nested {_STRIP_START}")
            skipping = True
            continue
        if s == _STRIP_END:
            if not skipping:
                raise BundleError(f"{origin}:{lineno}: {_STRIP_END} without a start")
            skipping = False
            continue
        if skipping:
            continue
        if _FUTURE.match(line):
            continue
        m = _CXRDEV_IMPORT.match(line)
        if m:
            deps.append(m.group(1))
            continue
        if _ANY_CXRDEV_IMPORT.match(line):
            raise BundleError(f"{origin}:{lineno}: unsupported cxrdev import (use one line "
                              f"'from cxrdev.<module> import <names>' at column 0, no aliases): {s}")
        out.append(line)
    if skipping:
        raise BundleError(f"{origin}: unterminated {_STRIP_START}")
    return "\n".join(out).strip("\n") + "\n", deps


def module_order(kind: str, repo_root: Path = REPO_ROOT) -> list[str]:
    """The ``cxrdev`` modules a kernel needs, every module after its dependencies.

    Topological order (Kahn) over the import closure of the kernel main; among modules that are
    ready at the same time, ``constants`` comes first and the rest alphabetically.
    """
    main_path = Path(repo_root) / "kaggle_kernels" / KERNEL_DIRS[kind] / "kernel_main.py"
    _, main_deps = strip_source(main_path.read_text(encoding="utf-8"), str(main_path))
    deps: dict[str, set[str]] = {}
    pending = sorted(set(main_deps))
    while pending:
        mod = pending.pop()
        if mod in deps:
            continue
        path = Path(repo_root) / "cxrdev" / f"{mod}.py"
        if not path.is_file():
            raise BundleError(f"cxrdev module not found: {path}")
        _, sub = strip_source(path.read_text(encoding="utf-8"), f"cxrdev/{mod}.py")
        deps[mod] = set(sub) - {mod}
        pending.extend(sorted(deps[mod] - set(deps)))
    order: list[str] = []
    remaining = {m: set(d) for m, d in deps.items()}
    while remaining:
        ready = sorted((m for m, d in remaining.items() if not d), key=lambda m: (m != "constants", m))
        if not ready:
            raise BundleError("import cycle among cxrdev modules: " + ", ".join(sorted(remaining)))
        mod = ready[0]
        order.append(mod)
        del remaining[mod]
        for d in remaining.values():
            d.discard(mod)
    return order


def _bindings(tree: ast.Module) -> tuple[set[str], dict[str, set[str]]]:
    """Top-level defined names and import aliases (alias -> imported targets), including if/try bodies."""
    defined: set[str] = set()
    imports: dict[str, set[str]] = {}

    def targets(node) -> list[str]:
        if isinstance(node, ast.Name):
            return [node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return [n for elt in node.elts for n in targets(elt)]
        return []

    def visit(body) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    defined.update(targets(t))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                defined.update(targets(node.target))
            elif isinstance(node, ast.Import):
                for a in node.names:
                    alias = a.asname or a.name.split(".")[0]
                    imports.setdefault(alias, set()).add(a.name if a.asname else a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    imports.setdefault(a.asname or a.name, set()).add(f"{node.module}.{a.name}")
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                visit(node.body)
                visit(getattr(node, "orelse", []))
            elif isinstance(node, ast.Try):
                visit(node.body)
                for h in node.handlers:
                    visit(h.body)
                visit(node.orelse)
                visit(node.finalbody)

    visit(tree.body)
    return defined, imports


def check_collisions(parts: list[tuple[str, str]]) -> None:
    """Raise BundleError when two source files bind the same top-level name differently."""
    owner: dict[str, str] = {}
    alias_owner: dict[str, tuple[str, frozenset]] = {}
    problems: list[str] = []
    for origin, src in parts:
        defined, imports = _bindings(ast.parse(src))
        for name in sorted(defined):
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in owner and owner[name] != origin:
                problems.append(f"{name!r} defined in {owner[name]} and {origin}")
            owner.setdefault(name, origin)
        for alias, tgts in sorted(imports.items()):
            key = frozenset(tgts)
            if alias in alias_owner and alias_owner[alias][1] != key:
                problems.append(f"import alias {alias!r} means {sorted(alias_owner[alias][1])} in "
                                f"{alias_owner[alias][0]} but {sorted(key)} in {origin}")
            alias_owner.setdefault(alias, (origin, key))
    for name, origin in owner.items():
        if name in alias_owner and alias_owner[name][0] != origin:
            problems.append(f"{name!r} defined in {origin} but imported in {alias_owner[name][0]}")
    if problems:
        raise BundleError("name collisions in the bundle: " + "; ".join(problems))


def _quote_of(token_text: str) -> tuple[str, bool]:
    body = token_text.lstrip("rRbBuUfF")
    return body[0], body[:3] in ('"""', "'''")


def check_python_compat(source: str, version: tuple[int, int] = MIN_PYTHON) -> list[str]:
    """Problems that would stop ``source`` from running on Python ``version`` (best effort)."""
    problems: list[str] = []
    try:
        ast.parse(source, feature_version=version)
    except SyntaxError as exc:
        problems.append(f"syntax not accepted by Python {version[0]}.{version[1]}: line {exc.lineno}: {exc.msg}")
    fstart = getattr(tokenize, "FSTRING_START", None)
    if fstart is None or version >= (3, 12):
        return problems
    fend = tokenize.FSTRING_END
    stack: list[tuple[str, bool]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == fstart:
                q = _quote_of(tok.string)
                if any(q[0] == outer[0] and (not outer[1] or q[1]) for outer in stack):
                    problems.append(f"line {tok.start[0]}: nested f-string reuses the enclosing quote")
                stack.append(q)
            elif tok.type == fend:
                if stack:
                    stack.pop()
            elif stack and tok.type == tokenize.STRING:
                q = _quote_of(tok.string)
                if any(q[0] == outer[0] and (not outer[1] or q[1]) for outer in stack):
                    problems.append(f"line {tok.start[0]}: string inside an f-string reuses the enclosing quote")
                if "\\" in tok.string:
                    problems.append(f"line {tok.start[0]}: backslash inside an f-string expression")
            elif stack and tok.type == tokenize.COMMENT:
                problems.append(f"line {tok.start[0]}: comment inside an f-string expression")
    except (tokenize.TokenError, SyntaxError) as exc:
        problems.append(f"tokenize failed: {exc}")
    return problems


def validate_bundle(source: str, name: str = "kernel.py") -> None:
    """Compile and check a bundle; raise BundleError on any problem."""
    try:
        tree = compile(source, name, "exec", flags=ast.PyCF_ONLY_AST, dont_inherit=True)
        compile(source, name, "exec", dont_inherit=True)
    except SyntaxError as exc:
        raise BundleError(f"{name} does not compile: line {exc.lineno}: {exc.msg}") from None
    problems = check_python_compat(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "cxrdev" for a in node.names):
            problems.append(f"line {node.lineno}: import cxrdev left in the bundle")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "cxrdev":
            problems.append(f"line {node.lineno}: from cxrdev import left in the bundle")
        elif isinstance(node, ast.Name) and node.id == "cxrdev":
            problems.append(f"line {node.lineno}: name cxrdev used in the bundle")
    if _STRIP_START in source or _STRIP_END in source:
        problems.append("strip markers left in the bundle")
    if problems:
        raise BundleError(f"{name}: " + "; ".join(problems))


def _apply_overrides(kind: str, source: str, hash_folders: list[str] | None, match_config: dict | None) -> str:
    if kind == "hash" and hash_folders:
        if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", f) for f in hash_folders):
            raise BundleError(f"invalid folder names: {hash_folders}")
        source, n = _HASH_FOLDERS_LINE.subn(f"FOLDERS_OVERRIDE = {sorted(set(hash_folders))!r}", source)
        if n != 1:
            raise BundleError("FOLDERS_OVERRIDE line not found exactly once in the hash kernel")
    if kind == "match" and match_config:
        from cxrdev.matching import MatchConfig
        allowed = {f.name for f in fields(MatchConfig)}
        unknown = sorted(set(match_config) - allowed)
        if unknown:
            raise BundleError(f"unknown MatchConfig fields {unknown}; allowed {sorted(allowed)}")
        literal = repr({k: match_config[k] for k in sorted(match_config)})
        ast.literal_eval(literal)
        source, n = _MATCH_CONFIG_LINE.subn(lambda _m: f"CONFIG_OVERRIDES = {literal}", source)
        if n != 1:
            raise BundleError("CONFIG_OVERRIDES line not found exactly once in the match kernel")
    return source


def bundle_kernel(kind: str, repo_root: Path = REPO_ROOT, hash_folders: list[str] | None = None,
                  match_config: dict | None = None) -> str:
    """Return the bundled ``kernel.py`` text for ``kind`` (hash, match or export)."""
    if kind not in KERNEL_DIRS:
        raise ValueError(f"unknown kernel {kind!r}; choose from {sorted(KERNEL_DIRS)}")
    repo_root = Path(repo_root)
    parts: list[tuple[str, str, str]] = []
    for mod in module_order(kind, repo_root):
        raw = (repo_root / "cxrdev" / f"{mod}.py").read_text(encoding="utf-8")
        stripped, _ = strip_source(raw, f"cxrdev/{mod}.py")
        parts.append((f"cxrdev/{mod}.py", _sha256_text(raw), stripped))
    main_rel = f"kaggle_kernels/{KERNEL_DIRS[kind]}/kernel_main.py"
    raw = (repo_root / main_rel).read_text(encoding="utf-8")
    stripped, _ = strip_source(raw, main_rel)
    stripped = _apply_overrides(kind, stripped, hash_folders, match_config)
    parts.append((main_rel, _sha256_text(raw), stripped))
    check_collisions([(origin, src) for origin, _, src in parts])
    slug = KERNEL_SLUGS[kind]
    lines = [f'"""{slug}: self-contained private Kaggle kernel generated by kaggle_kernels/build_kernels.py.',
             "",
             "Do not edit this file; edit the sources below and rebuild. Sources (sha256):"]
    lines += [f"  {origin}  {digest}" for origin, digest, _ in parts]
    lines += ['"""', "from __future__ import annotations", ""]
    chunks = ["\n".join(lines)]
    for origin, digest, src in parts:
        bar = "=" * 24
        chunks.append(f"\n# {bar} {origin} (sha256 {digest[:16]}) {bar}\n{src}")
    bundle = "\n".join(chunks)
    validate_bundle(bundle, f"{slug}/kernel.py")
    return bundle


def kernel_metadata(kind: str, username: str) -> dict:
    """``kernel-metadata.json`` content for one kernel (private, CPU, internet off)."""
    if not USERNAME_RE.match(username or ""):
        raise ValueError(f"invalid Kaggle username: {username!r}")
    slug = KERNEL_SLUGS[kind]
    datasets = [KAGGLE_NIH_DATASET]
    if kind == "export":
        datasets.append(f"{username}/{FILELIST_DATASET_SLUG}")
    return {
        "id": f"{username}/{slug}",
        "title": TITLES[kind],
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": datasets,
        "competition_sources": [KAGGLE_COMPETITION] if kind in ("hash", "match") else [],
        "kernel_sources": [f"{username}/{KERNEL_SLUGS['hash']}"] if kind == "match" else [],
        "model_sources": [],
    }


def build_all(username: str, out_dir: Path | str = DEFAULT_OUT, kinds=("hash", "match", "export"),
              repo_root: Path = REPO_ROOT, hash_folders: list[str] | None = None,
              match_config: dict | None = None) -> dict[str, Path]:
    """Write ``<out_dir>/<slug>/kernel.py`` and ``kernel-metadata.json`` for each kernel; returns the folders."""
    out_dir = Path(out_dir)
    built: dict[str, Path] = {}
    for kind in kinds:
        meta = kernel_metadata(kind, username)
        source = bundle_kernel(kind, repo_root, hash_folders=hash_folders, match_config=match_config)
        folder = out_dir / KERNEL_SLUGS[kind]
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "kernel.py", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(source)
        with open(folder / "kernel-metadata.json", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(meta, indent=2) + "\n")
        built[kind] = folder
    return built


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bundle the private Kaggle kernels (res201-hash, -match, -export).")
    ap.add_argument("--username", required=True, help="Kaggle username that owns the kernels")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (default kaggle_kernels/build)")
    ap.add_argument("--kernels", default="hash,match,export", help="comma-separated subset of hash,match,export")
    ap.add_argument("--hash-folders", default=None, help="hash only these NIH folders (comma-separated)")
    ap.add_argument("--match-config", default=None, help='JSON MatchConfig overrides, e.g. \'{"cut": 30}\'')
    args = ap.parse_args(argv)
    kinds = [k.strip() for k in args.kernels.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in KERNEL_DIRS]
    if unknown:
        ap.error(f"unknown kernels {unknown}")
    folders = [f.strip() for f in args.hash_folders.split(",") if f.strip()] if args.hash_folders else None
    config = json.loads(args.match_config) if args.match_config else None
    if config is not None and not isinstance(config, dict):
        ap.error("--match-config must be a JSON object")
    try:
        built = build_all(args.username, args.out, kinds, hash_folders=folders, match_config=config)
    except (BundleError, ValueError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 2
    for kind, folder in built.items():
        size = (folder / "kernel.py").stat().st_size
        print(f"{kind:6s} -> {folder} (kernel.py {size} bytes)")
    print("next: python kaggle_kernels/run_kernel.py <hash|match|export> --username " + args.username)
    return 0


if __name__ == "__main__":
    sys.exit(main())
