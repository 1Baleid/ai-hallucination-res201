"""Tests for tools/reading_core.py and the thin GUI in tools/read_tool.py.

Synthetic data only: tiny generated PNGs and made-up read_ids and filenames (no RANZCR
data). The GUI tests drive ReaderApp through its key handler with fake dialogs and skip
when Tk cannot start.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cxrdev.constants import derive_seed
from tools import reading_core as rc

REPO_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 11, 2, 9, 0, 0, tzinfo=timezone.utc)
ISO_Z = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")


# ---------------------------------------------------------------- helpers
class FakeClock:
    """Monotonic clock under test control."""

    def __init__(self, start: float = 100.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeNow:
    """UTC wall clock under test control."""

    def __init__(self, start: datetime = T0) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kwargs: float) -> None:
        self.t += timedelta(**kwargs)


def _png(path: Path, value: int = 90, size: tuple[int, int] = (12, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0]), value, dtype=np.uint8)
    arr[:, size[0] // 2] = 250
    Image.fromarray(arr).save(path)
    return path


def make_list(
    folder: Path,
    n: int = 6,
    *,
    name: str = "reading_list.csv",
    seed: int | None = 2026101205,
    sidecar: str | None = "stem",
    missing: tuple[int, ...] = (),
    relative: bool = True,
) -> tuple[Path, list[dict[str, str]]]:
    """Write a synthetic S10 reading list (rows deliberately not sorted) plus images."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(derive_seed("ids"))
    read_ids: list[str] = []
    while len(read_ids) < n:
        candidate = f"{int(rng.integers(0, 2 ** 32)):08X}"
        if candidate not in read_ids:
            read_ids.append(candidate)
    numbers = rng.permutation(9000)[:n] + 1000
    rows = []
    for k in range(n):
        nih = f"{int(numbers[k]):08d}_{k % 4:03d}.png"
        image = folder / "images" / nih
        if k not in missing:
            _png(image, value=40 + 10 * (k % 10))
        rows.append({"read_id": read_ids[k], "nih_filename": nih,
                     "image_path": f"images/{nih}" if relative else str(image)})
    list_path = folder / name
    with open(list_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rc.LIST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if sidecar is not None and seed is not None:
        side = list_path.with_suffix(".meta.json") if sidecar == "stem" else list_path.with_name(name + ".meta.json")
        side.write_text(json.dumps({"seed": seed}), encoding="utf-8")
    return list_path, rows


def new_session(list_path: Path, out: Path, mode: str = "read", **kwargs):
    clock = kwargs.pop("clock", FakeClock())
    now = kwargs.pop("now", FakeNow())
    return rc.ReadingSession(list_path, out, mode, clock=clock, now=now, **kwargs), clock, now


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finalised_session(folder: Path, n: int = 3, mode: str = "read", now: FakeNow | None = None, **list_kwargs):
    list_path, rows = make_list(folder, n=n, **list_kwargs)
    out = folder / ("verified_items.csv" if mode == "read" else "retest.csv")
    session, clock, now = new_session(list_path, out, mode, now=now or FakeNow())
    for _ in range(n):
        session.record("absent")
    return session, session.finalise(), list_path, rows, out


# ---------------------------------------------------------------- which files get opened (child process)
# A Python audit hook sees every file open (builtins.open, io.open, os.open, PIL). Audit hooks
# cannot be removed once added, and one left in the pytest process made later Tk start-ups
# fail intermittently, so the hook runs in a separate interpreter.
_CHILD_PRELUDE = """
import json, os, sys
from pathlib import Path
sys.path.insert(0, {root!r})
from tools import reading_core as rc
OPENED = []
def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        OPENED.append(os.fsdecode(args[0]))
sys.addaudithook(_hook)
"""


def run_in_child(body: str, *args: str):
    """Run ``body`` after the prelude in a fresh interpreter; return the JSON it prints last."""
    code = _CHILD_PRELUDE.format(root=str(REPO_ROOT)) + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-I", "-c", code, *args], capture_output=True, text=True,
                          timeout=180, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _inside(path: str, folder: Path) -> bool:
    return os.path.normcase(os.path.abspath(path)).startswith(os.path.normcase(str(folder)))


# ---------------------------------------------------------------- reading lists
def test_load_list_keeps_file_order_and_resolves_relative_paths(tmp_path):
    list_path, rows = make_list(tmp_path / "items", n=6)
    items = rc.load_reading_list(list_path)
    ids = [item.read_id for item in items]
    assert ids == [row["read_id"] for row in rows]
    assert ids != sorted(ids)                                    # the file order is not a sort order
    assert [item.nih_filename for item in items] != sorted(item.nih_filename for item in items)
    assert [item.order_index for item in items] == list(range(6))
    for item, row in zip(items, rows):
        assert item.nih_filename == row["nih_filename"]
        assert item.image_path.is_file() and item.image_path.name == row["nih_filename"]
    assert rows[0]["nih_filename"] not in repr(items[0])         # repr hides filenames and paths
    assert rc.load_reading_list(list_path, limit=3) == items[:3]
    with pytest.raises(ValueError):
        rc.load_reading_list(list_path, limit=0)


def test_absolute_image_paths_and_repo_relative_fallback(tmp_path, monkeypatch):
    list_path, rows = make_list(tmp_path / "items", n=2, relative=False)
    assert all(item.image_path.is_file() for item in rc.load_reading_list(list_path))
    # a path relative to the current folder (not to the list's folder) is found too
    monkeypatch.chdir(tmp_path)
    resolved = rc.resolve_image_path(f"items/images/{rows[0]['nih_filename']}", tmp_path / "elsewhere")
    assert resolved.is_file()
    # nothing found: the list-folder candidate is returned
    ghost = rc.resolve_image_path("nope/ghost.png", tmp_path / "items")
    assert ghost == (tmp_path / "items").resolve() / "nope" / "ghost.png" and not ghost.exists()


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("read_id,nih_filename\n0A1B2C3D,x.png\n", "must have the columns"),
        ("read_id,nih_filename,image_path,arm\n0A1B2C3D,x.png,a.png,B\n", "unexpected columns"),
        ("read_id,item_id,nih_filename,image_path\n0A1B2C3D,ab12,x.png,a.png\n", "unexpected columns"),
        ("read_id,nih_filename,image_path,read_id\n0A1B2C3D,x.png,a.png,0A1B2C3D\n", "must have the columns"),
        ("read_id,nih_filename,image_path\n0a1b2c3d,x.png,a.png\n", "8-char uppercase hex"),
        ("read_id,nih_filename,image_path\n0A1B2C3,x.png,a.png\n", "8-char uppercase hex"),
        ("read_id,nih_filename,image_path\n0A1B2C3D,x.png,a.png\n0A1B2C3D,y.png,b.png\n", "duplicate read_id"),
        ("read_id,nih_filename,image_path\n0A1B2C3D,x.png,a.png\n1A1B2C3D,x.png,b.png\n", "appears twice"),
        ("read_id,nih_filename,image_path\n0A1B2C3D,,a.png\n", "empty nih_filename"),
        ("read_id,nih_filename,image_path\n0A1B2C3D,x.png\n", "expected 3 fields"),
        ("read_id,nih_filename,image_path\n", "no rows"),
        ("", "is empty"),
    ],
)
def test_list_validation_errors(tmp_path, text, fragment):
    path = tmp_path / "reading_list.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(rc.ListError, match=re.escape(fragment)):
        rc.load_reading_list(path)


def test_list_errors_name_no_image_file(tmp_path):
    path = tmp_path / "reading_list.csv"
    path.write_text("read_id,nih_filename,image_path\n0A1B2C3D,00001234_000.png,a.png\n"
                    "1A1B2C3D,00001234_000.png,b.png\n", encoding="utf-8")
    with pytest.raises(rc.ListError) as info:
        rc.load_reading_list(path)
    assert "00001234" not in str(info.value)


def test_list_with_bom_crlf_and_blank_lines(tmp_path):
    path = tmp_path / "reading_list.csv"
    path.write_bytes("\ufeffread_id,nih_filename,image_path\r\n0A1B2C3D,x.png,a.png\r\n\r\n"
                     "1A1B2C3D,y.png,b.png\r\n".encode("utf-8"))
    assert [item.read_id for item in rc.load_reading_list(path)] == ["0A1B2C3D", "1A1B2C3D"]
    path.write_bytes(b"read_id,nih_filename,image_path\n\xff\xfe,x.png,a.png\n")
    with pytest.raises(rc.ListError, match="not UTF-8"):
        rc.load_reading_list(path)
    with pytest.raises(rc.ListError, match="must be a .csv"):
        rc.load_reading_list(tmp_path / "list.txt")
    with pytest.raises(rc.ListError, match="not found"):
        rc.load_reading_list(tmp_path / "absent_list.csv")


BLINDED_OR_OUTPUT_NAMES = [
    "reading_key.csv", "items.csv", "pod_items.csv", "parsed_results.csv", "detector_scores.csv",
    "exclusions.csv", "match_table.csv", "qwen3vl2b.jsonl", "verified_items.csv", "retest.csv",
]


def test_blinded_and_output_files_are_refused_without_being_opened(tmp_path):
    for name in BLINDED_OR_OUTPUT_NAMES:
        (tmp_path / name).write_text("read_id,nih_filename,image_path\n0A1B2C3D,x.png,a.png\n", encoding="utf-8")
    upper = tmp_path / "upper"
    upper.mkdir()
    (upper / "Reading_Key.CSV").write_text("read_id,item_id\n0A1B2C3D,ab\n", encoding="utf-8")
    result = run_in_child(
        """
        errors = {}
        for path in [Path(p) for p in json.loads(sys.argv[1])]:
            kinds = []
            for attempt in (lambda: rc.load_reading_list(path),
                            lambda: rc.ReadingSession(path, path.parent / "out.csv")):
                try:
                    attempt()
                    kinds.append("no error")
                except rc.ListError:
                    kinds.append("ListError")
                except Exception as exc:
                    kinds.append(type(exc).__name__)
            errors[path.name] = kinds
        print(json.dumps({"opened": OPENED, "errors": errors}))
        """,
        json.dumps([str(tmp_path / name) for name in BLINDED_OR_OUTPUT_NAMES] + [str(upper / "Reading_Key.CSV")]),
    )
    assert result["errors"] == {name: ["ListError", "ListError"]
                                for name in BLINDED_OR_OUTPUT_NAMES + ["Reading_Key.CSV"]}
    assert not [p for p in result["opened"] if _inside(p, tmp_path)]   # not one of them was opened


def test_session_never_opens_key_items_or_model_output(tmp_path):
    items_dir = tmp_path / "items"
    list_path, rows = make_list(items_dir, n=5)
    (items_dir / "reading_key.csv").write_text("read_id,item_id\n" + "".join(
        f"{row['read_id']},abcdef012{k}\n" for k, row in enumerate(rows)), encoding="utf-8")
    (items_dir / "items.csv").write_text("item_id,arm\nabcdef0120,B\n", encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "qwen3vl2b.jsonl").write_text("{}\n", encoding="utf-8")
    out = items_dir / "verified_items.csv"
    opened = run_in_child(
        """
        list_path, out = Path(sys.argv[1]), Path(sys.argv[2])
        first = rc.ReadingSession(list_path, out)
        first.load_image()
        first.record("present")
        first.load_image()
        first.record("absent", flags=["ambiguous"])
        del first                                            # crash
        second = rc.ReadingSession(list_path, out)
        while not second.complete:
            second.load_image()
            second.record("absent")
        second.view()
        second.summary_lines()
        second.finalise()
        print(json.dumps(OPENED))
        """,
        str(list_path), str(out),
    )
    assert rc.verify_output(out)["n"] == 5                       # the child really finalised
    names = {Path(p).name.lower() for p in opened}
    assert "reading_key.csv" not in names and "items.csv" not in names
    assert not [p for p in opened if p.lower().endswith(".jsonl")]
    own = {"reading_list.csv", "reading_list.meta.json", "reading_list.csv.meta.json",
           "verified_items.partial.csv", "verified_items.partial.csv.tmp",
           "verified_items.csv", "verified_items.csv.tmp",
           "verified_items.meta.json", "verified_items.meta.json.tmp"}
    images = {row["nih_filename"].lower() for row in rows}
    inside = {Path(p).name.lower() for p in opened if _inside(p, tmp_path)}
    assert inside <= own | images and images <= inside          # every image read, nothing else of ours


def test_view_and_summary_never_show_filenames_or_paths(tmp_path):
    list_path, rows = make_list(tmp_path / "items", n=4, missing=(2,))
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    texts: list[str] = list(session.summary_lines())
    for k in range(4):
        view = session.view()
        texts += [repr(view), view.progress]
        session.record("skip" if k == 2 else "present")
    texts += [repr(session.view()), session.view().progress, repr(session.items), repr(session.current_item)]
    blob = "\n".join(texts)
    for row in rows:
        assert row["nih_filename"] not in blob
        assert row["nih_filename"].split("_")[0] not in blob     # not even the patient prefix
    assert str(tmp_path / "items" / "images") not in blob
    assert "WARNING: 1 image file(s) are missing" in blob


# ---------------------------------------------------------------- answering
def test_reading_order_equals_list_order(tmp_path):
    folder = tmp_path / "items"
    folder.mkdir()
    ids = ["F0000001", "0000000A", "C0FFEE00", "00000002", "AB000000"]   # neither sorted nor reverse-sorted
    with open(folder / "reading_list.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(rc.LIST_COLUMNS)
        for k, read_id in enumerate(ids):
            name = f"{9 - k:08d}_000.png"
            _png(folder / "images" / name)
            writer.writerow([read_id, name, f"images/{name}"])
    out = folder / "verified_items.csv"
    session, clock, now = new_session(folder / "reading_list.csv", out)
    assert session.read_order == tuple(ids)
    shown = []
    while not session.complete:
        shown.append(session.view().read_id)
        assert session.current_item.read_id == shown[-1]
        session.record("absent")
    assert shown == ids
    session.finalise()
    rows = read_rows(out)
    assert [row["read_id"] for row in rows] == ids
    assert [row["order_index"] for row in rows] == ["0", "1", "2", "3", "4"]


def test_draft_flow_records_answer_flags_and_note(tmp_path):
    list_path, rows = make_list(tmp_path / "items", n=6)
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    view = session.view()
    assert (view.position, view.progress, view.response, view.flags, view.note) == (0, "1 of 6", None, frozenset(), "")
    assert view.read_id == rows[0]["read_id"] and not view.can_go_back and view.mode == "read"
    session.select("absent")
    session.select("present")                                    # a later key replaces the draft
    assert session.toggle_flag("ambiguous") is True
    assert session.toggle_flag("out_of_field") is True
    assert session.toggle_flag("out_of_field") is False
    session.set_note("  faint stripe,\n near the carina \t ")
    clock.advance(12.5)
    assert session.confirm() == 0
    assert session.responses[0] == rc.Response(
        "present", frozenset({"ambiguous"}), "faint stripe, near the carina", 12.5, "2026-11-02T09:00:00Z")
    view = session.view()
    assert view.position == 1 and view.response is None and view.flags == frozenset() and view.note == ""
    session.select("skip")
    session.toggle_flag("tracheostomy")
    session.clear_draft()
    assert session.view().response is None and session.view().flags == frozenset()


def test_invalid_answers_are_refused_and_nothing_is_saved(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=3)
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    with pytest.raises(rc.ResponseError, match="before confirming"):
        session.confirm()
    with pytest.raises(rc.ResponseError):
        session.select("maybe")
    with pytest.raises(rc.ResponseError):
        session.toggle_flag("pneumothorax")
    with pytest.raises(rc.ResponseError, match="unknown flags"):
        session.record("present", flags=["ambiguous", "nope"])
    with pytest.raises(rc.ResponseError):
        session.record("Present")
    assert session.responses == {} and session.position == 0
    assert not session.partial_path.exists()


def test_seconds_accumulate_over_visits_including_back(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=4)
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    clock.advance(10)
    session.record("absent")                                     # item 0: 10 s
    clock.advance(4)                                             # 4 s on item 1, then back
    assert session.back() is True and session.position == 0
    view = session.view()
    assert view.response == "absent" and view.stepped_back and not view.can_go_back
    clock.advance(3)
    session.select("present")
    session.confirm()                                            # item 0: 10 + 3
    assert session.responses[0].response == "present" and session.responses[0].seconds == 13.0
    assert session.position == 1
    clock.advance(5)
    session.record("skip")                                       # item 1: 4 before going back + 5
    assert session.responses[1].seconds == 9.0
    session.restart_timer()
    clock.advance(0.0004)
    session.record("absent")
    assert session.responses[2].seconds == 0.0                   # rounded to milliseconds


def test_back_is_one_step_only(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=5)
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    assert session.back() is False                               # first item
    for response in ("present", "absent", "skip"):
        session.record(response)
    assert session.back() is True and session.position == 2
    assert session.back() is False and session.position == 2     # no second step
    session.confirm()                                            # re-confirm the unchanged draft
    assert session.position == 3 and session.responses[2].response == "skip"
    assert session.back() is True                                # allowed again after confirming


def test_back_from_the_end_changes_the_last_answer(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=3)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out)
    for _ in range(3):
        session.record("present")
    view = session.view()
    assert view.complete and view.progress == "3 of 3 answered" and view.read_id == "" and view.image_problem is None
    with pytest.raises(rc.ReadingError, match="every item is answered"):
        session.record("absent")
    assert session.back() is True and session.position == 2
    session.select("absent")
    session.toggle_flag("out_of_field")
    session.confirm()
    assert session.complete
    session.finalise()
    last = read_rows(out)[2]
    assert last["response"] == "absent" and last["flag_out_of_field"] == "1"


# ---------------------------------------------------------------- autosave and resume
def test_autosave_after_every_response(tmp_path):
    list_path, rows = make_list(tmp_path / "items", n=4)
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    assert session.partial_path == tmp_path / "items" / "verified_items.partial.csv"
    assert not session.partial_path.exists()
    for k, response in enumerate(["present", "absent", "skip"]):
        session.record(response, flags=["tracheostomy"] if k == 1 else [], note=f"note {k}")
        saved = read_rows(session.partial_path)
        assert [row["order_index"] for row in saved] == [str(i) for i in range(k + 1)]
        assert [row["read_id"] for row in saved] == [row["read_id"] for row in rows[: k + 1]]
        assert saved[-1]["response"] == response and saved[-1]["note"] == f"note {k}"
    assert saved[1]["flag_tracheostomy"] == "1" and saved[0]["flag_tracheostomy"] == "0"
    header = session.partial_path.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == rc.OUTPUT_COLUMNS
    assert not list(session.partial_path.parent.glob("*.tmp"))


def test_resume_after_simulated_crash(tmp_path):
    list_path, rows = make_list(tmp_path / "items", n=6)
    out = tmp_path / "items" / "verified_items.csv"
    first, clock, now = new_session(list_path, out)
    clock.advance(7)
    first.record("present", flags=["ambiguous", "out_of_field"], note='quoted "tip", comma')
    now.advance(minutes=1)
    clock.advance(8)
    first.record("absent")
    clock.advance(2)
    first.select("skip")                                         # an unconfirmed draft is lost in the crash
    before = first.responses
    first.partial_path.with_name(first.partial_path.name + ".tmp").write_bytes(b"half-written")
    del first                                                    # the process dies without finalising

    second, clock2, now2 = new_session(list_path, out)
    assert second.resumed_count == 2 and second.position == 2
    assert second.responses == before                            # flags, notes, seconds, timestamps exact
    assert second.view().response is None
    assert any("Resuming: 2 of 6 answered; continuing at image 3" in line for line in second.summary_lines())
    assert second.back() is True and second.view().response == "absent"
    second.confirm()
    assert second.responses[1].seconds == 8.0
    for response in ("skip", "present", "absent", "present"):
        second.record(response)
    result = second.finalise()
    saved = read_rows(out)
    assert [row["read_id"] for row in saved] == [row["read_id"] for row in rows]
    assert [row["response"] for row in saved] == ["present", "absent", "skip", "present", "absent", "present"]
    assert saved[0]["note"] == 'quoted "tip", comma' and saved[0]["seconds"] == "7.000"
    assert (saved[0]["flag_ambiguous"], saved[0]["flag_tracheostomy"], saved[0]["flag_out_of_field"]) == ("1", "0", "1")
    assert saved[1]["timestamp_utc"] == "2026-11-02T09:00:00Z"   # the re-confirmation time (fresh clock)
    assert not second.partial_path.exists() and result.partial_removed


def test_resume_when_everything_was_answered_before_the_crash(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=2)
    out = tmp_path / "items" / "verified_items.csv"
    first, clock, now = new_session(list_path, out)
    first.record("present")
    first.record("absent")
    del first
    second, clock, now = new_session(list_path, out)
    assert second.complete and second.can_go_back
    assert any("all answered" in line for line in second.summary_lines())
    second.finalise()
    assert [row["response"] for row in read_rows(out)] == ["present", "absent"]


def test_resume_refuses_a_partial_made_with_another_list(tmp_path):
    folder = tmp_path / "items"
    list_a, rows = make_list(folder, n=4)
    out = folder / "verified_items.csv"
    session, clock, now = new_session(list_a, out)
    session.record("present")
    session.record("absent")
    del session
    list_b = folder / "other_order.csv"
    swapped = [rows[1], rows[0], *rows[2:]]
    with open(list_b, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rc.LIST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(swapped)
    with pytest.raises(rc.ResumeError, match="does not match this reading list"):
        rc.ReadingSession(list_b, out)
    with pytest.raises(rc.ResumeError, match="outside this list"):
        rc.ReadingSession(list_a, out, limit=1)


def _corrupt(path: Path, edit) -> None:
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))
    rows = edit(rows)
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _set(column: str, value: str, row: int = 1):
    def edit(rows):
        rows[row][rows[0].index(column)] = value
        return rows
    return edit


@pytest.mark.parametrize(
    "edit, fragment",
    [
        (_set("seconds", "secs", row=0), "S11 columns"),
        (lambda rows: [rows[0][:-1]] + [r[:-1] for r in rows[1:]], "S11 columns"),
        (_set("response", "maybe"), "response"),
        (_set("flag_ambiguous", "2"), "must be 0 or 1"),
        (_set("seconds", "-1"), "finite number"),
        (_set("seconds", "nan"), "finite number"),
        (_set("seconds", "abc"), "not a number"),
        (_set("order_index", "x"), "not an integer"),
        (_set("order_index", "99"), "outside this list"),
        (_set("timestamp_utc", "yesterday"), "timestamp_utc"),
        (_set("timestamp_utc", "2026-11-02T09:00:00"), "timestamp_utc"),
        (lambda rows: rows + [rows[1]], "appears twice"),
        (lambda rows: [rows[0], rows[1] + ["extra"]], "wrong number of fields"),
        (lambda rows: [rows[0], rows[1][:-2]], "wrong number of fields"),
    ],
)
def test_resume_rejects_a_corrupted_partial(tmp_path, edit, fragment):
    list_path, _ = make_list(tmp_path / "items", n=3)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out)
    session.record("present")
    session.record("absent")
    _corrupt(session.partial_path, edit)
    with pytest.raises(rc.ResumeError, match=re.escape(fragment)):
        rc.ReadingSession(list_path, out)


@pytest.mark.parametrize("payload, fragment", [(b"", "S11 columns"), (b"\xff\xfe\x00bad", "not UTF-8")])
def test_resume_rejects_empty_or_binary_partial(tmp_path, payload, fragment):
    list_path, _ = make_list(tmp_path / "items", n=2)
    out = tmp_path / "items" / "verified_items.csv"
    rc.partial_path_for(out).write_bytes(payload)
    with pytest.raises(rc.ResumeError, match=fragment):
        rc.ReadingSession(list_path, out)


def test_header_only_partial_resumes_at_the_start(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=2)
    out = tmp_path / "items" / "verified_items.csv"
    rc.partial_path_for(out).write_bytes(rc.render_csv(rc.OUTPUT_COLUMNS, []))
    session, clock, now = new_session(list_path, out)
    assert session.position == 0 and session.responses == {}
    session.record("present")                                    # the loaded file may be replaced
    assert len(read_rows(session.partial_path)) == 1


def test_autosave_failure_keeps_the_answer_and_recovers(tmp_path, monkeypatch):
    list_path, _ = make_list(tmp_path / "items", n=4)
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    session.record("present")
    real_replace = os.replace

    def locked(src, dst):
        raise PermissionError("the file is open in another program")

    monkeypatch.setattr(rc.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(rc.os, "replace", locked)
    with pytest.raises(rc.AutosaveError, match="kept in memory"):
        session.record("absent")
    monkeypatch.setattr(rc.os, "replace", real_replace)
    assert session.responses[1].response == "absent" and session.position == 2
    assert [row["response"] for row in read_rows(session.partial_path)] == ["present"]
    assert not list(session.partial_path.parent.glob("*.tmp"))
    session.record("skip")                                       # the next save catches up
    assert [row["response"] for row in read_rows(session.partial_path)] == ["present", "absent", "skip"]


def test_atomic_write_retries_transient_locks(tmp_path, monkeypatch):
    target = tmp_path / "x.partial.csv"
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("scanner lock")
        real_replace(src, dst)

    monkeypatch.setattr(rc.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(rc.os, "replace", flaky)
    rc._atomic_write(target, b"abc")
    assert target.read_bytes() == b"abc" and calls["n"] == 3


def test_second_window_on_the_same_output_is_detected(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=3)
    out = tmp_path / "items" / "verified_items.csv"
    first, clock, now = new_session(list_path, out)
    second, clock2, now2 = new_session(list_path, out)
    first.record("present")
    with pytest.raises(rc.ConcurrentWriteError, match="another program or tool window"):
        second.record("absent")
    assert [row["response"] for row in read_rows(first.partial_path)] == ["present"]


def test_partial_deleted_by_hand_is_written_again(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=3)
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    session.record("present")
    session.partial_path.unlink()
    session.record("absent")
    assert [row["response"] for row in read_rows(session.partial_path)] == ["present", "absent"]


# ---------------------------------------------------------------- finalising
def test_finalise_is_refused_while_items_are_unanswered(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=6)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out)
    session.record("present")
    with pytest.raises(rc.IncompleteError, match="5 of 6 images have no answer yet"):
        session.finalise()
    assert not out.exists() and not session.meta_path.exists()
    assert session.partial_path.exists() and not session.finalised


def test_skip_remaining_ends_early_and_allows_finalise(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=6)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out)
    session.record("present")
    clock.advance(4)                                             # 4 s on image 2 before ending early
    assert session.skip_remaining() == 5
    assert session.complete and session.unanswered() == []
    assert session.responses[1] == rc.Response("skip", frozenset(), rc.SKIP_REMAINING_NOTE, 4.0, "2026-11-02T09:00:00Z")
    assert session.responses[4].seconds == 0.0
    assert len(read_rows(session.partial_path)) == 6
    assert session.skip_remaining() == 0
    result = session.finalise()
    assert result.n_skip == 5 and result.n == 6


def test_finalise_writes_output_and_meta(tmp_path):
    list_path, rows = make_list(tmp_path / "items", n=6, seed=2026101205)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out, "read")
    answers = [("present", {"ambiguous"}), ("absent", set()), ("skip", set()),
               ("present", {"tracheostomy", "out_of_field"}), ("absent", set()), ("absent", set())]
    for response, flags in answers:
        clock.advance(3.25)
        now.advance(seconds=30)
        session.record(response, flags=flags)
    now.advance(minutes=5)
    result = session.finalise()

    data = out.read_bytes()
    assert b"\r\n" not in data and not data.startswith(b"\xef\xbb\xbf")
    meta_path = tmp_path / "items" / "verified_items.meta.json"
    assert result.meta_path == meta_path == rc.meta_path_for(out)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert list(meta) == list(rc.META_KEYS)
    assert meta == {"sha256": hashlib.sha256(data).hexdigest(), "saved_utc": "2026-11-02T09:08:00Z",
                    "seed": 2026101205, "n": 6, "mode": "read", "tool_version": rc.TOOL_VERSION}
    assert result.sha256 == meta["sha256"] == rc.sha256_file(out)
    assert result.earliest_retest_utc == "2026-11-04T09:08:00Z"
    assert rc.verify_output(out) == meta

    saved = read_rows(out)
    assert tuple(saved[0].keys()) == rc.OUTPUT_COLUMNS
    assert [row["read_id"] for row in saved] == [row["read_id"] for row in rows]
    assert [row["nih_filename"] for row in saved] == [row["nih_filename"] for row in rows]
    assert [row["order_index"] for row in saved] == [str(i) for i in range(6)]
    assert [row["response"] for row in saved] == [a for a, _ in answers]
    assert [(r["flag_ambiguous"], r["flag_tracheostomy"], r["flag_out_of_field"]) for r in saved][0::3] == [
        ("1", "0", "0"), ("0", "1", "1")]
    assert all(row["seconds"] == "3.250" for row in saved)
    assert saved[0]["timestamp_utc"] == "2026-11-02T09:00:30Z"
    assert all(ISO_Z.fullmatch(row["timestamp_utc"]) for row in saved)
    assert result.n_skip == 1 and result.n_flagged == 2
    assert not session.partial_path.exists() and result.partial_removed and session.finalised
    summary = "\n".join(result.summary_lines())
    assert meta["sha256"] in summary and "Earliest retest start" in summary

    with pytest.raises(rc.SessionClosedError):
        session.record("present")
    with pytest.raises(rc.SessionClosedError):
        session.finalise()
    with pytest.raises(rc.SessionClosedError):
        session.back()
    with pytest.raises(rc.SessionClosedError):
        session.skip_remaining()


def test_retest_mode_meta(tmp_path):
    session, result, list_path, rows, out = finalised_session(
        tmp_path / "items", n=3, mode="retest", name="retest_list.csv", seed=77, sidecar="full")
    meta = rc.read_meta(out)
    assert out.name == "retest.csv" and meta["mode"] == "retest" and meta["seed"] == 77 and meta["n"] == 3
    assert result.earliest_retest_utc is None


@pytest.mark.parametrize("sidecar, expected", [("stem", 11), ("full", 11), (None, None)])
def test_seed_comes_from_the_list_sidecar(tmp_path, sidecar, expected):
    session, result, list_path, rows, out = finalised_session(tmp_path / "items", n=2, seed=11, sidecar=sidecar)
    assert rc.read_list_seed(list_path) == expected and session.seed == expected
    assert json.loads(rc.meta_path_for(out).read_text(encoding="utf-8"))["seed"] == expected
    line = [text for text in session.summary_lines() if text.startswith("Order seed")][0]
    assert (str(expected) in line) if expected is not None else ("not recorded" in line)


def test_bad_list_sidecars(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=2, sidecar=None)
    stem, full = rc.list_sidecar_paths(list_path)
    assert (stem.name, full.name) == ("reading_list.meta.json", "reading_list.csv.meta.json")
    for payload, fragment in [("{not json", "not valid JSON"), ('{"seed": "abc"}', "integer"),
                              ('{"seed": true}', "integer"), ("[1, 2]", "JSON object")]:
        stem.write_text(payload, encoding="utf-8")
        with pytest.raises(rc.ListError, match=fragment):
            rc.read_list_seed(list_path)
    stem.write_text('{"other": 1}', encoding="utf-8")
    assert rc.read_list_seed(list_path) is None
    stem.write_text('{"seed": 5}', encoding="utf-8")
    full.write_text('{"seed": 6}', encoding="utf-8")
    with pytest.raises(rc.ListError, match="disagree"):
        rc.read_list_seed(list_path)
    full.write_text('{"seed": 5}', encoding="utf-8")
    assert rc.read_list_seed(list_path) == 5


def test_refuses_to_overwrite_a_finalised_output(tmp_path):
    session, result, list_path, rows, out = finalised_session(tmp_path / "items", n=2)
    before = out.read_bytes(), rc.meta_path_for(out).read_bytes()
    with pytest.raises(rc.OutputExistsError, match="already finalised"):
        rc.ReadingSession(list_path, out)
    assert (out.read_bytes(), rc.meta_path_for(out).read_bytes()) == before


def test_output_appearing_during_the_session_is_not_overwritten(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=2)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out)
    session.record("present")
    session.record("absent")
    out.write_text("someone else's file", encoding="utf-8")
    with pytest.raises(rc.OutputExistsError):
        session.finalise()
    assert out.read_text(encoding="utf-8") == "someone else's file"
    assert not session.meta_path.exists() and session.partial_path.exists() and not session.finalised


def test_interrupted_finalisation_states_are_refused_at_start(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=2)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out)
    session.record("present")
    out.write_text("partial output", encoding="utf-8")
    with pytest.raises(rc.OutputExistsError, match="finalisation was probably interrupted"):
        rc.ReadingSession(list_path, out)
    out.unlink()
    rc.meta_path_for(out).write_text("{}", encoding="utf-8")
    with pytest.raises(rc.OutputExistsError, match="without its output"):
        rc.ReadingSession(list_path, out)


def test_failed_meta_write_rolls_back_the_output(tmp_path, monkeypatch):
    list_path, _ = make_list(tmp_path / "items", n=2)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out)
    session.record("present")
    session.record("absent")
    real_publish = rc._publish_new

    def failing(path, data):
        if path.suffix == ".json":
            raise OSError("disk full")
        real_publish(path, data)

    monkeypatch.setattr(rc, "_publish_new", failing)
    with pytest.raises(OSError, match="disk full"):
        session.finalise()
    assert not out.exists() and not session.meta_path.exists()
    assert session.partial_path.exists() and not session.finalised
    monkeypatch.setattr(rc, "_publish_new", real_publish)
    session.finalise()
    assert rc.verify_output(out)["n"] == 2


def test_publish_new_without_hard_links(tmp_path, monkeypatch):
    def no_links(src, dst):
        raise OSError("hard links not supported")

    monkeypatch.setattr(rc.os, "link", no_links)
    target = tmp_path / "out.csv"
    rc._publish_new(target, b"first")
    assert target.read_bytes() == b"first"
    with pytest.raises(rc.OutputExistsError):
        rc._publish_new(target, b"second")
    assert target.read_bytes() == b"first" and not list(tmp_path.glob("*.tmp"))


def test_output_path_rules(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=2)
    with pytest.raises(ValueError, match=".csv"):
        rc.ReadingSession(list_path, tmp_path / "items" / "verified_items.txt")
    with pytest.raises(ValueError, match="overwrite the reading list"):
        rc.ReadingSession(list_path, list_path)
    clash = tmp_path / "items" / "verified_items.partial.csv"
    shutil.copy(list_path, clash)
    with pytest.raises(ValueError, match="overwrite the reading list"):
        rc.ReadingSession(clash, tmp_path / "items" / "verified_items.csv")
    with pytest.raises(ValueError, match="mode"):
        rc.ReadingSession(list_path, tmp_path / "items" / "verified_items.csv", "reread")
    assert rc.partial_path_for("items/retest.csv") == Path("items/retest.partial.csv")
    assert rc.meta_path_for("items/retest.csv") == Path("items/retest.meta.json")


def test_verify_output_detects_tampering(tmp_path):
    session, result, list_path, rows, out = finalised_session(tmp_path / "items", n=2)
    assert rc.verify_output(out)["sha256"] == result.sha256
    out.write_bytes(out.read_bytes() + b"0A1B2C3D,x.png,present,0,0,0,,1.000,2,2026-11-02T09:00:00Z\n")
    with pytest.raises(rc.MetaError, match="does not match"):
        rc.verify_output(out)


@pytest.mark.parametrize(
    "change, fragment",
    [
        (lambda m: m.pop("tool_version"), "exactly the keys"),
        (lambda m: m.update(extra=1), "exactly the keys"),
        (lambda m: m.update(sha256="ABC"), "sha256"),
        (lambda m: m.update(saved_utc="tomorrow"), "saved_utc"),
        (lambda m: m.update(seed="7"), "seed"),
        (lambda m: m.update(n=0), "positive integer"),
        (lambda m: m.update(mode="practice"), "mode"),
        (lambda m: m.update(tool_version=1), "tool_version"),
    ],
)
def test_read_meta_validates_the_sidecar(tmp_path, change, fragment):
    session, result, list_path, rows, out = finalised_session(tmp_path / "items", n=2)
    meta_path = rc.meta_path_for(out)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    change(meta)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(rc.MetaError, match=fragment):
        rc.read_meta(out)


def test_read_meta_missing_or_not_json(tmp_path):
    with pytest.raises(rc.MetaError, match="not found"):
        rc.read_meta(tmp_path / "verified_items.csv")
    (tmp_path / "verified_items.meta.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(rc.MetaError, match="not valid JSON"):
        rc.read_meta(tmp_path / "verified_items.csv")


# ---------------------------------------------------------------- the 48-hour retest clock
def test_retest_gate(tmp_path):
    session, result, list_path, rows, out = finalised_session(tmp_path / "read", n=2, now=FakeNow())
    saved = rc.parse_utc(result.saved_utc)
    assert rc.retest_earliest_start(out) == saved + timedelta(hours=48)
    with pytest.raises(rc.RetestGapError, match="48 h"):
        rc.check_retest_gap(out, now=saved + timedelta(hours=47, minutes=59, seconds=59))
    assert rc.check_retest_gap(out, now=saved + timedelta(hours=48)) == saved + timedelta(hours=48)
    with pytest.raises(ValueError, match="timezone-aware"):
        rc.check_retest_gap(out, now=datetime(2026, 11, 9))
    _, _, _, _, retest_out = finalised_session(tmp_path / "retest", n=2, mode="retest")
    with pytest.raises(rc.MetaError, match="not the first read"):
        rc.check_retest_gap(retest_out, now=saved + timedelta(days=30))
    with pytest.raises(rc.MetaError, match="not found"):
        rc.check_retest_gap(tmp_path / "nothing.csv")


# ---------------------------------------------------------------- images
def test_missing_image_can_only_be_skipped(tmp_path):
    list_path, rows = make_list(tmp_path / "items", n=3, missing=(1,))
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    assert session.missing_images() == [rows[1]["read_id"]]
    session.record("present")
    assert session.view().image_problem == "image file missing"
    with pytest.raises(rc.ResponseError, match="only be skipped"):
        session.select("present")
    with pytest.raises(rc.ResponseError, match="only be skipped"):
        session.record("absent")
    with pytest.raises(rc.ImageUnavailableError) as info:
        session.load_image()
    assert rows[1]["nih_filename"] not in str(info.value)
    session.toggle_flag("ambiguous")                             # flags are still allowed
    session.select("skip")
    session.confirm()
    assert session.responses[1].response == "skip" and session.responses[1].note == "image file missing"
    assert session.responses[1].flags == frozenset({"ambiguous"})


def test_unreadable_image(tmp_path):
    list_path, rows = make_list(tmp_path / "items", n=2)
    (tmp_path / "items" / "images" / rows[0]["nih_filename"]).write_bytes(b"not a png at all")
    session, clock, now = new_session(list_path, tmp_path / "items" / "verified_items.csv")
    assert session.image_problem() is None                       # unknown until decoding is tried
    with pytest.raises(rc.ImageUnavailableError, match="unreadable") as info:
        session.load_image()
    assert rows[0]["nih_filename"] not in str(info.value)
    assert session.image_problem() == "image file unreadable"
    with pytest.raises(rc.ResponseError):
        session.select("absent")
    session.record("skip", note="corrupt file")
    assert session.responses[0].note == "corrupt file"
    image = session.load_image()
    assert image.mode == "L" and image.size == (12, 10)


@pytest.mark.parametrize(
    "size, box, expected",
    [((1024, 1024), (800, 600), (600, 600)), ((1000, 500), (400, 400), (400, 200)),
     ((100, 50), (1000, 1000), (1000, 500)), ((3000, 2000), (1, 1), (1, 1)), ((10, 10), (0, -5), (1, 1))],
)
def test_fit_size(size, box, expected):
    assert rc.fit_size(*size, *box) == expected


def test_fit_size_property_and_errors():
    rng = np.random.default_rng(derive_seed("reading"))
    for _ in range(300):
        w, h, bw, bh = (int(v) for v in rng.integers(1, 3000, size=4))
        fw, fh = rc.fit_size(w, h, bw, bh)
        assert 1 <= fw <= bw and 1 <= fh <= bh
        assert fw == bw or fh == bh or fw == 1 or fh == 1
        assert abs(fw * h - fh * w) <= max(w, h)                 # aspect kept to within a pixel
    with pytest.raises(ValueError):
        rc.fit_size(0, 10, 100, 100)
    img = Image.new("L", (40, 20), 7)
    assert rc.fit_image(img, 40, 20) is img
    assert rc.fit_image(img, 20, 20).size == (20, 10)


def test_contrast_presets():
    rng = np.random.default_rng(derive_seed("retest"))
    arr = rng.integers(100, 140, size=(64, 64), dtype=np.uint8)
    img = Image.fromarray(arr)
    assert np.array_equal(np.asarray(rc.apply_contrast(img, "original")), arr)
    auto = np.asarray(rc.apply_contrast(img, "autocontrast"))
    equal = np.asarray(rc.apply_contrast(img, "equalised"))
    assert np.ptp(auto) > 200 and np.ptp(equal) > 200 and np.ptp(arr) < 40
    assert rc.apply_contrast(Image.fromarray(np.dstack([arr] * 3)), "original").mode == "L"
    assert [rc.next_contrast(p) for p in rc.CONTRAST_PRESETS] == ["autocontrast", "equalised", "original"]
    with pytest.raises(ValueError):
        rc.apply_contrast(img, "gamma")
    with pytest.raises(ValueError):
        rc.next_contrast("gamma")


def test_load_display_image_modes(tmp_path):
    rgba = tmp_path / "rgba.png"
    Image.new("RGBA", (8, 6), (200, 200, 200, 255)).save(rgba)
    assert rc.load_display_image(rgba).mode == "L"
    deep = tmp_path / "deep.png"
    Image.fromarray(np.arange(64, dtype=np.uint16).reshape(8, 8) * 64).save(deep)
    loaded = np.asarray(rc.load_display_image(deep))
    assert loaded.dtype == np.uint8 and loaded.min() == 0 and loaded.max() == 255
    flat = Image.fromarray(np.full((4, 4), 1000, dtype=np.uint16))
    assert np.asarray(rc.to_display_gray(flat)).max() == 0
    assert rc.to_display_gray(Image.fromarray(np.full((3, 3), 0.5, dtype=np.float32))).mode == "L"


# ---------------------------------------------------------------- small helpers
def test_clean_note_render_csv_and_time_helpers():
    assert rc.clean_note(" a\tb\r\nc  d \x00e ") == "a b c d e"
    assert rc.clean_note("tube\u2028tip") == "tube tip"
    assert rc.clean_note(None) == ""
    assert rc.render_csv(["a", "b"], [["x,y", 'q"q'], [1, ""]]) == b'a,b\n"x,y","q""q"\n1,\n'
    assert rc.format_utc(datetime(2026, 11, 2, 12, 0, tzinfo=timezone(timedelta(hours=3)))) == "2026-11-02T09:00:00Z"
    assert rc.parse_utc("2026-11-02T09:00:00Z") == T0
    assert rc.parse_utc("2026-11-02T12:00:00+03:00") == T0
    with pytest.raises(ValueError):
        rc.format_utc(datetime(2026, 11, 2))
    with pytest.raises(ValueError):
        rc.parse_utc("2026-11-02T09:00:00")
    assert ISO_Z.fullmatch(rc.utc_now())
    assert rc.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


# ---------------------------------------------------------------- practice and run planning
def test_plan_run_study_and_retest(tmp_path):
    list_path, _ = make_list(tmp_path / "items", n=2)
    out = tmp_path / "items" / "verified_items.csv"
    plan = rc.plan_run(list_path, out, "read")
    assert (plan.list_path, plan.out_path, plan.limit, plan.practice_dir, plan.messages) == (list_path, out, None, None, ())
    with pytest.raises(ValueError, match="required"):
        rc.plan_run(None, out)
    with pytest.raises(ValueError, match="required"):
        rc.plan_run(list_path, None)
    with pytest.raises(ValueError, match="mode"):
        rc.plan_run(list_path, out, "reread")
    with pytest.raises(ValueError, match="retest only"):
        rc.plan_run(list_path, out, "read", first_read=out)
    plan = rc.plan_run(list_path, tmp_path / "items" / "retest.csv", "retest")
    assert any(message.startswith("WARNING") for message in plan.messages)

    session, result, _, _, first_out = finalised_session(tmp_path / "first", n=2)
    saved = rc.parse_utc(result.saved_utc)
    with pytest.raises(rc.RetestGapError):
        rc.plan_run(list_path, tmp_path / "items" / "retest.csv", "retest", first_read=first_out,
                    now=saved + timedelta(hours=1))
    plan = rc.plan_run(list_path, tmp_path / "items" / "retest.csv", "retest", first_read=first_out,
                       now=saved + timedelta(hours=49))
    assert any("48-hour check passed" in message for message in plan.messages)


def test_plan_run_practice_with_a_practice_list(tmp_path):
    base = tmp_path / "temp"
    base.mkdir()
    for name in ("reading_list.csv", "retest_list.csv", "Reading_List.csv"):
        with pytest.raises(rc.ListError, match="practice run must not show them"):
            rc.plan_run(tmp_path / "items" / name, None, practice=True, practice_base=base)
    with pytest.raises(rc.ListError, match="must never open"):
        rc.plan_run(tmp_path / "items" / "items.csv", None, practice=True, practice_base=base)
    with pytest.raises(rc.ListError, match="not found"):
        rc.plan_run(tmp_path / "items" / "practice_list.csv", None, practice=True, practice_base=base)
    assert list(base.iterdir()) == []                            # nothing created when refused

    practice_list, rows = make_list(tmp_path / "items", n=12, name="practice_list.csv")
    plan = rc.plan_run(practice_list, Path("items/verified_items.csv"), practice=True, practice_base=base)
    assert plan.practice_dir.parent == base and plan.limit == rc.PRACTICE_N == 10
    assert plan.out_path == plan.practice_dir / "verified_items.csv"
    session, clock, now = new_session(plan.list_path, plan.out_path, plan.mode, limit=plan.limit)
    assert session.n == 10 and session.read_order == tuple(row["read_id"] for row in rows[:10])
    for _ in range(10):
        session.record("absent")
    result = session.finalise()
    assert result.out_path.parent == plan.practice_dir and rc.read_meta(result.out_path)["n"] == 10
    assert not (tmp_path / "items" / "verified_items.csv").exists()


def test_plan_run_practice_demo_without_a_list(tmp_path):
    base = tmp_path / "temp"
    base.mkdir()
    plan = rc.plan_run(None, None, "retest", practice=True, practice_base=base)
    assert plan.list_path.name == "demo_list.csv" and plan.out_path.name == "practice_retest.csv"
    assert plan.list_path.parent == plan.practice_dir and any("synthetic" in m for m in plan.messages)
    session, clock, now = new_session(plan.list_path, plan.out_path, plan.mode, limit=plan.limit)
    assert session.n == 10 and session.missing_images() == [] and session.seed == derive_seed("reading")
    image = session.load_image()
    assert image.mode == "L" and image.size == (512, 512)


def test_demo_list_is_deterministic_and_valid(tmp_path):
    a = rc.write_demo_list(tmp_path / "a", n=4, seed=5)
    b = rc.write_demo_list(tmp_path / "b", n=4, seed=5)
    c = rc.write_demo_list(tmp_path / "c", n=4, seed=6)
    assert a.read_bytes() == b.read_bytes() and a.read_bytes() != c.read_bytes()
    items = rc.load_reading_list(a)
    assert len(items) == 4 and len({item.read_id for item in items}) == 4
    bright = 0
    for item in items:
        twin = tmp_path / "b" / "demo_images" / item.nih_filename
        assert item.image_path.read_bytes() == twin.read_bytes()
        bright += int(np.asarray(rc.load_display_image(item.image_path))[100, 256] > 180)
    assert bright == 2                                           # two images with the stripe, two without
    assert rc.read_list_seed(a) == 5
    with pytest.raises(ValueError):
        rc.write_demo_list(tmp_path / "d", n=0)


# ---------------------------------------------------------------- the GUI (thin layer)
class FakeDialogs:
    def __init__(self, answer: bool = True, note: str | None = "faint line") -> None:
        self.answer = answer
        self.note = note
        self.calls: list[tuple] = []

    def ask_yes_no(self, title, message):
        self.calls.append(("yes_no", title, message))
        return self.answer

    def ask_ok_cancel(self, title, message):
        self.calls.append(("ok_cancel", title, message))
        return self.answer

    def ask_string(self, title, prompt, initial):
        self.calls.append(("string", title, prompt, initial))
        return self.note

    def info(self, title, message):
        self.calls.append(("info", title, message))

    def error(self, title, message):
        self.calls.append(("error", title, message))


@pytest.fixture(scope="module")
def tk_interp():
    """One Tk interpreter for the whole module. On this Windows machine a second Tcl
    interpreter created in the same process after a ReaderApp session intermittently failed
    to read its own library files (init.tcl, text.tcl); the tool itself creates only one
    interpreter per process, so the tests share one and give each app a Toplevel."""
    tk = pytest.importorskip("tkinter")
    try:
        interp = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk cannot start here: {exc}")
    interp.withdraw()
    yield interp
    try:
        interp.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def tk_root(tk_interp):
    """A hidden window that plays the app's root for one test."""
    import tkinter as tk

    window = tk.Toplevel(tk_interp)
    window.withdraw()
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass


def test_parse_args_and_key_map():
    pytest.importorskip("tkinter")
    from tools import read_tool

    args = read_tool.parse_args(["--list", "l.csv", "--out", "o.csv"])
    assert (args.list, args.out, args.mode, args.practice, args.first_read) == (Path("l.csv"), Path("o.csv"), "read", False, None)
    assert read_tool.parse_args(["--practice"]).list is None
    args = read_tool.parse_args(["--list", "l.csv", "--out", "o.csv", "--mode", "retest", "--first-read", "v.csv"])
    assert args.mode == "retest" and args.first_read == Path("v.csv")
    for bad in (["--list", "l.csv"], ["--list", "l.csv", "--out", "o.csv", "--mode", "x"],
                ["--list", "l.csv", "--out", "o.csv", "--first-read", "v.csv"]):
        with pytest.raises(SystemExit):
            read_tool.parse_args(bad)
    for name, argument in read_tool.KEY_ACTIONS.values():
        assert callable(getattr(read_tool.ReaderApp, f"do_{name}"))
        assert argument is None or argument in rc.RESPONSES + rc.FLAGS
    assert {read_tool.KEY_ACTIONS[k][1] for k in ("1", "2", "3")} == set(rc.FLAGS)
    units = read_tool.ReaderApp._wheel_units
    assert (units(120), units(-120), units(240), units(30), units(-30), units(0)) == (-1, 1, -2, -1, 1, 0)


def test_main_reports_startup_errors(tmp_path, monkeypatch):
    pytest.importorskip("tkinter")
    from tools import read_tool

    messages: list[str] = []
    monkeypatch.setattr(read_tool, "_startup_error", messages.append)
    key = tmp_path / "reading_key.csv"
    key.write_text("read_id,item_id\n", encoding="utf-8")
    assert read_tool.main(["--list", str(key), "--out", str(tmp_path / "verified_items.csv")]) == 2
    assert "must never open" in messages[-1]
    session, result, list_path, rows, out = finalised_session(tmp_path / "items", n=2)
    assert read_tool.main(["--list", str(list_path), "--out", str(out)]) == 2
    assert "already finalised" in messages[-1]


def test_gui_keys_drive_the_session(tmp_path, tk_root):
    from tools import read_tool

    list_path, rows = make_list(tmp_path / "items", n=3, missing=(2,))
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out)
    dialogs = FakeDialogs()
    app = read_tool.ReaderApp(tk_root, session, dialogs=dialogs)

    def screen_text() -> str:
        variables = (app.progress_var, app.read_id_var, app.answer_var, app.note_var, app.message_var, app.display_var)
        canvas_text = [app.canvas.itemcget(item, "text") for item in app.canvas.find_all()
                       if app.canvas.type(item) == "text"]
        return " | ".join([v.get() for v in variables] + canvas_text + [tk_root.title()])

    assert app.progress_var.get() == "1 of 3" and rows[0]["read_id"] in app.read_id_var.get()
    assert app.handle_key("Return") is True                      # nothing chosen yet
    assert session.responses == {} and "choose present" in app.message_var.get()
    assert app.handle_key("x") is False
    for key in ("P", "1", "n", "c", "z"):
        assert app.handle_key(key)
    assert app.contrast == "autocontrast" and app.zoom_100 and app.vbar.winfo_manager() == "grid"
    view = session.view()
    assert (view.response, view.flags, view.note) == ("present", frozenset({"ambiguous"}), "faint line")
    assert app.flag_labels["ambiguous"].cget("bg") == app.ON_BG
    app.handle_key("Return")
    assert session.responses[0].response == "present" and app.progress_var.get() == "2 of 3"
    assert not app.zoom_100 and app.vbar.winfo_manager() == ""
    app.handle_key("a")
    app.handle_key("Return")
    assert app.progress_var.get() == "3 of 3" and app._image_error == "image file missing"
    assert "Press S, then Enter" in screen_text()
    app.handle_key("p")
    assert "only be skipped" in app.message_var.get()
    app.handle_key("BackSpace")
    assert session.position == 1 and app.answer_var.get() == "Answer: ABSENT  (previous image)"
    app.handle_key("BackSpace")
    assert "back one image only" in app.message_var.get()
    app.handle_key("Return")
    app.handle_key("s")
    app.handle_key("Return")
    assert session.complete and app.progress_var.get() == "3 of 3 answered"
    text = screen_text()
    for row in rows:
        assert row["nih_filename"] not in text and row["nih_filename"].split("_")[0] not in text
    dialogs.answer = False
    app.handle_key("Return")                                     # finalise declined
    assert not session.finalised and not out.exists()
    dialogs.answer = True
    app.handle_key("Return")                                     # finalise accepted
    assert app.finished and session.finalised and out.exists() and app.result is not None
    assert any(call[0] == "info" and "SHA-256" in call[2] for call in dialogs.calls)
    assert app.handle_key("p") is True and session.finalised     # keys after the end do nothing
    assert [row["response"] for row in read_rows(out)] == ["present", "absent", "skip"]


def test_gui_end_early_and_quit_keep_the_autosave(tmp_path, tk_root):
    from tools import read_tool

    list_path, _ = make_list(tmp_path / "items", n=3)
    out = tmp_path / "items" / "verified_items.csv"
    session, clock, now = new_session(list_path, out, "retest")
    dialogs = FakeDialogs(answer=False)
    app = read_tool.ReaderApp(tk_root, session, dialogs=dialogs)
    assert tk_root.title() == "Blind ETT read - RETEST"
    app.do_skip_remaining()                                      # declined
    assert session.unanswered() == [0, 1, 2]
    dialogs.answer = True
    app.do_skip_remaining()
    assert session.complete and len(read_rows(session.partial_path)) == 3
    app.on_close()
    assert app.finished and not session.finalised and session.partial_path.exists() and not out.exists()
