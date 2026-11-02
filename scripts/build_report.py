"""Assemble the Stage 1 report from Markdown sections and fill result placeholders.

Placeholders (docs/PIPELINE_SPEC.md section 6):
  [[NUM:key]]          value from results/numbers.json, formatted by key type
  [[NUM:key|pct]]      as a percentage with no decimals (0.623 -> 62%)
  [[NUM:key|ci]]       "value [lo, hi]" from key, key.lo and key.hi
  [[NUM:key|p]]        p-value style ("<0.001" or three decimals)
  [[NUM:key|int]]      integer with thousands separators
  [[TODO:free text]]   a fact the student or a later step must supply

A key whose value is null in numbers.json (an undefined number, for example a p-value of an
untested cell) renders as "n/a" in every style. The script never invents a number: an unresolved placeholder stays in the text and is
listed in the build log, and the exit code is non-zero when --strict is given.

Usage:
  python scripts/build_report.py [--sections report/sections] [--numbers results/numbers.json]
                                 [--out report/out] [--strict] [--docx]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLACEHOLDER_RE = re.compile(r"\[\[NUM:([A-Za-z0-9_.\-]+)(?:\|(pct|ci|p|int))?\]\]")
TODO_RE = re.compile(r"\[\[TODO:([^\]]*)\]\]")
CITE_RE = re.compile(r"@([A-Za-z0-9_:\-]+)")
BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")

# Wording rules from STAGE1_PLAN.md section 7 / PIPELINE_SPEC section 0.
FORBIDDEN = [
    (re.compile(r"radiologist[- ]label", re.I), "say 'RANZCR CLiP labels (43 trained labellers)'"),
    (re.compile(r"counterfactual", re.I), "say 'within-patient contrast'"),
    (re.compile(r"\bunseen\b", re.I), "no model is called unseen without a card citation"),
    (re.compile(r"label[- ]clean", re.I), "describe what the training list contains instead"),
    (re.compile(r"\bverbatim\b", re.I), "the Kalai prompt is 'with t substituted as a percentage'"),
    (re.compile(r"inter-?rater audit", re.I), "there is no second rater; never call it an inter-rater audit"),
]

SECTION_ORDER_HINT = ("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "appendix")


def _fmt_number(key: str, value: object, style: str | None, numbers: dict) -> str:
    """Format one value for the text (null renders as "n/a")."""
    if style == "ci":
        if f"{key}.lo" not in numbers or f"{key}.hi" not in numbers:
            raise KeyError(f"{key}.lo/.hi")
        return f"{_plain(value)} [{_plain(numbers[key + '.lo'])}, {_plain(numbers[key + '.hi'])}]"
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    if style == "pct":
        return f"{float(value) * 100:.0f}%"
    if style == "p":
        v = float(value)
        return "<0.001" if v < 0.001 else f"{v:.3f}"
    if style == "int" or key.endswith(".n") or key.endswith("_n") or isinstance(value, bool):
        return f"{int(round(float(value))):,}"
    return _plain(value)


def _plain(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if abs(value) >= 100:
            return f"{value:,.0f}"
        return f"{value:.2f}"
    return str(value)


def fill_placeholders(text: str, numbers: dict) -> tuple[str, list[str]]:
    """Replace [[NUM:...]] placeholders; return the new text and the unresolved keys."""
    unresolved: list[str] = []

    def repl(m: re.Match) -> str:
        key, style = m.group(1), m.group(2)
        if key not in numbers:
            unresolved.append(key)
            return m.group(0)
        try:
            return _fmt_number(key, numbers[key], style, numbers)
        except (KeyError, TypeError, ValueError):
            unresolved.append(f"{key}|{style}")
            return m.group(0)

    return PLACEHOLDER_RE.sub(repl, text), unresolved


def wording_hits(text: str) -> list[tuple[int, str, str]]:
    """(line number, matched text, advice) for every forbidden wording, skipping rule statements."""
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        if "never" in low and ("say" in low or "call" in low or "called" in low):
            continue  # a sentence stating the rule itself
        for rx, advice in FORBIDDEN:
            for m in rx.finditer(line):
                hits.append((i, m.group(0), advice))
    return hits


def merge_bib(paths: list[Path]) -> tuple[str, list[str], list[str]]:
    """Concatenate BibTeX files, dropping exact duplicate keys; report conflicting duplicates."""
    entries: dict[str, str] = {}
    conflicts: list[str] = []
    for p in paths:
        if not p.exists():
            continue
        raw = p.read_text(encoding="utf-8")
        starts = [m.start() for m in re.finditer(r"(?m)^@\w+\s*\{", raw)]
        starts.append(len(raw))
        for a, b in zip(starts, starts[1:]):
            chunk = raw[a:b].strip()
            km = BIB_KEY_RE.match(chunk)
            if not km:
                continue
            key = km.group(1)
            if key in entries:
                if _norm(entries[key]) != _norm(chunk):
                    conflicts.append(key)
                continue
            entries[key] = chunk
    return "\n\n".join(entries.values()) + "\n", sorted(entries), sorted(set(conflicts))


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def cited_keys(text: str) -> set[str]:
    keys = set()
    for block in re.findall(r"\[([^\]]*@[^\]]*)\]", text):
        keys.update(CITE_RE.findall(block))
    return keys


def ordered_sections(sections_dir: Path) -> list[Path]:
    files = [p for p in sections_dir.glob("*.md") if not p.name.startswith("_")]

    def rank(p: Path) -> tuple[int, str]:
        for i, prefix in enumerate(SECTION_ORDER_HINT):
            if p.name.startswith(prefix):
                return (i, p.name)
        return (len(SECTION_ORDER_HINT), p.name)

    return sorted(files, key=rank)


def build(sections_dir: Path, numbers_path: Path, out_dir: Path, bib_paths: list[Path]) -> dict:
    numbers = json.loads(numbers_path.read_text(encoding="utf-8")) if numbers_path.exists() else {}
    parts, unresolved, todos, hits = [], [], [], []
    for p in ordered_sections(sections_dir):
        text = p.read_text(encoding="utf-8")
        filled, missing = fill_placeholders(text, numbers)
        unresolved += [f"{p.name}: {k}" for k in missing]
        todos += [f"{p.name}: {t.strip()}" for t in TODO_RE.findall(filled)]
        hits += [f"{p.name}:{ln}: '{w}' -> {adv}" for ln, w, adv in wording_hits(filled)]
        parts.append(filled.rstrip() + "\n")
    report = "\n".join(parts)
    bib_text, bib_keys, conflicts = merge_bib(bib_paths)
    missing_refs = sorted(cited_keys(report) - set(bib_keys))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    (out_dir / "references.bib").write_text(bib_text, encoding="utf-8", newline="\n")
    log = {
        "sections": [p.name for p in ordered_sections(sections_dir)],
        "numbers_file": str(numbers_path) if numbers_path.exists() else None,
        "unresolved_placeholders": unresolved,
        "todos": todos,
        "wording_hits": hits,
        "missing_references": missing_refs,
        "bib_conflicts": conflicts,
        "n_bib_entries": len(bib_keys),
    }
    (out_dir / "build_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8", newline="\n")
    return log


def to_docx(out_dir: Path) -> str:
    """Convert report.md to report.docx with citeproc when pypandoc_binary is installed."""
    try:
        import pypandoc  # type: ignore
    except ImportError:
        return "pypandoc not installed: pip install pypandoc_binary to enable --docx"
    pypandoc.convert_file(
        str(out_dir / "report.md"), "docx", outputfile=str(out_dir / "report.docx"),
        extra_args=["--citeproc", f"--bibliography={out_dir / 'references.bib'}"],
    )
    return "report.docx written"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default=str(ROOT / "report" / "sections"))
    ap.add_argument("--numbers", default=str(ROOT / "results" / "numbers.json"))
    ap.add_argument("--out", default=str(ROOT / "report" / "out"))
    ap.add_argument("--bib", nargs="*", default=[str(ROOT / "report" / "references.bib"), str(ROOT / "report" / "references_methods.bib")])
    ap.add_argument("--strict", action="store_true", help="exit 1 if anything is unresolved or a wording rule is broken")
    ap.add_argument("--docx", action="store_true")
    a = ap.parse_args(argv)
    log = build(Path(a.sections), Path(a.numbers), Path(a.out), [Path(b) for b in a.bib])
    print(f"sections: {len(log['sections'])} | unresolved: {len(log['unresolved_placeholders'])} | "
          f"todos: {len(log['todos'])} | wording hits: {len(log['wording_hits'])} | "
          f"missing refs: {len(log['missing_references'])} | bib conflicts: {len(log['bib_conflicts'])}")
    for k in ("wording_hits", "missing_references", "bib_conflicts"):
        for item in log[k]:
            print(f"  {k}: {item}")
    if a.docx:
        print(to_docx(Path(a.out)))
    bad = log["unresolved_placeholders"] or log["wording_hits"] or log["missing_references"] or log["bib_conflicts"]
    return 1 if (a.strict and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
