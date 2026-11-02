"""Tests for scripts/build_report.py (placeholder filling, wording rules, bibliography merge)."""
import json

from scripts import build_report as br


def test_fill_placeholders_styles():
    numbers = {"m.hit_A": 0.6234, "m.hit_A.lo": 0.53, "m.hit_A.hi": 0.701, "pairs.n": 1200, "h1.p": 0.0004, "h3.p": 0.0412}
    text = "hit [[NUM:m.hit_A]] ci [[NUM:m.hit_A|ci]] pct [[NUM:m.hit_A|pct]] n [[NUM:pairs.n]] p [[NUM:h1.p|p]] q [[NUM:h3.p|p]]"
    out, missing = br.fill_placeholders(text, numbers)
    assert missing == []
    assert out == "hit 0.62 ci 0.62 [0.53, 0.70] pct 62% n 1,200 p <0.001 q 0.041"


def test_unresolved_placeholders_are_kept_and_listed():
    out, missing = br.fill_placeholders("a [[NUM:nope]] b [[NUM:x|ci]]", {"x": 0.5})
    assert "[[NUM:nope]]" in out and "[[NUM:x|ci]]" in out
    assert missing == ["nope", "x|ci"]


def test_wording_hits_and_rule_sentences():
    text = "We use radiologist labels here.\nNever say counterfactual; say within-patient contrast.\nThe prompt is verbatim."
    hits = br.wording_hits(text)
    assert [(ln, w.lower()) for ln, w, _ in hits] == [(1, "radiologist label"), (3, "verbatim")]


def test_merge_bib_dedup_and_conflict(tmp_path):
    a = tmp_path / "a.bib"
    b = tmp_path / "b.bib"
    a.write_text("@article{k1,\n title={A}\n}\n\n@article{k2,\n title={B}\n}\n", encoding="utf-8")
    b.write_text("@article{k1,\n  title={A}\n}\n\n@article{k2,\n title={Different}\n}\n@misc{k3,\n title={C}\n}\n", encoding="utf-8")
    text, keys, conflicts = br.merge_bib([a, b])
    assert keys == ["k1", "k2", "k3"]
    assert conflicts == ["k2"]
    assert text.count("@") == 3


def test_build_end_to_end(tmp_path):
    sec = tmp_path / "sections"
    sec.mkdir()
    (sec / "02_related.md").write_text("Related [@k1; @k9].\n", encoding="utf-8")
    (sec / "01_intro.md").write_text("Intro [[NUM:x]] and [[TODO: summary]].\n", encoding="utf-8")
    (sec / "appendix_b.md").write_text("Appendix.\n", encoding="utf-8")
    nums = tmp_path / "numbers.json"
    nums.write_text(json.dumps({"x": 0.25}), encoding="utf-8")
    bib = tmp_path / "r.bib"
    bib.write_text("@article{k1,\n title={A}\n}\n", encoding="utf-8")
    log = br.build(sec, nums, tmp_path / "out", [bib])
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert report.index("Intro 0.25") < report.index("Related") < report.index("Appendix")
    assert log["missing_references"] == ["k9"]
    assert log["todos"] == ["01_intro.md: summary"]
    assert log["unresolved_placeholders"] == []


def test_null_numbers_render_as_na():
    """numbers.json stores undefined numbers as null; every style renders them as n/a."""
    numbers = json.loads('{"h1.m.p_value": null, "h1.m.estimate": 0.1, "h1.m.estimate.lo": null, '
                         '"h1.m.estimate.hi": null, "x": null, "x.lo": null, "x.hi": null, "k.n": null}')
    text = ("p [[NUM:h1.m.p_value|p]] e [[NUM:h1.m.estimate|ci]] x [[NUM:x]] [[NUM:x|pct]] [[NUM:x|ci]] "
            "[[NUM:x|int]] n [[NUM:k.n]]")
    out, missing = br.fill_placeholders(text, numbers)
    assert missing == []
    assert out == "p n/a e 0.10 [n/a, n/a] x n/a n/a n/a [n/a, n/a] n/a n n/a"
    out2, missing2 = br.fill_placeholders("[[NUM:h1.m.p_value|ci]]", {"h1.m.p_value": None})
    assert missing2 == ["h1.m.p_value|ci"] and out2 == "[[NUM:h1.m.p_value|ci]]"  # no .lo/.hi keys at all
