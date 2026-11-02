"""Select the Stage 1 items and write every derived list (PIPELINE_SPEC v1.0, section 3 scripts).

Usage, from the repository root::

    python scripts/select_items.py --match data/match_table.csv --train data/ranzcr/train.csv \
        --nih data/nih/Data_Entry_2017.csv --images-dir data/images --out items/

Writes into ``--out``: ``items.csv`` (S5), ``pod_items.csv`` (S6, the only item file that goes to
the pod), ``nih_filelist.csv``, ``reading_list.csv`` (every A, B and U image), ``reading_key.csv``,
``retest_list.csv`` (S10), ``practice_list.csv`` (10 dev-pool images under their own read ids, for
the student's practice run; never a test image), the seed sidecars ``reading_list.meta.json``,
``retest_list.meta.json`` and ``practice_list.meta.json`` (the reading tool reads the list seed from
them) and ``selection_report.json`` (with the SHA-256 of every CSV). ``--images-dir`` sets
``image_path`` in the reading lists; the directory need not exist yet (the PNGs arrive from the
export kernel, which itself needs ``nih_filelist.csv``).

Labels of record: RANZCR CLiP labels (43 trained labellers: radiologists, radiology trainees,
hospital medical officers), read from the local working copy of ``train.csv``. Everything written
here stays in the git-ignored ``items/`` directory except ``pod_items.csv`` (opaque ids and NIH
filenames only) and ``nih_filelist.csv`` (NIH filenames only); see PROTOCOL.md section 9. This
script never touches the network and prints aggregates only.

Optional flags implement the registered cut decisions (``--n-pairs 80``, ``--n-dev-pos 100
--n-dev-neg 100``) and the registered H1 decision from the Day-1 yield table
(``--h1-substitute yes|no``; default ``auto`` applies the rule to the selection pool). The count
options (``--n-u-per-stratum``, ``--n-retest``, ``--n-smoke``, ``--n-practice``) also let a test run
on a small simulated pool; the defaults are the registered values in ``cxrdev/constants.py``.

``--exclude-filenames PATH`` names a text file of NIH filenames (one per line; "#" comments and blank
lines ignored) removed from the matched pool before any selection (pairs, U and the dev pool); the
report records the file's SHA-256, the number of names listed and how many were in the matched pool.
The registered run passes ``results/probes/probe_seen_filenames.txt`` (public NIH images that
received model or detector outputs during software tests before the protocol freeze).

``train.csv`` is read with ``cxrdev.ranzcr.read_train_csv`` (StudyInstanceUID and PatientID kept as
text, no NA parsing), so identifiers with leading zeros or digits only join the match table as text.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
from collections.abc import Sequence  # noqa: E402

import pandas as pd  # noqa: E402

from cxrdev import constants as C  # noqa: E402
from cxrdev import items as items_mod  # noqa: E402

_H1_CHOICES = {"auto": None, "yes": True, "no": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (see the module docstring)."""
    p = argparse.ArgumentParser(
        prog="select_items.py",
        description="Select Stage 1 items (A/B pairs, U strata, dev pool) and write the derived lists.",
    )
    p.add_argument("--match", required=True, type=Path, help="S3 match table CSV (matching.match_all output)")
    p.add_argument("--train", required=True, type=Path, help="RANZCR CLiP train.csv (local working copy)")
    p.add_argument("--nih", required=True, type=Path, help="NIH Data_Entry_2017.csv (Kaggle or v2020 layout)")
    p.add_argument("--images-dir", required=True, help="directory of the selected NIH PNGs (sets image_path)")
    p.add_argument("--out", required=True, type=Path, help="output directory (git-ignored items/)")
    p.add_argument("--seed", type=int, default=C.SEED, help="master seed (default constants.SEED)")
    p.add_argument("--n-pairs", type=int, default=C.N_PAIRS)
    p.add_argument("--n-u-per-stratum", type=int, default=C.N_U_PER_STRATUM)
    p.add_argument("--n-dev-pos", type=int, default=C.N_DEV_POS)
    p.add_argument("--n-dev-neg", type=int, default=C.N_DEV_NEG)
    p.add_argument("--n-read-a", type=int, default=C.N_READ_A)
    p.add_argument("--n-retest", type=int, default=C.N_RETEST)
    p.add_argument("--n-smoke", type=int, default=C.N_SMOKE)
    p.add_argument("--n-practice", type=int, default=items_mod.PRACTICE_N,
                   help="dev-pool images in practice_list.csv (default 10)")
    p.add_argument(
        "--h1-substitute", choices=sorted(_H1_CHOICES), default="auto",
        help="auto: apply the H1 substitute rule to the selection pool; yes/no: force the registered decision",
    )
    p.add_argument(
        "--no-descriptive-noline", action="store_true",
        help="under the H1 substitute, drop the descriptive no-line U stratum (kept by default, PROTOCOL section 2)",
    )
    p.add_argument("--exclude-filenames", type=Path, default=None, metavar="PATH",
                   help="text file of NIH filenames (one per line, # comments) removed before any selection")
    p.add_argument("--force", action="store_true", help="overwrite existing output files")
    return p.parse_args(argv)


def read_match_table(path: Path) -> pd.DataFrame:
    """Read an S3 match table CSV with text identifiers kept as text ("" for unmatched filenames)."""
    text_cols = ("uid", "ranzcr_patient_id", "nih_filename", "status", "key_used", "route")
    df = pd.read_csv(path, dtype={c: str for c in text_cols})
    if "nih_filename" in df.columns:
        df["nih_filename"] = df["nih_filename"].fillna("")
    return df


def load_inputs(match_path: Path, train_path: Path, nih_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the S3, S1 and S2 frames; ``cxrdev.ranzcr`` and ``cxrdev.nih`` are imported lazily."""
    ranzcr = importlib.import_module("cxrdev.ranzcr")
    nih = importlib.import_module("cxrdev.nih")
    match_table = read_match_table(match_path)
    train_df = ranzcr.read_train_csv(train_path)
    presence = ranzcr.presence(train_df)
    nih_meta = nih.load_nih_meta(nih_path)
    return match_table, presence, nih_meta


def _summary_lines(report: dict, out_dir: Path, report_sha: str) -> list[str]:
    """Aggregate-only summary for the console."""
    counts = report["counts"]
    u = report["u"]
    return [
        f"rung: {report['rung']} (pairs {report['pairs']['n_selected']} of {report['pairs']['n_target']} requested; "
        f"{report['n_eligible_patients']} eligible NIH patients)",
        f"arms: {counts['by_arm']}",
        f"strata: {counts['by_stratum']}",
        f"h1_substitute: {report['h1_substitute']} ({u['h1_substitute_rule']}; "
        f"{u['n_noline_patients_available']} no-line patients available); short strata: {u['short_strata']}",
        f"view matched: {u['view_matched']} (target AP share {u['target_ap_share']})",
        f"dev short: {report['dev']['short']}; smoke: {report['smoke']}",
        *([f"excluded filenames: {report['excluded_filenames']['n_listed']} listed, "
           f"{report['excluded_filenames']['n_present_in_matched_pool']} were in the matched pool and were removed "
           "before selection"] if report.get("excluded_filenames") else []),
        f"reading list {report['reading']['n_reading_list']}, retest {report['reading']['n_retest']}, "
        f"practice {report['reading'].get('n_practice')}",
        f"wrote {len(items_mod.OUTPUT_FILES) + len(items_mod.META_FILES)} files to {out_dir}; "
        f"{items_mod.OUTPUT_FILES['selection_report']} sha256 {report_sha}",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selection; return 0 on success and 2 on a data, yield or overwrite error."""
    args = parse_args(argv)
    try:
        match_table, presence, nih_meta = load_inputs(args.match, args.train, args.nih)
        master = items_mod.build_master_table(match_table, presence, nih_meta)
        exclude, exclude_source = None, None
        if args.exclude_filenames is not None:
            exclude, sha = items_mod.read_filename_list(args.exclude_filenames)
            exclude_source = {"file": args.exclude_filenames.name, "sha256": sha}
        outputs = items_mod.build_outputs(
            master,
            args.images_dir,
            seed=args.seed,
            n_pairs=args.n_pairs,
            n_u_per_stratum=args.n_u_per_stratum,
            n_dev_pos=args.n_dev_pos,
            n_dev_neg=args.n_dev_neg,
            n_read_a=args.n_read_a,
            n_retest=args.n_retest,
            n_smoke=args.n_smoke,
            h1_substitute=_H1_CHOICES[args.h1_substitute],
            keep_noline_descriptive=not args.no_descriptive_noline,
            n_practice=args.n_practice,
            exclude_filenames=exclude,
            exclude_source=exclude_source,
        )
        report = items_mod.write_outputs(outputs, args.out, overwrite=args.force)
    except (ValueError, FileExistsError) as exc:
        print(f"select_items.py: error: {exc}", file=sys.stderr)
        return 2
    report_path = args.out / items_mod.OUTPUT_FILES["selection_report"]
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    for line in _summary_lines(report, args.out, report_sha):
        print(line)
    if not Path(args.images_dir).is_dir():
        print(f"note: {args.images_dir} does not exist yet; image_path points there for the reading tool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
