"""Single source of truth for Stage 1 constants (mirrors PROTOCOL.md v0.4).

Do not change a value here without a dated entry in PROTOCOL.md section 11.
This module must stay importable with the standard library only (it is copied to the
RunPod pod and bundled into Kaggle kernels).
"""
from __future__ import annotations

# ---------------------------------------------------------------- randomness
SEED = 20261012  # master seed; derive purpose seeds with derive_seed()

_PURPOSES = {
    "pairs": 1, "u": 2, "dev": 3, "ids": 4, "reading": 5, "retest": 6,
    "smoke": 7, "bootstrap": 8, "permutation": 9, "spotcheck": 10, "model": 11,
}


def derive_seed(purpose: str) -> int:
    """Deterministic per-purpose seed (purposes listed in _PURPOSES)."""
    return SEED * 100 + _PURPOSES[purpose]


# ---------------------------------------------------------------- statistics
ALPHA = 0.05
N_BOOT = 9999          # clustered bootstrap resamples
N_PERM = 9999          # shuffled-expert permutations (H5 primary inference)
MIN_CELL_N = 30        # cells below this get an interval and no test
UNPARSABLE_FLAG = 0.10  # flag a run with > 10% unparsable under P0

# H0 floor gate, evaluated in this order: NO-GO, then GO, else CONDITIONAL (under P0)
GATE = {
    "nogo_ba": 0.60,      # NO-GO if balanced accuracy < 0.60
    "const_lo": 0.10,     # ... or yes-rate < 0.10
    "const_hi": 0.90,     # ... or yes-rate > 0.90
    "go_ba": 0.70,        # GO if BA >= 0.70
    "go_hit": 0.70,       # ... and hit_A >= 0.70
    "go_yes_lo": 0.25,    # ... and 0.25 <= yes-rate <= 0.75
    "go_yes_hi": 0.75,
}

# ---------------------------------------------------------------- Arm 1 (detector-gated answers)
GAMMA = 5.0                 # clip for the expert log-odds
BETA_BASELINE = 0.0         # baseline row: answer yes iff z > 0
BETA_PRIMARY = 1.0          # pre-registered primary row
BETA_GRID = (0.5, 1.0, 2.0)  # dev-pool sensitivity grid (ties -> closest to 1)

# ---------------------------------------------------------------- item counts (CORE)
N_PAIRS = 120
N_PAIRS_RUNG_MIN = 80
N_U_PER_STRATUM = 60
N_DEV_POS = 150
N_DEV_NEG = 150
N_READ_A = 1_000_000   # read EVERY A image (PROTOCOL v0.4: RANZCR labels tracheostomy tubes as ETT, so all A are checked); callers take min(N_READ_A, number of A)
N_RETEST = 100
N_SMOKE = 20           # dev-pool items used for the smoke test
H1_SUBSTITUTE_MIN_PATIENTS = 30  # below this, H1 uses U(>=2 lines) vs U(exactly 1 line)

# ---------------------------------------------------------------- prompts and decoding
CORE_PROMPTS = ("P0", "P1", "P2hi")
DEV_PROMPTS = ("P0",)
PLUS_PROMPTS = ("P2lo",)
MAX_NEW_TOKENS = 16
YES_FORMS = ("Yes", "yes", " Yes", " yes", "YES", " YES")
NO_FORMS = ("No", "no", " No", " no", "NO", " NO")
ABSTAIN_FIRST_FORMS = ("I", " I", "Cannot", "cannot", " cannot", "Unknown", "unknown", " unknown")  # exploratory only
COVERAGE_MIN_MEDIAN = 0.90  # smoke-test assertion on P(yes forms) + P(no forms) under P0
GREY_LEVEL = 128            # BLIND-grey image value (1024 x 1024, 8-bit)
IMAGE_SIDE = 1024

# ---------------------------------------------------------------- models
MODELS = {
    "qwen3vl2b": {
        "hf_id": "Qwen/Qwen3-VL-2B-Instruct", "family": "qwen3vl", "tier": "core",
        "dtype": "bfloat16", "min_pixels": 1024 * 1024, "max_pixels": 1024 * 1024,
        "expected_visual_tokens": 1024, "licence": "Apache-2.0",
    },
    "medgemma15": {
        "hf_id": "google/medgemma-1.5-4b-it", "family": "gemma3", "tier": "core",
        "dtype": "bfloat16", "image_size": 896, "expected_visual_tokens": 256,
        "licence": "Health AI Developer Foundations terms (gated)",
    },
    "chexone": {
        "hf_id": "StanfordAIMI/CheXOne", "family": "qwen25vl", "tier": "plus",
        "dtype": "bfloat16", "min_pixels": 512 * 512, "max_pixels": 512 * 512,
        "expected_visual_tokens": None, "licence": "CC BY-NC 4.0 (Hub card); paper CC BY-NC-ND 4.0",
    },
    "qwen25vl3b": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct", "family": "qwen25vl", "tier": "plus",
        "dtype": "bfloat16", "min_pixels": 512 * 512, "max_pixels": 512 * 512,
        "expected_visual_tokens": None, "licence": "Qwen Research License",
    },
    "nvreason3b": {
        "hf_id": "nvidia/NV-Reason-CXR-3B", "family": "qwen25vl", "tier": "plus-fallback",
        "dtype": "bfloat16", "min_pixels": 512 * 512, "max_pixels": 512 * 512,
        "expected_visual_tokens": None, "licence": "NVIDIA OneWay Noncommercial + Qwen Research License",
    },
}
CORE_MODELS = ("qwen3vl2b", "medgemma15")

EXPERT = {
    "hf_id": "google/medsiglip-448",
    "texts": ("a chest x-ray with an endotracheal tube", "a chest x-ray without an endotracheal tube"),
    "image_size": 448,
    "licence": "Health AI Developer Foundations terms (gated)",
}

# ---------------------------------------------------------------- RANZCR CLiP label columns (train.csv)
RANZCR_UID = "StudyInstanceUID"
RANZCR_PATIENT = "PatientID"
ETT_COLS = ("ETT - Abnormal", "ETT - Borderline", "ETT - Normal")
NGT_COLS = ("NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal")
NGT_INCOMPLETE_COL = "NGT - Incompletely Imaged"
CVC_COLS = ("CVC - Abnormal", "CVC - Borderline", "CVC - Normal")
SWAN_COL = "Swan Ganz Catheter Present"

# ---------------------------------------------------------------- Kaggle sources
KAGGLE_COMPETITION = "ranzcr-clip-catheter-line-classification"
KAGGLE_NIH_DATASET = "nih-chest-xrays/data"
KERNEL_SLUGS = {"hash": "res201-hash", "match": "res201-match", "export": "res201-export"}
FILELIST_DATASET_SLUG = "res201-nih-filelist"
