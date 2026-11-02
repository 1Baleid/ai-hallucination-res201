"""Pytest root configuration: make the repository root importable (package `cxrdev`, `pod`, `tools`, `kaggle`)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
