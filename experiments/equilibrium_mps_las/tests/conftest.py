"""Test path setup for the isolated experiment."""

import sys
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
ALRIS_ROOT = Path(__file__).resolve().parents[5]
PROJECT = ALRIS_ROOT / "projects" / "quasisymmetry"

for path in (EXPERIMENT, PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
