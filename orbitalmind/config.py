"""Central configuration: where the data lives and orbit->file mapping.

The raw competition data is read from ``ORBITALMIND_DATA_DIR`` if set, else the
default download location. Nothing here is written to; outputs go to ``outputs/``.
"""
from __future__ import annotations

import os
from pathlib import Path

# Project root (this file is orbitalmind/config.py -> parent.parent is repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs"

# Raw data directory. Prefer an explicit machine-specific override, then the
# repository-local dataset directory, and finally the legacy download path.
_LOCAL_DATA_DIR = REPO_ROOT / "Data_PS-08"
DATA_DIR = Path(os.environ.get("ORBITALMIND_DATA_DIR", _LOCAL_DATA_DIR))
if not DATA_DIR.exists():
    DATA_DIR = Path(r"C:\Users\krant\Downloads\Data_PS-08")

# Reference dataset for validating our own Shapiro-Wilk implementation.
SW_REFERENCE_FILE = DATA_DIR / "SW_ReferenceData.xlsx"
SW_REFERENCE_TARGET = {"W": 0.9810, "p": 0.5840, "H": 0}

# Logical datasets. Each is (orbit_kind, train_file, test_file).
# MEO has two satellites (train/train2, test/test2).
DATASETS = {
    "GEO": {
        "kind": "GEO",
        "train": DATA_DIR / "DATA_GEO_Train.csv",
        "test": DATA_DIR / "DATA_GEO_Test.csv",
    },
    "MEO1": {
        "kind": "MEO",
        "train": DATA_DIR / "DATA_MEO_Train.csv",
        "test": DATA_DIR / "DATA_MEO_Test.csv",
    },
    "MEO2": {
        "kind": "MEO",
        "train": DATA_DIR / "DATA_MEO_Train2.csv",
        "test": DATA_DIR / "DATA_MEO_Test2.csv",
    },
}

# Candidate dominant periodicities (hours) per orbit kind, used as default
# Fourier periods for harmonic models. GEO ~ sidereal day; MEO ~ half-day class.
DEFAULT_PERIODS_HOURS = {
    "GEO": [23.9345, 11.9673, 7.9782, 5.9836],
    "MEO": [11.9673, 5.9836, 3.9891, 2.9918],
}

SIGNIFICANCE_ALPHA = 0.05
