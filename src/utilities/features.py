"""Peptide feature builders for the validation baselines (numpy only)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .binding_proxy import featurize as positional_features
from .common import AA20, AA_INDEX, MAX_LEN, MIN_LEN
from .peptide_sim import encode

# Kyte-Doolittle hydropathy, side-chain charge at pH 7, residue mass (Da), polarity flags
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
      "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}
MASS = {"A": 71.04, "R": 156.10, "N": 114.04, "D": 115.03, "C": 103.01, "E": 129.04, "Q": 128.06, "G": 57.02, "H": 137.06,
        "I": 113.08, "L": 113.08, "K": 128.09, "M": 131.04, "F": 147.07, "P": 97.05, "S": 87.03, "T": 101.05, "W": 186.08,
        "Y": 163.06, "V": 99.07}
GROUPS = {"aromatic": "FWY", "aliphatic": "AILMV", "small": "AGSTC", "polar": "NQSTY", "acidic": "DE", "basic": "KRH"}


def composition(peptides: Sequence[str]) -> np.ndarray:
    """(n, 20) amino-acid composition."""
    out = np.zeros((len(peptides), len(AA20)), dtype=np.float32)
    for i, p in enumerate(peptides):
        for c in p:
            out[i, AA_INDEX[c]] += 1
        out[i] /= len(p)
    return out


def length_onehot(peptides: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(peptides), MAX_LEN - MIN_LEN + 1), dtype=np.float32)
    for i, p in enumerate(peptides):
        out[i, len(p) - MIN_LEN] = 1.0
    return out


def dipeptide(peptides: Sequence[str]) -> np.ndarray:
    """(n, 400) overlapping 2-mer frequencies."""
    out = np.zeros((len(peptides), 400), dtype=np.float32)
    for i, p in enumerate(peptides):
        for a, b in zip(p[:-1], p[1:]):
            out[i, AA_INDEX[a] * 20 + AA_INDEX[b]] += 1
        out[i] /= max(len(p) - 1, 1)
    return out


def physchem(peptides: Sequence[str]) -> np.ndarray:
    """(n, 6 + len(GROUPS)) physicochemical descriptors."""
    cols = 6 + len(GROUPS)
    out = np.zeros((len(peptides), cols), dtype=np.float32)
    for i, p in enumerate(peptides):
        kd = [KD[c] for c in p]
        out[i, 0] = np.mean(kd)
        out[i, 1] = np.max(kd)
        out[i, 2] = np.min(kd)
        out[i, 3] = sum(CHARGE.get(c, 0.0) for c in p)
        out[i, 4] = np.mean([MASS[c] for c in p]) / 100.0
        out[i, 5] = np.mean(kd[2:-2]) if len(kd) > 4 else np.mean(kd)  # central hydropathy
        for j, g in enumerate(GROUPS.values()):
            out[i, 6 + j] = sum(c in g for c in p) / len(p)
    return out


def peptide_features(peptides: Sequence[str]) -> np.ndarray:
    """Full peptide feature block: positional terminal one-hots, length, composition, 2-mers, physico-chemistry."""
    return np.hstack([positional_features(peptides), dipeptide(peptides), physchem(peptides)]).astype(np.float32)
