"""Light-weight allele-specific MHC-I binding-strength proxy.

NOT a state-of-the-art predictor.  A per-allele ridge regression on N-/C-terminal residue identities, length and
composition, trained on IEDB measured affinities (log10 nM).  It exists only so that hard negatives (N3) can be
matched to positives on approximate binding strength when no measurement is available, and it is cross-validated
(see ``data/metadata/binding_proxy_cv.csv``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .common import AA20, MAX_LEN, MIN_LEN
from .peptide_sim import encode

N_TERM = 3
C_TERM = 3
N_FEATURES = (N_TERM + C_TERM) * len(AA20) + (MAX_LEN - MIN_LEN + 1) + len(AA20)


def featurize(peptides: Sequence[str]) -> np.ndarray:
    """(n, N_FEATURES) float32 feature matrix."""
    codes = encode(peptides)
    n = len(peptides)
    lens = (codes >= 0).sum(axis=1)
    x = np.zeros((n, N_FEATURES), dtype=np.float32)
    rows = np.arange(n)
    off = 0
    for k in range(N_TERM):
        x[rows, off + k * len(AA20) + codes[:, k]] = 1.0
    off += N_TERM * len(AA20)
    for k in range(C_TERM):
        x[rows, off + k * len(AA20) + codes[rows, lens - 1 - k]] = 1.0
    off += C_TERM * len(AA20)
    x[rows, off + (lens - MIN_LEN)] = 1.0
    off += MAX_LEN - MIN_LEN + 1
    for i in range(n):
        c = codes[i, : lens[i]]
        x[i, off:off + len(AA20)] = np.bincount(c, minlength=len(AA20)) / lens[i]
    return x


class BindingProxy:
    """Collection of per-allele linear models loaded from / saved to JSON."""

    def __init__(self, models: dict[str, dict]):
        self.models = models

    @classmethod
    def load(cls, path: Path) -> "BindingProxy":
        return cls(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.models))

    def alleles(self) -> list[str]:
        return sorted(self.models)

    def predict(self, allele: str, peptides: Sequence[str]) -> np.ndarray:
        """Predicted log10(nM); NaN if there is no model for ``allele``."""
        m = self.models.get(allele)
        if m is None or len(peptides) == 0:
            return np.full(len(peptides), np.nan)
        return featurize(peptides) @ np.asarray(m["coef"], dtype=np.float32) + m["intercept"]
