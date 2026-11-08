"""Peptide similarity engine used for leakage control.

Definition
----------
For two peptides ``a`` and ``b`` (length 8-11) the similarity is the best ungapped identity over relative shifts
``d`` in [-3, 3]::

    sim(a, b) = max_d  #{ i : a[i] == b[i + d] }  /  max(len(a), len(b))

so a 9-mer nested in a 10-mer scores 0.9 and a single-residue variant of a 9-mer scores 8/9 = 0.889.  Identical
peptides score 1.0; peptides of different length can never score 1.0.

All-pairs counts are computed as matrix products of one-hot encodings (BLAS), which is exact and fast enough for
~10^5 x 10^4 comparisons on a laptop CPU.  No third-party dependency beyond numpy/scipy.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .common import AA_INDEX, AA20, MAX_LEN

OFFSETS: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
BIN_LABELS: tuple[str, ...] = ("<40%", "40-59%", "60-79%", "80-99%", "100%")


def encode(peptides: Sequence[str]) -> np.ndarray:
    """Integer-encode peptides into an (n, MAX_LEN) int8 array, padded with -1."""
    out = np.full((len(peptides), MAX_LEN), -1, dtype=np.int8)
    for r, p in enumerate(peptides):
        if len(p) > MAX_LEN:
            raise ValueError(f"peptide longer than {MAX_LEN}: {p}")
        out[r, : len(p)] = [AA_INDEX[c] for c in p]
    return out


def _onehot(codes: np.ndarray, shift: int = 0) -> np.ndarray:
    """One-hot (n, MAX_LEN*20); position i holds residue ``codes[:, i + shift]`` (zero if outside)."""
    n = codes.shape[0]
    oh = np.zeros((n, MAX_LEN, len(AA20)), dtype=np.float32)
    for i in range(MAX_LEN):
        j = i + shift
        if 0 <= j < MAX_LEN:
            col = codes[:, j]
            ok = col >= 0
            oh[np.nonzero(ok)[0], i, col[ok]] = 1.0
    return oh.reshape(n, MAX_LEN * len(AA20))


def _lengths(codes: np.ndarray) -> np.ndarray:
    return (codes >= 0).sum(axis=1).astype(np.int16)


class SimilarityIndex:
    """Pre-encoded target set supporting repeated max-similarity queries."""

    def __init__(self, targets: Sequence[str]):
        self.targets = list(targets)
        self.codes = encode(self.targets)
        self.len = _lengths(self.codes)
        self._shifted = {d: _onehot(self.codes, d).T.copy() for d in OFFSETS}  # (MAX_LEN*20, m)

    def __len__(self) -> int:
        return len(self.targets)

    def _block_sim(self, q_oh: np.ndarray, q_len: np.ndarray) -> np.ndarray:
        counts = None
        for d in OFFSETS:
            c = q_oh @ self._shifted[d]
            counts = c if counts is None else np.maximum(counts, c, out=counts)
        maxlen = np.maximum(q_len[:, None], self.len[None, :]).astype(np.float64)
        return counts.astype(np.float64) / maxlen

    def max_similarity(self, queries: Sequence[str], block: int = 400,
                       exclude_identical: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return (max similarity, index of best target) for each query.

        ``exclude_identical`` ignores targets whose string equals the query (useful for within-set queries).
        """
        qc = encode(queries)
        ql = _lengths(qc)
        best = np.zeros(len(queries), dtype=np.float64)
        arg = np.full(len(queries), -1, dtype=np.int64)
        if len(self) == 0:
            return best, arg
        tarr = np.asarray(self.targets, dtype=object)
        for s in range(0, len(queries), block):
            e = min(s + block, len(queries))
            sim = self._block_sim(_onehot(qc[s:e]), ql[s:e])
            if exclude_identical:
                sim[tarr[None, :] == np.asarray(queries[s:e], dtype=object)[:, None]] = -1.0
            a = sim.argmax(axis=1)
            best[s:e] = sim[np.arange(e - s), a]
            arg[s:e] = a
        best[best < 0] = 0.0
        return best, arg

    def topk(self, queries: Sequence[str], k: int, block: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Indices and similarities of the ``k`` most similar targets per query (descending); identical strings are kept."""
        k = min(k, len(self))
        qc = encode(queries)
        ql = _lengths(qc)
        idx = np.zeros((len(queries), k), dtype=np.int64)
        val = np.zeros((len(queries), k), dtype=np.float64)
        for s in range(0, len(queries), block):
            e = min(s + block, len(queries))
            sim = self._block_sim(_onehot(qc[s:e]), ql[s:e])
            part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            ps = np.take_along_axis(sim, part, axis=1)
            o = np.argsort(-ps, axis=1)
            idx[s:e] = np.take_along_axis(part, o, axis=1)
            val[s:e] = np.take_along_axis(ps, o, axis=1)
        return idx, val

    def pairs_above(self, queries: Sequence[str], threshold: float, block: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All (query_idx, target_idx, sim) with sim >= threshold (identical-index pairs are NOT removed)."""
        qc = encode(queries)
        ql = _lengths(qc)
        qi, ti, sv = [], [], []
        for s in range(0, len(queries), block):
            e = min(s + block, len(queries))
            sim = self._block_sim(_onehot(qc[s:e]), ql[s:e])
            r, c = np.nonzero(sim >= threshold - 1e-9)
            qi.append(r + s); ti.append(c); sv.append(sim[r, c])
        if not qi:
            return np.empty(0, int), np.empty(0, int), np.empty(0)
        return np.concatenate(qi), np.concatenate(ti), np.concatenate(sv)


def components_from_pairs(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected-component label per node given an undirected edge list."""
    if len(i) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def similarity_bin(sim: np.ndarray | float) -> np.ndarray:
    """Map similarity in [0, 1] to bin labels ``BIN_LABELS`` (100% means identical sequence)."""
    s = np.atleast_1d(np.asarray(sim, dtype=np.float64))
    idx = np.select([s >= 1.0 - 1e-9, s >= 0.8 - 1e-9, s >= 0.6 - 1e-9, s >= 0.4 - 1e-9], [4, 3, 2, 1], default=0)
    return np.asarray(BIN_LABELS, dtype=object)[idx]


def naive_similarity(a: str, b: str) -> float:
    """Reference (slow) implementation used only by the unit tests."""
    best = 0
    for d in OFFSETS:
        m = sum(1 for i in range(len(a)) if 0 <= i + d < len(b) and a[i] == b[i + d])
        best = max(best, m)
    return best / max(len(a), len(b))


def unique_preserve(seq: Iterable[str]) -> list[str]:
    """Order-preserving de-duplication."""
    return list(dict.fromkeys(seq))
