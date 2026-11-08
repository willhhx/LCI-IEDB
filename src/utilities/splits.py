"""Deterministic grouped split utilities shared by split generation and evaluation.

Split definitions (``SPLITS``): the *unit* that is never divided between train and test.
  S1  antigen-cluster split     unit = leakage group (near-duplicate peptides + parent protein + antigen family)
  S2  held-out HLA split        unit = HLA allele        (training rows sharing a near-duplicate peptide component with test are purged)
  S3  pathogen-held-out split   unit = taxon group      (training rows sharing a full leakage group with test are purged; rows without
                                                         a source taxon are excluded)
  S4  random split              unit = row               (reference only: estimates leakage inflation)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS: tuple[str, ...] = ("S1", "S2", "S3", "S4")
PURGED_SPLITS: frozenset[str] = frozenset({"S2", "S3"})
# column whose values may not occur on both sides of a purged split
PURGE_COLUMN: dict[str, str] = {"S2": "peptide_component", "S3": "group_id"}
OFFICIAL_TARGETS: dict[str, float] = {"train": 0.7, "val": 0.1, "test": 0.2}
N_FOLDS: int = 5


def unit_series(df: pd.DataFrame, split: str) -> pd.Series:
    """Unit label per row ('' = row cannot take part in this split)."""
    if split == "S1":
        return df["group_id"].astype(str)
    if split == "S2":
        return df["allele"].astype(str)
    if split == "S3":
        return df["taxon_group"].where(df["taxon_group"].ne("unspecified"), "")
    if split == "S4":
        return df["row_id"].astype(str)
    raise ValueError(f"unknown split {split}")


def assign_to_targets(sizes: pd.Series, targets: dict[str, float], rng: np.random.Generator) -> dict[str, str]:
    """Greedy largest-first assignment of units to parts so that part sizes approach ``targets``."""
    order = sizes.iloc[rng.permutation(len(sizes))].sort_values(ascending=False, kind="stable")
    total = float(sizes.sum())
    load = {k: 0.0 for k in targets}
    out: dict[str, str] = {}
    for unit, size in order.items():
        best = max(targets, key=lambda k: (targets[k] * total - load[k]) / targets[k])
        out[unit] = best
        load[best] += float(size)
    return out


def assign_to_folds(sizes: pd.Series, k: int, rng: np.random.Generator) -> dict[str, int]:
    """Greedy largest-first assignment of units to ``k`` balanced folds."""
    order = sizes.iloc[rng.permutation(len(sizes))].sort_values(ascending=False, kind="stable")
    load = np.zeros(k)
    out: dict[str, int] = {}
    for unit, size in order.items():
        f = int(load.argmin())
        out[unit] = f
        load[f] += float(size)
    return out


def cv_train_mask(df: pd.DataFrame, split: str, fold_col: str, k: int) -> np.ndarray:
    """Boolean mask of rows allowed in training for CV fold ``k`` (purged for S2/S3)."""
    fold = df[fold_col].to_numpy()
    train = (fold != k) & (fold >= 0)
    if split in PURGED_SPLITS:
        col = PURGE_COLUMN[split]
        train &= ~df[col].isin(set(df.loc[fold == k, col])).to_numpy()
    return train
