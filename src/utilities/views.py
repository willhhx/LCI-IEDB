"""Benchmark "views": the row sets used to evaluate each negative tier under every split.

view(tier):
  N2  all positives + assay-verified negatives (the primary benchmark)
  N1  positives that have a matched same-source decoy + those decoys
  N3  positives that have a matched hard decoy + those decoys
  N0  ANALYSIS-ONLY easy reference: all positives + one uniform-random peptide of the same length and allele per positive.
      Random peptides are generated at run time from a seed and are NOT part of the released dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import AA20, data_dir
from .splits import SPLITS

TIERS = ("N2", "N1", "N3", "N0")
KEEP = ["row_id", "peptide", "allele", "label", "tier", "category", "taxon_group", "in_pathogen_track", "reference_ids",
        "group_id", "peptide_component", "matched_pair_id", "evidence_tier", "n2_confidence"] + [f"{s}_cv" for s in SPLITS]


def load_master(root: Path | None = None) -> pd.DataFrame:
    """Dataset rows joined with leakage groups and CV fold assignments."""
    base = data_dir("processed") if root is None else root / "processed"
    ds = pd.read_csv(base / "dataset_all_tiers.csv.gz", dtype=str, keep_default_na=False)
    gr = pd.read_csv(base / "leakage_groups.csv.gz", dtype={"row_id": str})[["row_id", "group_id", "peptide_component"]]
    sp_path = (data_dir("benchmark_splits") if root is None else root / "benchmark_splits") / "splits_all_tiers.csv.gz"
    sp = pd.read_csv(sp_path, dtype={"row_id": str}, usecols=["row_id"] + [f"{s}_cv" for s in SPLITS])
    m = ds.drop(columns=[c for c in ("group_id", "peptide_component") if c in ds.columns]).merge(gr, on="row_id").merge(sp, on="row_id")
    m["label"] = m["label"].astype(int)
    m["in_pathogen_track"] = m["in_pathogen_track"].astype(str).eq("True")
    return m


def random_peptides(lengths: list[int], forbidden: set[str], rng: np.random.Generator) -> list[str]:
    out: list[str] = []
    for n in lengths:
        while True:
            p = "".join(rng.choice(list(AA20), size=n))
            if p not in forbidden:
                forbidden.add(p)
                out.append(p)
                break
    return out


def build_view(master: pd.DataFrame, tier: str, seed: int, track: str = "all") -> pd.DataFrame:
    """Assemble the evaluation rows of ``tier`` (see module docstring); ``track='pathogen'`` keeps pathogen-derived rows."""
    pos = master[master["tier"] == "P"]
    if tier == "N2":
        v = pd.concat([pos, master[master["tier"] == "N2"]])
    elif tier in ("N1", "N3"):
        dec = master[master["tier"] == tier]
        v = pd.concat([pos[pos["row_id"].isin(set(dec["matched_pair_id"]))], dec])
    elif tier == "N0":
        rng = np.random.default_rng(seed)
        rp = random_peptides([len(p) for p in pos["peptide"]], set(master["peptide"]), rng)
        n0 = pos.copy()
        n0["peptide"] = rp
        n0["label"] = 0
        n0["tier"] = "N0"
        n0["row_id"] = [f"N0-{i:06d}" for i in range(1, len(n0) + 1)]
        n0["reference_ids"] = ""
        n0["matched_pair_id"] = pos["row_id"].to_numpy()
        n0["group_id"] = -1 - np.arange(len(n0))
        n0["peptide_component"] = -1 - np.arange(len(n0))
        v = pd.concat([pos, n0])
    else:
        raise ValueError(f"unknown tier {tier}")
    v = v[KEEP].reset_index(drop=True)
    if track == "pathogen":
        v = v[v["in_pathogen_track"]].reset_index(drop=True)
    return v
