#!/usr/bin/env python3
"""Generate the official benchmark splits S1-S4, 5-fold grouped CV assignments and similarity bins.

For every split ``Sx`` the master table gets
  Sx_split      official partition: train | val | test | purged | excluded
  Sx_cv         5-fold grouped CV fold (-1 = excluded from this split)
  Sx_maxsim     (test rows) maximum similarity of the peptide to any official TRAINING peptide of the same split
  Sx_bin        similarity bin of Sx_maxsim: 100% | 80-99% | 60-79% | 40-59% | <40%
  Sx_cv_maxsim  / Sx_cv_bin   the same, relative to the training portion of the row's own CV fold
S2 and S3 are *purged*: training/validation rows sharing a leakage unit (S2: near-duplicate peptide component;
S3: full leakage group = near-duplicate peptide, same antigen family, identical peptide) with any test row are removed (partition ``purged``), so allele- or taxon-holdout is not undermined
by conserved peptides.  S2 purges on near-duplicate peptide components; S3 on the full leakage group.

Outputs: ``data/benchmark_splits/splits_all_tiers.csv.gz`` and primary-benchmark (P vs N2) train/val/test CSV files per
split in ``data/benchmark_splits/<split>/``; split statistics in ``data/metadata/split_stats.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import DEFAULT_SEED, data_dir, setup_logging, update_manifest  # noqa: E402
from utilities.peptide_sim import SimilarityIndex, similarity_bin  # noqa: E402
from utilities.splits import (N_FOLDS, OFFICIAL_TARGETS, PURGED_SPLITS, SPLITS, assign_to_folds,  # noqa: E402
                              assign_to_targets, cv_train_mask, unit_series, PURGE_COLUMN)

NAMES = {"S1": "S1_antigen_cluster", "S2": "S2_heldout_HLA", "S3": "S3_heldout_pathogen", "S4": "S4_random_reference"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=data_dir("processed/dataset_all_tiers.csv.gz"))
    p.add_argument("--groups", type=Path, default=data_dir("processed/leakage_groups.csv.gz"))
    p.add_argument("--outdir", type=Path, default=data_dir("benchmark_splits"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def maxsim(query_pep: pd.Series, train_pep: pd.Series) -> np.ndarray:
    """Max similarity of each query peptide to the set of training peptides."""
    tr = sorted(set(train_pep))
    if not tr:
        return np.zeros(len(query_pep))
    uq = sorted(set(query_pep))
    best, _ = SimilarityIndex(tr).max_similarity(uq)
    lut = dict(zip(uq, best))
    return query_pep.map(lut).to_numpy(dtype=float)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("10_make_splits", args.verbose)
    for pth in (args.dataset, args.groups):
        if not pth.exists():
            log.error("input not found: %s", pth)
            return 2
    try:
        ds = pd.read_csv(args.dataset, dtype=str, keep_default_na=False)
        gr = pd.read_csv(args.groups, dtype=str)
    except (OSError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 1
    ds = ds.merge(gr[["row_id", "group_id", "peptide_component"]], on="row_id", how="left")
    ds["group_id"] = ds["group_id"].astype(int)
    ds["peptide_component"] = ds["peptide_component"].astype(int)
    rng = np.random.default_rng(args.seed)
    stats = []
    for sp in SPLITS:
        unit = unit_series(ds, sp)
        eligible = unit.ne("")
        sizes = unit[eligible].value_counts()
        off = assign_to_targets(sizes, OFFICIAL_TARGETS, np.random.default_rng(rng.integers(1 << 31)))
        part = unit.map(off).where(eligible, "excluded")
        folds = assign_to_folds(sizes, N_FOLDS, np.random.default_rng(rng.integers(1 << 31)))
        ds[f"{sp}_cv"] = unit.map(folds).where(eligible, -1).astype(int)
        if sp in PURGED_SPLITS:
            pc = PURGE_COLUMN[sp]
            purge = part.isin(["train", "val"]) & ds[pc].isin(set(ds.loc[part == "test", pc]))
            part = part.where(~purge, "purged")
        ds[f"{sp}_split"] = part
        # official similarity to training peptides (test rows only)
        tr_pep = ds.loc[part == "train", "peptide"]
        ds[f"{sp}_maxsim"] = np.nan
        te = part == "test"
        ds.loc[te, f"{sp}_maxsim"] = maxsim(ds.loc[te, "peptide"], tr_pep)
        ds[f"{sp}_bin"] = ""
        ds.loc[te, f"{sp}_bin"] = similarity_bin(ds.loc[te, f"{sp}_maxsim"].to_numpy())
        # CV similarity
        ds[f"{sp}_cv_maxsim"] = np.nan
        for k in range(N_FOLDS):
            tr_mask = cv_train_mask(ds, sp, f"{sp}_cv", k)
            te_mask = (ds[f"{sp}_cv"] == k).to_numpy()
            ds.loc[te_mask, f"{sp}_cv_maxsim"] = maxsim(ds.loc[te_mask, "peptide"], ds.loc[tr_mask, "peptide"])
        ok = ds[f"{sp}_cv"] >= 0
        ds[f"{sp}_cv_bin"] = ""
        ds.loc[ok, f"{sp}_cv_bin"] = similarity_bin(ds.loc[ok, f"{sp}_cv_maxsim"].to_numpy())
        for tier_set, tiers in (("P_vs_N2", ["P", "N2"]),):
            sub = ds[ds["tier"].isin(tiers)]
            for pname in ("train", "val", "test", "purged", "excluded"):
                s = sub[sub[f"{sp}_split"] == pname]
                stats.append({"split": sp, "benchmark": tier_set, "partition": pname, "rows": len(s),
                              "positives": int((s["label"] == "1").sum()), "negatives": int((s["label"] == "0").sum()),
                              "unique_peptides": s["peptide"].nunique(), "alleles": s["allele"].nunique(),
                              "units": unit[s.index].nunique()})
        log.info("%s partitions (all tiers): %s", sp, part.value_counts().to_dict())

    args.outdir.mkdir(parents=True, exist_ok=True)
    keep = (["row_id", "peptide", "allele", "tier", "label", "category", "taxon_group", "in_pathogen_track", "group_id"]
            + [f"{sp}_{c}" for sp in SPLITS for c in ("split", "cv", "maxsim", "bin", "cv_maxsim", "cv_bin")])
    ds[keep].to_csv(args.outdir / "splits_all_tiers.csv.gz", index=False, compression="gzip")
    cols = ["row_id", "peptide", "allele", "label", "tier", "category", "taxon_group", "species", "parent_name", "evidence_tier",
            "n2_confidence", "reference_ids", "group_id"]
    prim = ds[ds["tier"].isin(["P", "N2"])]
    for sp in SPLITS:
        d = args.outdir / NAMES[sp]
        d.mkdir(parents=True, exist_ok=True)
        for pname in ("train", "val", "test"):
            sub = prim[prim[f"{sp}_split"] == pname]
            sub[cols + [f"{sp}_maxsim", f"{sp}_bin"]].to_csv(d / f"primary_P_vs_N2_{pname}.csv", index=False)
    st = pd.DataFrame(stats)
    st.to_csv(data_dir("metadata/split_stats.csv"), index=False)
    log.info("primary benchmark (P vs N2) partition sizes:\n%s", st[st["partition"].isin(["train", "val", "test", "purged"])].to_string(index=False))
    update_manifest("10_make_splits", {"seed": args.seed, "n_folds": N_FOLDS, "targets": OFFICIAL_TARGETS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
