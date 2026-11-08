#!/usr/bin/env python3
"""Measure how many held-out rows are exposed to each leakage vector under each split (primary P vs N2 benchmark, 5-fold CV).

For every split and each test fold, a test row is "exposed" if the training portion of the fold (after purging) contains
  identical_peptide        the same peptide string (any allele)
  near_duplicate           a peptide with similarity >= 0.8 (single substitutions, nested windows)
  same_leakage_group       a row from the same leakage group (near-duplicate / same parent protein / antigen family)
  same_allele              the same HLA allele
  same_taxon_group         the same source taxon group (rows with unspecified taxon are ignored)
  same_study               the same first IEDB reference (rows with a reference only)
Output: ``data/metadata/results/leakage_exposure_<track>.csv``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import DEFAULT_SEED, data_dir, setup_logging, update_manifest  # noqa: E402
from utilities.peptide_sim import SimilarityIndex  # noqa: E402
from utilities.splits import N_FOLDS, SPLITS, cv_train_mask  # noqa: E402
from utilities.views import build_view, load_master  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="all", choices=["all", "pathogen"])
    p.add_argument("--outdir", type=Path, default=data_dir("metadata/results"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("13_leakage_exposure", args.verbose)
    try:
        view = build_view(load_master(), "N2", args.seed, args.track)
    except (OSError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    ref = view["reference_ids"].str.split(";").str[0].fillna("")
    rows = []
    for sp in SPLITS:
        fc = f"{sp}_cv"
        ok = (view[fc] >= 0).to_numpy()
        exp = {k: np.zeros(len(view), bool) for k in ("identical_peptide", "near_duplicate", "same_leakage_group", "same_allele",
                                                       "same_taxon_group", "same_study")}
        for k in range(N_FOLDS):
            tr = cv_train_mask(view, sp, fc, k)
            te = (view[fc] == k).to_numpy()
            if not te.any() or not tr.any():
                continue
            trp = sorted(set(view.loc[tr, "peptide"]))
            uq = sorted(set(view.loc[te, "peptide"]))
            _, val = SimilarityIndex(trp).topk(uq, 1)
            lut = dict(zip(uq, val[:, 0]))
            ms = view.loc[te, "peptide"].map(lut).to_numpy()
            exp["identical_peptide"][te] = ms >= 1 - 1e-9
            exp["near_duplicate"][te] = ms >= 0.8 - 1e-9
            exp["same_leakage_group"][te] = view.loc[te, "group_id"].isin(set(view.loc[tr, "group_id"])).to_numpy()
            exp["same_allele"][te] = view.loc[te, "allele"].isin(set(view.loc[tr, "allele"])).to_numpy()
            exp["same_taxon_group"][te] = view.loc[te, "taxon_group"].isin(set(view.loc[tr, "taxon_group"])).to_numpy()
            exp["same_study"][te] = ref[te].isin(set(ref[tr]) - {""}).to_numpy()
        for name, arr in exp.items():
            mask = ok.copy()
            if name == "same_study":
                mask &= (ref != "").to_numpy()
            if name == "same_taxon_group":
                mask &= (view["taxon_group"] != "unspecified").to_numpy()
            rows.append({"track": args.track, "split": sp, "vector": name, "rows": int(mask.sum()),
                         "fraction_exposed": float(arr[mask].mean()) if mask.any() else np.nan})
    out = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outdir / f"leakage_exposure_{args.track}.csv", index=False)
    log.info("\n%s", out.pivot(index="vector", columns="split", values="fraction_exposed").round(3).to_string())
    update_manifest(f"13_leakage_exposure_{args.track}", {"rows": len(view)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
