#!/usr/bin/env python3
"""Train and cross-validate the per-allele binding-strength proxy used for N3 matching.

Training data: exact (non-censored) measured affinities in ``ligand_pair_evidence.csv.gz``.
Only alleles with at least ``--min-peptides`` measured peptides receive a model.  5-fold cross-validated Spearman
correlation and binder (<=500 nM) AUC are written per allele to ``data/metadata/binding_proxy_cv.csv``.

Outputs: ``data/processed/binding_proxy_models.json``, ``data/metadata/binding_proxy_cv.csv``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.binding_proxy import BindingProxy, featurize  # noqa: E402
from utilities.common import DEFAULT_SEED, data_dir, setup_logging, update_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evidence", type=Path, default=data_dir("processed/ligand_pair_evidence.csv.gz"))
    p.add_argument("--min-peptides", type=int, default=150)
    p.add_argument("--alpha", type=float, default=3.0, help="ridge penalty")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("06_train_binding_proxy", args.verbose)
    if not args.evidence.exists():
        log.error("input not found: %s", args.evidence)
        return 2
    try:
        ev = pd.read_csv(args.evidence, dtype={"peptide": str, "allele": str}, keep_default_na=False,
                         usecols=["peptide", "allele", "mhc_log10_nM", "n_qb_pos"])
    except (OSError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    ev["mhc_log10_nM"] = pd.to_numeric(ev["mhc_log10_nM"], errors="coerce")
    ev = ev.dropna(subset=["mhc_log10_nM"])
    ev["y"] = ev["mhc_log10_nM"].clip(0.5, 5.0)
    counts = ev.groupby("allele").size()
    keep = counts[counts >= args.min_peptides].index
    log.info("alleles with >=%d measured peptides: %d (of %d)", args.min_peptides, len(keep), counts.size)
    models: dict[str, dict] = {}
    rows = []
    for allele in keep:
        d = ev[ev["allele"] == allele]
        x = featurize(d["peptide"].tolist())
        y = d["y"].to_numpy(dtype=np.float64)
        oof = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=args.seed).split(x):
            oof[te] = Ridge(alpha=args.alpha).fit(x[tr], y[tr]).predict(x[te])
        rho = spearmanr(oof, y).statistic
        binder = (y <= np.log10(500)).astype(int)
        auc = roc_auc_score(binder, -oof) if 0 < binder.sum() < len(binder) else np.nan
        full = Ridge(alpha=args.alpha).fit(x, y)
        models[allele] = {"coef": full.coef_.astype(float).tolist(), "intercept": float(full.intercept_), "n_train": int(len(d)),
                          "cv_spearman": float(rho)}
        rows.append({"allele": allele, "n_peptides": len(d), "cv_spearman": rho, "cv_auc_binder_500nM": auc})
    cv = pd.DataFrame(rows).sort_values("n_peptides", ascending=False)
    cv.to_csv(data_dir("metadata/binding_proxy_cv.csv"), index=False)
    BindingProxy(models).save(data_dir("processed/binding_proxy_models.json"))
    log.info("models=%d  median CV Spearman=%.3f  median CV AUC=%.3f", len(models), cv["cv_spearman"].median(),
             cv["cv_auc_binder_500nM"].median())
    log.info("top alleles:\n%s", cv.head(8).to_string(index=False))
    update_manifest("06_train_binding_proxy", {"alleles": len(models), "median_cv_spearman": float(cv["cv_spearman"].median()),
                                               "median_cv_auc": float(cv["cv_auc_binder_500nM"].median()), "seed": args.seed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
