#!/usr/bin/env python3
"""S5: study-held-out split for the primary P-vs-N2 set (addresses the study-identity shortcut left open by S1).

Unit = the first (primary) IEDB reference of a row. Whole units go to a fold. Training rows are purged when they share ANY IEDB
reference or any leakage group with the held-out rows, so S5 also keeps S1's guarantee.  Writes
  data/benchmark_splits/S5_study_heldout.csv.gz   (row_id, S5_unit, S5_cv, S5_split)
  data/metadata/results/study_heldout_N2.csv       (AUPRC/AUROC with cluster-bootstrap CIs, all nine reference models)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from utilities.common import DEFAULT_SEED, data_dir, setup_logging, update_manifest  # noqa: E402
from utilities.splits import N_FOLDS, OFFICIAL_TARGETS, assign_to_folds, assign_to_targets  # noqa: E402
from utilities.views import build_view, load_master  # noqa: E402


def _load_cv():
    spec = importlib.util.spec_from_file_location("cvb", SRC / "modeling" / "11_cv_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shares(df: pd.DataFrame, mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """rows in ``mask_a`` that share a reference id or a leakage group with any row in ``mask_b``."""
    refs_b = set(r for lst in df.loc[mask_b, "refs"] for r in lst)
    grp_b = set(df.loc[mask_b, "group_id"])
    return np.array([m and (g in grp_b or any(r in refs_b for r in lst)) for m, g, lst in zip(mask_a, df["group_id"], df["refs"])])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(argv)
    log = setup_logging("14_study_heldout")
    v = build_view(load_master(), "N2", a.seed, "all").reset_index(drop=True)
    v["refs"] = v["reference_ids"].fillna("").str.split(";").map(lambda x: [r for r in x if r])
    v["S5_unit"] = v["refs"].str[0]
    rng = np.random.default_rng(a.seed)
    sizes = v["S5_unit"].value_counts()
    fold_of = assign_to_folds(sizes, N_FOLDS, np.random.default_rng(a.seed))
    part_of = assign_to_targets(sizes, OFFICIAL_TARGETS, np.random.default_rng(a.seed))
    v["S5_cv"] = v["S5_unit"].map(fold_of).astype(int)
    part = v["S5_unit"].map(part_of)
    test = (part == "test").to_numpy(); val = (part == "val").to_numpy(); train = (part == "train").to_numpy()
    pv = shares(v, val, test); pt = shares(v, train, test | val)
    v["S5_split"] = np.select([test, val & ~pv, train & ~pt], ["test", "val", "train"], "purged")
    out = data_dir("benchmark_splits") / "S5_study_heldout.csv.gz"
    v[["row_id", "S5_unit", "S5_cv", "S5_split"]].to_csv(out, index=False, compression="gzip")
    log.info("official S5 counts: %s", v["S5_split"].value_counts().to_dict())

    cvb = _load_cv()
    def mask(df, split, fold_col, k):
        te = (df[fold_col] == k).to_numpy()
        return (~te) & ~shares(df, np.ones(len(df), bool), te)
    cvb.cv_train_mask = mask
    res = cvb.run_view(v, "S5", cvb.MODELS, a.seed, log)
    res["group_id"] = res["group_id"].astype(str)
    gid = res["group_id"].to_numpy(); y = res["label"].to_numpy()
    groups = {g: np.flatnonzero(gid == g) for g in np.unique(gid)}
    keys = list(groups)
    rows = []
    for m in cvb.MODELS:
        s = res[m].to_numpy(); ok = ~np.isnan(s)
        if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
            continue
        ap_, au = average_precision_score(y[ok], s[ok]), roc_auc_score(y[ok], s[ok])
        bs = []
        for _ in range(a.n_boot):
            idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]); idx = idx[ok[idx]]
            if len(np.unique(y[idx])) < 2:
                continue
            bs.append((average_precision_score(y[idx], s[idx]), roc_auc_score(y[idx], s[idx])))
        bs = np.array(bs)
        rows.append({"split": "S5", "model": m, "n_scored": int(ok.sum()), "n_total": len(res), "prevalence": float(y[ok].mean()),
                     "auprc": ap_, "auprc_lo": np.percentile(bs[:, 0], 2.5), "auprc_hi": np.percentile(bs[:, 0], 97.5),
                     "auroc": au, "auroc_lo": np.percentile(bs[:, 1], 2.5), "auroc_hi": np.percentile(bs[:, 1], 97.5)})
    pd.DataFrame(rows).to_csv(data_dir("metadata/results") / "study_heldout_N2.csv", index=False)
    update_manifest("14_study_heldout", {"seed": a.seed, "units": int(v["S5_unit"].nunique()), "official": v["S5_split"].value_counts().to_dict()})
    print(pd.DataFrame(rows).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
