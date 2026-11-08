#!/usr/bin/env python3
"""Grouped cross-validated validation baselines for every (negative tier x split) benchmark view.

Models are NOT the contribution; they quantify (i) how much apparent performance is shortcut / leakage and (ii) which simple
methods stay robust.  Peptide-level feature models never see organism, study or tier information.

  B1_length            logistic regression on length one-hot
  B2_composition       logistic regression on amino-acid composition
  B3_organism          logistic regression on source taxon group (the "source organism" shortcut)
  B3b_allele           logistic regression on HLA allele one-hot (allele prior)
  B4_nearest_neighbour similarity-weighted label of the 5 most similar training peptides (memorisation baseline)
  B5_study_prior       smoothed positive rate of the row's first IEDB reference seen in training (study leakage)
  LR / RF / GBM        logistic regression / random forest / histogram gradient boosting on terminal-residue one-hots,
                       length, composition, 2-mers, physicochemical descriptors and HLA allele one-hot
(GBM = scikit-learn HistGradientBoosting.)

Out-of-fold predictions go to ``data/processed/oof/oof_<track>_<tier>_<split>.csv.gz`` (resumable).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import DEFAULT_SEED, data_dir, setup_logging, update_manifest, package_versions  # noqa: E402
from utilities.features import composition, length_onehot, peptide_features  # noqa: E402
from utilities.peptide_sim import SimilarityIndex, similarity_bin  # noqa: E402
from utilities.splits import N_FOLDS, SPLITS, cv_train_mask  # noqa: E402
from utilities.views import TIERS, build_view, load_master  # noqa: E402

MODELS = ["B1_length", "B2_composition", "B3_organism", "B3b_allele", "B4_nearest_neighbour", "B5_study_prior", "LR", "RF", "GBM"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--track", default="all", choices=["all", "pathogen"])
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--outdir", type=Path, default=data_dir("processed/oof"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--force", action="store_true", help="recompute even if the output file exists")
    p.add_argument("--max-rows", type=int, default=0, help="debug: subsample the view to this many rows")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def onehot(values: pd.Series, vocab: list[str]) -> np.ndarray:
    lut = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for r, v in enumerate(values):
        j = lut.get(v)
        if j is not None:
            out[r, j] = 1.0
    return out


def lr_scores(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, c: float, scale: bool) -> np.ndarray:
    if len(np.unique(ytr)) < 2:
        return np.full(len(xte), float(ytr.mean()) if len(ytr) else 0.5)
    if scale:
        sc = StandardScaler().fit(xtr)
        xtr, xte = sc.transform(xtr), sc.transform(xte)
    return LogisticRegression(C=c, max_iter=2000).fit(xtr, ytr).predict_proba(xte)[:, 1]


def run_view(view: pd.DataFrame, split: str, models: list[str], seed: int, log) -> pd.DataFrame:
    fold_col = f"{split}_cv"
    v = view[view[fold_col] >= 0].reset_index(drop=True)
    peps = v["peptide"].tolist()
    uniq = sorted(set(peps))
    pos_of = {p: i for i, p in enumerate(uniq)}
    ridx = np.fromiter((pos_of[p] for p in peps), int, len(peps))
    block = peptide_features(uniq)[ridx]
    alleles = sorted(v["allele"].unique())
    x_allele = onehot(v["allele"], alleles)
    x_full = np.hstack([block, x_allele])
    x_len, x_comp = length_onehot(peps), composition(peps)
    x_tax = onehot(v["taxon_group"], sorted(v["taxon_group"].unique()))
    ref = v["reference_ids"].str.split(";").str[0].fillna("").to_numpy()
    y = v["label"].to_numpy()
    out = pd.DataFrame({"row_id": v["row_id"], "tier": v["tier"], "fold": v[fold_col], "label": y, "group_id": v["group_id"],
                        "maxsim": np.nan, "bin": ""})
    for m in models:
        out[m] = np.nan
    for k in range(N_FOLDS):
        tr = cv_train_mask(v, split, fold_col, k)
        te = (v[fold_col] == k).to_numpy()
        if te.sum() == 0 or tr.sum() < 30:
            log.warning("  fold %d skipped (train=%d test=%d)", k, tr.sum(), te.sum())
            continue
        ytr = y[tr]
        # nearest neighbours / similarity bins computed against the exact training set of this fold
        trp = v.loc[tr, ["peptide", "label"]].groupby("peptide")["label"].agg(["mean", "size"])
        index = SimilarityIndex(list(trp.index))
        uq = sorted(set(v.loc[te, "peptide"]))
        idx, val = index.topk(uq, 5)
        w = val ** 4 + 1e-9
        nn = (w * trp["mean"].to_numpy()[idx]).sum(1) / w.sum(1)
        lut_nn, lut_ms = dict(zip(uq, nn)), dict(zip(uq, val[:, 0]))
        out.loc[te, "maxsim"] = v.loc[te, "peptide"].map(lut_ms).to_numpy()
        for m in models:
            if m == "B1_length":
                s = lr_scores(x_len[tr], ytr, x_len[te], 1.0, False)
            elif m == "B2_composition":
                s = lr_scores(x_comp[tr], ytr, x_comp[te], 1.0, True)
            elif m == "B3_organism":
                s = lr_scores(x_tax[tr], ytr, x_tax[te], 1.0, False)
            elif m == "B3b_allele":
                s = lr_scores(x_allele[tr], ytr, x_allele[te], 1.0, False)
            elif m == "B4_nearest_neighbour":
                s = v.loc[te, "peptide"].map(lut_nn).to_numpy()
            elif m == "B5_study_prior":
                d = pd.DataFrame({"ref": ref[tr], "y": ytr})
                d = d[d["ref"] != ""].groupby("ref")["y"].agg(["sum", "size"])
                prior = float(ytr.mean())
                a = 2.0
                tab = ((d["sum"] + a * prior) / (d["size"] + a)).to_dict()
                s = np.array([tab.get(r, prior) if r else prior for r in ref[te]])
            elif m == "LR":
                s = lr_scores(x_full[tr], ytr, x_full[te], 0.05, True)
            elif m == "RF":
                s = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt", n_jobs=2,
                                           random_state=seed + k).fit(x_full[tr], ytr).predict_proba(x_full[te])[:, 1]
            elif m == "GBM":
                s = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=24, l2_regularization=1.0,
                                                   random_state=seed + k).fit(x_full[tr], ytr).predict_proba(x_full[te])[:, 1]
            else:
                raise ValueError(m)
            out.loc[te, m] = s
    out["bin"] = np.where(out["maxsim"].notna(), similarity_bin(out["maxsim"].fillna(0).to_numpy()), "")
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("11_cv_baselines", args.verbose)
    try:
        master = load_master()
    except (OSError, ValueError) as exc:
        log.error("cannot load benchmark tables: %s", exc)
        return 2
    args.outdir.mkdir(parents=True, exist_ok=True)
    for tier in args.tiers:
        view = build_view(master, tier, args.seed, args.track)
        if args.max_rows:
            view = view.sample(min(args.max_rows, len(view)), random_state=args.seed).reset_index(drop=True)
        log.info("== track=%s tier=%s rows=%d positives=%d", args.track, tier, len(view), int(view["label"].sum()))
        for sp in args.splits:
            path = args.outdir / f"oof_{args.track}_{tier}_{sp}.csv.gz"
            if path.exists() and not args.force:
                log.info("  %s exists, skipping", path.name)
                continue
            t0 = time.time()
            try:
                res = run_view(view, sp, args.models, args.seed, log)
            except (ValueError, MemoryError) as exc:
                log.error("  %s %s failed: %s", tier, sp, exc)
                return 1
            res.to_csv(path, index=False, compression="gzip")
            log.info("  %s %s: %d rows, %.0fs -> %s", tier, sp, len(res), time.time() - t0, path.name)
    update_manifest("11_cv_baselines", {"seed": args.seed, "track": args.track, "models": args.models,
                                        "packages": package_versions(["numpy", "pandas", "scikit-learn", "scipy"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
