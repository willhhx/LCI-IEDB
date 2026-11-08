#!/usr/bin/env python3
"""Metrics, leakage-gap, memorisation-ratio and shortcut-ratio analyses from the out-of-fold predictions.

Definitions
  * Threshold metrics (MCC, F1, precision, recall) use a *prevalence-matched* threshold per CV fold: the top q % of the fold's
    scores are called positive, q = label prevalence of the remaining folds.  No test labels are used to choose thresholds.
  * PPV@100 and PPV@Npos (= R-precision): precision among the 100 / Npos highest-scoring rows.
  * Lift = (AUPRC - prevalence) / (1 - prevalence): 0 = chance, 1 = perfect.
  * Leakage gap(model, S) = metric on random split S4 - metric on split S (same rows).  Positive = apparent performance was
    inflated by train-test similarity / shortcuts that S removes.
  * Memorisation ratio(model, S) = lift(B4 nearest neighbour) / lift(model) on split S (model in LR, RF, GBM); only interpretable
    when both lifts are > 0 (otherwise the table/figure scripts show n/a).
  * Shortcut ratio(baseline, S) = lift(baseline) / lift(best of LR, RF, GBM) on split S.
  * 95 % intervals: cluster bootstrap that resamples whole leakage groups (``group_id``), never single rows.

Outputs (``data/metadata/results``): metrics_overall.csv, metrics_by_bin.csv, leakage_gap.csv, memorization_ratio.csv,
shortcut_ratio.csv, study_leakage.csv.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import DEFAULT_SEED, data_dir, setup_logging, update_manifest  # noqa: E402
from utilities.peptide_sim import BIN_LABELS  # noqa: E402
from utilities.splits import N_FOLDS, SPLITS, cv_train_mask  # noqa: E402
from utilities.views import TIERS, build_view, load_master  # noqa: E402

MODELS = ["B1_length", "B2_composition", "B3_organism", "B3b_allele", "B4_nearest_neighbour", "B5_study_prior", "LR", "RF", "GBM"]
ML_MODELS = ["LR", "RF", "GBM"]
BASELINES = [m for m in MODELS if m not in ML_MODELS]
MIN_BIN_ROWS, MIN_BIN_CLASS = 40, 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oof-dir", type=Path, default=data_dir("processed/oof"))
    p.add_argument("--outdir", type=Path, default=data_dir("metadata/results"))
    p.add_argument("--track", default="all", choices=["all", "pathogen"])
    p.add_argument("--tiers", nargs="+", default=list(TIERS), choices=TIERS)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def mcc_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    tp = float(np.sum(p & (y == 1))); tn = float(np.sum(~p & (y == 0)))
    fp = float(np.sum(p & (y == 0))); fn = float(np.sum(~p & (y == 1)))
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return mcc, f1, prec, rec


def ppv_at(y: np.ndarray, s: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(y)))
    top = np.argpartition(-s, k - 1)[:k]
    return float(y[top].mean())


def pred_labels(df: pd.DataFrame, model: str) -> np.ndarray:
    """Prevalence-matched per-fold predicted labels."""
    pred = np.zeros(len(df), dtype=bool)
    y = df["label"].to_numpy()
    fold = df["fold"].to_numpy()
    s = df[model].to_numpy()
    for k in np.unique(fold):
        te = fold == k
        q = y[~te].mean() if (~te).any() else y.mean()
        n_top = max(1, int(round(q * te.sum())))
        thr = np.sort(s[te])[::-1][n_top - 1]
        pred[te] = s[te] >= thr
    return pred


def point_metrics(y: np.ndarray, s: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    prev = float(y.mean())
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": prev}
    ok = 0 < y.sum() < len(y)
    out["auprc"] = float(average_precision_score(y, s)) if ok else np.nan
    out["auroc"] = float(roc_auc_score(y, s)) if ok else np.nan
    out["lift"] = (out["auprc"] - prev) / (1 - prev) if ok and prev < 1 else np.nan
    mcc, f1, prec, rec = mcc_f1(y, pred)
    out.update({"mcc": mcc, "f1": f1, "precision": prec, "recall": rec,
                "ppv_at_100": ppv_at(y, s, 100) if ok else np.nan, "ppv_at_npos": ppv_at(y, s, int(y.sum())) if ok else np.nan})
    return out


class Cluster:
    """Cluster-bootstrap index sampler over ``group_id``."""

    def __init__(self, groups: np.ndarray):
        codes, _ = pd.factorize(groups)
        self.order = np.argsort(codes, kind="stable")
        _, self.start, self.count = np.unique(codes[self.order], return_index=True, return_counts=True)
        self.n = len(self.start)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        pick = rng.integers(0, self.n, self.n)
        return np.concatenate([self.order[self.start[c]:self.start[c] + self.count[c]] for c in pick])


def ci(vals: list[float]) -> tuple[float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) >= 20 else (np.nan, np.nan)


def boot_overall(y, s, pred, cl: Cluster, nb: int, rng) -> dict[str, tuple[float, float]]:
    acc: dict[str, list[float]] = {k: [] for k in ("auprc", "auroc", "mcc", "ppv_at_npos", "lift")}
    for _ in range(nb):
        i = cl.sample(rng)
        yy, ss, pp = y[i], s[i], pred[i]
        if yy.sum() in (0, len(yy)):
            continue
        ap = average_precision_score(yy, ss)
        acc["auprc"].append(ap)
        acc["auroc"].append(roc_auc_score(yy, ss))
        acc["mcc"].append(mcc_f1(yy, pp)[0])
        acc["ppv_at_npos"].append(ppv_at(yy, ss, int(yy.sum())))
        prev = yy.mean()
        acc["lift"].append((ap - prev) / (1 - prev))
    return {k: ci(v) for k, v in acc.items()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("12_metrics", args.verbose)
    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    oof: dict[tuple[str, str], pd.DataFrame] = {}
    for tier, sp in itertools.product(args.tiers, SPLITS):
        path = args.oof_dir / f"oof_{args.track}_{tier}_{sp}.csv.gz"
        if not path.exists():
            log.warning("missing %s", path.name)
            continue
        d = pd.read_csv(path, dtype={"row_id": str})
        oof[(tier, sp)] = d[d[MODELS].notna().all(axis=1)].reset_index(drop=True)
    if not oof:
        log.error("no OOF files found in %s", args.oof_dir)
        return 2

    overall, by_bin = [], []
    t0 = time.time()
    for (tier, sp), d in oof.items():
        y = d["label"].to_numpy()
        cl = Cluster(d["group_id"].to_numpy())
        for m in MODELS:
            s = d[m].to_numpy()
            pred = pred_labels(d, m)
            row = {"track": args.track, "tier": tier, "split": sp, "model": m, **point_metrics(y, s, pred)}
            for k, (lo, hi) in boot_overall(y, s, pred, cl, args.n_boot, rng).items():
                row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
            overall.append(row)
            for b in BIN_LABELS:
                mk = (d["bin"] == b).to_numpy()
                if mk.sum() >= MIN_BIN_ROWS and min(y[mk].sum(), (1 - y[mk]).sum()) >= MIN_BIN_CLASS:
                    by_bin.append({"track": args.track, "tier": tier, "split": sp, "model": m, "bin": b,
                                   **point_metrics(y[mk], s[mk], pred[mk])})
        log.info("overall metrics %s %s done (%.0fs)", tier, sp, time.time() - t0)
    ov = pd.DataFrame(overall)
    ov.to_csv(args.outdir / f"metrics_overall_{args.track}.csv", index=False)
    pd.DataFrame(by_bin).to_csv(args.outdir / f"metrics_by_bin_{args.track}.csv", index=False)

    # ---------------- leakage gap vs the random split S4 ----------------
    gaps = []
    for tier in args.tiers:
        if (tier, "S4") not in oof:
            continue
        ref = oof[(tier, "S4")]
        for sp in ("S1", "S2", "S3"):
            if (tier, sp) not in oof:
                continue
            tst = oof[(tier, sp)]
            common = ref.merge(tst, on="row_id", suffixes=("_ref", "_tst"))
            if len(common) < 100:
                continue
            y = common["label_ref"].to_numpy()
            cl = Cluster(common["group_id_ref"].to_numpy())
            for m in MODELS:
                sr, st = common[f"{m}_ref"].to_numpy(), common[f"{m}_tst"].to_numpy()
                dr = common[["label_ref", "fold_ref", f"{m}_ref"]].rename(columns={"label_ref": "label", "fold_ref": "fold", f"{m}_ref": m})
                dt = common[["label_tst", "fold_tst", f"{m}_tst"]].rename(columns={"label_tst": "label", "fold_tst": "fold", f"{m}_tst": m})
                pr, pt = pred_labels(dr, m), pred_labels(dt, m)
                a, b = point_metrics(y, sr, pr), point_metrics(y, st, pt)
                accg: dict[str, list[float]] = {"auprc": [], "auroc": [], "mcc": [], "lift": []}
                for _ in range(args.n_boot):
                    i = cl.sample(rng)
                    yy = y[i]
                    if yy.sum() in (0, len(yy)):
                        continue
                    prev = yy.mean()
                    apr, apt = average_precision_score(yy, sr[i]), average_precision_score(yy, st[i])
                    accg["auprc"].append(apr - apt)
                    accg["auroc"].append(roc_auc_score(yy, sr[i]) - roc_auc_score(yy, st[i]))
                    accg["mcc"].append(mcc_f1(yy, pr[i])[0] - mcc_f1(yy, pt[i])[0])
                    accg["lift"].append((apr - apt) / (1 - prev))
                row = {"track": args.track, "tier": tier, "model": m, "reference_split": "S4", "split": sp, "n_common_rows": len(common)}
                for k in ("auprc", "auroc", "mcc", "lift"):
                    ref_v = a[k] if k != "lift" else a["lift"]
                    tst_v = b[k] if k != "lift" else b["lift"]
                    row[f"{k}_S4"], row[f"{k}_split"], row[f"{k}_gap"] = ref_v, tst_v, ref_v - tst_v
                    row[f"{k}_gap_lo"], row[f"{k}_gap_hi"] = ci(accg[k])
                gaps.append(row)
        log.info("leakage gap %s done (%.0fs)", tier, time.time() - t0)
    pd.DataFrame(gaps).to_csv(args.outdir / f"leakage_gap_{args.track}.csv", index=False)

    # ---------------- memorisation and shortcut ratios (point estimates + bootstrap CI) ----------------
    mem, sc = [], []
    for (tier, sp), d in oof.items():
        y = d["label"].to_numpy()
        cl = Cluster(d["group_id"].to_numpy())
        scores = {m: d[m].to_numpy() for m in MODELS}
        lifts = {m: point_metrics(y, scores[m], np.zeros(len(y), bool))["lift"] for m in MODELS}
        best_ml = max(ML_MODELS, key=lambda m: lifts[m])
        boot = {m: [] for m in MODELS}
        for _ in range(args.n_boot):
            i = cl.sample(rng)
            yy = y[i]
            if yy.sum() in (0, len(yy)):
                for m in MODELS:
                    boot[m].append(np.nan)
                continue
            prev = yy.mean()
            for m in MODELS:
                boot[m].append((average_precision_score(yy, scores[m][i]) - prev) / (1 - prev))
        nn = np.asarray(boot["B4_nearest_neighbour"])
        for m in ML_MODELS:
            lo, hi = ci(list(nn / np.asarray(boot[m])))
            mem.append({"track": args.track, "tier": tier, "split": sp, "model": m, "lift_nn": lifts["B4_nearest_neighbour"],
                        "lift_model": lifts[m], "memorization_ratio": lifts["B4_nearest_neighbour"] / lifts[m] if lifts[m] > 0 else np.nan,
                        "ratio_lo": lo, "ratio_hi": hi})
        ref = np.asarray(boot[best_ml])
        for m in BASELINES:
            lo, hi = ci(list(np.asarray(boot[m]) / ref))
            sc.append({"track": args.track, "tier": tier, "split": sp, "baseline": m, "reference_model": best_ml,
                       "lift_baseline": lifts[m], "lift_reference": lifts[best_ml],
                       "shortcut_ratio": lifts[m] / lifts[best_ml] if lifts[best_ml] > 0 else np.nan, "ratio_lo": lo, "ratio_hi": hi})
    pd.DataFrame(mem).to_csv(args.outdir / f"memorization_ratio_{args.track}.csv", index=False)
    pd.DataFrame(sc).to_csv(args.outdir / f"shortcut_ratio_{args.track}.csv", index=False)

    # ---------------- study leakage: test rows whose first reference also contributes training rows ----------------
    master = load_master()
    rows = []
    view = build_view(master, "N2", args.seed, args.track)
    ref = view["reference_ids"].str.split(";").str[0].fillna("")
    for sp in SPLITS:
        fc = f"{sp}_cv"
        seen = np.zeros(len(view), dtype=bool)
        for k in range(N_FOLDS):
            tr = cv_train_mask(view, sp, fc, k)
            te = (view[fc] == k).to_numpy()
            seen[te] = ref[te].isin(set(ref[tr])).to_numpy()
        ok = (view[fc] >= 0).to_numpy() & (ref != "").to_numpy()
        rows.append({"track": args.track, "split": sp, "rows_with_reference": int(ok.sum()),
                     "fraction_sharing_reference_with_train": float(seen[ok].mean()),
                     "positives": float(seen[ok & (view["label"] == 1).to_numpy()].mean()),
                     "negatives": float(seen[ok & (view["label"] == 0).to_numpy()].mean())})
    pd.DataFrame(rows).to_csv(args.outdir / f"study_leakage_{args.track}.csv", index=False)
    update_manifest(f"12_metrics_{args.track}", {"n_boot": args.n_boot, "seed": args.seed, "tiers": args.tiers})
    log.info("done in %.0fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
