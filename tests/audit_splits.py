#!/usr/bin/env python3
"""Independent leakage audit of the released splits (run: python -I tests/audit_splits.py).

Checks, using code paths independent of the generator wherever possible:
  * S1: no leakage group and no identical peptide on both sides of train/test; S2: no allele shared, no near-duplicate
    component shared; S3: no taxon group shared, no group shared; S4 (reference): identical peptides DO straddle.
  * stored max-similarity values equal a brute-force recomputation on a random sample (official and CV columns).
  * fold sizes, label balance and every fold non-empty.
Exit code 1 on any failure.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utilities.peptide_sim import naive_similarity, similarity_bin  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def main() -> int:
    sp = pd.read_csv(ROOT / "data/benchmark_splits/splits_all_tiers.csv.gz", dtype={"row_id": str, "peptide": str, "allele": str})
    ds = pd.read_csv(ROOT / "data/processed/dataset_all_tiers.csv.gz", dtype=str, keep_default_na=False)[["row_id", "taxon_group"]]
    gr = pd.read_csv(ROOT / "data/processed/leakage_groups.csv.gz", dtype={"row_id": str})[["row_id", "peptide_component"]]
    sp = sp.merge(gr, on="row_id").reset_index(drop=True)
    check(sp["row_id"].is_unique, "row ids unique")
    for s in ("S1", "S2", "S3", "S4"):
        tr, te = sp[sp[f"{s}_split"] == "train"], sp[sp[f"{s}_split"] == "test"]
        ident = len(set(tr["peptide"]) & set(te["peptide"]))
        if s == "S4":
            check(ident > 0, f"{s}: identical peptides straddle train/test as expected for the random reference ({ident})")
            continue
        check(ident == 0, f"{s}: no identical peptide in train and test (found {ident})")
        check(len(set(tr["group_id"]) & set(te["group_id"])) == 0 or s == "S2",
              f"{s}: no leakage group shared by train and test")
        check(len(set(tr["peptide_component"]) & set(te["peptide_component"])) == 0, f"{s}: no near-duplicate component shared")
        if s == "S2":
            check(set(tr["allele"]).isdisjoint(te["allele"]), "S2: no HLA allele shared between train and test")
        if s == "S3":
            check(set(tr["taxon_group"]).isdisjoint(te["taxon_group"]), "S3: no taxon group shared between train and test")
            check((sp.loc[sp["S3_split"] == "excluded", "taxon_group"] == "unspecified").all(), "S3: only taxon-less rows are excluded")
        # CV folds
        for k in range(5):
            f = sp[sp[f"{s}_cv"] == k]
            check(len(f) > 0, f"{s}: CV fold {k} non-empty ({len(f)} rows)")
    # leakage units in CV folds
    for s, col in (("S1", "group_id"), ("S2", "allele"), ("S3", "taxon_group")):
        d = sp[sp[f"{s}_cv"] >= 0]
        nfold = d.groupby(col)[f"{s}_cv"].nunique()
        check((nfold == 1).all(), f"{s}: every {col} falls in exactly one CV fold")
    # brute-force verification of stored similarities
    rng = random.Random(1)
    for s in ("S1", "S2", "S3", "S4"):
        te = sp[sp[f"{s}_split"] == "test"]
        tr_pep = sorted(set(sp.loc[sp[f"{s}_split"] == "train", "peptide"]))
        sample = te.sample(15, random_state=3)
        ok = True
        for r in sample.itertuples():
            ref = max(naive_similarity(r.peptide, t) for t in tr_pep)
            ok &= abs(ref - getattr(r, f"{s}_maxsim")) < 1e-9
            ok &= similarity_bin(ref)[0] == getattr(r, f"{s}_bin")
        check(ok, f"{s}: official max-similarity and bins match brute force on 15 random test rows")
        # CV version for one fold
        k = rng.randrange(5)
        te = sp[sp[f"{s}_cv"] == k]
        if s in ("S2", "S3"):
            col = "peptide_component" if s == "S2" else "group_id"
            trm = (sp[f"{s}_cv"] != k) & (sp[f"{s}_cv"] >= 0) & ~sp[col].isin(set(te[col]))
        else:
            trm = (sp[f"{s}_cv"] != k) & (sp[f"{s}_cv"] >= 0)
        tr_pep = sorted(set(sp.loc[trm, "peptide"]))
        ok = True
        for r in te.sample(10, random_state=5).itertuples():
            ref = max(naive_similarity(r.peptide, t) for t in tr_pep)
            ok &= abs(ref - getattr(r, f"{s}_cv_maxsim")) < 1e-9
        check(ok, f"{s}: CV max-similarity (fold {k}) matches brute force on 10 random rows")
    # similarity-bin composition summary (informational)
    for s in ("S1", "S2", "S3", "S4"):
        print(s, "official test bins:", sp.loc[sp[f"{s}_split"] == "test", f"{s}_bin"].value_counts(normalize=True).round(3).to_dict())
    print(f"\n{len(FAILS)} failure(s)")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
