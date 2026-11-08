#!/usr/bin/env python3
"""Collapse MHC-ligand records to (peptide, allele) evidence and annotate the T-cell pair labels.

Per (peptide, allele):
  * measured affinity in nM from IC50 / KD / EC50 assays ('=' records -> geometric mean; '>' records -> censored lower bound)
  * qualitative binding calls, and mass-spectrometry "ligand presentation" (elution) positives
Derived class ``mhc_evidence_class``:
  strong_binder (<=50 nM) | binder (<=500 nM, qualitative-positive or eluted) | nonbinder (>500 nM or qualitative-negative) | none

Outputs (``data/processed``):
  ligand_pair_evidence.csv.gz     one row per (peptide, allele) incl. modal parent / position / species of the peptide
  pair_labels_annotated.csv.gz    pair_labels_all + mhc_log10_nM, mhc_eluted, mhc_evidence_class
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, setup_logging, update_manifest  # noqa: E402
from utilities.tables import modal  # noqa: E402

AFFINITY_RESPONSES = {
    "dissociation constant KD (~IC50)", "dissociation constant KD (~EC50)", "dissociation constant KD",
    "half maximal inhibitory concentration (IC50)", "half maximal effective concentration (EC50)",
}
PA = ["peptide", "allele"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ligand", type=Path, default=data_dir("processed/mhc_ligand_records_human_hlaI.csv.gz"))
    p.add_argument("--pairs", type=Path, default=data_dir("processed/pair_labels_all.csv.gz"))
    p.add_argument("--outdir", type=Path, default=data_dir("processed"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def classify(log10nm: pd.Series, qual_pos: pd.Series, qual_neg: pd.Series, eluted: pd.Series) -> pd.Series:
    cls = pd.Series("none", index=log10nm.index, dtype=object)
    cls[qual_neg & ~qual_pos] = "nonbinder"
    cls[qual_pos | eluted] = "binder"
    has_nm = log10nm.notna()
    cls[has_nm & (log10nm <= np.log10(500))] = "binder"
    cls[has_nm & (log10nm <= np.log10(50))] = "strong_binder"
    cls[has_nm & (log10nm > np.log10(500))] = "nonbinder"
    return cls


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("05_ligand_evidence", args.verbose)
    for pth in (args.ligand, args.pairs):
        if not pth.exists():
            log.error("input not found: %s", pth)
            return 2
    try:
        lig = pd.read_csv(args.ligand, dtype=str, keep_default_na=False)
        pairs = pd.read_csv(args.pairs, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 1
    log.info("ligand records: %d", len(lig))
    lig["is_aff"] = lig["response"].isin(AFFINITY_RESPONSES) & lig["units"].eq("nM")
    val = pd.to_numeric(lig["quantitative"], errors="coerce")
    lig["nm"] = val.where(lig["is_aff"] & (val > 0))
    lig["censored"] = lig["inequality"].isin([">", ">="])
    lig["nm_exact"] = lig["nm"].where(~lig["censored"])
    lig["nm_cens"] = lig["nm"].where(lig["censored"])
    lig["log_exact"] = np.log10(lig["nm_exact"])
    lig["is_elution"] = lig["response"].eq("ligand presentation") & lig["qualitative"].str.startswith("Positive")
    qb = lig["response"].eq("qualitative binding")
    lig["qb_pos"] = qb & lig["qualitative"].str.startswith("Positive")
    lig["qb_neg"] = qb & lig["qualitative"].eq("Negative")

    g = lig.groupby(PA, sort=False)
    ev = g.agg(n_records=("assay_id", "size"), n_elution=("is_elution", "sum"), n_qb_pos=("qb_pos", "sum"),
               n_qb_neg=("qb_neg", "sum"), mean_log_exact=("log_exact", "mean"), n_exact=("log_exact", "count"),
               max_censored_nm=("nm_cens", "max")).reset_index()
    ev["mhc_eluted"] = ev["n_elution"] > 0
    ev["mhc_log10_nM"] = ev["mean_log_exact"]
    cens_only = ev["mhc_log10_nM"].isna() & ev["max_censored_nm"].notna()
    ev.loc[cens_only, "mhc_log10_nM"] = np.log10(ev.loc[cens_only, "max_censored_nm"])  # lower bound
    ev["mhc_evidence_class"] = classify(ev["mhc_log10_nM"], ev["n_qb_pos"] > 0, ev["n_qb_neg"] > 0, ev["mhc_eluted"])
    # peptide-level modal metadata (allele independent)
    meta = pd.DataFrame(index=pd.Index(sorted(lig["peptide"].unique()), name="peptide"))
    for col in ("parent_name", "parent_iri", "uniprot", "start", "end", "species", "taxon_id"):
        meta[col] = modal(lig, ["peptide"], col)
    meta = meta.fillna("").reset_index()
    ev = ev.merge(meta, on="peptide", how="left").drop(columns=["mean_log_exact", "max_censored_nm", "n_exact"])
    args.outdir.mkdir(parents=True, exist_ok=True)
    ev.to_csv(args.outdir / "ligand_pair_evidence.csv.gz", index=False, compression="gzip")

    ann = pairs.merge(ev[PA + ["mhc_log10_nM", "mhc_eluted", "mhc_evidence_class", "n_records"]].rename(
        columns={"n_records": "n_ligand_records"}), on=PA, how="left")
    ann["mhc_evidence_class"] = ann["mhc_evidence_class"].fillna("none")
    ann["mhc_eluted"] = ann["mhc_eluted"].fillna(False)
    ann.to_csv(args.outdir / "pair_labels_annotated.csv.gz", index=False, compression="gzip")
    tab = pd.crosstab(ann["status"], ann["mhc_evidence_class"])
    log.info("ligand pairs: %d\nT-cell pairs by status x independent MHC evidence:\n%s", len(ev), tab.to_string())
    update_manifest("05_ligand_evidence", {"ligand_pairs": len(ev), "tcell_pairs_with_mhc_evidence": int((ann["mhc_evidence_class"] != "none").sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
