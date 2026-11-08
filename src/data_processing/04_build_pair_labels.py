#!/usr/bin/env python3
"""Aggregate record-level T-cell assays into one label per (peptide, HLA allele) pair.

Record eligibility (all four must hold; counted in ``data/metadata/funnel_labels.json``):
  * functional T-cell assay (ELISPOT, ICS, ELISA, cytokine bead array, 51Cr, activation/proliferation ...);
    MHC-multimer staining is NOT a functional assay and is used only to detect contradictions;
  * effector cell is a T-cell-containing population (PBMC / CD8+ / T cell / TIL / lymphocyte / mononuclear);
  * culture condition is Direct Ex Vivo or Short Term Restimulated (no clones, engineered cells, hybridomas);
  * the HLA restriction was actually established (IEDB evidence code != "Not determined").

Pair label (references = IEDB reference ids):
  P  = references with >=1 positive record,  N-only = references whose records for the pair are ALL negative
  positive  : P non-empty and N-only empty        (``mixed_within_study`` flags studies with both outcomes)
  negative  : P empty                              (candidate N2 negative)
  conflict  : P non-empty and N-only non-empty     -> excluded from labels, written to the conflicts table
A negative pair that also has a multimer-positive record from any reference is moved to conflicts too.

Allele-evidence tiers (how the HLA restriction was established):
  A = Single allele present / MHC binding assay / T-cell-assay based;  B = cited reference / statistical inference;
  C = MHC binding prediction.

Outputs in ``data/processed``: pair_labels_all.csv.gz, peptide_meta.csv.gz, pair_multimer_only_positive.csv.gz;
``data/metadata``: references_used.csv, funnel_labels.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, setup_logging, update_manifest, write_json  # noqa: E402
from utilities.tables import join_unique, modal  # noqa: E402

ALLOWED_EFFECTOR = {"PBMC", "T cell CD8+", "T cell", "T cell CD3+", "tumor infiltrating lymphocyte", "lymphocyte",
                    "mononuclear cell"}
ALLOWED_CULTURE = {"Direct Ex Vivo", "Short Term Restimulated"}
EVIDENCE_TIER = {
    "Single allele present": "A", "MHC binding assay": "A", "T cell assay -Mismatched MHC molecules": "A",
    "T cell assay -Biological process measured": "A", "T cell assay -MHC subset identification": "A",
    "T cell assay -T cell subset identification": "A",
    "Cited reference": "B", "Statistically inferred by motif or alleles present": "B",
    "MHC binding prediction": "C",
}
STRENGTH_RANK = {"High": 4, "Intermediate": 3, "Unspecified": 2, "Low": 1, "NA": 0}
PEP_ALLELE = ["peptide", "allele"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", type=Path, default=data_dir("processed/tcell_records_human_hlaI.csv.gz"))
    p.add_argument("--reference-csv", type=Path, default=data_dir("raw/reference/reference_full_v3.csv"))
    p.add_argument("--outdir", type=Path, default=data_dir("processed"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("04_build_pair_labels", args.verbose)
    for pth in (args.records, args.reference_csv):
        if not pth.exists():
            log.error("input not found: %s", pth)
            return 2
    try:
        rec = pd.read_csv(args.records, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        log.error("cannot read %s: %s", args.records, exc)
        return 1
    for c in ("n_tested", "n_positive_subjects"):
        rec[c] = pd.to_numeric(rec[c], errors="coerce")
    rec["label_record"] = rec["label_record"].astype(int)
    rec["tier"] = rec["mhc_evidence"].map(EVIDENCE_TIER).fillna("X")

    funnel: dict[str, int] = {"records_resolved_allele": len(rec)}
    el = rec["method_family"].eq("functional"); funnel["records_functional_assay"] = int(el.sum())
    el &= rec["effector_cell"].isin(ALLOWED_EFFECTOR); funnel["records_allowed_effector_cell"] = int(el.sum())
    el &= rec["culture_condition"].isin(ALLOWED_CULTURE); funnel["records_allowed_culture"] = int(el.sum())
    el &= rec["tier"].ne("X"); funnel["records_allele_established"] = int(el.sum())
    elig = rec[el].copy()
    mm_pos = rec[rec["method_family"].eq("multimer") & rec["label_record"].eq(1) & rec["tier"].ne("X")]

    # ---- peptide-level metadata (from ALL resolved records so that parents are consistent across alleles) ----
    pk = ["peptide"]
    pm = pd.DataFrame(index=pd.Index(sorted(rec["peptide"].unique()), name="peptide"))
    for col in ("parent_name", "parent_iri", "uniprot", "source_molecule", "start", "end", "source_organism", "species",
                "taxon_id"):
        pm[col] = modal(rec, pk, col)
    pm = pm.fillna("")
    pm["parents_all"] = rec.groupby("peptide")["parent_iri"].agg(join_unique)
    pm["taxa_all"] = rec.groupby("peptide")["taxon_id"].agg(join_unique)
    pm["length"] = pm.index.str.len()

    # ---- per pair x reference outcomes ----
    g = elig.groupby(PEP_ALLELE + ["reference_id"], sort=False)["label_record"].agg(npos="sum", n="count").reset_index()
    g["has_pos"] = g["npos"] > 0
    g["has_neg"] = g["npos"] < g["n"]
    g["neg_only"] = g["has_neg"] & ~g["has_pos"]
    g["mixed"] = g["has_pos"] & g["has_neg"]
    pair = g.groupby(PEP_ALLELE, sort=False).agg(n_ref_pos=("has_pos", "sum"), n_ref_neg=("has_neg", "sum"),
                                                n_ref_negonly=("neg_only", "sum"), n_ref_mixed=("mixed", "sum")).reset_index()
    pair["status"] = np.where(pair["n_ref_pos"].eq(0), "negative",
                              np.where(pair["n_ref_negonly"].eq(0), "positive", "conflict"))
    pair["mixed_within_study"] = pair["n_ref_mixed"].gt(0) & pair["status"].eq("positive")

    # multimer-positive contradicts a functionally-negative pair
    mm_keys = set(zip(mm_pos["peptide"], mm_pos["allele"]))
    is_mm = pd.Series([k in mm_keys for k in zip(pair["peptide"], pair["allele"])], index=pair.index)
    pair["multimer_positive_exists"] = is_mm
    pair.loc[pair["status"].eq("negative") & is_mm, "status"] = "conflict"

    # ---- evidence summary using the records that support the final label ----
    lab = pair[PEP_ALLELE + ["status"]]
    e = elig.merge(lab, on=PEP_ALLELE, how="inner")
    sup = e[(e["status"].eq("positive") & e["label_record"].eq(1)) | (e["status"].eq("negative") & e["label_record"].eq(0))
            | e["status"].eq("conflict")]
    sup = sup.assign(strength_rank=sup["strength"].map(STRENGTH_RANK))
    agg = sup.groupby(PEP_ALLELE, sort=False).agg(
        n_records=("assay_id", "size"), n_references=("reference_id", "nunique"),
        reference_ids=("reference_id", join_unique), pmids=("pmid", join_unique), methods=("method", join_unique),
        responses=("response", join_unique), effector_cells=("effector_cell", join_unique),
        culture_conditions=("culture_condition", join_unique), first_year=("ref_year", lambda s: min([x for x in s if x] or [""])),
        evidence_tier=("tier", "min"), max_strength_rank=("strength_rank", "max"),
        max_subjects_tested=("n_tested", "max"), has_cd8_annotated=("effector_cell", lambda s: bool((s == "T cell CD8+").any())),
    ).reset_index()
    inv = {v: k for k, v in STRENGTH_RANK.items()}
    agg["max_positive_strength"] = agg["max_strength_rank"].map(inv)
    agg = agg.drop(columns="max_strength_rank")
    ctx = modal(sup, PEP_ALLELE, "vivo_process").rename("exposure_context").reset_index()
    out = pair.merge(agg, on=PEP_ALLELE, how="left").merge(ctx, on=PEP_ALLELE, how="left")
    out["exposure_context"] = out["exposure_context"].fillna("")
    out.loc[out["status"].ne("positive"), "max_positive_strength"] = ""
    out = out.merge(pm.reset_index(), on="peptide", how="left")
    out["length"] = out["peptide"].str.len()
    out = out.sort_values(PEP_ALLELE).reset_index(drop=True)
    out.insert(0, "pair_id", [f"LCI{i:06d}" for i in range(1, len(out) + 1)])

    # negatives: confidence tier
    neg = out["status"].eq("negative")
    out["n2_confidence"] = ""
    out.loc[neg & out["evidence_tier"].isin(["A", "B"]), "n2_confidence"] = "high"
    out.loc[neg & out["evidence_tier"].eq("C"), "n2_confidence"] = "medium"
    out["negative_replicated"] = neg & out["n_references"].ge(2)

    # multimer-only positives (supplementary)
    func_keys = set(zip(out["peptide"], out["allele"]))
    mmo = mm_pos[[k not in func_keys for k in zip(mm_pos["peptide"], mm_pos["allele"])]]
    mmo = mmo.groupby(PEP_ALLELE).agg(n_records=("assay_id", "size"), reference_ids=("reference_id", join_unique),
                                      pmids=("pmid", join_unique)).reset_index()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.outdir / "pair_labels_all.csv.gz", index=False, compression="gzip")
    pm.reset_index().to_csv(args.outdir / "peptide_meta.csv.gz", index=False, compression="gzip")
    mmo.to_csv(args.outdir / "pair_multimer_only_positive.csv.gz", index=False, compression="gzip")

    # ---- reference metadata for every reference that supports a label ----
    ref = pd.read_csv(args.reference_csv, header=None, skiprows=2, dtype=str, keep_default_na=False,
                      names=["iri", "type", "pmid", "submission_id", "alt_iris", "authors", "journal", "date", "title"])
    ref["reference_id"] = ref["iri"].str.rsplit("/", n=1).str[-1]
    used = set(";".join(out.loc[out["status"].isin(["positive", "negative", "conflict"]), "reference_ids"]).split(";")) - {""}
    ref = ref[ref["reference_id"].isin(used)].copy()
    ref["first_author"] = ref["authors"].str.split(";").str[0].str.strip()
    ref[["reference_id", "type", "pmid", "submission_id", "first_author", "journal", "date", "title"]].to_csv(
        data_dir("metadata/references_used.csv"), index=False)

    counts = out["status"].value_counts().to_dict()
    funnel.update({"pairs_total": len(out), **{f"pairs_{k}": int(v) for k, v in counts.items()},
                   "pairs_negative_high_confidence": int((out["n2_confidence"] == "high").sum()),
                   "pairs_negative_medium_confidence": int((out["n2_confidence"] == "medium").sum()),
                   "pairs_positive_mixed_within_study": int(out["mixed_within_study"].sum()),
                   "pairs_multimer_only_positive": len(mmo), "references_used": len(ref)})
    write_json(data_dir("metadata/funnel_labels.json"), funnel)
    update_manifest("04_build_pair_labels", funnel)
    for k, v in funnel.items():
        log.info("%-42s %9d", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
