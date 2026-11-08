#!/usr/bin/env python3
"""Assemble the unified row-level dataset from positives, N2 negatives and the N1 / N3 decoy tiers.

Rows (``tier``):  P = positive (label 1) | N2 = assay-verified negative | N1 = same-source decoy | N3 = matched hard decoy.
N1 and N3 are ALTERNATIVE negative sets for the same positives (unverified: no T-cell assay record exists for them).
Conflicting pairs are written to a separate table and are not part of the benchmark.

Outputs: ``data/processed/dataset_all_tiers.csv.gz``, ``data/processed/pair_conflicts.csv.gz``,
``data/metadata/taxon_groups.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, normalise_name, setup_logging, update_manifest  # noqa: E402
from utilities.taxonomy import PATHOGEN_CATEGORIES, classify_species  # noqa: E402

GENERIC_NAMES = {"", "protein", "uncharacterized protein", "hypothetical protein", "putative uncharacterized protein",
                 "unknown", "unnamed protein product", "predicted protein", "polypeptide"}
COLUMNS = [
    "row_id", "peptide", "allele", "length", "label", "tier", "category", "taxon_group", "species", "taxon_id",
    "parent_iri", "parent_name", "antigen_name_key", "start", "end", "in_pathogen_track",
    "evidence_tier", "n2_confidence", "negative_replicated", "mixed_within_study", "max_positive_strength",
    "n_records", "n_references", "reference_ids", "pmids", "methods", "responses", "effector_cells", "culture_conditions",
    "first_year", "exposure_context", "has_cd8_annotated", "max_subjects_tested",
    "mhc_evidence_class", "mhc_log10_nM", "mhc_eluted",
    "matched_pair_id", "match_level", "n3_evidence", "binding_log10nM", "binding_source", "delta_binding", "delta_start",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, default=data_dir("processed/pair_labels_annotated.csv.gz"))
    p.add_argument("--n1", type=Path, default=data_dir("processed/negatives_N1.csv.gz"))
    p.add_argument("--n3", type=Path, default=data_dir("processed/negatives_N3.csv.gz"))
    p.add_argument("--outdir", type=Path, default=data_dir("processed"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("08_assemble_dataset", args.verbose)
    for pth in (args.pairs, args.n1, args.n3):
        if not pth.exists():
            log.error("input not found: %s", pth)
            return 2
    try:
        pairs = pd.read_csv(args.pairs, dtype=str, keep_default_na=False)
        n1 = pd.read_csv(args.n1, dtype=str, keep_default_na=False)
        n3 = pd.read_csv(args.n3, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 1

    conflicts = pairs[pairs["status"] == "conflict"]
    conflicts.to_csv(args.outdir / "pair_conflicts.csv.gz", index=False, compression="gzip")
    core = pairs[pairs["status"].isin(["positive", "negative"])].copy()
    core["label"] = (core["status"] == "positive").astype(int)
    core["tier"] = core["status"].map({"positive": "P", "negative": "N2"})
    core["row_id"] = core["pair_id"]
    core["binding_log10nM"] = core["mhc_log10_nM"]
    core["binding_source"] = ["measured" if x != "" else "" for x in core["mhc_log10_nM"]]
    dec = []
    for tier, df in (("N1", n1), ("N3", n3)):
        d = df.copy()
        d["label"] = 0
        d["tier"] = tier
        d["row_id"] = [f"{tier}-{i:06d}" for i in range(1, len(d) + 1)]
        d["length"] = d["peptide"].str.len()
        dec.append(d)
    allr = pd.concat([core] + dec, ignore_index=True, sort=False).fillna("")
    cg = allr["species"].map(classify_species)
    allr["category"] = cg.str[0]
    allr["taxon_group"] = cg.str[1]
    allr["in_pathogen_track"] = allr["category"].isin(PATHOGEN_CATEGORIES)
    nm = allr["parent_name"].map(normalise_name)
    allr["antigen_name_key"] = [f"{g}|{n}" if n not in GENERIC_NAMES and g != "unspecified" else "" for g, n in zip(allr["taxon_group"], nm)]
    allr["length"] = allr["peptide"].str.len()
    for c in COLUMNS:
        if c not in allr.columns:
            allr[c] = ""
    out = allr[COLUMNS]
    # flag peptides that carry both labels (different alleles)
    lab_sets = out.groupby("peptide")["label"].nunique()
    out = out.assign(peptide_label_ambiguous=out["peptide"].map(lab_sets).gt(1))
    out.to_csv(args.outdir / "dataset_all_tiers.csv.gz", index=False, compression="gzip")
    tg = (allr[["species", "taxon_id", "category", "taxon_group"]].drop_duplicates("species").sort_values("species"))
    tg.to_csv(data_dir("metadata/taxon_groups.csv"), index=False)

    log.info("rows by tier:\n%s", out["tier"].value_counts().to_string())
    log.info("category x tier:\n%s", pd.crosstab(out["category"], out["tier"]).to_string())
    log.info("conflicts: %d  ambiguous-label peptides: %d", len(conflicts), int(out.loc[out["peptide_label_ambiguous"], "peptide"].nunique()))
    update_manifest("08_assemble_dataset", {"rows": len(out), "tiers": out["tier"].value_counts().to_dict(), "conflicts": len(conflicts)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
