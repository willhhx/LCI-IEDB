#!/usr/bin/env python3
"""Write ``data/documentation/data_dictionary.csv`` and verify that every column of every released table is documented.

Exit status 1 if a released column has no entry (or an entry has no column), so the dictionary cannot silently go stale.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, setup_logging  # noqa: E402

D = "dataset_all_tiers.csv.gz"
SPLIT_TXT = {"S1": "antigen-group", "S2": "held-out HLA allele", "S3": "held-out source taxon group", "S4": "random"}

DATASET_COLUMNS: dict[str, tuple[str, str]] = {
    "row_id": ("string", "Stable row identifier (LCI000001...). Decoy rows N1/N3 carry their own ids."),
    "peptide": ("string", "Linear unmodified peptide, 20 canonical amino acids, length 8-11."),
    "allele": ("string", "Normalised HLA class I allele (HLA-A*02:01 style, two-field resolution) the label refers to."),
    "length": ("int", "Peptide length (8-11)."),
    "label": ("int", "1 = positive T-cell response; 0 = negative / decoy. The label unit is the (peptide, allele) pair."),
    "tier": ("category", "P positive; N2 assay-verified negative (primary); N1 same-protein decoy; N3 matched hard decoy. N1/N3 peptides have no T-cell record."),
    "category": ("category", "Source category: virus, human_self, parasite, bacterium, fungus, other_eukaryote, other, unspecified."),
    "taxon_group": ("string", "Leakage-control taxon group (virus family, bacterial/parasite genus, 'Homo sapiens (self)', or 'unspecified')."),
    "species": ("string", "Source species from IEDB (blank if not stated)."),
    "taxon_id": ("float", "NCBI taxon id of the source species (blank if not stated)."),
    "parent_iri": ("string", "UniProt IRI of the parent protein (blank if IEDB lists none)."),
    "parent_name": ("string", "Parent protein name from IEDB."),
    "antigen_name_key": ("string", "Normalised 'taxon group|protein name' key used to link antigen families."),
    "start": ("int", "Start position of the peptide in the parent protein (IEDB; 1-based)."),
    "end": ("int", "End position of the peptide in the parent protein."),
    "in_pathogen_track": ("bool", "True if the row belongs to the pathogen/vaccine evaluation track (non-human source taxon)."),
    "evidence_tier": ("category", "How the HLA restriction was established (best evidence over records) for P and N2 rows: A single allele present, MHC binding assay or T-cell-assay-based; B cited reference or statistical inference; C MHC-binding prediction only."),
    "n2_confidence": ("category", "N2 rows only: high (evidence tier A or B) or medium (tier C, restriction predicted only)."),
    "negative_replicated": ("bool", "N2 rows: negative reported in at least two records."),
    "mixed_within_study": ("bool", "P rows: at least one study reports both positive and negative outcomes for the pair (positive kept, flagged)."),
    "max_positive_strength": ("category", "P rows: strongest IEDB response-strength annotation (High, Intermediate, Low, Unspecified)."),
    "n_records": ("int", "Number of IEDB T-cell assay records supporting the label."),
    "n_references": ("int", "Number of distinct IEDB references supporting the label."),
    "reference_ids": ("string", "';'-joined IEDB reference ids (provenance)."),
    "pmids": ("string", "';'-joined PubMed ids of the supporting references. (The IEDB export carries no DOI.)"),
    "methods": ("string", "';'-joined assay methods (ELISPOT, ICS, ELISA, 51 chromium, ...)."),
    "responses": ("string", "';'-joined measured response types (IFNg release, activation, ...)."),
    "effector_cells": ("string", "';'-joined effector cell annotations."),
    "culture_conditions": ("string", "';'-joined culture conditions (Direct Ex Vivo, Short Term Restimulated, ...)."),
    "first_year": ("int", "Earliest publication year among supporting references."),
    "exposure_context": ("string", "';'-joined donor exposure context (cancer, infectious disease, no immunisation, ...)."),
    "has_cd8_annotated": ("bool", "True if a CD8+ effector annotation exists in a supporting record."),
    "max_subjects_tested": ("int", "Largest number of subjects tested in a supporting record (blank if not reported)."),
    "mhc_evidence_class": ("category", "Independent MHC-binding evidence from IEDB ligand assays for the pair: strong_binder, binder, nonbinder, none, or predicted_binder (N3 decoys selected by the proxy)."),
    "mhc_log10_nM": ("float", "log10 of the strongest measured IC50/Kd (nM) for the pair in IEDB MHC binding assays (blank if none)."),
    "mhc_eluted": ("bool", "True if the pair was observed in an eluted-ligand (mass spectrometry) record."),
    "matched_pair_id": ("string", "Decoy rows (N1/N3): row_id of the matched positive. Blank for P and N2 rows."),
    "match_level": ("category", "Decoy rows: matching scope reached, 'protein' (same parent protein) or 'species' (same source species)."),
    "n3_evidence": ("category", "N3 rows: ligand_evidence (observed MHC ligand/binder) or predicted (binding proxy)."),
    "binding_log10nM": ("float", "Decoy rows: measured or proxy-predicted log10 nM binding affinity used for N3 matching."),
    "binding_source": ("category", "measured (IEDB binding assay) or proxy (self-trained ridge regression on IEDB affinities, not NetMHCpan)."),
    "delta_binding": ("float", "N3 rows: |log10 nM difference| to the matched positive (tolerance 0.7)."),
    "delta_start": ("float", "Decoy rows: absolute difference in start position between decoy and matched positive within the parent protein (antigen-region match; blank if unknown)."),
    "peptide_label_ambiguous": ("bool", "True if the same peptide string is positive for one allele and negative for another (kept; flagged)."),
}
GROUP_COLUMNS = {
    "row_id": ("string", "Row identifier (joins to dataset_all_tiers)."), "peptide": ("string", "Peptide sequence."),
    "group_id": ("int", "Leakage group id: connected component of near-duplicate peptides (similarity >= 0.8), same parent protein and same antigen family."),
    "peptide_component": ("int", "Connected component of near-duplicate peptides only (used to purge S2)."),
}
SPLIT_COLUMNS: dict[str, tuple[str, str]] = {
    "row_id": ("string", "Row identifier."), "peptide": ("string", "Peptide."), "allele": ("string", "HLA allele."),
    "tier": ("category", "P, N2, N1 or N3."), "label": ("int", "Label."), "category": ("category", "Source category."),
    "taxon_group": ("string", "Taxon group."), "in_pathogen_track": ("bool", "Pathogen/vaccine track membership."),
    "group_id": ("int", "Leakage group id."),
}
for _s, _t in SPLIT_TXT.items():
    SPLIT_COLUMNS[f"{_s}_split"] = ("category", f"{_s} ({_t}) official partition: train | val | test | purged (removed to avoid leakage) | excluded (row cannot take part).")
    SPLIT_COLUMNS[f"{_s}_cv"] = ("int", f"{_s} five-fold grouped cross-validation fold 0-4; -1 = not used in this split.")
    SPLIT_COLUMNS[f"{_s}_maxsim"] = ("float", f"{_s} test rows: maximum peptide similarity to any official training peptide.")
    SPLIT_COLUMNS[f"{_s}_bin"] = ("category", f"{_s} test rows: similarity bin of {_s}_maxsim (<40%, 40-59%, 60-79%, 80-99%, 100%).")
    SPLIT_COLUMNS[f"{_s}_cv_maxsim"] = ("float", f"{_s}: maximum similarity to the training portion of the row's own CV fold.")
    SPLIT_COLUMNS[f"{_s}_cv_bin"] = ("category", f"{_s}: similarity bin of {_s}_cv_maxsim.")

CONFLICT_COLUMNS = {
    "pair_id": ("string", "Peptide|allele key."), "status": ("category", "Why the pair is unresolved (cross-study positive/negative conflict)."),
    "n_ref_pos": ("int", "References with a positive outcome."), "n_ref_neg": ("int", "References with a negative outcome."),
    "n_ref_negonly": ("int", "References reporting only negatives."), "n_ref_mixed": ("int", "References reporting mixed outcomes."),
    "multimer_positive_exists": ("bool", "A multimer/tetramer-only positive exists (not counted as functional)."),
    "uniprot": ("string", "UniProt accession of the parent protein."), "source_molecule": ("string", "Source molecule name from IEDB."),
    "source_organism": ("string", "Source organism name from IEDB."), "parents_all": ("string", "All parent proteins listed for the peptide."),
    "taxa_all": ("string", "All source taxa listed for the peptide."), "n_ligand_records": ("int", "IEDB ligand records supporting binding evidence."),
}
CONFLICT_SHARED = {c: DATASET_COLUMNS[c] for c in ["peptide", "allele", "mixed_within_study", "n_records", "n_references", "reference_ids", "pmids", "methods",
                                                    "responses", "effector_cells", "culture_conditions", "first_year", "evidence_tier", "max_subjects_tested",
                                                    "has_cd8_annotated", "max_positive_strength", "exposure_context", "parent_name", "parent_iri", "start", "end",
                                                    "species", "taxon_id", "length", "n2_confidence", "negative_replicated", "mhc_log10_nM", "mhc_eluted",
                                                    "mhc_evidence_class"]}
CONFLICT_COLUMNS.update(CONFLICT_SHARED)
MULTIMER_COLUMNS = {"peptide": ("string", "Peptide."), "allele": ("string", "HLA allele."), "n_records": ("int", "Multimer/tetramer-only records."),
                    "reference_ids": ("string", "IEDB reference ids."), "pmids": ("string", "PubMed ids.")}

TABLES = [("processed/dataset_all_tiers.csv.gz", "dataset_all_tiers", DATASET_COLUMNS),
          ("processed/leakage_groups.csv.gz", "leakage_groups", GROUP_COLUMNS),
          ("benchmark_splits/splits_all_tiers.csv.gz", "splits_all_tiers", SPLIT_COLUMNS),
          ("processed/pair_conflicts.csv.gz", "pair_conflicts", CONFLICT_COLUMNS),
          ("processed/pair_multimer_only_positive.csv.gz", "pair_multimer_only_positive", MULTIMER_COLUMNS)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=data_dir("documentation/data_dictionary.csv"))
    args = p.parse_args(argv)
    log = setup_logging("build_data_dictionary")
    rows, bad = [], 0
    for rel, name, doc in TABLES:
        try:
            cols = list(pd.read_csv(data_dir() / rel, nrows=1).columns)
        except OSError as exc:
            log.error("%s", exc)
            return 2
        missing, extra = [c for c in cols if c not in doc], [c for c in doc if c not in cols]
        if missing or extra:
            log.error("%s: undocumented %s, documented-but-absent %s", name, missing, extra)
            bad += 1
        rows += [{"table": name, "column": c, "type": doc[c][0], "description": doc[c][1]} for c in cols if c in doc]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    log.info("%d columns documented -> %s", len(rows), args.out)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
