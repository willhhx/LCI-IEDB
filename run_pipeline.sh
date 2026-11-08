#!/usr/bin/env bash
# Re-run the whole LCI-IEDB pipeline from the raw IEDB exports (data/raw/...). Every step is also runnable on its own.
# Python 3.13 or later, see requirements.txt.  Runtime on 2 CPU / 7 GB RAM: about 50 min (CV baselines + bootstrap metrics dominate).
set -euo pipefail
cd "$(dirname "$0")"
PY="python3 -I"
$PY src/data_processing/01_build_hla_allele_map.py
$PY src/data_processing/02_extract_tcell_records.py
$PY src/data_processing/03_extract_mhc_ligand.py
$PY src/data_processing/03b_extract_epitope_pool.py
$PY src/data_processing/04_build_pair_labels.py
$PY src/data_processing/05_ligand_evidence.py
$PY src/negative_curation/06_train_binding_proxy.py
$PY src/negative_curation/07_build_N1_N3_negatives.py
$PY src/data_processing/08_assemble_dataset.py
$PY src/clustering/09_build_groups.py --external-clusters data/processed/mmseqs_map.tsv   # MMseqs2 clusters of the parent proteins (see README)
$PY src/benchmark_generation/10_make_splits.py
for track in all pathogen; do
  $PY src/modeling/11_cv_baselines.py --track "$track"
  $PY src/evaluation/12_metrics.py --track "$track"
  $PY src/evaluation/13_leakage_exposure.py --track "$track"
done
$PY src/evaluation/14_study_heldout.py
$PY src/utilities/build_data_dictionary.py
$PY src/visualization/20_make_tables.py
$PY src/visualization/21_make_figures.py
$PY src/visualization/22_make_novelty_table.py
$PY src/utilities/make_release.py
$PY tests/test_peptide_sim.py
$PY tests/test_taxonomy.py
$PY tests/audit_splits.py
