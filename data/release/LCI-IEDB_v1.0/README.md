# LCI-IEDB v1.0: a leakage-controlled open dataset for human CD8⁺ T-cell immunogenicity

LCI-IEDB packages experimentally measured T-cell responses to human MHC-class-I peptides (8–11 aa) from the Immune Epitope Database (IEDB, export last updated **1 Nov 2026**) as a benchmark in which the usual sources of train-test leakage are measured and controlled.

* **Label unit:** (peptide, HLA allele) pair. Positives = functional T-cell response; primary negatives (**N2**) = assay-verified non-responses. Cross-study conflicts are removed to a separate table.
* **No shuffled or random negatives are released.** Optional decoy tiers **N1** (same protein) and **N3** (matched on length, predicted binding and antigen region) are *untested* peptides and are labelled as such.
* **Splits:** S1 antigen-group (primary), S2 held-out HLA, S3 held-out source taxon, S5 held-out IEDB study (`split_S5_study_heldout`, P/N2 only), S4 random (reference), each as official train/val/test and 5-fold grouped CV, with train-test similarity bins (<40, 40–59, 60–79, 80–99, 100 %).
* **Diagnostics:** nine baselines, leakage gap, memorisation ratio, shortcut ratio, similarity-bin analysis, cluster-bootstrap CIs.

## Dataset at a glance

| tier | rows | meaning |
|---|---|---|
| P | 6,675 | positive pair (5,512 peptides, 123 alleles) |
| N2 | 5,705 | assay-verified negative pair (1,795 high-, 3,910 medium-confidence) |
| N1 | 3,428 | same-protein decoy for a positive (untested) |
| N3 | 4,675 | matched hard decoy (untested) |

17,278 unique peptides; 5,432 leakage groups (largest 7.1 % of rows). Details: `datasheet.md`, `data_dictionary.csv`, tables in `data/documentation/tables/`.

## Headline results (P vs N2, grouped 5-fold CV; see Table 6–7, Fig 5–8)

* Random splitting (S4) lets 17 % of held-out rows see an identical training peptide and 47 % a near-duplicate; S1 removes both.
* AUPRC of the best peptide model falls from 0.80 (random forest, S4) to 0.64–0.65 (S1; chance 0.54); under S4 a nearest-neighbour lookup matches the ML models (memorisation ratio 0.8–1.2), under S1 it collapses (AUPRC 0.57).
* **Metadata shortcuts remain in P-vs-N2:** a study-identity prior reaches AUPRC 0.90 on S1 and a source-organism prior 0.75, because positives and negatives come from different studies and negatives are often organism-less neoantigen screens. Use S3 (which also removes shared studies) and always report the shortcut baselines.
* The pathogen/vaccine track (`in_pathogen_track`, 6,421 P+N2 rows, 70 % positive) behaves alike: feature models AUPRC 0.78 on S1 (chance 0.70) vs 0.91 for the study prior; results in `data/documentation/*/pathogen_track/` and `data/metadata/results/*_pathogen.csv`.
* Random peptides (N0, analysis-only) are trivially separable (composition alone AUPRC 0.85), showing why assay-verified negatives matter.

## Repository layout

```
data/raw/            IEDB exports (not redistributed; see "Obtaining the raw data")
data/processed/      intermediate and final tables (dataset_all_tiers.csv.gz, leakage_groups.csv.gz, ...)
data/benchmark_splits/  splits_all_tiers.csv.gz + primary P-vs-N2 train/val/test per split
data/metadata/       funnels, manifest.json (versions, hashes, seeds), results/*.csv, run_logs/
data/documentation/  figures (PNG+SVG), tables (CSV+MD), data_dictionary.csv, datasheet.md
data/release/        validated release package with SHA256SUMS.txt
src/{data_processing,negative_curation,clustering,benchmark_generation,modeling,evaluation,visualization,utilities}
tests/               unit tests and an independent split audit
run_pipeline.sh      runs everything end to end
```

## Quick start

```bash
pip install -r requirements.txt            # numpy, pandas, scipy, scikit-learn, matplotlib
python -I src/utilities/make_release.py    # validates the tables and writes data/release/LCI-IEDB_v1.0
```
Use the released data directly:
```python
import pandas as pd
ds = pd.read_csv("LCI-IEDB_v1.0_dataset_all_tiers.csv.gz")
sp = pd.read_csv("LCI-IEDB_v1.0_splits_all_tiers.csv.gz")
d = ds[ds.tier.isin(["P", "N2"])].merge(sp[["row_id", "S1_split", "S1_cv", "S1_bin"]], on="row_id")
train, test = d[d.S1_split == "train"], d[d.S1_split == "test"]
```
Rebuild from raw: `./run_pipeline.sh` (about 50 min on 2 CPUs / 7 GB RAM). Every script is runnable alone (`python -I <script> --help`), logs to `data/metadata/run_logs/`, and appends versions/hashes to `data/metadata/manifest.json`.

## Protein clusters
The MMseqs2 clusters of the parent proteins (`mmseqs easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 1` on UniProt sequences of the accessions in `data/metadata/parent_accessions.txt`) are included as `data/processed/mmseqs_map.tsv` (`parent_iri<TAB>cluster_id`). `09_build_groups.py --external-clusters data/processed/mmseqs_map.tsv` uses them.

## Obtaining the raw data
Download from https://www.iedb.org/database_export_v3.php (CC BY 4.0): `tcell_full_v3`, `mhc_ligand_full` (single file), `epitope_full_v3`, `reference_full_v3`, `MhcAlleleNameList`, and place them as shown in `data/raw/`. SHA-256 of the exports used are recorded in `data/metadata/manifest.json`. `antigen_full_v3` and `epitope_id_to_linear_seq_maps` were downloaded but are **not used** (the epitope export already carries sequence and parent protein; the antigen export has no sequences).

## Reproducibility record
Python 3.13.18; numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2. Global seed 42 (splits, bootstrap, N0, models, binding proxy). IEDB export "Last Updated: November 01, 2026"; processed 8 Nov 2026.

## Known limitations (read before using)
1. **Antigen clustering covers most, not all, parent proteins.** Leakage groups join near-duplicate peptides, shared parent proteins, antigen-family names and MMseqs2 clusters (30 % identity, 80 % coverage) of the parent proteins; 3,124 of 3,230 parent sequences could be retrieved from UniProt. Rows without a parent protein are grouped only by peptide similarity and antigen name. `09_build_groups.py --external-clusters` accepts a cluster map (`parent_iri<TAB>cluster_id`).
2. **Study / organism shortcuts** (see above) are not removed by S1. Report `B3_organism` and `B5_study_prior`.
3. **N1/N3 are untested peptides**, not experimental negatives; N3 matching uses a self-trained binding proxy (median CV AUC 0.91), not NetMHCpan.
4. **Tier-C restriction** (HLA from MHC-binding prediction only) makes 69 % of N2 "medium confidence"; filter with `n2_confidence == "high"` (1,795 rows) for stricter use.
5. **Not included:** XGBoost and protein language-model baselines. The nine reference models use scikit-learn only (GBM = HistGradientBoosting); they document the files and are not a ranking of methods.
6. **IEDB export has no DOI column**; PMIDs and IEDB reference IDs are kept.
7. S3 excludes pairs without a source taxon (3,937 rows, 72 % of them negatives); in the pathogen track S2 has an empty CV fold 0 (4 effective folds).
8. Ascertainment: IEDB is a literature-curated, human-donor, HLA-A*02:01-dominated resource (30 % of positives).

## Citation and licence
Data CC BY 4.0 (derived from IEDB; cite IEDB and the original studies via PMIDs); code MIT. See `LICENSE`.
