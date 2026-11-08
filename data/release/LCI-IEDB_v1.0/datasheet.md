# Datasheet for LCI-IEDB v1.0

(Following Gebru et al., *Datasheets for Datasets*.)

## Motivation
**Purpose.** A benchmark for human CD8⁺ T-cell immunogenicity prediction (MHC class I, linear peptides 8–11 aa) in which leakage between training and test data is measured and controlled, and negatives are experimentally verified rather than random.
**Created by** Haoxi Hu (corresponding author, hu.haoxi@northeastern.edu), 2026. **Funding:** none declared.

## Composition
* **Instances:** 20,483 (peptide, HLA allele) rows: P 6,675; N2 5,705; N1 3,428; N3 4,675. Unique peptides 17,278; HLA alleles 130 in P/N2 (123 in P).
* **Labels:** P = functional T-cell response (ELISpot, ICS, ELISA/CBA, 51Cr, proliferation, …) in ≥1 reference and no reference with only negative outcomes; N2 = assay-verified negative (no positive reference, no multimer-positive record); N1/N3 = untested decoys (no T-cell record), provided as alternative negative sets.
* **Record filters:** host human; linear unmodified peptide, 20 canonical residues, length 8–11; MHC class I; HLA allele resolved to two fields; functional assays on PBMC / CD8⁺ / T cells / TIL, direct ex vivo or short-term restimulated; HLA restriction established (evidence tiers A/B/C kept as a column).
* **Excluded to the conflicts table:** 1,103 pairs with positive and negative-only references (or a multimer-positive record); 3,525 multimer-only positives are in a separate supplementary table.
* **Metadata kept:** IEDB reference ids, PMIDs, methods, response, effector cells, culture conditions, exposure context, publication year, subjects tested, MHC-binding evidence, parent protein, source taxon (see `data_dictionary.csv`). **DOIs are not in the IEDB export.**
* **Splits:** S1 antigen-group, S2 held-out HLA (purged on near-duplicate peptide components), S3 held-out taxon group (purged on leakage group; unspecified taxon excluded), S4 random; official 70/10/20 plus 5-fold CV; similarity bins from max peptide similarity to training.
* **Sensitive data:** none; the data describe peptides and HLA alleles, not individuals. Donor-level data are not in the export.
* **Known gaps/biases:** HLA-A*02:01 accounts for 30 % of positives and the 8 most frequent alleles for 66 % of P and N2 rows; viral sources dominate positives (56 %); half of N2 rows have no stated source organism; positives and negatives originate from different studies (study prior AUPRC 0.90 on S1).

## Collection process
Derived from IEDB exports `tcell_full_v3`, `mhc_ligand_full`, `epitope_full_v3`, `reference_full_v3`, `MhcAlleleNameList` (Last Updated 2026-11-01; SHA-256 in `data/metadata/manifest.json`), processed 2026-11-08 with the scripts in `src/`. No new experimental data. Binding-strength matching for N3 uses a ridge regression trained on IEDB affinities (66 alleles, median CV AUC 0.91), not NetMHCpan.

## Preprocessing / cleaning
HLA names normalised through the IEDB allele list; modified, non-canonical and non-8–11-mer peptides removed; duplicate records merged per (peptide, allele); multimer-only evidence kept out of labels; peptides positive for one allele and negative for another are kept and flagged (`peptide_label_ambiguous`). The processing funnel is in Table 1 and `data/metadata/funnel_*.json`.

## Uses
**Intended:** benchmarking immunogenicity models under leakage-controlled splits; studying shortcut learning; negative-set design.
**Not suitable for:** clinical decisions; claims about immunogenicity in non-human hosts, MHC class II or long peptides; treating N1/N3 as true non-immunogens; estimating population-level epitope frequencies.
**Cautions:** report shortcut baselines (organism, study, allele, nearest neighbour) with any model; leakage groups use MMseqs2 clusters (30 % identity) for the 3,124 of 3,230 parent proteins that could be retrieved; the remaining rows are grouped by peptide similarity and antigen name.

## Distribution and maintenance
Released as a versioned package (`data/release/LCI-IEDB_v1.0`, SHA-256 checksums) hosted at https://github.com/willhhx/LCI-IEDB. Data CC BY 4.0, code MIT. Maintainer: the author above. Future IEDB releases can be re-processed with `run_pipeline.sh`.
