#!/usr/bin/env python3
"""Write the novelty-comparison table (hand-curated; each cell reflects what was actually checked).

"not described" means the source examined does not describe the feature; it is NOT a claim that the full publication lacks it.
Cells reading "not verified" mean the source text could not be read; they are not claims about the resource.
Output: ``data/documentation/tables/novelty_comparison.{csv,md}``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, setup_logging  # noqa: E402
from utilities.tables import to_md  # noqa: E402

COLS = ["resource", "primary task", "negatives", "homology / antigen-level control", "hard negatives", "assay-verified negatives",
        "allele-level control", "similarity-bin analysis", "how checked"]
ROWS = [
    ["LCI-IEDB (this work)", "human CD8+ T-cell immunogenicity, MHC-I, 8-11-mers", "N2 assay-verified (primary); N1/N3 decoys (optional)",
     "leakage groups (near-duplicate peptides + parent protein + antigen family); S1 antigen-group split; MMseqs2 clusters of parent proteins (30 % identity, 80 % coverage)",
     "yes: N3 matched on length, binding strength and antigen region", "yes: N2, with evidence tier and confidence", "yes: S2 held-out HLA",
     "yes: 100 / 80-99 / 60-79 / 40-59 / <40 %, in the released splits", "this study"],
    ["BigMHC (Nature Mach. Intell. 2023)", "MHC-I presentation and immunogenicity (method; public data and code)", "presentation: random decoys (900,592 in the test set); immunogenicity sets: assay-validated negatives",
     "split rules not readable (paywalled); T-SCAPE reports that overlapping peptides are allowed if the allele differs", "random decoys (presentation task)", "yes, in the immunogenicity sets (per article figure legend)",
     "not verified", "not verified", "Nature MI article page 2026-11-08 (abstract, data statement, figure legends); T-SCAPE preprint for the split description"],
    ["MUNIS (Nat Mach Intell 2025;7:232-243)", "HLA-I presented / CD8+ epitopes in foreign pathogens", "random Swiss-Prot decoys (about 1:5); influenza benchmark uses other peptides of the same viral proteins",
     "exact-epitope removal across all HLA restrictions; no similarity filtering described", "same-protein peptides in the influenza benchmark only", "no (benchmark negatives are untested peptides)",
     "no allele hold-out described; EBV held out as a pathogen", "not described", "PMC article, fetched 2026-11-08"],
    ["T-SCAPE (Sci. Adv. 2025; bioRxiv 10.1101/2025.05.11.653308)", "cross-domain immunogenicity scoring (method)", "negatives of the two evaluation sets (neoantigen 655, infectious disease 605); definition not found in the text",
     "9-mer CD-HIT filter: training peptides with fewer than 2 mismatches to any benchmark peptide removed; no antigen, protein, organism or study hold-out described",
     "not described", "not described", "no HLA hold-out described", "not described", "bioRxiv full text read 2026-11-08 (Methods and Results)"],
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=data_dir("documentation/tables"))
    args = p.parse_args(argv)
    log = setup_logging("22_make_novelty_table")
    df = pd.DataFrame(ROWS, columns=COLS)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "novelty_comparison.csv", index=False)
    (args.outdir / "novelty_comparison.md").write_text("**Novelty comparison with existing immunogenicity / epitope resources**\n\n" + to_md(df))
    log.info("novelty table written (%d rows)", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
