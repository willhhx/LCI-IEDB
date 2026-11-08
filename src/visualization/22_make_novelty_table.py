#!/usr/bin/env python3
"""Write the novelty-comparison table (hand-curated; each cell reflects what was actually checked).

"not described" means the source examined does not describe the feature; it is NOT a claim that the full publication lacks it.
Rows marked "(literature knowledge)" were not re-opened during this build and must be verified by the authors before submission.
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
    ["BigMHC (Nature Mach. Intell. 2023)", "MHC-I presentation and immunogenicity", "not verified (Methods not readable when checked)",
     "not verified", "not verified", "not verified", "not verified", "not verified", "Nature MI article page only (2026-11-08); authors to verify against the full text"],
    ["MUNIS (Nat Mach Intell 2025;7:232-243)", "HLA-I presented / CD8+ epitopes in foreign pathogens", "random Swiss-Prot decoys (about 1:5); influenza benchmark uses other peptides of the same viral proteins",
     "exact-epitope removal across all HLA restrictions; no similarity filtering described", "same-protein peptides in the influenza benchmark only", "no (benchmark negatives are untested peptides)",
     "no allele hold-out described; EBV held out as a pathogen", "not described", "PMC article, fetched 2026-11-08"],
    ["T-SCAPE (Dryad, 2025)", "cross-domain immunogenicity scoring", "label 0 given, selection not explained", "page states 'leakage-controlled benchmarks'; method not on the page",
     "not described", "not described", "not described", "not described", "Dryad record, fetched 2026-11-08"],
    ["Typical IEDB-derived immunogenicity training sets (literature knowledge)", "peptide-HLA immunogenicity", "assay negatives from IEDB, often mixed with random/proteome peptides",
     "random or peptide-identity splits", "rare", "partly", "usually trained over many alleles; held-out alleles rare", "rare", "UNVERIFIED generalisation; authors to verify against individual papers or remove"],
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
