#!/usr/bin/env python3
"""Extract the IEDB epitope table into a pool of unmodified linear 8-11-mer peptides with parent-protein metadata.

The pool supplies same-protein candidate peptides for the N1 negative control.  Peptides are kept regardless of
host/assay type; label safety is enforced later by removing every peptide that has ANY T-cell assay record.

Output: ``data/processed/epitope_pool.csv.gz``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import (MAX_LEN, MIN_LEN, data_dir, setup_logging, sha256_file, taxon_id_from_iri,  # noqa: E402
                              uniprot_from_iri, update_manifest, write_json)

COLS: dict[int, str] = {0: "epitope_iri", 1: "epi_type", 2: "peptide", 3: "modified_residues", 5: "start", 6: "end",
                        9: "source_molecule", 11: "parent_name", 12: "parent_iri", 13: "source_organism",
                        15: "species", 16: "species_iri"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epitope", type=Path, default=data_dir("raw/epitope/epitope_full_v3.csv"))
    p.add_argument("--out", type=Path, default=data_dir("processed/epitope_pool.csv.gz"))
    p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("03b_extract_epitope_pool", args.verbose)
    if not args.epitope.exists():
        log.error("input not found: %s", args.epitope)
        return 2
    try:
        hdr = pd.read_csv(args.epitope, header=None, nrows=2, dtype=str, keep_default_na=False)
        if (hdr.iat[1, 2], hdr.iat[1, 11], hdr.iat[1, 12]) != ("Name", "Molecule Parent", "Molecule Parent IRI"):
            raise ValueError("unexpected epitope_full_v3 header layout")
        funnel = {"rows": 0, "linear_unmodified": 0, "canonical_len_8_11": 0, "with_parent_or_species": 0}
        n_out = 0
        reader = pd.read_csv(args.epitope, header=None, skiprows=2, usecols=sorted(COLS), names=[COLS[i] for i in sorted(COLS)],
                             dtype=str, keep_default_na=False, chunksize=args.chunksize, encoding="utf-8")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for ch in reader:
            funnel["rows"] += len(ch)
            ch = ch[ch["epi_type"].eq("Linear peptide") & ch["modified_residues"].eq("")]
            funnel["linear_unmodified"] += len(ch)
            ch = ch[ch["peptide"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+") & ch["peptide"].str.len().between(MIN_LEN, MAX_LEN)]
            funnel["canonical_len_8_11"] += len(ch)
            ch = ch[ch["parent_iri"].ne("") | ch["species_iri"].ne("")].copy()
            funnel["with_parent_or_species"] += len(ch)
            ch["epitope_id"] = ch["epitope_iri"].str.rsplit("/", n=1).str[-1]
            ch["taxon_id"] = ch["species_iri"].map(taxon_id_from_iri)
            ch["uniprot"] = ch["parent_iri"].map(uniprot_from_iri)
            ch[["epitope_id", "peptide", "start", "end", "parent_name", "parent_iri", "uniprot", "source_molecule",
                "source_organism", "species", "taxon_id"]].to_csv(args.out, mode="w" if n_out == 0 else "a",
                                                                  header=(n_out == 0), index=False, compression="gzip")
            n_out += len(ch)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 1
    write_json(data_dir("metadata/funnel_epitope_pool.json"), funnel)
    update_manifest("03b_extract_epitope_pool", {"input": args.epitope.name, "input_sha256": sha256_file(args.epitope), **funnel})
    log.info("funnel: %s -> %s", funnel, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
