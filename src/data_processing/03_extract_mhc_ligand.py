#!/usr/bin/env python3
"""Stream the (10 GB) IEDB MHC-ligand export and keep human HLA class I linear-peptide records.

The MHC-ligand data are NOT label sources for immunogenicity.  They are used for
 (a) annotating each T-cell labelled peptide/allele with independent evidence of MHC binding or
     presentation, and
 (b) providing a candidate pool and a measured-binding training set for hard-negative (N3) matching.

Selection: host Homo sapiens; linear peptide; unmodified; canonical amino acids; length 8-11;
MHC class I; MHC restriction resolved to a single HLA allele (IEDB allele list).

Output: ``data/processed/mhc_ligand_records_human_hlaI.csv.gz``  and a funnel in
``data/metadata/funnel_mhc_ligand.json``.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import (MAX_LEN, MIN_LEN, data_dir, setup_logging, sha256_file, taxon_id_from_iri,  # noqa: E402
                              uniprot_from_iri, update_manifest, write_json)

COLS: dict[int, str] = {
    0: "assay_iri", 1: "reference_iri", 3: "pmid", 9: "epitope_iri", 10: "epi_type", 11: "peptide",
    13: "modified_residues", 15: "start", 16: "end", 21: "parent_name", 22: "parent_iri", 23: "source_organism",
    25: "species", 26: "species_iri", 43: "host", 90: "method", 91: "response", 92: "units", 94: "qualitative",
    95: "inequality", 96: "quantitative", 107: "mhc_raw", 111: "mhc_class",
}
EXPECTED_HEADER: dict[int, tuple[str, str]] = {
    0: ("Assay ID", "IEDB IRI"), 11: ("Epitope", "Name"), 43: ("Host", "Name"), 90: ("Assay", "Method"),
    94: ("Assay", "Qualitative Measurement"), 96: ("Assay", "Quantitative measurement"),
    107: ("MHC Restriction", "Name"), 111: ("MHC Restriction", "Class"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ligand", type=Path, default=data_dir("raw/mhc_ligand/mhc_ligand_full.csv"))
    p.add_argument("--allele-map", type=Path, default=data_dir("processed/hla_allele_map.csv"))
    p.add_argument("--outdir", type=Path, default=data_dir("processed"))
    p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("--max-rows", type=int, default=0, help="debug: stop after this many rows (0 = all)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("03_extract_mhc_ligand", args.verbose)
    for pth in (args.ligand, args.allele_map):
        if not pth.exists():
            log.error("input not found: %s", pth)
            return 2
    try:
        hdr = pd.read_csv(args.ligand, header=None, nrows=2, dtype=str, keep_default_na=False)
        for idx, (a, b) in EXPECTED_HEADER.items():
            if (hdr.iat[0, idx], hdr.iat[1, idx]) != (a, b):
                raise ValueError(f"unexpected IEDB header at column {idx}: {(hdr.iat[0, idx], hdr.iat[1, idx])}")
        amap = pd.read_csv(args.allele_map, dtype=str, keep_default_na=False)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 1
    alias_to = {r.alias: r.canonical for r in amap.itertuples() if r.is_resolved_class_I == "True"}
    alias_lower = {k.lower(): v for k, v in alias_to.items()}

    funnel: collections.OrderedDict[str, int] = collections.OrderedDict()
    def bump(k: str, n: int) -> None:
        funnel[k] = funnel.get(k, 0) + int(n)

    keep = ["assay_id", "reference_id", "pmid", "epitope_id", "peptide", "start", "end", "parent_name", "parent_iri",
            "uniprot", "source_organism", "species", "taxon_id", "mhc_raw", "allele", "method", "response", "units",
            "qualitative", "inequality", "quantitative"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "mhc_ligand_records_human_hlaI.csv.gz"
    n_out = 0
    seen = 0
    reader = pd.read_csv(args.ligand, header=None, skiprows=2, usecols=sorted(COLS), names=[COLS[i] for i in sorted(COLS)],
                         dtype=str, keep_default_na=False, chunksize=args.chunksize, encoding="utf-8")
    try:
        for ch in reader:
            seen += len(ch)
            bump("1_all_mhc_ligand_rows", len(ch))
            m = ch["host"].str.startswith("Homo sapiens"); bump("2_host_human", m.sum()); ch = ch[m]
            m = ch["epi_type"].eq("Linear peptide") & ch["modified_residues"].eq(""); bump("3_linear_unmodified", m.sum()); ch = ch[m]
            m = ch["mhc_class"].eq("I"); bump("4_mhc_class_I", m.sum()); ch = ch[m]
            m = ch["peptide"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+") & ch["peptide"].str.len().between(MIN_LEN, MAX_LEN)
            bump("5_canonical_len_8_11", m.sum()); ch = ch[m]
            al = ch["mhc_raw"].map(lambda s: alias_to.get(s) or alias_lower.get(s.lower()))
            ok = al.notna(); bump("6_resolved_hla_class_I_allele", ok.sum())
            ch = ch[ok].copy(); ch["allele"] = al[ok]
            ch["assay_id"] = ch["assay_iri"].str.rsplit("/", n=1).str[-1]
            ch["reference_id"] = ch["reference_iri"].str.rsplit("/", n=1).str[-1]
            ch["epitope_id"] = ch["epitope_iri"].str.rsplit("/", n=1).str[-1]
            ch["taxon_id"] = ch["species_iri"].map(taxon_id_from_iri)
            ch["uniprot"] = ch["parent_iri"].map(uniprot_from_iri)
            ch[keep].to_csv(out, mode="w" if n_out == 0 else "a", header=(n_out == 0), index=False, compression="gzip")
            n_out += len(ch)
            if args.max_rows and seen >= args.max_rows:
                log.warning("stopping early at %d rows (--max-rows)", seen)
                break
            if seen % 2_000_000 < args.chunksize:
                log.info("... %d rows read", seen)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        log.error("failed while reading %s: %s", args.ligand, exc)
        return 1
    write_json(data_dir("metadata/funnel_mhc_ligand.json"), dict(funnel))
    update_manifest("03_extract_mhc_ligand", {"input": args.ligand.name, "input_sha256": sha256_file(args.ligand),
                                              "funnel": dict(funnel), "records_out": n_out})
    for k, v in funnel.items():
        log.info("%-40s %10d", k, v)
    log.info("wrote %s (%d records)", out, n_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
