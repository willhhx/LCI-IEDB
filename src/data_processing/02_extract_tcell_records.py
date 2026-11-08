#!/usr/bin/env python3
"""Stream the IEDB T-cell assay export and keep record-level human HLA class I peptide assays.

Selection funnel (each step is counted and written to ``data/metadata/funnel_tcell.json``):

 1. all T-cell assay rows
 2. host = Homo sapiens
 3. epitope is a linear peptide
 4. MHC restriction class = I
 5. assay antigen is the epitope peptide itself (``Epitope Relation == Epitope``, same sequence)
 6. no modified residues and not an adoptive-transfer experiment
 7. canonical amino acids only, length 8-11
 8. restriction resolved to a single HLA class I allele ("complete molecule" in the IEDB allele list)

Two side products are written for the negative-curation stage:
 * ``tcell_peptides_any_positive.txt.gz``  - every linear peptide with a positive T-cell record in ANY host/class
 * ``tcell_peptides_any_tested.txt.gz``    - every linear peptide with ANY T-cell record

Outputs (``data/processed``): ``tcell_records_human_hlaI.csv.gz`` and the two peptide lists.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import (MAX_LEN, MIN_LEN, data_dir, setup_logging, sha256_file, taxon_id_from_iri,  # noqa: E402
                              uniprot_from_iri, update_manifest, write_json)

# column index in tcell_full_v3.csv (two header rows; verified against the file header at run time)
COLS: dict[int, str] = {
    0: "assay_iri", 1: "reference_iri", 3: "pmid", 4: "submission_id", 7: "ref_year", 9: "epitope_iri",
    10: "epi_type", 11: "peptide", 13: "modified_residues", 15: "start", 16: "end", 19: "source_molecule",
    21: "parent_name", 22: "parent_iri", 23: "source_organism", 25: "species", 26: "species_iri", 43: "host",
    49: "mhc_present", 50: "vivo_process", 51: "disease", 94: "vitro_process", 114: "adoptive_transfer",
    118: "method", 119: "response", 120: "units", 122: "qualitative", 124: "quantitative", 125: "n_tested",
    126: "n_positive_subjects", 131: "effector_cell", 133: "culture_condition", 141: "mhc_raw",
    143: "mhc_evidence", 145: "mhc_class", 146: "aa_relation", 147: "aa_type", 148: "aa_name",
}
EXPECTED_HEADER: dict[int, tuple[str, str]] = {
    0: ("Assay ID", "IEDB IRI"), 11: ("Epitope", "Name"), 43: ("Host", "Name"), 118: ("Assay", "Method"),
    122: ("Assay", "Qualitative Measurement"), 141: ("MHC Restriction", "Name"), 145: ("MHC Restriction", "Class"),
    148: ("Assay Antigen", "Name"), 133: ("Effector Cell", "Culture Condition"), 126: ("Assay", "Number of Subjects Positive"),
}

FUNCTIONAL_METHODS = {
    "ELISPOT", "ICS", "intracellular staining", "ELISA", "cytometric bead array", "51 chromium",
    "biological activity", "3H-thymidine", "CFSE", "BrdU", "High throughput multiplexed assay",
    "in vitro assay", "RNA/DNA detection", "reporter gene assay", "radio immuno assay (RIA)", "bioassay",
}
MULTIMER_METHODS = {"multimer/tetramer"}


def method_family(m: str) -> str:
    if m in FUNCTIONAL_METHODS:
        return "functional"
    if m in MULTIMER_METHODS:
        return "multimer"
    return "other"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tcell", type=Path, default=data_dir("raw/tcell/tcell_full_v3.csv"))
    p.add_argument("--allele-map", type=Path, default=data_dir("processed/hla_allele_map.csv"))
    p.add_argument("--outdir", type=Path, default=data_dir("processed"))
    p.add_argument("--chunksize", type=int, default=100_000)
    p.add_argument("--max-rows", type=int, default=0, help="debug: stop after this many rows (0 = all)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def check_header(path: Path) -> None:
    hdr = pd.read_csv(path, header=None, nrows=2, dtype=str, keep_default_na=False)
    for idx, (a, b) in EXPECTED_HEADER.items():
        if (hdr.iat[0, idx], hdr.iat[1, idx]) != (a, b):
            raise ValueError(f"unexpected IEDB header at column {idx}: {(hdr.iat[0, idx], hdr.iat[1, idx])} != {(a, b)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("02_extract_tcell_records", args.verbose)
    for pth in (args.tcell, args.allele_map):
        if not pth.exists():
            log.error("input not found: %s", pth)
            return 2
    try:
        check_header(args.tcell)
        amap = pd.read_csv(args.allele_map, dtype=str, keep_default_na=False)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 1
    alias_to = {r.alias: (r.canonical, r.restriction_level, r.is_resolved_class_I == "True") for r in amap.itertuples()}
    alias_lower = {k.lower(): v for k, v in alias_to.items()}

    funnel: collections.OrderedDict[str, int] = collections.OrderedDict()
    def bump(k: str, n: int) -> None:
        funnel[k] = funnel.get(k, 0) + int(n)

    unresolved: collections.Counter[str] = collections.Counter()
    any_pos: set[str] = set()
    any_tested: set[str] = set()
    kept: list[pd.DataFrame] = []
    reader = pd.read_csv(args.tcell, header=None, skiprows=2, usecols=sorted(COLS), names=[COLS[i] for i in sorted(COLS)],
                         dtype=str, keep_default_na=False, chunksize=args.chunksize, encoding="utf-8")
    seen = 0
    try:
        for ch in reader:
            seen += len(ch)
            bump("1_all_tcell_rows", len(ch))
            lin = ch["epi_type"].eq("Linear peptide")
            any_tested.update(ch.loc[lin, "peptide"])
            any_pos.update(ch.loc[lin & ch["qualitative"].str.startswith("Positive"), "peptide"])
            m = ch["host"].str.startswith("Homo sapiens")
            bump("2_host_human", m.sum()); ch = ch[m]
            m = ch["epi_type"].eq("Linear peptide")
            bump("3_linear_peptide", m.sum()); ch = ch[m]
            m = ch["mhc_class"].eq("I")
            bump("4_mhc_class_I", m.sum()); ch = ch[m]
            m = ch["aa_relation"].eq("Epitope") & ch["aa_type"].eq("Linear peptide") & ch["aa_name"].eq(ch["peptide"])
            bump("5_assay_antigen_is_epitope_peptide", m.sum()); ch = ch[m]
            m = ch["modified_residues"].eq("") & ch["adoptive_transfer"].ne("Y")
            bump("6_unmodified_not_adoptive_transfer", m.sum()); ch = ch[m]
            ln = ch["peptide"].str.len()
            m = ch["peptide"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+") & ln.between(MIN_LEN, MAX_LEN)
            bump("7_canonical_len_8_11", m.sum()); ch = ch[m]
            res = ch["mhc_raw"].map(lambda s: alias_to.get(s) or alias_lower.get(s.lower()))
            ok = res.map(lambda t: bool(t and t[2]))
            unresolved.update(ch.loc[~ok, "mhc_raw"])
            bump("8_resolved_hla_class_I_allele", ok.sum())
            ch = ch[ok].copy()
            res = res[ok]
            ch["allele"] = res.map(lambda t: t[0])
            kept.append(ch)
            if args.max_rows and seen >= args.max_rows:
                log.warning("stopping early at %d rows (--max-rows)", seen)
                break
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        log.error("failed while reading %s: %s", args.tcell, exc)
        return 1

    df = pd.concat(kept, ignore_index=True)
    df["assay_id"] = df["assay_iri"].str.rsplit("/", n=1).str[-1]
    df["reference_id"] = df["reference_iri"].str.rsplit("/", n=1).str[-1]
    df["epitope_id"] = df["epitope_iri"].str.rsplit("/", n=1).str[-1]
    df["label_record"] = df["qualitative"].str.startswith("Positive").astype(int)
    df["strength"] = df["qualitative"].map({"Positive-High": "High", "Positive-Intermediate": "Intermediate",
                                             "Positive-Low": "Low", "Positive": "Unspecified", "Negative": "NA"}).fillna("NA")
    df["method_family"] = df["method"].map(method_family)
    df["taxon_id"] = df["species_iri"].map(taxon_id_from_iri)
    df["uniprot"] = df["parent_iri"].map(uniprot_from_iri)
    keep = ["assay_id", "reference_id", "pmid", "submission_id", "ref_year", "epitope_id", "peptide", "start", "end",
            "parent_name", "parent_iri", "uniprot", "source_molecule", "source_organism", "species", "taxon_id",
            "mhc_raw", "allele", "mhc_evidence", "mhc_present", "method", "method_family", "response", "units",
            "quantitative", "qualitative", "label_record", "strength", "n_tested", "n_positive_subjects",
            "effector_cell", "culture_condition", "vivo_process", "vitro_process", "disease"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "tcell_records_human_hlaI.csv.gz"
    df[keep].to_csv(out, index=False, compression="gzip")
    for name, s in (("tcell_peptides_any_positive.txt.gz", any_pos), ("tcell_peptides_any_tested.txt.gz", any_tested)):
        with gzip.open(args.outdir / name, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(s)) + "\n")
    pd.Series(unresolved, name="n_rows").rename_axis("mhc_raw").sort_values(ascending=False).to_csv(
        args.outdir / "tcell_unresolved_allele_names.csv")
    write_json(data_dir("metadata/funnel_tcell.json"), dict(funnel))
    update_manifest("02_extract_tcell_records", {"input": args.tcell.name, "input_sha256": sha256_file(args.tcell),
                                                 "funnel": dict(funnel), "records_out": len(df),
                                                 "peptides_any_positive": len(any_pos), "peptides_any_tested": len(any_tested)})
    for k, v in funnel.items():
        log.info("%-45s %9d", k, v)
    log.info("wrote %s (%d records)", out, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
