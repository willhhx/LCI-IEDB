#!/usr/bin/env python3
"""Build an HLA allele-name normalisation table from the IEDB MhcAlleleNameList.

Every alias (displayed name, synonyms, "includes") of every human MHC entry is
mapped to the IEDB displayed restriction together with its resolution level.
Only human class I entries with ``RestrictionLevel == 'complete molecule'`` are
considered "resolved alleles" by the downstream pipeline; serotype-level or
generic names ("HLA-A2", "HLA class I") are kept in the table but flagged.

Output: ``data/processed/hla_allele_map.csv``
"""
from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, setup_logging, update_manifest, sha256_file  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", type=Path, default=data_dir("raw/mhc_alleles/MhcAlleleNameList.xml"),
                   help="IEDB MhcAlleleNameList.xml")
    p.add_argument("--out", type=Path, default=data_dir("processed/hla_allele_map.csv"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def split_aliases(text: str | None) -> list[str]:
    return [a.strip() for a in (text or "").split("|") if a.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("01_build_hla_allele_map", args.verbose)
    if not args.xml.exists():
        log.error("input not found: %s", args.xml)
        return 2
    rows: dict[str, dict[str, str]] = {}
    n_entries = 0
    try:
        for _, el in ET.iterparse(args.xml):
            if el.tag != "MhcAlleleName":
                continue
            n_entries += 1
            if el.findtext("OrganismNcbiTaxId") == "9606":
                disp = (el.findtext("DisplayedRestriction") or "").strip()
                level = el.findtext("RestrictionLevel") or ""
                cls = el.findtext("Class") or ""
                locus = el.findtext("Locus") or ""
                names = [disp] + split_aliases(el.findtext("Synonyms")) + split_aliases(el.findtext("Includes"))
                for nm in names:
                    # first writer wins so that the displayed name itself is never overridden
                    rows.setdefault(nm, {"alias": nm, "canonical": disp, "restriction_level": level,
                                         "mhc_class": cls, "locus": locus,
                                         "is_resolved_class_I": str(level == "complete molecule" and cls == "I"
                                                                    and disp.startswith("HLA-"))})
            el.clear()
    except (ET.ParseError, OSError) as exc:
        log.error("failed to parse %s: %s", args.xml, exc)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["alias", "canonical", "restriction_level", "mhc_class", "locus",
                                           "is_resolved_class_I"])
        w.writeheader()
        w.writerows(rows.values())
    n_res = sum(1 for r in rows.values() if r["is_resolved_class_I"] == "True")
    log.info("entries=%d  human aliases=%d  resolved class I aliases=%d -> %s", n_entries, len(rows), n_res, args.out)
    update_manifest("01_build_hla_allele_map", {"input": str(args.xml.name), "input_sha256": sha256_file(args.xml),
                                                "aliases": len(rows), "resolved_class_I_aliases": n_res})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
