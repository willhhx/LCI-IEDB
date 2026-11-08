#!/usr/bin/env python3
"""Validate the benchmark tables and assemble the release package ``data/release/LCI-IEDB_v<version>/`` with SHA-256 checksums.

Validation (exit status 1 on any failure; nothing is copied unless all checks pass):
  * row_id unique; tier in {P, N2, N1, N3}; label == 1 exactly for P; no random / shuffled negatives (N0 absent)
  * peptides canonical (20 amino acids) and 8-11 residues; alleles look like HLA-A/B/C*xx:xx
  * no (peptide, allele) pair is both P and N2; no (peptide, allele, tier) duplicates; no pair of the conflicts table is in the dataset
  * every row has a leakage group and split columns; group tables align with the dataset
  * primary benchmark files contain only P and N2 rows and train/val/test are disjoint
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, is_canonical_peptide, repo_root, setup_logging  # noqa: E402

VERSION = "1.0"
ALLELE_RE = r"^HLA-[ABC]\*\d{2,3}:\d{2,3}$"


def validate(log) -> tuple[bool, dict[str, pd.DataFrame]]:
    proc, spl = data_dir("processed"), data_dir("benchmark_splits")
    ds = pd.read_csv(proc / "dataset_all_tiers.csv.gz", dtype=str, keep_default_na=False)
    grp = pd.read_csv(proc / "leakage_groups.csv.gz", dtype=str)
    sp = pd.read_csv(spl / "splits_all_tiers.csv.gz", dtype={"row_id": str})
    conf = pd.read_csv(proc / "pair_conflicts.csv.gz", dtype=str, keep_default_na=False)
    mm = pd.read_csv(proc / "pair_multimer_only_positive.csv.gz", dtype=str, keep_default_na=False)
    fails: list[str] = []

    def check(ok: bool, msg: str) -> None:
        log.info("%s  %s", "PASS" if ok else "FAIL", msg)
        if not ok:
            fails.append(msg)

    check(ds["row_id"].is_unique, "row_id unique")
    check(set(ds["tier"]) == {"P", "N2", "N1", "N3"}, f"tiers present: {sorted(set(ds['tier']))} (no N0 / random negatives)")
    check(bool(((ds["label"] == "1") == (ds["tier"] == "P")).all()), "label == 1 exactly for tier P")
    check(bool(ds["peptide"].map(is_canonical_peptide).all()) and bool(ds["peptide"].str.len().between(8, 11).all()), "peptides canonical, length 8-11")
    check(bool(ds["allele"].str.match(ALLELE_RE).all()), "all alleles are two-field HLA-A/B/C names")
    check(bool((ds["length"].astype(int) == ds["peptide"].str.len()).all()), "length column equals peptide length")
    key = ds["peptide"] + "|" + ds["allele"]
    check(not (set(key[ds["tier"] == "P"]) & set(key[ds["tier"] == "N2"])), "no (peptide, allele) pair is both P and N2")
    check(not (key + "|" + ds["tier"]).duplicated().any(), "no duplicated (peptide, allele, tier)")
    check(not (set(conf["peptide"] + "|" + conf["allele"]) & set(key[ds["tier"].isin(["P", "N2"])])), "conflict pairs are excluded from P and N2")
    check(set(grp["row_id"]) == set(ds["row_id"]) and set(sp["row_id"]) == set(ds["row_id"]), "leakage groups and splits cover every row")
    check(bool(sp[[f"{s}_cv" for s in ("S1", "S2", "S3", "S4")]].notna().all().all()), "CV fold columns complete")
    for s in ("S1", "S2", "S3", "S4"):
        d = pd.read_csv(spl / {"S1": "S1_antigen_cluster", "S2": "S2_heldout_HLA", "S3": "S3_heldout_pathogen", "S4": "S4_random_reference"}[s] / "primary_P_vs_N2_train.csv",
                        dtype=str, keep_default_na=False)
        t = pd.read_csv(spl / {"S1": "S1_antigen_cluster", "S2": "S2_heldout_HLA", "S3": "S3_heldout_pathogen", "S4": "S4_random_reference"}[s] / "primary_P_vs_N2_test.csv",
                        dtype=str, keep_default_na=False)
        check(set(d["tier"]) <= {"P", "N2"} and set(t["tier"]) <= {"P", "N2"} and not (set(d["row_id"]) & set(t["row_id"])), f"{s}: primary files only P/N2 and train/test disjoint")
    check(int((ds["tier"] == "P").sum()) > 0 and int((ds["tier"] == "N2").sum()) > 0, "positives and N2 negatives present")
    return not fails, {"ds": ds, "grp": grp, "sp": sp, "conf": conf, "mm": mm}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=data_dir("release"))
    args = ap.parse_args(argv)
    log = setup_logging("make_release")
    try:
        ok, t = validate(log)
    except OSError as exc:
        log.error("missing input: %s", exc)
        return 2
    if not ok:
        log.error("validation failed; release not written")
        return 1
    root = repo_root()
    out = args.outdir / f"LCI-IEDB_v{VERSION}"
    if out.exists():
        shutil.rmtree(out)
    (out / "metadata").mkdir(parents=True)
    ds = t["ds"]
    ds.to_csv(out / f"LCI-IEDB_v{VERSION}_dataset_all_tiers.csv.gz", index=False)
    ds[ds["tier"].isin(["P", "N2"])].to_csv(out / f"LCI-IEDB_v{VERSION}_primary_P_vs_N2.csv.gz", index=False)
    for src, dst in [("processed/leakage_groups.csv.gz", "leakage_groups.csv.gz"), ("benchmark_splits/splits_all_tiers.csv.gz", "splits_all_tiers.csv.gz"),
                     ("processed/pair_conflicts.csv.gz", "conflicts_excluded_pairs.csv.gz"),
                     ("processed/pair_multimer_only_positive.csv.gz", "multimer_only_positive_pairs.csv.gz")]:
        shutil.copy2(data_dir() / src, out / f"LCI-IEDB_v{VERSION}_{dst}")
    shutil.copytree(data_dir("benchmark_splits"), out / "splits_primary_P_vs_N2", ignore=shutil.ignore_patterns("splits_all_tiers.csv.gz"))
    for f in ["manifest.json", "funnel_tcell.json", "funnel_mhc_ligand.json", "funnel_epitope_pool.json", "funnel_labels.json", "split_stats.csv",
              "group_stats.json", "taxon_groups.csv", "references_used.csv", "decoy_summary.json", "decoy_match_report.csv", "binding_proxy_cv.csv", "parent_accessions.txt"]:
        if (data_dir("metadata") / f).exists():
            shutil.copy2(data_dir("metadata") / f, out / "metadata" / f)
    shutil.copytree(data_dir("metadata/results"), out / "metadata" / "results")
    shutil.copytree(data_dir("documentation/figures"), out / "documentation" / "figures")
    shutil.copytree(data_dir("documentation/tables"), out / "documentation" / "tables")
    shutil.copy2(data_dir("documentation/data_dictionary.csv"), out / "data_dictionary.csv")
    for name in ["README.md", "LICENSE", "CHANGELOG.md", "requirements.txt", "environment.yml"]:
        if not (root / name).exists():
            log.error("missing %s", name)
            return 2
        shutil.copy2(root / name, out / name)
    if not (data_dir("documentation") / "datasheet.md").exists():
        log.error("missing datasheet.md")
        return 2
    shutil.copy2(data_dir("documentation/datasheet.md"), out / "datasheet.md")
    lines = [f"{sha256(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*")) if p.is_file()]
    (out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    log.info("release written: %s (%d files)", out, len(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
