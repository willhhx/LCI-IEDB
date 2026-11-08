#!/usr/bin/env python3
"""Build Tables 1-7 (CSV + Markdown) from the released dataset, split statistics and evaluation results.

Table 1  dataset statistics (processing funnel and tier summary)
Table 2  HLA allele distribution
Table 3  source-organism distribution
Table 4  positive / negative composition (evidence, assays, lengths)
Table 5  split statistics
Table 6  baseline performance per split (N2 benchmark)
Table 7  leakage-gap analysis (S4 minus S1-S3) with memorisation ratio
Outputs: ``data/documentation/tables/tableN_*.csv`` and ``.md``.  Tables 6-7 need ``12_metrics.py`` results and are skipped if absent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, setup_logging  # noqa: E402
from utilities.tables import to_md  # noqa: E402
from visualization.style import MODEL_LABEL, MODEL_ORDER, SPLIT_LABEL  # noqa: E402

TIER_NAME = {"P": "Positive (P)", "N2": "N2 assay-verified negative", "N1": "N1 same-protein decoy", "N3": "N3 matched hard decoy"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=data_dir("documentation/tables"))
    p.add_argument("--results", type=Path, default=data_dir("metadata/results"))
    p.add_argument("--track", default="all", choices=["all", "pathogen"])
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def save(df: pd.DataFrame, outdir: Path, name: str, caption: str) -> None:
    df.to_csv(outdir / f"{name}.csv", index=False)
    (outdir / f"{name}.md").write_text(f"**{caption}**\n\n{to_md(df)}")


def fmt_ci(v: float, lo: float, hi: float, nd: int = 3) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" if np.isfinite(lo) else f"{v:.{nd}f}"


def table1(ds: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    rows = []
    f2, f3 = manifest["02_extract_tcell_records"]["funnel"], manifest["03_extract_mhc_ligand"]["funnel"]
    for k, v in f2.items():
        rows.append({"section": "T-cell assay records (IEDB tcell_full_v3)", "step": k.split("_", 1)[1].replace("_", " "), "count": v})
    for k, v in f3.items():
        rows.append({"section": "MHC ligand records (IEDB mhc_ligand_full; used for binding evidence only)",
                     "step": k.split("_", 1)[1].replace("_", " "), "count": v})
    fl = json.loads((data_dir("metadata") / "funnel_labels.json").read_text())
    for k in ("pairs_total", "pairs_positive", "pairs_negative", "pairs_negative_high_confidence", "pairs_negative_medium_confidence",
              "pairs_conflict", "pairs_multimer_only_positive", "references_used"):
        rows.append({"section": "Peptide-allele pair labels", "step": k.replace("_", " "), "count": fl[k]})
    for t, g in ds.groupby("tier", sort=False):
        rows.append({"section": "Released dataset", "step": f"{TIER_NAME[t]}: rows", "count": len(g)})
        rows.append({"section": "Released dataset", "step": f"{TIER_NAME[t]}: unique peptides", "count": g["peptide"].nunique()})
        rows.append({"section": "Released dataset", "step": f"{TIER_NAME[t]}: HLA alleles", "count": g["allele"].nunique()})
    rows.append({"section": "Released dataset", "step": "unique peptides (all tiers)", "count": ds["peptide"].nunique()})
    rows.append({"section": "Released dataset", "step": "rows in the primary benchmark (P + N2)", "count": int(ds["tier"].isin(["P", "N2"]).sum())})
    return pd.DataFrame(rows)


def table2(ds: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    piv = ds.pivot_table(index="allele", columns="tier", values="row_id", aggfunc="count", fill_value=0)
    for t in TIER_NAME:
        if t not in piv:
            piv[t] = 0
    piv = piv[list(TIER_NAME)]
    piv["total"] = piv.sum(axis=1)
    piv["locus"] = piv.index.str.extract(r"HLA-([ABC])\*", expand=False).fillna("other").to_numpy()
    piv["positive_fraction_P_vs_N2"] = (piv["P"] / (piv["P"] + piv["N2"]).replace(0, np.nan)).round(3)
    out = piv.sort_values("total", ascending=False).head(top).reset_index()
    out = out.rename(columns=TIER_NAME)
    rest = piv.sort_values("total", ascending=False).iloc[top:]
    tail = {"allele": f"other {len(rest)} alleles", "locus": "", "total": int(rest["total"].sum()), "positive_fraction_P_vs_N2": np.nan}
    tail.update({TIER_NAME[t]: int(rest[t].sum()) for t in TIER_NAME})
    return pd.concat([out, pd.DataFrame([tail])], ignore_index=True)[["allele", "locus"] + list(TIER_NAME.values()) + ["total", "positive_fraction_P_vs_N2"]]


def table3(ds: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    d = ds.copy()
    d["group"] = np.where(d["taxon_group"].eq("") | d["taxon_group"].eq("unspecified"), "unspecified", d["taxon_group"])
    piv = d.pivot_table(index=["category", "group"], columns="tier", values="row_id", aggfunc="count", fill_value=0)
    for t in TIER_NAME:
        if t not in piv:
            piv[t] = 0
    piv = piv[list(TIER_NAME)]
    piv["total"] = piv.sum(axis=1)
    piv["peptides_P"] = d[d["tier"] == "P"].groupby(["category", "group"])["peptide"].nunique().reindex(piv.index).fillna(0).astype(int)
    piv = piv.reset_index().sort_values("total", ascending=False)
    head = piv.head(top)
    rest = piv.iloc[top:]
    tail = {"category": "(all)", "group": f"other {len(rest)} source groups", "total": int(rest["total"].sum()), "peptides_P": int(rest["peptides_P"].sum())}
    tail.update({t: int(rest[t].sum()) for t in TIER_NAME})
    out = pd.concat([head, pd.DataFrame([tail])], ignore_index=True).rename(columns=TIER_NAME)
    return out.rename(columns={"group": "source taxon group"})


def table4(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(block: str, label: str, mask_fn) -> None:
        r = {"block": block, "category": label}
        for t in TIER_NAME:
            g = ds[ds["tier"] == t]
            r[TIER_NAME[t]] = int(mask_fn(g).sum())
        rows.append(r)

    for L in range(8, 12):
        add("peptide length", f"{L}-mer", lambda g, L=L: g["length"].astype(int) == L)
    for c in ["virus", "human_self", "parasite", "bacterium", "other_eukaryote", "fungus", "unspecified"]:
        add("source category", c, lambda g, c=c: g["category"] == c)
    for e in ["A", "B", "C"]:
        add("HLA evidence tier", {"A": "A: single allele present, MHC binding assay or T-cell-assay-based restriction",
                                  "B": "B: cited reference or statistical inference (motif / alleles present)",
                                  "C": "C: restriction from MHC-binding prediction only"}[e], lambda g, e=e: g["evidence_tier"] == e)
    add("N2 confidence", "high (evidence tier A/B)", lambda g: g["n2_confidence"] == "high")
    add("N2 confidence", "medium (evidence tier C)", lambda g: g["n2_confidence"] == "medium")
    groups = {"ELISpot": {"ELISPOT"}, "intracellular cytokine staining": {"ICS", "intracellular staining"},
              "ELISA / cytometric bead array": {"ELISA", "cytometric bead array"}, "51Cr release (cytotoxicity)": {"51 chromium"},
              "proliferation (CFSE, 3H-thymidine, BrdU)": {"CFSE", "3H-thymidine", "BrdU"},
              "RNA/DNA detection": {"RNA/DNA detection"},
              "other / unspecified functional readout": {"biological activity", "in vitro assay", "bioassay", "High throughput multiplexed assay"}}
    for label, toks in groups.items():
        add("assay (a record may use several)", label, lambda g, toks=toks: g["methods"].map(lambda m: bool(toks & set(m.split(";")))))
    add("replication", "negative replicated in >=2 records", lambda g: g["negative_replicated"].astype(str) == "True")
    add("replication", "positive with mixed outcomes inside one study", lambda g: g["mixed_within_study"].astype(str) == "True")
    add("track", "pathogen / vaccine track", lambda g: g["in_pathogen_track"].astype(str) == "True")
    return pd.DataFrame(rows)


def table5(path: Path) -> pd.DataFrame:
    s = pd.read_csv(path)
    s["split"] = s["split"].map(SPLIT_LABEL)
    s["positive_fraction"] = (s["positives"] / s["rows"].replace(0, np.nan)).round(3)
    return s.drop(columns=["benchmark"]).rename(columns={"units": "split_units"})


def table6(res: Path, track: str) -> pd.DataFrame | None:
    p = res / f"metrics_overall_{track}.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p)
    m = m[m["tier"] == "N2"]
    rows = []
    for mod in MODEL_ORDER:
        r = {"model": MODEL_LABEL[mod]}
        for sp in SPLIT_LABEL:
            g = m[(m["model"] == mod) & (m["split"] == sp)]
            if len(g):
                g = g.iloc[0]
                r[f"{sp} AUPRC [95% CI]"] = fmt_ci(g["auprc"], g["auprc_lo"], g["auprc_hi"])
                r[f"{sp} AUROC"] = f"{g['auroc']:.3f}"
                r[f"{sp} MCC"] = f"{g['mcc']:.3f}"
        rows.append(r)
    prev = {sp: m[(m["split"] == sp)].iloc[0]["prevalence"] for sp in SPLIT_LABEL if (m["split"] == sp).any()}
    rows.append({"model": "Chance (prevalence)", **{f"{sp} AUPRC [95% CI]": f"{v:.3f}" for sp, v in prev.items()}})
    return pd.DataFrame(rows)


def table7(res: Path, track: str) -> pd.DataFrame | None:
    p, q = res / f"leakage_gap_{track}.csv", res / f"memorization_ratio_{track}.csv"
    if not p.exists():
        return None
    g = pd.read_csv(p)
    g = g[g["tier"] == "N2"]
    mem = pd.read_csv(q) if q.exists() else pd.DataFrame()
    rows = []
    for mod in MODEL_ORDER:
        for sp in ("S1", "S2", "S3"):
            r = g[(g["model"] == mod) & (g["split"] == sp)]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {"model": MODEL_LABEL[mod], "split": SPLIT_LABEL[sp], "AUPRC on S4": f"{r['auprc_S4']:.3f}", "AUPRC on split": f"{r['auprc_split']:.3f}",
                   "AUPRC leakage gap [95% CI]": fmt_ci(r["auprc_gap"], r["auprc_gap_lo"], r["auprc_gap_hi"]),
                   "AUROC gap [95% CI]": fmt_ci(r["auroc_gap"], r["auroc_gap_lo"], r["auroc_gap_hi"]),
                   "lift gap [95% CI]": fmt_ci(r["lift_gap"], r["lift_gap_lo"], r["lift_gap_hi"])}
            if not mem.empty:
                mm = mem[(mem["tier"] == "N2") & (mem["split"] == sp) & (mem["model"] == mod)]
                ok = len(mm) and mm.iloc[0]["lift_nn"] > 0 and mm.iloc[0]["lift_model"] > 0
                row["memorisation ratio [95% CI]"] = (fmt_ci(mm.iloc[0]["memorization_ratio"], mm.iloc[0]["ratio_lo"], mm.iloc[0]["ratio_hi"], 2)
                                                      if ok else "n/a (a lift <= 0)")
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("20_make_tables", args.verbose)
    if args.track != "all":  # track-independent tables 1-5 are written once; track tables go to a sub-folder
        args.outdir = args.outdir / f"{args.track}_track"
    try:
        ds = pd.read_csv(data_dir("processed") / "dataset_all_tiers.csv.gz", dtype=str, keep_default_na=False)
        manifest = json.loads((data_dir("metadata") / "manifest.json").read_text())
        args.outdir.mkdir(parents=True, exist_ok=True)
        tabs = [("table1_dataset_statistics", "Table 1. Dataset statistics.", table1(ds, manifest)),
                ("table2_hla_allele_distribution", "Table 2. HLA class I allele distribution (25 most frequent alleles).", table2(ds)),
                ("table3_source_organism_distribution", "Table 3. Source-organism distribution (20 most frequent taxon groups).", table3(ds)),
                ("table4_composition", "Table 4. Composition of the positive and negative sets.", table4(ds)),
                ("table5_split_statistics", "Table 5. Split statistics for the primary benchmark (P vs N2).", table5(data_dir("metadata") / "split_stats.csv"))]
    except (OSError, KeyError, ValueError) as exc:
        log.error("cannot build tables: %s", exc)
        return 2
    for name, cap, df in tabs if args.track == "all" else []:
        save(df, args.outdir, name, cap)
        log.info("%s: %d rows", name, len(df))
    for name, cap, df in [("table6_model_performance", "Table 6. Grouped five-fold cross-validated baseline performance, P vs N2.", table6(args.results, args.track)),
                          ("table7_leakage_gap", "Table 7. Leakage gap (S4 minus split) and memorisation ratio, P vs N2.", table7(args.results, args.track))]:
        if df is None:
            log.warning("%s skipped: run src/evaluation/12_metrics.py first", name)
            continue
        save(df, args.outdir, name, cap)
        log.info("%s: %d rows", name, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
