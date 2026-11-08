#!/usr/bin/env python3
"""Render Figures 1-8 (PNG + SVG) from the released tables and evaluation results.

Fig 1 pipeline workflow                          Fig 5 performance by split
Fig 2 leakage vectors removed by each split       Fig 6 performance by train-test similarity bin
Fig 3 dataset composition                         Fig 7 negative-set comparison
Fig 4 train-test similarity distributions         Fig 8 leakage gap and memorisation ratio
Outputs: ``data/documentation/figures/figN_*.{png,svg}``.  Figures 2 and 4-8 need ``11_cv_baselines.py``, ``12_metrics.py`` and
``13_leakage_exposure.py`` results; a missing input skips that figure with a warning.
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
from visualization import style as S  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

KEY_MODELS = ["B3_organism", "B5_study_prior", "B4_nearest_neighbour", "LR", "RF", "GBM"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=data_dir("documentation/figures"))
    p.add_argument("--results", type=Path, default=data_dir("metadata/results"))
    p.add_argument("--oof-dir", type=Path, default=data_dir("processed/oof"))
    p.add_argument("--track", default="all", choices=["all", "pathogen"])
    p.add_argument("--figures", nargs="+", type=int, default=list(range(1, 9)), choices=range(1, 9))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


TRACK_NOTE = ""  # set in main(): label drawn on figures of the pathogen/vaccine track


def save(fig, outdir: Path, name: str) -> None:
    if TRACK_NOTE:
        fig.text(0.005, 1.0, TRACK_NOTE, fontsize=8, color=S.INK2, fontweight="bold", va="bottom")
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def need(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------- Figure 1
def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.012", fc="#f6f7fa", ec=color, lw=1.6))
    ax.text(x + 0.012, y + h - 0.02, title, fontsize=8.6, fontweight="bold", color=S.INK, va="top")
    ax.text(x + 0.012, y + h - 0.062, body, fontsize=7, color=S.INK2, va="top", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.2, shrinkA=0, shrinkB=0))


def fig1(a: argparse.Namespace) -> None:
    man = json.loads((data_dir("metadata") / "manifest.json").read_text())
    f2, f3 = man["02_extract_tcell_records"]["funnel"], man["03_extract_mhc_ligand"]["funnel"]
    fl = json.loads((data_dir("metadata") / "funnel_labels.json").read_text())
    ds = man["08_assemble_dataset"]["tiers"]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    c = S.PALETTE
    box(ax, 0.00, 0.60, 0.22, 0.34, "1  IEDB T-cell assays", f"{f2['1_all_tcell_rows']:,} rows\nhuman host, linear, MHC-I\nunmodified, 8-11 aa canonical\nHLA allele resolved\n-> {f2['8_resolved_hla_class_I_allele']:,} records", c[0])
    box(ax, 0.00, 0.14, 0.22, 0.34, "1b  IEDB ligand + epitope", f"{f3['1_all_mhc_ligand_rows']:,} ligand rows\n-> {f3['6_resolved_hla_class_I_allele']:,} human HLA-I\nrecords (binding evidence only)\nepitope + antigen tables:\nparent protein, taxon", c[0])
    box(ax, 0.27, 0.60, 0.22, 0.34, "2  Pair labels", f"peptide x HLA allele units\n{fl['pairs_positive']:,} positive (functional)\n{fl['pairs_negative']:,} negative (assay-verified)\n{fl['pairs_conflict']:,} cross-study conflicts\n   -> conflicts table\n{fl['pairs_multimer_only_positive']:,} multimer-only (suppl.)", c[1])
    box(ax, 0.27, 0.14, 0.22, 0.34, "3  Decoy negatives", f"N1 same-protein decoys: {ds['N1']:,}\nN3 length/binding/region-\n  matched decoys: {ds['N3']:,}\n(untested peptides; optional\n alternative negative sets)", c[3])
    box(ax, 0.54, 0.60, 0.22, 0.34, "4  Dataset (v1.0)", f"{sum(ds.values()):,} rows, 17,278 peptides\nP {ds['P']:,}   N2 {ds['N2']:,}\nN1 {ds['N1']:,}   N3 {ds['N3']:,}\nevidence tier, PMID, assay,\nconfidence retained", c[2])
    box(ax, 0.54, 0.14, 0.22, 0.34, "5  Leakage groups", f"connected components of\nnear-duplicate peptides\n(similarity >= 0.8), same parent\nprotein, same antigen family;\n{_gs()[0]:,} groups, largest {_gs()[1]:.1f} %", c[6])
    box(ax, 0.80, 0.60, 0.19, 0.34, "6  Splits", "S1 antigen-group (primary)\nS2 held-out HLA\nS3 held-out taxon\nS4 random reference\n70/10/20 + 5-fold CV", c[5])
    box(ax, 0.80, 0.14, 0.19, 0.34, "7  Evaluation", "9 baselines, 4 negative tiers\nAUPRC, AUROC, MCC,\nPPV@K, leakage gap, similarity bins,\ncluster-bootstrap CIs", c[7])
    for xy in [(0.22, 0.77, 0.27, 0.77), (0.22, 0.31, 0.27, 0.31), (0.49, 0.77, 0.54, 0.77), (0.49, 0.31, 0.54, 0.31),
               (0.38, 0.60, 0.38, 0.48), (0.65, 0.60, 0.65, 0.48), (0.76, 0.77, 0.80, 0.77), (0.76, 0.31, 0.80, 0.31), (0.895, 0.60, 0.895, 0.48)]:
        arrow(ax, *xy)
    ax.text(0.0, 0.03, "No shuffled or random negatives are released; random peptides (N0) are generated only at analysis time to quantify inflation.",
            fontsize=7.6, color=S.INK2, style="italic")
    save(fig, a.outdir, "fig1_workflow")


# ---------------------------------------------------------------- Figure 2
def fig2(a: argparse.Namespace) -> None:
    ex = need(a.results / f"leakage_exposure_{a.track}.csv")
    vec = [("identical_peptide", "Identical peptide in training"), ("near_duplicate", "Near-duplicate peptide (similarity >= 0.8)"),
           ("same_leakage_group", "Same leakage group (antigen family)"), ("same_study", "Same IEDB reference"),
           ("same_taxon_group", "Same source taxon group"), ("same_allele", "Same HLA allele")]
    piv = ex.pivot(index="vector", columns="split", values="fraction_exposed")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.grid(False)
    mat = np.array([[piv.loc[v, s] for s in S.SPLIT_LABEL] for v, _ in vec])
    ax.imshow(mat, cmap=plt.cm.Blues, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=9, color="white" if mat[i, j] > 0.55 else S.INK, fontweight="bold")
    ax.set_xticks(range(4), [S.SPLIT_LABEL[s] for s in S.SPLIT_LABEL]); ax.set_yticks(range(len(vec)), [l for _, l in vec])
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Share of held-out rows exposed to each leakage vector (5-fold CV, P vs N2)", pad=34, loc="left")
    save(fig, a.outdir, "fig2_leakage_sources")


# ---------------------------------------------------------------- Figure 3
def fig3(a: argparse.Namespace) -> None:
    ds = pd.read_csv(data_dir("processed") / "dataset_all_tiers.csv.gz", dtype=str, keep_default_na=False)
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1.25, 0.85]})
    tiers = ["P", "N2", "N1", "N3"]
    tcol = {"P": S.INK, "N2": "#9aa3b5"}  # neutral pair so tier colours never collide with the organism hues in panel a
    # a: source category by tier (share)
    ax = axs[0]
    cats = ["virus", "human_self", "parasite", "bacterium", "unspecified"]
    cc = {"virus": S.PALETTE[0], "human_self": S.PALETTE[1], "parasite": S.PALETTE[2], "bacterium": S.PALETTE[3], "unspecified": "#b4bac6"}
    left = np.zeros(len(tiers))
    for c in cats:
        v = np.array([(ds[ds.tier == t]["category"].isin([c] if c != "unspecified" else ["unspecified", "other", "fungus", "other_eukaryote"])).mean() for t in tiers])
        ax.barh(range(len(tiers)), v, left=left, color=cc[c], edgecolor="white", linewidth=1.5, label=c.replace("_", " "), height=0.62)
        for i, (l, w) in enumerate(zip(left, v)):
            if w > 0.09:
                ax.text(l + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7.5, color="white" if c in ("virus", "human_self", "parasite") else S.INK)
        left += v
    ax.set_yticks(range(len(tiers)), [f"{t}\n(n={int((ds.tier == t).sum()):,})" for t in tiers]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5, handlelength=1, columnspacing=1)
    ax.set_title("Source organism"); S.panel_label(ax, "a")
    # b: top alleles, P vs N2
    ax = axs[1]
    d = ds[ds.tier.isin(["P", "N2"])]
    top = d["allele"].value_counts().head(12).index[::-1]
    y = np.arange(len(top))
    for off, t in ((-0.19, "P"), (0.19, "N2")):
        n = d[d.tier == t]["allele"].value_counts().reindex(top).fillna(0)
        ax.barh(y + off, n.values, height=0.36, color=tcol[t], label="Positive" if t == "P" else "N2 negative")
    ax.set_yticks(y, [s.replace("HLA-", "") for s in top]); ax.set_xlabel("rows"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right"); ax.set_title(f"Top alleles ({d['allele'].nunique()} alleles in P/N2)"); S.panel_label(ax, "b")
    # c: length
    ax = axs[2]
    x = np.arange(8, 12)
    for off, t in ((-0.2, "P"), (0.2, "N2")):
        g = d[d.tier == t]["length"].astype(int)
        ax.bar(x + off, [(g == L).mean() for L in x], width=0.4, color=tcol[t], label="Positive" if t == "P" else "N2 negative")
    ax.set_xticks(x, [f"{L}-mer" for L in x]); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.grid(axis="x", visible=False)
    ax.set_title("Peptide length"); S.panel_label(ax, "c")
    fig.tight_layout()
    save(fig, a.outdir, "fig3_dataset_composition")


# ---------------------------------------------------------------- Figures 4-8
def load_oof(a: argparse.Namespace, tier: str, sp: str) -> pd.DataFrame:
    d = pd.read_csv(a.oof_dir / f"oof_{a.track}_{tier}_{sp}.csv.gz", dtype={"row_id": str})
    return d[d["maxsim"].notna()].reset_index(drop=True)


def fig4(a: argparse.Namespace) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    frac = {}
    for sp in S.SPLIT_LABEL:
        d = load_oof(a, "N2", sp)
        x = np.sort(d["maxsim"].to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=S.SPLIT_COLOR[sp], lw=2, label=S.SPLIT_LABEL[sp])
        frac[sp] = d["bin"].value_counts(normalize=True).reindex(S.BIN_ORDER).fillna(0)
    for v in (0.4, 0.6, 0.8):
        ax.axvline(v, color=S.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xlabel("maximum similarity to any training peptide"); ax.set_ylabel("cumulative share of test rows"); ax.set_xlim(0.2, 1.0)
    ax.legend(loc="upper left"); ax.set_title("Train-test similarity (P vs N2, 5-fold CV)"); S.panel_label(ax, "a")
    ax = axs[1]
    left = np.zeros(4)
    shades = ["#dce9f8", "#a9c9ee", "#6aa3e0", "#2a78d6", "#123f7a"]
    for b, col in zip(S.BIN_ORDER, shades):
        v = np.array([frac[sp][b] for sp in S.SPLIT_LABEL])
        ax.barh(range(4), v, left=left, color=col, edgecolor="white", linewidth=1.5, height=0.62, label=b)
        for i, (l, w) in enumerate(zip(left, v)):
            if w > 0.07:
                ax.text(l + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7.5, color="white" if col in shades[2:] else S.INK)
        left += v
    ax.set_yticks(range(4), [S.SPLIT_LABEL[s] for s in S.SPLIT_LABEL]); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(title="similarity bin", ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7.5, handlelength=1, columnspacing=0.9, title_fontsize=7.5)
    ax.set_title("Share of test rows per bin"); S.panel_label(ax, "b")
    fig.tight_layout()
    save(fig, a.outdir, "fig4_similarity_distributions")


def dotplot(ax, df: pd.DataFrame, models: list[str], val: str, lo: str | None, hi: str | None) -> None:
    for i, m in enumerate(models):
        r = df[df["model"] == m]
        if r.empty:
            continue
        r = r.iloc[0]
        if lo and np.isfinite(r[lo]):
            ax.plot([r[lo], r[hi]], [i, i], color=S.MODEL_COLOR[m], lw=2, solid_capstyle="round")
        ax.scatter(r[val], i, s=44, color=S.MODEL_COLOR[m], zorder=3, edgecolor="white", linewidth=1.2)
    ax.set_yticks(range(len(models)), [S.MODEL_LABEL[m] for m in models]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)


def fig5(a: argparse.Namespace) -> None:
    m = need(a.results / f"metrics_overall_{a.track}.csv")
    m = m[m["tier"] == "N2"]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.7), sharey=True, sharex=True)
    for ax, sp in zip(axs, S.SPLIT_LABEL):
        g = m[m["split"] == sp]
        dotplot(ax, g, S.MODEL_ORDER, "auprc", "auprc_lo", "auprc_hi")
        ax.axvline(g["prevalence"].iloc[0], color=S.MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(S.SPLIT_LABEL[sp]); ax.set_xlabel("AUPRC (95% cluster-bootstrap CI)")
    fig.text(0.5, -0.03, "Dashed line = share of positives (chance-level AUPRC). Intervals resample whole leakage groups.", ha="center", fontsize=7.5, color=S.INK2)
    fig.tight_layout()
    save(fig, a.outdir, "fig5_performance_by_split")


def fig6(a: argparse.Namespace) -> None:
    b = need(a.results / f"metrics_by_bin_{a.track}.csv")
    b = b[b["tier"] == "N2"]
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    x = np.arange(len(S.BIN_ORDER))
    for ax, sp, lab in zip(axs, ("S4", "S1"), "ab"):
        for mod in KEY_MODELS:
            g = b[(b["split"] == sp) & (b["model"] == mod)].set_index("bin").reindex(S.BIN_ORDER)
            ok = g["auroc"].notna().to_numpy()
            if ok.sum() == 0:
                continue
            ax.plot(x[ok], g["auroc"].to_numpy()[ok], marker="o", ms=6, lw=2, color=S.MODEL_COLOR[mod], markeredgecolor="white", markeredgewidth=1.2, label=S.MODEL_LABEL[mod])
        ax.axhline(0.5, color=S.MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(x, S.BIN_ORDER); ax.set_xlabel("maximum similarity to training peptides"); ax.set_ylim(0.3, 1.0)
        ax.set_title(f"{S.SPLIT_LABEL[sp]}"); S.panel_label(ax, lab)
    axs[0].set_ylabel("AUROC within bin")
    axs[0].legend(loc="lower right", ncol=2, fontsize=7.5)
    fig.text(0.5, -0.03, "Bins with fewer than 40 rows or fewer than 10 rows of either class are omitted.", ha="center", fontsize=7.5, color=S.INK2)
    fig.tight_layout()
    save(fig, a.outdir, "fig6_performance_by_similarity_bin")


def fig7(a: argparse.Namespace) -> None:
    m = need(a.results / f"metrics_overall_{a.track}.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)
    tiers = [t for t in S.TIER_LABEL if (m["tier"] == t).any()]
    for ax, sp, lab in zip(axs, ("S1", "S4"), "ab"):
        for j, mod in enumerate(["B3_organism", "B4_nearest_neighbour", "LR", "GBM"]):
            for i, t in enumerate(tiers):
                r = m[(m["tier"] == t) & (m["split"] == sp) & (m["model"] == mod)]
                if r.empty:
                    continue
                r = r.iloc[0]
                yy = i + (j - 1.5) * 0.17
                ax.plot([r["lift_lo"], r["lift_hi"]], [yy, yy], color=S.MODEL_COLOR[mod], lw=2, solid_capstyle="round")
                ax.scatter(r["lift"], yy, s=38, color=S.MODEL_COLOR[mod], zorder=3, edgecolor="white", linewidth=1.1, label=S.MODEL_LABEL[mod] if i == 0 else None)
        ax.set_yticks(range(len(tiers)), [S.TIER_LABEL[t] for t in tiers]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.axvline(0, color=S.MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_xlabel("AUPRC lift over prevalence (0 = chance, 1 = perfect)"); ax.set_title(S.SPLIT_LABEL[sp]); S.panel_label(ax, lab)
    axs[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, a.outdir, "fig7_negative_set_comparison")


def fig8(a: argparse.Namespace) -> None:
    g = need(a.results / f"leakage_gap_{a.track}.csv")
    g = g[g["tier"] == "N2"]
    mem = need(a.results / f"memorization_ratio_{a.track}.csv")
    mem = mem[mem["tier"] == "N2"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axs[0]
    models = ["B3_organism", "B4_nearest_neighbour", "B5_study_prior", "LR", "RF", "GBM"]
    sps = ["S1", "S2", "S3"]
    for j, sp in enumerate(sps):
        for i, mod in enumerate(models):
            r = g[(g["model"] == mod) & (g["split"] == sp)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = i + (j - 1) * 0.24
            ax.plot([r["auprc_gap_lo"], r["auprc_gap_hi"]], [yy, yy], color=S.SPLIT_COLOR[sp], lw=2, solid_capstyle="round")
            ax.scatter(r["auprc_gap"], yy, s=38, color=S.SPLIT_COLOR[sp], zorder=3, edgecolor="white", linewidth=1.1, label=S.SPLIT_LABEL[sp] if i == 0 else None)
    ax.axvline(0, color=S.INK2, lw=1)
    ax.set_yticks(range(len(models)), [S.MODEL_LABEL[m] for m in models]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("leakage gap = AUPRC(S4) - AUPRC(split)  (95% CI)"); ax.legend(loc="lower right"); ax.set_title("Apparent performance removed by each split"); S.panel_label(ax, "a")
    ax = axs[1]
    for j, sp in enumerate(["S4", "S1", "S2"]):
        for i, mod in enumerate(["LR", "RF", "GBM"]):
            r = mem[(mem["model"] == mod) & (mem["split"] == sp)]
            if r.empty:
                continue
            r = r.iloc[0]
            yy = i + (j - 1) * 0.24
            ax.plot([r["ratio_lo"], r["ratio_hi"]], [yy, yy], color=S.SPLIT_COLOR[sp], lw=2, solid_capstyle="round")
            ax.scatter(r["memorization_ratio"], yy, s=38, color=S.SPLIT_COLOR[sp], zorder=3, edgecolor="white", linewidth=1.1, label=S.SPLIT_LABEL[sp] if i == 0 else None)
    ax.axvline(1, color=S.INK2, lw=1)
    ax.set_yticks(range(3), [S.MODEL_LABEL[m] for m in ["LR", "RF", "GBM"]]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.7)
    ax.set_xlabel("memorisation ratio = lift(nearest neighbour) / lift(model)"); ax.set_title("Is a model beating a lookup table?"); S.panel_label(ax, "b")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=7.5)
    ax.text(0.99, 0.02, "S3 omitted: nearest-neighbour lift <= 0", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=S.INK2)
    fig.tight_layout()
    save(fig, a.outdir, "fig8_leakage_gap")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8}


def _gs():
    import json as _j
    g=_j.load(open(data_dir()/"metadata"/"group_stats.json"))
    return g["groups"], 100*g["largest_group_fraction"]


def main(argv: list[str] | None = None) -> int:
    global TRACK_NOTE
    a = parse_args(argv)
    TRACK_NOTE = "" if a.track == "all" else "Pathogen / vaccine track (non-human source taxa)"
    log = setup_logging("21_make_figures", a.verbose)
    S.apply()
    figures = a.figures
    if a.track != "all":  # figures 1 and 3 do not depend on the track; the others go to a sub-folder
        a.outdir = a.outdir / f"{a.track}_track"
        figures = [n for n in a.figures if n not in (1, 3)]
    a.outdir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for n in figures:
        try:
            FIGS[n](a)
            log.info("figure %d written", n)
        except FileNotFoundError as exc:
            log.warning("figure %d skipped, missing input: %s", n, exc)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
