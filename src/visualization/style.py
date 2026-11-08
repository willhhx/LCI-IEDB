"""Shared figure style: validated categorical palette (light surface), fixed hue order, recessive grid, direct labels."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated with the dataviz validator (light mode): lightness band, chroma floor and CVD separation pass;
# aqua/yellow/magenta have low contrast on white, so every series is also labelled directly or in a legend.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#1d2433", "#4a5368", "#7a8396", "#e3e6ec", "#ffffff"

MODEL_ORDER = ["B1_length", "B2_composition", "B3_organism", "B3b_allele", "B4_nearest_neighbour", "B5_study_prior", "LR", "RF", "GBM"]
MODEL_LABEL = {"B1_length": "Length", "B2_composition": "Composition", "B3_organism": "Source organism", "B3b_allele": "HLA allele prior",
               "B4_nearest_neighbour": "Nearest neighbour", "B5_study_prior": "Study prior", "LR": "Logistic regression",
               "RF": "Random forest", "GBM": "Gradient boosting"}
# fixed hue per entity (never re-assigned when a subset is plotted)
MODEL_COLOR = {"B1_length": "#7a8396", "B2_composition": "#b4bac6", "B3_organism": PALETTE[3], "B3b_allele": PALETTE[4],
               "B4_nearest_neighbour": PALETTE[1], "B5_study_prior": PALETTE[7], "LR": PALETTE[0], "RF": PALETTE[2], "GBM": PALETTE[6]}
SPLIT_LABEL = {"S1": "S1 antigen-group", "S2": "S2 held-out HLA", "S3": "S3 held-out taxon", "S4": "S4 random"}
SPLIT_COLOR = {"S1": PALETTE[0], "S2": PALETTE[2], "S3": PALETTE[6], "S4": "#7a8396"}
TIER_LABEL = {"N2": "N2 assay-verified", "N1": "N1 same-protein decoy", "N3": "N3 matched hard decoy", "N0": "N0 random (analysis only)"}
TIER_COLOR = {"N2": PALETTE[0], "N1": PALETTE[3], "N3": PALETTE[1], "N0": "#7a8396"}
BIN_ORDER = ["<40%", "40-59%", "60-79%", "80-99%", "100%"]


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
        "legend.fontsize": 8, "svg.fonttype": "none", "figure.dpi": 100, "savefig.dpi": 300,
    })


def panel_label(ax, letter: str) -> None:
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom", ha="right")
