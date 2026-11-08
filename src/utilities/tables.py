"""Small pandas helpers shared by several pipeline scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def modal(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    """Most frequent non-empty value of ``col`` per key group ('' if none); ties broken alphabetically."""
    d = df.loc[df[col].ne(""), keys + [col]]
    cnt = d.groupby(keys + [col], sort=False).size().rename("n").reset_index()
    cnt = cnt.sort_values(keys + ["n", col], ascending=[True] * len(keys) + [False, True])
    return cnt.drop_duplicates(keys).set_index(keys)[col]


def join_unique(s: pd.Series) -> str:
    """Sorted ';'-joined unique non-empty values."""
    return ";".join(sorted({x for x in s if x != ""}))


def to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (NaN -> empty cell)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"
