#!/usr/bin/env python3
"""Construct the N1 (same-source) and N3 (matched hard) negative tiers.

Both tiers are *decoys*: peptides with NO T-cell assay record of any kind in IEDB (so they are disjoint from the
assay-verified N2 negatives and from every positive).  Absence of evidence is not evidence of absence, hence the tiers
are labelled "unverified" in the release.  N1 and N3 are ALTERNATIVE negative sets for the same positives (each is unique
within itself, but a peptide may occur in both); they are not meant to be mixed.

N1 (composition / source control)
    same parent protein as a positive (fallback: same species), same peptide length; the positive's allele is
    inherited.  No association with binding is required.
N3 (hard negatives)
    same length and same parent protein (fallback: same species) as the positive, matched on binding strength and on
    position in the protein.  Two evidence levels (column ``n3_evidence``):
      ligand_evidence : IEDB MHC-ligand records show binding or presentation by the SAME allele (measured IC50/KD <= 500 nM,
                        positive qualitative binding, or eluted); strength = measured log10 nM, else the proxy
      predicted       : no ligand record; strength predicted by the binding proxy for that allele (within the tolerance)
    Ligand-evidence candidates are always preferred.  Candidates must not overlap a positive in the parent sequence and
    must be < ``--near-copy`` similar to every positive peptide.

Outputs: ``data/processed/negatives_N1.csv.gz``, ``negatives_N3.csv.gz``, ``data/metadata/decoy_match_report.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.binding_proxy import BindingProxy  # noqa: E402
from utilities.common import DEFAULT_SEED, data_dir, open_text, setup_logging, update_manifest, write_json  # noqa: E402
from utilities.peptide_sim import SimilarityIndex  # noqa: E402

POOL_COLS = ["peptide", "parent_iri", "parent_name", "start", "end", "species", "taxon_id"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, default=data_dir("processed/pair_labels_annotated.csv.gz"))
    p.add_argument("--ligand-evidence", type=Path, default=data_dir("processed/ligand_pair_evidence.csv.gz"))
    p.add_argument("--epitope-pool", type=Path, default=data_dir("processed/epitope_pool.csv.gz"))
    p.add_argument("--tested", type=Path, default=data_dir("processed/tcell_peptides_any_tested.txt.gz"))
    p.add_argument("--proxy", type=Path, default=data_dir("processed/binding_proxy_models.json"))
    p.add_argument("--outdir", type=Path, default=data_dir("processed"))
    p.add_argument("--n1-per-positive", type=int, default=1)
    p.add_argument("--n3-per-positive", type=int, default=2)
    p.add_argument("--max-delta-bind", type=float, default=0.7, help="max |delta log10 nM| for an N3 match")
    p.add_argument("--near-copy", type=float, default=0.7, help="reject decoys this similar to any positive")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def overlaps(intervals: dict[str, list[tuple[float, float]]], parent: str, start: float, end: float) -> bool:
    if not parent or np.isnan(start) or np.isnan(end):
        return False
    return any(not (end < a or start > b) for a, b in intervals.get(parent, ()))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("07_build_N1_N3_negatives", args.verbose)
    for pth in (args.pairs, args.ligand_evidence, args.epitope_pool, args.tested, args.proxy):
        if not pth.exists():
            log.error("input not found: %s", pth)
            return 2
    rng = np.random.default_rng(args.seed)
    try:
        pairs = pd.read_csv(args.pairs, dtype=str, keep_default_na=False)
        lig = pd.read_csv(args.ligand_evidence, dtype=str, keep_default_na=False)
        pool = pd.read_csv(args.epitope_pool, dtype=str, keep_default_na=False, usecols=POOL_COLS)
        with open_text(args.tested) as fh:
            tested = {ln.strip() for ln in fh if ln.strip()}
        proxy = BindingProxy.load(args.proxy)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 1

    # ---------------- positives ----------------
    pos = pairs[pairs["status"] == "positive"].copy()
    for c in ("start", "end", "mhc_log10_nM"):
        pos[c] = to_num(pos[c])
    pos["bind"] = pos["mhc_log10_nM"]
    pos["bind_src"] = np.where(pos["bind"].notna(), "measured", "")
    for allele, idx in pos.groupby("allele").groups.items():
        miss = idx[pos.loc[idx, "bind"].isna()]
        if len(miss):
            pred = proxy.predict(allele, pos.loc[miss, "peptide"].tolist())
            pos.loc[miss, "bind"] = pred
            pos.loc[miss, "bind_src"] = np.where(np.isnan(pred), "", "proxy")
    pos_peptides = sorted(pos["peptide"].unique())
    pos_set = set(pos_peptides)
    intervals: dict[str, list[tuple[float, float]]] = {}
    for r in pos.itertuples():
        if r.parent_iri and not (np.isnan(r.start) or np.isnan(r.end)):
            intervals.setdefault(r.parent_iri, []).append((r.start, r.end))
    pos_parents = set(pos["parent_iri"]) - {""}
    pos_taxa = set(pos["taxon_id"]) - {""}
    log.info("positives=%d  distinct peptides=%d  with parent=%d  with species=%d", len(pos), len(pos_peptides),
             int((pos["parent_iri"] != "").sum()), int((pos["taxon_id"] != "").sum()))

    # ---------------- candidate pools ----------------
    lig_pep = lig.drop_duplicates("peptide")[["peptide"] + POOL_COLS[1:]]
    pool_all = pd.concat([pool.assign(src="epitope"), lig_pep.assign(src="ligand")], ignore_index=True)
    pool_all = pool_all[~pool_all["peptide"].isin(tested) & ~pool_all["peptide"].isin(pos_set)]
    pool_all = pool_all.drop_duplicates("peptide")
    pool_all = pool_all[pool_all["parent_iri"].isin(pos_parents) | pool_all["taxon_id"].isin(pos_taxa)].copy()
    for c in ("start", "end"):
        pool_all[c] = to_num(pool_all[c])
    log.info("untested same-source/species candidate peptides: %d", len(pool_all))

    # N3 candidates: ligand evidence for an allele present among the positives, binder-class, untested peptide
    pos_alleles = set(pos["allele"])
    l3 = lig[lig["allele"].isin(pos_alleles) & lig["mhc_evidence_class"].isin(["binder", "strong_binder"])
             & ~lig["peptide"].isin(tested) & ~lig["peptide"].isin(pos_set)].copy()
    l3 = l3[l3["parent_iri"].isin(pos_parents) | l3["taxon_id"].isin(pos_taxa)].copy()
    for c in ("start", "end", "mhc_log10_nM"):
        l3[c] = to_num(l3[c])
    l3["bind"] = l3["mhc_log10_nM"]
    l3["bind_src"] = np.where(l3["bind"].notna(), "measured", "")
    for allele, idx in l3.groupby("allele").groups.items():
        miss = idx[l3.loc[idx, "bind"].isna()]
        if len(miss):
            pred = proxy.predict(allele, l3.loc[miss, "peptide"].tolist())
            l3.loc[miss, "bind"] = pred
            l3.loc[miss, "bind_src"] = np.where(np.isnan(pred), "", "proxy")
    l3 = l3.dropna(subset=["bind"]).copy()
    log.info("N3 candidate pairs (binder evidence, same-source/species): %d", len(l3))

    # ---------------- exclusion: near copies / overlap with positives ----------------
    all_cand = sorted(set(pool_all["peptide"]) | set(l3["peptide"]))
    index = SimilarityIndex(pos_peptides)
    log.info("computing similarity of %d candidate peptides to %d positives ...", len(all_cand), len(pos_peptides))
    msim, _ = index.max_similarity(all_cand)
    too_close = {p for p, s in zip(all_cand, msim) if s >= args.near_copy}
    log.info("rejected as near-copies (sim >= %.2f): %d", args.near_copy, len(too_close))

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        keep = ~df["peptide"].isin(too_close)
        ov = np.fromiter((overlaps(intervals, r.parent_iri, r.start, r.end) for r in df.itertuples()), bool, len(df))
        log.info("  rejected by sequence-position overlap: %d", int((ov & keep).sum()))
        return df[keep & ~ov].reset_index(drop=True)

    pool_all = clean(pool_all)
    l3 = clean(l3)

    def build_groups(df: pd.DataFrame, keys_parent: list[str], keys_taxon: list[str]):
        d1 = df[df["parent_iri"] != ""]
        d2 = df[df["taxon_id"] != ""]
        return d1.groupby(keys_parent).indices, d1, d2.groupby(keys_taxon).indices, d2

    # ---------------- N3 matching ----------------
    l3["length"] = l3["peptide"].str.len()
    g3p, d3p, g3t, d3t = build_groups(l3, ["allele", "length", "parent_iri"], ["allele", "length", "taxon_id"])
    pool_all["length"] = pool_all["peptide"].str.len()
    gpp, dpp, gpt, dpt = build_groups(pool_all, ["length", "parent_iri"], ["length", "taxon_id"])
    pred_cache: dict[tuple, np.ndarray] = {}
    used_pep: set[str] = set()
    order = rng.permutation(len(pos))
    n3_rows, report = [], []

    def pick(cand: pd.DataFrame, r: pd.Series, need: int, level: str, evidence: str, out: list[dict]) -> int:
        cand = cand[~cand["peptide"].isin(used_pep)]
        if cand.empty or need <= 0:
            return 0
        dbind = np.abs(cand["bind"].to_numpy() - r["bind"])
        known = cand["start"].notna().to_numpy() & (not np.isnan(r["start"]))
        dstart = np.where(known, np.abs(cand["start"].to_numpy() - r["start"]), np.nan)
        cost = dbind + np.where(known, dstart / 200.0, 0.5)
        got = 0
        for j in np.argsort(cost):
            if got >= need or dbind[j] > args.max_delta_bind:
                break
            c = cand.iloc[j]
            if c["peptide"] in used_pep:
                continue
            used_pep.add(c["peptide"])
            got += 1
            out.append({"peptide": c["peptide"], "allele": r["allele"], "tier": "N3", "matched_pair_id": r["pair_id"],
                        "match_level": level, "n3_evidence": evidence, "parent_iri": c["parent_iri"],
                        "parent_name": c["parent_name"], "start": c["start"], "end": c["end"], "species": c["species"],
                        "taxon_id": c["taxon_id"], "binding_log10nM": c["bind"], "binding_source": c["bind_src"],
                        "delta_binding": dbind[j], "delta_start": dstart[j],
                        "mhc_evidence_class": c.get("mhc_evidence_class", "predicted_binder")})
        return got

    def predicted_pool(allele: str, level: str, key: tuple, frame: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
        ck = (allele, level, key)
        if ck not in pred_cache:
            sub = frame.iloc[idx]
            if len(sub) > 20000:
                sub = sub.iloc[np.random.default_rng(args.seed).permutation(len(sub))[:20000]]
            pred = proxy.predict(allele, sub["peptide"].tolist())
            sub = sub.assign(bind=pred, bind_src="proxy", mhc_evidence_class="predicted_binder")
            pred_cache[ck] = sub.dropna(subset=["bind"]).reset_index(drop=True)
        return pred_cache[ck]

    for k in order:
        r = pos.iloc[k]
        row = {"pair_id": r["pair_id"], "peptide": r["peptide"], "allele": r["allele"], "n3": 0, "n3_level": "", "n3_reason": "",
               "n1": 0, "n1_level": "", "n1_reason": ""}
        if np.isnan(r["bind"]):
            row["n3_reason"] = "no_binding_estimate"
        elif r["parent_iri"] == "" and r["taxon_id"] == "":
            row["n3_reason"] = "no_source_annotation"
        else:
            L = len(r["peptide"])
            got = 0
            for level, ident in (("protein", r["parent_iri"]), ("species", r["taxon_id"])):
                if got >= args.n3_per_positive or ident == "":
                    continue
                gA, fA = (g3p, d3p) if level == "protein" else (g3t, d3t)
                if (r["allele"], L, ident) in gA:
                    got += pick(fA.iloc[gA[(r["allele"], L, ident)]], r, args.n3_per_positive - got, level, "ligand_evidence", n3_rows)
                if got < args.n3_per_positive and r["allele"] in proxy.models:
                    gB, fB = (gpp, dpp) if level == "protein" else (gpt, dpt)
                    if (L, ident) in gB:
                        cand = predicted_pool(r["allele"], level, (L, ident), fB, gB[(L, ident)])
                        got += pick(cand, r, args.n3_per_positive - got, level, "predicted", n3_rows)
                if got and not row["n3_level"]:
                    row["n3_level"] = level
            row["n3"] = got
            if got == 0:
                row["n3_reason"] = "no_candidate_within_binding_tolerance"
        report.append(row)
    n3 = pd.DataFrame(n3_rows)

    # ---------------- N1 matching (independent of N3: the tiers are ALTERNATIVE negative sets) ----------------
    used_pep1: set[str] = set()
    g1p, d1p, g1t, d1t = gpp, dpp, gpt, dpt
    rep = {r["pair_id"]: r for r in report}
    n1_rows = []
    for k in order:
        r = pos.iloc[k]
        row = rep[r["pair_id"]]
        if r["parent_iri"] == "" and r["taxon_id"] == "":
            row["n1_reason"] = "no_source_annotation"
            continue
        L = len(r["peptide"])
        got = 0
        for level, groups, frame, key in (("protein", g1p, d1p, (L, r["parent_iri"])), ("species", g1t, d1t, (L, r["taxon_id"]))):
            if got >= args.n1_per_positive or key not in groups or key[1] == "":
                continue
            cand = frame.iloc[groups[key]]
            cand = cand[~cand["peptide"].isin(used_pep1)]
            if cand.empty:
                continue
            take = cand.iloc[rng.permutation(len(cand))[: args.n1_per_positive - got]]
            for c in take.itertuples():
                used_pep1.add(c.peptide)
                got += 1
                n1_rows.append({"peptide": c.peptide, "allele": r["allele"], "tier": "N1", "matched_pair_id": r["pair_id"],
                                "match_level": level, "parent_iri": c.parent_iri, "parent_name": c.parent_name,
                                "start": c.start, "end": c.end, "species": c.species, "taxon_id": c.taxon_id})
            row["n1_level"] = level if not row["n1_level"] else row["n1_level"]
        row["n1"] = got
        if got == 0:
            row["n1_reason"] = "no_candidate"
    n1 = pd.DataFrame(n1_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    n1.to_csv(args.outdir / "negatives_N1.csv.gz", index=False, compression="gzip")
    n3.to_csv(args.outdir / "negatives_N3.csv.gz", index=False, compression="gzip")
    rep_df = pd.DataFrame(report).sort_values("pair_id")
    rep_df.to_csv(data_dir("metadata/decoy_match_report.csv"), index=False)
    summary = {"positives": len(pos), "n1_rows": len(n1), "n3_rows": len(n3),
               "positives_with_n1": int((rep_df["n1"] > 0).sum()), "positives_with_n3": int((rep_df["n3"] > 0).sum()),
               "n1_protein_level": int((n1["match_level"] == "protein").sum()) if len(n1) else 0,
               "n3_protein_level": int((n3["match_level"] == "protein").sum()) if len(n3) else 0,
               "n3_measured_binding": int((n3["binding_source"] == "measured").sum()) if len(n3) else 0,
               "n3_ligand_evidence": int((n3["n3_evidence"] == "ligand_evidence").sum()) if len(n3) else 0,
               "near_copy_threshold": args.near_copy, "max_delta_binding": args.max_delta_bind, "seed": args.seed}
    write_json(data_dir("metadata/decoy_summary.json"), summary)
    update_manifest("07_build_N1_N3_negatives", summary)
    log.info("summary: %s", summary)
    log.info("N3 failure reasons: %s", rep_df.loc[rep_df["n3"] == 0, "n3_reason"].value_counts().to_dict())
    log.info("N1 failure reasons: %s", rep_df.loc[rep_df["n1"] == 0, "n1_reason"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
