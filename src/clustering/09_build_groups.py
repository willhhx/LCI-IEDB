#!/usr/bin/env python3
"""Build leakage groups: the units that must never straddle train and test.

Nodes are unique peptide strings and antigen keys.  Edges (union-find / connected components):
  1. near-duplicate peptides            sim >= ``--near-dup`` (default 0.8; single-substitution variants, nested windows)
  2. peptide <-> parent protein          same IEDB parent-protein IRI (UniProt where available)
  3. peptide <-> antigen name key        same taxon group + same normalised protein name (links strain variants and
                                         homologous proteins of the same virus family; an offline stand-in for 30 % identity
                                         clustering of parent proteins)
  4. (optional) external antigen clusters, e.g. from MMseqs2 at 30 % identity: ``--external-clusters map.tsv`` with columns
     ``parent_iri<TAB>cluster_id``

Because identical strings are one node, identical peptides (also across alleles) always share a group.

Outputs: ``data/processed/leakage_groups.csv.gz`` (row_id, peptide, group_id, peptide_component), ``data/metadata/group_stats.json``,
``data/metadata/parent_accessions.txt`` (UniProt accessions to feed an external clustering run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utilities.common import data_dir, setup_logging, uniprot_from_iri, update_manifest, write_json  # noqa: E402
from utilities.peptide_sim import SimilarityIndex, components_from_pairs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=data_dir("processed/dataset_all_tiers.csv.gz"))
    p.add_argument("--near-dup", type=float, default=0.8)
    p.add_argument("--external-clusters", type=Path, default=None)
    p.add_argument("--no-antigen-name-links", action="store_true", help="disable edge type 3 (ablation)")
    p.add_argument("--outdir", type=Path, default=data_dir("processed"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("09_build_groups", args.verbose)
    if not args.dataset.exists():
        log.error("input not found: %s", args.dataset)
        return 2
    try:
        ds = pd.read_csv(args.dataset, dtype=str, keep_default_na=False)
        ext = None
        if args.external_clusters:
            ext = pd.read_csv(args.external_clusters, sep="\t", dtype=str, header=None, names=["parent_iri", "cluster_id"])
    except (OSError, pd.errors.ParserError) as exc:
        log.error("%s", exc)
        return 1

    peps = sorted(ds["peptide"].unique())
    pid = {p: i for i, p in enumerate(peps)}
    n = len(peps)
    log.info("rows=%d  unique peptides=%d", len(ds), n)
    index = SimilarityIndex(peps)
    qi, ti, _ = index.pairs_above(peps, args.near_dup)
    m = qi < ti
    ei, ej = qi[m], ti[m]
    log.info("near-duplicate peptide edges (sim >= %.2f): %d", args.near_dup, len(ei))

    node_of: dict[str, int] = {}
    def node(key: str) -> int:
        if key not in node_of:
            node_of[key] = n + len(node_of)
        return node_of[key]

    ai, aj = [], []
    n_parent = n_name = n_ext = 0
    ext_map = dict(zip(ext["parent_iri"], ext["cluster_id"])) if ext is not None else {}
    for pep, par, nk in zip(ds["peptide"], ds["parent_iri"], ds["antigen_name_key"]):
        if par:
            ai.append(pid[pep]); aj.append(node("P:" + par)); n_parent += 1
            if par in ext_map:
                ai.append(pid[pep]); aj.append(node("X:" + ext_map[par])); n_ext += 1
        if nk and not args.no_antigen_name_links:
            ai.append(pid[pep]); aj.append(node("N:" + nk)); n_name += 1
    # NOTE: decoys are deliberately NOT linked to the positive they were matched to.  Decoys matched at species level come
    # from a different protein; linking them would chain thousands of unrelated proteins into one giant group.  Protein-level
    # decoys share the positive's parent protein and are therefore grouped with it through edge type 2 anyway.
    total = n + len(node_of)
    I = np.concatenate([ei, np.asarray(ai, int)])
    J = np.concatenate([ej, np.asarray(aj, int)])
    comp = components_from_pairs(total, I, J)
    pep_comp = comp[:n]
    only_pep = components_from_pairs(n, ei, ej)  # peptide-only components (edge type 1) for reporting

    ds["peptide_component"] = [int(only_pep[pid[p]]) for p in ds["peptide"]]
    ds["group_id"] = [int(pep_comp[pid[p]]) for p in ds["peptide"]]
    ds[["row_id", "peptide", "group_id", "peptide_component"]].to_csv(args.outdir / "leakage_groups.csv.gz", index=False, compression="gzip")

    sizes = ds.groupby("group_id").size().sort_values(ascending=False)
    stats = {"rows": len(ds), "unique_peptides": n, "near_dup_threshold": args.near_dup,
             "near_dup_edges": int(len(ei)), "parent_edges": n_parent, "antigen_name_edges": n_name,
             "external_cluster_edges": n_ext, "groups": int(sizes.size),
             "largest_group_rows": int(sizes.iloc[0]), "largest_group_fraction": float(sizes.iloc[0] / len(ds)),
             "top5_group_sizes": sizes.head(5).tolist(), "singleton_groups": int((sizes == 1).sum()),
             "peptide_only_components": int(len(set(only_pep)))}
    write_json(data_dir("metadata/group_stats.json"), stats)
    accessions = sorted({a for a in (uniprot_from_iri(x) for x in ds["parent_iri"]) if a})
    (data_dir("metadata/parent_accessions.txt")).write_text("\n".join(accessions) + "\n")
    update_manifest("09_build_groups", stats)
    log.info("groups=%d  largest=%d rows (%.1f%%)  top5=%s", stats["groups"], stats["largest_group_rows"],
             100 * stats["largest_group_fraction"], stats["top5_group_sizes"])
    big = ds[ds["group_id"] == sizes.index[0]]
    log.info("largest group composition: %s", big["taxon_group"].value_counts().head(6).to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
