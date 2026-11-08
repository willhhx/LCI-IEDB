#!/usr/bin/env python3
"""Unit tests for the peptide similarity engine (run: python -I tests/test_peptide_sim.py)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
from utilities.common import AA20  # noqa: E402
from utilities.peptide_sim import (SimilarityIndex, components_from_pairs, naive_similarity,  # noqa: E402
                                   similarity_bin)


def rand_pep(rng: random.Random) -> str:
    return "".join(rng.choice(AA20) for _ in range(rng.randint(8, 11)))


def main() -> int:
    rng = random.Random(0)
    base = [rand_pep(rng) for _ in range(150)]
    # add near-duplicates / nested windows so that high-similarity cases are exercised
    extra = []
    for p in base[:40]:
        i = rng.randrange(len(p))
        extra.append(p[:i] + rng.choice(AA20) + p[i + 1:])
        extra.append(p[1:] if len(p) > 8 else p)
        extra.append(p + rng.choice(AA20) if len(p) < 11 else p)
    targets = base + extra
    queries = [rand_pep(rng) for _ in range(60)] + targets[:60]
    idx = SimilarityIndex(targets)
    best, arg = idx.max_similarity(queries)
    for q, b, a in zip(queries, best, arg):
        ref = max(naive_similarity(q, t) for t in targets)
        assert abs(b - ref) < 1e-12, (q, b, ref)
        assert abs(naive_similarity(q, targets[a]) - b) < 1e-12
    # exclude_identical must never return an identical string
    b2, a2 = idx.max_similarity(targets[:50], exclude_identical=True)
    assert all(targets[a] != t for t, a in zip(targets[:50], a2))
    # pairs_above equals the brute-force edge list
    qi, ti, sv = idx.pairs_above(targets[:40], 0.8)
    got = {(int(a), int(b)) for a, b in zip(qi, ti)}
    want = {(i, j) for i in range(40) for j in range(len(targets)) if naive_similarity(targets[i], targets[j]) >= 0.8 - 1e-9}
    assert got == want, (len(got), len(want))
    # top-k agrees with brute force (values; ties may be ordered differently)
    tk_i, tk_v = idx.topk(queries[:30], 5)
    for q, vals, ids in zip(queries[:30], tk_v, tk_i):
        ref = sorted((naive_similarity(q, t) for t in targets), reverse=True)[:5]
        assert np.allclose(vals, ref), (q, vals, ref)
        assert all(abs(naive_similarity(q, targets[j]) - v) < 1e-12 for j, v in zip(ids, vals))
    # bins
    assert list(similarity_bin([1.0, 0.99, 0.8, 0.79, 0.6, 0.59, 0.4, 0.39, 0.0])) == \
        ["100%", "80-99%", "80-99%", "60-79%", "60-79%", "40-59%", "40-59%", "<40%", "<40%"]
    assert naive_similarity("ABCDEFGHI", "BCDEFGHIK") == 8 / 9          # shifted window
    lab = components_from_pairs(5, [0, 1], [1, 2])
    assert lab[0] == lab[1] == lab[2] and lab[3] != lab[0] and lab[4] != lab[3]
    print("all peptide_sim tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
