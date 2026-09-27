"""Exchangeable permutation null: merge the two samples, shuffle, and re-cut.

A fixed-reference null fixes a single reference set R and samples the candidate
pool, so the threshold reflects only candidate-subsampling variance — not the
variance of *which* clean set plays the reference. A strong encoder (electra) then
resolves benign clean-split structure and the clean FPR fails to transport
(cal-threshold over-rejects a held-out test split).

The exchangeable null fixes this: each draw merges a clean pool, shuffles, and
recuts into two disjoint halves of sizes (n_ref, n), computing MMD between them.
This marginalizes over the reference half, so the threshold captures the full
clean-vs-clean variance and the clean FPR transports. The attack statistic still
uses the designated fixed reference R (a representative clean set of size n_ref),
which is comparable to the null because R is itself a valid clean draw.
"""

from __future__ import annotations

import numpy as np

from chord.metrics.distribution import rbf_mmd


def exchangeable_null(pool, n_ref, n, *, draws, sigma, block_size, rng):
    """MMD between two disjoint clean halves (sizes n_ref, n) drawn from `pool`."""
    out = np.empty(draws, dtype=np.float64)
    need = n_ref + n
    for d in range(draws):
        idx = rng.choice(len(pool), size=need, replace=False)
        a = pool[idx[:n_ref]]
        b = pool[idx[n_ref:]]
        out[d] = rbf_mmd(a, b, sigma, False, block_size)
    return out


def exchangeable_attack(pool, candidate, n_ref, n, *, draws, sigma, block_size, rng):
    """MMD(clean reference half of size n_ref, candidate sample of size n).

    The reference half is RESAMPLED from the clean pool each draw — matching the
    exchangeable null exactly, so a clean candidate reproduces the null (FPR=alpha,
    transporting) while a corrupted candidate lands far above it. Replaces the
    fixed-R `attack_distances` for the same consistency reason as the null.
    """
    out = np.empty(draws, dtype=np.float64)
    m = len(candidate)
    for d in range(draws):
        a = pool[rng.choice(len(pool), size=n_ref, replace=False)]
        b = candidate[rng.choice(m, size=n, replace=m < n)]
        out[d] = rbf_mmd(a, b, sigma, False, block_size)
    return out
