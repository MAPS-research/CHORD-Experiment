"""Official MAUVE / FI-KL on *precomputed* features (no fork required).

We feed our own encoder features (ELECTRA-mean/-last, gpt2-large-last, ...) straight
into ``mauve.compute_mauve(p_features=..., q_features=...)``. This is equivalent to
the ``jackjyzhang/mauve`` featurizer unlock, but keeps the encoder *identical* to the
one feeding our RBF-MMD, so the comparison isolates the statistic.

MAUVE configuration follows the reproduction recipe: ``num_buckets = N/10``,
``mauve_scaling_factor = 5``, reference set != evaluation set. Multiple k-means
seeds expose MAUVE's quantization variance.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def mauve_scores(
    p_features: np.ndarray,
    q_features: np.ndarray,
    *,
    num_buckets: int,
    scaling: float = 5.0,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    kmeans_explained_var: float = 0.9,
    kmeans_num_redo: int = 5,
    kmeans_max_iter: int = 500,
) -> Dict[str, float]:
    """Return mean/std of MAUVE and FI-KL across k-means seeds (quantization var)."""
    import mauve

    p = np.ascontiguousarray(p_features, dtype=np.float32)
    q = np.ascontiguousarray(q_features, dtype=np.float32)
    nb = max(2, int(num_buckets))
    mauve_vals: List[float] = []
    fikl_vals: List[float] = []
    for seed in seeds:
        res = mauve.compute_mauve(
            p_features=p,
            q_features=q,
            num_buckets=nb,
            pca_max_data=-1,
            kmeans_explained_var=kmeans_explained_var,
            kmeans_num_redo=kmeans_num_redo,
            kmeans_max_iter=kmeans_max_iter,
            divergence_curve_discretization_size=25,
            mauve_scaling_factor=float(scaling),
            seed=int(seed),
            verbose=False,
        )
        mauve_vals.append(float(res.mauve))
        fikl_vals.append(float(res.frontier_integral))
    return {
        "mauve_mean": float(np.mean(mauve_vals)),
        "mauve_std": float(np.std(mauve_vals)),
        "mauve_min": float(np.min(mauve_vals)),
        "mauve_max": float(np.max(mauve_vals)),
        "fi_kl_mean": float(np.mean(fikl_vals)),
        "fi_kl_std": float(np.std(fikl_vals)),
        "num_buckets": nb,
        "n_seeds": len(seeds),
    }


def kmeans_kl_frontier(
    p_features: np.ndarray,
    q_features: np.ndarray,
    *,
    num_buckets: int,
    seed: int = 0,
    kmeans_explained_var: float = 0.9,
    kmeans_num_redo: int = 1,
    kmeans_max_iter: int = 200,
) -> float:
    """MAUVE's k-means-KL frontier integral as a single scalar DIVERGENCE.

    This is the "k-means KL" distance of the representation x distance comparison: it
    quantizes the joint pool with k-means (MAUVE's representation-agnostic
    estimator) and returns the divergence-curve frontier integral, which is ~0
    when the two samples match and grows as they diverge, up to 1 for disjoint
    histograms. Unlike ``mauve`` (a [0,1] similarity that saturates near 0), it
    is a divergence computed directly from the two histograms, so it slots into
    the same exchangeable-null z-standardization as RBF-MMD / Fréchet / energy.
    ``kmeans_num_redo``/``kmeans_max_iter`` default lower than ``mauve_scores``
    because the null/power loops call this thousands of times; quantization noise
    is absorbed by the null standardization.

    Returns NaN when MAUVE cannot fit (e.g. too few points for the requested
    buckets) — the caller records the cell as undefined rather than faking it.
    """
    import mauve

    p = np.ascontiguousarray(p_features, dtype=np.float32)
    q = np.ascontiguousarray(q_features, dtype=np.float32)
    nb = max(2, int(num_buckets))
    try:
        res = mauve.compute_mauve(
            p_features=p,
            q_features=q,
            num_buckets=nb,
            pca_max_data=-1,
            kmeans_explained_var=kmeans_explained_var,
            kmeans_num_redo=kmeans_num_redo,
            kmeans_max_iter=kmeans_max_iter,
            divergence_curve_discretization_size=25,
            mauve_scaling_factor=5.0,
            seed=int(seed),
            verbose=False,
        )
        return float(res.frontier_integral)
    except Exception:
        return float("nan")
