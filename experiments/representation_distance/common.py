"""Shared machinery for the representation x distance experiments (Section 4).

Every (representation, distance) cell is read on one null-standardized scale:
an exchangeable permutation null (two disjoint clean halves) turns each distance
into a z-score, the clean-vs-perturbed comparison gives z(harmful) and z(benign),
and selectivity is S = z(harmful) - z(benign). The same bootstrap index draws feed
all four distances, so within a representation only the estimator varies, and
every representation uses the same protocol.

Distances:
  rbf_mmd    biased squared RBF-MMD (CHORD's distance), median-heuristic bandwidth
             fitted on the held-out dev split
  energy     biased energy distance (Szekely & Rizzo)
  frechet    Frechet distance between Gaussian fits (FBD's distance)
  kmeans_kl  MAUVE's k-means-quantized KL divergence frontier

The two estimators that fit a Gaussian or a quantizer (frechet, kmeans_kl) are
evaluated in a PCA-128 subspace fitted once on the clean reference pool: at
n << d a raw-dimension Gaussian fit is rank-deficient and the per-draw cost is
cubic in d. The kernel distances act on the raw features. The retained PCA
variance is written to the calibration CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from chord.metrics.distribution import frechet_distance, rbf_kernel
from chord.utils.hashing import seeded_rng  # noqa: F401  (re-exported for the experiment scripts)

from ..counterfactual_eval.mauve_eval import kmeans_kl_frontier

# Paper configuration: the counterfactual evaluation set (Qwen3-30B-A3B editor),
# highest severity of each of the nine perturbation types, benign paraphrase as
# the control. Feature directories are the ones written by
# experiments/counterfactual_eval/configs/encoders_{baselines,chord_27b}.yaml.
REPRESENTATIONS: Dict[str, str] = {
    "GPT-2 (MAUVE)": "gpt2-large-last",
    "ELECTRA (MAUVE-electra)": "electra-mean",
    "BERT (FBD)": "fbd-bert",
    "prompteol-coherence@Qwen3.5-27B-L62 (ours)": "qwen35-27b-prompteol-coherence-l62",
}
LABEL_BY_DIR = {v: k for k, v in REPRESENTATIONS.items()}

FAMILIES: List[Tuple[str, str]] = [
    ("contradiction__dose3", "relation"),
    ("causal_reverse__dose3", "relation"),
    ("broken_transition__dose3", "discourse"),
    ("topic_drift__dose3", "discourse"),
    ("sentence_permutation__r100", "order"),
    ("local_shuffle__r100", "order"),
    ("repetition__r100", "repetition"),
    ("dlm_splice__r050", "dlm_mix"),
    ("corpus_mix__r075_d8", "corpus_mix"),
]
BENIGN: Tuple[str, str] = ("benign_paraphrase__dose3", "control")
ALL_CONDITIONS = [c for c, _ in FAMILIES] + [BENIGN[0]]

DISTANCES = ["rbf_mmd", "energy", "frechet", "kmeans_kl"]
DISTANCE_LABEL = {
    "rbf_mmd": "RBF-MMD (ours)",
    "energy": "energy distance",
    "frechet": "Fréchet-Gaussian (FBD)",
    "kmeans_kl": "k-means-KL frontier (MAUVE)",
}
PCA_ESTIMATORS = {"frechet", "kmeans_kl"}

POWER_NS = [16, 32, 64, 128, 256]
SEED = 20260621
# Random-stream tags. They seed every bootstrap draw, so they are kept verbatim
# from the run that produced the paper's figures.
SELECTIVITY_RNG_TAG = "meta-eval"
POWER_RNG_TAG = "ea1-meta-power"


def load_matrix(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float64)


def split_reference(matrix: np.ndarray, seed: int, dev_size: int):
    """dev (bandwidth / PCA) | calibration region | test region, as in selectivity.py."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(matrix))
    dev = matrix[idx[:dev_size]]
    rest = matrix[idx[dev_size:]]
    cut = len(rest) // 2
    return dev, rest[:cut], rest[cut:]


def pca_basis(pool: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Frozen PCA basis: (mean, projection matrix d x k, retained variance)."""
    mu = pool.mean(axis=0)
    _, sv, vt = np.linalg.svd(pool - mu, full_matrices=False)
    k = min(k, vt.shape[0])
    var = sv**2
    retained = float(np.cumsum(var)[k - 1] / var.sum())
    return mu, vt[:k].T, retained


def _euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    ln = np.sum(left * left, axis=1)[:, None]
    rn = np.sum(right * right, axis=1)[None, :]
    return np.sqrt(np.maximum(ln + rn - 2 * left @ right.T, 0.0))


class CellEvaluator:
    """Precomputed Gram / distance blocks for one (reference pool, candidate pool)
    pairing; evaluates all four distances on sampled index sets. For the null the
    candidate pool is the reference pool itself."""

    def __init__(self, ref_raw, ref_pca, cand_raw, cand_pca, sigma, num_buckets):
        kernel = rbf_kernel(sigma)
        self.num_buckets = num_buckets
        self.ref_pca = ref_pca
        self.cand_pca = cand_pca
        self.g_rr = kernel(ref_raw, ref_raw)
        self.g_cc = kernel(cand_raw, cand_raw)
        self.g_rc = kernel(ref_raw, cand_raw)
        self.d_rr = _euclidean(ref_raw, ref_raw)
        self.d_cc = _euclidean(cand_raw, cand_raw)
        self.d_rc = _euclidean(ref_raw, cand_raw)

    def evaluate(self, distance: str, ix: np.ndarray, iy: np.ndarray) -> float:
        if distance == "rbf_mmd":
            xx = self.g_rr[np.ix_(ix, ix)].mean()
            yy = self.g_cc[np.ix_(iy, iy)].mean()
            xy = self.g_rc[np.ix_(ix, iy)].mean()
            return float(xx + yy - 2.0 * xy)
        if distance == "energy":
            xx = self.d_rr[np.ix_(ix, ix)].mean()
            yy = self.d_cc[np.ix_(iy, iy)].mean()
            xy = self.d_rc[np.ix_(ix, iy)].mean()
            return float(2.0 * xy - xx - yy)
        a, b = self.ref_pca[ix], self.cand_pca[iy]
        if distance == "frechet":
            try:
                return frechet_distance(a, b)
            except Exception:
                return float("nan")
        if distance == "kmeans_kl":
            return kmeans_kl_frontier(
                a,
                b,
                num_buckets=self.num_buckets,
                seed=0,
                kmeans_explained_var=0.99,
                kmeans_num_redo=1,
                kmeans_max_iter=100,
            )
        raise ValueError(distance)


def draw_distances(ev: CellEvaluator, distances, draws, n_ref, n, rng, *, null):
    """`draws` bootstrap evaluations -> {distance: array[draws]}.

    null=True  draws two disjoint halves of the same pool;
    null=False draws n_ref from the reference pool and n from the candidate pool.
    The index draws are shared across distances (one rng stream)."""
    out = {d: np.empty(draws, dtype=np.float64) for d in distances}
    n_pool, m_cand = len(ev.g_rr), len(ev.g_cc)
    for t in range(draws):
        if null:
            idx = rng.choice(n_pool, size=n_ref + n, replace=False)
            ix, iy = idx[:n_ref], idx[n_ref:]
        else:
            ix = rng.choice(n_pool, size=n_ref, replace=False)
            iy = rng.choice(m_cand, size=n, replace=m_cand < n)
        for d in distances:
            out[d][t] = ev.evaluate(d, ix, iy)
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
