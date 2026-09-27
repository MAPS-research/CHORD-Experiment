"""CHORD z and MAUVE of one truncation length, fold by fold on Table 2's folds.

Paper result: the cells of tab:chord-length and tab:mauve-length (appendix
"Sensitivity to Evaluation Window Length") for one length.

Input   --dir <out>/L{L} from build_truncated_corpora.py, after featurizing its
        chord.yaml (CHORD 27B) and mauve.yaml (GPT-2-large).
Folds   as experiments.generator_eval.score_chord_folds: the reference pool is
        permuted with the config seed, the first ``dev`` windows fit the RBF
        bandwidth, the next folds x size windows are the reference folds, and
        generator fold k is its samples [k*size, (k+1)*size) in run order.
CHORD   biased RBF-MMD^2 of reference fold k vs generator fold k, standardized by
        this length's exchangeable null (MMD^2 between two disjoint size-window
        subsets of the reference folds): z_k = (MMD^2_k - null mean) / null std.
MAUVE   official MAUVE of reference fold k vs generator fold k (size/10 buckets,
        c = 5), averaged over five k-means seeds.
Output  <dir>/scores/length_folds.csv (one row per generator x fold) and
        <dir>/scores/length_calibration.csv

    python -m experiments.window_length.score_folds \\
        --dir outputs/experiments/window_length_folds/L512
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from chord.metrics.distribution import median_bandwidth, rbf_mmd
from chord.utils.hashing import seeded_rng
from chord.utils.io import load_runs, write_csv

from ..counterfactual_eval.mauve_eval import mauve_scores

CHORD_ENCODER = "qwen35-27b-prompteol-coherence-l62"
MAUVE_SEEDS = (0, 1, 2, 3, 4)


def fold_matrices(
    root: Path, encoder: str, seed: int, folds: int, size: int, dev: int
) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, List[np.ndarray]]]:
    """(dev rows, reference folds, {generator: its folds}) of one encoder's features."""
    feats = root / "features" / encoder
    ref = np.load(feats / "reference.npy").astype(np.float64)
    need = folds * size
    assert len(ref) >= dev + need, f"{encoder}: reference has {len(ref)} rows"
    order = np.random.default_rng(seed).permutation(len(ref))
    ref_rows = ref[order[dev : dev + need]]
    runs = load_runs(root / "runs.jsonl")
    gens: Dict[str, List[np.ndarray]] = {}
    for model in dict.fromkeys(r.model for r in runs):
        x = np.concatenate([np.load(feats / f"{r.run_id}.npy") for r in runs if r.model == model])
        assert len(x) >= need, f"{encoder}/{model}: {len(x)} samples, need {need}"
        gens[model] = [x[k * size : (k + 1) * size].astype(np.float64) for k in range(folds)]
    ref_folds = [ref_rows[k * size : (k + 1) * size] for k in range(folds)]
    return ref[order[:dev]], ref_folds, gens


def chord_folds(root: Path, seed: int, folds: int, size: int, dev: int, draws: int):
    dev_rows, ref_folds, gens = fold_matrices(root, CHORD_ENCODER, seed, folds, size, dev)
    sigma = float(median_bandwidth(dev_rows, max_samples=2000, seed=seed))
    pool = np.concatenate(ref_folds)
    rng = seeded_rng("window-length-null", root.name, size)
    null = np.empty(draws)
    for d in range(draws):
        idx = rng.choice(len(pool), size=2 * size, replace=False)
        null[d] = rbf_mmd(pool[idx[:size]], pool[idx[size:]], sigma, False, 1024)
    mean, std = float(null.mean()), float(null.std())
    raw = {
        m: [rbf_mmd(ref_folds[k], g[k], sigma, False, 1024) for k in range(folds)]
        for m, g in gens.items()
    }
    calib = {"length": root.name, "sigma": sigma, "null_mean": mean, "null_std": std}
    return raw, {m: [(v - mean) / std for v in vals] for m, vals in raw.items()}, calib


def _mauve_cell(args):
    ref, gen = args
    return mauve_scores(ref, gen, num_buckets=max(2, len(ref) // 10), seeds=MAUVE_SEEDS)


def mauve_folds(root: Path, encoder: str, seed: int, folds: int, size: int, dev: int, workers):
    _dev, ref_folds, gens = fold_matrices(root, encoder, seed, folds, size, dev)
    cells = [(m, k) for m in gens for k in range(folds)]
    jobs = [(ref_folds[k], gens[m][k]) for m, k in cells]
    if workers <= 1:
        return dict(zip(cells, map(_mauve_cell, jobs)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return dict(zip(cells, pool.map(_mauve_cell, jobs)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", required=True, help="<out>/L{L} of build_truncated_corpora")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--dev", type=int, default=150)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8, help="parallel MAUVE computations")
    args = ap.parse_args()
    root = Path(args.dir).resolve()
    seed = int(yaml.safe_load((root / "chord.yaml").read_text(encoding="utf-8"))["seed"])
    mauve_cfg = yaml.safe_load((root / "mauve.yaml").read_text(encoding="utf-8"))
    mauve_encoder = mauve_cfg["encoders"][0]["name"]
    fold_args = (seed, args.folds, args.size, args.dev)

    raw, z, calib = chord_folds(root, *fold_args, args.draws)
    print(f"[length] {root.name}: sigma={calib['sigma']:.3f} null={calib['null_mean']:.5f}")
    mauve = mauve_folds(root, mauve_encoder, *fold_args, args.workers)
    family = {r.model: r.family for r in load_runs(root / "runs.jsonl")}
    rows = []
    for model in raw:
        for k in range(args.folds):
            m = mauve[(model, k)]
            rows.append(
                {
                    "model": model,
                    "family": family[model],
                    "fold": k,
                    "n": args.size,
                    "chord_mmd_x100": round(100 * raw[model][k], 5),
                    "chord_z": round(z[model][k], 4),
                    "mauve": round(m["mauve_mean"], 5),
                    "mauve_min": round(m["mauve_min"], 5),
                    "mauve_max": round(m["mauve_max"], 5),
                }
            )
        print(
            f"[length] {root.name} {model:16s} z={np.mean(z[model]):8.1f} "
            f"mauve={np.mean([mauve[(model, k)]['mauve_mean'] for k in range(args.folds)]):.3f}",
            flush=True,
        )
    write_csv(root / "scores" / "length_folds.csv", rows)
    write_csv(root / "scores" / "length_calibration.csv", [calib])
    print(f"[length] wrote {root / 'scores' / 'length_folds.csv'}")


if __name__ == "__main__":
    main()
