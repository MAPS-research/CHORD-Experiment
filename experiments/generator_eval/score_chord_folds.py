#!/usr/bin/env python3
"""Table 2 as mean +- std over disjoint folds (CPU).

Each generator has `folds` x `size` samples and the human reference has
dev + `folds` x `size` windows. Fold k scores MMD^2(reference fold k, candidate
fold k) with the paper's estimator (biased RBF-MMD^2, bandwidth = median
pairwise distance on the dev windows, fitted once per encoder). No text is used
in two folds. Reported: mean and std (ddof=1) over folds, x100; rank by mean;
Spearman against the first encoder; and, per pair of generators, in how many
folds each encoder orders them the same way as the first encoder's mean.

  python -m experiments.generator_eval.score_chord_folds \
      --config experiments/generator_eval/configs/featurize.yaml \
      --encoders qwen35-27b-prompteol-coherence-l62 chord-qwen3.5-2b-student
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from chord.metrics.distribution import median_bandwidth, rbf_mmd
from chord.utils.io import load_runs


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--encoders", nargs="+", required=True, help="first = the reference ordering (teacher)"
    )
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--dev", type=int, default=150)
    ap.add_argument(
        "--unbiased", action="store_true", help="unbiased MMD^2 instead of the paper's biased one"
    )
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = (cfg_path.parent / cfg["output_dir"]).resolve()
    runs = load_runs((cfg_path.parent / cfg["runs_manifest"]).resolve(), require_samples=False)
    seed = int(cfg.get("seed", 0))
    need = args.folds * args.size

    models = list(dict.fromkeys(r.model for r in runs))
    table, rows = {}, []
    for enc in args.encoders:
        ref = np.load(root / "features" / enc / "reference.npy").astype(np.float64)
        assert len(ref) >= args.dev + need, (
            f"{enc}: reference has {len(ref)} rows, need {args.dev + need}"
        )
        order = np.random.default_rng(seed).permutation(len(ref))
        dev, ref_folds = ref[order[: args.dev]], ref[order[args.dev : args.dev + need]]
        sigma = median_bandwidth(dev, max_samples=2000, seed=seed)
        per_model = {}
        for model in models:
            cand = np.concatenate(
                [
                    np.load(root / "features" / enc / f"{r.run_id}.npy").astype(np.float64)
                    for r in runs
                    if r.model == model
                ]
            )
            assert len(cand) >= need, f"{enc}/{model}: {len(cand)} samples, need {need}"
            # runs are one seed each, already in generation order:
            # fold k = rows [k*size, (k+1)*size)
            vals = (
                np.array(
                    [
                        rbf_mmd(
                            ref_folds[k * args.size : (k + 1) * args.size],
                            cand[k * args.size : (k + 1) * args.size],
                            sigma,
                            args.unbiased,
                            1024,
                        )
                        for k in range(args.folds)
                    ]
                )
                * 100
            )
            per_model[model] = vals
            for k, v in enumerate(vals):
                rows.append(
                    {
                        "encoder": enc,
                        "model": model,
                        "fold": k,
                        "n": args.size,
                        "sigma": sigma,
                        "mmd_x100": v,
                    }
                )
        table[enc] = per_model
        print(f"[{enc}] sigma={sigma:.4f} dim={ref.shape[1]}", flush=True)

    base = args.encoders[0]
    base_mean = np.array([table[base][m].mean() for m in models])
    print(
        f"\nTABLE 2, {args.folds} disjoint folds x {args.size} samples per side, "
        f"{'unbiased' if args.unbiased else 'biased'} RBF-MMD^2 x100, mean +- std (rank)"
    )
    print(f"{'generator':24s}" + "".join(f"{e[:26]:>30s}" for e in args.encoders))
    ranks = {
        e: np.argsort(np.argsort([table[e][m].mean() for m in models])) + 1 for e in args.encoders
    }
    for i, m in enumerate(models):
        print(
            f"{m:24s}"
            + "".join(
                f"{table[e][m].mean():16.2f} +- {table[e][m].std(ddof=1):5.2f} ({ranks[e][i]})  "
                for e in args.encoders
            )
        )
    for e in args.encoders[1:]:
        mean_e = np.array([table[e][m].mean() for m in models])
        fold_rho = [
            spearman(base_mean, np.array([table[e][m][k] for m in models]))
            for k in range(args.folds)
        ]
        print(
            f"\n{e}: Spearman of means vs {base} = {spearman(base_mean, mean_e):.3f}; "
            f"per-fold rho mean {np.mean(fold_rho):.3f} min {np.min(fold_rho):.3f}"
        )
    print(
        "\npairwise order, folds (of %d) in which the left generator scores LOWER than the right:"
        % args.folds
    )
    print(f"{'pair':44s}" + "".join(f"{e[:26]:>30s}" for e in args.encoders))
    by_mean = [m for _, m in sorted(zip(base_mean, models))]
    for i in range(len(by_mean) - 1):
        for j in range(i + 1, min(i + 4, len(by_mean))):
            a, b = by_mean[i], by_mean[j]
            line = f"{a + ' < ' + b:44s}"
            for e in args.encoders:
                wins = int((table[e][a] < table[e][b]).sum())
                gap = table[e][b].mean() - table[e][a].mean()
                sd = np.std(table[e][b] - table[e][a], ddof=1)
                line += f"{wins:>12d}/{args.folds}  gap {gap:6.2f} sd {sd:5.2f}"
            print(line)

    out = root / "scores"
    out.mkdir(parents=True, exist_ok=True)
    tag = "unbiased" if args.unbiased else "biased"
    with open(out / f"repeat_folds_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out / f'repeat_folds_{tag}.csv'}")


if __name__ == "__main__":
    main()
