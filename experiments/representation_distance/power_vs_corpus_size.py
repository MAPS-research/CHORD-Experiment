"""Detection power vs corpus size N (Section 4 Figure "factorial" panel b, and the
appendix figure "Effect of corpus size").

For each distance of the representation x distance comparison (see common.py)
and corpus sizes N in {16, 32, 64, 128, 256}, draws 200 bootstrap pairs of N
reference vs N candidate passages per perturbation type and 200 null pairs of
two disjoint clean halves. It reports every draw's distance and its z against
the null at that N, and the detection rate: the fraction of draws above the
null's 95th percentile, with a Wilson 95% interval. The benign paraphrase is
included; its rate rising with N is why detection is defined against the benign
response rather than against zero.

Inputs:  <corpus-dir>/feats/<representation>/{reference,<condition>}.npy
Outputs: <out-dir>/draws_vs_n_<representation>.csv  every draw (distance, type, N, D, z)
         <out-dir>/power_vs_n_<representation>.csv  detection rate per (distance, type, N)

    python -m experiments.representation_distance.power_vs_corpus_size \\
        --rep qwen35-27b-prompteol-coherence-l62 \\
        --corpus-dir outputs/counterfactual/meta_eval \\
        --out-dir outputs/experiments/representation_distance
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chord.metrics.distribution import median_bandwidth

from ..metric_utils.statistics import binomial_confidence_interval
from .common import (
    BENIGN,
    DISTANCE_LABEL,
    DISTANCES,
    FAMILIES,
    LABEL_BY_DIR,
    POWER_NS,
    POWER_RNG_TAG,
    SEED,
    CellEvaluator,
    draw_distances,
    load_matrix,
    pca_basis,
    seeded_rng,
    write_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rep", default="qwen35-27b-prompteol-coherence-l62", help="feature directory")
    ap.add_argument("--corpus-dir", default="outputs/counterfactual/meta_eval")
    ap.add_argument("--out-dir", default="outputs/experiments/representation_distance")
    ap.add_argument("--dev-size", type=int, default=1500, help="bandwidth split")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--nominal-alpha", type=float, default=0.05)
    ap.add_argument(
        "--distances",
        default=",".join(DISTANCES),
        help=f"comma-separated subset of {DISTANCES}",
    )
    args = ap.parse_args()
    distances = [d.strip() for d in args.distances.split(",") if d.strip()]
    unknown = sorted(set(distances) - set(DISTANCES))
    if unknown:
        raise SystemExit(f"unknown distances: {unknown}")

    rep_dir = args.rep
    feat = Path(args.corpus_dir) / "feats" / rep_dir
    ref = load_matrix(feat / "reference.npy")
    idx = np.random.default_rng(SEED).permutation(len(ref))
    dev, clean = ref[idx[: args.dev_size]], ref[idx[args.dev_size :]]
    sigma = float(median_bandwidth(dev, max_samples=2000, seed=SEED))

    mu, proj, _ = pca_basis(ref, args.pca_dim)

    def to_pca(x):
        return (x - mu) @ proj

    conditions = list(FAMILIES) + [BENIGN]
    clean_pca = to_pca(clean)
    cond_raw = {c: load_matrix(feat / f"{c}.npy") for c, _ in conditions}
    evaluators = {
        c: CellEvaluator(clean, clean_pca, cond_raw[c], to_pca(cond_raw[c]), sigma, 2)
        for c, _ in conditions
    }
    null_ev = CellEvaluator(clean, clean_pca, clean, clean_pca, sigma, 2)

    label = LABEL_BY_DIR.get(rep_dir, rep_dir)
    draw_rows, power_rows = [], []
    for n in POWER_NS:
        buckets = max(2, n // 10)
        null_ev.num_buckets = buckets
        null = draw_distances(
            null_ev,
            distances,
            args.draws,
            n,
            n,
            seeded_rng(POWER_RNG_TAG, rep_dir, "null", n),
            null=True,
        )
        null_stats = {}
        for d in distances:
            nd = null[d][np.isfinite(null[d])]
            null_stats[d] = (
                (
                    float(nd.mean()),
                    float(nd.std(ddof=1)),
                    float(np.quantile(nd, 1 - args.nominal_alpha)),
                )
                if len(nd)
                else (np.nan, np.nan, np.nan)
            )
        pools = [("null", "null", null)]
        for cond, group in conditions:
            ev = evaluators[cond]
            ev.num_buckets = buckets
            values = draw_distances(
                ev,
                distances,
                args.draws,
                n,
                n,
                seeded_rng(POWER_RNG_TAG, rep_dir, cond, n),
                null=False,
            )
            pools.append((cond, group, values))

        for cond, group, values in pools:
            for d in distances:
                nmean, nstd, threshold = null_stats[d]
                arr = values[d]
                for t, v in enumerate(arr):
                    z = (v - nmean) / nstd if np.isfinite(v) and nstd > 0 else np.nan
                    draw_rows.append(
                        {
                            "rep_dir": rep_dir,
                            "distance": d,
                            "distance_label": DISTANCE_LABEL[d],
                            "family": cond,
                            "group": group,
                            "N": n,
                            "draw": t,
                            "D": v,
                            "z": z,
                            "null_mean": nmean,
                            "null_std": nstd,
                            "null_q95": threshold,
                        }
                    )
                if cond == "null":
                    continue
                valid = arr[np.isfinite(arr)]
                if len(valid) and np.isfinite(threshold):
                    k, m = int(np.sum(valid > threshold)), len(valid)
                    rate, ci = k / m, binomial_confidence_interval(k, m)
                else:
                    m, rate, ci = 0, float("nan"), (float("nan"), float("nan"))
                power_rows.append(
                    {
                        "representation": label,
                        "rep_dir": rep_dir,
                        "distance": d,
                        "distance_label": DISTANCE_LABEL[d],
                        "family": cond,
                        "group": group,
                        "N": n,
                        "detection_rate": round(rate, 4) if np.isfinite(rate) else rate,
                        "det_ci_low": round(ci[0], 4),
                        "det_ci_high": round(ci[1], 4),
                        "n_valid": m,
                        "is_control": cond == BENIGN[0],
                    }
                )
                print(f"[power] N={n:4d} {d:10s} {cond:28s} detection={rate:.3f}", flush=True)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / f"draws_vs_n_{rep_dir}.csv", draw_rows)
    write_csv(out_dir / f"power_vs_n_{rep_dir}.csv", power_rows)
    print(f"[power] wrote {out_dir}/{{draws,power}}_vs_n_{rep_dir}.csv", flush=True)


if __name__ == "__main__":
    main()
