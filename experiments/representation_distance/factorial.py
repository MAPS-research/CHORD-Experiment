"""Representation x distance factorial (Section 4, Figure "factorial", panel a).

Separates the two ingredients a corpus-level metric combines: the representation
each passage is embedded with, and the distance between the two embedded corpora.
Four representations (GPT-2-large last token as in MAUVE, ELECTRA masked mean as
in MAUVE-ELECTRA, BERT pooler output as in FBD, and CHORD's coherence-prompted
Qwen3.5-27B layer -3 readout) are crossed with four distances (RBF-MMD, energy
distance, Frechet distance, MAUVE's k-means KL) on the nine highest-severity
perturbation types of the counterfactual evaluation set. Each cell reports z for
the perturbation, z for the benign paraphrase, the selectivity S = z(harmful) -
z(benign) with a bootstrap 95% interval, and whether S is selectively positive.
The paper's grid counts, per (representation, distance), the perturbation types
with a selectively positive S. See common.py for the shared protocol.

Inputs:  <corpus-dir>/feats/<representation>/{reference,<condition>}.npy, written by
         experiments.counterfactual_eval.featurize with
         experiments/counterfactual_eval/configs/encoders_baselines.yaml
         (gpt2-large-last, electra-mean, fbd-bert) and encoders_chord_27b.yaml
         (qwen35-27b-prompteol-coherence-l62).
Outputs: <out-dir>/selectivity_grid_<representation>.csv  one row per (distance, type)
         <out-dir>/calibration_<representation>.csv       null statistics per distance

    python -m experiments.representation_distance.factorial \\
        --corpus-dir outputs/counterfactual/meta_eval \\
        --out-dir outputs/experiments/representation_distance
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from chord.metrics.distribution import median_bandwidth

from .common import (
    ALL_CONDITIONS,
    BENIGN,
    DISTANCE_LABEL,
    DISTANCES,
    FAMILIES,
    LABEL_BY_DIR,
    PCA_ESTIMATORS,
    REPRESENTATIONS,
    SEED,
    SELECTIVITY_RNG_TAG,
    CellEvaluator,
    draw_distances,
    load_matrix,
    pca_basis,
    seeded_rng,
    split_reference,
    write_csv,
)


def _null_statistics(null: np.ndarray, transport: np.ndarray, alpha: float) -> Dict:
    nd = null[np.isfinite(null)]
    td = transport[np.isfinite(transport)]
    if not len(nd):
        nan = float("nan")
        return dict(nmean=nan, nstd=nan, thr=nan, q95=nan, transport_fpr=nan, within_fpr=nan, n=0)
    thr = float(np.quantile(nd, 1 - alpha))
    return dict(
        nmean=float(nd.mean()),
        nstd=float(nd.std()),
        thr=thr,
        q95=float(np.quantile(nd, 0.95)),
        transport_fpr=float(np.mean(td > thr)) if len(td) else float("nan"),
        within_fpr=float(np.mean(nd > thr)),
        n=int(len(nd)),
    )


def _z(values: np.ndarray, st: Dict) -> Tuple[float, float, float]:
    v = values[np.isfinite(values)]
    if not len(v) or not (st["nstd"] > 0):
        return float("nan"), float("nan"), float("nan")

    def scale(x):
        return (x - st["nmean"]) / st["nstd"]

    return scale(float(v.mean())), scale(np.quantile(v, 0.025)), scale(np.quantile(v, 0.975))


def run_representation(
    rep_dir: str,
    corpus_dir: Path,
    dev_size: int,
    draws: int,
    pca_dim: int,
    alpha: float,
    distances: List[str],
) -> Tuple[List[Dict], List[Dict]]:
    feat = corpus_dir / "feats" / rep_dir
    ref = load_matrix(feat / "reference.npy")
    dev, cal_pool, test_pool = split_reference(ref, SEED, dev_size)
    n = min(256, len(cal_pool) // 2, len(test_pool) // 2)
    sigma = float(median_bandwidth(dev, max_samples=2000, seed=SEED))
    num_buckets = max(2, n // 10)

    mu, proj, retained = pca_basis(ref, pca_dim)

    def to_pca(x):
        return (x - mu) @ proj

    cal_pca, test_pca = to_pca(cal_pool), to_pca(test_pool)
    cond_raw = {c: load_matrix(feat / f"{c}.npy") for c in ALL_CONDITIONS}
    cond_pca = {c: to_pca(v) for c, v in cond_raw.items()}

    tag = SELECTIVITY_RNG_TAG
    null_ev = CellEvaluator(cal_pool, cal_pca, cal_pool, cal_pca, sigma, num_buckets)
    null = draw_distances(null_ev, distances, draws, n, n, seeded_rng(tag, "null", n), null=True)
    test_ev = CellEvaluator(test_pool, test_pca, test_pool, test_pca, sigma, num_buckets)
    transport = draw_distances(
        test_ev, distances, draws, n, n, seeded_rng(tag, "transport", n), null=True
    )
    stats = {d: _null_statistics(null[d], transport[d], alpha) for d in distances}

    attack = {}
    for cond in ALL_CONDITIONS:
        ev = CellEvaluator(cal_pool, cal_pca, cond_raw[cond], cond_pca[cond], sigma, num_buckets)
        attack[cond] = draw_distances(
            ev, distances, draws, n, n, seeded_rng(tag, cond, n), null=False
        )

    label = LABEL_BY_DIR.get(rep_dir, rep_dir)
    grid_rows, calib_rows = [], []
    for d in distances:
        st = stats[d]
        calib_rows.append(
            {
                "representation": label,
                "rep_dir": rep_dir,
                "distance": d,
                "distance_label": DISTANCE_LABEL[d],
                "uses_pca_adapter": d in PCA_ESTIMATORS,
                "pca_dim": pca_dim,
                "pca_retained_var": round(retained, 4),
                "sigma": round(sigma, 6),
                "n": n,
                "draws": draws,
                "num_buckets": num_buckets,
                "null_mean": st["nmean"],
                "null_std": st["nstd"],
                "q95": st["q95"],
                "within_fpr": st["within_fpr"],
                "transport_fpr": st["transport_fpr"],
                "n_valid_null": st["n"],
                "nominal_alpha": alpha,
            }
        )
        z_benign, _, _ = _z(attack[BENIGN[0]][d], st)
        for cond, group in FAMILIES:
            z_harm, z_lo, z_hi = _z(attack[cond][d], st)
            harm, benign = attack[cond][d], attack[BENIGN[0]][d]
            mask = np.isfinite(harm) & np.isfinite(benign)
            if mask.sum() and st["nstd"] > 0:
                sel = (harm[mask] - benign[mask]) / st["nstd"]
                s_mean = float(sel.mean())
                s_lo, s_hi = float(np.quantile(sel, 0.025)), float(np.quantile(sel, 0.975))
            else:
                s_mean = s_lo = s_hi = float("nan")
            verdict = "SEL" if s_lo > 0 else ("ANTI" if s_hi < 0 else "n.s.")
            grid_rows.append(
                {
                    "representation": label,
                    "rep_dir": rep_dir,
                    "distance": d,
                    "distance_label": DISTANCE_LABEL[d],
                    "family": cond,
                    "group": group,
                    "z_harmful": round(z_harm, 4),
                    "z_harmful_ci_low": round(z_lo, 4),
                    "z_harmful_ci_high": round(z_hi, 4),
                    "z_benign": round(z_benign, 4),
                    "selectivity": round(s_mean, 4) if np.isfinite(s_mean) else s_mean,
                    "sel_ci_low": round(s_lo, 4) if np.isfinite(s_lo) else s_lo,
                    "sel_ci_high": round(s_hi, 4) if np.isfinite(s_hi) else s_hi,
                    "verdict": verdict,
                    "transport_fpr": st["transport_fpr"],
                    "uses_pca_adapter": d in PCA_ESTIMATORS,
                }
            )
            print(
                f"[grid] {rep_dir:32s} {d:10s} {cond:28s} z={z_harm:8.2f} "
                f"(benign {z_benign:6.2f}) S={s_mean:8.2f} [{s_lo:.2f},{s_hi:.2f}] {verdict}",
                flush=True,
            )
    return grid_rows, calib_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--rep",
        default="all",
        help="feature directory name, comma list, or 'all' (the four paper representations)",
    )
    ap.add_argument("--corpus-dir", default="outputs/counterfactual/meta_eval")
    ap.add_argument("--out-dir", default="outputs/experiments/representation_distance")
    ap.add_argument("--dev-size", type=int, default=1500, help="bandwidth / PCA split")
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
    if args.rep == "all":
        reps = list(REPRESENTATIONS.values())
    else:
        reps = [r.strip() for r in args.rep.split(",") if r.strip()]

    corpus_dir, out_dir = Path(args.corpus_dir), Path(args.out_dir)
    for rep_dir in reps:
        t0 = time.time()
        grid, calib = run_representation(
            rep_dir,
            corpus_dir,
            args.dev_size,
            args.draws,
            args.pca_dim,
            args.nominal_alpha,
            distances,
        )
        write_csv(out_dir / f"selectivity_grid_{rep_dir}.csv", grid)
        write_csv(out_dir / f"calibration_{rep_dir}.csv", calib)
        print(f"[grid] {rep_dir} done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
