"""Score cached generator features with raw and null-standardized CHORD.

All encoders share reference splits and bootstrap streams, so comparisons vary
only the representation. CSV names retain the historical ``chord_scores`` prefix
for compatibility with published results and downstream analysis.

    python -m experiments.generation_scoring.score_chord \
        --config experiments/single_fold/configs/single_fold_chord_27b.yaml \
        --encoders qwen35-27b-prompteol-coherence-l62
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from chord.metrics.distribution import median_bandwidth
from chord.utils.hashing import seeded_rng
from chord.utils.io import load_runs, write_csv

from ..counterfactual_eval.nulls import exchangeable_attack, exchangeable_null
from .score import _load, _setting, _split_reference


def score_encoder(root: Path, runs, enc: str, *, seed, draws, n_eval, dev_size, block_size):
    ref = _load(root / "features" / enc / "reference.npy")
    dev, cal, test = _split_reference(ref, seed, dev_size)
    n = min(n_eval, len(cal) // 2, len(test) // 2)
    if n < 2:
        raise ValueError(f"{enc}: reference pool too small (n={n})")
    sigma = median_bandwidth(dev, max_samples=2000, seed=seed)
    null = exchangeable_null(
        cal,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng("real-dlm", "null", n),
    )
    nmean, nstd = float(null.mean()), float(null.std())
    q95 = float(np.quantile(null, 0.95))
    transport = exchangeable_null(
        test,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng("real-dlm", "transport", n),
    )
    rows = []
    raw_rows = []
    for run in runs:
        cand = _load(root / "features" / enc / f"{run.run_id}.npy")
        if len(cand) < n:
            raise ValueError(f"{run.run_id}: only {len(cand)} samples (<{n})")
        d = exchangeable_attack(
            cal,
            cand,
            n,
            n,
            draws=draws,
            sigma=sigma,
            block_size=block_size,
            rng=seeded_rng("real-dlm", run.run_id, n),
        )
        z = (float(d.mean()) - nmean) / nstd if nstd > 0 else float("nan")
        rows.append(
            {
                "run_id": run.run_id,
                "model": run.model,
                "family": run.family,
                "seed": run.seed,
                "setting": _setting(run),
                "n": n,
                "encoder": enc,
                "metric": "chord_mmd_z",
                "value": z,
                "ci_low": (float(np.quantile(d, 0.025)) - nmean) / nstd,
                "ci_high": (float(np.quantile(d, 0.975)) - nmean) / nstd,
            }
        )
        raw_rows.append(
            {
                "run_id": run.run_id,
                "model": run.model,
                "family": run.family,
                "seed": run.seed,
                "setting": _setting(run),
                "n": n,
                "encoder": enc,
                "metric": "chord_mmd_raw",
                "value": float(d.mean()),
                "ci_low": float(np.quantile(d, 0.025)),
                "ci_high": float(np.quantile(d, 0.975)),
            }
        )
    calib = {
        "encoder": enc,
        "sigma": float(sigma),
        "n": n,
        "null_mean": nmean,
        "null_std": nstd,
        "q95": q95,
        "transport_fpr": float(np.mean(transport > q95)),
    }
    return rows, raw_rows, calib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--encoders",
        required=True,
        help="comma-separated encoder names; first is the comparison baseline",
    )
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    runs = load_runs((cfg_path.parent / cfg["runs_manifest"]).resolve(), require_samples=False)
    root = (cfg_path.parent / cfg["output_dir"]).resolve()
    sc = cfg["scoring"]
    seed = int(cfg.get("seed", 0))
    encs = [e.strip() for e in args.encoders.split(",") if e.strip()]

    by_enc = {}
    calibs = []
    for enc in encs:
        rows, raw_rows, calib = score_encoder(
            root,
            runs,
            enc,
            seed=seed,
            draws=int(sc.get("draws", 200)),
            n_eval=int(sc.get("n_eval", 256)),
            dev_size=int(sc.get("dev_size", 300)),
            block_size=int(sc.get("block_size", 1024)),
        )
        by_enc[enc] = {r["run_id"]: r for r in rows}
        calibs.append(calib)
        write_csv(root / "scores" / f"chord_scores_{enc}.csv", rows)
        write_csv(root / "scores" / f"chord_scores_raw_{enc}.csv", raw_rows)
        print(
            f"[{enc}] sigma={calib['sigma']:.4f} transport_fpr={calib['transport_fpr']:.3f}",
            flush=True,
        )

    print(
        f"\n{'run_id':22s} "
        + " ".join(f"{e:>14s}" for e in encs)
        + ("  Δ(last-first)" if len(encs) > 1 else "")
    )
    for run in runs:
        rid = run.run_id
        vals = [by_enc[e][rid]["value"] for e in encs]
        line = f"{rid:22s} " + " ".join(f"{v:14.2f}" for v in vals)
        if len(encs) > 1:
            line += f"   {vals[-1] - vals[0]:+8.2f}"
        print(line, flush=True)
    write_csv(root / "scores" / "chord_calibration.csv", calibs)


if __name__ == "__main__":
    main()
