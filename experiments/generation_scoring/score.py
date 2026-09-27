"""Score real DLM operating points with CHORD, two MAUVEs, and FBD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from chord.metrics.distribution import frechet_distance, median_bandwidth
from chord.utils.hashing import seeded_rng
from chord.utils.io import load_runs, write_csv

from ..counterfactual_eval.mauve_eval import mauve_scores
from ..counterfactual_eval.nulls import exchangeable_attack, exchangeable_null


def _load(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float64)


def _split_reference(matrix: np.ndarray, seed: int, dev_size: int):
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(len(matrix))
    dev = matrix[indexes[:dev_size]]
    remaining = matrix[indexes[dev_size:]]
    cut = len(remaining) // 2
    return dev, remaining[:cut], remaining[cut:]


def _setting(run) -> str:
    return json.dumps(run.setting, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-missing-runs", action="store_true")
    parser.add_argument("--mmd-encoder", help="override scoring.mmd_encoder (e.g. compare prompts)")
    parser.add_argument("--tag", default="", help="suffix for output CSVs when comparing variants")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = load_runs(
        (config_path.parent / cfg["runs_manifest"]).resolve(),
        require_samples=not args.allow_missing_runs,
    )
    root = (config_path.parent / cfg["output_dir"]).resolve()
    score_cfg = cfg["scoring"]
    draws = int(score_cfg.get("draws", 200))
    n_eval = int(score_cfg.get("n_eval", 1000))
    dev_size = int(score_cfg.get("dev_size", 1500))
    block_size = int(score_cfg.get("block_size", 1024))
    seeds = [int(value) for value in score_cfg.get("mauve_seeds", [0, 1, 2, 3, 4])]

    chord_encoder = args.mmd_encoder or str(score_cfg.get("mmd_encoder", "prompteol-coherence"))
    chord_ref = _load(root / "features" / chord_encoder / "reference.npy")
    dev, cal_pool, test_pool = _split_reference(chord_ref, int(cfg.get("seed", 0)), dev_size)
    n = min(n_eval, len(cal_pool) // 2, len(test_pool) // 2)
    if n < 2:
        raise ValueError("reference pool is too small for exchangeable calibration")
    sigma = median_bandwidth(dev, max_samples=2000, seed=int(cfg.get("seed", 0)))
    null = exchangeable_null(
        cal_pool,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng("real-dlm", "null", n),
    )
    null_mean, null_std = float(null.mean()), float(null.std())
    threshold = float(np.quantile(null, 0.95))
    transport = exchangeable_null(
        test_pool,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng("real-dlm", "transport", n),
    )
    calibration = [
        {
            "encoder": chord_encoder,
            "sigma": sigma,
            "n": n,
            "draws": draws,
            "null_mean": null_mean,
            "null_std": null_std,
            "q95": threshold,
            "transport_fpr": float(np.mean(transport > threshold)),
        }
    ]

    fixed_reference: Dict[str, np.ndarray] = {}
    for name in ("mauve-gpt2", "mauve-electra", "fbd-bert"):
        ref = _load(root / "features" / name / "reference.npy")
        rng = seeded_rng("real-dlm", name, "reference")
        fixed_reference[name] = ref[rng.choice(len(ref), size=min(n, len(ref)), replace=False)]

    rows: List[Dict[str, Any]] = []
    for run in runs:
        candidate = _load(root / "features" / chord_encoder / f"{run.run_id}.npy")
        if len(candidate) < n:
            raise ValueError(
                f"{run.run_id}: only {len(candidate)} samples; calibrated scoring "
                f"requires at least {n}"
            )
        run_n = n
        distances = exchangeable_attack(
            cal_pool,
            candidate,
            run_n,
            run_n,
            draws=draws,
            sigma=sigma,
            block_size=block_size,
            rng=seeded_rng("real-dlm", run.run_id, run_n),
        )
        z = (float(distances.mean()) - null_mean) / null_std if null_std > 0 else float("nan")
        base = {
            "run_id": run.run_id,
            "model": run.model,
            "family": run.family,
            "seed": run.seed,
            "setting": _setting(run),
            "n": run_n,
        }
        rows.append(
            {
                **base,
                "metric": "chord_mmd_z",
                "value": z,
                "ci_low": (float(np.quantile(distances, 0.025)) - null_mean) / null_std,
                "ci_high": (float(np.quantile(distances, 0.975)) - null_mean) / null_std,
                "higher_is_worse": True,
            }
        )
        for name, metric_name in (
            ("mauve-gpt2", "mauve"),
            ("mauve-electra", "mauve_electra"),
        ):
            candidate_features = _load(root / "features" / name / f"{run.run_id}.npy")
            take = min(len(fixed_reference[name]), len(candidate_features), run_n)
            values = mauve_scores(
                fixed_reference[name][:take],
                candidate_features[:take],
                num_buckets=max(2, take // 10),
                seeds=seeds,
            )
            rows.append(
                {
                    **base,
                    "metric": metric_name,
                    "value": values["mauve_mean"],
                    "ci_low": values["mauve_min"],
                    "ci_high": values["mauve_max"],
                    "higher_is_worse": False,
                }
            )
        fbd_candidate = _load(root / "features" / "fbd-bert" / f"{run.run_id}.npy")
        take = min(len(fixed_reference["fbd-bert"]), len(fbd_candidate), run_n)
        rows.append(
            {
                **base,
                "metric": "fbd_bert",
                "value": frechet_distance(fixed_reference["fbd-bert"][:take], fbd_candidate[:take]),
                "ci_low": "",
                "ci_high": "",
                "higher_is_worse": True,
            }
        )
        print(f"[real-dlm score] {run.run_id}: prompteol-z={z:.3f}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    write_csv(root / "scores" / f"distribution_metrics{suffix}.csv", rows)
    write_csv(root / "scores" / f"calibration{suffix}.csv", calibration)


if __name__ == "__main__":
    main()
