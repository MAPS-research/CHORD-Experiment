"""Paired unsafe-vs-matched-safe contrast for the CHORD rows of the safety table.

Produces the CHORD rows of the table "Selectivity to unsafe-content prevalence":
Delta z = z_unsafe - z_safe at each replacement rate, with a paired bootstrap CI.

``experiments.counterfactual_eval.selectivity`` scores every condition against one benign
anchor. Here each rate has a control arm built from the same slots, length bins
and prompts, and the quantity of interest is the difference between the two at
the same rate. All statistics are those of ``selectivity``: the same
dev/calibration/test split, dev-fitted median bandwidth, exchangeable
clean-vs-clean null and null standardization. The only addition is pairing:
within a draw, the parent bootstrap, the candidate rows and the reference half
are drawn once and applied to every arm. The arms are row-aligned by
construction, so a paired draw differs between arms only in the injected
responses. Each arm's marginal distribution is unchanged, so its z column equals
a ``selectivity`` run.

Outputs ``<corpus>/selectivity/safety_{arms,contrast,calibration}<suffix>.csv``.

    python -m experiments.safety_transfer.paired_contrast \
        --config experiments/safety_transfer/configs/paired_contrast.yaml \
        --corpus-dir outputs/experiments/safety_transfer

With 5,120-d features and n = 1,000 one readout takes several CPU hours; run one
readout per job with ``--only-encoders <name> --out-suffix _<name>`` and join
the parts with ``merge_contrast_parts``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from chord.metrics.distribution import median_bandwidth, rbf_mmd
from chord.utils.hashing import seeded_rng

from ..counterfactual_eval.nulls import exchangeable_null
from ..counterfactual_eval.selectivity import (
    _load,
    _parent_groups,
    _parent_ids,
    _split,
    _write_csv,
)
from ..metric_utils.statistics import binomial_confidence_interval


def paired_attack(
    clean_pool, arms: Dict[str, np.ndarray], groups, n_ref, n, *, draws, sigma, block_size, rng
) -> Dict[str, np.ndarray]:
    """``selectivity.hierarchical_attack`` run jointly over row-aligned arms: the
    parent bootstrap, the row selection and the clean reference half are drawn
    once per draw and reused for every arm."""
    out = {k: np.empty(draws, dtype=np.float64) for k in arms}
    n_groups = len(groups)
    for d in range(draws):
        chosen = rng.choice(n_groups, size=n_groups, replace=True)
        rows = np.concatenate([groups[c] for c in chosen])
        pick = rows[rng.choice(len(rows), size=n, replace=len(rows) < n)]
        a = clean_pool[rng.choice(len(clean_pool), size=n_ref, replace=False)]
        for name, emb in arms.items():
            out[name][d] = rbf_mmd(a, emb[pick], sigma, False, block_size)
    return out


def calibrate(enc, corpus, cal_pool, test_pool, n, *, draws, sigma, block_size, alpha) -> Dict:
    """Clean null on the calibration pool plus within/transport false-positive rates."""
    null_cal = exchangeable_null(
        cal_pool,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng(enc, corpus, "cal", n),
    )
    thr = float(np.quantile(null_cal, 1 - alpha))
    within = exchangeable_null(
        cal_pool,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng(enc, corpus, "calfpr", n),
    )
    trans = exchangeable_null(
        test_pool,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng(enc, corpus, "test", n),
    )
    k_in, k_tr = int((within > thr).sum()), int((trans > thr).sum())
    ci_in = binomial_confidence_interval(k_in, draws)
    ci_tr = binomial_confidence_interval(k_tr, draws)
    return {
        "encoder": enc,
        "sigma": round(sigma, 6),
        "n": n,
        "draws": draws,
        "null_mean": float(null_cal.mean()),
        "null_std": float(null_cal.std()),
        "q95": float(np.quantile(null_cal, 0.95)),
        "within_fpr": k_in / draws,
        "transport_fpr": k_tr / draws,
        "within_ci_low": ci_in[0],
        "within_ci_high": ci_in[1],
        "transport_ci_low": ci_tr[0],
        "transport_ci_high": ci_tr[1],
        "nominal_alpha": alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--only-encoders", default="", help="comma list (default: all)")
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    corpus_dir = Path(args.corpus_dir)
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    corpus = manifest.get("corpus", corpus_dir.name)

    draws = int(cfg.get("draws", 200))
    n_eval = int(cfg.get("n_eval", 1000))
    dev_size = int(cfg.get("dev_size", 1500))
    block_size = int(cfg.get("block_size", 1024))
    alpha = float(cfg.get("nominal_alpha", 0.05))
    harmful_arm = cfg.get("harmful_arm", "unsafe")
    control_arms = list(cfg.get("control_arms", ["safe", "ood"]))

    conds = manifest["conditions"]
    anchor = next(c["name"] for c in conds if c.get("kind") == "benign")
    by_dose: Dict[float, Dict[str, str]] = defaultdict(dict)
    for c in conds:
        if c.get("arm") and c["arm"] != "clean":
            by_dose[c["dose"]][c["arm"]] = c["name"]
    doses = sorted(by_dose)

    encoders = [e.strip() for e in args.only_encoders.split(",") if e.strip()] or sorted(
        p.name for p in (corpus_dir / "feats").iterdir() if (p / "reference.npy").exists()
    )

    out_dir = corpus_dir / "selectivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_rows: List[Dict] = []
    contrast_rows: List[Dict] = []
    calib_rows: List[Dict] = []

    def write_all() -> None:
        _write_csv(out_dir / f"safety_arms{args.out_suffix}.csv", arm_rows)
        _write_csv(out_dir / f"safety_contrast{args.out_suffix}.csv", contrast_rows)
        _write_csv(out_dir / f"safety_calibration{args.out_suffix}.csv", calib_rows)

    for enc in encoders:
        feat_dir = corpus_dir / "feats" / enc
        need = ["reference", anchor] + [n for d in doses for n in by_dose[d].values()]
        missing = [s for s in need if not (feat_dir / f"{s}.npy").exists()]
        if missing:
            print(f"[safety] {enc}: incomplete feats (missing {missing[:3]}), skip", flush=True)
            continue

        ref_emb = _load(feat_dir / "reference.npy")
        dev, cal_pool, test_pool = _split(ref_emb, enc, corpus, dev_size=dev_size)
        n = min(n_eval, len(cal_pool) // 2, len(test_pool) // 2)
        sigma = float(median_bandwidth(dev, max_samples=2000, seed=0))
        calib = calibrate(
            enc,
            corpus,
            cal_pool,
            test_pool,
            n,
            draws=draws,
            sigma=sigma,
            block_size=block_size,
            alpha=alpha,
        )
        calib_rows.append(calib)
        nmean, nstd, q95 = calib["null_mean"], calib["null_std"], calib["q95"]
        print(
            f"[safety] {enc}: sigma={sigma:.4f} n={n} FPR within={calib['within_fpr']:.3f} "
            f"transport={calib['transport_fpr']:.3f}",
            flush=True,
        )

        groups = _parent_groups(_parent_ids(corpus_dir, anchor))
        d_anchor = paired_attack(
            cal_pool,
            {"clean": _load(feat_dir / f"{anchor}.npy")},
            groups,
            n,
            n,
            draws=draws,
            sigma=sigma,
            block_size=block_size,
            rng=seeded_rng(enc, corpus, "paired", anchor, n),
        )["clean"]
        z_anchor = (float(d_anchor.mean()) - nmean) / nstd if nstd > 0 else float("nan")
        arm_rows.append(
            {
                "encoder": enc,
                "arm": "clean",
                "dose": 0.0,
                "condition": anchor,
                "n_injected": 0,
                "z": round(z_anchor, 4),
                "detected": bool(float(np.quantile(d_anchor, 0.025)) > q95),
                "mmd_mean": float(d_anchor.mean()),
            }
        )

        for dose in doses:
            names = by_dose[dose]
            arms = {a: _load(feat_dir / f"{names[a]}.npy") for a in names}
            g = _parent_groups(_parent_ids(corpus_dir, names[harmful_arm]))
            draws_by_arm = paired_attack(
                cal_pool,
                arms,
                g,
                n,
                n,
                draws=draws,
                sigma=sigma,
                block_size=block_size,
                rng=seeded_rng(enc, corpus, "paired", f"p{dose}", n),
            )
            n_inj = next(c["n_injected"] for c in conds if c["name"] == names[harmful_arm])
            zs = {}
            for a, d_arm in draws_by_arm.items():
                zs[a] = (float(d_arm.mean()) - nmean) / nstd if nstd > 0 else float("nan")
                arm_rows.append(
                    {
                        "encoder": enc,
                        "arm": a,
                        "dose": dose,
                        "condition": names[a],
                        "n_injected": n_inj,
                        "z": round(zs[a], 4),
                        "detected": bool(float(np.quantile(d_arm, 0.025)) > q95),
                        "mmd_mean": float(d_arm.mean()),
                    }
                )
            for ctrl in control_arms:
                if ctrl not in draws_by_arm:
                    continue
                d_harm = draws_by_arm[harmful_arm]
                if nstd > 0:
                    delta = (d_harm - draws_by_arm[ctrl]) / nstd
                else:
                    delta = np.full(draws, np.nan)
                lo, hi = float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))
                contrast_rows.append(
                    {
                        "encoder": enc,
                        "dose": dose,
                        "n_injected": n_inj,
                        "harmful": names[harmful_arm],
                        "control": names[ctrl],
                        "control_arm": ctrl,
                        "z_harmful": round(zs[harmful_arm], 4),
                        "z_control": round(zs[ctrl], 4),
                        "delta_z": round(float(delta.mean()), 4),
                        "ci_low": round(lo, 4),
                        "ci_high": round(hi, 4),
                        "separated": bool(lo > 0),
                        "detected_harmful": bool(float(np.quantile(d_harm, 0.025)) > q95),
                    }
                )
                print(
                    f"[safety]   {enc} p={dose:.2f}: dz(vs {ctrl})={delta.mean():.2f} "
                    f"CI[{lo:.2f},{hi:.2f}] {'separated' if lo > 0 else 'n.s.'}",
                    flush=True,
                )
        write_all()  # after each readout, so a long run is inspectable

    write_all()
    print(f"[safety] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
