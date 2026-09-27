"""MAUVE (and RBF-MMD z) for every safety-transfer condition on GPT-2-large features.

Backs the appendix sentence "MAUVE also does not track unsafe prevalence
consistently". For each encoder: the reference is split into dev | calibration |
test exactly as in ``experiments.counterfactual_eval.selectivity``; a fixed calibration
sample P is compared with every condition by MAUVE (num_buckets = n/10, scaling
factor 5, several k-means seeds, via ``experiments.counterfactual_eval.mauve_eval``), and
the same condition also gets the null-standardized RBF-MMD z. MAUVE has no null,
so a condition counts as detected when its drop from the clean-candidate MAUVE
exceeds 1.645 pooled seed standard deviations.

Outputs ``<corpus>/scores/{scores_all,calibration}.csv``.

    python -m experiments.safety_transfer.mauve_scores \
        --config experiments/safety_transfer/configs/mauve.yaml \
        --corpus-dir outputs/experiments/safety_transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from chord.metrics.distribution import median_bandwidth
from chord.utils.hashing import seeded_rng

from ..counterfactual_eval.mauve_eval import mauve_scores
from ..counterfactual_eval.nulls import exchangeable_attack, exchangeable_null
from ..counterfactual_eval.selectivity import _load, _split, _write_csv
from ..metric_utils.statistics import binomial_confidence_interval


def z_row(clean_pool, cond_emb, n, draws, sigma, block_size, enc, corpus, name, nmean, nstd, q95):
    dist = exchangeable_attack(
        clean_pool,
        cond_emb,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng(enc, corpus, "cond", name, n),
    )
    ci_low = float(np.quantile(dist, 0.025))
    z = (float(dist.mean()) - nmean) / nstd if nstd > 0 else float("nan")
    return {
        "z": z,
        "D_mean": float(dist.mean()),
        "D_ci_low": ci_low,
        "D_ci_high": float(np.quantile(dist, 0.975)),
        "detected": bool(ci_low > q95),
    }


def mauve_detected(mc, mc_std, mv, mv_std, z_thr: float = 1.645) -> bool:
    """Drop from the clean MAUVE larger than z_thr pooled k-means seed std."""
    pooled = (mc_std**2 + mv_std**2) ** 0.5
    return bool((mc - mv) > z_thr * max(pooled, 1e-9))


def score_row(enc, name, meta, *, z, mauve, mauve_clean, mauve_clean_std, n, nb) -> Dict:
    if mauve_clean:
        drop = 100 * (mauve_clean - mauve["mauve_mean"]) / mauve_clean
    else:
        drop = float("nan")
    return {
        "encoder": enc,
        "condition": name,
        "experiment": meta.get("experiment", ""),
        "dose": meta.get("dose", ""),
        "n": n,
        "z": round(z["z"], 4),
        "detected": z["detected"],
        "D_mean": z["D_mean"],
        "D_ci_low": z["D_ci_low"],
        "mauve_mean": round(mauve["mauve_mean"], 5),
        "mauve_std": round(mauve["mauve_std"], 5),
        "mauve_drop_pct": round(drop, 3),
        "mauve_detected": mauve_detected(
            mauve_clean, mauve_clean_std, mauve["mauve_mean"], mauve["mauve_std"]
        ),
        "fi_kl_mean": round(mauve["fi_kl_mean"], 5),
        "num_buckets": nb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-dir", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    corpus_dir = Path(args.corpus_dir)
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    cond_meta = {c["name"]: c for c in manifest["conditions"]}
    corpus = manifest.get("corpus", corpus_dir.name)

    draws = int(cfg.get("draws", 200))
    n_eval = int(cfg.get("n_eval", 1000))
    dev_size = int(cfg.get("dev_size", 1500))
    block_size = int(cfg.get("block_size", 1024))
    alpha = float(cfg.get("nominal_alpha", 0.05))
    seeds = list(cfg.get("mauve_seeds", [0, 1, 2, 3, 4]))
    scaling = float(cfg.get("mauve_scaling_factor", 5.0))
    encoders = cfg.get("encoders") or [p.name for p in (corpus_dir / "feats").iterdir()]

    out_dir = corpus_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    score_rows: List[Dict] = []
    calib_rows: List[Dict] = []
    set_names = [c["name"] for c in manifest["conditions"] if c.get("experiment") != "control"]

    for enc in encoders:
        feat_dir = corpus_dir / "feats" / enc
        missing = [
            s
            for s in ["reference", "clean_candidates"] + set_names
            if not (feat_dir / f"{s}.npy").exists()
        ]
        if missing:
            print(f"[mauve] {enc}: incomplete features (missing {missing[:3]}), skip", flush=True)
            continue
        ref_emb = _load(feat_dir / "reference.npy")
        dev, cal_pool, test_pool = _split(ref_emb, enc, corpus, dev_size=dev_size)
        n = min(n_eval, len(cal_pool) // 2, len(test_pool) // 2)
        sigma = float(median_bandwidth(dev, max_samples=2000, seed=0))

        def null(pool, tag, e=enc, nn=n, s=sigma):
            return exchangeable_null(
                pool,
                nn,
                nn,
                draws=draws,
                sigma=s,
                block_size=block_size,
                rng=seeded_rng(e, corpus, tag, nn),
            )

        null_cal = null(cal_pool, "cal")
        nmean, nstd = float(null_cal.mean()), float(null_cal.std())
        q95 = float(np.quantile(null_cal, 0.95))
        thr = float(np.quantile(null_cal, 1 - alpha))
        within, trans = null(cal_pool, "calfpr"), null(test_pool, "test")
        k_in, k_tr = int((within > thr).sum()), int((trans > thr).sum())
        ci_in = binomial_confidence_interval(k_in, draws)
        ci_tr = binomial_confidence_interval(k_tr, draws)
        calib_rows.append(
            {
                "encoder": enc,
                "sigma": round(sigma, 6),
                "n": n,
                "null_mean": nmean,
                "null_std": nstd,
                "q95": q95,
                "within_fpr": k_in / draws,
                "within_ci_low": ci_in[0],
                "within_ci_high": ci_in[1],
                "transport_fpr": k_tr / draws,
                "transport_ci_low": ci_tr[0],
                "transport_ci_high": ci_tr[1],
                "nominal_alpha": alpha,
            }
        )

        p_ref = cal_pool[seeded_rng(enc, corpus, "P").choice(len(cal_pool), size=n, replace=False)]
        clean_cand = _load(feat_dir / "clean_candidates.npy")
        nb = max(2, min(len(p_ref), len(clean_cand)) // 10)
        clean = mauve_scores(p_ref, clean_cand, num_buckets=nb, scaling=scaling, seeds=seeds)
        mc, mc_std = clean["mauve_mean"], clean["mauve_std"]
        print(f"[mauve] {enc}: clean MAUVE={mc:.4f}±{mc_std:.4f} (nb={nb})", flush=True)
        zc = z_row(
            cal_pool,
            clean_cand,
            n,
            draws,
            sigma,
            block_size,
            enc,
            corpus,
            "clean_candidates",
            nmean,
            nstd,
            q95,
        )
        score_rows.append(
            score_row(
                enc,
                "clean_candidates",
                cond_meta.get("clean_candidates", {}),
                z=zc,
                mauve=clean,
                mauve_clean=mc,
                mauve_clean_std=mc_std,
                n=n,
                nb=nb,
            )
        )

        for s in set_names:
            cond_emb = _load(feat_dir / f"{s}.npy")
            zinfo = z_row(
                cal_pool, cond_emb, n, draws, sigma, block_size, enc, corpus, s, nmean, nstd, q95
            )
            nb_c = max(2, min(len(p_ref), len(cond_emb)) // 10)
            mv = mauve_scores(p_ref, cond_emb, num_buckets=nb_c, scaling=scaling, seeds=seeds)
            score_rows.append(
                score_row(
                    enc,
                    s,
                    cond_meta.get(s, {}),
                    z=zinfo,
                    mauve=mv,
                    mauve_clean=mc,
                    mauve_clean_std=mc_std,
                    n=n,
                    nb=nb_c,
                )
            )
            print(f"[mauve]   {s}: z={zinfo['z']:.1f} MAUVE={mv['mauve_mean']:.4f}", flush=True)
        _write_csv(out_dir / "scores_all.csv", score_rows)
        _write_csv(out_dir / "calibration.csv", calib_rows)

    _write_csv(out_dir / "scores_all.csv", score_rows)
    _write_csv(out_dir / "calibration.csv", calib_rows)
    print(f"[mauve] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
