"""Benign-contrast selectivity S for BASELINE metrics on the meta-eval set.

The paper's comparison columns (FBD / MAUVE's FI-KL / gen-PPL / entropy) do not
share the RBF-MMD ruler, but they CAN and MUST share the ENDPOINT, or the
comparison is apples-to-oranges. This module reuses the exact statistical frame
of ``selectivity.py`` — exchangeable clean-vs-clean null, null-standardized z,
matched-benign contrast S = z(harmful) - z(benign anchor), hierarchical
parent-level bootstrap CI — with the distance statistic swapped:

  fbd      Gaussian Frechet distance on cached encoder feats (set-level)
  fikl     MAUVE's k-means-KL frontier integral on cached feats (set-level;
           a [0,1] divergence, 0 for identical histograms — the MAUVE score
           itself is not used for this null-standardized endpoint)
  genppl   |Δ mean per-passage log-PPL| (scalar; reads genppl/<set>.npy from
           score_genppl.py's GPU half)
  entropy  |Δ mean per-passage unigram token entropy| (scalar; computed from
           texts on the fly, cached to entropy/<set>.npy)

Scalar statistics use the ABSOLUTE mean shift: corruption can push a scalar in
either direction (repetition LOWERS entropy and log-PPL; shuffling raises PPL
but leaves the unigram histogram untouched), and "detection" means moving away
from clean in either direction. The clean-vs-clean null folds the same way, so
z is comparable.

STRUCTURE-MATCHED SAMPLING (the FBD duplicate-bias fix). The naive scheme —
null = two disjoint clean halves, attack = parent-bootstrap + row-resample —
gives the attack side duplicate rows the null never has, and the duplicate RATE
depends on the condition's pool size (benign anchor ~990 rows vs harmful
conditions ~470). Fréchet is hyper-sensitive to duplicates (they shrink the
sample covariance), which inflated z for EVERY set (clean included) and,
asymmetrically, S for small-pool conditions: the one-shot FBD showed
sentence_permutation ~ at the null while the naive bootstrap called it
SELECTIVE. Fix: every set (clean_candidates, benign anchor, every target) is
first reduced to ONE ROW PER PARENT, deterministically subsampled to the
COMMON minimum parent count m, and each draw parent-bootstraps exactly m rows.
The z/detection anchor is clean_candidates pushed through the SAME machinery
(D_clean), so the duplicate structure is identical on all three sides and
cancels exactly in z and S.

    python -m experiments.counterfactual_eval.baseline_selectivity \
        --config experiments/counterfactual_eval/configs/baseline_selectivity.yaml \
        --corpus-dir outputs/counterfactual/meta_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import yaml

from chord.metrics.distribution import frechet_distance, median_bandwidth, rbf_mmd
from chord.utils.hashing import seeded_rng

from .featurize import _load_texts, _set_path, text_set_names
from .selectivity import _load, _parent_ids, _split, _write_csv


def one_per_parent(vals: np.ndarray, parent_ids: List[str]):
    """First row of each parent -> (sub_vals, sub_parent_ids)."""
    seen = set()
    keep: List[int] = []
    for i, pid in enumerate(parent_ids):
        key = pid if pid is not None else f"_row{i}"
        if key not in seen:
            seen.add(key)
            keep.append(i)
    idx = np.asarray(keep, dtype=np.int64)
    return vals[idx], [parent_ids[i] for i in keep]


def matched_attack(clean_pool, cand_m, n_ref, *, draws, stat, rng):
    """Structure-matched bootstrap: ``cand_m`` holds EXACTLY one row per parent,
    already subsampled to the block's common parent count m. Each draw
    parent-bootstraps m rows (with replacement) and draws a fresh clean
    reference half of n_ref rows. Because clean_candidates, the benign anchor,
    and every target pass through this with the SAME m, the duplicate-row
    structure is identical on all sides and cancels in z and S."""
    out = np.empty(draws, dtype=np.float64)
    m = len(cand_m)
    for d in range(draws):
        b = cand_m[rng.choice(m, size=m, replace=True)]
        a = clean_pool[rng.choice(len(clean_pool), size=n_ref, replace=False)]
        out[d] = stat(a, b)
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def _stat_fbd(a: np.ndarray, b: np.ndarray) -> float:
    return frechet_distance(a, b)


def _stat_fikl(a: np.ndarray, b: np.ndarray) -> float:
    # cheap-mode FI-KL (kmeans_num_redo=1); quantization noise is absorbed by
    # the null standardization (see mauve_eval.kmeans_kl_frontier docstring).
    from .mauve_eval import kmeans_kl_frontier

    nb = max(2, min(len(a), len(b)) // 10)
    return kmeans_kl_frontier(a, b, num_buckets=nb, seed=0)


def _stat_abs_mean_shift(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    return float(abs(a.mean() - b.mean()))


def _stat_mmd_chan(a: np.ndarray, b: np.ndarray) -> float:
    """MMD-BERT statistic per Chan et al. (LREC-COLING 2024): biased RBF-MMD
    with bandwidth = (median pairwise distance)/2, recomputed on the joint
    sample of every comparison (their per-set protocol — deliberately NOT the
    dev-frozen bandwidth of our ruler). The null standardization downstream is
    the shared endpoint frame, same as every other baseline lane."""
    joint = np.concatenate([a, b], axis=0)
    sigma = 0.5 * float(median_bandwidth(joint, max_samples=2000, seed=0))
    if not sigma > 0:
        return float("nan")
    return float(rbf_mmd(a, b, sigma, unbiased=False))


STATS: Dict[str, Callable] = {
    "fbd": _stat_fbd,
    "fikl": _stat_fikl,
    "abs_mean_shift": _stat_abs_mean_shift,
    "mmd_chan": _stat_mmd_chan,
}


# --------------------------------------------------------------------------
# scalar-lane value caches
# --------------------------------------------------------------------------


def passage_entropy(text: str) -> float:
    """Unigram token entropy (bits) of one passage — whitespace-lowercase tokens,
    the lexical.py convention. Order-invariant BY CONSTRUCTION: shuffling
    sentences/words leaves it untouched (that blindness is the point of the
    baseline); repetition collapses it."""
    toks = text.lower().split()
    if not toks:
        return 0.0
    _, counts = np.unique(np.asarray(toks, dtype=object), return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    return float(-(p * np.log2(p)).sum())


def ensure_entropy_cache(corpus_dir: Path, sets: List[str]) -> Path:
    """Compute + cache per-passage entropy for every text set (CPU, fast)."""
    out_dir = corpus_dir / "entropy"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in sets:
        outp = out_dir / f"{s}.npy"
        if outp.exists():
            continue
        texts = _load_texts(_set_path(corpus_dir / "texts", s))
        np.save(outp, np.asarray([passage_entropy(t) for t in texts], dtype=np.float64))
        print(f"[base] entropy cache {s}: n={len(texts)}", flush=True)
    return out_dir


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--out-suffix", default="")
    parser.add_argument(
        "--only-statistic", default=None, help="run only this statistic block (for job splitting)"
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    corpus_dir = Path(args.corpus_dir)
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    corpus = manifest.get("corpus", corpus_dir.name)

    n_eval = int(cfg.get("n_eval", 1000))
    dev_size = int(cfg.get("dev_size", 1500))

    # kind=probe (meaning_probe scope-probe families) shares the z + S endpoint
    # for comparability but never counts toward the harmful-detection tally
    # (mirrors the selectivity.py convention).
    conds = [
        c
        for c in manifest["conditions"]
        if c.get("kind") in ("harmful", "benign", "benign_dose", "probe")
    ]
    benign = next((c["name"] for c in conds if c.get("kind") == "benign"), None)
    if benign is None:
        raise ValueError("no benign condition (kind=benign) in manifest")
    targets = [c["name"] for c in conds if c.get("kind") in ("harmful", "benign_dose", "probe")]
    group_of = {c["name"]: c.get("group", "") for c in conds}
    kind_of = {c["name"]: c.get("kind", "") for c in conds}

    out_dir = corpus_dir / "selectivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_rows: List[Dict] = []
    calib_rows: List[Dict] = []

    def run_block(stat_label: str, source_label: str, load_set, stat, draws: int):
        """One (statistic, feature-source) selectivity pass. load_set(name) must
        return the per-row array for a set name ('reference', condition names)."""
        ref = load_set("reference")
        dev, cal_pool, test_pool = _split(
            ref, f"{stat_label}:{source_label}", corpus, dev_size=dev_size
        )
        n_ref = min(n_eval, len(cal_pool) // 2, len(test_pool) // 2)
        rng_key = (stat_label, source_label, corpus)

        # ---- structure-matched pools: one row per parent, common size m ------
        all_sets = ["clean_candidates", benign] + targets
        pools: Dict[str, np.ndarray] = {}
        for name in all_sets:
            vals, pids = one_per_parent(load_set(name), _parent_ids(corpus_dir, name))
            pools[name] = vals
        m = min(len(v) for v in pools.values())
        for name, vals in pools.items():
            if len(vals) > m:
                sub = seeded_rng(*rng_key, "sub", name).choice(len(vals), size=m, replace=False)
                pools[name] = vals[np.sort(sub)]

        # ---- matched null anchor: clean_candidates through the SAME machinery -
        D_clean = matched_attack(
            cal_pool,
            pools["clean_candidates"],
            n_ref,
            draws=draws,
            stat=stat,
            rng=seeded_rng(*rng_key, "hier", "clean_candidates", m),
        )
        nmean, nstd = float(D_clean.mean()), float(D_clean.std())
        q975 = float(np.quantile(D_clean, 0.975))
        calib_rows.append(
            {
                "statistic": stat_label,
                "source": source_label,
                "n_ref": n_ref,
                "m": m,
                "draws": draws,
                "null_mean": nmean,
                "null_std": nstd,
                "q975": q975,
            }
        )
        print(
            f"[base] {stat_label}/{source_label}: n_ref={n_ref} m={m} "
            f"draws={draws} D_clean={nmean:.4g}±{nstd:.4g}",
            flush=True,
        )

        def z_hier(name):
            D = matched_attack(
                cal_pool,
                pools[name],
                n_ref,
                draws=draws,
                stat=stat,
                rng=seeded_rng(*rng_key, "hier", name, m),
            )
            z = (float(D.mean()) - nmean) / nstd if nstd > 0 else float("nan")
            detected = bool(float(np.quantile(D, 0.025)) > q975)
            return D, z, detected

        D_benign, z_benign, det_b = z_hier(benign)
        sel_rows.append(
            {
                "statistic": stat_label,
                "source": source_label,
                "family": benign,
                "group": group_of.get(benign, ""),
                "kind": kind_of.get(benign, "benign"),
                "z_harmful": round(z_benign, 4),
                "z_benign": round(z_benign, 4),
                "detected_harmful": det_b,
                "selectivity": 0.0,
                "sel_ci_low": 0.0,
                "sel_ci_high": 0.0,
                "selective": False,
                "anti_selective": False,
            }
        )
        for name in targets:
            D_fam, z_fam, det = z_hier(name)
            sel = (D_fam - D_benign) / nstd if nstd > 0 else np.full(draws, np.nan)
            s_mean = float(np.mean(sel))
            s_low, s_high = float(np.quantile(sel, 0.025)), float(np.quantile(sel, 0.975))
            sel_rows.append(
                {
                    "statistic": stat_label,
                    "source": source_label,
                    "family": name,
                    "group": group_of.get(name, ""),
                    "kind": kind_of.get(name, ""),
                    "z_harmful": round(z_fam, 4),
                    "z_benign": round(z_benign, 4),
                    "detected_harmful": det,
                    "selectivity": round(s_mean, 4),
                    "sel_ci_low": round(s_low, 4),
                    "sel_ci_high": round(s_high, 4),
                    "selective": bool(s_low > 0),
                    "anti_selective": bool(s_high < 0),
                }
            )
            verdict = "SELECTIVE" if s_low > 0 else ("ANTI" if s_high < 0 else "n.s.")
            tag = "" if kind_of.get(name) == "harmful" else f" [{kind_of.get(name)}]"
            print(
                f"[base]   {stat_label}/{source_label}/{name}{tag}: z={z_fam:.1f} "
                f"(benign {z_benign:.1f}) S={s_mean:.2f} CI[{s_low:.2f},{s_high:.2f}] "
                f"{verdict}",
                flush=True,
            )
        # incremental write so a long fikl pass is inspectable mid-run
        _write_csv(out_dir / f"baseline_selectivity{args.out_suffix}.csv", sel_rows)
        _write_csv(out_dir / f"baseline_calibration{args.out_suffix}.csv", calib_rows)

    for block in cfg.get("statistics", []):
        name = block["name"]
        if args.only_statistic and name != args.only_statistic:
            continue
        draws = int(block.get("draws", 200))
        lane = block.get("lane", "feats")
        if lane == "feats":
            stat = STATS[block.get("stat", name)]
            for enc in block.get("encoders", []):
                feat_dir = corpus_dir / "feats" / enc
                needed = ["reference", "clean_candidates", benign] + targets
                missing = [s for s in needed if not (feat_dir / f"{s}.npy").exists()]
                if missing:
                    print(
                        f"[base] {name}/{enc}: incomplete feats (missing {missing[:3]}), skip",
                        flush=True,
                    )
                    continue
                run_block(name, enc, lambda s, d=feat_dir: _load(d / f"{s}.npy"), stat, draws)
        elif lane == "scalar":
            vdir_name = block.get("values_dir", name)
            if vdir_name == "entropy":
                vdir = ensure_entropy_cache(corpus_dir, text_set_names(manifest))
            else:
                vdir = corpus_dir / vdir_name
            needed = ["reference", "clean_candidates", benign] + targets
            missing = [s for s in needed if not (vdir / f"{s}.npy").exists()]
            if missing:
                print(
                    f"[base] {name}: missing values {missing[:3]} in {vdir}, skip "
                    f"(run the GPU half first)",
                    flush=True,
                )
                continue
            run_block(
                name,
                vdir_name,
                lambda s, d=vdir: _load(d / f"{s}.npy"),
                STATS["abs_mean_shift"],
                draws,
            )
        else:
            raise ValueError(f"unknown lane: {lane}")

    _write_csv(out_dir / f"baseline_selectivity{args.out_suffix}.csv", sel_rows)
    _write_csv(out_dir / f"baseline_calibration{args.out_suffix}.csv", calib_rows)
    print(f"[base] COMPLETE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
