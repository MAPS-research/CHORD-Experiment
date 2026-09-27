"""Table 1 detection: S = z(harmful) - z(matched benign) per perturbation family and dose.

The per-condition z is the null-standardized RBF-MMD against the human reference:
the exchangeable permutation null (``nulls.py``) and the median-heuristic bandwidth
frozen on a dev split. On top of it we add the contrast against the matched benign
paraphrase control and a HIERARCHICAL CI that resamples at the PARENT-document
level: perturbed texts that share a clean parent are correlated, so a draw
resamples parents with replacement, then bootstraps n candidate rows, then computes
MMD against a freshly-resampled clean reference half.

    S_low > 0  -> SELECTIVE     (harmful moves the embedding more than benign)
    S_high < 0 -> ANTI-SELECTIVE (the documented frozen-encoder disease)
    CI spans 0 -> blind         (cannot tell harmful from benign)

NB the reused null/attack estimator is the BIASED RBF-MMD V-statistic
(``rbf_mmd(..., unbiased=False)``), as in every calibration path in this repo.
Detection validity is unaffected: the
bias cancels under null-standardization (null and attack use the same estimator),
and the clean FPR is reported as the operating-point anchor.

    python -m experiments.counterfactual_eval.selectivity \
        --config experiments/counterfactual_eval/configs/selectivity.yaml \
        --corpus-dir outputs/counterfactual/meta_eval
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from chord.metrics.distribution import median_bandwidth, rbf_mmd
from chord.utils.hashing import seeded_rng

from ..metric_utils.statistics import binomial_confidence_interval
from .nulls import exchangeable_null


def _load(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float64)


def _split(ref_emb, enc_name, corpus, *, dev_size):
    """dev (bandwidth) | cal-region | test-region — identical to score.py."""
    rng = seeded_rng(enc_name, corpus, "split")
    idx = rng.permutation(len(ref_emb))
    dev_i = idx[:dev_size]
    rest = idx[dev_size:]
    cut = len(rest) // 2
    return ref_emb[dev_i], ref_emb[rest[:cut]], ref_emb[rest[cut:]]


def _parent_ids(corpus_dir: Path, name: str) -> List[str]:
    """parent_id per row, aligned to the cached .npy (featurize preserves order)."""
    path = corpus_dir / "texts" / "conditions" / f"{name}.jsonl"
    return [json.loads(line).get("parent_id") for line in open(path)]


def _parent_groups(parent_ids: List[str]) -> List[np.ndarray]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, pid in enumerate(parent_ids):
        groups[pid if pid is not None else f"_row{i}"].append(i)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def hierarchical_attack(clean_pool, cand_emb, groups, n_ref, n, *, draws, sigma, block_size, rng):
    """2-level bootstrap MMD(clean ref half, candidate sample).

    Each draw: (1) resample PARENT groups with replacement, pool their rows;
    (2) sample n candidate rows from that parent-resampled pool; (3) resample a
    fresh clean reference half of size n_ref — matching the exchangeable null, so a
    clean candidate reproduces the null while a corrupted one lands above it. The
    parent-level resample propagates document-level correlation into the CI."""
    out = np.empty(draws, dtype=np.float64)
    n_groups = len(groups)
    for d in range(draws):
        chosen = rng.choice(n_groups, size=n_groups, replace=True)
        rows = np.concatenate([groups[c] for c in chosen])
        pick = rows[rng.choice(len(rows), size=n, replace=len(rows) < n)]
        b = cand_emb[pick]
        a = clean_pool[rng.choice(len(clean_pool), size=n_ref, replace=False)]
        out[d] = rbf_mmd(a, b, sigma, False, block_size)
    return out


@dataclass(frozen=True)
class ScoringParams:
    draws: int = 300
    n_eval: int = 1000
    dev_size: int = 1500
    block_size: int = 1024
    nominal_alpha: float = 0.05

    @classmethod
    def from_config(cls, cfg: Dict) -> "ScoringParams":
        return cls(
            draws=int(cfg.get("draws", 300)),
            n_eval=int(cfg.get("n_eval", 1000)),
            dev_size=int(cfg.get("dev_size", 1500)),
            block_size=int(cfg.get("block_size", 1024)),
            nominal_alpha=float(cfg.get("nominal_alpha", 0.05)),
        )


def score_features(
    feat_dir: Path,
    corpus_dir: Path,
    *,
    label: str,
    rng_name: str,
    corpus: str,
    benign: str,
    targets: List[str],
    params: ScoringParams,
    group_of: Dict[str, str],
    kind_of: Dict[str, str],
    wall_families=frozenset(),
) -> Tuple[Dict, List[Dict]]:
    """Calibration row and z / S rows of one feature directory.

    ``rng_name`` seeds the split, the nulls and every bootstrap; ``label`` only names
    the rows. Table 1 passes the encoder name for both. The layer-depth ablation
    scores every layer under one locked encoder's ``rng_name``, so the layer is the
    only variable."""
    draws, block_size = params.draws, params.block_size
    ref_emb = _load(feat_dir / "reference.npy")
    dev, cal_pool, test_pool = _split(ref_emb, rng_name, corpus, dev_size=params.dev_size)
    n = min(params.n_eval, len(cal_pool) // 2, len(test_pool) // 2)
    sigma = float(median_bandwidth(dev, max_samples=2000, seed=0))

    # ---- calibration: exchangeable null + within/transport clean FPR --------
    null_cal = exchangeable_null(
        cal_pool,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng(rng_name, corpus, "cal", n),
    )
    nmean, nstd = float(null_cal.mean()), float(null_cal.std())
    q95 = float(np.quantile(null_cal, 0.95))
    thr = float(np.quantile(null_cal, 1 - params.nominal_alpha))
    within = exchangeable_null(
        cal_pool,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng(rng_name, corpus, "calfpr", n),
    )
    trans = exchangeable_null(
        test_pool,
        n,
        n,
        draws=draws,
        sigma=sigma,
        block_size=block_size,
        rng=seeded_rng(rng_name, corpus, "test", n),
    )
    k_in, k_tr = int((within > thr).sum()), int((trans > thr).sum())
    calib = {
        "encoder": label,
        "sigma": round(sigma, 6),
        "n": n,
        "null_mean": nmean,
        "null_std": nstd,
        "q95": q95,
        "within_fpr": k_in / draws,
        "transport_fpr": k_tr / draws,
        "within_ci_low": binomial_confidence_interval(k_in, draws)[0],
        "within_ci_high": binomial_confidence_interval(k_in, draws)[1],
        "transport_ci_low": binomial_confidence_interval(k_tr, draws)[0],
        "transport_ci_high": binomial_confidence_interval(k_tr, draws)[1],
        "nominal_alpha": params.nominal_alpha,
    }
    print(
        f"[sel] {label}: sigma={sigma:.4f} n={n} FPR within={k_in / draws:.3f} "
        f"transport={k_tr / draws:.3f}",
        flush=True,
    )

    def z_hier(name):
        emb = _load(feat_dir / f"{name}.npy")
        groups = _parent_groups(_parent_ids(corpus_dir, name))
        D = hierarchical_attack(
            cal_pool,
            emb,
            groups,
            n,
            n,
            draws=draws,
            sigma=sigma,
            block_size=block_size,
            rng=seeded_rng(rng_name, corpus, "hier", name, n),
        )
        z = (float(D.mean()) - nmean) / nstd if nstd > 0 else float("nan")
        detected = bool(float(np.quantile(D, 0.025)) > q95)
        return D, z, detected

    rows: List[Dict] = []
    D_benign, z_benign, det_benign = z_hier(benign)
    # the anchor's own dose-response point (S = 0 by construction) so the
    # benign curve in dose_response includes its top dose.
    rows.append(
        {
            "encoder": label,
            "family": benign,
            "group": group_of.get(benign, ""),
            "is_wall": benign in wall_families,
            "z_harmful": round(z_benign, 4),
            "z_benign": round(z_benign, 4),
            "detected_harmful": det_benign,
            "selectivity": 0.0,
            "sel_ci_low": 0.0,
            "sel_ci_high": 0.0,
            "selective": False,
            "anti_selective": False,
            "kind": kind_of.get(benign, "benign"),
        }
    )
    for name in targets:
        D_fam, z_fam, det = z_hier(name)
        # hierarchical selectivity: difference of the two parent-resampled
        # bootstrap distributions, null-standardized.
        sel = (D_fam - D_benign) / nstd if nstd > 0 else np.full(draws, np.nan)
        s_mean = float(np.mean(sel))
        s_low, s_high = float(np.quantile(sel, 0.025)), float(np.quantile(sel, 0.975))
        rows.append(
            {
                "encoder": label,
                "family": name,
                "group": group_of.get(name, ""),
                "is_wall": name in wall_families,
                "z_harmful": round(z_fam, 4),
                "z_benign": round(z_benign, 4),
                "detected_harmful": det,
                "selectivity": round(s_mean, 4),
                "sel_ci_low": round(s_low, 4),
                "sel_ci_high": round(s_high, 4),
                "selective": bool(s_low > 0),
                "anti_selective": bool(s_high < 0),
                "kind": kind_of.get(name, ""),
            }
        )
        verdict = "SELECTIVE" if s_low > 0 else ("ANTI" if s_high < 0 else "n.s.")
        tag = "" if kind_of.get(name) == "harmful" else f" [{kind_of.get(name)}]"
        print(
            f"[sel]   {label}/{name}{tag}: z={z_fam:.1f} (benign {z_benign:.1f}) "
            f"S={s_mean:.2f} CI[{s_low:.2f},{s_high:.2f}] {verdict}",
            flush=True,
        )
    return calib, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument(
        "--out-suffix",
        default="",
        help="append to output filenames, e.g. '_moe' -> "
        "selectivity_by_family_moe.csv (non-destructive)",
    )
    parser.add_argument(
        "--only-encoders",
        default="",
        help="comma-separated encoder names: compute ONLY these and "
        "merge into the existing output CSVs, carrying every "
        "other encoder's rows over unchanged. Exactly equivalent "
        "to a full rerun (rows are per-encoder independent and "
        "RNGs are seeded per encoder), but minutes instead of "
        "hours when adding one column.",
    )
    parser.add_argument(
        "--only-families",
        default="",
        help="comma-separated target condition names: score ONLY these "
        "conditions (the benign anchor is always scored). Each "
        "condition's z is independent of the others (per-condition "
        "RNG), so this yields the same numbers as a full run for "
        "the listed conditions — used to score a costly high-dim "
        "token-space encoder on just the family under study.",
    )
    args = parser.parse_args()
    suffix = args.out_suffix
    cfg = yaml.safe_load(Path(args.config).read_text())
    corpus_dir = Path(args.corpus_dir)
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    corpus = manifest.get("corpus", corpus_dir.name)

    params = ScoringParams.from_config(cfg)
    wall_families = set(cfg.get("wall_families", []))

    # Score the harmful families AND the graded benign-dose conditions; both get
    # z + S vs the SINGLE benign ANCHOR (kind=benign). Scoring benign_dose lets
    # dose_response draw the benign curve, which should stay flat/low (S~0) while
    # harmful curves rise — the negative-control counterpart of the dose story.
    # kind=probe (meaning_probe scope-probe families) is scored with the SAME
    # z + S endpoint for comparability, but is deliberately NOT kind=harmful so
    # it never counts toward the paper's harmful-detection tally.
    conds = [
        c
        for c in manifest["conditions"]
        if c.get("kind") in ("harmful", "benign", "benign_dose", "probe")
    ]
    benign = next((c["name"] for c in conds if c.get("kind") == "benign"), None)
    if benign is None:
        raise ValueError("no benign condition (kind=benign) in manifest")
    targets = [c["name"] for c in conds if c.get("kind") in ("harmful", "benign_dose", "probe")]
    only_families = {f.strip() for f in args.only_families.split(",") if f.strip()}
    if only_families:
        missing_fams = only_families - set(targets)
        if missing_fams:
            raise SystemExit(f"--only-families: unknown conditions {sorted(missing_fams)}")
        targets = [t for t in targets if t in only_families]
        print(
            f"[sel] only-families: scoring {len(targets)} conditions {sorted(only_families)}",
            flush=True,
        )
    group_of = {c["name"]: c.get("group", "") for c in conds}
    kind_of = {c["name"]: c.get("kind", "") for c in conds}

    encoders = cfg.get("encoders") or [
        p.name for p in (corpus_dir / "feats").iterdir() if (p / "reference.npy").exists()
    ]

    out_dir = corpus_dir / "selectivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_rows: List[Dict] = []
    calib_rows: List[Dict] = []
    only = [e.strip() for e in args.only_encoders.split(",") if e.strip()]
    if only:
        missing = [e for e in only if not (corpus_dir / "feats" / e / "reference.npy").exists()]
        if missing:
            raise SystemExit(f"--only-encoders: no features for {missing}")
        encoders = only
        sel_rows = _carry_rows(out_dir / f"selectivity_by_family{suffix}.csv", set(only))
        calib_rows = _carry_rows(out_dir / f"calibration{suffix}.csv", set(only))
        print(
            f"[sel] incremental: computing {only}, carrying "
            f"{len({r['encoder'] for r in sel_rows})} existing encoder columns",
            flush=True,
        )

    for enc in encoders:
        feat_dir = corpus_dir / "feats" / enc
        if not (feat_dir / "reference.npy").exists():
            print(f"[sel] {enc}: no features, skip", flush=True)
            continue
        missing = [s for s in [benign] + targets if not (feat_dir / f"{s}.npy").exists()]
        # kind=probe conditions (meaning probe) are ADDITIVE and only featurized for the
        # encoders that report them: an encoder lacking probe feats keeps its 40-harmful
        # + benign scoring and simply drops the probe rows (2026-08-17). Every other
        # missing set is a hard skip, as before.
        missing_required = [s for s in missing if kind_of.get(s) != "probe"]
        if missing_required:
            print(
                f"[sel] {enc}: incomplete features (missing {missing_required[:3]}), skip",
                flush=True,
            )
            continue
        enc_targets = targets
        if missing:
            print(
                f"[sel] {enc}: no probe feats for {missing} -> probe rows dropped for this encoder",
                flush=True,
            )
            enc_targets = [t for t in targets if t not in missing]
        calib, rows = score_features(
            feat_dir,
            corpus_dir,
            label=enc,
            rng_name=enc,
            corpus=corpus,
            benign=benign,
            targets=enc_targets,
            params=params,
            group_of=group_of,
            kind_of=kind_of,
            wall_families=wall_families,
        )
        calib_rows.append(calib)
        sel_rows.extend(rows)
        _write_csv(out_dir / f"selectivity_by_family{suffix}.csv", sel_rows)
        _write_csv(out_dir / f"calibration{suffix}.csv", calib_rows)

    _write_csv(out_dir / f"selectivity_by_family{suffix}.csv", sel_rows)
    _write_csv(out_dir / f"calibration{suffix}.csv", calib_rows)
    print(f"[sel] COMPLETE -> {out_dir}", flush=True)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _carry_rows(path: Path, exclude_encoders) -> List[Dict]:
    """Rows of an existing output CSV whose encoder is NOT being recomputed
    (--only-encoders incremental mode). Values stay as strings; DictWriter
    round-trips them identically to a fresh full run."""
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["encoder"] not in exclude_encoders]


if __name__ == "__main__":
    main()
