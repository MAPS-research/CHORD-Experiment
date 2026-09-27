"""Prompt-sweep selection WITHOUT tuning on the reported evaluation set.

Picking the readout template or ensemble from full evaluation-set selectivity
would tune on the reported test set. This module
implements the guardrail: split the 500 shared parents into a SELECTION half
(even trailing parent index) and a CONFIRMATION half (odd), score every prompt
column on both halves with the standard machinery (same dev-frozen bandwidth,
same exchangeable null, same parent-level hierarchical bootstrap — the clean
reference pool is shared between halves; only CONDITION rows are split), then:

  - the RANKING (and any ensemble composition choice) reads ONLY the selection
    half;
  - the reported numbers for the chosen prompt/ensemble come from the
    confirmation half (and the full evaluation-set table is an appendix ablation).

Mechanical winner rule (encoded here, not eyeballed):
  1. DISQUALIFY any prompt whose benign paraphrase turns selective on the
     selection half (fires on meaning-preserving rewrites);
  2. wall guard: the order-group score must stay >= 0.8x the best prompt's
     (order is the standing capacity wall — do not trade it away silently);
  3. rank the qualified by mean per-condition normalized S over harmful
     conditions (S_ec / max_e' S_e'c, clipped at 0).

    python -m experiments.counterfactual_eval.prompt_split_eval \
        --corpus-dir outputs/counterfactual/meta_eval --draws 80
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from chord.metrics.distribution import median_bandwidth
from chord.utils.hashing import seeded_rng

from .nulls import exchangeable_null
from .selectivity import _load, _parent_groups, _parent_ids, _split, _write_csv, hierarchical_attack

ORDER_GROUPS = {"order"}  # the wall families' manifest group
WALL_GUARD = 0.8  # min fraction of the best order score


def parent_half(pid: str) -> str:
    """Deterministic selection/confirmation split by trailing parent index
    (mev-<domain>-<n> -> parity of n; domain-balanced by construction)."""
    try:
        n = int(str(pid).rsplit("-", 1)[1])
    except (ValueError, IndexError):
        n = hash(str(pid))
    return "selection" if n % 2 == 0 else "confirmation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument(
        "--encoders",
        nargs="*",
        default=None,
        help="default: q27b-* and q35-* prompt features plus the adopted coherence encoder",
    )
    parser.add_argument("--draws", type=int, default=80)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--dev-size", type=int, default=1500)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--out-suffix", default="")
    parser.add_argument(
        "--feats-root",
        default=None,
        help="feature root (default <corpus-dir>/feats); e.g. the "
        "feats_layergrid/ sibling written by featurize_layer_grid",
    )
    parser.add_argument(
        "--split-key",
        default=None,
        help="fix the reference dev/cal/test partition + null/bootstrap "
        "seeds to this key for EVERY encoder (default: the encoder "
        "name itself, as in selectivity.py). Use one fixed key when "
        "comparing read depths of one backbone so only the layer varies.",
    )
    args = parser.parse_args()
    corpus_dir = Path(args.corpus_dir)
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    corpus = manifest.get("corpus", corpus_dir.name)

    conds = [
        c for c in manifest["conditions"] if c.get("kind") in ("harmful", "benign", "benign_dose")
    ]
    benign = next(c["name"] for c in conds if c.get("kind") == "benign")
    targets = [c["name"] for c in conds if c.get("kind") in ("harmful", "benign_dose")]
    group_of = {c["name"]: c.get("group", "") for c in conds}
    kind_of = {c["name"]: c.get("kind", "") for c in conds}

    feats_root = Path(args.feats_root) if args.feats_root else corpus_dir / "feats"
    if args.encoders:
        encoders = args.encoders
    else:
        encoders = sorted(
            p.name
            for p in feats_root.iterdir()
            if (p / "reference.npy").exists()
            and (
                p.name.startswith(("q27b-", "q35-"))
                or p.name
                in {"qwen3-27b-prompteol-coherence-l62", "qwen35-27b-prompteol-coherence-l62"}
            )
        )
    print(f"[split] encoders: {encoders}", flush=True)

    rows: List[Dict] = []
    for enc in encoders:
        feat_dir = feats_root / enc
        missing = [
            s for s in [benign] + targets + ["reference"] if not (feat_dir / f"{s}.npy").exists()
        ]
        if missing:
            print(f"[split] {enc}: incomplete ({missing[:3]}), skip", flush=True)
            continue
        ref_emb = _load(feat_dir / "reference.npy")
        # IDENTICAL split + bandwidth keys to selectivity.py, so z here is the
        # same quantity restricted to half the parents. --split-key overrides the
        # seed key (layer sweeps: one partition for every read depth).
        key = args.split_key or enc
        dev, cal_pool, test_pool = _split(ref_emb, key, corpus, dev_size=args.dev_size)
        n_full = min(args.n_eval, len(cal_pool) // 2, len(test_pool) // 2)
        sigma = float(median_bandwidth(dev, max_samples=2000, seed=0))
        null = exchangeable_null(
            cal_pool,
            n_full,
            n_full,
            draws=args.draws,
            sigma=sigma,
            block_size=args.block_size,
            rng=seeded_rng(key, corpus, "splitnull", n_full),
        )
        nmean, nstd = float(null.mean()), float(null.std())

        def z_half(name, half):
            emb = _load(feat_dir / f"{name}.npy")
            pids = _parent_ids(corpus_dir, name)
            keep = np.asarray(
                [i for i, p in enumerate(pids) if parent_half(p) == half], dtype=np.int64
            )
            if len(keep) == 0:
                return None, float("nan")
            groups = _parent_groups([pids[i] for i in keep])
            sub = emb[keep]
            n_c = min(n_full, len(sub))
            D = hierarchical_attack(
                cal_pool,
                sub,
                groups,
                n_full,
                n_c,
                draws=args.draws,
                sigma=sigma,
                block_size=args.block_size,
                rng=seeded_rng(key, corpus, "split", half, name, n_c),
            )
            z = (float(D.mean()) - nmean) / nstd if nstd > 0 else float("nan")
            return D, z

        for half in ("selection", "confirmation"):
            D_b, z_b = z_half(benign, half)
            for name in targets:
                D_f, z_f = z_half(name, half)
                if D_f is None or D_b is None:
                    continue
                sel = (D_f - D_b) / nstd
                s_mean = float(np.mean(sel))
                s_low = float(np.quantile(sel, 0.025))
                s_high = float(np.quantile(sel, 0.975))
                rows.append(
                    {
                        "encoder": enc,
                        "half": half,
                        "family": name,
                        "group": group_of.get(name, ""),
                        "kind": kind_of.get(name, ""),
                        "z_harmful": round(z_f, 4),
                        "z_benign": round(z_b, 4),
                        "selectivity": round(s_mean, 4),
                        "sel_ci_low": round(s_low, 4),
                        "sel_ci_high": round(s_high, 4),
                        "selective": bool(s_low > 0),
                    }
                )
            print(f"[split] {enc}/{half}: done ({len(targets)} conds)", flush=True)
        out_csv = corpus_dir / "selectivity" / f"prompt_split_eval{args.out_suffix}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(out_csv, rows)

    # ---- mechanical ranking on the SELECTION half ---------------------------
    sel_half = [r for r in rows if r["half"] == "selection"]
    by_enc: Dict[str, List[Dict]] = defaultdict(list)
    for r in sel_half:
        by_enc[r["encoder"]].append(r)
    # per-condition max S over encoders (harmful only)
    max_s: Dict[str, float] = defaultdict(float)
    for r in sel_half:
        if r["kind"] == "harmful":
            max_s[r["family"]] = max(max_s[r["family"]], r["selectivity"])
    ranking: List[Dict] = []
    for enc, rs in by_enc.items():
        benign_fire = any(r["kind"] == "benign_dose" and r["selective"] for r in rs)
        norm, order_norm = [], []
        for r in rs:
            if r["kind"] != "harmful" or max_s.get(r["family"], 0.0) <= 0:
                continue
            s = max(r["selectivity"], 0.0) / max_s[r["family"]]
            norm.append(s)
            if r["group"] in ORDER_GROUPS:
                order_norm.append(s)
        ranking.append(
            {
                "encoder": enc,
                "mean_norm_S": round(float(np.mean(norm)), 4) if norm else 0.0,
                "order_score": round(float(np.mean(order_norm)), 4) if order_norm else 0.0,
                "n_selective_harmful": sum(
                    1 for r in rs if r["kind"] == "harmful" and r["selective"]
                ),
                "benign_fires": benign_fire,
            }
        )
    best_order = max((r["order_score"] for r in ranking), default=0.0)
    for r in ranking:
        r["qualified"] = (not r["benign_fires"]) and (r["order_score"] >= WALL_GUARD * best_order)
    ranking.sort(key=lambda r: (-int(r["qualified"]), -r["mean_norm_S"]))
    rank_csv = corpus_dir / "selectivity" / f"prompt_ranking{args.out_suffix}.csv"
    _write_csv(rank_csv, ranking)
    print(
        f"[split] RANKING (selection half only; wall guard {WALL_GUARD}x, benign disqualifier):",
        flush=True,
    )
    for i, r in enumerate(ranking):
        print(
            f"[split]   #{i + 1} {r['encoder']}: mean_norm_S={r['mean_norm_S']:.3f} "
            f"order={r['order_score']:.3f} nSEL={r['n_selective_harmful']} "
            f"{'DISQ-benign' if r['benign_fires'] else ''}"
            f"{'' if r['qualified'] else ' [unqualified]'}",
            flush=True,
        )
    print(f"[split] COMPLETE -> {rank_csv}", flush=True)


if __name__ == "__main__":
    main()
