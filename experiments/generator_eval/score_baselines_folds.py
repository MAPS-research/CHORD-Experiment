#!/usr/bin/env python3
"""Table-2 baseline columns (gen-PPL, MAUVE, unigram entropy) under the 10 x 500
fold protocol, so every column of the table uses the same samples.

Definitions are unchanged from experiments.generation_scoring.score / score_generation:
gen-PPL is GPT-2-large corpus perplexity of a fold (fp16, 512 tokens); entropy is
tokenizer unigram entropy of a fold; MAUVE uses GPT-2-large last-token features,
num_buckets = n // 10, averaged over the same five k-means seeds, with fold k of
the human reference pool as P and fold k of the generator as Q. Folds are the
ones score_chord_folds.py uses (same permutation seed), so no document is reused.

  python -m experiments.generator_eval.score_baselines_folds \
      --config experiments/generator_eval/configs/featurize.yaml
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from chord.utils.io import load_runs, load_texts

from ..counterfactual_eval.mauve_eval import mauve_scores
from ..metric_utils.external_metrics import GenerationPerplexityScorer

MAUVE_SEEDS = [20260620, 20260621, 20260622, 20260623, 20260624]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--dev", type=int, default=150)
    ap.add_argument("--mauve-encoder", default="mauve-gpt2")
    ap.add_argument("--skip-mauve", action="store_true")
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = (cfg_path.parent / cfg["output_dir"]).resolve()
    runs = load_runs((cfg_path.parent / cfg["runs_manifest"]).resolve(), require_samples=True)
    seed = int(cfg.get("seed", 0))
    norm = str(cfg.get("text_normalization", "none"))
    need = args.folds * args.size

    gen_cfg = dict(cfg["generation_metrics"])
    cp = Path(gen_cfg["cache_path"])
    gen_cfg["cache_path"] = str((cfg_path.parent / cp).resolve() if not cp.is_absolute() else cp)
    Path(gen_cfg["cache_path"]).parent.mkdir(parents=True, exist_ok=True)
    scorer = GenerationPerplexityScorer(gen_cfg)

    # reference folds: identical permutation to score_chord_folds.py
    ref_path = (cfg_path.parent / cfg["reference"]["path"]).resolve()
    ref_texts = load_texts(ref_path, cfg["reference"].get("text_key", "text"), norm)[
        : int(cfg["reference"]["limit"])
    ]
    order = np.random.default_rng(seed).permutation(len(ref_texts))
    fold_idx = order[args.dev : args.dev + need]
    ref_feats = None
    if not args.skip_mauve:
        ref_feats = np.load(root / "features" / args.mauve_encoder / "reference.npy").astype(
            np.float32
        )
        assert len(ref_feats) == len(ref_texts), (len(ref_feats), len(ref_texts))

    models = list(dict.fromkeys(r.model for r in runs))
    rows, table = [], {}
    for model in models:
        mruns = [r for r in runs if r.model == model]
        texts = [t for r in mruns for t in load_texts(r.samples_path, r.text_key, norm)]
        assert len(texts) >= need, f"{model}: {len(texts)} docs, need {need}"
        feats = None
        if not args.skip_mauve:
            feats = np.concatenate(
                [np.load(root / "features" / args.mauve_encoder / f"{r.run_id}.npy") for r in mruns]
            ).astype(np.float32)
        per = {"gen_ppl": [], "entropy": [], "mauve": []}
        for k in range(args.folds):
            sl = slice(k * args.size, (k + 1) * args.size)
            fold_texts = texts[sl]
            per["gen_ppl"].append(scorer.perplexity(fold_texts))
            per["entropy"].append(scorer.unigram_entropy(fold_texts))
            if feats is not None:
                m = mauve_scores(
                    ref_feats[fold_idx[sl]],
                    feats[sl],
                    num_buckets=max(2, args.size // 10),
                    seeds=MAUVE_SEEDS,
                )
                per["mauve"].append(m["mauve_mean"])
            for metric, vals in per.items():
                if len(vals) == k + 1:
                    rows.append(
                        {
                            "model": model,
                            "fold": k,
                            "n": args.size,
                            "metric": metric,
                            "value": vals[-1],
                        }
                    )
        table[model] = {
            m: (float(np.mean(v)), float(np.std(v, ddof=1))) if v else (float("nan"), float("nan"))
            for m, v in per.items()
        }
        print(
            f"[baselines] {model:24s} "
            + "  ".join(f"{m} {table[model][m][0]:7.3f}±{table[model][m][1]:.3f}" for m in per),
            flush=True,
        )

    print(
        f"\nTABLE 2 baseline columns, {args.folds} folds x {args.size}: mean ± std "
        "(rank; gen-PPL low=1, MAUVE/entropy high=1)"
    )
    for metric, high in (("gen_ppl", False), ("mauve", True), ("entropy", True)):
        means = {m: table[m][metric][0] for m in models}
        order_m = sorted(models, key=lambda m: -means[m] if high else means[m])
        print(
            f"  {metric}: "
            + " | ".join(
                f"{m} {table[m][metric][0]:.2f}±{table[m][metric][1]:.2f} (#{order_m.index(m) + 1})"
                for m in models
            )
        )
    out = root / "scores" / "repeat_folds_baselines.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
