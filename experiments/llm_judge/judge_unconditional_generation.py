"""Direct LLM-judge ratings on the unconditional-generation folds of Table 2.

Produces the "Judge mean" and "Judge z_M" columns of the appendix table
tab:judge-casestudy on exactly the texts behind Table 2: ``fold_config`` is
Table 2's own config (experiments/generator_eval/configs/featurize.yaml), so the
runs, the reference pool and its seed permutation (150 dev windows, then
10 x 500 reference folds) are the ones ``experiments.generator_eval.score_chord_folds``
scores. Generator fold k = samples [k*500, (k+1)*500) of that generator's runs in
manifest order. The judge (same Qwen3.5-27B, prompt and settings as
``judge_evaluation_set``) rates every text once, cut to its first ``max_tokens``
GPT-2 tokens, the window every Table 2 metric reads (only ELF-L's ~930-token
documents are longer).

Per fold k the statistic is |mean rating of generator fold k - mean rating of
reference fold k|. The null is the same statistic between two disjoint fold-sized
subsets of the reference folds, and z_k = (statistic_k - null mean) / null std.
A generator counts as detected when its fold-mean statistic exceeds the 95th
percentile of fold-means of ``folds`` null draws.

    python -m experiments.llm_judge.judge_unconditional_generation \\
        --config experiments/llm_judge/configs/judge_unconditional_generation.yaml

Outputs (``output_dir``): judge_folds.csv (one row per generator x fold),
judge_summary.csv (one row per generator), judge_calibration.csv, and
judge_ratings.jsonl (every rating with its one-sentence reason).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from chord.utils.hashing import seeded_rng
from chord.utils.io import load_runs, load_texts, write_csv

from ..counterfactual_eval.baseline_selectivity import _stat_abs_mean_shift
from .judge_evaluation_set import PROMPT_REVISION, JudgeClient, judge_texts


def truncate_tokens(texts: List[str], truncation: Dict) -> List[str]:
    """Cut each text to its first ``max_tokens`` tokens of ``tokenizer``."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(truncation["tokenizer"], revision=truncation["revision"])
    limit = int(truncation["max_tokens"])
    out = []
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out.append(text if len(ids) <= limit else " ".join(tok.decode(ids[:limit]).split()))
    return out


def load_folds(fold_cfg_path: Path, folds: int, size: int, dev: int):
    """(reference folds, {generator: its folds}) exactly as score_chord_folds splits them."""
    cfg = yaml.safe_load(fold_cfg_path.read_text(encoding="utf-8"))
    base = fold_cfg_path.parent
    norm = str(cfg.get("text_normalization", "whitespace"))
    ref_cfg = cfg["reference"]
    ref = load_texts((base / ref_cfg["path"]).resolve(), ref_cfg.get("text_key", "text"), norm)
    ref = ref[: int(ref_cfg.get("limit", len(ref)))]
    need = folds * size
    assert len(ref) >= dev + need, f"reference has {len(ref)} windows, need {dev + need}"
    order = np.random.default_rng(int(cfg.get("seed", 0))).permutation(len(ref))[dev : dev + need]
    ref_folds = [[ref[i] for i in order[k * size : (k + 1) * size]] for k in range(folds)]

    runs = load_runs((base / cfg["runs_manifest"]).resolve(), require_samples=False)
    gen_folds: Dict[str, List[List[str]]] = {}
    for model in dict.fromkeys(r.model for r in runs):
        model_runs = [r for r in runs if r.model == model]
        texts = [t for r in model_runs for t in load_texts(r.samples_path, r.text_key, norm)]
        assert len(texts) >= need, f"{model}: {len(texts)} samples, need {need}"
        gen_folds[model] = [texts[k * size : (k + 1) * size] for k in range(folds)]
    return ref_folds, gen_folds


def rate(judge: JudgeClient, texts: List[str], truncation: Dict | None, workers: int):
    texts = truncate_tokens(texts, truncation) if truncation else texts
    vals, reasons, failed = judge_texts(judge, texts, workers)
    judge.flush()
    return vals, reasons, failed


def null_draws(pool: np.ndarray, size: int, draws: int, rng) -> np.ndarray:
    """|mean shift| between two disjoint fold-sized subsets of the reference ratings."""
    out = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        idx = rng.choice(len(pool), size=2 * size, replace=False)
        out[d] = _stat_abs_mean_shift(pool[idx[:size]], pool[idx[size:]])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    folds, size, dev = int(cfg["folds"]), int(cfg["size"]), int(cfg["dev"])
    draws, workers = int(cfg.get("draws", 1000)), int(cfg.get("concurrency", 32))
    truncation = cfg.get("truncation")
    out_dir = (base / cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_folds, gen_folds = load_folds((base / cfg["fold_config"]).resolve(), folds, size, dev)
    judge = JudgeClient(cfg["judge"], out_dir / "judge_cache.json")
    print(
        f"[judge] model={judge.model} rev={PROMPT_REVISION} generators={len(gen_folds)} "
        f"folds={folds}x{size}",
        flush=True,
    )
    ref_vals, ref_reasons, ref_failed = rate(
        judge, [t for f in ref_folds for t in f], truncation, workers
    )
    gen_vals, gen_reasons, gen_failed = {}, {}, {}
    for model, model_folds in gen_folds.items():
        gen_vals[model], gen_reasons[model], gen_failed[model] = rate(
            judge, [t for f in model_folds for t in f], truncation, workers
        )
        print(f"[judge] {model}: mean={np.nanmean(gen_vals[model]):.2f}", flush=True)

    rng = seeded_rng("judge-folds", PROMPT_REVISION, size)
    null = null_draws(ref_vals, size, draws, rng)
    nmean, nstd = float(null.mean()), float(null.std())
    fold_mean_null = null[rng.integers(0, draws, size=(draws, folds))].mean(axis=1)
    q95_fold_mean = float(np.quantile(fold_mean_null, 0.95))

    fold_rows: List[Dict] = []
    summary: List[Dict] = []
    for model, vals in gen_vals.items():
        stats = np.array(
            [
                _stat_abs_mean_shift(
                    ref_vals[k * size : (k + 1) * size], vals[k * size : (k + 1) * size]
                )
                for k in range(folds)
            ]
        )
        z = (stats - nmean) / nstd
        for k in range(folds):
            fold_rows.append(
                {
                    "model": model,
                    "fold": k,
                    "n": size,
                    "judge_mean": round(float(np.nanmean(vals[k * size : (k + 1) * size])), 4),
                    "abs_mean_shift": round(float(stats[k]), 5),
                    "z": round(float(z[k]), 4),
                }
            )
        summary.append(
            {
                "model": model,
                "n_total": len(vals),
                "n_failed": gen_failed[model],
                "judge_mean": round(float(np.nanmean(vals)), 4),
                "judge_mean_fold_std": round(
                    float(np.std([r["judge_mean"] for r in fold_rows[-folds:]], ddof=1)), 4
                ),
                "z_mean": round(float(z.mean()), 4),
                "z_std": round(float(z.std(ddof=1)), 4),
                "detected": bool(stats.mean() > q95_fold_mean),
            }
        )
        print(
            f"[judge] {model}: z={z.mean():.1f}+-{z.std(ddof=1):.1f} "
            f"detected={summary[-1]['detected']}",
            flush=True,
        )

    calib = {
        "statistic": "judge_abs_mean_shift",
        "prompt_revision": PROMPT_REVISION,
        "model": judge.model,
        "folds": folds,
        "size": size,
        "draws": draws,
        "ref_n": len(ref_vals),
        "ref_failed": ref_failed,
        "ref_mean": round(float(np.nanmean(ref_vals)), 4),
        "null_mean": nmean,
        "null_std": nstd,
        "q95_fold_mean": q95_fold_mean,
    }
    write_csv(out_dir / "judge_folds.csv", fold_rows)
    write_csv(out_dir / "judge_summary.csv", summary)
    write_csv(out_dir / "judge_calibration.csv", [calib])
    with (out_dir / "judge_ratings.jsonl").open("w", encoding="utf-8") as fh:
        groups = [("reference", ref_vals, ref_reasons)] + [
            (m, gen_vals[m], gen_reasons[m]) for m in gen_vals
        ]
        for name, vals, reasons in groups:
            for i, (score, reason) in enumerate(zip(vals.tolist(), reasons)):
                row = {"corpus": name, "fold": i // size, "idx": i, "score": score}
                row["reason"] = reason
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[judge] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
