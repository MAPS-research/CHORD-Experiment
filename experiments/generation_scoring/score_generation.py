"""Score gen-PPL and tokenizer unigram entropy for normalized DLM samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from chord.utils.io import load_runs, load_texts, write_csv

from ..metric_utils.external_metrics import GenerationPerplexityScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-missing-runs", action="store_true")
    parser.add_argument("--entropy-only", action="store_true")
    parser.add_argument(
        "--tag", default="", help="suffix for the output CSV when comparing variants"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = load_runs(
        (config_path.parent / cfg["runs_manifest"]).resolve(),
        require_samples=not args.allow_missing_runs,
    )
    root = (config_path.parent / cfg["output_dir"]).resolve()
    normalization = str(cfg.get("text_normalization", "none"))
    scorer_cfg = dict(cfg["generation_metrics"])
    cache_path = Path(scorer_cfg["cache_path"])
    if not cache_path.is_absolute():
        scorer_cfg["cache_path"] = str((config_path.parent / cache_path).resolve())
    scorer = GenerationPerplexityScorer(scorer_cfg)
    rows: List[Dict[str, Any]] = []
    for run in runs:
        texts = load_texts(run.samples_path, run.text_key, normalization)
        limit = int(cfg["scoring"].get("n_eval", len(texts)))
        texts = texts[:limit]
        base = {
            "run_id": run.run_id,
            "model": run.model,
            "family": run.family,
            "seed": run.seed,
            "setting": json.dumps(run.setting, sort_keys=True, separators=(",", ":")),
            "n": len(texts),
        }
        entropy = scorer.unigram_entropy(texts)
        if args.entropy_only:
            rows.append(
                {
                    **base,
                    "metric": "unigram_entropy",
                    "value": entropy,
                    "higher_is_worse": False,
                }
            )
            print(
                f"[real-dlm generation metrics] {run.run_id}: entropy={entropy:.3f}",
                flush=True,
            )
        else:
            ppl = scorer.perplexity(texts)
            rows.extend(
                [
                    {
                        **base,
                        "metric": "gen_ppl",
                        "value": ppl,
                        "higher_is_worse": True,
                    },
                    {
                        **base,
                        "metric": "unigram_entropy",
                        "value": entropy,
                        "higher_is_worse": False,
                    },
                ]
            )
            print(
                f"[real-dlm generation metrics] {run.run_id}: ppl={ppl:.3f} entropy={entropy:.3f}",
                flush=True,
            )
    suffix = f"_{args.tag}" if args.tag else ""
    filename = (
        f"entropy_preliminary{suffix}.csv"
        if args.entropy_only
        else f"generation_metrics{suffix}.csv"
    )
    write_csv(root / "scores" / filename, rows)


if __name__ == "__main__":
    main()
