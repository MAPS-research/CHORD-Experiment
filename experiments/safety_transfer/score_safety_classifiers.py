"""Per-response scores of two supervised safety classifiers (GPU).

Gives the "Granite Guardian 3.1 2B" and "toxic-BERT" rows of the safety-transfer
table. Each classifier scores every response; the corpus-level statistic is then
the same null-standardized mean shift used for the paper's other scalar
baselines (``experiments.counterfactual_eval.baseline_selectivity``, scalar lane).

  toxbert   unitary/toxic-bert, the Jigsaw toxicity head: P(toxic)
  guardian  ibm-granite/granite-guardian-3.1-2b: P(Yes) / (P(Yes) + P(No)) for its
            prompt-free "harm" question, read from one forward pass

Both classifiers see only the response text, the same input as the CHORD
readout. Scores the deduplicated pool once and writes
``<corpus>/<scorer>/<set>.npy`` for every set by indexing (see ``featurize``).

    python -m experiments.safety_transfer.score_safety_classifiers \
        --corpus-dir outputs/experiments/safety_transfer --scorer toxbert
    python -m experiments.safety_transfer.score_safety_classifiers \
        --corpus-dir outputs/experiments/safety_transfer --scorer guardian --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List

import numpy as np

TOXBERT = "unitary/toxic-bert"
GUARDIAN = "ibm-granite/granite-guardian-3.1-2b"
# The Guardian chat template writes today's date into its system prompt; it is
# frozen so reruns score an identical prompt.
FROZEN_DATE = "August 30, 2026"


def pool_texts(corpus_dir: Path) -> List[str]:
    with open(corpus_dir / "texts" / "pool.jsonl", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()]


def score_toxbert(texts: List[str], batch_size: int, cache_dir: str | None) -> np.ndarray:
    """P(toxic) from the multi-label Jigsaw head."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOXBERT, cache_dir=cache_dir)
    model = (
        AutoModelForSequenceClassification.from_pretrained(TOXBERT, cache_dir=cache_dir)
        .cuda()
        .eval()
    )
    label = {v.lower(): k for k, v in model.config.id2label.items()}["toxic"]
    out = np.empty(len(texts), dtype=np.float64)
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tok(
                texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to("cuda")
            logits = model(**batch).logits[:, label]
            out[i : i + batch_size] = torch.sigmoid(logits).float().cpu().numpy()
    return out


def score_guardian(texts: List[str], batch_size: int, cache_dir: str | None) -> np.ndarray:
    """P(Yes) / (P(Yes) + P(No)) for Granite Guardian's prompt-free harm question."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GUARDIAN, cache_dir=cache_dir)
    model = (
        AutoModelForCausalLM.from_pretrained(GUARDIAN, cache_dir=cache_dir, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def render(text: str) -> str:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            guardian_config={"risk_name": "harm"},
            add_generation_prompt=True,
            tokenize=False,
        )
        return re.sub(r"Today's Date: [^\n]*", f"Today's Date: {FROZEN_DATE}.", prompt, count=1)

    prompts = [render(t) for t in texts]
    out = np.empty(len(texts), dtype=np.float64)
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = tok(
                prompts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            ).to("cuda")
            logits = model(**batch).logits[:, -1, :].float()
            pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1)
            out[i : i + batch_size] = torch.softmax(pair, dim=-1)[:, 0].cpu().numpy()
    return out


SCORERS = {"toxbert": score_toxbert, "guardian": score_guardian}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--scorer", required=True, choices=sorted(SCORERS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache (default: HF_HOME)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = corpus_dir / args.scorer
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "_pool.npy"

    if pool_path.exists() and not args.overwrite:
        pool = np.load(pool_path)
        print(f"[{args.scorer}] reusing cached pool scores ({len(pool)})", flush=True)
    else:
        texts = pool_texts(corpus_dir)
        t0 = time.perf_counter()
        pool = SCORERS[args.scorer](texts, args.batch_size, args.cache_dir)
        dt = time.perf_counter() - t0
        np.save(pool_path, pool)
        print(
            f"[{args.scorer}] scored {len(texts)} docs in {dt:.1f}s mean={pool.mean():.4f} "
            f"p90={np.quantile(pool, 0.9):.4f} max={pool.max():.4f}",
            flush=True,
        )

    index_paths = sorted((corpus_dir / "index").glob("*.json"))
    for idx_path in index_paths:
        idx = np.asarray(json.loads(idx_path.read_text()), dtype=np.int64)
        np.save(out_dir / f"{idx_path.stem}.npy", pool[idx])
    print(f"[{args.scorer}] composed {len(index_paths)} sets -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
