"""Hypothesis-only BARTScore of the counterfactual evaluation set (table tab:modern-baselines).

BARTScore (Yuan et al., NeurIPS 2021) is the length-normalized log-likelihood
log p(y | x) / |y| of a target text under ``facebook/bart-large-cnn`` (the
official ``BARTScorer.score``: token NLL over the labels, padding ignored,
divided by the target length including BOS/EOS). Every published direction
conditions on a source or a paired reference; the evaluation set has neither,
so the passage is scored with an empty source, log p(y | "") / |y| (config
``source: ""``).

    python -m experiments.embedding_evaluator_baselines.bartscore_scores \\
        --config experiments/embedding_evaluator_baselines/configs/bartscore.yaml \\
        --corpus-dir outputs/counterfactual/meta_eval

Scores land in ``<corpus-dir>/bartscore/<set>.npy`` (line order of each text set)
for the scalar lane of ``experiments.counterfactual_eval.baseline_selectivity``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml

from ..counterfactual_eval.featurize import _load_texts, _set_path, text_set_names


class BARTScorer:
    def __init__(
        self,
        model: str,
        revision: str | None,
        cache_dir: str | None,
        max_length: int = 1024,
        batch_size: int = 8,
        dtype: str = "float32",
    ) -> None:
        import torch
        from transformers import AutoTokenizer, BartForConditionalGeneration

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, cache_dir=cache_dir
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            model, revision=revision, cache_dir=cache_dir, torch_dtype=getattr(torch, dtype)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.pad_id = int(self.model.config.pad_token_id)

    def score(self, sources: List[str], targets: List[str]) -> np.ndarray:
        """Per-item log p(target | source) / |target| (higher is better)."""
        torch = self.torch
        out = np.empty(len(targets), dtype=np.float64)
        order = sorted(range(len(targets)), key=lambda i: len(targets[i]))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            kw = dict(
                max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
            )
            enc_src = self.tokenizer([sources[i] for i in idx], **kw)
            enc_tgt = self.tokenizer([targets[i] for i in idx], **kw)
            tgt_ids = enc_tgt["input_ids"].to(self.device)
            tgt_len = enc_tgt["attention_mask"].sum(dim=1).to(self.device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc_src["input_ids"].to(self.device),
                    attention_mask=enc_src["attention_mask"].to(self.device),
                    labels=tgt_ids,
                ).logits.float()
                nll = (
                    -torch.log_softmax(logits, dim=-1).gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
                )
                nll = nll.masked_fill(tgt_ids == self.pad_id, 0.0)
                out[np.asarray(idx)] = (-(nll.sum(dim=1) / tgt_len)).cpu().numpy()
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--only-sets", default="", help="comma-separated text sets")
    parser.add_argument("--limit", type=int, default=0, help="first N docs per set (not saved)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cdir = Path(args.corpus_dir)
    sets = text_set_names(json.loads((cdir / "conditions_manifest.json").read_text()))
    only = [s.strip() for s in args.only_sets.split(",") if s.strip()]
    if only:
        unknown = [s for s in only if s not in sets]
        if unknown:
            raise SystemExit(f"--only-sets: unknown text sets {unknown}")
        sets = only
    vdir = cdir / str(cfg.get("values_dir", "bartscore"))
    vdir.mkdir(parents=True, exist_ok=True)
    source = str(cfg.get("source", ""))
    scorer = BARTScorer(
        cfg["model"],
        cfg.get("revision"),
        cfg.get("cache_dir"),
        max_length=int(cfg.get("max_length", 1024)),
        batch_size=int(cfg.get("batch_size", 8)),
        dtype=str(cfg.get("dtype", "float32")),
    )
    for name in sets:
        out = vdir / f"{name}.npy"
        if out.exists() and not args.limit:
            print(f"[bartscore] {name} exists, skip", flush=True)
            continue
        texts = _load_texts(_set_path(cdir / "texts", name))[: args.limit or None]
        t0 = time.perf_counter()
        vals = scorer.score([source] * len(texts), texts)
        msg = f"n={len(texts)} mean={vals.mean():.4f} ({time.perf_counter() - t0:.1f}s)"
        if args.limit:
            print(f"[bartscore] {name} (limit, not saved): {msg}", flush=True)
            continue
        np.save(out, vals.astype(np.float64))
        print(f"[bartscore] {name}: {msg}", flush=True)
    print(f"[bartscore] done -> {vdir}", flush=True)


if __name__ == "__main__":
    main()
