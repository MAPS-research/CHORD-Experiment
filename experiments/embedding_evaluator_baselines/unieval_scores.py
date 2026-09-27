"""UniEval scores of the counterfactual evaluation set (table tab:modern-baselines).

UniEval (Zhong et al., EMNLP 2022) is a T5 Boolean-QA evaluator: each dimension
is a yes/no question and the score is P(Yes) / (P(Yes) + P(No)) at the first
decoder step (the official ``UniEvaluator.score``, batched here). The passages
have no source or reference, so only source-free dimensions are used:

  fluency    ``unieval-sum`` as released: each sentence is scored with
             "Is this a fluent paragraph?" and the sentence scores are averaged
             (sentences split with nltk ``sent_tokenize``, as in the official
             evaluator, falling back to ``chord.text.sentences``)
  coherence  ``unieval-intermediate`` with the zero-shot question
             "Is this a coherent and logically consistent passage?"
  overall    the mean of the two ("fluency + coherence" in the table)

    python -m experiments.embedding_evaluator_baselines.unieval_scores \\
        --config experiments/embedding_evaluator_baselines/configs/unieval.yaml \\
        --corpus-dir outputs/counterfactual/meta_eval

Scores land in ``<corpus-dir>/{unieval_fluency,unieval_coherence,unieval_overall}/<set>.npy``
(line order of each text set) for the scalar lane of
``experiments.counterfactual_eval.baseline_selectivity``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import yaml

from chord.text import sentences as regex_sentences

from ..counterfactual_eval.featurize import _load_texts, _set_path, text_set_names

FLUENCY_QUESTION = "question: Is this a fluent paragraph? </s> paragraph: "


def sentence_splitter() -> Callable[[str], List[str]]:
    """nltk sent_tokenize when its punkt data is installed, else the regex splitter."""
    try:
        from nltk.tokenize import sent_tokenize

        sent_tokenize("A test. Another one.")
        print("[unieval] sentence splitter: nltk sent_tokenize", flush=True)
        return sent_tokenize
    except Exception as exc:  # noqa: BLE001 - nltk or punkt missing
        print(f"[unieval] nltk unavailable ({exc}); using chord.text.sentences", flush=True)
        return regex_sentences


def custom_question(question: str, field: str, text: str) -> str:
    """UniEval Boolean-QA input for a source-free custom dimension."""
    return f"question: {question} </s> {field}: {text}"


class UniEvalScorer:
    """Official scoring rule: softmax at the first decoder position, P(Yes)/(P(Yes)+P(No))."""

    def __init__(
        self,
        model: str,
        revision: str | None,
        cache_dir: str | None,
        max_length: int = 1024,
        batch_size: int = 32,
        dtype: str = "float32",
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision=revision, cache_dir=cache_dir
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model, revision=revision, cache_dir=cache_dir, torch_dtype=getattr(torch, dtype)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.pos_id = self.tokenizer("Yes")["input_ids"][0]
        self.neg_id = self.tokenizer("No")["input_ids"][0]
        # T5 needs decoder inputs; the label token does not affect first-step logits.
        self.tgt_id = self.tokenizer("No")["input_ids"][0]

    def score(self, inputs: List[str]) -> np.ndarray:
        torch = self.torch
        out = np.empty(len(inputs), dtype=np.float64)
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))  # fewer pad tokens
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer(
                [inputs[i] for i in idx],
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            labels = torch.full((len(idx), 1), self.tgt_id, dtype=torch.long)
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                    labels=labels.to(self.device),
                ).logits
                probs = torch.softmax(logits[:, 0, :].float(), dim=-1)
                pos, neg = probs[:, self.pos_id], probs[:, self.neg_id]
                out[np.asarray(idx)] = (pos / (pos + neg)).cpu().numpy()
        return out


def fluency_scores(scorer: UniEvalScorer, texts: List[str], split) -> np.ndarray:
    """Sentence-level fluency averaged per passage (official evaluator)."""
    inputs: List[str] = []
    counts: List[int] = []
    for t in texts:
        sents = [s for s in split(t) if s.strip()] or [t]
        counts.append(len(sents))
        inputs.extend(FLUENCY_QUESTION + s for s in sents)
    flat = scorer.score(inputs)
    out = np.empty(len(texts), dtype=np.float64)
    pos = 0
    for i, n in enumerate(counts):
        out[i] = float(flat[pos : pos + n].mean())
        pos += n
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
    dims: Dict[str, str] = dict(
        cfg.get("values_dirs")
        or {
            "fluency": "unieval_fluency",
            "coherence": "unieval_coherence",
            "overall": "unieval_overall",
        }
    )
    vdirs = {k: cdir / v for k, v in dims.items()}
    for d in vdirs.values():
        d.mkdir(parents=True, exist_ok=True)

    common = dict(
        cache_dir=cfg.get("cache_dir"),
        max_length=int(cfg.get("max_length", 1024)),
        batch_size=int(cfg.get("batch_size", 32)),
        dtype=str(cfg.get("dtype", "float32")),
    )
    fluency_model = UniEvalScorer(cfg["sum_model"], cfg.get("sum_revision"), **common)
    coherence_model = UniEvalScorer(
        cfg["intermediate_model"], cfg.get("intermediate_revision"), **common
    )
    question = str(
        cfg.get("coherence_question", "Is this a coherent and logically consistent passage?")
    )
    field = str(cfg.get("coherence_field", "passage"))
    split = sentence_splitter()

    for name in sets:
        if all((vdirs[k] / f"{name}.npy").exists() for k in dims) and not args.limit:
            print(f"[unieval] {name} exists, skip", flush=True)
            continue
        texts = _load_texts(_set_path(cdir / "texts", name))[: args.limit or None]
        t0 = time.perf_counter()
        flu = fluency_scores(fluency_model, texts, split)
        coh = coherence_model.score([custom_question(question, field, t) for t in texts])
        vals = {"fluency": flu, "coherence": coh, "overall": (flu + coh) / 2.0}
        msg = (
            f"n={len(texts)} fluency={flu.mean():.3f} coherence={coh.mean():.3f} "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        if args.limit:
            print(f"[unieval] {name} (limit, not saved): {msg}", flush=True)
            continue
        for k in dims:
            np.save(vdirs[k] / f"{name}.npy", vals[k].astype(np.float64))
        print(f"[unieval] {name}: {msg}", flush=True)
    print("[unieval] done", flush=True)


if __name__ == "__main__":
    main()
