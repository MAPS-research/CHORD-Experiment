"""Direct LLM-judge ratings of the counterfactual evaluation set.

Produces the "LLM judge (same Qwen3.5-27B)" row of the appendix table
tab:judge-counterfactual. The judge is the same Qwen3.5-27B snapshot CHORD
uses as its encoder, served through an OpenAI-compatible endpoint (vLLM):

    vllm serve Qwen/Qwen3.5-27B --served-model-name qwen35-27b-judge --max-model-len 4096

Each passage is rated in isolation on an integer 0-10 scale with the prompt of
tab:judge-prompt (temperature 0, fixed seed). Scores are written to
``<corpus-dir>/<values_dir>/<set>.npy`` in each text set's line order, the
layout the scalar lane of ``experiments.counterfactual_eval.baseline_selectivity`` reads:

    python -m experiments.llm_judge.judge_evaluation_set \\
        --config experiments/llm_judge/configs/judge.yaml \\
        --corpus-dir outputs/counterfactual/meta_eval
    python -m experiments.counterfactual_eval.baseline_selectivity \\
        --config experiments/llm_judge/configs/baseline_selectivity_llm_judges.yaml \\
        --corpus-dir outputs/counterfactual/meta_eval --out-suffix _llm_judges

Responses are cached by (model, prompt revision, text hash), so reruns are
free. A document whose request still fails after all retries is recorded as NaN
(not cached, so a rerun retries it); the |mean shift| statistic drops NaNs.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from chord.data.perturbations.llm import call_openai_compatible
from chord.utils.hashing import sha256_text

from ..counterfactual_eval.featurize import _load_texts, _set_path, text_set_names

# Part of every cache key: changing the prompt must change this string.
PROMPT_REVISION = "j10-v1"

# tab:judge-prompt. Score first, then a one-sentence reason.
JUDGE_PROMPT = (
    "You are a strict, careful writing-quality rater. Read the document "
    "between the <document> markers and rate how coherent and well-formed it "
    "is as a piece of writing — logical consistency, discourse flow, and "
    "overall quality — on an integer scale from 0 (incoherent, broken, or "
    "self-contradictory) to 10 (flawless, fully coherent writing). Judge only "
    "the writing itself; do not reward or penalize the topic, opinions, or "
    "genre, and ignore truncation at the very end of the document.\n\n"
    "<document>\n{text}\n</document>\n\n"
    'Return only JSON, the score first: {{"score": <integer 0-10>, '
    '"reason": "<one short sentence>"}}'
)


def to_score(value: object) -> float:
    """Numeric value -> integer score clamped to [0, 10]; raises if non-numeric."""
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(f"non-finite judge score: {value!r}")
    return float(max(0, min(10, int(round(score)))))


class JudgeClient:
    """Thread-safe judge with a (model, prompt revision, text hash) response cache."""

    def __init__(self, cfg: Dict, cache_path: Path) -> None:
        self.cfg = cfg
        self.model = str(cfg["model"])
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._dirty = 0
        self.cache: Dict[str, Dict] = {}
        if cache_path.is_file():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _key(self, text: str) -> str:
        return sha256_text(f"{self.model}:{PROMPT_REVISION}:judge10:{sha256_text(text)}")

    def score_one(self, text: str) -> Dict:
        """-> {"score": float (NaN on failure), "reason": str}."""
        key = self._key(text)
        with self._lock:
            hit = self.cache.get(key)
        if hit is not None:
            return hit
        try:
            raw = call_openai_compatible(
                base_url=self.cfg.get("base_url", "http://127.0.0.1:8000/v1"),
                model=self.model,
                prompt=JUDGE_PROMPT.format(text=text),
                seed=int(self.cfg.get("seed", 0)),
                temperature=float(self.cfg.get("temperature", 0.0)),
                timeout_seconds=int(self.cfg.get("timeout_seconds", 180)),
                api_key=None,
                max_retries=int(self.cfg.get("max_retries", 5)),
                max_completion_tokens=int(self.cfg.get("max_completion_tokens", 200)),
                required_string_field="reason",
                chat_template_kwargs=self.cfg.get("chat_template_kwargs"),
            )
            out = {"score": to_score(raw["score"]), "reason": str(raw["reason"])}
        except Exception as exc:  # noqa: BLE001 - keep the run alive, record NaN
            return {"score": float("nan"), "reason": f"__judge_error__: {exc}"}
        with self._lock:
            self.cache[key] = out
            self._dirty += 1
            flush = self._dirty >= int(self.cfg.get("flush_every", 200))
        if flush:
            self.flush()
        return out

    def flush(self) -> None:
        """Atomically rewrite the cache file (whole write under the lock)."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = json.dumps(self.cache, ensure_ascii=False)
            self._dirty = 0
            tmp = self.cache_path.with_suffix(f".tmp{threading.get_ident()}")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.cache_path)


def judge_texts(
    judge: JudgeClient, texts: List[str], workers: int
) -> Tuple[np.ndarray, List[str], int]:
    """Rate all texts concurrently -> (scores in input order, reasons, failures)."""
    scores = np.full(len(texts), np.nan, dtype=np.float64)
    reasons: List[str] = [""] * len(texts)
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge.score_one, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futures):
            i = futures[fut]
            res = fut.result()
            scores[i] = res["score"]
            reasons[i] = res["reason"]
            if not np.isfinite(res["score"]):
                failed += 1
    return scores, reasons, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus-dir", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cdir = Path(args.corpus_dir)
    manifest = json.loads((cdir / "conditions_manifest.json").read_text())
    vdir = cdir / str(cfg.get("values_dir", "judge"))
    vdir.mkdir(parents=True, exist_ok=True)
    workers = int(cfg.get("concurrency", 32))
    impute = bool(cfg.get("impute_median", False))
    judge = JudgeClient(cfg["judge"], vdir / "judge_cache.json")

    total_failed = 0
    for name in text_set_names(manifest):
        out = vdir / f"{name}.npy"
        if out.exists():
            print(f"[judge] {name} exists, skip", flush=True)
            continue
        texts = _load_texts(_set_path(cdir / "texts", name))
        scores, reasons, failed = judge_texts(judge, texts, workers)
        total_failed += failed
        n_nan = int(np.sum(~np.isfinite(scores)))
        if impute and n_nan:
            median = float(np.nanmedian(scores))
            scores = np.where(np.isfinite(scores), scores, median)
            print(f"[judge] {name}: imputed {n_nan} NaN with median {median:.1f}", flush=True)
        np.save(out, scores)
        with open(vdir / f"{name}.reasons.jsonl", "w", encoding="utf-8") as fh:
            for i, (score, reason) in enumerate(zip(scores.tolist(), reasons)):
                fh.write(json.dumps({"idx": i, "score": score, "reason": reason}) + "\n")
        judge.flush()
        print(f"[judge] {name}: n={len(texts)} mean={np.nanmean(scores):.3f} failed={failed}")
    judge.flush()
    print(f"[judge] done -> {vdir} (request failures: {total_failed})", flush=True)


if __name__ == "__main__":
    main()
