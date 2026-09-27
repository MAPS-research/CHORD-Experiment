"""Step 2 of the QA-faithfulness evaluation set: the faithful paraphrase control (LLM).

The supporting sentence S of each parent is rewritten by the editor LLM
(Qwen3-30B-A3B, the evaluation-set editor, a different family from the metric
backbone) under a strict faithfulness contract: every number, name, date and
the polarity kept. A paraphrase is accepted only if it differs from S, keeps
0.5-2x its word count and keeps the multiset of numbers verbatim. Accepted
rows form ``benign_paraphrase.jsonl`` (paraphrase draw B), the faithful control
that should sit near the null.

Re-derives the parents of step 1 and checks them against ``reference.jsonl``
before calling the editor. Requires an OpenAI-compatible server.

    python -m experiments.qa_faithfulness.paraphrase_benign_control \\
        --config experiments/qa_faithfulness/configs/build.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import yaml

from chord.data.perturbations.llm import call_openai_compatible

from .build_verbatim_conditions import (
    check_parents,
    compose,
    derive_items,
    write_rows,
)

PROMPT = (
    "Rewrite the statement below as a natural, fluent paraphrase that preserves its "
    "meaning EXACTLY. Keep every fact unchanged: every number, date, name, place, and "
    "whether the statement is affirmative or negated. Change the wording and sentence "
    "structure freely, but do not add, drop, or alter any information. "
    'Respond with JSON only: {"perturbed_text": "<the paraphrase>"}.\n'
    'Statement: "{s}"'
)

_WORD = re.compile(r"\b\w+\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def valid(src: str, para: str) -> bool:
    """Differs from the source, similar length, numbers preserved verbatim."""
    p = " ".join(para.split())
    if not p or p.casefold() == src.casefold():
        return False
    if not (0.5 <= len(_WORD.findall(p)) / max(1, len(_WORD.findall(src))) <= 2.0):
        return False
    return sorted(_NUMBER.findall(src)) == sorted(_NUMBER.findall(p))


def paraphrase_one(
    base_url: str, model: str, parent_id: str, statement: str, seed_base: int
) -> Optional[str]:
    """One validated paraphrase of ``statement`` (up to 3 seeded attempts) or None."""
    idx = int(parent_id.split(":")[1])
    for attempt in range(3):
        try:
            r = call_openai_compatible(
                base_url=base_url,
                model=model,
                prompt=PROMPT.replace("{s}", statement),
                seed=seed_base + idx * 10 + attempt,
                temperature=0.7,
                timeout_seconds=180,
                max_completion_tokens=400,
                required_string_field="perturbed_text",
                chat_template_kwargs={"enable_thinking": False},
            )
            p = " ".join(str(r["perturbed_text"]).split()).strip().strip('"')
            if valid(statement, p):
                return p
        except RuntimeError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    para = cfg["paraphrase"]
    texts_dir = Path(cfg["texts_dir"])
    items = derive_items(
        cfg["squad"], int(cfg["max_source_chars"]), int(cfg["limit"]), int(cfg["seed"])
    )
    check_parents(items, texts_dir / "reference.jsonl")

    seed_base = int(para["benign_seed_base"])
    with ThreadPoolExecutor(max_workers=int(para.get("workers", 8))) as ex:
        paras = list(
            ex.map(
                lambda it: paraphrase_one(
                    para["base_url"], para["model"], it["parent_id"], it["statement"], seed_base
                ),
                items,
            )
        )
    kept = [(it, p) for it, p in zip(items, paras) if p]
    print(f"[qa] {len(kept)}/{len(items)} draw-B paraphrases passed validation")
    if len(kept) < int(para["min_valid_benign"]):
        raise SystemExit(f"only {len(kept)} valid paraphrases (< {para['min_valid_benign']})")
    write_rows(
        texts_dir / "benign_paraphrase.jsonl",
        [
            {"parent_id": it["parent_id"], "text": compose(it["source"], it["question"], p)}
            for it, p in kept
        ],
    )
    audit = {
        "n_parents": len(items),
        "n_valid": len(kept),
        "editor": para["model"],
        "seed_base": seed_base,
        "examples": [
            {"pid": it["parent_id"], "support": it["statement"], "paraphrase": p}
            for it, p in kept[:10]
        ],
    }
    (texts_dir / "benign_paraphrase_audit.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
