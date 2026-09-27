"""Step 3 of the QA-faithfulness evaluation set: paraphrase reference and wrong-fact conditions.

If the wrong-fact edits were applied to the verbatim supporting sentence, each
wrong-fact corpus would differ from a paraphrased reference by the injected fact
and by verbatim-vs-paraphrase wording. This step removes the wording gap: an
independent LLM paraphrase of S (draw A) is both the faithful reference
(``reference_paraphrase.jsonl``) and the base for the three FactCC edits
(``contra_{number,entity,negation}_para.jsonl``), so a wrong-fact condition differs
from the reference only by the injected fact. The faithful control is draw B
from step 2. Parents whose draw A fails validation fall back to the verbatim S.

Re-derives the parents of step 1 and checks them against ``reference.jsonl``
before calling the editor. Requires an OpenAI-compatible server.

    python -m experiments.qa_faithfulness.build_paraphrase_conditions \\
        --config experiments/qa_faithfulness/configs/build.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .build_verbatim_conditions import (
    check_parents,
    compose,
    derive_items,
    write_rows,
)
from .fact_edits import negate, swap_entity, swap_number
from .paraphrase_benign_control import paraphrase_one

WRONG_FACT_EDITS = {
    "contra_number": lambda s, src, rng: swap_number(s, rng),
    "contra_entity": lambda s, src, rng: swap_entity(s, src, rng),
    "contra_negation": lambda s, src, rng: negate(s, rng),
}


def apply_wrong_fact_edits(
    items: List[Dict], bases: List[Optional[str]], seed: int
) -> Tuple[List[Tuple[Dict, str]], Dict[str, List[Tuple[Dict, str]]]]:
    """Reference rows (item, base) and, per wrong-fact condition, (item, edited base).

    ``bases[i]`` is the draw-A paraphrase of item i (None -> verbatim fallback).
    An edit is kept only where it fires and changes the base.
    """
    rng = random.Random(seed)
    reference: List[Tuple[Dict, str]] = []
    wrong: Dict[str, List[Tuple[Dict, str]]] = {k: [] for k in WRONG_FACT_EDITS}
    for it, para in zip(items, bases):
        base = para if para else it["statement"]
        reference.append((it, base))
        for name, edit in WRONG_FACT_EDITS.items():
            y = edit(base, it["source"], rng)
            if y and y != base:
                wrong[name].append((it, y))
    return reference, wrong


def _rows(pairs: List[Tuple[Dict, str]]) -> List[Dict]:
    return [
        {"parent_id": it["parent_id"], "text": compose(it["source"], it["question"], s)}
        for it, s in pairs
    ]


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

    seed_base = int(para["reference_seed_base"])
    with ThreadPoolExecutor(max_workers=int(para.get("workers", 8))) as ex:
        draws = list(
            ex.map(
                lambda it: paraphrase_one(
                    para["base_url"], para["model"], it["parent_id"], it["statement"], seed_base
                ),
                items,
            )
        )
    n_valid = sum(1 for d in draws if d)
    print(f"[qa] draw A: {n_valid}/{len(items)} valid (verbatim fallback otherwise)")

    reference, wrong = apply_wrong_fact_edits(items, draws, int(para["contra_edit_seed"]))
    counts = {k: len(v) for k, v in wrong.items()}
    print(f"[qa] wrong-fact counts: {counts}")
    if min(counts.values()) < int(para["min_contra_items"]):
        raise SystemExit(f"a wrong-fact condition has < {para['min_contra_items']} items: {counts}")

    write_rows(texts_dir / "reference_paraphrase.jsonl", _rows(reference))
    for name, pairs in wrong.items():
        write_rows(texts_dir / f"{name}_para.jsonl", _rows(pairs))
    (texts_dir / "paraphrase_conditions_audit.json").write_text(
        json.dumps(
            {
                "n_parents": len(items),
                "draw_A_valid": n_valid,
                "contra_counts": counts,
                "edit_seed": int(para["contra_edit_seed"]),
                "reference": f"paraphrase draw A (seed base {seed_base})",
                "faithful_control": "benign_paraphrase, draw B (step 2)",
            },
            indent=2,
        )
    )
    print(f"[qa] wrote reference_paraphrase + contra_*_para -> {texts_dir}")


if __name__ == "__main__":
    main()
