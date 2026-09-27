"""Step 1 of the QA-faithfulness evaluation set: parents and verbatim conditions (CPU).

For each SQuAD (context C, question Q, gold answer A) it takes the supporting
sentence S (the sentence of C containing A) and, when at least one fluent
factual edit applies, writes per condition one JSON row ``{parent_id, text}``
where ``text`` is the source-conditioned prompt (``compose``):

  reference         (C, Q, S)                   faithful
  benign_reword     (C, Q, synonym reword of S) faithful; the evaluation set's
                                                clean candidates
  contra_number     (C, Q, S with a number swapped)   wrong w.r.t. C
  contra_entity     (C, Q, S with two entities swapped)
  contra_negation   (C, Q, S negated)

Deterministic from ``seed``. The later steps re-derive the same parents with
``derive_items`` (same RNG consumption order) and check them against
``reference.jsonl``.

    python -m experiments.qa_faithfulness.build_verbatim_conditions \\
        --config experiments/qa_faithfulness/configs/build.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from chord.text import sentences

from .fact_edits import (
    benign_reword,
    cap_source,
    clean_claim,
    negate,
    swap_entity,
    swap_number,
)

HARMFUL = ["contra_number", "contra_entity", "contra_negation"]
CONDITIONS = ["reference", "benign_reword"] + HARMFUL


def compose(source: str, question: str, statement: str) -> str:
    """The source-conditioned prompt every encoder reads (final-token readout)."""
    return (
        f'Source: "{source}"\nQuestion: "{question}"\nStatement: "{statement}"\n'
        "Judging only whether the statement is faithful to and consistent with the "
        "source, in one word:"
    )


def support_sentence(context: str, answer: str) -> Optional[str]:
    """The sentence of the context that contains the gold answer span."""
    a = answer.strip()
    if not a:
        return None
    for s in sentences(context):
        if a.lower() in s.lower():
            s = clean_claim(s)
            n_words = len(s.split())
            if 6 <= n_words <= 45 and s.endswith((".", "!", "?", "”", '"')):
                return s
    return None


def derive_items(squad_path: str, max_source_chars: int, limit: int, seed: int) -> List[Dict]:
    """Parents in build order with their verbatim edits.

    The RNG is consumed exactly as the original build did (number swap, entity
    swap, negation, then the benign reword for kept parents), so every step
    that calls this recovers identical parents and edits.
    """
    rng = random.Random(seed)
    data = json.load(open(squad_path, encoding="utf-8"))["data"]
    items: List[Dict] = []
    for article in data:
        for para in article["paragraphs"]:
            full = para["context"]
            source = cap_source(full, max_source_chars)
            for qa in para["qas"]:
                if len(items) >= limit:
                    break
                if not qa.get("answers"):
                    continue
                question = qa["question"].strip()
                answer = qa["answers"][0]["text"].strip()
                statement = support_sentence(full, answer)
                if not statement or answer.lower() not in source.lower():
                    continue  # need S, and A inside the (capped) source
                y_num = swap_number(statement, rng)
                y_ent = swap_entity(statement, source, rng)
                y_neg = negate(statement, rng)
                if not (y_num or y_ent or y_neg):
                    continue  # need at least one fluent factual edit
                items.append(
                    {
                        "parent_id": f"squad:{len(items) + 1:06d}",
                        "source": source,
                        "question": question,
                        "answer": answer,
                        "statement": statement,
                        "benign_reword": benign_reword(statement, rng),
                        "contra_number": y_num,
                        "contra_entity": y_ent,
                        "contra_negation": y_neg,
                    }
                )
    return items


def check_parents(items: List[Dict], reference_path: Path) -> None:
    """Abort unless ``items`` reproduce the written ``reference.jsonl`` exactly."""
    rows = [json.loads(line) for line in open(reference_path, encoding="utf-8")]
    if len(rows) != len(items):
        raise SystemExit(f"parent count {len(items)} != {len(rows)} in {reference_path}")
    for row, it in zip(rows, items):
        expected = compose(it["source"], it["question"], it["statement"])
        if row["parent_id"] != it["parent_id"] or row["text"] != expected:
            raise SystemExit(f"parent mismatch at {it['parent_id']} vs {reference_path}")
    print(f"[qa] parent identity check passed ({len(items)} parents)")


def write_rows(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--texts-dir", default=None, help="override texts_dir")
    parser.add_argument("--squad", default=None, help="override the SQuAD path")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    texts_dir = Path(args.texts_dir or cfg["texts_dir"])
    items = derive_items(
        args.squad or cfg["squad"],
        int(cfg["max_source_chars"]),
        int(cfg["limit"]),
        int(cfg["seed"]),
    )

    counts = {}
    for cond in CONDITIONS:
        rows = []
        for it in items:
            statement = it["statement"] if cond == "reference" else it[cond]
            if statement:
                rows.append(
                    {
                        "parent_id": it["parent_id"],
                        "text": compose(it["source"], it["question"], statement),
                    }
                )
        write_rows(texts_dir / f"{cond}.jsonl", rows)
        counts[cond] = len(rows)

    examples = [
        {
            "q": it["question"],
            "gold": it["answer"],
            "support": it["statement"],
            "num": it["contra_number"],
            "ent": it["contra_entity"],
            "neg": it["contra_negation"],
        }
        for it in items[:12]
    ]
    (texts_dir / "verbatim_build_audit.json").write_text(
        json.dumps({"seed": int(cfg["seed"]), "counts": counts, "examples": examples}, indent=2)
    )
    print("[qa] verbatim conditions:", counts, "->", texts_dir)


if __name__ == "__main__":
    main()
