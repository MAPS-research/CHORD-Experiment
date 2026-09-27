"""Build the position-robustness corpus: one topic-drift sentence at three positions.

Paper result: appendix "Effect of Coherence-Failure Position on Detection"
(Figure fig_position_robustness). A last-token representation of a causal LM can
over-weight the end of a passage; this corpus isolates position from content.

For every clean passage that received exactly one inserted topic-drift sentence,
the SAME sentence is re-inserted at three sentence boundaries:

    topicdrift_pos_prefix   after the first sentence
    topicdrift_pos_middle   at the middle sentence boundary
    topicdrift_pos_suffix   after the last sentence

Only the position changes; the inserted text, its normalization and the insertion
method are identical. The human reference, the clean candidates and the benign
paraphrase control are copied verbatim from the source counterfactual corpus, so
the same null and benign contrast apply.

Inputs
    --source-corpus  a built counterfactual corpus directory containing
                     texts/reference.jsonl, texts/clean_candidates.jsonl and
                     texts/conditions/<benign>.jsonl
    --perturbations  LLM-edit records (``chord.data.generate_perturbations`` output);
                     rows with ``perturbation == "topic_drift"`` and a single
                     inserted span are used
Outputs (under --out, default outputs/experiments/position_robustness)
    texts/..., conditions_manifest.json

    python -m experiments.position_robustness.build_corpus \\
        --source-corpus outputs/counterfactual/meta_eval \\
        --perturbations data/interim/perturbations_qwen_editor.jsonl

The paper's corpus was built from an earlier version of the counterfactual set
and its topic-drift edits; the built texts are what the released features
describe.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from chord.text import sentences

POSITIONS = ("prefix", "middle", "suffix")


def read_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_sentence(text: str) -> str:
    """Normalize an extracted span into a standalone sentence (capitalized,
    terminal punctuation). Applied identically at every position."""
    s = text.strip().lstrip(",;:").strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if s[-1] not in ".!?":
        s += "."
    return s


def inject(clean: str, insert: str, position: str) -> Optional[str]:
    parts = sentences(clean)
    if len(parts) < 2:
        return None
    if position == "prefix":
        index = 1
    elif position == "middle":
        index = max(1, len(parts) // 2)
    else:
        index = len(parts)
    new = parts[:index] + [insert] + parts[index:]
    return " ".join(s.strip() for s in new)


def single_insert_topic_drift(path: Path, cap: int, seed: int) -> List[Dict]:
    """(parent id, clean text, inserted sentence) for topic-drift edits that add
    exactly one span and change nothing else."""
    rows: List[Dict] = []
    for record in read_jsonl(path):
        if record.get("perturbation") != "topic_drift":
            continue
        spans = record.get("changed_spans") or []
        inserted = [s for s in spans if s.get("before", "") == "" and s.get("after", "")]
        if len(spans) != 1 or len(inserted) != 1:
            continue
        clean = (record.get("clean_text") or "").strip()
        insert = as_sentence(inserted[0]["after"])
        if clean and insert:
            rows.append(
                {"parent_id": record.get("parent_sample_id"), "clean": clean, "insert": insert}
            )
    random.Random(seed).shuffle(rows)
    return rows[:cap]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source-corpus", required=True)
    ap.add_argument("--perturbations", required=True)
    ap.add_argument("--out", default="outputs/experiments/position_robustness")
    ap.add_argument(
        "--benign-condition",
        default="benign_paraphrase",
        help="file stem of the benign control under texts/conditions/ in the source corpus",
    )
    ap.add_argument("--cap", type=int, default=1500, help="maximum number of parents")
    ap.add_argument("--seed", type=int, default=20260618)
    args = ap.parse_args()

    source, out = Path(args.source_corpus), Path(args.out)
    for name in ("reference", "clean_candidates"):
        write_jsonl(out / "texts" / f"{name}.jsonl", read_jsonl(source / "texts" / f"{name}.jsonl"))
    benign = read_jsonl(source / "texts" / "conditions" / f"{args.benign_condition}.jsonl")
    write_jsonl(out / "texts" / "conditions" / "benign_paraphrase.jsonl", benign)

    edits = single_insert_topic_drift(Path(args.perturbations), args.cap, args.seed)
    print(f"[position] topic-drift parents with a single inserted sentence: {len(edits)}")

    conditions: List[Dict] = []
    for position in POSITIONS:
        rows = []
        for edit in edits:
            text = inject(edit["clean"], edit["insert"], position)
            if text is not None:
                rows.append(
                    {
                        "parent_id": edit["parent_id"],
                        "text": text,
                        "position": position,
                        "insert": edit["insert"],
                    }
                )
        name = f"topicdrift_pos_{position}"
        write_jsonl(out / "texts" / "conditions" / f"{name}.jsonl", rows)
        conditions.append(
            {
                "name": name,
                "experiment": "position",
                "op": "topic_drift_inject",
                "group": "position",
                "criterion": name,
                "kind": "harmful",
                "position": position,
                "n": len(rows),
            }
        )
        print(f"[position]   {name}: {len(rows)} texts")
    conditions.append(
        {
            "name": "benign_paraphrase",
            "experiment": "benign",
            "op": "paraphrase",
            "group": "benign",
            "kind": "benign",
        }
    )
    conditions.append(
        {"name": "clean_candidates", "experiment": "control", "op": "none", "kind": "clean"}
    )
    manifest = {
        "corpus": "position_probe",
        "n_reference": len(read_jsonl(out / "texts" / "reference.jsonl")),
        "source": "single topic-drift sentence re-inserted at three positions",
        "conditions": conditions,
    }
    (out / "conditions_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[position] wrote {out / 'conditions_manifest.json'}")


if __name__ == "__main__":
    main()
