"""Step 4 of the QA-faithfulness evaluation set: lay out the texts for scoring (CPU).

Copies the per-condition texts into the evaluation-set layout that
``experiments.counterfactual_eval.featurize`` / ``selectivity`` read, and writes its
``conditions_manifest.json``:

  texts/reference.jsonl         <- reference_paraphrase.jsonl (draw A)
  texts/clean_candidates.jsonl  <- benign_reword.jsonl
  texts/conditions/benign_paraphrase.jsonl       faithful control (draw B)
  texts/conditions/contra_{number,entity,negation}_para.jsonl   wrong facts

    python -m experiments.qa_faithfulness.assemble_evalset \\
        --config experiments/qa_faithfulness/configs/build.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

# evaluation-set file -> file written by steps 1-3
LAYOUT = {
    "reference.jsonl": "reference_paraphrase.jsonl",
    "clean_candidates.jsonl": "benign_reword.jsonl",
    "conditions/benign_paraphrase.jsonl": "benign_paraphrase.jsonl",
    "conditions/contra_number_para.jsonl": "contra_number_para.jsonl",
    "conditions/contra_entity_para.jsonl": "contra_entity_para.jsonl",
    "conditions/contra_negation_para.jsonl": "contra_negation_para.jsonl",
}

CONDITIONS = [
    {"name": "benign_paraphrase", "family": "benign_paraphrase", "group": "benign",
     "experiment": "benign", "kind": "benign", "role": "benign"},
    {"name": "contra_number_para", "family": "number", "group": "wrong_fact",
     "experiment": "faithfulness", "kind": "harmful", "role": "harmful"},
    {"name": "contra_entity_para", "family": "entity", "group": "wrong_fact",
     "experiment": "faithfulness", "kind": "harmful", "role": "harmful"},
    {"name": "contra_negation_para", "family": "negation", "group": "wrong_fact",
     "experiment": "faithfulness", "kind": "harmful", "role": "harmful"},
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--texts-dir", default=None, help="override texts_dir")
    parser.add_argument("--evalset-dir", default=None, help="override evalset_dir")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    texts_dir = Path(args.texts_dir or cfg["texts_dir"])
    evalset_dir = Path(args.evalset_dir or cfg["evalset_dir"])

    missing = [src for src in LAYOUT.values() if not (texts_dir / src).exists()]
    if missing:
        raise SystemExit(f"missing in {texts_dir}: {missing} (run steps 1-3 first)")
    for dst, src in LAYOUT.items():
        target = evalset_dir / "texts" / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        shutil.copyfile(texts_dir / src, target)
    manifest = {
        "corpus": "qa_faithfulness",
        "seed": int(cfg.get("evalset_seed", 20260920)),
        "conditions": CONDITIONS,
    }
    (evalset_dir / "conditions_manifest.json").write_text(json.dumps(manifest, indent=1))
    counts = {
        dst: sum(1 for _ in open(evalset_dir / "texts" / dst, encoding="utf-8")) for dst in LAYOUT
    }
    print(f"[qa] evaluation set at {evalset_dir}: {counts}")


if __name__ == "__main__":
    main()
