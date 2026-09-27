from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from typing import Dict, List, Tuple

from chord.data.perturbations.llm import derive_changed_spans
from chord.data.perturbations.validation import validate_record
from chord.utils.config import load_config, resolve_path
from chord.utils.io import load_records, write_jsonl
from chord.utils.records import PassageRecord

SEVERITY_ORDER = {"mild": 0, "moderate": 1, "severe": 2}


def reject_duplicate_outputs_within_parent(
    accepted: List[Dict], rejected: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    grouped = defaultdict(list)
    for row in accepted:
        key = (row["split"], row.get("parent_sample_id"), row["text_hash"])
        grouped[key].append(row)

    filtered = []
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda row: (
                row["perturbation"],
                SEVERITY_ORDER.get(row["severity"], 99),
                row["sample_id"],
            ),
        )
        winner, *duplicates = ordered
        winner_validation = dict(winner["validation"])
        winner_validation["unique_output_within_parent"] = True
        filtered.append({**winner, "validation": winner_validation})
        for row in duplicates:
            validation = dict(row["validation"])
            validation["unique_output_within_parent"] = False
            validation["passed"] = False
            rejected.append({**row, "validation": validation})
    return filtered, rejected


def run(config_path: str) -> None:
    config = load_config(config_path)
    source_values = config["input"]["perturbations"]
    if not isinstance(source_values, list):
        source_values = [source_values]
    sources = [resolve_path(config, value) for value in source_values]
    accepted_path = resolve_path(config, config["output"]["accepted"])
    rejected_path = resolve_path(config, config["output"]["rejected"])
    accepted = []
    rejected = []
    edit_bounds = config.get("edit_ratio_bounds", {})
    source_rows = []
    for source in sources:
        source_rows.extend(load_records(source))
    by_sample_id = {row["sample_id"]: row for row in source_rows}
    for row in by_sample_id.values():
        record = PassageRecord.from_dict(row)
        if record.generator != "rule" and record.perturbation != "clean":
            record = replace(
                record,
                changed_spans=derive_changed_spans(record.clean_text, record.perturbed_text or ""),
            )
        validation = dict(record.validation)
        perturbation_bounds = edit_bounds.get(record.perturbation, {})
        bounds = perturbation_bounds.get(record.severity)
        validation.update(
            validate_record(
                record,
                min_length_ratio=float(config.get("min_length_ratio", 0.5)),
                max_length_ratio=float(config.get("max_length_ratio", 1.75)),
                edit_ratio_bounds=bounds,
                enforce_edit_ratio_bounds=bool(config.get("enforce_edit_ratio_bounds", False)),
                require_operation_evidence=bool(config.get("require_operation_evidence", False)),
            )
        )
        updated = replace(record, validation=validation).to_dict()
        (accepted if validation["passed"] else rejected).append(updated)
    if config.get("reject_duplicate_outputs_within_parent", False):
        accepted, rejected = reject_duplicate_outputs_within_parent(accepted, rejected)
    write_jsonl(accepted_path, accepted)
    write_jsonl(rejected_path, rejected)
    print(f"accepted {len(accepted)}; rejected {len(rejected)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
