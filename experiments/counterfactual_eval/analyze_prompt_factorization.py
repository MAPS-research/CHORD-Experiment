"""Aggregate the frozen prompt-factorization experiment without selecting a winner.

Inputs are the per-condition, per-half rows from ``prompt_split_eval`` and the
factor metadata frozen in the encoder YAML.  Outputs retain every encoder cell
and summarize wording variation by median/range.  Detection counts are a
secondary diagnostic; the primary value is harmful-minus-benign selectivity.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import yaml


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _median(values: Iterable[float]) -> float:
    return float(np.median(list(values)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    meta = {e["name"]: dict(e.get("factors", {})) for e in cfg["encoders"]}
    analysis = cfg.get("analysis", {})
    neutral = analysis.get("neutral_encoder")
    raw = list(analysis.get("raw_encoders", []))
    if neutral:
        meta[neutral] = {"attribute": "neutral", "wording": "fixed", "format": "compact"}
    for enc in raw:
        meta[enc] = {"attribute": "raw", "wording": enc.rsplit("-", 1)[-1], "format": "none"}

    rows = _read_csv(Path(args.split_csv))
    unknown = sorted({r["encoder"] for r in rows} - set(meta))
    if unknown:
        raise ValueError(f"split CSV contains encoders absent from factor metadata: {unknown}")

    # One row per encoder x half x intervention group.  Benign-dose rows remain
    # a dedicated group rather than being mixed with harmful groups.
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        group = "benign" if r["kind"] == "benign_dose" else r["group"]
        grouped[(r["encoder"], r["half"], group)].append(r)

    cells: List[Dict] = []
    for (enc, half, group), rs in sorted(grouped.items()):
        sels = [float(r["selectivity"]) for r in rs]
        zs = [float(r["z_harmful"]) for r in rs]
        factors = meta[enc]
        cells.append(
            {
                "encoder": enc,
                "half": half,
                "attribute": factors["attribute"],
                "wording": factors["wording"],
                "format": factors["format"],
                "intervention_group": group,
                "mean_selectivity": round(float(np.mean(sels)), 4),
                "median_selectivity": round(_median(sels), 4),
                "mean_z_harmful": round(float(np.mean(zs)), 4),
                "n_conditions": len(rs),
                "n_selective": sum(str(r["selective"]).lower() == "true" for r in rs),
            }
        )

    # Family summaries expose exact-wording variance.  Panel A has three
    # wordings per attribute/compact cell; Panel B has three per format.
    fam_groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in cells:
        fam_groups[(r["half"], r["attribute"], r["format"], r["intervention_group"])].append(r)
    families: List[Dict] = []
    for (half, attr, fmt, group), rs in sorted(fam_groups.items()):
        vals = [float(r["mean_selectivity"]) for r in rs]
        families.append(
            {
                "half": half,
                "attribute": attr,
                "format": fmt,
                "intervention_group": group,
                "median_selectivity": round(_median(vals), 4),
                "min_selectivity": round(min(vals), 4),
                "max_selectivity": round(max(vals), 4),
                "n_wordings": len(vals),
                "n_wordings_majority_selective": sum(
                    int(r["n_selective"]) * 2 >= int(r["n_conditions"]) for r in rs
                ),
            }
        )

    out = Path(args.out_dir)
    _write_csv(out / "prompt_factorization_cells.csv", cells)
    _write_csv(out / "prompt_factorization_families.csv", families)

    primary = str(analysis.get("primary_half", "confirmation"))
    print(f"\nPrimary half: {primary}")
    print("attribute/format x intervention-group median [min,max] across wordings")
    for r in families:
        if r["half"] != primary:
            continue
        print(
            f"{r['attribute']:10s} {r['format']:8s} {r['intervention_group']:12s} "
            f"{r['median_selectivity']:9.2f} "
            f"[{r['min_selectivity']:.2f},{r['max_selectivity']:.2f}] "
            f"n={r['n_wordings']}"
        )
    print(f"\nwrote {out / 'prompt_factorization_cells.csv'}")
    print(f"wrote {out / 'prompt_factorization_families.csv'}")


if __name__ == "__main__":
    main()
