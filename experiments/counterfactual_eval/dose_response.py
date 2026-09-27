"""Dose-response / monotonicity report for the comprehensive meta-eval set.

The paper claims the metric responds *monotonically* to increasing perturbation
strength (evidence it tracks coherence, not a surface proxy). This reads the
per-condition selectivity output and the evaluation set manifest, joins each condition
back to its (family, dose) coordinate, and per (encoder, family) reports:

  * the dose-response curve  z_harmful(dose)  (long table, plot-ready);
  * a Spearman rank correlation between dose and z_harmful (and dose and S);
  * whether the curve is monotone non-decreasing.

    python -m experiments.counterfactual_eval.dose_response \
        --corpus-dir outputs/counterfactual/meta_eval

Writes ``dose_response/dose_response_long.csv`` and
``dose_response/monotonicity.csv`` under the corpus dir.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rho = Pearson on ranks (average ranks for ties). NaN if <2 points
    or a variable is constant."""
    if len(x) < 2:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(values: List[float]) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _monotone_nondecreasing(y: List[float], tol: float = 1e-9) -> bool:
    return all(b - a >= -tol for a, b in zip(y, y[1:]))


def run(corpus_dir: str) -> None:
    corpus = Path(corpus_dir)
    manifest = json.loads((corpus / "conditions_manifest.json").read_text())
    meta = {c["name"]: c for c in manifest["conditions"]}

    sel_path = corpus / "selectivity" / "selectivity_by_family.csv"
    if not sel_path.exists():
        raise SystemExit(f"no selectivity output at {sel_path}; run selectivity.py first")
    with open(sel_path) as fh:
        sel_rows = list(csv.DictReader(fh))

    long_rows: List[Dict] = []
    # group points by (encoder, family)
    curves: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in sel_rows:
        cond = meta.get(r["family"])  # selectivity "family" col == the condition name
        if cond is None or cond.get("dose_kind") in (None, "none"):
            continue
        fam = cond["family"]
        point = {
            "encoder": r["encoder"],
            "family": fam,
            "group": cond.get("group", ""),
            "dose_kind": cond.get("dose_kind", ""),
            "role": cond.get("role", "harmful"),
            "condition": r["family"],
            "dose": float(cond["dose"]),
            "z_harmful": float(r["z_harmful"]),
            "z_benign": float(r["z_benign"]),
            "selectivity": float(r["selectivity"]),
            "selective": r.get("selective", ""),
        }
        long_rows.append(point)
        curves[(r["encoder"], fam)].append(point)

    long_rows.sort(key=lambda p: (p["encoder"], p["family"], p["dose"]))

    mono_rows: List[Dict] = []
    for (enc, fam), pts in sorted(curves.items()):
        pts = sorted(pts, key=lambda p: p["dose"])
        doses = [p["dose"] for p in pts]
        zs = [p["z_harmful"] for p in pts]
        sels = [p["selectivity"] for p in pts]
        mono_rows.append(
            {
                "encoder": enc,
                "family": fam,
                "group": pts[0]["group"],
                "dose_kind": pts[0]["dose_kind"],
                "role": pts[0]["role"],
                "n_levels": len(pts),
                "spearman_z_dose": round(_spearman(doses, zs), 4),
                "spearman_sel_dose": round(_spearman(doses, sels), 4),
                "monotone_increasing": _monotone_nondecreasing(zs),
                "z_at_min_dose": round(zs[0], 4),
                "z_at_max_dose": round(zs[-1], 4),
                "dose_min": doses[0],
                "dose_max": doses[-1],
            }
        )

    out_dir = corpus / "dose_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "dose_response_long.csv", long_rows)
    _write_csv(out_dir / "monotonicity.csv", mono_rows)

    for row in mono_rows:
        flag = "MONO↑" if row["monotone_increasing"] else "non-mono"
        print(
            f"[dose] {row['encoder']}/{row['family']} ({row['dose_kind']}, "
            f"n={row['n_levels']}): rho(z,dose)={row['spearman_z_dose']:+.2f} "
            f"z {row['z_at_min_dose']:.1f}->{row['z_at_max_dose']:.1f} [{flag}]",
            flush=True,
        )
    print(f"[dose] COMPLETE -> {out_dir}", flush=True)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        print(f"[dose] no rows for {path.name}", flush=True)
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    args = parser.parse_args()
    run(args.corpus_dir)


if __name__ == "__main__":
    main()
