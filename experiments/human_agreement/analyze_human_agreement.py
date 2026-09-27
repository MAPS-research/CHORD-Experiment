"""Correlate every metric's 8-setting ranking with the MAUVE human BT scores.

Reads the score CSVs produced on the judged-corpora lanes plus bt_scores.csv
(build_mauve_human_eval.py) and reports Spearman rho / Kendall tau between each
metric and the Bradley-Terry scores for q1 (interesting), q2 (sensible) and
q3 (human-like), over the 8 model settings (human dropped, as in the MAUVE
notebook).

Metric orientation ("higher = better" before correlating):
- CHORD (27B / 2B student) and MMD-MiniLM (Chan) z, FBD: negated (divergences).
- MAUVE (gpt2 / electra): used as-is.
- gen-PPL and unigram entropy are TWO-SIDED (a candidate can be "better than
  human" by being degenerate — Table 2's gamed gen-PPL), so they are oriented
  as -|value - value(human-judged lane)|. The signed raw Spearman is reported
  alongside for transparency.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd
from scipy import stats

HUMAN_RUN = "human-judged"

# (metric label, source file, filter column/value, value column, orientation)
# orientation: "neg" | "pos" | "abs_human" (two-sided, anchored at human lane)
SOURCES = [
    ("CHORD-Qwen3.5-27B", "chord_scores_qwen35-27b-prompteol-coherence-l62.csv", None, "neg"),
    ("CHORD-Qwen3.5-2B", "chord_scores_chord-qwen3.5-2b-student.csv", None, "neg"),
    ("MMD-MiniLM-Chan", "chord_scores_minilm-cls.csv", None, "neg"),
    ("MAUVE-gpt2", "distribution_metrics.csv", "mauve", "pos"),
    ("MAUVE-electra", "distribution_metrics.csv", "mauve_electra", "pos"),
    ("FBD-BERT", "distribution_metrics.csv", "fbd_bert", "neg"),
    ("gen-PPL", "generation_metrics.csv", "gen_ppl", "abs_human"),
    ("unigram-entropy", "generation_metrics.csv", "unigram_entropy", "abs_human"),
]


def load_metric(scores_dir: Path, filename: str, metric: str | None) -> Dict[str, float]:
    df = pd.read_csv(scores_dir / filename)
    if metric is not None:
        df = df[df["metric"] == metric]
    return dict(zip(df["run_id"], df["value"].astype(float)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="outputs/casestudy/human_agreement")
    args = ap.parse_args()
    root = Path(args.dir).resolve()
    scores_dir = root / "scores"
    bt = pd.read_csv(root / "bt_scores.csv")
    model_runs = sorted(set(bt["run_id"]) - {HUMAN_RUN})

    rows: List[Dict] = []
    table: List[Dict] = []
    for label, filename, metric, orientation in SOURCES:
        values = load_metric(scores_dir, filename, metric)
        missing = [r for r in model_runs if r not in values]
        if missing:
            print(f"[analyze] {label}: missing runs {missing}; skipped")
            continue
        raw = {r: values[r] for r in model_runs}
        human_value = values.get(HUMAN_RUN)
        if orientation == "neg":
            oriented = {r: -v for r, v in raw.items()}
        elif orientation == "pos":
            oriented = dict(raw)
        else:  # abs_human
            if human_value is None:
                print(f"[analyze] {label}: human lane missing; skipped")
                continue
            oriented = {r: -abs(v - human_value) for r, v in raw.items()}
        table.append({"metric": label, "human_lane": human_value, **raw})
        for question in sorted(set(bt["question"])):
            b = bt[(bt["question"] == question) & (bt["run_id"].isin(model_runs))]
            b = b.set_index("run_id")["bt_score"]
            x = [oriented[r] for r in model_runs]
            xr = [raw[r] for r in model_runs]
            y = [float(b[r]) for r in model_runs]
            rho, rho_p = stats.spearmanr(x, y)
            tau, tau_p = stats.kendalltau(x, y)
            raw_rho, _ = stats.spearmanr(xr, y)
            rows.append(
                {
                    "metric": label,
                    "question": question,
                    "orientation": orientation,
                    "spearman": round(float(rho), 4),
                    "spearman_p": round(float(rho_p), 4),
                    "kendall": round(float(tau), 4),
                    "kendall_p": round(float(tau_p), 4),
                    "spearman_raw_signed": round(float(raw_rho), 4),
                    "n_settings": len(model_runs),
                }
            )

    out_csv = scores_dir / "human_agreement_correlations.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(table).to_csv(scores_dir / "human_agreement_metric_values.csv", index=False)

    print("\nPer-lane metric values (human sanity column first):")
    print(pd.DataFrame(table).to_string(index=False))
    print("\nCorrelations with Bradley-Terry human scores (8 settings):")
    piv = pd.DataFrame(rows).pivot(index="metric", columns="question", values="spearman")
    print(piv.to_string())
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
