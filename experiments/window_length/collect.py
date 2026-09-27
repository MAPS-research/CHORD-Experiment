"""Collect the window-length scores into tidy per-corpus x length tables.

Paper result: the data behind tab:mauve-length and tab:chord-length (appendix
"Sensitivity to Evaluation Window Length"). Each value is the mean over Table 2's
ten folds (the fold standard deviation is kept alongside). The script also prints
the three-tier check the appendix states for CHORD (human < autoregressive <
diffusion/flow at every length).

Inputs   <out>/L{L}/scores/length_folds.csv   (score_folds.py)
         <out>/L{L}/trunc_stats.json          (build_truncated_corpora.py)
Outputs  <out>/chord_length_sensitivity.csv, <out>/mauve_length_sensitivity.csv

    python -m experiments.window_length.collect --lengths 128,256,384,512,1024
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

AUTOREGRESSIVE = ("GPT2-large-AR", "GPT2-medium-AR")
DIFFUSION_FLOW = ("SEDD-small", "MDLM-OWT", "LangFlow-OWT", "ELF-L-OWT")
HUMAN = "Human-packed"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(base: Path, lengths: List[int]) -> Dict[str, List[Dict]]:
    """{"chord": rows, "mauve": rows}, one row per corpus x length (fold means)."""
    out: Dict[str, List[Dict]] = {"chord": [], "mauve": []}
    for length in lengths:
        folder = base / f"L{length}"
        scores = folder / "scores" / "length_folds.csv"
        if not scores.is_file():
            print(f"[collect] missing {scores}; L{length} skipped")
            continue
        stats = json.loads((folder / "trunc_stats.json").read_text())["runs"]
        tokens: Dict[str, List[float]] = defaultdict(list)
        for s in stats.values():
            tokens[s["model"]].append(s["gen_tok_mean"])
        groups: Dict[str, Dict[str, List[float]]] = OrderedDict()
        family: Dict[str, str] = {}
        for r in read_csv(scores):
            g = groups.setdefault(r["model"], defaultdict(list))
            family[r["model"]] = r["family"]
            for key in ("chord_z", "chord_mmd_x100", "mauve"):
                g[key].append(float(r[key]))
        for model, g in groups.items():
            common = {"model": model, "family": family[model], "length_tokens": length}
            tok = round(mean(tokens[model]), 1)
            out["chord"].append(
                dict(
                    common,
                    chord_z=round(mean(g["chord_z"]), 3),
                    chord_z_fold_std=round(stdev(g["chord_z"]), 3),
                    chord_mmd_x100=round(mean(g["chord_mmd_x100"]), 4),
                    n_folds=len(g["chord_z"]),
                    qwen_tok_mean=tok,
                )
            )
            out["mauve"].append(
                dict(
                    common,
                    mauve=round(mean(g["mauve"]), 4),
                    mauve_fold_std=round(stdev(g["mauve"]), 4),
                    n_folds=len(g["mauve"]),
                    qwen_tok_mean=tok,
                )
            )
    return out


def tier_check(rows: List[Dict], key: str, lower_is_better: bool) -> None:
    by_length: Dict[int, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_length[r["length_tokens"]][r["model"]] = r[key]
    for length, v in sorted(by_length.items()):
        if not all(m in v for m in (HUMAN,) + AUTOREGRESSIVE + DIFFUSION_FLOW):
            continue
        sign = 1 if lower_is_better else -1
        human, ar = sign * v[HUMAN], [sign * v[m] for m in AUTOREGRESSIVE]
        dlm = [sign * v[m] for m in DIFFUSION_FLOW]
        holds = human < min(ar) and max(ar) < min(dlm)
        print(f"  L{length:<5d} human < AR < diffusion/flow: {'holds' if holds else 'broken'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default="outputs/experiments/window_length_folds")
    ap.add_argument("--lengths", default="128,256,384,512,1024")
    args = ap.parse_args()
    base, lengths = Path(args.out), [int(x) for x in args.lengths.split(",") if x.strip()]

    tables = collect(base, lengths)
    for name, key, lower in (("chord", "chord_z", True), ("mauve", "mauve", False)):
        rows = tables[name]
        if not rows:
            continue
        path = base / f"{name}_length_sensitivity.csv"
        write_csv(path, rows)
        print(f"[collect] wrote {path} ({len(rows)} rows)")
        print(f"{name.upper()} tier check")
        tier_check(rows, key, lower_is_better=lower)


if __name__ == "__main__":
    main()
