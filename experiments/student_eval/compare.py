"""Compare a distilled student against the teacher on the two paper lanes.

This is the "did my variant get better?" tool. It reads the same score files the
paper tables are built from and prints, in the paper's own units:

  * TABLE-1 lane (counterfactual evaluation set) — null-standardized z per damage
    family at its highest severity, plus whether the benign-contrast bootstrap
    calls it detected. Students are read from the CONFIRMATION half of
    `prompt_split_eval`, which is how the released student is reported: the
    student's training touched this protocol, so its numbers come from the held
    out half of an honest split, never from the half used for any selection.

  * TABLE-2 lane (unconditional generation) — raw RBF-MMD (x1e-2) per generator,
    the induced ranking, Spearman rank agreement with the teacher, and every
    rank inversion, annotated with the across-seed spread so noise-level swaps
    are not mistaken for ranking errors.

Usage:

    python -m experiments.student_eval.compare \\
        --student chord-qwen3.5-2b-student \\
        --counterfactual-dir outputs/counterfactual/meta_eval \\
        --ladder-dir  outputs/distill/eval_ladder \\
        --teacher-ladder-dir outputs/casestudy/single_fold_chord_27b \\
        --split-suffix _chord_2b_student

Add `--extras` to also print the diagnostic runs that are NOT in the paper table
(the short-document format probe and ELF-B-1024). Those are useful signal for
distillation work but must not be read as paper-facing ranking errors: the
format probe is deliberately format-mismatched from the reference, so a large
value is the intended behavior, not a defect.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path
from typing import Dict, List

TEACHER = "qwen35-27b-prompteol-coherence-l62"

# Table-1 damage families at their highest reported severity, in the paper's
# column order (hardest to easiest).
TABLE1_FAMILIES = [
    ("causal_reverse__dose3", "Causal reversal"),
    ("contradiction__dose3", "Contradiction"),
    ("broken_transition__dose3", "Broken transition"),
    ("topic_drift__dose3", "Topic drift"),
    ("sentence_permutation__r100", "Sentence permutation"),
    ("local_shuffle__r100", "Word shuffle"),
    ("repetition__r100", "Repetition"),
    ("dlm_splice__r050", "DLM mix"),
    ("corpus_mix__r075_d8", "Document mix"),
]
BENIGN = "benign_paraphrase__dose3"

# Generators reported in the unconditional-generation table.
TABLE2_MODELS = [
    "Human-packed",
    "GPT2-large-AR",
    "GPT2-medium-AR",
    "ELF-L-OWT",
    "MDLM-OWT",
    "LangFlow-OWT",
    "SEDD-small",
]
EXTRA_MODELS = ["Real-OWT (short docs)", "ELF-B-OWT (1024)"]


def _rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing score file: {path}")
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def _spearman(a: Dict[str, int], b: Dict[str, int], keys: List[str]) -> float:
    n = len(keys)
    if n < 2:
        return float("nan")
    d2 = sum((a[k] - b[k]) ** 2 for k in keys)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


# ---------------------------------------------------------------- counterfactual lane


def _counterfactual_teacher(counterfactual_dir: Path, encoder: str) -> Dict[str, Dict[str, object]]:
    """Full-counterfactual z per family (the teacher / non-student rows of Table 1)."""
    out = {}
    for r in _rows(counterfactual_dir / "selectivity" / "selectivity_by_family.csv"):
        if r["encoder"] != encoder:
            continue
        out[r["family"]] = {"z": float(r["z_harmful"]), "sel": _truthy(r["selective"])}
    return out


def _counterfactual_student(
    counterfactual_dir: Path, encoder: str, suffix: str, half: str = "confirmation"
) -> Dict[str, Dict[str, object]]:
    """Honest-split z per family (how a student is reported)."""
    path = counterfactual_dir / "selectivity" / f"prompt_split_eval{suffix}.csv"
    out = {}
    for r in _rows(path):
        if r["encoder"] != encoder or r["half"] != half:
            continue
        out[r["family"]] = {"z": float(r["z_harmful"]), "sel": _truthy(r["selective"])}
        # The benign control is a COLUMN here (every harmful row carries the
        # z it was contrasted against), not a family row as in the full evaluation-set
        # CSV. Surface it under the same key so both sides print alike.
        if BENIGN not in out and r.get("z_benign") not in (None, ""):
            out[BENIGN] = {"z": float(r["z_benign"]), "sel": False}
    return out


def _print_counterfactual(teacher: Dict, student: Dict, student_name: str) -> None:
    print("\n" + "=" * 78)
    print(
        "TABLE-1 LANE — counterfactual evaluation set (null-standardized z, higher = "
        "more damage detected)"
    )
    print("=" * 78)
    if not student:
        print("  no student rows found — check --split-suffix / --student")
        return
    print(f"  {'family':22s} {'27B teacher':>14s}   {student_name:>18s}   ratio")
    t_hits = s_hits = t_tot = s_tot = 0
    for cond, label in TABLE1_FAMILIES:
        t, s = teacher.get(cond), student.get(cond)
        if t:
            t_tot += 1
            t_hits += bool(t["sel"])
        if s:
            s_tot += 1
            s_hits += bool(s["sel"])
        tv = f"{t['z']:9.1f} {'*' if t['sel'] else ' '}" if t else "        --  "
        sv = f"{s['z']:13.1f} {'*' if s['sel'] else ' '}" if s else "            --  "
        ratio = f"{s['z'] / t['z']:6.2f}" if t and s and abs(t["z"]) > 1e-9 else "     --"
        flag = (
            "   <-- teacher detects, student does not"
            if (t and s and t["sel"] and not s["sel"])
            else ""
        )
        print(f"  {label:22s} {tv}   {sv}   {ratio}{flag}")
    tb, sb = teacher.get(BENIGN), student.get(BENIGN)
    tv = f"{tb['z']:9.1f}  " if tb else "        --  "
    sv = f"{sb['z']:13.1f}  " if sb else "            --  "
    print(f"  {'benign paraphrase':22s} {tv}   {sv}   (must stay low)")
    print(
        f"\n  detected: teacher {t_hits}/{t_tot}   {student_name} {s_hits}/{s_tot}"
        "   (* = selective vs the benign control)"
    )


# ----------------------------------------------------------------- ladder lane


def _ladder(scores_dir: Path, encoder: str) -> Dict[str, Dict[str, float]]:
    """Mean raw RBF-MMD (x1e-2) per generator, with the across-seed spread."""
    path = scores_dir / "scores" / f"chord_scores_raw_{encoder}.csv"
    agg = collections.defaultdict(list)
    for r in _rows(path):
        agg[r["model"]].append(float(r["value"]) * 100.0)
    out = {}
    for model, vals in agg.items():
        mean = sum(vals) / len(vals)
        spread = (max(vals) - min(vals)) / 2.0 if len(vals) > 1 else 0.0
        out[model] = {"mean": mean, "spread": spread, "n": len(vals)}
    return out


def _print_ladder(
    teacher: Dict, student: Dict, student_name: str, models: List[str], extras: bool
) -> None:
    print("\n" + "=" * 78)
    print("TABLE-2 LANE — unconditional generation (raw RBF-MMD x1e-2, lower = closer to human)")
    print("=" * 78)
    present = [m for m in models if m in teacher and m in student]
    missing = [m for m in models if m not in teacher or m not in student]
    if missing:
        print(f"  (missing from the score files: {missing})")
    if len(present) < 2:
        print("  not enough shared generators to compare")
        return

    rt = {m: i + 1 for i, m in enumerate(sorted(present, key=lambda m: teacher[m]["mean"]))}
    rs = {m: i + 1 for i, m in enumerate(sorted(present, key=lambda m: student[m]["mean"]))}
    print(
        f"  {'generator':24s} {'27B teacher':>16s} {'rank':>5s}   {student_name:>16s} {'rank':>5s}"
    )
    for m in sorted(present, key=lambda m: teacher[m]["mean"]):
        t, s = teacher[m], student[m]
        ts = f"{t['mean']:9.2f}+-{t['spread']:<4.1f}" if t["n"] > 1 else f"{t['mean']:9.2f}     "
        ss = f"{s['mean']:9.2f}+-{s['spread']:<4.1f}" if s["n"] > 1 else f"{s['mean']:9.2f}     "
        print(f"  {m:24s} {ts} {rt[m]:5d}   {ss} {rs[m]:5d}")

    rho = _spearman(rt, rs, present)
    print(f"\n  Spearman rank agreement with the teacher: rho = {rho:.3f}")

    substantive = 0
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            if (teacher[a]["mean"] - teacher[b]["mean"]) * (
                student[a]["mean"] - student[b]["mean"]
            ) >= 0:
                continue
            tgap = abs(teacher[a]["mean"] - teacher[b]["mean"])
            sgap = abs(student[a]["mean"] - student[b]["mean"])
            tnoise = teacher[a]["spread"] + teacher[b]["spread"]
            snoise = student[a]["spread"] + student[b]["spread"]
            noise = tgap <= tnoise or sgap <= snoise
            tag = "within seed spread" if noise else "SUBSTANTIVE"
            substantive += not noise
            print(
                f"    inversion [{tag:18s}] {a} vs {b}: "
                f"teacher {teacher[a]['mean']:.1f}/{teacher[b]['mean']:.1f} "
                f"(gap {tgap:.1f}, spread {tnoise:.1f}) | "
                f"student {student[a]['mean']:.1f}/{student[b]['mean']:.1f} "
                f"(gap {sgap:.1f}, spread {snoise:.1f})"
            )
    print(f"  substantive inversions: {substantive}")

    if extras:
        rows = [m for m in EXTRA_MODELS if m in teacher and m in student]
        if rows:
            print(
                "\n  diagnostic runs NOT in the paper table "
                "(the short-document probe is deliberately format-mismatched,"
            )
            print("   so a large value there is the intended behavior, not a ranking error):")
            for m in rows:
                print(
                    f"    {m:24s} teacher {teacher[m]['mean']:8.2f}   "
                    f"student {student[m]['mean']:8.2f}"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--student", required=True, help="student encoder name")
    ap.add_argument("--teacher", default=TEACHER, help=f"teacher encoder name (default {TEACHER})")
    ap.add_argument("--counterfactual-dir", default="outputs/counterfactual/meta_eval")
    ap.add_argument(
        "--split-suffix",
        default="",
        help="suffix of prompt_split_eval<suffix>.csv holding the student's honest-split rows",
    )
    ap.add_argument("--half", default="confirmation", choices=["confirmation", "selection"])
    ap.add_argument(
        "--ladder-dir",
        default="outputs/distill/eval_ladder",
        help="scores dir for the STUDENT's unconditional-generation run",
    )
    ap.add_argument(
        "--teacher-ladder-dir",
        default="outputs/casestudy/single_fold_chord_27b",
        help="scores dir for the TEACHER's unconditional-generation run",
    )
    ap.add_argument(
        "--extras",
        action="store_true",
        help="also print the diagnostic runs absent from the paper table",
    )
    ap.add_argument("--skip-counterfactual", action="store_true")
    ap.add_argument("--skip-ladder", action="store_true")
    args = ap.parse_args()

    if not args.skip_counterfactual:
        bdir = Path(args.counterfactual_dir)
        _print_counterfactual(
            _counterfactual_teacher(bdir, args.teacher),
            _counterfactual_student(bdir, args.student, args.split_suffix, args.half),
            args.student,
        )
    if not args.skip_ladder:
        _print_ladder(
            _ladder(Path(args.teacher_ladder_dir), args.teacher),
            _ladder(Path(args.ladder_dir), args.student),
            args.student,
            TABLE2_MODELS,
            args.extras,
        )
    print()


if __name__ == "__main__":
    main()
