"""Data-quality inspection for the comprehensive meta-eval set.

Runs on CPU with no model, right after ``build_corpus.py``. It verifies the
evaluation set is clean BEFORE any expensive featurization and dumps human-readable
samples so a person can eyeball every family. Two outputs under the corpus dir:

  * ``qc/qc_report.md`` — per-condition and per-family stats + PASS/WARN/FAIL:
      - row counts and per-dose count balance (families should be dose-balanced);
      - char-level edit fraction and #changed sentences (recomputed vs the parent);
      - INPUT-dose monotonicity (stronger dose => more actual change);
      - normalized edit-position distribution + last-decile mass (UNIFORM placement
        check — a localized family should not pile edits at the end / recency);
      - length ratio, no-op (text==parent) rate, truncation rate.
  * ``qc/qc_samples.md`` — ``--samples`` random rows per condition, showing the
    parent snippet, the perturbed snippet, and the top changed spans.

Hard failures (missing/empty condition, high no-op or truncation rate, severe
count imbalance in a deterministic family) exit non-zero so the pipeline STOPS
before wasting GPU. Position skew and non-monotone INPUT dose are WARN only.

    python -m experiments.counterfactual_eval.inspect_corpus \
        --corpus-dir outputs/counterfactual/meta_eval [--samples 5]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

from chord.text import sentences, words

# families whose edits REPLACE a localized span (so uniform placement is testable
# via changed-span midpoints). Order families (permute/shuffle) rearrange the whole
# passage, and repetition INSERTS duplicates (its difflib midpoint is the insertion
# point, not a replaced region), so both are exempt from the placement-skew check.
_PLACEMENT_CHECK_GROUPS = {"relation", "discourse", "dlm_mix", "corpus_mix", "benign"}
# thresholds
_STAT_SAMPLE = 150  # rows/condition for the expensive char-diff stats
_NOOP_FAIL = 0.01  # >1% no-op rows in a condition -> FAIL
_TRUNC_FAIL = 0.05  # >5% severely short rows -> FAIL
_TRUNC_WARN = 0.01
_LEN_LOW, _LEN_HIGH = 0.40, 2.50  # word-length ratio bounds (severe outside)
_COUNT_IMBAL_WARN = 0.05  # per-dose count spread within a family
_COUNT_IMBAL_FAIL = 0.20
_LASTDECILE_WARN = 0.22  # >22% of localized edits in the final 10% of the doc


def _read_rows(path: Path) -> List[Dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _edit_stats(parent: str, text: str) -> Tuple[float, List[float], int]:
    """(char-level edit fraction, normalized changed-span midpoints, #changed sentences)."""
    sm = SequenceMatcher(a=parent, b=text, autojunk=False)
    edit_frac = 1.0 - sm.ratio()
    positions: List[float] = []
    denom = max(len(parent), 1)
    for op, i1, i2, _j1, _j2 in sm.get_opcodes():
        if op == "equal":
            continue
        positions.append(((i1 + i2) / 2.0) / denom)
    ps, ts = set(sentences(parent)), set(sentences(text))
    changed_sent = len(ps - ts)
    return edit_frac, positions, changed_sent


def _changed_spans(parent: str, text: str, cap: int = 3) -> List[Tuple[str, str]]:
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(a=parent, b=text, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        out.append((parent[i1:i2], text[j1:j2]))
        if len(out) >= cap:
            break
    return out


def run(corpus_dir: str, n_samples: int, seed: int) -> int:
    corpus = Path(corpus_dir)
    manifest = json.loads((corpus / "conditions_manifest.json").read_text())
    conds = manifest["conditions"]
    cond_dir = corpus / "texts" / "conditions"

    # parent text lookup (for edit stats): clean_candidates carries parent_id->text.
    parent_text: Dict[str, str] = {}
    cc = corpus / "texts" / "clean_candidates.jsonl"
    if cc.exists():
        for r in _read_rows(cc):
            parent_text[r.get("parent_id")] = r.get("text", "")

    rng = random.Random(seed)
    per_cond: List[Dict] = []
    sample_blocks: List[str] = []
    fails: List[str] = []
    warns: List[str] = []

    for cond in conds:
        name = cond["name"]
        path = cond_dir / f"{name}.jsonl"
        if not path.exists():
            fails.append(f"{name}: condition file missing ({path})")
            continue
        rows = _read_rows(path)
        n = len(rows)
        if n == 0:
            fails.append(f"{name}: 0 rows")
            continue

        # cheap full-scan checks: no-op / empty / length ratio (needs parent).
        noop = empty = trunc = long = with_parent = 0
        for r in rows:
            t = r.get("text", "")
            if not t:
                empty += 1
                continue
            p = parent_text.get(r.get("parent_id"))
            if p is not None:
                with_parent += 1
                if t == p:
                    noop += 1
                lr = len(words(t)) / max(len(words(p)), 1)
                if lr < _LEN_LOW:
                    trunc += 1
                elif lr > _LEN_HIGH:
                    long += 1
        noop_rate = (noop + empty) / n
        trunc_rate = trunc / max(with_parent, 1)

        # expensive stats on a deterministic subsample (with a recoverable parent).
        pool = [r for r in rows if parent_text.get(r.get("parent_id")) and r.get("text")]
        rng.shuffle(pool)
        stat_rows = pool[:_STAT_SAMPLE]
        edit_fracs, changed_sents, all_pos, len_ratios = [], [], [], []
        for r in stat_rows:
            p = parent_text[r["parent_id"]]
            ef, pos, cs = _edit_stats(p, r["text"])
            edit_fracs.append(ef)
            changed_sents.append(cs)
            all_pos.extend(pos)
            len_ratios.append(len(words(r["text"])) / max(len(words(p)), 1))
        mean_ef = sum(edit_fracs) / len(edit_fracs) if edit_fracs else float("nan")
        mean_cs = sum(changed_sents) / len(changed_sents) if changed_sents else float("nan")
        mean_lr = sum(len_ratios) / len(len_ratios) if len_ratios else float("nan")
        last_decile = (
            (sum(1 for x in all_pos if x >= 0.9) / len(all_pos)) if all_pos else float("nan")
        )

        # per-condition FAIL/WARN. clean_candidates is the identity/parent set
        # (text == parent by construction) -> exempt from no-op/truncation checks.
        # clean_candidates is the raw parent set (identity, no-op by construction),
        # whether tagged kind=clean (control) or the fallback kind=benign anchor.
        is_identity = cond.get("kind") == "clean" or name == "clean_candidates"
        if not is_identity and noop_rate > _NOOP_FAIL:
            fails.append(f"{name}: no-op/empty rate {noop_rate:.1%} (> {_NOOP_FAIL:.0%})")
        if not is_identity and trunc_rate > _TRUNC_FAIL:
            fails.append(
                f"{name}: truncated (len<{_LEN_LOW}) rate {trunc_rate:.1%} (> {_TRUNC_FAIL:.0%})"
            )
        elif not is_identity and trunc_rate > _TRUNC_WARN:
            warns.append(f"{name}: truncated rate {trunc_rate:.1%}")

        per_cond.append(
            {
                "name": name,
                "family": cond.get("family", ""),
                "group": cond.get("group", ""),
                "kind": cond.get("kind", ""),
                "dose": cond.get("dose"),
                "dose_kind": cond.get("dose_kind", ""),
                "n": n,
                "edit_frac": mean_ef,
                "changed_sent": mean_cs,
                "len_ratio": mean_lr,
                "noop_rate": noop_rate,
                "trunc_rate": trunc_rate,
                "last_decile": last_decile,
            }
        )

        # samples
        picks = pool[:n_samples] if pool else rows[:n_samples]
        block = [
            f"### {name}  (kind={cond.get('kind')}, group={cond.get('group')}, "
            f"dose={cond.get('dose')}, n={n})\n"
        ]
        for r in picks:
            p = parent_text.get(r.get("parent_id"), "")
            t = r.get("text", "")
            block.append(f"- **parent** `{r.get('parent_id')}`: {p[:360]}")
            block.append(f"  **perturbed**: {t[:360]}")
            if p:
                spans = _changed_spans(p, t)
                for before, after in spans:
                    block.append(f"    - changed: `{before[:80]}` -> `{after[:80]}`")
            block.append("")
        sample_blocks.append("\n".join(block))

    # ---- per-family aggregation: count balance + INPUT-dose monotonicity ------
    fam_rows: Dict[str, List[Dict]] = defaultdict(list)
    for pc in per_cond:
        if pc["dose_kind"] not in (None, "none"):
            fam_rows[pc["family"]].append(pc)
    fam_report: List[Dict] = []
    for fam, pcs in sorted(fam_rows.items()):
        pcs = sorted(pcs, key=lambda x: x["dose"] if x["dose"] is not None else 0.0)
        counts = [pc["n"] for pc in pcs]
        cmax, cmin = max(counts), min(counts)
        imbal = (cmax - cmin) / cmax if cmax else 0.0
        efs = [pc["edit_frac"] for pc in pcs]
        # monotone non-decreasing INPUT change with dose (tolerant). Only meaningful
        # when the dose is an AMOUNT of change (rate / n_edits / paraphrase strength);
        # for n_donors the amount replaced is fixed and diversity varies instead.
        dose_kind = pcs[0]["dose_kind"]
        amount_dose = dose_kind in ("rate", "n_edits", "paraphrase_strength")
        mono = all(b - a >= -1e-6 for a, b in zip(efs, efs[1:]))
        group = pcs[0]["group"]
        localized = group in _PLACEMENT_CHECK_GROUPS
        ld = [pc["last_decile"] for pc in pcs if pc["last_decile"] == pc["last_decile"]]
        max_ld = max(ld) if ld else float("nan")
        fam_report.append(
            {
                "family": fam,
                "group": group,
                "n_levels": len(pcs),
                "counts": counts,
                "count_imbalance": imbal,
                "edit_fracs": [round(x, 3) for x in efs],
                "input_monotone": mono,
                "max_last_decile": max_ld,
                "localized": localized,
            }
        )
        det = group in ("order", "repetition", "corpus_mix", "dlm_mix")
        if imbal > _COUNT_IMBAL_FAIL and det:
            fails.append(f"{fam}: per-dose count imbalance {imbal:.1%} (counts {counts})")
        elif imbal > _COUNT_IMBAL_WARN:
            warns.append(f"{fam}: per-dose count imbalance {imbal:.1%} (counts {counts})")
        if amount_dose and not mono:
            warns.append(
                f"{fam}: INPUT edit-fraction not monotone with dose ({[round(x, 3) for x in efs]})"
            )
        if localized and max_ld == max_ld and max_ld > _LASTDECILE_WARN:
            warns.append(
                f"{fam}: {max_ld:.0%} of edits in final 10% of doc (uniform-placement skew)"
            )

    # ---- write reports --------------------------------------------------------
    qc_dir = corpus / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
    lines = [
        f"# Meta-eval QC report — **{verdict}**",
        "",
        f"- corpus: `{corpus}`  |  conditions: {len(conds)}  |  "
        f"parents with recoverable text: {len(parent_text)}",
        f"- {len(fails)} FAIL, {len(warns)} WARN",
        "",
    ]
    if fails:
        lines += ["## FAIL", *[f"- ❌ {m}" for m in fails], ""]
    if warns:
        lines += ["## WARN", *[f"- ⚠️ {m}" for m in warns], ""]
    lines += [
        "## Per-family (dose-graded)",
        "",
        "| family | group | levels | counts | imbalance | edit_frac by dose "
        "| input-mono | max last-decile |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in fam_report:
        lines.append(
            f"| {f['family']} | {f['group']} | {f['n_levels']} | {f['counts']} | "
            f"{f['count_imbalance']:.1%} | {f['edit_fracs']} | "
            f"{'yes' if f['input_monotone'] else 'NO'} | "
            f"{f['max_last_decile']:.0%} |"
        )
    lines += [
        "",
        "## Per-condition",
        "",
        "| condition | kind | group | dose | n | edit_frac | chg_sent | len_ratio "
        "| noop% | trunc% | last-decile% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for pc in per_cond:
        lines.append(
            f"| {pc['name']} | {pc['kind']} | {pc['group']} | {pc['dose']} | {pc['n']} | "
            f"{pc['edit_frac']:.3f} | {pc['changed_sent']:.2f} | {pc['len_ratio']:.2f} | "
            f"{pc['noop_rate']:.1%} | {pc['trunc_rate']:.1%} | {pc['last_decile']:.0%} |"
        )
    (qc_dir / "qc_report.md").write_text("\n".join(lines) + "\n")
    (qc_dir / "qc_samples.md").write_text(
        f"# Meta-eval QC samples ({n_samples}/condition, seed {seed})\n\n"
        + "\n".join(sample_blocks)
        + "\n"
    )

    print(f"[qc] {verdict}: {len(fails)} FAIL, {len(warns)} WARN over {len(conds)} conditions")
    for m in fails:
        print(f"[qc]   FAIL {m}")
    for m in warns[:20]:
        print(f"[qc]   WARN {m}")
    print(f"[qc] wrote {qc_dir}/qc_report.md + qc_samples.md", flush=True)
    return 1 if fails else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()
    raise SystemExit(run(args.corpus_dir, args.samples, args.seed))


if __name__ == "__main__":
    main()
