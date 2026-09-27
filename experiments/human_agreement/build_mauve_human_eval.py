"""Build the MAUVE human-evaluation correlation lanes (appendix experiment).

Source data = the released MAUVE human study (Pillutla et al. 2021,
`mauve-experiments/human_evaluation.md`): 3,240 pairwise AMT comparisons over
9 players (8 GPT-2 decoding settings + human) on webtext continuations, three
questions (q1 interesting / q2 sensible / q3 human-like).

Data-cut facts established by audit (2026-07-13) — every lane below follows them:

1. Raters were shown the FIRST 256 GPT-2 TOKENS of (35-token webtext prompt +
   completion): the human-player rows reconstruct webtext docs cut at exactly
   35+221 tokens (466/516 exact k=221; remainder are whitespace off-by-a-few or
   docs shorter than 256). `Input.len_*` is the FULL generation length — the
   display was truncated, so metrics must score the displayed 256-token cut,
   not the full generation.
2. The released generations archive contains ONLY gpt2 and gpt2-large (medium
   and xl directories are empty in the tarball) and used prompt_size=10, while
   the human study used prompt_size=35 — the judged completions are NOT in the
   archive (membership test negative). Therefore the only corpora exactly
   aligned with the human judgments are the ones reconstructed from the CSV.
3. Display pipeline = HTML wrapping (<p> paragraphs, <strong> ctx, rare
   list/blockquote markup) + entity escaping. We strip a tag whitelist,
   unescape entities, and collapse all whitespace; the scoring configs apply
   the same `whitespace` normalization to reference and candidates (one surface
   policy for every corpus; MAUVE is sensitive to formatting differences).
4. The study only used prompts whose texts run >= ~200 tokens (min observed
   total length 200), so the reference pool applies the same >=200-token
   filter before the 256-token cut, and excludes every judged prompt idx so
   the human-judged lane shares no document with the reference.

Outputs (under --output-dir):
    judged/<run_id>.jsonl        9 candidate lanes (one text per unique idx)
    reference_judged256.jsonl    held-out webtext docs, same 256-token cut
    bt_scores.csv                Bradley-Terry scores per question (notebook port)
    build_audit.json             counts, reconstruction checks, length stats
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Player name (CSV) -> lane run_id. Order matters: it is the notebook's player
# order (8 model settings, then human).
PLAYERS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("('gpt2', 'p0.9')", "gpt2-p0.9"),
        ("('gpt2', 'p1.0')", "gpt2-p1.0"),
        ("('gpt2-large', 'p0.95')", "gpt2-large-p0.95"),
        ("('gpt2-large', 'p1.0')", "gpt2-large-p1.0"),
        ("('gpt2-medium', 'p0.9')", "gpt2-medium-p0.9"),
        ("('gpt2-medium', 'p1.0')", "gpt2-medium-p1.0"),
        ("('gpt2-xl', 'p0.95')", "gpt2-xl-p0.95"),
        ("('gpt2-xl', 'p1.0')", "gpt2-xl-p1.0"),
        ("human", "human-judged"),
    ]
)

DISPLAY_TAG = re.compile(r"</?(?:p|strong|em|ul|ol|li|blockquote|a|hr)(?:\s[^>]{0,40})?/?>")
PROMPT_TOKENS = 35
DISPLAY_TOKENS = 256  # prompt 35 + completion 221
MIN_DOC_TOKENS = 200
THRESHOLD_TIME = 25  # notebook quality-control filter on Answer.te
QUESTIONS = ("Answer.q1", "Answer.q2", "Answer.q3")


def clean_display(text: str) -> str:
    """Strip the AMT display markup and collapse whitespace."""
    text = DISPLAY_TAG.sub(" ", str(text))
    return " ".join(html.unescape(text).split())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bradley_terry(
    df: pd.DataFrame, field: str, players: List[str], max_iterations: int = 1000
) -> Dict[str, float]:
    """Zermelo iteration, reimplemented from human_eval-compute_BT_scores.ipynb.

    Ties/slight-A share code '1a' (randomized A/B assignment absorbs ties).
    Deterministic uniform init instead of the notebook's np.random.rand: the
    BT likelihood has a unique normalized maximizer, so the fixed point is
    init-independent.
    """
    k = len(players)
    wins = np.zeros((k, k), dtype=np.int64)
    for i, m1 in enumerate(players):
        for j, m2 in enumerate(players):
            if i == j:
                continue
            d1 = df[(df["Input.model_a"] == m1) & (df["Input.model_b"] == m2)]
            d2 = df[(df["Input.model_b"] == m1) & (df["Input.model_a"] == m2)]
            wins[i, j] = d1[field].isin(["1a", "2a"]).sum() + d2[field].isin(["1b", "2b"]).sum()
    wins_per = wins.sum(axis=1)
    ps = np.full(k, 1.0 / k)
    qs = np.zeros_like(ps)
    for _ in range(max_iterations):
        for i in range(k):
            denom = sum((wins[i, j] + wins[j, i]) / (ps[i] + ps[j]) for j in range(k) if j != i)
            qs[i] = wins_per[i] / denom
        ps_new = qs / qs.sum()
        if np.linalg.norm(ps_new - ps, 1) < 1e-16:
            break
        ps = ps_new
    scores = np.log(ps)
    scores -= scores.mean()
    scores *= 100.0
    return dict(zip(players, scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="mauve human-eval anon CSV")
    ap.add_argument("--webtext", required=True, help="webtext.test.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--qwen-tokenizer",
        default="Qwen/Qwen3.5-27B",
        help="tokenizer path for the cap532 body-budget audit ('' to skip)",
    )
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    audit: Dict = {
        "source_csv": str(Path(args.csv).resolve()),
        "source_csv_sha256": sha256_file(Path(args.csv)),
        "source_webtext": str(Path(args.webtext).resolve()),
        "source_webtext_sha256": sha256_file(Path(args.webtext)),
        "display_tokens": DISPLAY_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "min_doc_tokens": MIN_DOC_TOKENS,
        "threshold_time": THRESHOLD_TIME,
    }

    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")

    df = pd.read_csv(args.csv, index_col=0)
    audit["csv_rows"] = int(len(df))
    docs = [json.loads(line) for line in open(args.webtext, encoding="utf-8")]
    judged_idx = set(df["Input.idx"].astype(int))
    audit["judged_prompt_idx"] = len(judged_idx)

    # ---- candidate lanes: exactly the displayed texts, one per unique idx ----
    lane_rows: Dict[str, "OrderedDict[int, str]"] = {p: OrderedDict() for p in PLAYERS}
    distinct_conflicts = 0
    for _, row in df.iterrows():
        ctx = clean_display(row["Input.ctx"])
        for side in ("a", "b"):
            player = row[f"Input.model_{side}"]
            if player not in PLAYERS:
                raise ValueError(f"unexpected player {player!r}")
            idx = int(row["Input.idx"])
            text = ctx + " " + clean_display(row[f"Input.completion{side}"])
            prev = lane_rows[player].get(idx)
            if prev is None:
                lane_rows[player][idx] = text
            elif prev != text:
                distinct_conflicts += 1  # keep first occurrence (row order)
    audit["post_normalization_conflicts"] = distinct_conflicts

    lane_stats = {}
    for player, run_id in PLAYERS.items():
        items = sorted(lane_rows[player].items())
        rows = [{"idx": idx, "text": text} for idx, text in items]
        write_jsonl(out / "judged" / f"{run_id}.jsonl", rows)
        lens = [len(tok.encode(r["text"])) for r in rows]
        lane_stats[run_id] = {
            "n": len(rows),
            "gpt2_tokens_q": [int(q) for q in np.percentile(lens, [0, 25, 50, 75, 100])],
        }
    audit["lanes"] = lane_stats

    # ---- human-lane reconstruction fidelity vs webtext (rule: doc[:256]) ----
    human_rows = [{"idx": idx, "text": text} for idx, text in sorted(lane_rows["human"].items())]
    match = 0
    for r in human_rows:
        ref = " ".join(tok.decode(tok.encode(docs[r["idx"]]["text"])[:DISPLAY_TOKENS]).split())
        a, b = r["text"], ref
        head = min(len(a), len(b)) - 30  # tolerate boundary off-by-a-few
        if head > 0 and a[:head] == b[:head]:
            match += 1
    audit["human_lane_exact_256cut_match"] = {
        "matched": match,
        "total": len(human_rows),
    }

    # ---- reference: held-out docs, same >=200-token filter and 256-token cut ----
    ref_rows = []
    for i, doc in enumerate(docs):
        if i in judged_idx:
            continue
        ids = tok.encode(doc["text"])
        if len(ids) < MIN_DOC_TOKENS:
            continue
        text = " ".join(tok.decode(ids[:DISPLAY_TOKENS]).split())
        ref_rows.append({"idx": i, "text": text})
    write_jsonl(out / "reference_judged256.jsonl", ref_rows)
    audit["reference_n"] = len(ref_rows)

    # ---- optional cap532 body-budget audit under the Qwen tokenizer ----
    if args.qwen_tokenizer:
        try:
            from transformers import AutoTokenizer

            qtok = AutoTokenizer.from_pretrained(args.qwen_tokenizer)
            sample = [r["text"] for r in ref_rows[:200]]
            for run_id in ("gpt2-xl-p1.0", "human-judged"):
                path = out / "judged" / f"{run_id}.jsonl"
                sample += [json.loads(line)["text"] for line in open(path, encoding="utf-8")][:200]
            qlens = [len(qtok(t)["input_ids"]) for t in sample]
            audit["qwen_tokens_sampled"] = {
                "q": [int(q) for q in np.percentile(qlens, [0, 50, 95, 100])],
                "frac_over_body512": float(np.mean(np.asarray(qlens) > 512)),
            }
        except Exception as exc:  # noqa: BLE001 — audit is advisory
            audit["qwen_tokens_sampled"] = f"skipped: {exc}"

    # ---- Bradley-Terry scores (notebook port) ----
    kept = df[df["Answer.te"] > THRESHOLD_TIME]
    audit["bt_annotations_kept"] = int(len(kept))
    players = list(PLAYERS)
    bt_rows = []
    for field in QUESTIONS:
        scores = bradley_terry(kept, field, players)
        for player, score in scores.items():
            bt_rows.append(
                {
                    "question": field.split(".")[-1],
                    "player": player,
                    "run_id": PLAYERS[player],
                    "bt_score": score,
                }
            )
    pd.DataFrame(bt_rows).to_csv(out / "bt_scores.csv", index=False)

    (out / "build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print("\nBT scores (q3 human-like):")
    for row in bt_rows:
        if row["question"] == "q3":
            print(f"  {row['player']:28s} {row['bt_score']:8.2f}")


if __name__ == "__main__":
    main()
