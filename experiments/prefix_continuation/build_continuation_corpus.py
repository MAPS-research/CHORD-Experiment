"""Build a prefix-to-continuation benchmark on OWT (mirrors TextLDM 2605.07748).

Each held-out OWT passage is tokenized with the GPT-2 tokenizer and split at a fixed
point into a PREFIX (condition) and a human CONTINUATION (the reference target). All
generators continue from the SAME prefixes, so model continuations are directly
comparable to the human continuations under any metric (Figure 5: CHORD vs MAUVE and the
other baselines).

Fixed token lengths keep DLM (SEDD/MDLM) infill batching simple and fast: prefix
L_pre, continuation L_cont, total length = L_pre + L_cont.

Writes (outputs/casestudy/prefix_continuation/):
  prefixes.jsonl   {id, prefix_text, prefix_ids}     # drives conditional generation
  human.jsonl      {id, text}                         # human continuation = reference target
  wrong_source.jsonl {id, text}                       # a human continuation from a DIFFERENT
                                                       # passage = unfaithful-to-prefix probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="outputs/counterfactual/counterfactual_l512/texts/clean_candidates.jsonl",
    )
    ap.add_argument(
        "--ref-source",
        default="outputs/casestudy/single_fold/reference_score.jsonl",
        help="separate (larger, longer) OWT source for the human reference pool",
    )
    ap.add_argument("--out-dir", default="outputs/casestudy/prefix_continuation")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument(
        "--ref-n",
        type=int,
        default=1024,
        help="size of a DISJOINT human-continuation reference pool (for the null)",
    )
    ap.add_argument("--prefix-len", type=int, default=128)
    ap.add_argument("--cont-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260621)
    args = ap.parse_args()

    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    need = args.prefix_len + args.cont_len

    # prefixes (+ their human continuations) from --source
    rows = []
    for rec in (json.loads(line) for line in open(args.source)):
        ids = tok(rec["text"]).input_ids
        if len(ids) < need:
            continue
        rows.append(
            {
                "id": rec.get("parent_id", "")[:16] or f"p{len(rows)}",
                "prefix_text": tok.decode(ids[: args.prefix_len]),
                "prefix_ids": ids[: args.prefix_len],
                "human_cont": tok.decode(ids[args.prefix_len : need]),
            }
        )
        if len(rows) >= args.n:
            break
    # a larger, DISJOINT human-continuation reference pool from --ref-source (for the null)
    prefix_ids_used = {tuple(r["prefix_ids"]) for r in rows}
    ref_pool = []
    for rec in (json.loads(line) for line in open(args.ref_source)):
        ids = tok(rec["text"]).input_ids
        if len(ids) < need or tuple(ids[: args.prefix_len]) in prefix_ids_used:
            continue
        ref_pool.append(
            {
                "id": rec.get("parent_id", "")[:16] or f"r{len(ref_pool)}",
                "text": tok.decode(ids[args.prefix_len : need]),
            }
        )
        if len(ref_pool) >= args.ref_n:
            break

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "prefixes.jsonl", "w") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {"id": r["id"], "prefix_text": r["prefix_text"], "prefix_ids": r["prefix_ids"]}
                )
                + "\n"
            )
    with open(out / "human.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "text": r["human_cont"]}) + "\n")
    # wrong-source: pair each prefix with another passage's human continuation (seeded shift)
    shift = max(1, len(rows) // 2 + 1)
    with open(out / "wrong_source.jsonl", "w") as f:
        for i, r in enumerate(rows):
            other = rows[(i + shift) % len(rows)]
            f.write(json.dumps({"id": r["id"], "text": other["human_cont"]}) + "\n")

    with open(out / "reference_pool.jsonl", "w") as f:
        for r in ref_pool:
            f.write(json.dumps(r) + "\n")

    print(
        f"[continuation corpus] {len(rows)} prefixes "
        f"(L_pre={args.prefix_len}, L_cont={args.cont_len}); "
        f"reference pool {len(ref_pool)}; "
        f"wrote prefixes/human/wrong_source/reference_pool.jsonl to {out}"
    )
    if rows:
        print("  example prefix tail:", repr(rows[0]["prefix_text"][-90:]))
        print("  example human cont :", repr(rows[0]["human_cont"][:90]))


if __name__ == "__main__":
    main()
