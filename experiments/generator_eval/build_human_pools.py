#!/usr/bin/env python3
"""Human pools for the repeated Table-2 protocol (10 disjoint folds of 500).

Packs fresh OpenWebText documents into 512-token windows, exactly as
experiments.generation_scoring.build_packed_reference does, after dropping every document
that shares an 8-word shingle with any text the student was trained,
validated or previously evaluated on. Reference and held-out windows come from
disjoint documents.

  python -m experiments.generator_eval.build_human_pools \
      --out outputs/casestudy/unconditional_generation/human
"""

import argparse
import json
import random
import re
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "Skylion007/openwebtext"
SHINGLE = 8
BAN_GLOBS = [
    "data/distill/training_data/**/*.jsonl",  # every text of the distillation training data
    "data/distill/eval_ban/*.jsonl",
    "data/interim/*.jsonl",
    "outputs/casestudy/single_fold/*.jsonl",
]
WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, stride: int = 1) -> np.ndarray:
    words = WORD.findall(text.lower())
    if len(words) < SHINGLE:
        return np.empty(0, dtype=np.uint64)
    joined = [" ".join(words[i : i + SHINGLE]) for i in range(0, len(words) - SHINGLE + 1, stride)]
    return np.fromiter(
        ((zlib.crc32(s.encode()) << 32) | zlib.adler32(s.encode()) for s in joined),
        dtype=np.uint64,
        count=len(joined),
    )


def ban_set() -> np.ndarray:
    parts, n_texts = [], 0
    for pattern in BAN_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            for line in open(path, encoding="utf-8"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("text", "clean_text", "perturbed_text", "parent_text"):
                    if isinstance(row.get(key), str):
                        parts.append(shingles(row[key]))
                        n_texts += 1
    ban = np.unique(np.concatenate(parts))
    print(f"[ban] {n_texts} texts -> {len(ban):,} distinct {SHINGLE}-word shingles", flush=True)
    return ban


def pack(docs, tok, window):
    stream = []
    for d in docs:
        stream.extend(tok(d)["input_ids"] + [tok.eos_token_id])
    return [
        tok.decode(stream[i : i + window], skip_special_tokens=True)
        for i in range(0, len(stream) - window, window)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "outputs/casestudy/unconditional_generation/human"))
    ap.add_argument(
        "--reference", type=int, default=5150, help="reference windows (150 dev + 10 x 500)"
    )
    ap.add_argument("--held", type=int, default=5000, help="held-out human windows (10 x 500)")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--shard", type=int, default=41, help="first OWT parquet shard to read")
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ban = ban_set()
    files = sorted(
        f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet")
    )
    need = {"reference": args.reference, "held": args.held}
    docs = {"reference": [], "held": []}
    tokens = {"reference": 0, "held": 0}
    seen = dropped = 0
    rng = random.Random(args.seed)
    for shard in range(args.shard, len(files)):
        local = hf_hub_download(REPO_ID, files[shard], repo_type="dataset")
        texts = [t for t in pq.read_table(local, columns=["text"])["text"].to_pylist() if t]
        rng.shuffle(texts)
        for text in texts:
            seen += 1
            probe = shingles(text, stride=4)
            at = np.clip(np.searchsorted(ban, probe), 0, len(ban) - 1)
            if (ban[at] == probe).any():
                dropped += 1
                continue
            side = (
                "reference"
                if tokens["reference"] * need["held"] <= tokens["held"] * need["reference"]
                else "held"
            )
            docs[side].append(text)
            tokens[side] += len(tok(text)["input_ids"]) + 1
            if all(tokens[k] > (need[k] + 2) * args.window for k in need):
                break
        print(f"[owt] shard {shard}: seen {seen} dropped {dropped} tokens {tokens}", flush=True)
        if all(tokens[k] > (need[k] + 2) * args.window for k in need):
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = pack(docs["reference"], tok, args.window)[: args.reference]
    held = pack(docs["held"], tok, args.window)[: args.held]
    assert len(ref) == args.reference and len(held) == args.held, (len(ref), len(held))
    rng.shuffle(ref)
    rng.shuffle(held)
    with open(out / "reference_pool.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(ref):
            f.write(json.dumps({"sample_id": f"r10ref:{i:05d}", "text": t}) + "\n")
    with open(out / "human_held.jsonl", "w", encoding="utf-8") as f:
        for i, t in enumerate(held):
            f.write(
                json.dumps(
                    {
                        "sample_id": f"r10held:{i:05d}",
                        "text": t,
                        "run_id": "human-packed",
                        "model": "Human-packed",
                        "seed": 0,
                    }
                )
                + "\n"
            )
    print(
        f"[done] docs ref {len(docs['reference'])} held {len(docs['held'])}; "
        f"dropped {dropped}/{seen} "
        f"for overlap -> {len(ref)} reference + {len(held)} held windows"
    )


if __name__ == "__main__":
    main()
