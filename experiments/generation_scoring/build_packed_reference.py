"""Build a PACKED OWT reference matching the training-corpus format of
ELF/SEDD/MDLM/LangFlow (documents concatenated, fixed-token windows).

A short single-document reference (mean ~241 GPT-2 tokens) is a format mismatch
against packed-1024-trained generators whose samples legitimately contain
document switches. Windows of 512 tokens = exactly what the CHORD encoder reads
of any generation. Emits reference_packed512.jsonl (reference pool) +
human_packed_held.jsonl (disjoint windows, human floor check)."""

import argparse
import json
import random
from pathlib import Path

from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = [
    ROOT / "outputs/casestudy/single_fold/reference_score.jsonl",
    ROOT / "outputs/casestudy/single_fold/real_owt_held.jsonl",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sources",
        nargs="*",
        default=[str(p) for p in DEFAULT_SOURCES],
        help="JSONL files with a `text` field: human OWT documents to pack",
    )
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/casestudy/single_fold"))
    ap.add_argument("--window", type=int, default=512, help="tokens per packed window")
    ap.add_argument("--held", type=int, default=148, help="held-out human windows")
    ap.add_argument("--seed", type=int, default=20260707)
    args = ap.parse_args()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    docs = [json.loads(line)["text"] for p in args.sources for line in open(p)]
    random.Random(args.seed).shuffle(docs)
    stream = []
    for d in docs:
        stream.extend(tok(d)["input_ids"] + [tok.eos_token_id])
    W = args.window
    wins = [
        tok.decode(stream[i : i + W], skip_special_tokens=True)
        for i in range(0, len(stream) - W, W)
    ]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_held = args.held
    with open(out / f"reference_packed{W}.jsonl", "w") as f:
        for i, t in enumerate(wins[n_held:]):
            f.write(json.dumps({"text": t, "sample_id": f"packref:{i:05d}"}) + "\n")
    with open(out / "human_packed_held.jsonl", "w") as f:
        for i, t in enumerate(wins[:n_held]):
            f.write(
                json.dumps(
                    {
                        "sample_id": f"packheld:{i:05d}",
                        "text": t,
                        "run_id": "human-packed",
                        "model": "Human-packed",
                        "seed": 0,
                    }
                )
                + "\n"
            )
    print(
        f"[packed-ref] {len(docs)} docs -> {len(wins)} windows of {W} tok "
        f"({len(wins) - n_held} reference + {n_held} held)"
    )


if __name__ == "__main__":
    main()
