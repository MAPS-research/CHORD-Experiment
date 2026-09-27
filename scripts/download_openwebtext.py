"""
Download OpenWebText from HuggingFace (Skylion007/openwebtext) via parquet files
and chunk documents into ~200-word passages.

Output: data/raw/openwebtext_passages.jsonl
Usage: python scripts/download_openwebtext.py [--target 250000]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "Skylion007/openwebtext"
TARGET_MIN_WORDS = 140
TARGET_MAX_WORDS = 320
PASSAGE_TARGET_WORDS = 200


def split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_document(text: str) -> list[str]:
    sents = split_sentences(text)
    passages: list[str] = []
    bucket: list[str] = []
    bucket_words = 0
    for sent in sents:
        w = len(sent.split())
        if bucket_words + w > PASSAGE_TARGET_WORDS and bucket:
            passages.append(" ".join(bucket))
            bucket = [sent]
            bucket_words = w
        else:
            bucket.append(sent)
            bucket_words += w
    if bucket:
        passages.append(" ".join(bucket))
    return passages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=int, default=250_000, help="Stop after collecting this many passages"
    )
    parser.add_argument("--out", default="data/raw/openwebtext_passages.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in {REPO_ID} ...")
    all_files = sorted(
        f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet")
    )
    print(f"Found {len(all_files)} parquet files")

    count = 0
    with out.open("w", encoding="utf-8") as fh:
        for file_idx, repo_path in enumerate(all_files):
            if count >= args.target:
                break
            print(f"[{file_idx + 1}/{len(all_files)}] {repo_path}  passages={count}", flush=True)
            local = hf_hub_download(REPO_ID, repo_path, repo_type="dataset")
            table = pq.read_table(local, columns=["text"])
            for row_idx, text in enumerate(table["text"].to_pylist()):
                if not text:
                    continue
                doc_id = f"owt-{file_idx}-{row_idx}"
                for j, passage in enumerate(chunk_document(text)):
                    nwords = len(passage.split())
                    if TARGET_MIN_WORDS <= nwords <= TARGET_MAX_WORDS:
                        fh.write(
                            json.dumps(
                                {
                                    "source_document_id": doc_id,
                                    "passage_index": j,
                                    "text": passage,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
                if count >= args.target:
                    break

    print(f"Done. Wrote {count} passages to {out}")


if __name__ == "__main__":
    main()
