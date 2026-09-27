#!/usr/bin/env python3
"""Download the CHORD data bundle from the Hugging Face Hub into this checkout.

The bundle's tree mirrors the repository (`data/...`, `outputs/...`), so the
files land exactly where the configs expect them. By default only the text
corpora are fetched; `--features` adds the cached encoder features (teacher
PCA targets, teacher fold features), which let the teacher columns be
reproduced without a 27B GPU. `--checkpoint qwen3.5-2b` (or `qwen3.5-0.8b`) also
fetches a trained student into `outputs/distill/student_<s>/final`. The students'
training data is in the library's dataset (`mikezhu/chord-distill-data`).

    python scripts/download_data.py [--repo mikezhu/chord-experiments-data] [--features] \
        [--checkpoint qwen3.5-2b]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
# The dataset card (README.md at the bundle root) is deliberately not fetched:
# the bundle root is the checkout root, so it would overwrite this repository's README.
MANIFEST = "MANIFEST-experiments.tsv"
TEXT_PATTERNS = ["outputs/**/*.jsonl", "outputs/**/*.json", "outputs/**/*.csv", MANIFEST]
FEATURE_PATTERNS = ["outputs/**"]


def verify(root: Path, manifest: Path, sample: int) -> None:
    """Check sizes of every downloaded file and the sha256 of a sample of them."""
    rows = list(csv.DictReader(open(manifest), delimiter="\t"))
    present = [r for r in rows if (root / r["path"]).exists()]
    bad = [r["path"] for r in present if (root / r["path"]).stat().st_size != int(r["bytes"])]
    for r in present[:: max(1, len(present) // sample)]:
        digest = hashlib.sha256((root / r["path"]).read_bytes()).hexdigest()
        if digest != r["sha256"]:
            bad.append(r["path"])
    print(f"{len(present)}/{len(rows)} bundle files present; {len(bad)} failed verification")
    if bad:
        raise SystemExit("\n".join(bad))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="mikezhu/chord-experiments-data", help="dataset repo id")
    ap.add_argument("--features", action="store_true", help="also fetch cached features (~2.4 GB)")
    ap.add_argument("--checkpoint", choices=["qwen3.5-2b", "qwen3.5-0.8b"], action="append",
                    default=[], help="also fetch this trained student")
    ap.add_argument("--dest", default=str(ROOT), help="checkout root (default: this repository)")
    ap.add_argument("--verify-sample", type=int, default=20, help="files to sha256-check")
    args = ap.parse_args()

    dest = Path(args.dest)
    patterns = TEXT_PATTERNS + (FEATURE_PATTERNS if args.features else [])
    snapshot_download(args.repo, repo_type="dataset", allow_patterns=patterns, local_dir=dest)
    verify(dest, dest / MANIFEST, args.verify_sample)

    for student in args.checkpoint:
        target = dest / f"outputs/distill/student_{student}/final"
        snapshot_download(f"{args.repo.split('/')[0]}/chord-{student}-student", local_dir=target)
        print(f"student {student} -> {target}")


if __name__ == "__main__":
    main()
