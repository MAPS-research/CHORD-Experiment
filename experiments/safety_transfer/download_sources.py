"""Download the PKU-SafeRLHF files the safety-transfer corpus is built from.

Fetches the train-split release files of two generators from the Hugging Face
dataset ``PKU-Alignment/PKU-SafeRLHF`` (CC-BY-NC-4.0) into ``data/raw/pku_saferlhf``:
Alpaca-7B (reference, clean candidates and matched donor pairs) and Alpaca3-8B
(the generator-shift control).

    python -m experiments.safety_transfer.download_sources
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ID = "PKU-Alignment/PKU-SafeRLHF"
GENERATORS = ("Alpaca-7B", "Alpaca3-8B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/raw/pku_saferlhf")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for generator in GENERATORS:
        out = out_dir / f"{generator}_train.jsonl"
        if out.exists():
            print(f"{generator}: cached ({out.stat().st_size / 1e6:.1f} MB)")
            continue
        path = hf_hub_download(REPO_ID, f"data/{generator}/train.jsonl", repo_type="dataset")
        shutil.copy(path, out)
        print(f"{generator}: {out.stat().st_size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
