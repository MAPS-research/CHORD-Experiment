"""Featurize the safety-transfer corpus once per response, then compose each set.

The 19 conditions of 1,000 responses are drawn from about 8,000 distinct
responses, and every arm shares the same unreplaced clean rows. ``pool`` runs one
forward pass per unique response (``texts/pool.jsonl``); ``compose`` builds each
set's feature matrix by indexing (``index/<set>.json``). Besides being about three
times cheaper, this makes the dose curve move only because the corpus changed:
a response always carries the same bf16 feature vector in every condition.

Outputs ``<corpus>/feats/<encoder>/<set>.npy``, the layout read by
``paired_contrast``, ``mauve_scores`` and
``experiments.counterfactual_eval.baseline_selectivity``.

    python -m experiments.safety_transfer.featurize pool \
        --config experiments/safety_transfer/configs/encoders_chord_prompts.yaml \
        --corpus-dir outputs/experiments/safety_transfer
    python -m experiments.safety_transfer.featurize compose \
        --corpus-dir outputs/experiments/safety_transfer
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml

from chord.feature_encoding import embed_batched


def pool_texts(corpus_dir: Path) -> List[str]:
    with open(corpus_dir / "texts" / "pool.jsonl", encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()]


def set_names(corpus_dir: Path) -> List[str]:
    return sorted(p.stem for p in (corpus_dir / "index").glob("*.json"))


def cmd_pool(args) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    corpus_dir = Path(args.corpus_dir)
    texts = pool_texts(corpus_dir)
    batch_size = int(cfg.get("batch_size", 8))
    for enc in cfg["encoders"]:
        name = enc["name"]
        if args.only_encoder and name != args.only_encoder:
            continue
        out_enc = corpus_dir / "feats" / name
        out_enc.mkdir(parents=True, exist_ok=True)
        outp = out_enc / "_pool.npy"
        if outp.exists() and not args.overwrite:
            print(f"[pool] {name} exists, skip", flush=True)
            continue
        t0 = time.perf_counter()
        emb = embed_batched(texts, dict(enc["protocol"]), batch_size=batch_size)
        np.save(outp, emb.astype("float32"))
        dt = time.perf_counter() - t0
        print(
            f"[pool] {name}: n={len(texts)} dim={emb.shape[1]} {dt:.1f}s "
            f"({len(texts) / dt:.1f} docs/s) -> {outp}",
            flush=True,
        )


def cmd_compose(args) -> None:
    corpus_dir = Path(args.corpus_dir)
    names = set_names(corpus_dir)
    n_pool = len(pool_texts(corpus_dir))
    if args.only_encoder:
        encoders = [args.only_encoder]
    else:
        encoders = sorted(p.name for p in (corpus_dir / "feats").iterdir() if p.is_dir())
    for enc in encoders:
        pool_path = corpus_dir / "feats" / enc / "_pool.npy"
        if not pool_path.exists():
            print(f"[compose] {enc}: no _pool.npy, skip", flush=True)
            continue
        pool = np.load(pool_path)
        if len(pool) != n_pool:
            raise SystemExit(
                f"[compose] {enc}: pool has {len(pool)} rows, texts/pool.jsonl has {n_pool}"
            )
        for name in names:
            idx = np.asarray(
                json.loads((corpus_dir / "index" / f"{name}.json").read_text()), dtype=np.int64
            )
            np.save(corpus_dir / "feats" / enc / f"{name}.npy", pool[idx])
        print(
            f"[compose] {enc}: {len(names)} sets from {len(pool)}x{pool.shape[1]} pool",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool", help="one forward pass over texts/pool.jsonl")
    p.add_argument("--config", required=True)
    p.add_argument("--corpus-dir", required=True)
    p.add_argument("--only-encoder", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_pool)
    c = sub.add_parser("compose", help="materialize per-set .npy by indexing the pool")
    c.add_argument("--corpus-dir", required=True)
    c.add_argument("--only-encoder", default=None)
    c.set_defaults(func=cmd_compose)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
