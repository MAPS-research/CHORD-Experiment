"""Stage 2 (GPU): featurize every text set with every encoder -> cached .npy.

One forward pass per (encoder, text-set); resumable (skips existing .npy). The
*identical* feature matrices are later consumed by both our RBF-MMD ruler and the
official ``mauve.compute_mauve`` quantized-KL frontier, which is what makes the
ELECTRA+MMD vs MAUVE-ELECTRA comparison apples-to-apples.

    python -m experiments.counterfactual_eval.featurize \
        --config experiments/counterfactual_eval/configs/encoders_chord_27b.yaml \
        --corpus-dir outputs/counterfactual/meta_eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from chord.feature_encoding import embed_batched


def _load_texts(path: Path) -> List[str]:
    return [json.loads(line)["text"] for line in open(path)]


def _set_path(texts_dir: Path, name: str) -> Path:
    if name in ("reference", "clean_candidates"):
        return texts_dir / f"{name}.jsonl"
    return texts_dir / "conditions" / f"{name}.jsonl"


def text_set_names(manifest: Dict) -> List[str]:
    names = ["reference", "clean_candidates"]
    for c in manifest["conditions"]:
        if c.get("experiment") != "control":
            names.append(c["name"])
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="encoder protocols")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-encoder", default=None, help="featurize only this encoder name (for env-split jobs)"
    )
    parser.add_argument(
        "--only-sets",
        default="",
        help="comma-separated text-set names: featurize ONLY these "
        "(additive; existing .npy caches untouched). Used to "
        "add new conditions to an already-featurized evaluation set.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    corpus_dir = Path(args.corpus_dir)
    texts_dir = corpus_dir / "texts"
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    sets = text_set_names(manifest)
    only_sets = [s.strip() for s in args.only_sets.split(",") if s.strip()]
    if only_sets:
        unknown = [s for s in only_sets if s not in sets]
        if unknown:
            raise SystemExit(f"--only-sets: unknown text sets {unknown}")
        sets = only_sets
        print(f"[feat] only-sets: featurizing {sets}", flush=True)
    batch_size = int(cfg.get("batch_size", 128))

    for enc in cfg["encoders"]:
        name = enc["name"]
        if args.only_encoder and name != args.only_encoder:
            continue
        protocol = dict(enc["protocol"])
        out_enc = corpus_dir / "feats" / name
        out_enc.mkdir(parents=True, exist_ok=True)
        for s in sets:
            outp = out_enc / f"{s}.npy"
            if outp.exists() and not args.overwrite:
                print(f"[feat] {name}/{s} exists, skip", flush=True)
                continue
            texts = _load_texts(_set_path(texts_dir, s))
            t0 = time.perf_counter()
            emb = embed_batched(texts, protocol, batch_size=batch_size)
            np.save(outp, emb.astype("float32"))
            print(
                f"[feat] {name}/{s}: n={len(texts)} dim={emb.shape[1]} "
                f"{time.perf_counter() - t0:.1f}s -> {outp}",
                flush=True,
            )

    print("[feat] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
