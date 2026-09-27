"""Layer-depth ablation, step 1 of 2 (GPU): CHORD's readout at every second layer.

Appendix "Effect of Extraction Layer" sweeps the read depth on the Table 1
evaluation set. The encoder protocol (model, revision, max_length, prompt
template, dtype) and the batch size come from the Table 1 encoder config, so only
the read depth differs from Table 1. One forward pass per batch with
``output_hidden_states=True`` yields every layer: the coherence-prompt final-token
state is saved at every second block (hidden-state index 2, 4, ..., always
including the final block) and at the post-norm ``last_hidden_state`` (tag
``last``). The text sets are the human reference, the benign anchor of the
manifest and the nine Table 1 perturbation types at the highest severity.

Outputs: <out-dir>/feats/L<k>/<set>.npy (k = block index) and <out-dir>/feats/Llast/<set>.npy

    python -m experiments.layer_depth.featurize \\
        --encoder-config experiments/counterfactual_eval/configs/encoders_chord_27b.yaml \\
        --corpus-dir outputs/counterfactual/meta_eval \\
        --out-dir outputs/experiments/layer_depth/qwen35_27b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from chord.embeddings import HuggingFaceEncoder, template_input_ids

# The nine Table 1 perturbation types at the highest severity (Table 1's columns).
TABLE1_TOP_SEVERITY = [
    "causal_reverse__dose3",
    "contradiction__dose3",
    "broken_transition__dose3",
    "topic_drift__dose3",
    "sentence_permutation__r100",
    "local_shuffle__r100",
    "repetition__r100",
    "dlm_splice__r050",
    "corpus_mix__r075_d8",
]


def benign_anchor(corpus_dir: Path) -> str:
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    return next(c["name"] for c in manifest["conditions"] if c.get("kind") == "benign")


def text_sets(corpus_dir: Path) -> Dict[str, Path]:
    texts = corpus_dir / "texts"
    sets = {"reference": texts / "reference.jsonl"}
    for name in [benign_anchor(corpus_dir)] + TABLE1_TOP_SEVERITY:
        sets[name] = texts / "conditions" / f"{name}.jsonl"
    return sets


def load_texts(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()]


def table1_encoder(config: Path) -> tuple[Dict, int]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    (encoder,) = cfg["encoders"]
    return encoder["protocol"], int(encoder.get("batch_size", cfg.get("batch_size", 8)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder-config", required=True, help="Table 1 encoder config (one encoder)")
    ap.add_argument("--corpus-dir", default="outputs/counterfactual/meta_eval")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import torch

    protocol, batch_size = table1_encoder(Path(args.encoder_config))
    enc = HuggingFaceEncoder(protocol)
    tok, model, template = enc.tokenizer, enc.model, enc.prompteol_template
    max_length = int(protocol["max_length"])

    def forward(texts: List[str]):
        ids = [template_input_ids(tok, template, t, max_length) for t in texts]
        prev = tok.padding_side
        tok.padding_side = "left"  # the genuine final token sits at position -1
        try:
            batch = tok.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        finally:
            tok.padding_side = prev
        batch = {k: v.to(enc.device) for k, v in batch.items()}
        with torch.inference_mode():
            return model(**batch, output_hidden_states=True)

    # depth from a probe forward: len(hidden_states) == blocks + 1 for any backbone
    n_states = len(forward(["probe"]).hidden_states)
    grid = sorted(set(range(2, n_states, 2)) | {n_states - 1})
    tags: List = grid + ["last"]
    print(f"[layers] {protocol['model']}: {n_states - 1} blocks, grid {grid} + last", flush=True)

    out_root = Path(args.out_dir) / "feats"

    def layer_dir(tag) -> Path:
        d = out_root / f"L{tag}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    sets = text_sets(Path(args.corpus_dir))
    for name, path in sets.items():
        if not args.overwrite and all((layer_dir(t) / f"{name}.npy").exists() for t in tags):
            print(f"[layers] {name}: all layers present, skipped", flush=True)
            continue
        texts = load_texts(path)
        acc = {t: [] for t in tags}
        t0 = time.perf_counter()
        for start in range(0, len(texts), batch_size):
            out = forward(texts[start : start + batch_size])
            for layer in grid:
                acc[layer].append(out.hidden_states[layer][:, -1, :].float().cpu().numpy())
            acc["last"].append(out.last_hidden_state[:, -1, :].float().cpu().numpy())
        for t in tags:
            np.save(layer_dir(t) / f"{name}.npy", np.concatenate(acc[t]).astype("float32"))
        print(f"[layers] {name}: n={len(texts)} {time.perf_counter() - t0:.1f}s", flush=True)

    # sanity: on the first batch (same texts, same padding) the Table 1 read depth
    # must match CHORD's own encoder path up to rounding
    read = int(protocol["layer"])
    depth = n_states + read if read < 0 else read
    sample = load_texts(sets["reference"])[:batch_size]
    table1 = np.load(layer_dir(depth) / "reference.npy")[:batch_size]
    rel = float(np.abs(enc.encode(sample) - table1).max() / np.abs(table1).max())
    print(f"[layers] L{depth} vs encoder, first batch: max|diff| / max|x| = {rel:.1e} (expect ~0)")


if __name__ == "__main__":
    main()
