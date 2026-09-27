"""Selectivity per failure position (the numbers behind the position-robustness figure).

Paper result: appendix "Effect of Coherence-Failure Position on Detection"
(Figure fig_position_robustness and the suffix-to-prefix ratios quoted in the
text: raw last token 16.4x, MetaEOL multi-concat 9.7x, CHORD coherence prompt
1.7x, generic PromptEOL 2.3x; mean pooling near zero at every position).

For each readout of the fixed Qwen3.5-9B backbone, selectivity at a position is
    Delta z = z(topic drift at that position) - z(benign paraphrase),
with z the null-standardized RBF-MMD against the human reference, using the same
estimator as ``experiments.counterfactual_eval.selectivity`` (exchangeable null, dev-split
median bandwidth, parent-level hierarchical bootstrap; 200 draws, n = 1000 per
side, 1,500 dev texts).

Inputs   <corpus>/conditions_manifest.json, <corpus>/texts/, <corpus>/feats/<encoder>/
         (build_corpus.py, then experiments.counterfactual_eval.featurize with
         experiments/position_robustness/configs/encoders.yaml)
Outputs  <out>/position_robustness.csv

    python -m experiments.position_robustness.analyze \\
        --corpus-dir outputs/experiments/position_robustness
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from chord.metrics.distribution import median_bandwidth
from chord.utils.hashing import seeded_rng

from ..counterfactual_eval.nulls import exchangeable_null
from ..counterfactual_eval.selectivity import (
    _load,
    _parent_groups,
    _parent_ids,
    _split,
    hierarchical_attack,
)

DRAWS, N_EVAL, DEV_SIZE, BLOCK = 200, 1000, 1500, 1024
POSITIONS = ("prefix", "middle", "suffix")

# (feature directory, label)
READOUTS = [
    ("q35-9b-prompteol-coherence-l512", "CHORD (coherence prompt, Qwen3.5-9B)"),
    ("q35-9b-last-l512", "raw last token (Qwen3.5-9B)"),
    ("q35-9b-multi-concat-l512", "MetaEOL multi-concat (Qwen3.5-9B)"),
    ("q35-9b-prompteol-l512", "PromptEOL (Qwen3.5-9B)"),
    ("q35-9b-mean-l512", "mean pooling (Qwen3.5-9B)"),
]


def selectivity_by_position(corpus_dir: Path, encoder: str, corpus: str) -> Dict[str, float]:
    feats = corpus_dir / "feats" / encoder
    reference = _load(feats / "reference.npy")
    dev, cal, test = _split(reference, encoder, corpus, dev_size=DEV_SIZE)
    n = min(N_EVAL, len(cal) // 2, len(test) // 2)
    sigma = float(median_bandwidth(dev, max_samples=2000, seed=0))
    null = exchangeable_null(
        cal,
        n,
        n,
        draws=DRAWS,
        sigma=sigma,
        block_size=BLOCK,
        rng=seeded_rng(encoder, corpus, "cal", n),
    )
    null_mean, null_std = float(null.mean()), float(null.std())

    def z(name: str) -> float:
        emb = _load(feats / f"{name}.npy")
        groups = _parent_groups(_parent_ids(corpus_dir, name))
        draws = hierarchical_attack(
            cal,
            emb,
            groups,
            n,
            n,
            draws=DRAWS,
            sigma=sigma,
            block_size=BLOCK,
            rng=seeded_rng(encoder, corpus, "h", name),
        )
        return (float(draws.mean()) - null_mean) / null_std

    z_benign = z("benign_paraphrase")
    return {p: z(f"topicdrift_pos_{p}") - z_benign for p in POSITIONS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--corpus-dir", default="outputs/experiments/position_robustness")
    ap.add_argument("--out-dir", default=None, help="default: <corpus-dir>")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir) if args.out_dir else corpus_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    corpus = manifest.get("corpus", "position_probe")

    rows: List[Dict] = []
    for encoder, label in READOUTS:
        feats = corpus_dir / "feats" / encoder
        if not (feats / "reference.npy").exists():
            print(f"[position] no features for {encoder}; skipped")
            continue
        s = selectivity_by_position(corpus_dir, encoder, corpus)
        ratio = s["suffix"] / s["prefix"] if s["prefix"] > 0 else float("nan")
        rows.append(
            {
                "encoder": encoder,
                "label": label,
                "S_prefix": round(s["prefix"], 1),
                "S_middle": round(s["middle"], 1),
                "S_suffix": round(s["suffix"], 1),
                "suffix_prefix_ratio": round(ratio, 2),
            }
        )
        print(
            f"[position] {label:40s} prefix={s['prefix']:7.1f} middle={s['middle']:7.1f} "
            f"suffix={s['suffix']:7.1f}  suffix/prefix={ratio:.2f}x"
        )

    with open(out_dir / "position_robustness.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[position] wrote {out_dir / 'position_robustness.csv'}")


if __name__ == "__main__":
    main()
