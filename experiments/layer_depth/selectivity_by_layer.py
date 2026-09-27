"""Layer-depth ablation, step 2 of 2 (CPU): Table 1 selectivity at every layer.

Scores each layer's features from ``featurize.py`` with Table 1's endpoint
(``experiments.counterfactual_eval.selectivity.score_features``: dev-frozen median
bandwidth, exchangeable null, parent-level hierarchical bootstrap, S = z(harmful) -
z(benign) with its 95% interval). Every random stream (split, null, bootstrap) is
seeded by the locked Table 1 encoder name (``--rng-encoder``) rather than by the
layer, so the layer is the only variable and the Table 1 read depth reproduces the
Table 1 row up to bf16 re-featurization noise. The appendix figure plots, per
layer, the mean S over the nine types normalized by the backbone's peak and the
number of types whose S interval lies above zero.

Inputs:  <sweep-dir>/feats/L<k>/<set>.npy (featurize.py), <corpus-dir>/texts/conditions
         (parent ids for the bootstrap) and the Table 1 selectivity config
Outputs: <sweep-dir>/layer_selectivity.csv  columns: layer, family, z_harmful,
         z_benign, selectivity, ci_low, ci_high, verdict, sigma
         <sweep-dir>/layer_calibration.csv  one calibration row per layer

    python -m experiments.layer_depth.selectivity_by_layer \\
        --sweep-dir outputs/experiments/layer_depth/qwen35_27b \\
        --rng-encoder qwen35-27b-prompteol-coherence-l62
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import yaml

from ..counterfactual_eval.selectivity import ScoringParams, score_features
from .featurize import TABLE1_TOP_SEVERITY, benign_anchor

FIELDS = [
    "layer",
    "family",
    "z_harmful",
    "z_benign",
    "selectivity",
    "ci_low",
    "ci_high",
    "verdict",
    "sigma",
]


def discover_layers(sweep_dir: Path) -> List[str]:
    tags = [d.name[1:] for d in (sweep_dir / "feats").iterdir() if (d / "reference.npy").exists()]
    order = [str(n) for n in sorted(int(t) for t in tags if t.isdigit())]
    return order + (["last"] if "last" in tags else [])


def write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--corpus-dir", default="outputs/counterfactual/meta_eval")
    ap.add_argument("--config", default="experiments/counterfactual_eval/configs/selectivity.yaml")
    ap.add_argument("--rng-encoder", required=True, help="Table 1 encoder name seeding every draw")
    args = ap.parse_args()

    sweep_dir, corpus_dir = Path(args.sweep_dir), Path(args.corpus_dir)
    params = ScoringParams.from_config(yaml.safe_load(Path(args.config).read_text()))
    manifest = json.loads((corpus_dir / "conditions_manifest.json").read_text())
    corpus = manifest.get("corpus", corpus_dir.name)
    kind_of = {c["name"]: c.get("kind", "") for c in manifest["conditions"]}
    group_of = {c["name"]: c.get("group", "") for c in manifest["conditions"]}
    benign = benign_anchor(corpus_dir)

    rows: List[Dict] = []
    calib_rows: List[Dict] = []
    for layer in discover_layers(sweep_dir):
        calib, scored = score_features(
            sweep_dir / "feats" / f"L{layer}",
            corpus_dir,
            label=f"L{layer}",
            rng_name=args.rng_encoder,
            corpus=corpus,
            benign=benign,
            targets=TABLE1_TOP_SEVERITY,
            params=params,
            group_of=group_of,
            kind_of=kind_of,
        )
        calib_rows.append(dict(calib, layer=layer))
        for r in scored:
            if r["family"] == benign:
                continue
            verdict = "SEL" if r["selective"] else ("ANTI" if r["anti_selective"] else "n.s.")
            rows.append(
                {
                    "layer": layer,
                    "family": r["family"],
                    "z_harmful": r["z_harmful"],
                    "z_benign": r["z_benign"],
                    "selectivity": r["selectivity"],
                    "ci_low": r["sel_ci_low"],
                    "ci_high": r["sel_ci_high"],
                    "verdict": verdict,
                    "sigma": calib["sigma"],
                }
            )
        n_sel = sum(r["selective"] for r in scored)
        print(f"[layer L{layer}] sigma={calib['sigma']:.4f} selective={n_sel}/9", flush=True)
        write_csv(sweep_dir / "layer_selectivity.csv", rows)
        write_csv(sweep_dir / "layer_calibration.csv", calib_rows)
    print(f"[layer] wrote {sweep_dir / 'layer_selectivity.csv'}", flush=True)


if __name__ == "__main__":
    main()
