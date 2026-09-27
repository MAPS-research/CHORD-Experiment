"""gen-PPL token-level baseline column (E1/E7 comparison).

GPU half: per-text mean token NLL (= log perplexity) under gpt2-large -> npy.
CPU half: the *same* bootstrap-detection framework as our ruler — clean null from
the clean candidates' log-ppl, condition detected if its 2.5% bootstrap CI clears
the clean q95. This makes gen-PPL a like-for-like detection column: it should peak
on intra-sentence (fluency) corruption and stay blunt on pure discourse reordering.

    python -m experiments.counterfactual_eval.score_genppl \
        --config experiments/counterfactual_eval/configs/genppl.yaml \
        --score-config experiments/counterfactual_eval/configs/selectivity.yaml \
        --corpus-dir outputs/counterfactual/meta_eval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from chord.utils.hashing import seeded_rng

from .featurize import _load_texts, _set_path, text_set_names


def _mean_nll(texts: List[str], cfg: Dict) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]
    rev = cfg.get("revision")
    max_length = int(cfg.get("max_length", 256))
    bs = int(cfg.get("batch_size", 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, revision=rev)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg.get("model_dtype", "float32"))
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=rev, torch_dtype=dtype)
    model.to(device).eval()
    out = np.empty(len(texts), dtype=np.float64)
    for start in range(0, len(texts), bs):
        batch = texts[start : start + bs]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        shift_mask = mask[:, 1:].bool()
        losses = torch.nn.functional.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels, reduction="none"
        )
        for i in range(len(batch)):
            m = shift_mask[i]
            out[start + i] = float(losses[i][m].mean().item()) if bool(m.any()) else float("nan")
    return out


def _boot(vals: np.ndarray, n: int, draws: int, rng) -> np.ndarray:
    # Proper bootstrap of the mean: ALWAYS with replacement. Sampling n without
    # replacement from a pool of exactly n would pick the whole pool every draw
    # (zero variance -> degenerate z). With replacement, null std = std/sqrt(n).
    vals = vals[np.isfinite(vals)]
    out = np.empty(draws)
    for d in range(draws):
        pick = rng.choice(len(vals), size=n, replace=True)
        out[d] = vals[pick].mean()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="gen-PPL model config")
    parser.add_argument("--score-config", required=True, help="reuse n_eval/draws")
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument(
        "--out-suffix",
        default="",
        help="append to the detection CSV name, e.g. "
        "'_meaning_probe' -> genppl_detection_meaning_probe.csv "
        "(non-destructive; per-set NLL caches are shared)",
    )
    args = parser.parse_args()

    gcfg = yaml.safe_load(Path(args.config).read_text())
    scfg = yaml.safe_load(Path(args.score_config).read_text())
    cdir = Path(args.corpus_dir)
    manifest = json.loads((cdir / "conditions_manifest.json").read_text())
    cond_meta = {c["name"]: c for c in manifest["conditions"]}
    sets = text_set_names(manifest)
    nll_dir = cdir / "genppl"
    nll_dir.mkdir(parents=True, exist_ok=True)

    # ---- GPU half: per-text mean NLL ----------------------------------------
    if not args.detect_only:
        for s in sets:
            outp = nll_dir / f"{s}.npy"
            if outp.exists():
                print(f"[genppl] {s} exists, skip", flush=True)
                continue
            texts = _load_texts(_set_path(cdir / "texts", s))
            nll = _mean_nll(texts, gcfg)
            np.save(outp, nll)
            print(f"[genppl] {s}: n={len(texts)} mean_logppl={np.nanmean(nll):.3f}", flush=True)

    # ---- CPU half: detection in the ruler framework -------------------------
    if not (nll_dir / "clean_candidates.npy").exists():
        print("[genppl] no NLL cache yet (run the GPU half first); skipping detection.", flush=True)
        return
    n_eval = int(scfg.get("n_eval", 1000))
    draws = int(scfg.get("draws", 200))
    clean = np.load(nll_dir / "clean_candidates.npy")
    n = min(n_eval, int(np.isfinite(clean).sum()))
    null = _boot(clean, n, draws, seeded_rng("genppl", "clean", n))
    nmean, nstd, q95 = float(null.mean()), float(null.std()), float(np.quantile(null, 0.95))
    rows: List[Dict] = []
    for s in ["clean_candidates"] + sets:
        if not (nll_dir / f"{s}.npy").exists():
            continue
        vals = np.load(nll_dir / f"{s}.npy")
        D = _boot(vals, n, draws, seeded_rng("genppl", "cond", s, n))
        z = (float(D.mean()) - nmean) / nstd if nstd > 0 else float("nan")
        detected = bool(float(np.quantile(D, 0.025)) > q95)
        m = cond_meta.get(s, {})
        rows.append(
            {
                "condition": s,
                "experiment": m.get("experiment", ""),
                "op": m.get("op", ""),
                "layer": m.get("layer", ""),
                "dose": m.get("dose", ""),
                "n": n,
                "mean_logppl": round(float(np.nanmean(vals)), 4),
                "z": round(z, 4),
                "detected": detected,
            }
        )
        print(f"[genppl] {s}: z={z:.1f} det={detected} logppl={np.nanmean(vals):.3f}", flush=True)

    out_csv = cdir / "scores" / f"genppl_detection{args.out_suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[genppl] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
