#!/usr/bin/env python3
"""Compute-cost appendix, re-measured (one GPU, one process, all rows).

Per encoder: parameters, output dim, analytic FLOPs per document
(2 x params x tokens, one prefill forward), throughput at the production batch
size (3 repeats, mean +- std), batch-1 latency per document (3 repeats on the
first 100 docs), peak GPU memory (both max_memory_allocated and
max_memory_reserved), weight bytes in the served dtype, and the CPU statistic
at two protocols: n = 137 / B = 200 null draws on the real features (capped
by the workload size), and the 10 x 500-fold protocol of Table 2 on Gaussian
features of the same width (the kernel cost depends on n and d only).
Workload: the first --docs documents of the packed OpenWebText reference
(500 in the paper, one Table-2 fold). Model loading and one warm-up batch are excluded.

gen-PPL is timed at its production batch (8) and at MAUVE-GPT2's batch (32)
with the same GPT-2-large, to settle the wall-clock gap between the two.

  python -m experiments.compute_cost.bench_cost \
      --out outputs/casestudy/unconditional_generation/scores/cost.csv
"""

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from chord.feature_encoding import embed_batched, get_cached_encoder
from chord.metrics.distribution import frechet_distance, median_bandwidth, rbf_mmd

from ..counterfactual_eval.nulls import exchangeable_attack, exchangeable_null
from ..generation_scoring.score import _split_reference
from ..metric_utils.external_metrics import GenerationPerplexityScorer

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "outputs/casestudy/single_fold/reference_packed512.jsonl"
PROMPTEOL = {
    "backend": "huggingface",
    "max_length": 532,
    "pooling": "prompteol",
    "prompteol_template": "coherence",
    "normalization": "none",
    "model_dtype": "bfloat16",
    "cache_dtype": "float32",
}
# name -> (table label, extractor label, production batch, protocol)
ENCODERS = {
    "chord-qwen3.5-27b": (
        "CHORD (Qwen3.5-27B)",
        "Qwen3.5-27B",
        8,
        dict(
            PROMPTEOL,
            model="Qwen/Qwen3.5-27B",
            layer=-3,
            revision="fc05daec18b0a78c049392ed2e771dde82bdf654",
            tokenizer_revision="fc05daec18b0a78c049392ed2e771dde82bdf654",
        ),
    ),
    "chord-qwen3.5-9b": (
        "CHORD (Qwen3.5-9B)",
        "Qwen3.5-9B",
        16,
        dict(
            PROMPTEOL,
            model="Qwen/Qwen3.5-9B",
            layer=-3,
            max_length=512,
            revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            tokenizer_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        ),
    ),
    "chord-qwen3.5-2b-student": (
        "CHORD (Qwen3.5-2B distilled)",
        "Qwen3.5-2B + LoRA + P_S",
        32,
        dict(PROMPTEOL, model=str(ROOT / "outputs/distill/student_qwen3.5-2b/final")),
    ),
    "chord-qwen3.5-0.8b-student": (
        "CHORD (Qwen3.5-0.8B distilled)",
        "Qwen3.5-0.8B + LoRA + P_S",
        32,
        dict(PROMPTEOL, model=str(ROOT / "outputs/distill/student_qwen3.5-0.8b/final")),
    ),
    "mauve-gpt2": (
        "MAUVE (GPT-2)",
        "GPT-2-large",
        32,
        {
            "backend": "huggingface",
            "model": "gpt2-large",
            "max_length": 512,
            "pooling": "last",
            "normalization": "none",
            "model_dtype": "float16",
            "cache_dtype": "float32",
            "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
            "tokenizer_revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        },
    ),
    "mauve-electra": (
        "MAUVE (ELECTRA)",
        "ELECTRA-large",
        64,
        {
            "backend": "huggingface",
            "model": "google/electra-large-discriminator",
            "max_length": 512,
            "pooling": "masked_mean",
            "normalization": "none",
            "model_dtype": "float32",
            "cache_dtype": "float32",
            "revision": "c13c3df7efadc2162f42588bd28eb4e187d602a5",
            "tokenizer_revision": "c13c3df7efadc2162f42588bd28eb4e187d602a5",
        },
    ),
    "fbd-bert": (
        "FBD",
        "BERT-base",
        64,
        {
            "backend": "huggingface",
            "model": "bert-base-uncased",
            "max_length": 512,
            "pooling": "pooler_output",
            "normalization": "none",
            "model_dtype": "float32",
            "cache_dtype": "float32",
            "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
            "tokenizer_revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        },
    ),
    "minilm-cls": (
        "MMD-MiniLM",
        "MiniLM-L6",
        64,
        {
            "backend": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 512,
            "pooling": "cls",
            "normalization": "none",
            "model_dtype": "float32",
            "cache_dtype": "float32",
            "revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "tokenizer_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        },
    ),
}
GENPPL = {
    "model": "gpt2-large",
    "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
    "tokenizer_revision": "32b71b12589c2f8d625668d2335a01cac3249519",
    "max_length": 512,
    "device": "cuda:0",
    "model_dtype": "float16",
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peaks():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


def timed(fn, repeats):
    out = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t)
    return float(np.mean(out)), float(np.std(out, ddof=1)) if repeats > 1 else 0.0


def param_stats(module):
    n = sum(p.numel() for p in module.parameters())
    b = sum(p.numel() * p.element_size() for p in module.parameters())
    return n, b


def statistic_costs(feats, n_small, draws, n_fold, folds, seed):
    """(calib_137, persys_137, sigma_fit_500, persys_500_per_fold, persys_500_total)."""
    ref = feats.astype(np.float64)
    dev, cal, test = _split_reference(ref, seed, 150)
    n = min(n_small, len(cal) // 2, len(test) // 2)
    t = time.perf_counter()
    sigma = median_bandwidth(dev, max_samples=2000, seed=seed)
    exchangeable_null(
        cal, n, n, draws=draws, sigma=sigma, block_size=1024, rng=np.random.default_rng(0)
    )
    calib = time.perf_counter() - t
    t = time.perf_counter()
    exchangeable_attack(
        cal, test[:n], n, n, draws=draws, sigma=sigma, block_size=1024, rng=np.random.default_rng(1)
    )
    persys = time.perf_counter() - t
    # fold protocol: same width, Gaussian features (kernel cost depends on n, d only)
    rng = np.random.default_rng(seed)
    d = ref.shape[1]
    g_dev = rng.standard_normal((150, d))
    g_ref = rng.standard_normal((n_fold * folds, d))
    g_cand = rng.standard_normal((n_fold * folds, d))
    t = time.perf_counter()
    s2 = median_bandwidth(g_dev, max_samples=2000, seed=seed)
    fit = time.perf_counter() - t
    t = time.perf_counter()
    for k in range(folds):
        rbf_mmd(
            g_ref[k * n_fold : (k + 1) * n_fold],
            g_cand[k * n_fold : (k + 1) * n_fold],
            s2,
            False,
            1024,
        )
    total = time.perf_counter() - t
    return calib, persys, fit, total / folds, total, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    ap.add_argument("--docs", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--latency-docs", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-genppl", action="store_true")
    ap.add_argument(
        "--out", default=str(ROOT / "outputs/casestudy/unconditional_generation/scores/cost.csv")
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seed = 20260707

    texts = [json.loads(line)["text"] for line in open(REFERENCE, encoding="utf-8")][: args.docs]
    env = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "cpu_threads": args.threads,
        "docs": len(texts),
        "repeats": args.repeats,
    }
    print("[bench]", json.dumps(env), flush=True)
    rows = []

    for name in args.encoders:
        label, extractor, batch, protocol = ENCODERS[name]
        if protocol["model"].startswith("/") and not Path(protocol["model"]).is_dir():
            print(f"[bench] {name}: checkpoint missing, skip", flush=True)
            continue
        # warm-up (builds + caches the encoder; excluded from every timing)
        embed_batched(texts[:batch], protocol, batch_size=batch)
        enc = get_cached_encoder(protocol)
        enc._ensure_model() if hasattr(enc, "_ensure_model") else None
        model = enc.model
        n_params, w_bytes = param_stats(model)
        proj = getattr(enc, "projector", None)
        p_params = sum(p.numel() for p in proj.parameters()) if proj is not None else 0
        tok = enc.tokenizer
        toks = [
            min(len(tok(t, add_special_tokens=True)["input_ids"]), enc.max_length) for t in texts
        ]
        mean_tok = float(np.mean(toks))

        reset_peak()
        wall, wall_sd = timed(
            lambda: embed_batched(texts, protocol, batch_size=batch), args.repeats
        )
        alloc, reserved = peaks()
        feats = np.asarray(embed_batched(texts, protocol, batch_size=batch), dtype=np.float32)
        sub = texts[: args.latency_docs]
        lat, lat_sd = timed(lambda: embed_batched(sub, protocol, batch_size=1), args.repeats)
        calib, persys, fit500, per_fold, total500, n_used = statistic_costs(
            feats, 137, 200, 500, 10, seed
        )
        row = {
            "encoder": name,
            "label": label,
            "extractor": extractor,
            "batch": batch,
            "params_M": round(n_params / 1e6, 1),
            "projector_params_K": round(p_params / 1e3, 1),
            "weights_gb": round(w_bytes / 2**30, 2),
            "dim": int(feats.shape[1]),
            "mean_tokens": round(mean_tok, 1),
            "gflops_per_doc": round(2 * (n_params + p_params) * mean_tok / 1e9, 1),
            "wall_s": round(wall, 2),
            "wall_sd": round(wall_sd, 2),
            "docs_per_s": round(len(texts) / wall, 1),
            "tokens_per_s": round(len(texts) * mean_tok / wall, 0),
            "latency_ms_b1": round(1000 * lat / len(sub), 2),
            "latency_sd_ms": round(1000 * lat_sd / len(sub), 2),
            "peak_alloc_gb": round(alloc, 2),
            "peak_reserved_gb": round(reserved, 2),
            "calib_s_n137": round(calib, 2),
            "persys_s_n137": round(persys, 2),
            "n_small_used": n_used,
            "sigma_fit_s_n500": round(fit500, 3),
            "persys_s_n500_per_fold": round(per_fold, 2),
            "persys_s_n500_10folds": round(total500, 1),
        }
        # baseline statistics on the same features, for the CPU columns
        if name == "fbd-bert":
            half = len(feats) // 2
            t = time.perf_counter()
            frechet_distance(feats[:half], feats[half:])
            row["fbd_s"] = round(time.perf_counter() - t, 3)
        if name in ("mauve-gpt2", "mauve-electra"):
            try:
                import mauve

                half = len(feats) // 2
                t = time.perf_counter()
                mauve.compute_mauve(
                    p_features=feats[:half],
                    q_features=feats[half:],
                    num_buckets="auto",
                    kmeans_explained_var=0.9,
                    kmeans_num_redo=5,
                    kmeans_max_iter=500,
                    verbose=False,
                    seed=seed,
                )
                row["mauve_s"] = round(time.perf_counter() - t, 2)
            except Exception as e:  # noqa: BLE001
                row["mauve_s"] = f"n/a ({type(e).__name__})"
        rows.append(row)
        print(
            f"[bench] {name:14s} {n_params / 1e9:5.2f}B dim {feats.shape[1]:5d} "
            f"tok {mean_tok:5.0f} | "
            f"wall {wall:6.1f}±{wall_sd:.1f}s {len(texts) / wall:6.1f} docs/s | "
            f"b1 {1000 * lat / len(sub):6.1f} ms/doc | "
            f"peak {alloc:5.1f}/{reserved:5.1f} GB | n137 calib {calib:.1f}s sys {persys:.1f}s | "
            f"n500 fold {per_fold:.1f}s x10 {total500:.0f}s",
            flush=True,
        )

    if not args.skip_genppl:
        for bs in (8, 32):
            times, peak_a, peak_r = [], 0.0, 0.0
            for r in range(args.repeats):
                tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name
                scorer = GenerationPerplexityScorer(dict(GENPPL, batch_size=bs, cache_path=tmp))
                scorer.perplexity(texts[:bs])  # warm-up + model load, excluded
                reset_peak()
                sync()
                t = time.perf_counter()
                scorer.perplexity(texts[bs:] if r else texts)  # first repeat scores the full set
                sync()
                dt = time.perf_counter() - t
                a, rr = peaks()
                peak_a, peak_r = max(peak_a, a), max(peak_r, rr)
                # normalize every repeat to the full --docs workload
                times.append(dt * len(texts) / (len(texts) if r == 0 else len(texts) - bs))
                n_params, w_bytes = param_stats(scorer._ensure_model())
                del scorer
                os.unlink(tmp)
                import gc

                gc.collect()
                torch.cuda.empty_cache()
            wall, wall_sd = float(np.mean(times)), float(np.std(times, ddof=1))
            rows.append(
                {
                    "encoder": f"genppl-b{bs}",
                    "label": f"gen-PPL (batch {bs})",
                    "extractor": "GPT-2-large",
                    "batch": bs,
                    "params_M": round(n_params / 1e6, 1),
                    "projector_params_K": 0,
                    "weights_gb": round(w_bytes / 2**30, 2),
                    "dim": "",
                    "mean_tokens": "",
                    "gflops_per_doc": "",
                    "wall_s": round(wall, 2),
                    "wall_sd": round(wall_sd, 2),
                    "docs_per_s": round(len(texts) / wall, 1),
                    "tokens_per_s": "",
                    "latency_ms_b1": "",
                    "latency_sd_ms": "",
                    "peak_alloc_gb": round(peak_a, 2),
                    "peak_reserved_gb": round(peak_r, 2),
                }
            )
            print(
                f"[bench] gen-PPL batch {bs}: wall {wall:.1f}±{wall_sd:.1f}s  "
                f"{len(texts) / wall:.1f} docs/s  "
                f"peak {peak_a:.1f}/{peak_r:.1f} GB",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out.with_suffix(".env.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    print("\nwrote", out, "and", out.with_suffix(".env.json"))


if __name__ == "__main__":
    main()
