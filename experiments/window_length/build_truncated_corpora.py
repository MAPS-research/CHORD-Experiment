"""Sentence-boundary truncations of the unconditional-generation corpora.

Paper result: appendix "Sensitivity to Evaluation Window Length" (Tables
tab:mauve-length and tab:chord-length): MAUVE and CHORD on Table 2's folds
truncated to 128, 256, 384, 512 and 1024 tokens.

Every document of Table 2 (the human reference pool, the held-out human folds and
every generator run of experiments/generator_eval) keeps its position, so the
folds of Table 2 carry over unchanged, and is cut to its longest WHOLE-SENTENCE
prefix within a Qwen-token budget
L, so the coherence readout never sees a dangling clause right before the prompt
suffix. A document whose first sentence alone exceeds L falls back to its
L-token prefix. The length knob lives entirely in the text:

* CHORD (Qwen3.5-27B, coherence prompt, layer -3) gets an encoder budget of
  max(L + 80, longest emitted text + 40) tokens, so the prompt suffix always fits;
* MAUVE (GPT-2-large, last token) keeps GPT-2's full 1024-token window, so it
  never truncates a second time.

For each L this writes a self-contained directory (all paths relative)

    <out>/L{L}/reference.jsonl, samples/<run_id>.jsonl, runs.jsonl,
              trunc_stats.json, chord.yaml, mauve.yaml

Inputs   --source-config  Table 2's config: runs manifest, human reference pool and
                          the seed of its fold permutation, plus the CHORD 27B encoder
                          protocol (default experiments/generator_eval/configs/featurize.yaml)

    python -m experiments.window_length.build_truncated_corpora --lengths 128,256,384,512,1024
    # then, per length (window_length.slurm; GPU for featurize, CPU for scoring):
    python -m chord.featurize --config outputs/experiments/window_length_folds/L512/chord.yaml
    python -m chord.featurize --config outputs/experiments/window_length_folds/L512/mauve.yaml
    python -m experiments.window_length.score_folds \\
        --dir outputs/experiments/window_length_folds/L512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import yaml

from chord.text import sentences
from chord.utils.io import load_runs, load_texts

CHORD_ENCODER = "qwen35-27b-prompteol-coherence-l62"
HEADROOM = 80  # encoder budget = L + HEADROOM (the prompt template is ~24-30 tokens)
GPT2_CONTEXT = 1024
MAUVE_PROTOCOL = {
    "backend": "huggingface",
    "model": "gpt2-large",
    "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
    "tokenizer_revision": "32b71b12589c2f8d625668d2335a01cac3249519",
    "max_length": GPT2_CONTEXT,
    "pooling": "last",
    "normalization": "none",
    "model_dtype": "float16",
    "cache_dtype": "float32",
}

Truncation = Tuple[str, int, int]  # (text, sentences kept, Qwen tokens)


def cumulative_token_counts(text: str, tokenizer, cap: int) -> List[int]:
    """Qwen-token count of each whole-sentence prefix of ``text`` (measured on the
    exact string that will be written), stopping once the count exceeds ``cap``."""
    parts = sentences(text)
    counts: List[int] = []
    for k in range(1, len(parts) + 1):
        n = len(tokenizer(" ".join(parts[:k]), add_special_tokens=False)["input_ids"])
        counts.append(n)
        if n > cap:
            break
    return counts


def token_prefix(text: str, tokenizer, length: int) -> Tuple[str, int]:
    """Fixed-token prefix of ``text`` whose re-tokenized length is <= ``length``."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:length]
    while True:
        emitted = " ".join(tokenizer.decode(ids, skip_special_tokens=True).split())
        n = len(tokenizer(emitted, add_special_tokens=False)["input_ids"])
        if n <= length or not ids:
            return emitted, n
        ids = ids[:-1]


def truncate(texts: List[str], tokenizer, lengths: List[int]) -> Dict[int, List[Truncation]]:
    cap = max(lengths)
    out: Dict[int, List[Truncation]] = {length: [] for length in lengths}
    for text in texts:
        parts = sentences(text)
        counts = cumulative_token_counts(text, tokenizer, cap)
        for length in lengths:
            k = sum(1 for n in counts if n <= length)
            if k:
                out[length].append((" ".join(parts[:k]), k, counts[k - 1]))
            else:
                emitted, n = token_prefix(text, tokenizer, length)
                out[length].append((emitted, 1, n))
    return out


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(recs: List[Truncation], length: int) -> Dict:
    return {
        "n": len(recs),
        "n_sentences_mean": mean(r[1] for r in recs),
        "gen_tok_mean": mean(r[2] for r in recs),
        "n_over_budget": sum(r[2] > length for r in recs),
    }


def dump_config(path: Path, header: str, cfg: Dict) -> None:
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source-config", default="experiments/generator_eval/configs/featurize.yaml")
    ap.add_argument("--lengths", default="128,256,384,512,1024")
    ap.add_argument("--out", default="outputs/experiments/window_length_folds")
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer that measures the budget (default: the CHORD encoder's model)",
    )
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    source_path = Path(args.source_config).resolve()
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    encoder = next(e for e in source["encoders"] if e["name"] == CHORD_ENCODER)
    normalization = str(source.get("text_normalization", "none"))
    runs = load_runs(source_path.parent / source["runs_manifest"], require_samples=True)
    reference_cfg = source["reference"]
    reference = load_texts(
        source_path.parent / reference_cfg["path"],
        reference_cfg.get("text_key", "text"),
        normalization,
    )[: int(reference_cfg.get("limit", 10**9))]

    from transformers import AutoTokenizer

    protocol = encoder["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or protocol["model"],
        revision=None if args.tokenizer else protocol.get("tokenizer_revision"),
    )

    print(f"[window] truncating the reference ({len(reference)} docs)")
    ref_trunc = truncate(reference, tokenizer, lengths)
    run_trunc = {}
    for run in runs:
        texts = load_texts(run.samples_path, run.text_key, normalization)
        run_trunc[run.run_id] = (run, truncate(texts, tokenizer, lengths))
        print(f"[window]   {run.run_id:22s} n={len(texts)}")

    for length in lengths:
        out = Path(args.out) / f"L{length}"
        write_jsonl(out / "reference.jsonl", [{"text": r[0]} for r in ref_trunc[length]])
        manifest, stats = [], {"length": length, "reference": summarize(ref_trunc[length], length)}
        stats["runs"] = {}
        for run_id, (run, per_length) in run_trunc.items():
            recs = per_length[length]
            write_jsonl(out / "samples" / f"{run_id}.jsonl", [{"text": r[0]} for r in recs])
            manifest.append(
                {
                    "run_id": run.run_id,
                    "model": run.model,
                    "family": run.family,
                    "seed": run.seed,
                    "samples_path": f"samples/{run_id}.jsonl",
                    "setting": run.setting,
                }
            )
            stats["runs"][run_id] = {"model": run.model, "family": run.family}
            stats["runs"][run_id].update(summarize(recs, length))
        write_jsonl(out / "runs.jsonl", manifest)

        emitted = ref_trunc[length] + [r for _, pl in run_trunc.values() for r in pl[length]]
        longest = max(r[2] for r in emitted)
        chord_protocol = dict(protocol, max_length=max(length + HEADROOM, longest + 40))
        stats["max_emitted_text_tokens"] = int(longest)
        stats["encoder_max_length"] = int(chord_protocol["max_length"])
        (out / "trunc_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

        common = {
            "seed": int(source.get("seed", 0)),
            "runs_manifest": "runs.jsonl",
            "output_dir": ".",
            "text_normalization": normalization,
            "reference": {"path": "reference.jsonl", "text_key": "text", "limit": len(reference)},
        }
        dump_config(
            out / "chord.yaml",
            f"# CHORD 27B on sentence-boundary truncations, L={length} Qwen tokens.\n"
            f"# Generated by experiments.window_length.build_truncated_corpora;\n"
            f"# encoder budget {chord_protocol['max_length']} keeps the prompt suffix.\n",
            dict(
                common,
                encoders=[
                    {
                        "name": CHORD_ENCODER,
                        "batch_size": int(encoder.get("batch_size", 8)),
                        "protocol": chord_protocol,
                    }
                ],
            ),
        )
        dump_config(
            out / "mauve.yaml",
            f"# MAUVE (GPT-2-large, last token) on the same truncations, L={length}.\n"
            f"# Generated by experiments.window_length.build_truncated_corpora;\n"
            f"# the extractor keeps GPT-2's full {GPT2_CONTEXT}-token window.\n",
            dict(
                common,
                encoders=[
                    {
                        "name": f"mauve-gpt2-matched-L{length}",
                        "batch_size": 8,
                        "protocol": dict(MAUVE_PROTOCOL),
                    }
                ],
            ),
        )
        print(f"[window] L{length}: wrote {out}")


if __name__ == "__main__":
    main()
