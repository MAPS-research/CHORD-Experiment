"""Next-token predictions at CHORD's readout position (figure fig_token_verdict_decoding).

CHORD reads the hidden state at the final token of the coherence PromptEOL
template. This script asks what the language-model head of the same model
predicts at that position, i.e. the verdict an LLM would literally emit. The
input is built exactly as CHORD featurizes it (raw template, left padding so the
final prompt token is at index -1, no chat template), and decoding is manual
greedy over raw forward passes (``generate()`` would inject generation-config
behavior). next-k = the first k predicted tokens, detokenized and normalized
(strip, lowercase, surrounding punctuation removed).

Two analyses:
  content  (the figure) softmax at the readout position restricted to content
           tokens (no special, scaffold or whitespace tokens), renormalized;
           next-1 = top content token, next-3/5 = content-masked greedy
  argmax   the unmasked greedy argmax, reported as is (for a chat-tuned model
           this is typically the reasoning scaffold)

Conditions: benign paraphrase, full sentence shuffle and contradiction at the
highest severity, one passage per parent. CHORD's hidden-state z for the same
conditions is copied from ``<corpus-dir>/selectivity/selectivity_by_family.csv``
into the summary for comparison (not recomputed).

    python -m experiments.token_verdict_decoding.decode --analysis content \\
        --model Qwen/Qwen3.5-27B --corpus-dir outputs/counterfactual/meta_eval \\
        --out-dir outputs/experiments/token_verdict_decoding
    python -m experiments.token_verdict_decoding.decode --smoke   # CPU self-test

Outputs (out-dir): per-passage JSONL, per-horizon tally CSVs
(``tally_content_next{1,3,5}.csv`` feed the figure) and a summary JSON with the
total-variation distance between benign and harmful decoded-string distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from chord.embeddings import PROMPTEOL_TEMPLATES

COHERENCE_TEMPLATE = PROMPTEOL_TEMPLATES["coherence"]
CHORD_ENCODER = "qwen35-27b-prompteol-coherence-l62"

# condition -> role; benign first (every contrast is harmful minus benign)
GROUPS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("benign_paraphrase__dose3", "benign"),
        ("sentence_permutation__r100", "harmful"),
        ("contradiction__dose3", "harmful_llm"),
    ]
)
BENIGN = "benign_paraphrase__dose3"
HORIZONS = ("next1", "next3", "next5")
TOP_K = 15  # decoded strings kept per condition in the tally CSVs


def normalize_decoded(s: str) -> str:
    """Strip, lowercase, and remove surrounding whitespace / quotes / punctuation."""
    s = s.strip().lower()
    s = re.sub(r'^[\s"\'`“”‘’.,:;!?()\[\]{}\-–—*]+', "", s)
    s = re.sub(r'[\s"\'`“”‘’.,:;!?()\[\]{}\-–—*]+$', "", s)
    return s.strip()


def load_group(conditions_dir: Path, name: str) -> List[Tuple[str, str]]:
    """[(parent_id, text)] for one condition file, first row per parent."""
    seen = set()
    rows: List[Tuple[str, str]] = []
    with (conditions_dir / f"{name}.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = d.get("parent_id", f"idx-{len(rows)}")
            if pid in seen:
                continue
            seen.add(pid)
            rows.append((pid, d["text"]))
    return rows


def tally(norm_strings: List[str]) -> "OrderedDict[str, Tuple[int, float]]":
    """decoded string -> (count, share), most frequent first."""
    total = len(norm_strings)
    return OrderedDict(
        (s, (c, c / total if total else 0.0)) for s, c in Counter(norm_strings).most_common()
    )


def top_k_with_other(counts: "OrderedDict[str, Tuple[int, float]]", k: int):
    """Keep the top-k strings and collapse the rest into '(other)'."""
    items = list(counts.items())
    out = OrderedDict(items[:k])
    rest = items[k:]
    if rest:
        out["(other)"] = (sum(c for _, (c, _) in rest), sum(p for _, (_, p) in rest))
    return out


def shares(norm_strings: List[str]) -> Dict[str, float]:
    total = len(norm_strings)
    return {s: c / total for s, c in Counter(norm_strings).items()} if total else {}


def tv_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Total-variation distance between two decoded-string distributions."""
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def chord_reference(corpus_dir: Path) -> Dict[str, Dict[str, float]]:
    """CHORD's z for the decoded conditions, read from the selectivity CSV."""
    path = corpus_dir / "selectivity" / "selectivity_by_family.csv"
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for row in csv.DictReader(open(path)):
        if row.get("encoder") == CHORD_ENCODER and row.get("family") in GROUPS:
            out[row["family"]] = {
                "z_harmful": float(row["z_harmful"]),
                "z_benign": float(row["z_benign"]),
                "selectivity": float(row["selectivity"]),
            }
    return out


def load_model(name: str, revision: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    tok.padding_side = "left"  # the final prompt token sits at index -1 for every row
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return tok, model.eval()


def content_mask(tok, vocab_size: int) -> List[bool]:
    """True for content tokens: not special/added scaffold, surface has a letter."""
    special = set(int(i) for i in (tok.all_special_ids or []))
    scaffold = [
        "<think>",
        "</think>",
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_sep|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
    ]
    for t in scaffold:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            special.add(tid)
    try:
        for surface, tid in tok.get_added_vocab().items():
            s = surface.strip()
            if s in {"<think>", "</think>"} or (s.startswith("<|") and s.endswith("|>")):
                special.add(int(tid))
    except Exception:  # noqa: BLE001 - tokenizers without an added vocab
        pass
    pieces = tok.convert_ids_to_tokens(list(range(vocab_size)))
    return [
        i not in special
        and piece is not None
        and any(ch.isalpha() for ch in tok.convert_tokens_to_string([piece]))
        for i, piece in enumerate(pieces)
    ]


def decode_condition_sets(args) -> None:
    import torch

    corpus_dir = Path(args.corpus_dir)
    conditions_dir = corpus_dir / "texts" / "conditions"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(args.model, args.revision)
    content = args.analysis == "content"
    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    ban = None
    if content:
        mask = torch.tensor(content_mask(tok, vocab_size), dtype=torch.bool, device=model.device)
        ban = ~mask
        print(f"[decode] content vocabulary: {int(mask.sum())} / {vocab_size}", flush=True)

    def greedy(texts: List[str]) -> List[List[int]]:
        batch = tok(
            [COHERENCE_TEMPLATE.format(text=t) for t in texts],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        ids = batch["input_ids"].to(model.device)
        attn = batch["attention_mask"].to(model.device)
        out: List[List[int]] = [[] for _ in texts]
        for _ in range(args.n_tokens):
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=attn).logits[:, -1, :].float()
            if ban is not None:
                logits = logits.masked_fill(ban.unsqueeze(0), float("-inf"))
            nxt = logits.argmax(dim=-1)
            for i in range(len(texts)):
                out[i].append(int(nxt[i]))
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            ones = torch.ones((len(texts), 1), dtype=attn.dtype, device=attn.device)
            attn = torch.cat([attn, ones], dim=1)
        return out

    prefix = "tally_content" if content else "tally"
    raw_path = out_dir / ("token_decode_content_raw.jsonl" if content else "token_decode_raw.jsonl")
    decoded: Dict[str, Dict[str, List[str]]] = {c: {h: [] for h in HORIZONS} for c in GROUPS}
    with raw_path.open("w") as raw:
        for cond, group in GROUPS.items():
            rows = load_group(conditions_dir, cond)[: args.limit or None]
            print(f"[decode] {cond} ({group}): {len(rows)} passages", flush=True)
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                for (pid, _), ids in zip(batch, greedy([t for _, t in batch])):
                    rec = {"parent_id": pid, "condition": cond, "group": group, "token_ids": ids}
                    for h, k in zip(HORIZONS, (1, 3, 5)):
                        text = tok.decode(ids[:k], skip_special_tokens=content)
                        rec[h] = text
                        rec[f"{h}_norm"] = normalize_decoded(text)
                        decoded[cond][h].append(rec[f"{h}_norm"])
                    raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_tallies_and_summary(out_dir, decoded, prefix, chord_reference(corpus_dir), args.analysis)
    print(f"[decode] done -> {out_dir}", flush=True)


def write_tallies_and_summary(out_dir: Path, decoded, prefix: str, chord, analysis: str) -> None:
    for h in HORIZONS:
        with (out_dir / f"{prefix}_{h}.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["decoded_string", "condition", "count", "proportion"])
            for cond in GROUPS:
                for s, (c, p) in top_k_with_other(tally(decoded[cond][h]), TOP_K).items():
                    writer.writerow([s, cond, c, f"{p:.6f}"])
    summary = {
        "analysis": analysis,
        "coherence_template": COHERENCE_TEMPLATE,
        "conditions": dict(GROUPS),
        "n_passages": {c: len(decoded[c]["next1"]) for c in GROUPS},
        "horizons": {},
        "chord_hidden_state_reference": {"encoder": CHORD_ENCODER, "per_condition": chord},
    }
    for h in HORIZONS:
        top1 = {}
        for cond in GROUPS:
            counts = tally(decoded[cond][h])
            s, (c, p) = next(iter(counts.items())) if counts else ("", (0, 0.0))
            top1[cond] = {"string": s, "count": c, "share": round(p, 6)}
        tv = {
            f"TV_benign_vs_{cond}": round(
                tv_distance(shares(decoded[BENIGN][h]), shares(decoded[cond][h])), 6
            )
            for cond, role in GROUPS.items()
            if role.startswith("harmful")
        }
        summary["horizons"][h] = {"top1_per_condition": top1, "tv_distance": tv}
    name = "summary_content.json" if analysis == "content" else "summary.json"
    (out_dir / name).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def smoke() -> None:
    """CPU self-test: template identity, normalization and tally arithmetic."""
    expected = (
        'This passage : "{text}" , in terms of its logical coherence and the '
        "order of its ideas, means in one word:"
    )
    assert COHERENCE_TEMPLATE == expected, COHERENCE_TEMPLATE
    assert normalize_decoded('  "Coherent." ') == "coherent"
    assert normalize_decoded("**Jumbled**") == "jumbled"
    benign = ["coherent"] * 8 + ["clear"] * 2
    harmful = ["coherent"] * 7 + ["jumbled"] * 3
    assert abs(tv_distance(shares(benign), shares(harmful)) - 0.3) < 1e-9
    assert top_k_with_other(tally(benign), 1)["(other)"][0] == 2
    print("[smoke] template, normalization and tally arithmetic OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="CPU self-test, no model")
    parser.add_argument("--model", default=None, help="HF id or local path of the backbone")
    parser.add_argument(
        "--revision",
        default=None,
        help="Hub revision; use CHORD's pinned revision so the decode reads the same weights",
    )
    parser.add_argument("--analysis", choices=["content", "argmax"], default="content")
    parser.add_argument("--corpus-dir", default="outputs/counterfactual/meta_eval")
    parser.add_argument("--out-dir", default="outputs/experiments/token_verdict_decoding")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-tokens", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="passages per condition (0 = all)")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return
    if not args.model:
        parser.error("--model is required unless --smoke")
    decode_condition_sets(args)


if __name__ == "__main__":
    main()
