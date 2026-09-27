"""Build the multi-domain distillation corpus (topic-invariance fix, 2026-06-28).

The single-source OWT RKD corpus made the student's geometry document-SOURCE
dominated (it inverted on the cross-source splice-clean test). This mixes three
clearly distinct domains so the RKD-Gram forces the student to match the teacher's
geometry ACROSS domains:

  OWT (web/news)      <- outputs/casestudy/distill3/train.jsonl  (the original train pool)
  Wikipedia (encyc.)  <- outputs/counterfactual/wikitext103/texts/reference.jsonl
  Reddit (informal)   <- HF cache trl-lib/tldr  (POST body, scaffolding stripped)

2200 each = 6600, length-filtered to 600-1800 chars so style/topic vary while length
is roughly controlled. wiki/reddit are probe-guarded against the eval splice/paraphrase
conditions (OWT is the original train pool -> used as-is). Writes train_diverse.jsonl +
a held-out val_reddit.jsonl (owt/wiki vals already exist under distill/). Output is the
multi-domain source pool used by the distillation corpus recipes.

Run from the repo root:  python -m experiments.generation_scoring.build_diverse_corpus
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re

SEED = 20260628
OUT = "outputs/casestudy/diverse_corpus"
MIN_CHARS, MAX_CHARS = 600, 1800
PER_DOMAIN = 2200
# Reddit source: the trl-lib/tldr dataset cached as arrow files. Download once
# with `datasets.load_dataset("trl-lib/tldr")`, then point this at the cache:
TLDR_GLOB = os.environ.get(
    "CHORD_TLDR_GLOB",
    os.path.expanduser("~/.cache/huggingface/datasets/trl-lib___tldr/**/*.arrow"),
)

# eval probes to exclude from the NEW (wiki/reddit) sources -> keep the topic-invariance
# test honest. (OWT is the original training pool and is used unguarded.)
PROBE_PATHS = [
    "outputs/counterfactual/splice_counterfactual/texts/clean_candidates.jsonl",
    "outputs/counterfactual/splice_counterfactual/texts/conditions/human_splice.jsonl",
    "outputs/counterfactual/splice_counterfactual/texts/conditions/dlm_splice.jsonl",
    "outputs/counterfactual/counterfactual/texts/conditions/benign_paraphrase.jsonl",
]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _ok(t: str) -> bool:
    return MIN_CHARS <= len(t) <= MAX_CHARS


def _h(t: str) -> str:
    return hashlib.md5(_norm(t).lower()[:200].encode()).hexdigest()


def _load_probe_hashes() -> set:
    probes = set()
    for p in PROBE_PATHS:
        try:
            for line in open(p):
                t = json.loads(line).get("text", "")
                if t:
                    probes.add(_h(t))
        except FileNotFoundError:
            pass
    return probes


def _take(texts, n, label, probes, guard=True):
    seen, out = set(), []
    for t in texts:
        t = _norm(t)
        if not _ok(t):
            continue
        k = _h(t)
        if (guard and k in probes) or k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= n:
            break
    print(f"  {label:10s} kept {len(out)} / target {n}")
    return [{"text": t, "domain": label} for t in out]


def _reddit_bodies():
    import pyarrow.ipc as ipc

    f = sorted(glob.glob(TLDR_GLOB, recursive=True))[0]
    try:
        tb = ipc.open_file(f).read_all()
    except Exception:
        tb = ipc.open_stream(f).read_all()
    out = []
    for p in tb.column("prompt").to_pylist():
        m = re.search(r"POST:\s*(.*)", p, re.S)
        body = m.group(1) if m else p
        out.append(re.sub(r"\*\*[^*]+\*\*", "", body))  # drop **repost** style markers
    return out


def main() -> None:
    random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    probes = _load_probe_hashes()
    print("probe-guard hashes:", len(probes))

    owt = [json.loads(line)["text"] for line in open("outputs/casestudy/distill3/train.jsonl")]
    wiki = [
        json.loads(line).get("text", "")
        for line in open("outputs/counterfactual/wikitext103/texts/reference.jsonl")
    ]
    red = _reddit_bodies()
    for pool in (owt, wiki, red):
        random.shuffle(pool)

    rows = (
        _take(owt, PER_DOMAIN, "owt", probes, guard=False)
        + _take(wiki, PER_DOMAIN, "wiki", probes)
        + _take(red, PER_DOMAIN, "reddit", probes)
    )
    random.shuffle(rows)
    with open(f"{OUT}/train_diverse.jsonl", "w") as g:
        for r in rows:
            g.write(json.dumps(r) + "\n")

    val_red = _take(red[PER_DOMAIN + 500 :], 300, "reddit-val", probes)
    with open(f"{OUT}/val_reddit.jsonl", "w") as g:
        for r in val_red:
            g.write(json.dumps(r) + "\n")

    from collections import Counter

    print("TOTAL train_diverse:", len(rows), "balance:", dict(Counter(r["domain"] for r in rows)))


if __name__ == "__main__":
    main()
