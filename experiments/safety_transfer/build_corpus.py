"""Build the safety-transfer corpus: unsafe-content prevalence in a response set.

Produces the corpus of the appendix "Preliminary Transfer to Unsafe-Content
Prevalence" (Table "Construction of the safety-prevalence experiment").

PKU-SafeRLHF pairs two responses to the same prompt with per-response safety
labels. Rows where exactly one response is safe give a (safe, unsafe) pair from
the same prompt and the same generator. Those donors replace documents of a
clean response corpus at a nested prevalence ladder:

  unsafe__pXXX   donor = the unsafe response                    harmful arm
  safe__pXXX     donor = the same prompt's safe sibling         matched control
  ood__pXXX      donor = a safe response from another generator generator-shift control
  clean__p000    no replacement                                 benign anchor

Three properties hold by construction, so the arms differ only in the content of
the injected text:

  * same slots: every arm replaces the same clean documents in the same nested
    order, so the 2% corpus contains the 1% replacements;
  * same length bin: donor pairs, generator-shift donors and the replaced clean
    documents share a word-count bin, so the length histogram is identical across
    arms and rates;
  * disjoint prompts: reference, clean candidates, donors and generator-shift
    donors use non-overlapping prompts.

The reference consists of safe responses to the same red-team prompts (rows
where both responses are safe), so it already contains refusals and safe
handling of harmful topics.

Inputs:  the two PKU-SafeRLHF files named in the config (download_sources).
Outputs: ``<output_dir>/texts/{reference,clean_candidates,pool}.jsonl``,
``texts/conditions/<condition>.jsonl``, ``index/<set>.json`` (row -> pool index,
used by ``featurize compose``), ``conditions_manifest.json`` and
``safety_corpus_summary.json``.

    python -m experiments.safety_transfer.build_corpus \
        --config experiments/safety_transfer/configs/corpus.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

from chord.utils.hashing import stable_seed

# Word-count bins. The top bin is open-ended: few responses exceed 110 words and
# same-bin donor pairs are too scarce above that for a finer split.
BINS: List[Tuple[int, float]] = [(0, 40), (40, 60), (60, 80), (80, 110), (110, float("inf"))]

# Diagnostic only, never used to select documents: a donor arm that is more
# refusal-like than the reference would be a confound, so it is reported.
REFUSAL = re.compile(
    r"^\s*(i'?m sorry|i am sorry|sorry,|i cannot|i can'?t|i will not|i won'?t"
    r"|as an ai|it is not appropriate|it'?s not appropriate)",
    re.I,
)


def word_count(text: str) -> int:
    return len(text.split())


def bin_of(text: str) -> int:
    n = word_count(text)
    for b, (lo, hi) in enumerate(BINS):
        if lo <= n < hi:
            return b
    return len(BINS) - 1


def read_rows(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pick_response(row: Dict, key: str) -> str:
    """Pick one of two equally safe responses by a stable hash of the prompt,
    so neither response slot of the release is systematically preferred."""
    return row["response_%d" % (stable_seed(key, row["prompt"]) % 2)]


def dedup_by_prompt(rows: Sequence[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for row in rows:
        if row["prompt"] in seen:
            continue
        seen.add(row["prompt"])
        out.append(row)
    return out


def quota_from_histogram(hist: Counter, total: int) -> List[int]:
    """Per-bin counts summing to `total`, proportional to `hist` (largest remainder)."""
    n = sum(hist.values())
    exact = [hist.get(b, 0) / n * total for b in range(len(BINS))]
    quota = [int(x) for x in exact]
    order = sorted(range(len(BINS)), key=lambda b: exact[b] - quota[b], reverse=True)
    for b in order[: total - sum(quota)]:
        quota[b] += 1
    return quota


def take_per_bin(
    items_by_bin: Dict[int, List], quota: Sequence[int], rng: random.Random, what: str
) -> List:
    """Sample quota[b] items from each bin; a short bin is an error, since a
    silently smaller arm would break the length matching."""
    out = []
    for b, k in enumerate(quota):
        avail = items_by_bin.get(b, [])
        if len(avail) < k:
            raise SystemExit(
                f"{what}: bin {b} ({BINS[b][0]}-{BINS[b][1]} words) needs {k} "
                f"items but only {len(avail)} are available"
            )
        out.extend(rng.sample(avail, k))
    return out


def arm_stats(texts: Sequence[str]) -> Dict:
    wc = [word_count(t) for t in texts]
    return {
        "n": len(texts),
        "words_mean": round(sum(wc) / max(1, len(wc)), 1),
        "words_median": sorted(wc)[len(wc) // 2] if wc else 0,
        "refusal_rate": round(sum(bool(REFUSAL.match(t)) for t in texts) / max(1, len(texts)), 4),
        "bins": dict(Counter(bin_of(t) for t in texts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    out_dir = Path(cfg["output_dir"])
    corpus_name = cfg.get("corpus_name", out_dir.name)
    seed = int(cfg.get("seed", 20260829))
    n_reference = int(cfg["n_reference"])
    n_candidates = int(cfg["n_candidates"])
    doses = [float(d) for d in cfg["doses"]]
    n_donors = max(int(round(d * n_candidates)) for d in doses)
    arms = list(cfg.get("arms", ["unsafe", "safe", "ood"]))

    rng = random.Random(seed)
    main_rows = read_rows(Path(cfg["sources"]["main"]))
    ood_rows = read_rows(Path(cfg["sources"]["ood"])) if "ood" in arms else []

    # ---- pools, disjoint by prompt ------------------------------------------
    both_safe = dedup_by_prompt(
        [r for r in main_rows if r["is_response_0_safe"] and r["is_response_1_safe"]]
    )
    one_safe = dedup_by_prompt(
        [r for r in main_rows if r["is_response_0_safe"] != r["is_response_1_safe"]]
    )
    used_prompts = {r["prompt"] for r in both_safe}
    one_safe = [r for r in one_safe if r["prompt"] not in used_prompts]

    rng.shuffle(both_safe)
    if len(both_safe) < n_reference + n_candidates:
        raise SystemExit(
            f"both-safe pool has {len(both_safe)} prompts, need {n_reference + n_candidates}"
        )
    ref_rows = both_safe[:n_reference]
    cand_rows = both_safe[n_reference : n_reference + n_candidates]

    # ---- donor pairs: same length bin on both sides -------------------------
    pairs = []
    for r in one_safe:
        i = 0 if r["is_response_0_safe"] else 1
        safe_t, unsafe_t = r["response_%d" % i], r["response_%d" % (1 - i)]
        if bin_of(safe_t) == bin_of(unsafe_t):
            pairs.append(
                {"prompt": r["prompt"], "safe": safe_t, "unsafe": unsafe_t, "bin": bin_of(safe_t)}
            )
    pairs_by_bin: Dict[int, List] = defaultdict(list)
    for p in pairs:
        pairs_by_bin[p["bin"]].append(p)

    cand_hist = Counter(bin_of(pick_response(r, "cand")) for r in cand_rows)
    quota = quota_from_histogram(cand_hist, n_donors)
    donors = take_per_bin(pairs_by_bin, quota, rng, "donor pairs")
    rng.shuffle(donors)  # slot order; each rate uses a prefix of it

    # ---- generator-shift donors: same bin sequence, other generator, safe ----
    ood_by_bin: Dict[int, List] = defaultdict(list)
    if "ood" in arms:
        donor_prompts = {p["prompt"] for p in donors}
        ood_pool = dedup_by_prompt(
            [
                r
                for r in ood_rows
                if r["is_response_0_safe"]
                and r["is_response_1_safe"]
                and r["prompt"] not in used_prompts
                and r["prompt"] not in donor_prompts
            ]
        )
        for r in ood_pool:
            t = pick_response(r, "ood")
            ood_by_bin[bin_of(t)].append(t)
        ood_quota = [0] * len(BINS)
        for d in donors:
            ood_quota[d["bin"]] += 1
        ood_texts = take_per_bin(ood_by_bin, ood_quota, rng, "ood donors")
        by_bin_iter: Dict[int, List[str]] = defaultdict(list)
        for t in ood_texts:
            by_bin_iter[bin_of(t)].append(t)
        for d in donors:  # one generator-shift donor per slot, same bin
            d["ood"] = by_bin_iter[d["bin"]].pop()

    # ---- replacement slots: same bin as the donor, distinct clean documents --
    cand_texts = [pick_response(r, "cand") for r in cand_rows]
    free_by_bin: Dict[int, List[int]] = defaultdict(list)
    for i, t in enumerate(cand_texts):
        free_by_bin[bin_of(t)].append(i)
    for b in free_by_bin:
        rng.shuffle(free_by_bin[b])
    slots: List[int] = []
    for d in donors:
        if not free_by_bin[d["bin"]]:
            raise SystemExit(f"no free clean document left in bin {d['bin']}")
        slots.append(free_by_bin[d["bin"]].pop())

    # ---- emit ----------------------------------------------------------------
    texts_dir = out_dir / "texts"
    (texts_dir / "conditions").mkdir(parents=True, exist_ok=True)
    (out_dir / "index").mkdir(parents=True, exist_ok=True)

    pool: List[str] = []
    pool_index: Dict[str, int] = {}

    def pool_id(text: str) -> int:
        if text not in pool_index:
            pool_index[text] = len(pool)
            pool.append(text)
        return pool_index[text]

    def write_set(name: str, texts: Sequence[str], ids: Sequence[str], path: Path) -> None:
        with path.open("w") as fh:
            for t, rid in zip(texts, ids):
                fh.write(json.dumps({"id": rid, "parent_id": rid, "text": t}) + "\n")
        idx = [pool_id(t) for t in texts]
        (out_dir / "index" / f"{name}.json").write_text(json.dumps(idx))

    ref_texts = [pick_response(r, "ref") for r in ref_rows]
    write_set(
        "reference",
        ref_texts,
        [f"ref{i:05d}" for i in range(len(ref_texts))],
        texts_dir / "reference.jsonl",
    )
    cand_ids = [f"cand{i:05d}" for i in range(len(cand_texts))]
    write_set("clean_candidates", cand_texts, cand_ids, texts_dir / "clean_candidates.jsonl")
    # baseline_selectivity reads parent ids of every set, clean_candidates
    # included, from texts/conditions/.
    shutil.copy(
        texts_dir / "clean_candidates.jsonl", texts_dir / "conditions" / "clean_candidates.jsonl"
    )

    conditions = [
        {
            "name": "clean__p000",
            "group": "safety",
            "experiment": "safety",
            "kind": "benign",
            "role": "benign",
            "dose": 0.0,
            "dose_kind": "rate",
            "criterion": "safety",
            "arm": "clean",
            "n_injected": 0,
        }
    ]
    write_set("clean__p000", cand_texts, cand_ids, texts_dir / "conditions" / "clean__p000.jsonl")

    summary = {
        "corpus": corpus_name,
        "seed": seed,
        "bins": [list(b) for b in BINS],
        "pools": {
            "both_safe_prompts": len(both_safe),
            "one_safe_prompts": len(one_safe),
            "same_bin_pairs": len(pairs),
            "donors_selected": len(donors),
        },
        "donor_quota_by_bin": quota,
        "arms": {},
        "conditions": [],
    }
    for arm in arms:
        summary["arms"][arm] = arm_stats([d[arm] for d in donors])
    summary["arms"]["clean_candidates"] = arm_stats(cand_texts)
    summary["arms"]["reference"] = arm_stats(ref_texts)
    summary["arms"]["replaced_slots"] = arm_stats([cand_texts[s] for s in slots])

    for arm in arms:
        for dose in doses:
            k = int(round(dose * n_candidates))
            texts = list(cand_texts)
            ids = list(cand_ids)
            for j in range(k):
                texts[slots[j]] = donors[j][arm]
                ids[slots[j]] = f"{arm}{j:04d}"
            name = f"{arm}__p{int(round(dose * 100)):03d}"
            write_set(name, texts, ids, texts_dir / "conditions" / f"{name}.jsonl")
            conditions.append(
                {
                    "name": name,
                    "group": "safety",
                    "experiment": "safety",
                    "kind": "harmful" if arm == "unsafe" else "benign_dose",
                    "role": "harmful" if arm == "unsafe" else "control",
                    "dose": dose,
                    "dose_kind": "rate",
                    "criterion": "safety",
                    "arm": arm,
                    "n_injected": k,
                }
            )
            summary["conditions"].append({"name": name, "arm": arm, "dose": dose, "n_injected": k})
            print(f"[safety] {name}: {k}/{n_candidates} replaced", flush=True)

    with (texts_dir / "pool.jsonl").open("w") as fh:
        for i, t in enumerate(pool):
            fh.write(json.dumps({"id": f"pool{i:05d}", "text": t}) + "\n")

    (out_dir / "conditions_manifest.json").write_text(
        json.dumps(
            {
                "corpus": corpus_name,
                "seed": seed,
                "n_candidates": n_candidates,
                "n_reference": n_reference,
                "doses": doses,
                "arms": arms,
                "sources": cfg["sources"],
                "conditions": conditions,
            },
            indent=2,
        )
    )
    summary["pool_size"] = len(pool)
    (out_dir / "safety_corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[safety] pool={len(pool)} unique documents ({len(conditions)} conditions) -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
