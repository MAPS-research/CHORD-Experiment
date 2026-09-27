"""Fluency-preserving factual edits used to build the QA-faithfulness conditions.

Operators follow FactCC (Kryscinski et al., 2020): number swap, entity swap and
sentence negation, each changing exactly one fact so the edited statement stays
fluent and coherent on its own and is wrong only with respect to the source.
``benign_reword`` is a deterministic fact-preserving synonym rewrite (the
``benign_reword`` control, also used as the evaluation set's clean candidates).
All randomness goes through the caller's ``random.Random``; the build scripts
consume it in a fixed order, so the conditions are reproducible from ``SEED``.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

from chord.text import sentences

SEED = 20260621

NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

AUX_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|can|could|will|would|should|does|do|did)\b",
    flags=re.IGNORECASE,
)

# capitalized tokens that are not really named entities
ENTITY_STOP = {
    "The",
    "This",
    "That",
    "There",
    "These",
    "Those",
    "And",
    "But",
    "For",
    "Nor",
    "When",
    "While",
    "After",
    "Before",
    "However",
    "Although",
    "Though",
    "Because",
    "Since",
    "Their",
    "They",
    "His",
    "Her",
    "Its",
    "Our",
    "Your",
    "She",
    "Here",
    "What",
    "Where",
    "Which",
    "While",
    "With",
    "From",
    "Into",
    "Over",
    "Under",
    "Also",
    "Then",
    "Than",
    "Such",
    "Some",
    "Many",
    "Most",
    "More",
    "One",
    "Two",
    "First",
    "Second",
    "Third",
    "Last",
    "Next",
    "Now",
    "Today",
    "Yesterday",
}

SYNONYMS = {  # fact-preserving surface rewrites (never touch numbers/entities); only
    # context-safe, meaning-preserving pairs (risky polysemes excluded)
    "said": "stated",
    "says": "states",
    "big": "large",
    "small": "little",
    "important": "significant",
    "however": "nevertheless",
    "also": "additionally",
    "many": "numerous",
    "showed": "demonstrated",
    "shows": "demonstrates",
    "made": "produced",
    "helps": "assists",
    "began": "commenced",
    "started": "commenced",
    "found": "discovered",
    "told": "informed",
    "huge": "enormous",
    "quickly": "rapidly",
    "often": "frequently",
    "good": "favorable",
    "difficult": "challenging",
    "main": "primary",
    "purchased": "acquired",
    "additional": "further",
    "rapidly": "swiftly",
    "significant": "considerable",
    "numerous": "various",
}

# faithful re-frames used when no synonym fires (varied so benign is not a uniform prefix)
BENIGN_FRAMES = [
    "According to the passage, {s}",
    "The passage states that {s}",
    "As the source notes, {s}",
    "Per the text, {s}",
]


def clean_claim(s: str) -> str:
    """Strip leading footnote/line-number markers ('11However', '2 For') and '(...)'
    truncation artifacts so the swappable fact is a real claim, not a doc artifact."""
    s = re.sub(r"^\s*\d+\s*", "", s)
    s = re.sub(r"\s*\(\.\.\.\)\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _preserve_case(before: str, after: str) -> str:
    if before.isupper():
        return after.upper()
    if before[:1].isupper():
        return after[:1].upper() + after[1:]
    return after


def cap_source(text: str, max_chars: int) -> str:
    """Trim the source to <= max_chars at a sentence boundary (token budget headroom)."""
    if len(text) <= max_chars:
        return text.strip()
    sents = sentences(text)
    out = ""
    for s in sents:
        if len(out) + len(s) + 1 > max_chars:
            break
        out += (" " if out else "") + s.strip()
    return (out or text[:max_chars]).strip()


def entities_in(text: str) -> List[str]:
    """MID-sentence capitalized tokens only (skip sentence-initial words, which are
    capitalized by position, not because they are named entities). This keeps entity
    swaps FLUENT and on-type instead of mangling leading adverbs like 'Ultimately'."""
    seen, out = set(), []
    for sent in sentences(text):
        if not sent.strip():
            continue
        first = len(sent) - len(sent.lstrip())
        for m in CAP_RE.finditer(sent):
            tok = m.group()
            if m.start() == first or tok in ENTITY_STOP or tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
    return out


def swap_number(claim: str, rng: random.Random) -> Optional[str]:
    ms = list(NUM_RE.finditer(claim))
    if not ms:
        return None
    m = ms[rng.randrange(len(ms))]
    raw = m.group()
    digits = raw.replace(",", "")
    try:
        if "." in digits:
            val = float(digits)
            new = val * (rng.choice([0.5, 1.7, 2.3, 0.3]))
            new_s = f"{new:.2f}".rstrip("0").rstrip(".")
        else:
            val = int(digits)
            if len(digits) == 4 and 1500 <= val <= 2100:  # looks like a year
                new = val + rng.choice([-13, -7, 9, 17, 23])
            else:
                new = val + max(1, round(val * 0.4)) * rng.choice([-1, 1])
                if new == val:
                    new = val + 1
            new_s = str(int(new))
    except ValueError:
        return None
    if new_s == raw:
        return None
    return claim[: m.start()] + new_s + claim[m.end() :]


def swap_entity(claim: str, source: str, rng: random.Random) -> Optional[str]:
    """Transpose TWO mid-sentence entities WITHIN the claim (role reversal). Same-
    context => type-matched + fluent (e.g. 'Carson ... endorsed Trump' -> 'Trump ...
    endorsed Carson'), a clean contradiction-to-source, unlike swapping in a random
    source token. Requires >=2 distinct entities in the claim."""
    uniq: List[str] = []
    for e in entities_in(claim):
        if e not in uniq:
            uniq.append(e)
    if len(uniq) < 2:
        return None
    a, b = rng.sample(uniq, 2)
    out = re.sub(r"\b" + re.escape(a) + r"\b", "\x00", claim, count=1)
    out = re.sub(r"\b" + re.escape(b) + r"\b", a, out, count=1)
    out = out.replace("\x00", b, 1)
    return out if out != claim else None


def benign_frame(claim: str, rng: random.Random) -> str:
    frame = BENIGN_FRAMES[rng.randrange(len(BENIGN_FRAMES))]
    return frame.format(s=claim[:1].lower() + claim[1:])


def negate(claim: str, rng: random.Random) -> Optional[str]:
    """Insert a clean auxiliary negation (FactCC 'Sentence Negation'). Skip claims
    that already contain a negation (avoids confusing double negatives) and claims
    with no clean auxiliary verb (keeps every contradiction FLUENT)."""
    if (
        re.search(r"\b(not|never|no|none|cannot|nobody|nothing)\b", claim, re.I)
        or "n't" in claim.lower()
    ):
        return None
    m = AUX_RE.search(claim)
    if not m:
        return None
    if m.group().lower() == "can":
        return claim[: m.start()] + "cannot" + claim[m.end() :]
    return claim[: m.end()] + " not" + claim[m.end() :]


def benign_reword(claim: str, rng: random.Random) -> str:
    def repl(mo):
        w = mo.group()
        low = w.lower()
        return _preserve_case(w, SYNONYMS[low]) if low in SYNONYMS else w

    out = re.sub(r"\b[A-Za-z]+\b", repl, claim)
    if out == claim:  # no synonym fired -> a varied faithful re-frame
        out = benign_frame(claim, rng)
    return out
