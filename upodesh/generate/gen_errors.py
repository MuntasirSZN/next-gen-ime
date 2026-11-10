#!/usr/bin/env python3
"""Generate synthetic (typo -> correct) Bengali pairs from a CC-0 corpus.

License-clean replacement for DPCSpell (CC BY-NC-SA) in the calibration /
evaluation pipeline: base text is IndicCorp V2 (CC-0), error operators are
derived from the vendored IPA alignment table (``data/ipa-alignments.md``)
plus keyboard-shaped doubling — no copied error-model text anywhere.

Usage:
    python3 generate/gen_errors.py --corpus /tmp/opencode/bn.txt \
        --out generate/out/pairs.tsv --pairs 40000 --seed 42

Output: TSV with columns (typo, correct, class). Rows whose typo equals the
correct word are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DICTIONARY = ROOT / "data/openbangla-keyboard/data/dictionary.json"
FREQUENCY = ROOT / "data/frequency.json"
ALIGNMENTS = ROOT / "data/ipa-alignments.md"

# Error-operator classes follow the taxonomy of habibsifat's Bengali error
# generator (reference-only: that repo carries no license, so its table
# contents are not reproduced here). The adjacency data below is derived
# from QWERTY keyboard geometry + the vendored IPA table instead.
QWERTY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

# Bengali grapheme -> canonical roman (dhoni mapping). Keys below are
# factual keyboard adjacencies.
ROMAN_OF: dict[str, str] = {
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "c",
    "ছ": "ch",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "ng",
    "ট": "t",
    "ঠ": "th",
    "ড": "d",
    "ঢ": "dh",
    "ণ": "n",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "f",
    "ব": "b",
    "ভ": "bh",
    "ম": "m",
    "য": "z",
    "র": "r",
    "ল": "l",
    "শ": "s",
    "ষ": "x",
    "স": "s",
    "হ": "h",
    "ড়": "r",
    "ঢ়": "r",
    "য়": "y",
    "ৎ": "t",
    "ং": "ng",
    "ঃ": "",
    "ঁ": "",
}
# Reverse: roman digraphs that start at a key -> candidate graphemes for neighbor collisions.
DIGRAPHS: dict[str, list[str]] = defaultdict(list)
for g, r in ROMAN_OF.items():
    if r:
        DIGRAPHS[r[0]].append(g)


def qwerty_neighbors(key: str) -> list[str]:
    """Letters adjacent to `key` on QWERTY rows (same row +/-1, plus the
    nearest keys on the rows directly above and below)."""
    out: set[str] = set()
    for r, row in enumerate(QWERTY_ROWS):
        if key not in row:
            continue
        pos = row.index(key)
        for c in (pos - 1, pos + 1):
            if 0 <= c < len(row):
                out.add(row[c])
        for dr in (-1, 1):
            rr = r + dr
            if 0 <= rr < len(QWERTY_ROWS):
                other = QWERTY_ROWS[rr]
                for c in (pos - 1, pos, pos + 1):
                    if 0 <= c < len(other) and other[c] != key:
                        out.add(other[c])
    out.discard(key)
    return sorted(out)


PUNCT = ",.;:!?()[]{}\"'-\u0964\u0965\u0967\u0970"

# Bengali dependent signs that attach to the preceding base when splitting
# into coarse clusters (enough for typo simulation fidelity).
MARKS = "\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c5\u09c6\u09c7\u09c8\u09c9\u09ca\u09cb\u09cc\u09fa\u09fc\u09bc\u09d1\u0981\u0982\u0983\u09a4"


def load_dictionary_words() -> set[str]:
    buckets = json.loads(DICTIONARY.read_text(encoding="utf-8"))
    return {
        unicodedata.normalize("NFC", w.strip())
        for words in buckets.values()
        for w in words
        if w.strip()
    }


def load_ipa_classes() -> dict[str, list[str]]:
    """Parse the vendored IPA table's grapheme-group column.

    Rows look like ``| ই, ি | i |``; members share an IPA class and are
    sampled for phonetic-confusion substitution.
    """
    classes: dict[str, list[str]] = defaultdict(list)
    for line in ALIGNMENTS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "---" in line or "Grapheme" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2 or not cells[1] or cells[1].startswith("("):
            continue
        members = [m.strip() for m in cells[0].split(",")]
        if len(members) >= 2:
            for a in members:
                for b in members:
                    if a != b:
                        classes[a].append(b)
    return dict(classes)


def tokenize(sentence: str) -> list[str]:
    out = []
    for tok in unicodedata.normalize("NFC", sentence).split():
        tok = tok.strip(PUNCT)
        if tok and any("\u0980" <= ch <= "\u09ff" for ch in tok):
            out.append(tok)
    return out


def graphemes(word: str) -> list[str]:
    """Coarse clusters: base + following dependent marks/virama stay together."""
    out: list[str] = []
    current: list[str] = []
    for ch in word:
        if current and (ch == "\u09cd" or ch in MARKS):
            current.append(ch)
        else:
            if current:
                out.append("".join(current))
            current = [ch]
    if current:
        out.append("".join(current))
    return out


def op_transpose(word: str, rng: random.Random) -> str | None:
    g = graphemes(word)
    if len(g) < 2:
        return None
    i = rng.randrange(len(g) - 1)
    g[i], g[i + 1] = g[i + 1], g[i]
    return "".join(g)


def op_delete(word: str, rng: random.Random) -> str | None:
    g = graphemes(word)
    if len(g) < 2:
        return None
    del g[rng.randrange(len(g))]
    return "".join(g)


def op_duplicate(word: str, rng: random.Random) -> str | None:
    """Doubled grapheme — the classic 'banngla' slip."""
    g = graphemes(word)
    if not g:
        return None
    i = rng.randrange(len(g))
    g.insert(i, g[i])
    return "".join(g)


def op_phonetic(
    word: str, rng: random.Random, classes: dict[str, list[str]]
) -> str | None:
    """Same-IPA-class cluster swap (ই/ঈ, শ/ষ, ন/ণ ...)."""
    g = graphemes(word)
    candidates = [(i, classes[c]) for i, c in enumerate(g) if c in classes]
    if not candidates:
        return None
    i, alternatives = rng.choice(candidates)
    g[i] = rng.choice(alternatives)
    return "".join(g)


def op_mispress(word: str, rng: random.Random) -> str | None:
    """Mispress substitution: replace one grapheme with the grapheme whose
    canonical roman shares a QWERTY-adjacent key (habibsifat class 1)."""
    g = graphemes(word)
    candidates = [i for i, c in enumerate(g) if c in ROMAN_OF and ROMAN_OF[c]]
    if not candidates:
        return None
    i = rng.choice(candidates)
    key = ROMAN_OF[g[i]][0]
    neighbours = qwerty_neighbors(key[0])
    pool = [
        cand
        for neighbor in neighbours
        for cand in DIGRAPHS.get(neighbor, [])
        if cand != g[i]
    ]
    if not pool:
        return None
    g[i] = rng.choice(pool)
    return "".join(g)


def op_mispress_insert(word: str, rng: random.Random) -> str | None:
    """Mispress insertion: slip an adjacent erroneous grapheme after one
    already typed (habibsifat class 2)."""
    g = graphemes(word)
    candidates = [i for i, c in enumerate(g) if c in ROMAN_OF and ROMAN_OF[c]]
    if not candidates:
        return None
    i = rng.choice(candidates)
    key = ROMAN_OF[g[i]][0]
    pool = [
        cand
        for neighbor in qwerty_neighbors(key[0])
        for cand in DIGRAPHS.get(neighbor, [])
    ]
    if not pool:
        return None
    g.insert(i + 1, rng.choice(pool))
    return "".join(g)


OPERATORS = [
    ("transpose", op_transpose),
    ("delete", op_delete),
    ("duplicate", op_duplicate),
    ("phonetic", op_phonetic),
    ("mispress", op_mispress),
    ("mispress-insert", op_mispress_insert),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", required=True, help="one sentence per line, e.g. bn.txt"
    )
    parser.add_argument("--out", required=True, help="output TSV path")
    parser.add_argument("--pairs", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    words = load_dictionary_words()
    classes = load_ipa_classes()
    try:
        freq = json.loads(FREQUENCY.read_text(encoding="utf-8"))
    except FileNotFoundError:
        freq = {}

    counts: Counter[str] = Counter()
    with open(args.corpus, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            for tok in tokenize(line):
                if tok in words:
                    counts[tok] += 1
            if sum(counts.values()) > args.pairs * 10:  # bound runtime/memory
                break

    pool = list(counts)
    weights = [math.sqrt(freq.get(w, 1)) for w in pool]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        while written < args.pairs:
            word = rng.choices(pool, weights=weights, k=1)[0]
            name, fn = rng.choice(OPERATORS)
            typo = (
                fn(word, rng, classes)
                if name == "phonetic"
                else fn(word, rng)
                if name in ("transpose", "delete", "duplicate")
                else fn(word, rng)
            )
            if not typo or typo == word:
                continue
            out.write(f"{typo}\t{word}\t{name}\n")
            class_counts[name] += 1
            written += 1
    print(f"wrote {written} pairs -> {out_path} (seed={args.seed})")
    print("classes:", dict(class_counts))


if __name__ == "__main__":
    main()
