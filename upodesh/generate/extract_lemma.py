#!/usr/bin/env python3
"""Extract lemma/POS metadata from a bnwiktionary dump.

bnwiktionary is CC BY-SA 4.0 (Wikimedia). Only derived facts ship here —
a word->POS table; the dump itself stays offline. Wiktextract does not
support the bn edition, so POS headings are parsed directly: pages carry a
``{{ভাষা|bn}}`` language section containing POS subsections such as
``===বিশেষ্য===`` (noun), ``===বিশেষণ===`` (adjective), ``===ক্রিয়া===``
(verb). Wikilink forms (``=== [[বিশেষ্য]] ===``) are unwrapped.

Usage:
    python3 generate/extract_lemma.py --dump /tmp/opencode/bnwiktionary.xml.bz2 \
        --out data/lemma.json

Licence note: emitted table is a derivative work under CC BY-SA 4.0;
see data/ATTRIBUTION.md for the attribution statement.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import unicodedata
from pathlib import Path

POS_MAP = {
    "বিশেষ্য": "noun",
    "বিশেষণ": "adj",
    "ক্রিয়া": "verb",
    "ক্রিয়াবিশেষণ": "adv",
    "অব্যয়": "indeclinable",
    "সর্বনাম": "pronoun",
    "যোজক": "conjunction",
    "অনুসর্গ": "postposition",
    "উপসর্গ": "prefix",
    "সংখ্যাবাচক": "numeral",
    "নির্দেশক": "determiner",
    "ধ্বনি": "onomatopoeia",
    "নামবাচক বিশেষ্য": "proper_noun",
}

PAGE_RE = re.compile(r"<page>(.*?)</page>", re.S)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
TEXT_RE = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
HEADING_RE = re.compile(r"^={1,6}\s*(.*?)\s*={1,6}\s*$", re.M)
LANG_RE = re.compile(r"\{\{ভাষা\|([a-z]+)\}\}")


def clean_heading(label: str) -> str:
    label = label.strip()
    if label.startswith("[[") and label.endswith("]]"):
        label = label[2:-2]
    return unicodedata.normalize("NFC", label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    digest = hashlib.sha256()
    lemma: dict[str, str] = {}
    unknown_pos: dict[str, int] = {}
    pages = 0

    def handle_page(page: str) -> None:
        nonlocal pages
        title_m = TITLE_RE.search(page)
        text_m = TEXT_RE.search(page)
        if not title_m or not text_m:
            return
        title = unicodedata.normalize("NFC", title_m.group(1).strip())
        if not title or not any("\u0980" <= c <= "\u09ff" for c in title):
            return
        wikitext = text_m.group(1)
        in_bn = False
        for heading in HEADING_RE.finditer(wikitext):
            label = heading.group(1).strip()
            lang = LANG_RE.search(label)
            if lang:
                in_bn = lang.group(1) == "bn"
                continue
            if not in_bn:
                continue
            pos = POS_MAP.get(clean_heading(label))
            if pos:
                lemma.setdefault(title, pos)
            elif label:
                unknown_pos[label] = unknown_pos.get(label, 0) + 1
        pages += 1
        if pages % 50000 == 0:
            print(f"  pages so far: {pages}, lemmas: {len(lemma)}", flush=True)

    digest_text = []
    buffer = ""
    with bz2.open(args.dump, "rt", encoding="utf-8", errors="ignore") as handle:
        while True:
            piece = handle.read(1 << 20)
            if not piece:
                break
            digest.update(piece.encode("utf-8"))
            buffer += piece
            if len(buffer) > 4 << 20:
                cut = buffer.rfind("</page>")
                if cut >= 0:
                    for match in PAGE_RE.finditer(buffer[: cut + len("</page>")]):
                        handle_page(match.group(1))
                    buffer = buffer[cut + len("</page>") :]
    for match in PAGE_RE.finditer(buffer):
        handle_page(match.group(1))

    out = Path(args.out)
    out.write_text(
        json.dumps(lemma, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    top_unknown = sorted(unknown_pos.items(), key=lambda kv: -kv[1])[:8]
    print(f"pages={pages} lemmas={len(lemma)} sha256={digest.hexdigest()}")
    print("unknown POS headings (top):", top_unknown)


if __name__ == "__main__":
    main()
