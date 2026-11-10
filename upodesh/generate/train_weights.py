#!/usr/bin/env python3
"""Learn per-class error weights from synthetic typo pairs.

Reads a pairs TSV produced by ``generate/gen_errors.py`` and reports the
empirical class distribution as ``data/weights.json`` with full provenance
(corpus checksum, pair count, seeds). The output is a config artifact: it
does not change the library by itself. A future phase can wire these ratios
into `EditCosts` / `RomanCosts` defaults; until then the hand-tuned defaults
remain authoritative.

Provenance: base corpus IndicCorp V2 (CC-0) + synthetic operators derived
from the vendored IPA table. Derived weights are CC-0-clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "weights.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="pairs TSV from gen_errors.py")
    parser.add_argument(
        "--corpus", required=True, help="corpus file the pairs were sampled from"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    corpus_path = Path(args.corpus)
    counts: Counter[str] = Counter()
    total = 0
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            counts[parts[2]] += 1
            total += 1
    if total == 0:
        raise SystemExit(f"no pairs found in {pairs_path}")

    weights = {name: count / total for name, count in sorted(counts.items())}
    payload = {
        "generated_by": "generate/train_weights.py",
        "generated_from": {
            "corpus": str(corpus_path.name),
            "corpus_sha256": sha256(corpus_path),
            "pairs": str(pairs_path),
            "pairs_sha256": sha256(pairs_path),
            "pairs_total": total,
            "seed": args.seed,
            "license": "CC-0 (IndicCorp V2 base + synthetic operators)",
        },
        "class_weights": weights,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT}")
    print("class weights:", weights)


if __name__ == "__main__":
    main()
