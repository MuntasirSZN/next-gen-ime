# upodesh/ — typo-tolerant suggestion experiment (this repo)

This folder is a self-contained copy of the `upodesh` suggestion crate,
pushed here per maintainer request so the typo-tolerance work can iterate
alongside the GRU next-word model in the rest of this repo.

## What it is

`upodesh` takes **already-converted Bangla** and returns ranked,
typo-tolerant suggestions. Conversion (Avro/Khipro → Bangla) stays in
`riti`/`okkhor`; this crate never autocorrects — the correct spelling is
always among the candidates.

- Grapheme trie + bounded weighted Damerau/OSA search (insertion,
  deletion, substitution, transposition, Bangla confusions like শ/ষ/স,
  repeated-char discount).
- Frequency prior from IndicCorp V2 (CC-0) — see `data/ATTRIBUTION.md`
  and `data/frequency.LICENSE` for provenance.
- Optional bigram/trigram context scoring (`with_language_model`,
  `with_trigram_language_model`).
- Dual-input `suggest_input`: Bangla typo path + Roman (Avro/Khipro)
  byte tries via `dhoni` (git dep, Khipro mode tracked in
  OpenBangla/dhoni#1).
- Build-time `UPDSHIX3` index blob (~14 MB), `simd-normalizer` (Unicode 17).

## How it relates to the GRU model here

The GRU model in the repo root predicts the *next word* given context.
`upodesh/` fixes *typos in the current word* and completes prefixes.
Complementary layers: upodesh candidates can be re-ranked by the neural
model later; neither replaces the other.

## Run

```sh
cd upodesh
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo run --release --example eval -- data/eval/banglasec-test.csv.gz
cargo run --release --example percentiles
```

Baselines: BanglaSEC Top-1 18.4% / Top-5 23.2% (177,919 pairs);
typo suggest ~200 µs, prefix ~5 µs (GBoard bar is <5 ms).

Submodules `data/openbangla-keyboard` and `data/khipro-m17n` are pinned
in this repo at `upodesh/data/…`.
