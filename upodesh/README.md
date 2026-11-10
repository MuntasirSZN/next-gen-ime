# upodesh

`upodesh` (In Bangla, `উপদেশ` is a synonym of the word `পরামর্শ` which means
`suggestion`) is a Bangla word suggestion library.

It takes **already-converted Bangla text** and returns ranked, typo-tolerant
suggestions. Phonetic conversion (Avro, Khipro, …) deliberately lives in
`riti`/`okkhor`, not here.

## How it works

The index is a single build-time grapheme trie (UPDSHIX3 blob) with
frequency-bearing terminals. The same index answers both operations:

- **Typo correction** — a bounded weighted Damerau/optimal-string-alignment
  DP over the trie. It supports insertion, deletion, substitution, adjacent
  transposition, Bangla-specific confusions (for example শ/ষ/স, ন/ণ, ই/ঈ, ি/ী),
  and a repeated-char discount on doublings (`banngla` ~ `bangla`).
- **Prefix completion** — a second automaton over the same keys. The final
  query grapheme may be incomplete, so typing `ক` still matches words beginning
  with `কা`.
- **Dual-input typo matching** — `suggest_input` also scores the original
  Roman input against per-mode (Avro/Khipro) byte tries, merging both
  signals with calibrated weights. Suggest needs no input mode at all:
  Bangla-only typo tolerance is its hot path.

Input is NFC-normalized, and grapheme IDs are stored as big-endian `u16`s so
edits never split a Bangla consonant from its dependent vowel sign. Ranking uses
weighted edit cost plus the dictionary frequency prior; optional Kneser–Ney
bigram and trigram models can add context scoring, and a lemma/POS table can
gate morphological suffix rules.

The upstream OpenBangla-Keyboard dictionary is imported as a git submodule at
`data/openbangla-keyboard`. Only its `data/dictionary.json` is used.

### Frequency prior

`upodesh` ships `data/frequency.json`, derived from **IndicCorp V2**
(`ai4bharat/IndicCorpV2`, CC-0). It covers 99,422 / 159,384 dictionary
words (62.4%) from 922M tokens and is loaded at build time. See
`data/frequency.LICENSE` for provenance, corpus checksum, and citation;
regenerate it with the Rust streaming counter:

```sh
cargo run --release --manifest-path generate/Cargo.toml --bin corpus-count -- \
    --corpus /path/to/bn.txt --out data/frequency.json
```

### Offline eval

`cargo run --release --example eval -- data/eval/banglasec-test.csv.gz`
reports Top-1/Top-5/Recall@5 of the Bangla-side suggester against the
BanglaSEC test split (MIT, official DPCSpell splits). Current baseline:
Top-1 18.4%, Top-5 23.2% (177,919 pairs). The synthetic generator
(`python3 generate/gen_errors.py`) produces a compatible pair set from
the same CC-0 corpus for calibration runs. For latency numbers, see
`cargo run --release --example percentiles`.

## Usage

```rust
use upodesh::{corrector, CorrectionConfig};

let corrector = corrector();

// Typo-tolerant suggestions.
let suggestions = corrector.suggest("বাংরা");

// Prefix completion over the same index.
let completions = corrector.complete_prefix("বাংলা", 10);

// Ranked candidates with edit cost and frequency.
for candidate in corrector.correct_with_context("বাং", &["আমি"], 10) {
    println!("{} (cost {}, frequency {})", candidate.word, candidate.edit_cost, candidate.frequency);
}
```

`corrector()` returns a process-wide index built once on first use. Building a
custom index is possible with `Corrector::from_entries`, `with_edit_costs`,
`with_config`, `with_language_model` and `with_trigram_language_model`.

> **Frequency data:** corpus counts come from IndicCorp V2 (CC-0,
> see `data/frequency.LICENSE`). Words absent from that corpus keep
> frequency `1`, so rare dictionary entries rank below attested words. Supplying
> your own frequencies through `WordEntry` is also supported.

For performance numbers, see `examples/percentiles.rs` (p50/p95/p99 against
the 5–15 µs readiness target).

## Development

```sh
git submodule update --init --recursive
cargo test
cargo clippy --all-targets --all-features
cargo bench --bench suggestions
```

`cargo run --manifest-path generate/Cargo.toml` reports the size and shape of
the embedded dictionary.

## Acknowledgement

- [Mehdi Hasan Khan](https://github.com/mugli) and
  [Tahmid Sadik](https://github.com/tahmidsadik/) for their
  [`libavrophonetic`](https://github.com/mugli/libavrophonetic/) project.
- [Andrew Gallant](https://github.com/BurntSushi) for the
  [`fst`](https://github.com/BurntSushi/fst) crate and
  [Index 1,600,000,000 Keys with Automata and Rust](https://burntsushi.net/transducers/).
