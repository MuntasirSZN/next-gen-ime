//! Varied-input benchmark matrix: short/medium/long word sizes across
//! Bangla (typo-tolerant), prefix and membership paths, plus Roman-path
//! probes once the dual-input API lands. Inputs are derived from the
//! embedded dictionary so they stay representative of real usage.
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use simd_normalizer::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;
use upodesh::corrector;

fn word_at_len(words: &[String], target: usize) -> String {
    words
        .iter()
        .find(|w| UnicodeSegmentation::graphemes(w.as_str(), true).count() == target)
        .cloned()
        .expect("dictionary has a word of each length")
}

fn varied(c: &mut Criterion) {
    let corrector = corrector();
    // Representative words drawn from the shipped dictionary, picked by
    // grapheme count so short/medium/long tiers track real lengths.
    let root: serde_json::Value = serde_json::from_str(include_str!(
        "../data/openbangla-keyboard/data/dictionary.json"
    ))
    .expect("embedded dictionary is valid JSON");
    let mut words: Vec<String> = root
        .as_object()
        .expect("dictionary root is an object")
        .values()
        .flat_map(|v| v.as_array().into_iter().flatten())
        .filter_map(|v| v.as_str())
        .map(|w| {
            let n: String = w.nfc().into_owned();
            n
        })
        .collect();
    words.sort();
    words.dedup();

    let cases = [
        ("short1exact", word_at_len(&words, 1)),
        ("short2exact", word_at_len(&words, 2)),
        ("med4exact", word_at_len(&words, 4)),
        ("med5exact", word_at_len(&words, 5)),
        ("long8exact", word_at_len(&words, 8)),
        ("long10exact", word_at_len(&words, 10)),
    ];
    for (label, input) in cases {
        assert!(
            corrector.contains(&input),
            "{label} input {input} must be in dictionary"
        );
        c.bench_function(&format!("suggest {label}"), |b| {
            b.iter(|| corrector.suggest_with(black_box(&input), 10))
        });
    }

    // Typo probes on a medium word: substitute the last grapheme, delete
    // one, transpose the last two, and prepend a random valid grapheme.
    let medium = word_at_len(&words, 5);
    let gs: Vec<&str> = UnicodeSegmentation::graphemes(medium.as_str(), true).collect();
    let joined = |parts: Vec<&str>| parts.join("");
    let mut subbed = gs.clone();
    subbed[4] = "x";
    let deleted = joined(gs[..4].to_vec());
    let mut transposed = gs.clone();
    transposed.swap(3, 4);
    let transposed = joined(transposed);
    for (label, input) in [
        ("typo sub", joined(subbed)),
        ("typo del", deleted),
        ("typo trans", transposed),
    ] {
        c.bench_function(&format!("suggest {label} med1"), |b| {
            b.iter(|| corrector.suggest_with(black_box(&input), 10))
        });
    }

    let prefix = UnicodeSegmentation::graphemes(word_at_len(&words, 8).as_str(), true)
        .take(3)
        .collect::<Vec<_>>()
        .join("");
    c.bench_function("prefix narrow widelite", |b| {
        b.iter(|| corrector.complete_prefix(black_box(&prefix), 10))
    });
    c.bench_function("prefix wide first grapheme", |b| {
        b.iter(|| corrector.complete_prefix(black_box("ক"), 10))
    });
    c.bench_function("contains hit", |b| {
        b.iter(|| corrector.contains(black_box(&medium)))
    });
    c.bench_function("contains miss", |b| {
        b.iter(|| corrector.contains(black_box("অজানাশব্দ")))
    });
}

criterion_group!(benches, varied);
criterion_main!(benches);
