//! Streaming word counter for corpus frequency tables.
//!
//! Rebuilds the frequency table from a large CC-0 corpus (e.g. IndicCorp V2
//! `bn.txt`, ~800 MB) — a single pass, O(1) memory beyond the dictionary
//! itself. Python does this in minutes; this does it in seconds.
//!
//! Usage:
//!   cargo run --release --manifest-path generate/Cargo.toml \
//!       --bin corpus-count -- --corpus /tmp/opencode/bn.txt \
//!       --out data/frequency.json

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use simd_normalizer::UnicodeNormalization;

const PUNCT: &[char] = &[
    ',', '.', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '"', '\'', '-', '।', '॥', 'ך', '։',
];

fn main() {
    let mut args = env::args().skip(1);
    let mut corpus = PathBuf::new();
    let mut out = PathBuf::new();
    let mut dictionary = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"))
        .join("../data/openbangla-keyboard/data/dictionary.json");
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--corpus" => corpus = PathBuf::from(args.next().expect("--corpus value")),
            "--out" => out = PathBuf::from(args.next().expect("--out value")),
            "--dictionary" => dictionary = PathBuf::from(args.next().expect("--dictionary value")),
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(!corpus.as_os_str().is_empty(), "--corpus is required");
    assert!(!out.as_os_str().is_empty(), "--out is required");

    // Dictionary words (NFC, trimmed), intersected with corpus tokens.
    let dict_text = std::fs::read_to_string(&dictionary).expect("read dictionary");
    let root: serde_json::Value = serde_json::from_str(&dict_text).expect("dictionary JSON");
    let mut words = HashSet::new();
    if let Some(buckets) = root.as_object() {
        for bucket in buckets.values() {
            for word in bucket
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|w| w.as_str())
            {
                let word: String = word.trim().nfc().into_owned();
                if !word.is_empty() {
                    words.insert(word);
                }
            }
        }
    }

    let mut reader = BufReader::new(File::open(&corpus).expect("open corpus"));
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    let (mut tokens, mut kept) = (0u64, 0u64);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf).expect("read line");
        if n == 0 {
            break;
        }
        let line = String::from_utf8_lossy(&buf);
        let normalized: String = line.nfc().into_owned();
        for token in normalized.split_whitespace() {
            let token = token.trim_matches(PUNCT);
            if token.is_empty() {
                continue;
            }
            tokens += 1;
            if words.contains(token) {
                *counts.entry(token.to_string()).or_default() += 1;
                kept += 1;
            }
        }
    }

    let mut json = Vec::with_capacity(counts.len() * 24);
    json.extend_from_slice(b"{");
    for (i, (word, count)) in counts.iter().enumerate() {
        if i > 0 {
            json.extend_from_slice(b",");
        }
        json.extend_from_slice(serde_json::to_string(word).expect("serialize").as_bytes());
        json.push(b':');
        json.extend_from_slice(count.to_string().as_bytes());
    }
    json.extend_from_slice(b"}\n");
    let mut file = File::create(&out).expect("create output");
    file.write_all(&json).expect("write output");

    println!(
        "tokens={tokens} kept={kept} covered={}/{} ({:.1}%) bytes={}",
        counts.len(),
        words.len(),
        100.0 * counts.len() as f64 / words.len() as f64,
        json.len(),
    );
}
