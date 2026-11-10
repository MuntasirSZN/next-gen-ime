//! Offline sanity checks for the embedded OpenBangla-Keyboard dictionary.
//!
//! The correction index is built once, lazily, by the `upodesh` library. This
//! binary exists so `cargo run` can report what the submodule actually ships
//! before a release.

use std::{env, fs, path::PathBuf};

use serde_json::Value;

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let path = root.join("../data/openbangla-keyboard/data/dictionary.json");
    let text = fs::read_to_string(&path).expect("read dictionary.json from the submodule");
    let value: Value = serde_json::from_str(&text).expect("dictionary.json is valid JSON");

    let buckets = value.as_object().expect("dictionary.json is a JSON object");
    let mut words = 0_usize;
    let mut empty_buckets = 0_usize;
    let mut longest = 0_usize;

    for bucket in buckets.values() {
        let Some(bucket) = bucket.as_array() else {
            continue;
        };
        if bucket.is_empty() {
            empty_buckets += 1;
        }
        for word in bucket.iter().filter_map(Value::as_str) {
            words += 1;
            longest = longest.max(word.chars().count());
        }
    }

    println!("dictionary: {}", path.display());
    println!("buckets: {}", buckets.len());
    println!("empty buckets: {empty_buckets}");
    println!("words: {words}");
    println!("longest word (chars): {longest}");
}
