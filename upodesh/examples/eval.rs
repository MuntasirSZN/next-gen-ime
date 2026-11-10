//! Offline evaluation against typo corpora.
//!
//! Two pair sources, two formats:
//! - **Synthetic TSV** (typo, correct, class) from `generate/gen_errors.py`
//!   (IndicCorp CC-0 base + our operators; reproducible from corpus + seed).
//! - **BanglaSEC CSV** (Word, Error, ErrorType, ...) — the HuggingFace
//!   `mehedihasanbijoy/BanglaSEC` test split (MIT, official DPCSpell
//!   corpus). Committed gzipped at `data/eval/banglasec-test.csv.gz`;
//!   only Word/Error/ErrorType are read. Eval-only, never used for training.
//!
//! Usage:
//!   cargo run --release --example eval -- data/eval/banglasec-test.csv.gz

use std::collections::HashMap;
use std::process::Command;

use upodesh::corrector;

struct Pair {
    typo: String,
    want: String,
    class: String,
}

fn load(path: &str) -> Vec<Pair> {
    let text = if path.ends_with(".gz") {
        let output = Command::new("gzip")
            .args(["-dc", path])
            .output()
            .expect("run gzip -dc");
        assert!(output.status.success(), "gzip failed on {path}");
        String::from_utf8_lossy(&output.stdout).into_owned()
    } else {
        std::fs::read_to_string(path).unwrap_or_else(|_| panic!("pairs file not found: {path}"))
    };
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    let mut pairs = Vec::new();
    if first.starts_with("Word,") {
        // BanglaSEC CSV: first three fields never contain commas.
        for line in lines {
            let Some(i) = line.find(',') else { continue };
            let Some(j) = line[i + 1..].find(',').map(|o| i + 1 + o) else {
                continue;
            };
            let Some(k) = line[j + 1..].find(',').map(|o| j + 1 + o) else {
                continue;
            };
            pairs.push(Pair {
                typo: line[i + 1..j].to_string(),
                want: line[..i].to_string(),
                class: line[j + 1..k].to_string(),
            });
        }
    } else {
        for line in std::iter::once(first).chain(lines) {
            let mut cols = line.split('\t');
            let (Some(typo), Some(want), Some(class)) = (cols.next(), cols.next(), cols.next())
            else {
                continue;
            };
            pairs.push(Pair {
                typo: typo.to_string(),
                want: want.to_string(),
                class: class.to_string(),
            });
        }
    }
    pairs
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "generate/out/pairs.tsv".into());
    let pairs = load(&path);
    let corrector = corrector();

    let (mut top1, mut top5, mut total) = (0usize, 0usize, 0usize);
    let mut by_class: HashMap<&str, (usize, usize)> = HashMap::new();

    for pair in &pairs {
        let suggestions = corrector.suggest_with(&pair.typo, 5);
        total += 1;
        let rank = suggestions.iter().position(|s| s == &pair.want);
        top1 += usize::from(rank == Some(0));
        top5 += usize::from(rank.is_some());
        let entry = by_class.entry(pair.class.as_str()).or_default();
        entry.0 += usize::from(rank == Some(0));
        entry.1 += 1;
    }

    let pct = |n: usize| 100.0 * n as f64 / total.max(1) as f64;
    println!(
        "pairs={total} Top-1={:.1}% Top-5={:.1}% Recall@5={:.1}%",
        pct(top1),
        pct(top5),
        pct(top5)
    );
    let mut classes: Vec<_> = by_class.iter().collect();
    classes.sort_by_key(|(_, count)| std::cmp::Reverse(count.1));
    for (class, (hits, count)) in classes {
        println!("  {class}: Top-1={:.1}% ({hits}/{count})", pct(*hits));
    }
}
