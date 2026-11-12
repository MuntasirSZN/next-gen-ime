//! Latency distribution percentiles (p50/p95/p99) for suggestion paths.
//!
//! Reports per-query latency distributions for the pin CPU this binary runs
//! on. Reference bar: GBoard targets <5 ms per keystroke; the typo path runs
//! ~20× under that bar, and the exact/prefix paths ~1000× under it.
//! Candidate counts show how much of the time goes into result collection vs
//! traversal; allocations are not tracked (instrumentation would change the
//! APIs it measures).
//!
//! Usage:
//!   cargo run --release --example percentiles

use std::hint::black_box;
use std::time::{Duration, Instant};
use peak_alloc::PeakAlloc;

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

use upodesh::corrector;

fn percentile(mut samples: Vec<Duration>, p: f64) -> Duration {
    samples.sort();
    let idx = ((samples.len() as f64 - 1.0) * p / 100.0) as usize;
    samples[idx.min(samples.len() - 1)]
}

fn measure(name: &str, n: usize, f: impl Fn()) -> (String, Duration, Duration, Duration) {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    (
        name.to_string(),
        percentile(samples.clone(), 50.0),
        percentile(samples.clone(), 95.0),
        percentile(samples, 99.0),
    )
}

fn main() {
    let corrector = corrector();
    let stats = corrector.stats();
    println!(
        "index: {} words, {} graphemes, max_len {}, max_freq {}",
        stats.words, stats.graphemes, stats.max_dict_len, stats.max_frequency
    );

    let n = 5_000;
    let rows = [
        measure("suggest বাংলা", n, || {
            black_box(corrector.suggest_with("বাংলা", 10));
        }),
        measure("suggest বাংরা", n, || {
            black_box(corrector.suggest_with("বাংরা", 10));
        }),
        measure("suggest আমি", n, || {
            black_box(corrector.suggest_with("আমি", 10));
        }),
        measure("prefix কম্পি", n, || {
            black_box(corrector.complete_prefix("কম্পি", 10));
        }),
        measure("suggest_input roman", 2_000, || {
            let query =
                upodesh::SuggestionInput::with_roman("amerika", "আমেরিকা", upodesh::InputMode::Avro);
            black_box(corrector.suggest_input(&query, 10));
        }),
    ];
    for (name, p50, p95, p99) in rows {
        println!("{name}: p50={p50:?} p95={p95:?} p99={p99:?}");
    }
    println!("reference bar: GBoard tracks <5 ms/keystroke; we're µs-range.");

    let current_mem = PEAK_ALLOC.current_usage_as_mb();
	println!("\nThis program currently uses {} MB of RAM.", current_mem);
	let peak_mem = PEAK_ALLOC.peak_usage_as_mb();
	println!("The max amount that was used {} MB of RAM.", peak_mem);
}
