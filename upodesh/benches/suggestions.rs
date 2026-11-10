use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use upodesh::corrector;

fn suggestions(c: &mut Criterion) {
    let corrector = corrector();
    // Pay the one-time index build before timing anything.
    assert!(!corrector.suggest("বাংলা").is_empty());

    for (label, input) in [
        ("exact বাংলা", "বাংলা"),
        ("typo বাংরা", "বাংরা"),
        ("short আমি", "আমি"),
        ("long কম্পিউটার", "কম্পিউটার"),
    ] {
        c.bench_function(label, |b| {
            b.iter(|| corrector.suggest_with(black_box(input), 10))
        });
    }

    c.bench_function("prefix কম্পি", |b| {
        b.iter(|| corrector.complete_prefix(black_box("কম্পি"), 10))
    });
}

criterion_group!(benches, suggestions);
criterion_main!(benches);
