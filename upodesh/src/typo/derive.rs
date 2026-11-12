// Build-time dictionary derivations shared verbatim by `build.rs` and the
// in-memory index builder.
//
// Upstream dictionaries list lemmas but not every inflected form typists
// expect. The only derivation here is the genitive `-র` for noun lemmas
// ending in `ী` (পৃথিবী → পৃথিবীর): it is regular, and the lemma table gives
// the POS needed to keep the rule off verbs and adjectives.

/// Words to add to the index, with a frequency prior.
///
/// `entries` is the current word → frequency map; `lemma` maps a lemma to its
/// coarse POS; `frequencies` is the upstream frequency table (used for the
/// derived form when it is already attested there). Derived forms never
/// replace an existing entry.
pub fn genitive_forms(
    entries: &std::collections::BTreeMap<String, u64>,
    lemma: &std::collections::BTreeMap<String, String>,
    frequencies: &std::collections::BTreeMap<String, u64>,
) -> Vec<(String, u64)> {
    entries
        .iter()
        .filter(|(word, _)| {
            word.ends_with('ী') && lemma.get(word.as_str()).is_some_and(|pos| pos == "noun")
        })
        .filter_map(|(word, &frequency)| {
            let derived = format!("{word}র");
            if entries.contains_key(&derived) {
                return None;
            }
            let derived_frequency = frequencies
                .get(&derived)
                .copied()
                .unwrap_or(frequency / 2)
                .max(1);
            Some((derived, derived_frequency))
        })
        .collect()
}
