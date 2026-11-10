use unicode_segmentation::UnicodeSegmentation;

use super::WordEntry;
use super::lemma::{LemmaPos, LemmaTable};
use super::normalize::normalize_bengali;

/// A productive suffix rule for dictionary expansion.
///
/// `stem_ending` is removed from a lemma and `replacement` is appended. An
/// empty `stem_ending` means a plain suffix append. Rules are explicit because
/// Bengali inflection and stem alternation should not be guessed from a short
/// hard-coded suffix list.
///
/// `pos` optionally gates the rule on the lemma's part of speech (see
/// [`LemmaTable`]). When the table has no entry for a lemma, the rule applies
/// anyway — unknown words are never dropped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SuffixRule {
    stem_ending: String,
    replacement: String,
    min_stem_graphemes: usize,
    frequency_divisor: u64,
    pos: Option<LemmaPos>,
}

impl SuffixRule {
    /// Create a suffix rule that appends `suffix` unchanged.
    pub fn append(suffix: impl Into<String>) -> Self {
        Self {
            stem_ending: String::new(),
            replacement: normalize_bengali(&suffix.into()),
            min_stem_graphemes: 2,
            frequency_divisor: 3,
            pos: None,
        }
    }

    /// Replace a known stem ending with `replacement`. This supports simple
    /// orthographic alternations without pretending to be a full analyzer.
    pub fn replace(stem_ending: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            stem_ending: normalize_bengali(&stem_ending.into()),
            replacement: normalize_bengali(&replacement.into()),
            min_stem_graphemes: 2,
            frequency_divisor: 3,
            pos: None,
        }
    }

    pub fn with_min_stem_graphemes(mut self, minimum: usize) -> Self {
        self.min_stem_graphemes = minimum;
        self
    }

    /// Generated forms inherit `lemma_frequency / divisor` (minimum 1).
    /// A larger divisor gives derived forms a weaker lexical prior.
    pub fn with_frequency_divisor(mut self, divisor: u64) -> Self {
        self.frequency_divisor = divisor.max(1);
        self
    }

    /// Gate this rule on the lemma's part of speech from the
    /// [`LemmaTable`]. Lemmas missing from the table are unaffected.
    pub fn with_pos(mut self, pos: LemmaPos) -> Self {
        self.pos = Some(pos);
        self
    }
}

/// Small, explicit suffix transducer used to expand a lemma dictionary into
/// surface forms before indexing. It does not recursively apply rules.
#[derive(Clone, Debug, Default)]
pub struct Morphology {
    suffix_rules: Vec<SuffixRule>,
    lemma_table: Option<LemmaTable>,
}

impl Morphology {
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict POS-gated rules using a Wiktextract-derived lemma table.
    pub fn with_lemma_table(mut self, table: LemmaTable) -> Self {
        self.lemma_table = Some(table);
        self
    }

    pub fn with_rule(mut self, rule: SuffixRule) -> Self {
        self.suffix_rules.push(rule);
        self
    }

    pub fn add_rule(&mut self, rule: SuffixRule) {
        self.suffix_rules.push(rule);
    }

    /// Expand each lemma once with each applicable rule. The original lemma is
    /// retained. Duplicate generated forms are merged by the index builder.
    pub fn expand(&self, entries: impl IntoIterator<Item = WordEntry>) -> Vec<WordEntry> {
        let entries: Vec<WordEntry> = entries
            .into_iter()
            .map(|entry| {
                WordEntry::new(normalize_bengali(entry.word.trim()), entry.frequency.max(1))
            })
            .filter(|entry| !entry.word.is_empty())
            .collect();

        let mut expanded = entries.clone();
        for entry in &entries {
            for rule in &self.suffix_rules {
                if let (Some(required), Some(table)) = (rule.pos, &self.lemma_table)
                    && table
                        .pos(&entry.word)
                        .is_some_and(|actual| actual != required)
                {
                    continue;
                }
                let stem = if rule.stem_ending.is_empty() {
                    Some(entry.word.as_str())
                } else {
                    entry.word.strip_suffix(&rule.stem_ending)
                };
                let Some(stem) = stem else {
                    continue;
                };

                let stem_len = UnicodeSegmentation::graphemes(stem, true).count();
                if stem_len < rule.min_stem_graphemes {
                    continue;
                }

                let mut surface = String::with_capacity(stem.len() + rule.replacement.len());
                surface.push_str(stem);
                surface.push_str(&rule.replacement);
                if surface == entry.word {
                    continue;
                }

                expanded.push(WordEntry::new(
                    surface,
                    (entry.frequency / rule.frequency_divisor).max(1),
                ));
            }
        }
        expanded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_suffix_and_retains_lemma() {
        let morphology =
            Morphology::new().with_rule(SuffixRule::append("গুলো").with_min_stem_graphemes(2));
        let expanded = morphology.expand([WordEntry::new("মানুষ", 90)]);
        assert!(expanded.iter().any(|entry| entry.word == "মানুষ"));
        assert!(expanded.iter().any(|entry| entry.word == "মানুষগুলো"));
    }

    #[test]
    fn can_replace_stem_ending() {
        let morphology = Morphology::new().with_rule(SuffixRule::replace("ী", "িগুলো"));
        let expanded = morphology.expand([WordEntry::new("নদী", 30)]);
        assert!(expanded.iter().any(|entry| entry.word == "নদিগুলো"));
    }

    #[test]
    fn pos_gated_rule_skips_mismatched_lemma_and_keeps_unknown() {
        let table = LemmaTable::from_json(r#"{"মানুষ":"noun","আমি":"pronoun"}"#.as_bytes());
        let morphology = Morphology::new()
            .with_lemma_table(table)
            .with_rule(SuffixRule::append("গুলো").with_pos(LemmaPos::Noun));
        let expanded = morphology.expand([
            WordEntry::new("মানুষ", 90),
            WordEntry::new("আমি", 90),
            WordEntry::new("কম্পিউটার", 90),
        ]);
        assert!(expanded.iter().any(|entry| entry.word == "মানুষগুলো"));
        assert!(!expanded.iter().any(|entry| entry.word == "আমিগুলো"));
        // Unknown to the table: rule still applies (never drops).
        assert!(expanded.iter().any(|entry| entry.word == "কম্পিউটারগুলো"));
    }

    #[test]
    fn rule_without_pos_gate_ignores_table() {
        let table = LemmaTable::from_json(r#"{"আমি":"pronoun"}"#.as_bytes());
        let morphology = Morphology::new()
            .with_lemma_table(table)
            .with_rule(SuffixRule::append("গুলো"));
        let expanded = morphology.expand([WordEntry::new("আমি", 90)]);
        assert!(expanded.iter().any(|entry| entry.word == "আমিগুলো"));
    }
}
