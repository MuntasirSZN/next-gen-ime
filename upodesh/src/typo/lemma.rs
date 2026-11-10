//! Lemma/POS metadata (from Wiktextract-derived `data/lemma.json`).
//!
//! The table maps dictionary lemmas to their coarse part of speech. Morphology
//! suffix rules can then be gated on POS, replacing blind suffix-append
//! expansion (a plural suffix on a verb lemma makes no sense; a verbal suffix
//! on a noun is wrong). Entries missing from the table keep legacy behavior —
//! unknown words are unrestricted, never dropped.

use std::collections::HashMap;

/// Coarse part-of-speech label emitted by `generate/extract_lemma.py`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum LemmaPos {
    Noun,
    Adjective,
    Verb,
    Adverb,
    ProperNoun,
    Indeclinable,
    Pronoun,
    Conjunction,
    Postposition,
    Prefix,
    Numeral,
    Determiner,
    Onomatopoeia,
}

impl LemmaPos {
    /// Map a lemma-table label (as written by `generate/extract_lemma.py`).
    pub fn from_label(s: &str) -> Option<Self> {
        Some(match s {
            "noun" => Self::Noun,
            "adj" => Self::Adjective,
            "verb" => Self::Verb,
            "adv" => Self::Adverb,
            "proper_noun" => Self::ProperNoun,
            "indeclinable" => Self::Indeclinable,
            "pronoun" => Self::Pronoun,
            "conjunction" => Self::Conjunction,
            "postposition" => Self::Postposition,
            "prefix" => Self::Prefix,
            "numeral" => Self::Numeral,
            "determiner" => Self::Determiner,
            "onomatopoeia" => Self::Onomatopoeia,
            _ => return None,
        })
    }
}

/// Word → POS lookup. Empty entries are fine — unknown words behave as if
/// no gating were configured.
#[derive(Clone, Debug, Default)]
pub struct LemmaTable {
    map: HashMap<String, LemmaPos>,
}

impl LemmaTable {
    /// Parse a `{"lemma": "pos", ...}` JSON object as written by
    /// `generate/extract_lemma.py`. Returns an empty table on parse failure.
    pub fn from_json(bytes: &[u8]) -> Self {
        let Ok(root) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            return Self::default();
        };
        let mut map = HashMap::new();
        if let Some(obj) = root.as_object() {
            for (word, pos) in obj {
                if let Some(pos) = pos.as_str().and_then(LemmaPos::from_label) {
                    map.insert(word.clone(), pos);
                }
            }
        }
        Self { map }
    }

    pub fn pos(&self, word: &str) -> Option<LemmaPos> {
        self.map.get(word).copied()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lemma_json() {
        let table = LemmaTable::from_json(r#"{"আলো":"noun","যাওয়া":"verb"}"#.as_bytes());
        assert_eq!(table.pos("আলো"), Some(LemmaPos::Noun));
        assert_eq!(table.pos("যাওয়া"), Some(LemmaPos::Verb));
        assert!(table.pos("অজানা").is_none());
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn unknown_labels_are_ignored() {
        // Defensive: forward-compat labels are skipped, never misclassified.
        let table = LemmaTable::from_json(r#"{"x":"interjection"}"#.as_bytes());
        assert!(table.is_empty());
    }
}
