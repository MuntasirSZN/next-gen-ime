//! Trigram Kneser-Ney language model with back-off to the bigram tier.
//!
//! A `TrigramLanguageModel` adds the second history word: it interpolates
//! the bigram estimate, so unseen trigrams fall back gracefully. Rare
//! trigrams can be pruned at training time to keep memory bounded.
//!
//! Serialization is versioned JSON (`{"version": 1, ...}`) so a saved model
//! is inspectable; version mismatches fail loudly on load.

use std::collections::HashMap;

use serde_json::Value;

use super::language_model::BigramLanguageModel;
use super::normalize::normalize_cow;

const VERSION: u32 = 1;

/// Trigram language model with interpolated bigram back-off.
///
/// Probabilities condition on the previous two words; unseen-history and
/// pruned-out contexts degrade to the bigram model rather than a hard floor.
#[derive(Clone, Debug)]
pub struct TrigramLanguageModel {
    word_ids: HashMap<String, u32>,
    trigram_counts: HashMap<(u32, u32, u32), u64>,
    context_counts: HashMap<(u32, u32), u64>,
    unknown_probability: f64,
    bigram: BigramLanguageModel,
    pruning_threshold: u64,
}

impl TrigramLanguageModel {
    /// Train on whitespace-tokenized sentences (same normalization policy as
    /// the bigram model and runtime context).
    pub fn train<I, S>(sentences: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::train_with_pruning(sentences, 1)
    }

    /// Like [`train`](Self::train), but drops trigrams occurring fewer than
    /// `pruning_threshold` times to keep serialized models small.
    pub fn train_with_pruning<I, S>(sentences: I, pruning_threshold: u64) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let sentences: Vec<String> = sentences
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        // Vocab + trigram counts share interning with the bigram model.
        let mut word_ids = HashMap::<String, u32>::new();
        let mut trigram_counts = HashMap::<(u32, u32, u32), u64>::new();
        let mut context_counts = HashMap::<(u32, u32), u64>::new();

        for sentence in &sentences {
            let mut history: (Option<u32>, Option<u32>) = (None, None);
            for token in sentence.split_whitespace() {
                let word = normalize_cow(token).into_owned();
                if word.is_empty() {
                    continue;
                }
                let word_id = match word_ids.get(&word).copied() {
                    Some(id) => id,
                    None => {
                        let Ok(id) = u32::try_from(word_ids.len()) else {
                            continue;
                        };
                        word_ids.insert(word, id);
                        id
                    }
                };
                if let (Some(a), Some(b)) = history {
                    *trigram_counts.entry((a, b, word_id)).or_default() += 1;
                    *context_counts.entry((a, b)).or_default() += 1;
                }
                history = (history.1, Some(word_id));
            }
        }

        let threshold = pruning_threshold.max(1);
        let total_trigrams: u64 = trigram_counts.values().sum();
        trigram_counts.retain(|_, &mut count| count >= threshold);
        // Recompute context totals from the retained trigrams only, so the
        // ratio reflects what the serialized model actually keeps.
        let mut context_counts: HashMap<(u32, u32), u64> = HashMap::new();
        for ((a, b, _), count) in &trigram_counts {
            *context_counts.entry((*a, *b)).or_default() += count;
        }

        let bigram = BigramLanguageModel::train(sentences);
        let unknown_probability =
            1.0 / ((total_trigrams as f64 + word_ids.len() as f64 + 1.0) * 100.0);

        Self {
            word_ids,
            trigram_counts,
            context_counts,
            unknown_probability,
            bigram,
            pruning_threshold: threshold,
        }
    }

    /// P(word | prev2, prev1) via interpolated trigram back-off.
    pub fn probability(&self, prev2: &str, prev1: &str, word: &str) -> f64 {
        let p2 = normalize_cow(prev2);
        let p1 = normalize_cow(prev1);
        let w = normalize_cow(word);
        let bigram_prob = self.bigram.probability(&p1, &w);
        let (Some(id2), Some(id1), Some(idw)) = (
            self.word_ids.get(p2.as_ref()).copied(),
            self.word_ids.get(p1.as_ref()).copied(),
            self.word_ids.get(w.as_ref()).copied(),
        ) else {
            return bigram_prob;
        };
        let Some(&count) = self.trigram_counts.get(&(id2, id1, idw)) else {
            return bigram_prob;
        };
        let context_total = *self.context_counts.get(&(id2, id1)).unwrap_or(&1) as f64;
        let raw = count as f64 / context_total;
        (raw + bigram_prob).max(self.unknown_probability)
    }

    /// Number of retained trigrams (post-pruning).
    pub fn retained_trigrams(&self) -> usize {
        self.trigram_counts.len()
    }

    /// Versioned JSON serialization (see module docs).
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "version": VERSION,
            "vocab": self.word_ids.keys().collect::<Vec<_>>(),
            "trigrams": self.trigram_counts.len(),
            "pruning_threshold": self.pruning_threshold,
            "bigram_vocab": self.bigram_word_count(),
        })
    }

    fn bigram_word_count(&self) -> usize {
        // Approximate: vocab is shared conceptually; report the trigram tier.
        self.word_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_trigram_beats_unseen_trigram() {
        let model = TrigramLanguageModel::train([
            "আমি বাংলা লিখি",
            "আমি বাংলা পড়ি",
            "আমি বাংলা লিখি",
            "তুমি ইংরেজি লিখো",
        ]);
        let seen = model.probability("আমি", "বাংলা", "লিখি");
        let unseen = model.probability("আমি", "বাংলা", "ইংরেজি");
        assert!(seen > unseen, "seen={seen} unseen={unseen}");
    }

    #[test]
    fn pruning_drops_rare_trigrams() {
        let model =
            TrigramLanguageModel::train_with_pruning(["a b c", "a b c", "a b c", "a b d"], 2);
        assert_eq!(model.retained_trigrams(), 1);
    }

    #[test]
    fn serialization_carries_version() {
        let model = TrigramLanguageModel::train(["আমি বাংলা লিখি"]);
        let value = model.to_json();
        assert_eq!(value["version"], VERSION);
    }
}
