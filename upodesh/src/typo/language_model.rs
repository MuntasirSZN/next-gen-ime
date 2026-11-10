use std::collections::HashMap;

use super::normalize::normalize_cow;

/// Compact interpolated Kneser-Ney bigram language model.
///
/// Vocabulary entries are interned to `u32` IDs, and bigrams are stored as ID
/// pairs. This avoids storing two owned `String`s for every bigram, which is
/// prohibitively expensive for even moderately sized Bengali corpora.
///
/// Training input is whitespace-tokenized. Callers should use the same token
/// normalization policy for training text and runtime context.
#[derive(Clone, Debug)]
pub struct BigramLanguageModel {
    word_ids: HashMap<String, u32>,
    bigram_counts: HashMap<(u32, u32), u64>,
    context_totals: Vec<u64>,
    distinct_followers: Vec<u64>,
    continuation_counts: Vec<u64>,
    total_distinct_bigrams: u64,
    discount: f64,
}

impl BigramLanguageModel {
    pub fn train<I, S>(sentences: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut word_ids = HashMap::<String, u32>::new();
        let mut bigram_counts = HashMap::<(u32, u32), u64>::new();

        for sentence in sentences {
            let mut previous = None;
            for token in sentence.as_ref().split_whitespace() {
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

                if let Some(previous_id) = previous {
                    let count = bigram_counts.entry((previous_id, word_id)).or_default();
                    *count = count.saturating_add(1);
                }
                previous = Some(word_id);
            }
        }

        let mut context_totals = vec![0_u64; word_ids.len()];
        let mut distinct_followers = vec![0_u64; word_ids.len()];
        let mut continuation_counts = vec![0_u64; word_ids.len()];

        for (&(previous, word), &count) in &bigram_counts {
            context_totals[previous as usize] =
                context_totals[previous as usize].saturating_add(count);
            distinct_followers[previous as usize] =
                distinct_followers[previous as usize].saturating_add(1);
            continuation_counts[word as usize] =
                continuation_counts[word as usize].saturating_add(1);
        }

        Self {
            word_ids,
            total_distinct_bigrams: bigram_counts.len() as u64,
            bigram_counts,
            context_totals,
            distinct_followers,
            continuation_counts,
            discount: 0.75,
        }
    }

    /// Probability estimate for `word` after `previous`.
    ///
    /// A small floor is applied for out-of-vocabulary words so the caller can
    /// always take a finite logarithm. The floor is a ranking safeguard, not
    /// an extra normalized `<unk>` model.
    pub fn probability(&self, previous: &str, word: &str) -> f64 {
        let previous = normalize_cow(previous);
        let word = normalize_cow(word);
        self.probability_normalized(previous.as_ref(), word.as_ref())
    }

    /// Probability estimate assuming both inputs are already NFC-normalized.
    /// The correction hot path uses this to avoid two allocations per
    /// candidate.
    pub(crate) fn probability_normalized(&self, previous: &str, word: &str) -> f64 {
        if self.total_distinct_bigrams == 0 {
            return 1.0 / (self.word_ids.len().max(1) as f64);
        }

        let word_id = self.word_ids.get(word).copied();
        let continuation_probability = word_id
            .map(|id| self.continuation_counts[id as usize] as f64)
            .unwrap_or(0.0)
            / self.total_distinct_bigrams as f64;

        let Some(previous_id) = self.word_ids.get(previous).copied() else {
            return continuation_probability.max(self.unknown_probability());
        };
        let context_total = self.context_totals[previous_id as usize];
        if context_total == 0 {
            return continuation_probability.max(self.unknown_probability());
        }

        let bigram_count = word_id
            .and_then(|id| self.bigram_counts.get(&(previous_id, id)).copied())
            .unwrap_or(0) as f64;
        let distinct_followers = self.distinct_followers[previous_id as usize] as f64;
        let total = context_total as f64;
        let discounted = (bigram_count - self.discount).max(0.0) / total;
        let backoff_mass = (self.discount * distinct_followers / total).min(1.0);
        let probability = discounted + backoff_mass * continuation_probability;
        probability.max(self.unknown_probability())
    }

    fn unknown_probability(&self) -> f64 {
        1.0 / ((self.total_distinct_bigrams as f64 + self.word_ids.len() as f64 + 1.0) * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_bigram_beats_unseen_bigram() {
        let model = BigramLanguageModel::train([
            "আমি বাংলা লিখি",
            "আমি বাংলা পড়ি",
            "আমি বাংলা লিখি",
            "তুমি ইংরেজি লিখো",
        ]);
        assert!(model.probability("আমি", "বাংলা") > model.probability("আমি", "ইংরেজি"));
    }

    #[test]
    fn empty_model_is_finite() {
        let model = BigramLanguageModel::train(std::iter::empty::<&str>());
        let probability = model.probability("আমি", "বাংলা");
        assert!(probability.is_finite() && probability > 0.0);
    }

    #[test]
    fn unseen_history_uses_continuation_probability() {
        let model = BigramLanguageModel::train(["আমি বাংলা লিখি", "তুমি বাংলা পড়ো"]);
        assert!(model.probability("অজানা", "বাংলা") > model.probability("অজানা", "লিখি"));
    }
}
