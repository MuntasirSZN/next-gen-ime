use super::class_weights::ClassWeights;
use super::normalize::normalize_bengali;

/// Weighted edit model over grapheme clusters.
///
/// Substitution costs can be directional (`with_substitution`) or symmetric
/// (`with_confusion`). A lower cost means a more plausible spelling confusion.
/// Costs are relative penalties, not calibrated probabilities.
#[derive(Clone, Debug)]
pub struct EditCosts {
    insertion: u16,
    deletion: u16,
    substitution: u16,
    transposition: u16,
    // Small linear-scan tables. Confusion sets are tiny (a handful of learned
    // probabilities), so a hash map's SipHash overhead would dominate — a
    // linear scan over <= dozens of entries is much cheaper per DP cell.
    confusions: Vec<(String, String, u16)>,
    character_confusions: Vec<(char, char, u16)>,
}

impl Default for EditCosts {
    fn default() -> Self {
        Self::bengali()
    }
}

impl EditCosts {
    /// Apply empirical class weights (see [`ClassWeights`]): insertion becomes
    /// cheaper with more duplicate slips, deletion/transposition shift by
    /// their own masses. Confusion entries are untouched — phonetic
    /// calibration wants the learned confusion matrix, not a uniform tweak.
    pub fn with_class_weights(self, weights: &ClassWeights) -> Self {
        let adjusted_insertion =
            (self.insertion as f64 * weights.insertion_multiplier()).round() as u16;
        let adjusted_deletion =
            (self.deletion as f64 * weights.deletion_multiplier()).round() as u16;
        let adjusted_transposition =
            (self.transposition as f64 * weights.transposition_multiplier()).round() as u16;
        Self {
            insertion: adjusted_insertion.clamp(1, 200),
            deletion: adjusted_deletion.clamp(1, 200),
            transposition: adjusted_transposition.clamp(1, 200),
            ..self
        }
    }

    /// A conservative starter model for common Bengali orthographic
    /// confusions. These defaults are deliberately small and configurable;
    /// production costs should be learned from aligned IME error data.
    pub fn bengali() -> Self {
        let mut costs = Self {
            insertion: 100,
            deletion: 100,
            substitution: 100,
            transposition: 65,
            confusions: Vec::new(),
            character_confusions: Vec::new(),
        };

        costs = costs
            .with_confusion("শ", "ষ", 35)
            .with_confusion("শ", "স", 45)
            .with_confusion("ষ", "স", 45)
            .with_confusion("ন", "ণ", 50)
            .with_confusion("ই", "ঈ", 60)
            .with_confusion("উ", "ঊ", 60)
            .with_confusion("ি", "ী", 60)
            .with_confusion("ু", "ূ", 60);

        costs
    }

    /// Replace the base insertion, deletion, substitution and transposition
    /// costs. Use the same scale as `CorrectionConfig::max_edit_cost`.
    pub fn with_operation_costs(
        mut self,
        insertion: u16,
        deletion: u16,
        substitution: u16,
        transposition: u16,
    ) -> Self {
        self.insertion = insertion;
        self.deletion = deletion;
        self.substitution = substitution;
        self.transposition = transposition;
        self
    }

    /// Add a directed substitution cost from the observed (mistyped) grapheme
    /// to the intended dictionary grapheme. Direction matters for learned
    /// keyboard-error models.
    pub fn with_substitution(
        mut self,
        observed: impl Into<String>,
        intended: impl Into<String>,
        cost: u16,
    ) -> Self {
        let observed = normalize_bengali(&observed.into());
        let intended = normalize_bengali(&intended.into());
        if let Some(slot) = self
            .confusions
            .iter_mut()
            .find(|(o, i, _)| o == &observed && i == &intended)
        {
            slot.2 = cost;
        } else {
            self.confusions
                .push((observed.clone(), intended.clone(), cost));
        }

        // Generalize a one-codepoint difference to graphemes carrying the same
        // other marks. This covers both শি -> ষি and কি -> কী.
        let observed_chars: Vec<char> = observed.chars().collect();
        let intended_chars: Vec<char> = intended.chars().collect();
        if observed_chars.len() == intended_chars.len() {
            let mut difference = None;
            let mut multiple_differences = false;
            for (&observed_char, &intended_char) in observed_chars.iter().zip(intended_chars.iter())
            {
                if observed_char != intended_char {
                    if difference.is_some() {
                        multiple_differences = true;
                        break;
                    }
                    difference = Some((observed_char, intended_char));
                }
            }
            if !multiple_differences && let Some((observed_char, intended_char)) = difference {
                if let Some(slot) = self
                    .character_confusions
                    .iter_mut()
                    .find(|(o, i, _)| *o == observed_char && *i == intended_char)
                {
                    slot.2 = cost;
                } else {
                    self.character_confusions
                        .push((observed_char, intended_char, cost));
                }
            }
        }
        self
    }

    /// Add a symmetric pair of grapheme confusions. Use `with_substitution`
    /// when the two directions have different probabilities.
    pub fn with_confusion(
        self,
        left: impl Into<String>,
        right: impl Into<String>,
        cost: u16,
    ) -> Self {
        let left = left.into();
        let right = right.into();
        self.with_substitution(left.clone(), right.clone(), cost)
            .with_substitution(right, left, cost)
    }

    pub(crate) fn insertion(&self) -> u16 {
        self.insertion
    }

    pub(crate) fn deletion(&self) -> u16 {
        self.deletion
    }

    pub(crate) fn transposition(&self) -> u16 {
        self.transposition
    }

    pub(crate) fn base_substitution(&self) -> u16 {
        self.substitution
    }

    pub(crate) fn char_confusion_cost(&self, left: char, right: char) -> Option<u16> {
        self.character_confusions
            .iter()
            .find(|(o, i, _)| *o == left && *i == right)
            .map(|(_, _, cost)| *cost)
    }

    #[cfg(test)]
    pub(crate) fn substitution_cost(&self, left: &str, right: &str) -> u16 {
        if left == right {
            return 0;
        }
        if let Some((_, _, cost)) = self
            .confusions
            .iter()
            .find(|(o, i, _)| o.as_str() == left && i.as_str() == right)
        {
            return *cost;
        }

        let mut left_chars = left.chars();
        let mut right_chars = right.chars();
        let mut difference = None;
        loop {
            match (left_chars.next(), right_chars.next()) {
                (Some(left_char), Some(right_char)) if left_char == right_char => {}
                (Some(left_char), Some(right_char)) => {
                    if difference.is_some() {
                        return self.substitution;
                    }
                    difference = Some((left_char, right_char));
                }
                (None, None) => break,
                _ => return self.substitution,
            }
        }

        difference
            .and_then(|(left_char, right_char)| {
                self.character_confusions
                    .iter()
                    .find(|(o, i, _)| *o == left_char && *i == right_char)
                    .map(|(_, _, cost)| *cost)
            })
            .unwrap_or(self.substitution)
    }

    pub(crate) fn confusions(&self) -> &[(String, String, u16)] {
        &self.confusions
    }

    pub(crate) fn character_confusions(&self) -> &[(char, char, u16)] {
        &self.character_confusions
    }
}
