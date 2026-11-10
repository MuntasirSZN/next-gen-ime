//! Empirical error-class weights (synthetic pipeline artifact).
//!
//! `data/weights.json` carries calibrated class frequencies produced offline
//! by `generate/train_weights.py` (IndicCorp CC-0 base + synthetic operators).
//! Consumers opt in — the hand-tuned defaults in `EditCosts` stay
//! authoritative so existing rankings and tests are stable.

use serde_json::Value;

/// Calibrated class proportions from `generate/train_weights.py`.
#[derive(Clone, Debug)]
pub struct ClassWeights {
    pub delete: f64,
    pub duplicate: f64,
    pub mispress: f64,
    pub mispress_insert: f64,
    pub phonetic: f64,
    pub transpose: f64,
}

impl ClassWeights {
    /// Parse a `weights.json` payload
    /// (`{"class_weights": {"delete": 0.2, ...}}`). Missing keys default to 0
    /// (identity multiplier).
    pub fn from_json(bytes: &[u8]) -> Self {
        let Ok(root) = serde_json::from_slice::<Value>(bytes) else {
            return Self {
                delete: 0.0,
                duplicate: 0.0,
                mispress: 0.0,
                mispress_insert: 0.0,
                phonetic: 0.0,
                transpose: 0.0,
            };
        };
        let w = &root["class_weights"];
        let get = |k: &str| w[k].as_f64().unwrap_or(0.0);
        Self {
            delete: get("delete"),
            duplicate: get("duplicate"),
            mispress: get("mispress"),
            mispress_insert: get("mispress-insert"),
            phonetic: get("phonetic"),
            transpose: get("transpose"),
        }
    }

    /// Doubling is the most common typing slip; more duplicate mass makes
    /// plain insertion cheaper.
    pub fn insertion_multiplier(&self) -> f64 {
        (1.0 - self.duplicate / 2.0).clamp(0.5, 1.5)
    }

    pub fn deletion_multiplier(&self) -> f64 {
        (1.0 - self.delete / 2.0).clamp(0.5, 1.5)
    }

    pub fn transposition_multiplier(&self) -> f64 {
        (1.0 - self.transpose / 2.0).clamp(0.5, 1.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_class_weights() {
        let weights = ClassWeights::from_json(
            br#"{"class_weights": {"delete": 0.2, "duplicate": 0.21, "mispress": 0.15, "mispress-insert": 0.15, "phonetic": 0.08, "transpose": 0.2}}"#,
        );
        assert!((weights.delete - 0.2).abs() < 1e-9);
        assert!(weights.insertion_multiplier() < 1.0);
        assert!(weights.transposition_multiplier() < 1.0);
    }

    #[test]
    fn empty_object_defaults_to_zero() {
        let weights = ClassWeights::from_json(b"{}");
        assert_eq!(weights.insertion_multiplier(), 1.0);
    }
}
