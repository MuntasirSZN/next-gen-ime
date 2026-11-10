/// Search and ranking parameters. Edit costs use hundredths: a normal edit
/// costs 100, so 150 permits one ordinary edit plus one inexpensive confusion.
#[derive(Clone, Debug)]
pub struct CorrectionConfig {
    /// Maximum weighted edit cost accepted during FST traversal.
    pub max_edit_cost: u16,
    /// Default number of suggestions returned by `suggest`.
    pub max_results: usize,
    /// Weight of the dictionary-frequency prior. Applied with or without context.
    pub frequency_weight: f64,
    /// Weight of the negative log bigram probability when context is supplied.
    pub language_model_weight: f64,
    /// Weight of the Roman alignment cost relative to the Bangla score.
    /// `0.0` disables the Roman side entirely (pure Bangla lookup).
    pub roman_weight: f64,
    /// Maximum Roman-side alignment cost (hundredths).
    pub max_roman_cost: u16,
    /// Weight of the Riti transliteration cost (reserved; 0 until Riti
    /// exposes a cost channel).
    pub transliteration_weight: f64,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            max_edit_cost: 150,
            max_results: 10,
            frequency_weight: 0.03,
            language_model_weight: 0.18,
            roman_weight: 1.0,
            max_roman_cost: 200,
            transliteration_weight: 0.0,
        }
    }
}

/// Which transliteration convention a Roman-side input follows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputMode {
    Bangla,
    Avro,
    Khipro,
}
