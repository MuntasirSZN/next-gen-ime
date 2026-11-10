use std::error::Error;
use std::fmt;

/// One dictionary entry. `frequency` should ideally come from a Bengali text
/// corpus or usage telemetry, not a made-up uniform count.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WordEntry {
    pub word: String,
    pub frequency: u64,
}

impl WordEntry {
    pub fn new(word: impl Into<String>, frequency: u64) -> Self {
        Self {
            word: word.into(),
            frequency,
        }
    }
}

/// Errors that can occur while constructing the immutable FST index.
#[derive(Debug)]
pub enum BuildError {
    TooManyGraphemes(usize),
    InvalidIndex(&'static str),
    Fst(fst::Error),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyGraphemes(n) => write!(
                f,
                "dictionary contains {n} distinct graphemes; this index supports at most 65536"
            ),
            Self::InvalidIndex(reason) => write!(f, "invalid embedded dictionary index: {reason}"),
            Self::Fst(err) => write!(f, "failed to build dictionary FST: {err}"),
        }
    }
}

impl Error for BuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Fst(err) => Some(err),
            Self::TooManyGraphemes(_) | Self::InvalidIndex(_) => None,
        }
    }
}

impl From<fst::Error> for BuildError {
    fn from(value: fst::Error) -> Self {
        Self::Fst(value)
    }
}

/// One suggestion query: original Roman input (if any) plus Riti's single
/// Bangla conversion for this keystroke.
pub struct SuggestionInput<'a> {
    pub roman: Option<&'a str>,
    pub bangla: &'a str,
    pub context: &'a [&'a str],
    pub mode: super::config::InputMode,
}

impl<'a> SuggestionInput<'a> {
    pub fn bangla_only(bangla: &'a str) -> Self {
        Self {
            roman: None,
            bangla,
            context: &[],
            mode: super::config::InputMode::Bangla,
        }
    }

    pub fn with_roman(roman: &'a str, bangla: &'a str, mode: super::config::InputMode) -> Self {
        Self {
            roman: Some(roman),
            bangla,
            context: &[],
            mode,
        }
    }
}

/// One ranked suggestion.
#[derive(Clone, Debug, PartialEq)]
pub struct Suggestion {
    pub word: String,
    pub score: f64,
    pub edit_cost: u16,
    pub roman_cost: Option<u16>,
    pub frequency: u64,
    pub context_probability: Option<f64>,
}

/// A ranked correction candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct Correction {
    pub word: String,
    /// Weighted edit cost in hundredths. Zero means an exact match.
    pub edit_cost: u16,
    pub frequency: u64,
    /// Negative-log-like combined score. Lower is better.
    pub score: f64,
    /// Present when a contextual language-model score was used.
    pub context_probability: Option<f64>,
}

/// Counts/limits for the embedded index (visibility for bench reporting,
/// candidate counting, memory accounting).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexStats {
    pub words: usize,
    pub graphemes: usize,
    pub max_dict_len: usize,
    pub max_frequency: u64,
}
