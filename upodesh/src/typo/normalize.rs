use std::borrow::Cow;

use simd_normalizer::UnicodeNormalization;

/// Canonical Unicode normalization shared by indexing and lookup.
///
/// Do not strip Bengali marks, virama, ZWJ or ZWNJ here: those can change the
/// spelling/orthography. NFC only makes canonically equivalent encodings equal.
pub(crate) fn normalize_bengali(input: &str) -> String {
    input.nfc().into_owned()
}

/// Borrow the input when it is already NFC, avoiding an allocation on the
/// common path. `simd-normalizer` returns `Cow::Borrowed` for already-normalized
/// input, so this is a zero-allocation fast path on clean text.
pub(crate) fn normalize_cow(input: &str) -> Cow<'_, str> {
    input.nfc()
}
