#[cfg(test)]
use super::EditCosts;

#[cfg(test)]
const INF: u16 = u16::MAX;

/// Weighted optimal-string-alignment distance (restricted Damerau-Levenshtein)
/// over grapheme-cluster strings. Adjacent transpositions are supported; unlike
/// unrestricted Damerau-Levenshtein, a substring cannot be edited repeatedly
/// through overlapping transpositions.
///
/// The FST automaton already tracks this exact distance during traversal
/// (see `WeightedDamerauState::cost`). This function remains as the
/// brute-force reference used by tests to validate the automaton.
#[cfg(test)]
pub(crate) fn weighted_osa_distance(
    query: &[String],
    candidate: &[String],
    costs: &EditCosts,
) -> u16 {
    let m = query.len();
    if m == 0 {
        return u16::try_from(candidate.len())
            .unwrap_or(u16::MAX)
            .saturating_mul(costs.insertion());
    }
    if candidate.is_empty() {
        return u16::try_from(m)
            .unwrap_or(u16::MAX)
            .saturating_mul(costs.deletion());
    }

    // `previous` is D[i][j - 1], and `two_back` is D[i][j - 2].
    let mut previous: Vec<u16> = (0..=m)
        .map(|i| {
            u16::try_from(i)
                .unwrap_or(u16::MAX)
                .saturating_mul(costs.deletion())
        })
        .collect();
    let mut two_back: Option<Vec<u16>> = None;
    let mut previous_candidate: Option<&str> = None;

    for current_candidate in candidate {
        let mut current = vec![INF; m + 1];
        current[0] = previous[0].saturating_add(costs.insertion());

        for i in 1..=m {
            let delete = previous[i].saturating_add(costs.deletion());
            let insert = current[i - 1].saturating_add(costs.insertion());
            let substitute = previous[i - 1]
                .saturating_add(costs.substitution_cost(&query[i - 1], current_candidate));
            current[i] = delete.min(insert).min(substitute);

            if i >= 2
                && previous_candidate == Some(query[i - 1].as_str())
                && query[i - 2] == *current_candidate
                && let Some(row_two_back) = two_back.as_ref()
            {
                let transpose = row_two_back[i - 2].saturating_add(costs.transposition());
                current[i] = current[i].min(transpose);
            }
        }

        two_back = Some(previous);
        previous = current;
        previous_candidate = Some(current_candidate);
    }

    previous[m]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_is_zero() {
        let costs = EditCosts::default();
        let value = vec!["বা".to_owned(), "ং".to_owned(), "লা".to_owned()];
        assert_eq!(weighted_osa_distance(&value, &value, &costs), 0);
    }

    #[test]
    fn insertion_and_deletion_are_weighted() {
        let costs = EditCosts::default();
        let a = vec!["ক".to_owned(), "খ".to_owned()];
        let b = vec!["ক".to_owned(), "গ".to_owned(), "খ".to_owned()];
        assert_eq!(weighted_osa_distance(&a, &b, &costs), 100);
        assert_eq!(weighted_osa_distance(&b, &a, &costs), 100);
    }

    #[test]
    fn adjacent_transposition_is_one_operation() {
        let costs = EditCosts::default();
        let a = vec!["ক".to_owned(), "খ".to_owned(), "গ".to_owned()];
        let b = vec!["খ".to_owned(), "ক".to_owned(), "গ".to_owned()];
        assert_eq!(weighted_osa_distance(&a, &b, &costs), 65);
    }

    #[test]
    fn common_bengali_confusion_is_cheaper_than_substitution() {
        let costs = EditCosts::default();
        let a = vec!["শ".to_owned()];
        let b = vec!["ষ".to_owned()];
        assert_eq!(weighted_osa_distance(&a, &b, &costs), 35);
    }
}
