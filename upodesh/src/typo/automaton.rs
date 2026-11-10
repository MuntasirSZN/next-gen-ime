pub(crate) const INF: u16 = u16::MAX;
/// Maximum query graphemes supported by the fixed-size DP rows.
pub(crate) const MAX_QUERY: usize = 24;
/// Row length including index 0.
pub(crate) const ROW_LEN: usize = MAX_QUERY + 1;

pub(crate) type Row = [u16; ROW_LEN];

#[inline]
pub(crate) fn start_row(deletion: u16, query_len: usize) -> (Row, u16) {
    let mut row = [INF; ROW_LEN];
    row[0] = 0;
    for (offset, slot) in row[1..=query_len].iter_mut().enumerate() {
        let i = u16::try_from(offset + 1).unwrap_or(INF);
        *slot = i.saturating_mul(deletion);
    }
    (row, 0)
}

/// Substitution cost for one (query position, candidate grapheme) pair.
/// `specials` holds cheaper (or explicitly configured) alternatives for the
/// position; everything else costs `base`.
#[inline(always)]
pub(crate) fn substitution(
    base: u16,
    candidate: u16,
    query_id: Option<u16>,
    specials: &[(u16, u16)],
) -> u16 {
    if query_id == Some(candidate) {
        return 0;
    }
    for &(id, c) in specials {
        if id == candidate {
            return c;
        }
    }
    base
}

/// Advance one weighted OSA row by a single candidate grapheme.
///
/// Returns the new row and its minimum over `0..=query_len`. Transposition
/// uses `two_back` (`row[j-1]`), `prev_candidate` (`c[j]`) and the candidate
/// (`c[j+1]`). Grapheme equality for the transposition test uses IDs when
/// both query graphemes are in the table and string slices otherwise.
///
/// Arguments are bundled (clippy `too_many_arguments`).
pub(crate) struct RowAdvance<'a> {
    pub parent: &'a Row,
    pub two_back: Option<&'a Row>,
    pub prev_candidate: Option<u16>,
    pub candidate: u16,
    pub query_len: usize,
    pub query_ids: &'a [Option<u16>],
    pub query_slices: &'a [&'a str],
    pub specials: &'a [&'a [(u16, u16)]],
    pub graphemes: &'a [String],
    pub insertion: u16,
    pub deletion: u16,
    pub base: u16,
    pub transposition: u16,
}

#[inline]
pub(crate) fn advance_row(args: RowAdvance<'_>) -> (Row, u16) {
    let RowAdvance {
        parent,
        two_back,
        prev_candidate,
        candidate,
        query_len,
        query_ids,
        query_slices,
        specials,
        graphemes,
        insertion,
        deletion,
        base,
        transposition,
    } = args;
    let mut row = [INF; ROW_LEN];
    row[0] = parent[0].saturating_add(insertion);
    let mut min = row[0];
    for i in 1..=query_len {
        let delete = parent[i].saturating_add(deletion);
        let insert = row[i - 1].saturating_add(insertion);
        let sub = parent[i - 1].saturating_add(substitution(
            base,
            candidate,
            query_ids[i - 1],
            specials[i - 1],
        ));
        let mut best = delete.min(insert).min(sub);
        if i >= 2
            && let Some(prev) = prev_candidate
            && let Some(two) = two_back
        {
            let prev_match = match query_ids[i - 1] {
                Some(qid) => qid == prev,
                None => query_slices[i - 1] == graphemes[prev as usize],
            };
            let cand_match = match query_ids[i - 2] {
                Some(qid) => qid == candidate,
                None => query_slices[i - 2] == graphemes[candidate as usize],
            };
            if prev_match && cand_match {
                let t = two[i - 2].saturating_add(transposition);
                if t < best {
                    best = t;
                }
            }
        }
        row[i] = best;
        if best < min {
            min = best;
        }
    }
    (row, min)
}

// Note: batch-over-siblings SIMD (fearless_simd) was prototyped and
// measured ~2.1x slower than scalar for these short DP rows (gather +
// transpose copies dominate), so the hot search stays scalar.
