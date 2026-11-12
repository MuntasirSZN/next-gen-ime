use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

use fst::{Map, Streamer};
use unicode_segmentation::UnicodeSegmentation;

use super::config::{CorrectionConfig, InputMode};
use super::costs::EditCosts;
use super::distance::weighted_osa_distance;
use super::language_model::BigramLanguageModel;
use super::morphology::Morphology;
use super::normalize::normalize_cow;
use super::trie::Trie;
use super::trigram::TrigramLanguageModel;
use super::types::{BuildError, Correction, IndexStats, Suggestion, SuggestionInput, WordEntry};
use super::{automaton, roman};

/// FNV-1a over a byte slice (blob integrity offsets must mirror build.rs).
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Compare two corrections in the order exposed by the public API.
/// Exact matches are always preferred over autocorrections.
fn compare_corrections(left: &Correction, right: &Correction) -> Ordering {
    right
        .edit_cost
        .eq(&0)
        .cmp(&left.edit_cost.eq(&0))
        .then_with(|| left.score.total_cmp(&right.score))
        .then_with(|| right.frequency.cmp(&left.frequency))
        .then_with(|| left.word.cmp(&right.word))
}

/// `BinaryHeap` is a max-heap, so this ordering deliberately makes the worst
/// retained candidate the heap's first element.
struct HeapCandidate(Correction);

impl PartialEq for HeapCandidate {
    fn eq(&self, other: &Self) -> bool {
        compare_corrections(&self.0, &other.0) == Ordering::Equal
    }
}

impl Eq for HeapCandidate {}

impl PartialOrd for HeapCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_corrections(&self.0, &other.0)
    }
}

/// Precompute cheaper substitution costs for every query grapheme ID.
///
/// Returns `specials[q_id] = [(candidate_id, cost)]` covering explicit
/// grapheme confusions and the single-character generalization (e.g.
/// শ → ষ also applies to শি → ষি). This reproduces
/// [`EditCosts::substitution_cost`] exactly, but once per index build
/// instead of per DP cell. Direct grapheme costs take precedence over
/// generalized costs, matching the reference implementation.
fn build_specials(
    graphemes: &[String],
    grapheme_ids: &HashMap<String, u16>,
    costs: &EditCosts,
) -> Vec<Vec<(u16, u16)>> {
    let n = graphemes.len();
    let mut specials: Vec<Vec<(u16, u16)>> = vec![Vec::new(); n];
    let add = |specials: &mut Vec<Vec<(u16, u16)>>, q: u16, g: u16, cost: u16| {
        let slot = &mut specials[q as usize];
        if let Some(entry) = slot.iter_mut().find(|(id, _)| *id == g) {
            entry.1 = entry.1.min(cost);
        } else {
            slot.push((g, cost));
        }
    };

    // Generalized single-character pairs by neighbor generation: for each
    // directed character confusion, rewrite every grapheme containing the
    // observed character and look the result up in the grapheme table.
    // This is O(table size x confusions) instead of an all-pairs scan.
    let mut buf = String::new();
    for &(observed_char, intended_char, cost) in costs.character_confusions() {
        for (gid, grapheme) in graphemes.iter().enumerate() {
            if !grapheme.contains(observed_char) {
                continue;
            }
            for (byte_pos, ch) in grapheme.char_indices() {
                if ch != observed_char {
                    continue;
                }
                buf.clear();
                buf.push_str(&grapheme[..byte_pos]);
                buf.push(intended_char);
                buf.push_str(&grapheme[byte_pos + ch.len_utf8()..]);
                if let Some(&nid) = grapheme_ids.get(buf.as_str()) {
                    add(&mut specials, gid as u16, nid, cost);
                }
            }
        }
    }

    // Explicit grapheme confusions overwrite generalized costs.
    for (observed, intended, cost) in costs.confusions() {
        if let (Some(&q), Some(&g)) = (grapheme_ids.get(observed), grapheme_ids.get(intended)) {
            let slot = &mut specials[q as usize];
            if let Some(entry) = slot.iter_mut().find(|(id, _)| *id == g) {
                entry.1 = *cost;
            } else {
                slot.push((g, *cost));
            }
        }
    }
    specials
}

/// Specials for an out-of-vocabulary query grapheme: direct confusions whose
/// observed string equals the query, plus single-character generalizations
/// against every dictionary grapheme.
fn specials_for_oov(
    query: &str,
    graphemes: &[String],
    grapheme_ids: &HashMap<String, u16>,
    costs: &EditCosts,
) -> Vec<(u16, u16)> {
    let mut out: Vec<(u16, u16)> = Vec::new();
    for (observed, intended, cost) in costs.confusions() {
        if observed == query
            && let Some(&g) = grapheme_ids.get(intended)
        {
            out.push((g, *cost));
        }
    }
    let q_chars: Vec<char> = query.chars().collect();
    for (gid, grapheme) in graphemes.iter().enumerate() {
        let g_len = grapheme.chars().count();
        if g_len != q_chars.len() {
            continue;
        }
        let mut diff: Option<(char, char)> = None;
        let mut multiple = false;
        for (&qc, gc) in q_chars.iter().zip(grapheme.chars()) {
            if qc != gc {
                if diff.is_some() {
                    multiple = true;
                    break;
                }
                diff = Some((qc, gc));
            }
        }
        if multiple {
            continue;
        }
        if let Some((qc, gc)) = diff
            && let Some(cost) = costs.char_confusion_cost(qc, gc)
        {
            let id = gid as u16;
            if out.iter().any(|(existing, _)| *existing == id) {
                continue;
            }
            out.push((id, cost));
        }
    }
    out
}

/// Scored ranking key without the word: (is_exact, score, -frequency).
/// A candidate strictly worse than the heap worst under this key can be
/// discarded without decoding its grapheme path.
fn frequency_score(frequency: u64, weight: f64) -> f64 {
    if frequency <= 1 || weight == 0.0 {
        0.0
    } else {
        weight * (frequency as f64).ln()
    }
}

/// Immutable Bengali correction index.
///
/// Each dictionary word is NFC-normalized and segmented into Unicode extended
/// grapheme clusters. Each distinct grapheme is assigned a stable `u16` ID
/// and a grapheme trie indexes the ID sequences. The edit search performs
/// its weighted OSA dynamic program over grapheme clusters, never over
/// UTF-8 bytes.
pub struct Corrector {
    trie: Trie,
    roman: [roman::RomanTrie; 2],
    graphemes: Vec<String>,
    grapheme_ids: HashMap<String, u16>,
    specials: Vec<Vec<(u16, u16)>>,
    word_count: usize,
    max_dict_len: usize,
    max_freq: u64,
    config: CorrectionConfig,
    edit_costs: EditCosts,
    language_model: Option<BigramLanguageModel>,
    trigram_language_model: Option<TrigramLanguageModel>,
}

/// Build both Roman tries from word list using `dhoni` (git dependency).
/// Khipro forms use Khipro mode, so they carry `/` candrabindu markers,
/// `t/` for ৎ, and trailing-`/` variants for virama words; Avro forms use
/// Avro mode.
fn build_roman_tries(words: Vec<String>) -> [roman::RomanTrie; 2] {
    let mut words_blob = Vec::new();
    let mut word_off: Vec<u32> = Vec::with_capacity(words.len());
    for w in &words {
        word_off.push(words_blob.len() as u32);
        words_blob.extend_from_slice(w.as_bytes());
    }
    let mut avro_pairs: Vec<(Vec<u8>, u32)> = Vec::new();
    let mut khipro_pairs: Vec<(Vec<u8>, u32)> = Vec::new();
    for (i, w) in words.iter().enumerate() {
        let wid = i as u32;
        for form in dhoni::convert_to_phonetic_variants(w, dhoni::Mode::Avro) {
            if !form.is_empty() && form != *w {
                avro_pairs.push((form.into_bytes(), wid));
            }
        }
        for form in dhoni::convert_to_phonetic_variants(w, dhoni::Mode::Khipro) {
            if !form.is_empty() && form != *w {
                khipro_pairs.push((form.into_bytes(), wid));
            }
        }
    }
    avro_pairs.sort();
    avro_pairs.dedup();
    khipro_pairs.sort();
    khipro_pairs.dedup();
    let a = roman::RomanTrie::build(&mut avro_pairs, words_blob.clone(), word_off.clone());
    let k = roman::RomanTrie::build(&mut khipro_pairs, words_blob, word_off);
    [a, k]
}

/// One merged `suggest_input` candidate: (edit cost, best roman cost,
/// frequency, context probability).
type MergedCandidate = (u16, Option<u16>, u64, Option<f64>);

impl Corrector {
    /// Counts and limits for the embedded index (words, graphemes, lengths).
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            graphemes: self.graphemes.len(),
            words: self.word_count,
            max_dict_len: self.max_dict_len,
            max_frequency: self.max_freq,
        }
    }

    /// A minimal empty corrector that cannot fail to construct.
    pub(crate) fn empty_fallback() -> Self {
        let trie = Trie::from_sorted_paths(&mut []);
        Self {
            trie,
            roman: [roman::RomanTrie::empty(), roman::RomanTrie::empty()],
            graphemes: Vec::new(),
            grapheme_ids: HashMap::new(),
            specials: Vec::new(),
            word_count: 0,
            max_dict_len: 0,
            max_freq: 0,
            config: CorrectionConfig::default(),
            edit_costs: EditCosts::bengali(),
            language_model: None,
            trigram_language_model: None,
        }
    }

    /// Build an index from words, assigning frequency 1 to every distinct word.
    pub fn from_words<I, S>(words: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::from_entries(
            words
                .into_iter()
                .map(|word| WordEntry::new(word.as_ref(), 1)),
        )
    }

    /// Build an index from lemmas and explicitly configured suffix rules.
    /// The rules are expanded once at build time; they are not recursively
    /// composed. Supply only rules that are valid for the dictionary's lemma
    /// classes.
    pub fn from_entries_with_morphology<I>(
        entries: I,
        morphology: &Morphology,
    ) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = WordEntry>,
    {
        Self::from_entries(morphology.expand(entries))
    }

    /// Build an index from words and corpus frequencies. Duplicate normalized
    /// entries are merged by summing their frequencies.
    pub fn from_entries<I>(entries: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = WordEntry>,
    {
        let mut normalized_entries = BTreeMap::<String, u64>::new();
        let mut grapheme_set = BTreeSet::<String>::new();

        for entry in entries {
            let trimmed = entry.word.trim();
            let word = normalize_cow(trimmed);
            let word_str: &str = &word;
            if word_str.is_empty() {
                continue;
            }
            for grapheme in UnicodeSegmentation::graphemes(word_str, true) {
                grapheme_set.insert(grapheme.to_owned());
            }
            let frequency = entry.frequency.max(1);
            let slot = normalized_entries.entry(word_str.to_owned()).or_default();
            *slot = (*slot).saturating_add(frequency);
        }

        if grapheme_set.len() > (u16::MAX as usize) + 1 {
            return Err(BuildError::TooManyGraphemes(grapheme_set.len()));
        }

        let graphemes: Vec<String> = grapheme_set.into_iter().collect();
        let grapheme_ids: HashMap<String, u16> = graphemes
            .iter()
            .enumerate()
            .map(|(id, grapheme)| (grapheme.clone(), id as u16))
            .collect();

        let mut paths: Vec<(Vec<u16>, u64)> = Vec::with_capacity(normalized_entries.len());
        let mut max_dict_len = 0usize;
        for (word, frequency) in &normalized_entries {
            let mut path = Vec::new();
            for grapheme in UnicodeSegmentation::graphemes(word.as_str(), true) {
                path.push(
                    *grapheme_ids
                        .get(grapheme)
                        .ok_or(BuildError::InvalidIndex("grapheme table incomplete"))?,
                );
            }
            max_dict_len = max_dict_len.max(path.len());
            paths.push((path, *frequency));
        }
        paths.sort_by(|left, right| left.0.cmp(&right.0));
        let max_freq: u64 = paths.iter().map(|(_, f)| *f).max().unwrap_or(0);
        let word_count = paths.len();
        let trie = Trie::from_sorted_paths(&mut paths);
        let edit_costs = EditCosts::bengali();
        let specials = build_specials(&graphemes, &grapheme_ids, &edit_costs);
        let words: Vec<String> = normalized_entries.keys().cloned().collect();
        let roman = build_roman_tries(words);

        Ok(Self {
            trie,
            roman,
            graphemes,
            grapheme_ids,
            specials,
            word_count,
            max_dict_len,
            max_freq,
            config: CorrectionConfig::default(),
            edit_costs,
            language_model: None,
            trigram_language_model: None,
        })
    }

    /// Load a build-time generated index blob produced by `build.rs`.
    ///
    /// The blob contains the grapheme table followed by the raw FST bytes.
    /// The FST is streamed once into the grapheme trie; lookups never touch
    /// the FST again.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BuildError> {
        const MAGIC_V3: &[u8; 8] = b"UPDSHIX3";
        if bytes.get(..8) != Some(MAGIC_V3) {
            return Err(BuildError::InvalidIndex("bad magic or truncated header"));
        }
        let magic_len = 8;
        let mut cursor = magic_len;

        let read_u32 = |cursor: &mut usize, reason: &'static str| {
            let chunk = bytes
                .get(*cursor..*cursor + 4)
                .ok_or(BuildError::InvalidIndex(reason))?;
            *cursor += 4;
            Ok::<u32, BuildError>(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        };

        let grapheme_count = read_u32(&mut cursor, "truncated grapheme count")? as usize;
        if grapheme_count > (u16::MAX as usize) + 1 {
            return Err(BuildError::InvalidIndex("too many graphemes"));
        }

        let mut graphemes = Vec::with_capacity(grapheme_count);
        let mut grapheme_ids = HashMap::with_capacity(grapheme_count);
        for id in 0..grapheme_count {
            let len = read_u32(&mut cursor, "truncated grapheme length")? as usize;
            let end = cursor
                .checked_add(len)
                .ok_or(BuildError::InvalidIndex("grapheme length overflow"))?;
            let raw = bytes
                .get(cursor..end)
                .ok_or(BuildError::InvalidIndex("truncated grapheme data"))?;
            cursor = end;
            let grapheme = String::from_utf8(raw.to_vec())
                .map_err(|_| BuildError::InvalidIndex("grapheme is not UTF-8"))?;
            grapheme_ids.insert(grapheme.clone(), id as u16);
            graphemes.push(grapheme);
        }

        let fst_len = read_u32(&mut cursor, "truncated FST length")? as usize;
        let end = cursor
            .checked_add(fst_len)
            .ok_or(BuildError::InvalidIndex("FST length overflow"))?;
        let fst_bytes = bytes
            .get(cursor..end)
            .ok_or(BuildError::InvalidIndex("truncated FST data"))?;
        cursor = end;
        let map = Map::new(fst_bytes.to_vec())?;

        let mut max_dict_len = 0usize;
        let mut max_freq = 0u64;
        let mut word_count = 0usize;
        // FST streams are already in lexicographic key order, which equals
        // grapheme-ID path order, so the trie is built incrementally.
        let mut builder = crate::typo::trie::TrieBuilder::new();
        let mut path: Vec<u16> = Vec::new();
        let mut stream = map.stream();
        while let Some((key, frequency)) = stream.next() {
            if !key.len().is_multiple_of(2) {
                return Err(BuildError::InvalidIndex("odd-length FST key"));
            }
            path.clear();
            for pair in key.as_chunks::<2>().0 {
                let id = u16::from_be_bytes(*pair) as usize;
                if id >= grapheme_count {
                    return Err(BuildError::InvalidIndex("grapheme ID out of range"));
                }
                path.push(id as u16);
            }
            max_dict_len = max_dict_len.max(path.len());
            max_freq = max_freq.max(frequency);
            builder.add(&path, frequency.max(1));
            word_count += 1;
        }
        let trie = builder.finish();
        let edit_costs = EditCosts::bengali();
        let specials = build_specials(&graphemes, &grapheme_ids, &edit_costs);
        let roman: [roman::RomanTrie; 2] = {
            let mut avro = roman::RomanTrie::empty();
            let mut khipro = roman::RomanTrie::empty();
            for slot in [&mut avro, &mut khipro] {
                let (parsed, _) = roman::RomanTrie::read_from(bytes, &mut cursor)?;
                *slot = parsed;
            }
            let word_count = {
                let chunk = bytes
                    .get(cursor..cursor + 4)
                    .ok_or(BuildError::InvalidIndex("truncated word count"))?;
                cursor += 4;
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize
            };
            let mut word_off = Vec::with_capacity(word_count);
            for _ in 0..word_count {
                let chunk = bytes
                    .get(cursor..cursor + 4)
                    .ok_or(BuildError::InvalidIndex("truncated word offsets"))?;
                cursor += 4;
                word_off.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }

            let wl = {
                let chunk = bytes
                    .get(cursor..cursor + 4)
                    .ok_or(BuildError::InvalidIndex("truncated words len"))?;
                cursor += 4;
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize
            };
            let end = cursor
                .checked_add(wl)
                .ok_or(BuildError::InvalidIndex("words overflow"))?;
            let words = bytes
                .get(cursor..end)
                .ok_or(BuildError::InvalidIndex("truncated words"))?
                .to_vec();
            cursor = end;
            // Trailer checksum (Phase 6 integrity): FNV-1a of everything
            // before it. Older blobs without a trailer stay readable.
            if let Some(checksum) = bytes.get(cursor..cursor + 8) {
                let expect = if let Ok(arr) = checksum.try_into() {
                    u64::from_le_bytes(arr)
                } else {
                    return Err(BuildError::InvalidIndex("bad checksum slice"));
                };
                let computed = fnv1a(&bytes[..cursor]);
                if expect != computed {
                    return Err(BuildError::InvalidIndex("blob checksum mismatch"));
                }
            }
            avro.set_words(words.clone(), word_off.clone());
            khipro.set_words(words, word_off);
            [avro, khipro]
        };

        Ok(Self {
            trie,
            roman,
            graphemes,
            grapheme_ids,
            specials,
            word_count,
            max_dict_len,
            max_freq,
            config: CorrectionConfig::default(),
            edit_costs,
            language_model: None,
            trigram_language_model: None,
        })
    }

    pub fn with_config(mut self, mut config: CorrectionConfig) -> Self {
        if !config.frequency_weight.is_finite() {
            config.frequency_weight = 0.0;
        }
        if !config.language_model_weight.is_finite() {
            config.language_model_weight = 0.0;
        }
        config.frequency_weight = config.frequency_weight.max(0.0);
        config.language_model_weight = config.language_model_weight.max(0.0);
        self.config = config;
        self
    }

    /// Number of distinct normalized surface forms in the dictionary.
    pub fn len(&self) -> usize {
        self.word_count
    }

    pub fn is_empty(&self) -> bool {
        self.word_count == 0
    }

    /// Whether the normalized word occurs in the dictionary.
    pub fn contains(&self, word: &str) -> bool {
        let normalized = normalize_cow(word.trim());
        let mut node = 0u32;
        let mut any = false;
        for grapheme in UnicodeSegmentation::graphemes(normalized.as_ref(), true) {
            any = true;
            let Some(&id) = self.grapheme_ids.get(grapheme) else {
                return false;
            };
            let Some(next) = self.trie.child(node, id) else {
                return false;
            };
            node = next;
        }
        any && self.trie.term[node as usize] > 0
    }

    fn decode_path(&self, path: &[u16]) -> Option<String> {
        let len: usize = path
            .iter()
            .map(|id| {
                self.graphemes
                    .get(*id as usize)
                    .map(|g| g.len())
                    .unwrap_or(0)
            })
            .sum();
        let mut word = String::with_capacity(len);
        for &id in path {
            word.push_str(self.graphemes.get(id as usize)?);
        }
        Some(word)
    }

    fn score(&self, edit_cost: u16, frequency: u64, context_probability: Option<f64>) -> f64 {
        let edit_score = edit_cost as f64 / 100.0;
        let frequency_score = frequency_score(frequency, self.config.frequency_weight);
        let context_score = context_probability
            .map(|probability| self.config.language_model_weight * probability.max(1e-12).ln())
            .unwrap_or(0.0);
        edit_score - frequency_score - context_score
    }

    /// Complete an exact Bengali prefix using the grapheme trie.
    /// Results are ranked by dictionary frequency, then spelling.
    /// Empty prefixes intentionally return no results.
    pub fn complete_prefix(&self, prefix: &str, limit: usize) -> Vec<String> {
        if limit == 0 || prefix.trim().is_empty() {
            return Vec::new();
        }

        let normalized = normalize_cow(prefix.trim());
        let query: Vec<&str> = UnicodeSegmentation::graphemes(normalized.as_ref(), true).collect();
        if query.is_empty() {
            return Vec::new();
        }
        let n = query.len();
        let mut node = 0u32;
        let mut prefix_path: Vec<u16> = Vec::with_capacity(n.saturating_sub(1));
        for &grapheme in &query[..n - 1] {
            let Some(&id) = self.grapheme_ids.get(grapheme) else {
                return Vec::new();
            };
            let Some(next) = self.trie.child(node, id) else {
                return Vec::new();
            };
            node = next;
            prefix_path.push(id);
        }
        let last = query[n - 1];
        let mut best = BinaryHeap::<(Reverse<u64>, String)>::new();

        // DFS over the subtree; depth == n - 1 + relative depth.
        struct PrefixFrame {
            node: u32,
            next_child: u32,
            visited: bool,
        }
        let mut stack = vec![PrefixFrame {
            node,
            next_child: 0,
            visited: false,
        }];
        let mut path = prefix_path;
        while let Some(frame) = stack.last_mut() {
            if !frame.visited {
                frame.visited = true;
                // Prune subtrees that cannot beat the heap worst.
                if best.len() == limit
                    && let Some((Reverse(worst_freq), _)) = best.peek()
                {
                    let worst_freq = *worst_freq;
                    if self.trie.mx[frame.node as usize] < worst_freq {
                        stack.pop();
                        if path.len() > n - 1 {
                            path.pop();
                        }
                        continue;
                    }
                }
                let depth = path.len();
                if depth >= n && self.trie.term[frame.node as usize] > 0 {
                    let freq = self.trie.term[frame.node as usize];
                    let candidate_word: Option<String> = if best.len() < limit {
                        self.decode_path(&path)
                    } else if let Some((worst, _)) = best.peek() {
                        let worst_freq = worst.0;
                        if freq < worst_freq {
                            None
                        } else if freq > worst_freq {
                            self.decode_path(&path)
                        } else {
                            let word = self.decode_path(&path);
                            let wins = match (&word, best.peek()) {
                                (Some(word), Some((_, worst_word))) => word < worst_word,
                                _ => false,
                            };
                            if wins { word } else { None }
                        }
                    } else {
                        self.decode_path(&path)
                    };
                    if let Some(word) = candidate_word {
                        let candidate = (Reverse(freq), word);
                        if best.len() < limit {
                            best.push(candidate);
                        } else if best
                            .peek()
                            .map(|worst| candidate.cmp(worst).is_lt())
                            .unwrap_or(true)
                        {
                            best.pop();
                            best.push(candidate);
                        }
                    }
                }
            }
            let (start, cnt) = self.trie.children(frame.node);
            if (frame.next_child as usize) >= cnt {
                // Leaving a node must also drop the path entry it pushed.
                // The root frame's path is the mandatory prefix (n-1
                // graphemes) and must survive.
                stack.pop();
                if path.len() > n - 1 {
                    path.pop();
                }
                continue;
            }
            let edge = start + frame.next_child as usize;
            frame.next_child += 1;
            let cand = self.trie.eid[edge];
            let child = self.trie.ech[edge];
            let depth = path.len();
            // The candidate is the nth grapheme exactly when the current
            // path holds the first n-1 graphemes.
            if depth == n - 1 && !self.graphemes[cand as usize].starts_with(last) {
                continue;
            }
            path.push(cand);
            stack.push(PrefixFrame {
                node: child,
                next_child: 0,
                visited: false,
            });
        }

        best.into_sorted_vec()
            .into_iter()
            .map(|(_, word)| word)
            .collect()
    }

    pub fn with_edit_costs(mut self, edit_costs: EditCosts) -> Self {
        self.specials = build_specials(&self.graphemes, &self.grapheme_ids, &edit_costs);
        self.edit_costs = edit_costs;
        self
    }

    pub fn with_language_model(mut self, model: BigramLanguageModel) -> Self {
        self.language_model = Some(model);
        self
    }

    /// Add the trigram tier (uses the last two context words; falls back to
    /// the bigram tier for shorter context).
    pub fn with_trigram_language_model(mut self, model: TrigramLanguageModel) -> Self {
        self.trigram_language_model = Some(model);
        self
    }

    pub fn config(&self) -> &CorrectionConfig {
        &self.config
    }

    /// Unified dual-input suggestion: Bangla typo trie + Roman index.
    ///
    /// `roman: None` (or `InputMode::Bangla`) behaves like the pure Bangla
    /// API. Otherwise each candidate is scored as
    /// `edit/100 + roman_weight * roman/100 - frequency - context`, lower is
    /// better: the two channels are correlated evidence, so their costs add
    /// with a calibrated weight rather than multiplying as independent
    /// probabilities. Words found only through the Roman index get their
    /// exact Bangla edit cost; Bangla candidates the Roman search did not
    /// reach are charged `max_roman_cost + 1`, a lower bound on their true
    /// Roman cost, so a missing Roman match never beats a found one.
    pub fn suggest_input(&self, query: &SuggestionInput<'_>, limit: usize) -> Vec<Suggestion> {
        let bangla_hits = self.correct_with_context(query.bangla, query.context, limit.max(20));
        let roman_trie = match query.mode {
            InputMode::Avro => Some(&self.roman[0]),
            InputMode::Khipro => Some(&self.roman[1]),
            InputMode::Bangla => None,
        };
        let (Some(roman), Some(trie)) = (query.roman, roman_trie) else {
            return bangla_hits
                .into_iter()
                .take(limit)
                .map(|c| Suggestion {
                    word: c.word,
                    score: c.score,
                    edit_cost: c.edit_cost,
                    roman_cost: None,
                    frequency: c.frequency,
                    context_probability: c.context_probability,
                })
                .collect();
        };
        if limit == 0 || self.config.roman_weight <= 0.0 {
            return self.suggest_input(
                &SuggestionInput {
                    roman: None,
                    ..*query
                },
                limit,
            );
        }

        let mut candidates: HashMap<String, MergedCandidate> = bangla_hits
            .into_iter()
            .map(|c| {
                let merged = (c.edit_cost, None, c.frequency, c.context_probability);
                (c.word, merged)
            })
            .collect();
        let costs = roman::RomanCosts {
            insertion: 100,
            deletion: 100,
            transposition: 65,
        };
        let hits = roman::roman_search(trie, roman, self.config.max_roman_cost, &costs);
        let normalized = normalize_cow(query.bangla.trim());
        let query_graphemes: Vec<String> =
            UnicodeSegmentation::graphemes(normalized.as_ref(), true)
                .map(str::to_owned)
                .collect();
        for (word, roman_cost) in hits {
            let entry = candidates.entry(word).or_insert_with_key(|word| {
                // Found only via the Roman path: score its Bangla side exactly.
                let graphemes: Vec<String> = UnicodeSegmentation::graphemes(word.as_str(), true)
                    .map(str::to_owned)
                    .collect();
                let edit_cost =
                    weighted_osa_distance(&query_graphemes, &graphemes, &self.edit_costs);
                (
                    edit_cost,
                    None,
                    self.frequency_of(word),
                    self.context_probability(query.context, word),
                )
            });
            // A word can have several Roman forms; keep its best match.
            entry.1 = Some(entry.1.map_or(roman_cost, |best| best.min(roman_cost)));
        }

        let missing_roman_cost = self.config.max_roman_cost.saturating_add(1);
        let mut out: Vec<Suggestion> = candidates
            .into_iter()
            .map(
                |(word, (edit_cost, roman_cost, frequency, context_probability))| {
                    let roman_score = self.config.roman_weight
                        * (roman_cost.unwrap_or(missing_roman_cost) as f64 / 100.0);
                    Suggestion {
                        score: self.score(edit_cost, frequency, context_probability) + roman_score,
                        word,
                        edit_cost,
                        roman_cost,
                        frequency,
                        context_probability,
                    }
                },
            )
            .collect();
        // The typed Roman string is the strongest signal: rank by Roman
        // evidence first. When two words share the same Roman cost (identical
        // spelling), the lexical prior decides — a common word beats a rare
        // homophone — and the combined score breaks any remaining tie.
        out.sort_by(|a, b| {
            let left = a.roman_cost.unwrap_or(missing_roman_cost);
            let right = b.roman_cost.unwrap_or(missing_roman_cost);
            left.cmp(&right)
                .then(b.frequency.cmp(&a.frequency))
                .then(a.score.total_cmp(&b.score))
                .then(a.word.cmp(&b.word))
        });
        out.truncate(limit);
        out
    }

    /// Dictionary frequency of a normalized word, or 1 when it is absent.
    fn frequency_of(&self, word: &str) -> u64 {
        let mut node = 0u32;
        for grapheme in UnicodeSegmentation::graphemes(word, true) {
            let next = self
                .grapheme_ids
                .get(grapheme)
                .and_then(|&id| self.trie.child(node, id));
            let Some(next) = next else {
                return 1;
            };
            node = next;
        }
        match self.trie.term[node as usize] {
            0 => 1,
            frequency => frequency,
        }
    }

    /// Language-model probability of `word` after `context`: the trigram
    /// tier when two context words are available, else the bigram tier.
    fn context_probability(&self, context: &[&str], word: &str) -> Option<f64> {
        if let (Some(model), [.., p2, p1]) = (&self.trigram_language_model, context) {
            return Some(model.probability(
                &normalize_cow(p2.trim()),
                &normalize_cow(p1.trim()),
                word,
            ));
        }
        match (&self.language_model, context.last()) {
            (Some(model), Some(previous)) => {
                Some(model.probability_normalized(&normalize_cow(previous.trim()), word))
            }
            _ => None,
        }
    }

    /// Return ranked corrections using dictionary frequency as the lexical
    /// prior. The input is NFC-normalized before lookup.
    pub fn suggest(&self, input: &str) -> Vec<String> {
        self.suggest_with(input, self.config.max_results)
    }

    pub fn suggest_with(&self, input: &str, limit: usize) -> Vec<String> {
        self.correct_with_context(input, &[], limit)
            .into_iter()
            .map(|candidate| candidate.word)
            .collect()
    }

    /// Return ranked corrections. `context` should contain the preceding words
    /// in order; the bigram model uses the final item as the history word.
    pub fn correct_with_context(
        &self,
        input: &str,
        context: &[&str],
        limit: usize,
    ) -> Vec<Correction> {
        if limit == 0 || input.trim().is_empty() || self.graphemes.is_empty() {
            return Vec::new();
        }

        let normalized = normalize_cow(input.trim());
        // Stack query buffers: no heap allocation per lookup.
        let mut buf_slices = [""; automaton::MAX_QUERY];
        let mut raw_count: usize = 0;
        for g in UnicodeSegmentation::graphemes(normalized.as_ref(), true) {
            if raw_count < automaton::MAX_QUERY {
                buf_slices[raw_count] = g;
            }
            raw_count += 1;
        }
        if raw_count == 0 {
            return Vec::new();
        }

        // Fast reject: no dictionary word is long enough to be within the
        // edit budget when insertion/deletion are costly.
        let min_step = self.edit_costs.insertion().min(self.edit_costs.deletion());
        if min_step > 0 && self.max_dict_len > 0 {
            let over = raw_count.saturating_sub(self.max_dict_len) as u64;
            let min_cost_over = over.saturating_mul(min_step as u64);
            if min_cost_over > self.config.max_edit_cost as u64 {
                return Vec::new();
            }
        }

        let query_len = raw_count.min(automaton::MAX_QUERY);
        let truncated = raw_count > query_len;
        // When truncation actually splits a longer query, treat it as
        // unmatched unless the part beyond the row would be free.
        if truncated && min_step == 0 {
            // Custom zero-cost edits cannot be decided with a fixed row;
            // fall back to the truncated query (behavior matches the
            // original fixed-row automaton for queries within range).
        }
        let query_slices: &[&str] = &buf_slices[..query_len];
        let mut query_ids = [None::<u16>; automaton::MAX_QUERY];
        for (i, &g) in query_slices.iter().enumerate() {
            query_ids[i] = self.grapheme_ids.get(g).copied();
        }
        // Per-position specials: borrowed for in-vocabulary positions, owned
        // for out-of-vocabulary positions.
        let mut owned_by_pos: [Option<Vec<(u16, u16)>>; automaton::MAX_QUERY] =
            std::array::from_fn(|_| None);
        for i in 0..query_len {
            if query_ids[i].is_none() {
                owned_by_pos[i] = Some(specials_for_oov(
                    query_slices[i],
                    &self.graphemes,
                    &self.grapheme_ids,
                    &self.edit_costs,
                ));
            }
        }
        let mut pos_buf: [&[(u16, u16)]; automaton::MAX_QUERY] = [&[]; automaton::MAX_QUERY];
        for (i, &qid) in query_ids[..query_len].iter().enumerate() {
            pos_buf[i] = match qid {
                Some(id) => &self.specials[id as usize],
                None => owned_by_pos[i].as_deref().unwrap_or(&[]),
            };
        }
        let query_ids: &[Option<u16>] = &query_ids[..query_len];
        let pos_specials: &[&[(u16, u16)]] = &pos_buf[..query_len];

        let insertion = self.edit_costs.insertion();
        let deletion = self.edit_costs.deletion();
        let base = self.edit_costs.base_substitution();
        let transposition = self.edit_costs.transposition();
        let max_cost = self.config.max_edit_cost;
        let weight = self.config.frequency_weight;

        let previous_word = context.last().map(|word| normalize_cow(word.trim()));
        let mut best = BinaryHeap::<HeapCandidate>::new();
        let mut threshold: u16 = max_cost;
        let log_max_freq = if self.max_freq > 1 {
            (self.max_freq as f64).ln()
        } else {
            0.0
        };

        let recompute_threshold = |best: &BinaryHeap<HeapCandidate>| -> u16 {
            if best.len() < limit {
                return max_cost;
            }
            let worst = best.peek().map(|worst| &worst.0);
            let Some(worst) = worst else {
                return max_cost;
            };
            // Candidates that could displace `worst` need
            // edit/100 - w*ln(max_freq) <= worst.score, i.e.
            // edit <= 100 * (worst.score + w*ln(max_freq)).
            let bound = 100.0 * (worst.score + weight * log_max_freq);
            if bound <= 0.0 {
                return 0;
            }
            let t = bound.floor() as i64 + 1;
            if t > max_cost as i64 {
                max_cost
            } else {
                t.max(0) as u16
            }
        };

        // Exact-match fast path: also seeds the heap so threshold pruning
        // is effective from the start.
        let query_path: Option<Vec<u16>> = query_ids.iter().copied().collect();
        if let Some(path) = &query_path
            && let Some(node) = self.trie.walk(path)
        {
            let freq = self.trie.term[node as usize];
            if freq > 0 {
                let edit_cost = 0u16;
                let word = normalized.as_ref().to_owned();
                // Truncated queries cannot be exact dictionary words;
                // the trie walk already guarantees the match, but keep
                // the untruncated string only when no truncation happened.
                let word = if truncated {
                    // The walked path is the truncated prefix; decode it.
                    self.decode_path(path).unwrap_or(word)
                } else {
                    word
                };
                let previous_two: Option<(String, String)> = (context.len() >= 2).then(|| {
                    (
                        normalize_cow(context[context.len() - 2].trim()).into_owned(),
                        normalize_cow(context[context.len() - 1].trim()).into_owned(),
                    )
                });
                let context_probability = match (&self.trigram_language_model, &previous_two) {
                    (Some(model), Some((p2, p1))) => Some(model.probability(p2, p1, &word)),
                    _ => match (&self.language_model, previous_word.as_deref()) {
                        (Some(model), Some(previous)) => {
                            Some(model.probability_normalized(previous, &word))
                        }
                        _ => None,
                    },
                };
                let score = self.score(edit_cost, freq, context_probability);
                if limit == 1 {
                    return vec![Correction {
                        word,
                        edit_cost,
                        frequency: freq,
                        score,
                        context_probability,
                    }];
                }
                best.push(HeapCandidate(Correction {
                    word,
                    edit_cost,
                    frequency: freq,
                    score,
                    context_probability,
                }));
                threshold = recompute_threshold(&best);
            }
        }

        // Depth-first edit search over the trie.
        //
        // DP rows live in a small arena indexed by frame, truncated on
        // backtrack, so expanding an edge never copies two full rows into a
        // new frame. A frame whose row minimum exceeds the threshold can
        // only survive through a transposition-completing child, which
        // `can_match`/`recovers` test from the two-back row.
        const NO_ROW: usize = usize::MAX;
        struct Frame {
            node: u32,
            next_child: u32,
            row_idx: usize,
            two_idx: usize,
            row_min: u16,
            prev_id: u16,
            depth: usize,
            visited: bool,
        }
        let (root_row, root_min) = automaton::start_row(deletion, query_len);
        let mut rows: Vec<automaton::Row> = vec![root_row];
        let mut stack = vec![Frame {
            node: 0,
            next_child: 0,
            row_idx: 0,
            two_idx: NO_ROW,
            row_min: root_min,
            prev_id: 0,
            depth: 0,
            visited: false,
        }];
        let mut path: Vec<u16> = Vec::new();

        let can_match_row =
            |row_min: u16, two: &automaton::Row, prev: u16, threshold: u16| -> bool {
                if row_min <= threshold {
                    return true;
                }
                for i in 2..=query_len {
                    if query_ids[i - 1] == Some(prev)
                        && two[i - 2].saturating_add(transposition) <= threshold
                    {
                        return true;
                    }
                }
                false
            };
        let recovers_row = |two: &automaton::Row, prev: u16, cand: u16, threshold: u16| -> bool {
            for i in 2..=query_len {
                if query_ids[i - 1] == Some(prev)
                    && query_ids[i - 2] == Some(cand)
                    && two[i - 2].saturating_add(transposition) <= threshold
                {
                    return true;
                }
            }
            false
        };

        while let Some(frame) = stack.last_mut() {
            if !frame.visited {
                frame.visited = true;
                let live = if frame.row_min <= threshold {
                    true
                } else if frame.two_idx == NO_ROW {
                    false
                } else {
                    can_match_row(
                        frame.row_min,
                        &rows[frame.two_idx],
                        frame.prev_id,
                        threshold,
                    )
                };
                if !live {
                    let idx = frame.row_idx;
                    stack.pop();
                    rows.truncate(idx);
                    if idx > 0 {
                        path.pop();
                    }
                    continue;
                }
                // Frequency-aware subtree bound: even with the subtree's
                // most frequent word, this branch cannot displace the
                // heap worst. Upper-bounded log via floor-log2 bins keeps
                // the check to a couple of flops with no `ln` call.
                if best.len() == limit
                    && let Some(worst) = best.peek().map(|worst| &worst.0)
                {
                    let mx = self.trie.mx[frame.node as usize];
                    let upper_ln = if mx <= 1 {
                        0.0
                    } else {
                        ((64 - mx.leading_zeros()) as f64) * std::f64::consts::LN_2
                    };
                    let allowed = (100.0 * (worst.score + weight * upper_ln)).floor() as i64 + 1;
                    let mut lb_edit = frame.row_min;
                    if frame.two_idx != NO_ROW {
                        let two = &rows[frame.two_idx];
                        for i in 2..=query_len {
                            if query_ids[i - 1] == Some(frame.prev_id) {
                                let c = two[i - 2].saturating_add(transposition);
                                if c < lb_edit {
                                    lb_edit = c;
                                }
                            }
                        }
                    }
                    if (lb_edit as i64) > allowed {
                        let idx = frame.row_idx;
                        stack.pop();
                        rows.truncate(idx);
                        if idx > 0 {
                            path.pop();
                        }
                        continue;
                    }
                }
                let depth = path.len();
                if depth > 0 && self.trie.term[frame.node as usize] > 0 {
                    let freq = self.trie.term[frame.node as usize];
                    let edit_cost = rows[frame.row_idx][query_len];
                    if !(query_path
                        .as_ref()
                        .map(|qp| qp.len() == depth && qp.iter().zip(&path).all(|(a, b)| a == b))
                        .unwrap_or(false))
                        && edit_cost <= threshold
                    {
                        // Approximate score without the language model
                        // for heap pruning without decoding.
                        let approx = edit_cost as f64 / 100.0 - frequency_score(freq, weight);
                        let mut keep = best.len() < limit;
                        if !keep && let Some(worst) = best.peek().map(|worst| &worst.0) {
                            if approx < worst.score - 1e-9 {
                                keep = true;
                            } else if (approx - worst.score).abs() <= 1e-9 {
                                keep = freq > worst.frequency
                                    || (freq == worst.frequency && best.peek().is_some());
                            }
                        }
                        if keep && let Some(word) = self.decode_path(&path) {
                            let context_probability =
                                match (&self.language_model, previous_word.as_deref()) {
                                    (Some(model), Some(previous)) => {
                                        Some(model.probability_normalized(previous, &word))
                                    }
                                    _ => None,
                                };
                            let score = self.score(edit_cost, freq, context_probability);
                            let candidate = HeapCandidate(Correction {
                                word,
                                edit_cost,
                                frequency: freq,
                                score,
                                context_probability,
                            });
                            if best.len() < limit {
                                best.push(candidate);
                                threshold = recompute_threshold(&best);
                            } else if let Some(worst) = best.peek().map(|worst| &worst.0)
                                && compare_corrections(&candidate.0, worst).is_lt()
                            {
                                best.pop();
                                best.push(candidate);
                                threshold = recompute_threshold(&best);
                            }
                            // With a single slot the exact match is
                            // final regardless of score.
                            if limit == 1
                                && best
                                    .peek()
                                    .map(|worst| worst.0.edit_cost == 0)
                                    .unwrap_or(false)
                            {
                                break;
                            }
                        }
                    }
                }
            }

            let (start, cnt) = self.trie.children(frame.node);
            if (frame.next_child as usize) >= cnt {
                let idx = frame.row_idx;
                stack.pop();
                rows.truncate(idx);
                if idx > 0 {
                    path.pop();
                }
                continue;
            }
            let edge = start + frame.next_child as usize;
            frame.next_child += 1;
            let cand = self.trie.eid[edge];
            let child = self.trie.ech[edge];
            let parent_idx = frame.row_idx;
            let parent_two_idx = frame.two_idx;
            let parent_prev = frame.prev_id;
            let parent_depth = frame.depth;
            let has_link = parent_depth > 0;
            // Recovery-only parents keep only transposition children.
            if has_link
                && frame.row_min > threshold
                && !recovers_row(&rows[parent_two_idx], parent_prev, cand, threshold)
            {
                continue;
            }
            let two_ref: Option<&automaton::Row> = if has_link {
                Some(&rows[parent_two_idx])
            } else {
                None
            };
            let (row, min) = automaton::advance_row(automaton::RowAdvance {
                parent: &rows[parent_idx],
                two_back: two_ref,
                prev_candidate: has_link.then_some(parent_prev),
                candidate: cand,
                query_len,
                query_ids,
                query_slices,
                specials: pos_specials,
                graphemes: &self.graphemes,
                insertion,
                deletion,
                base,
                transposition,
            });
            if !can_match_row(min, &rows[parent_idx], cand, threshold) {
                continue;
            }
            let child_idx = rows.len();
            rows.push(row);
            path.push(cand);
            stack.push(Frame {
                node: child,
                next_child: 0,
                row_idx: child_idx,
                two_idx: parent_idx,
                row_min: min,
                prev_id: cand,
                depth: parent_depth + 1,
                visited: false,
            });
        }

        let mut candidates: Vec<Correction> =
            best.into_iter().map(|candidate| candidate.0).collect();
        candidates.sort_unstable_by(compare_corrections);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::CorrectionConfig;
    use super::super::costs::EditCosts;
    use super::super::morphology::{Morphology, SuffixRule};
    use super::super::types::WordEntry;
    use super::*;
    use simd_normalizer::UnicodeNormalization;
    use std::collections::BTreeSet;
    use unicode_segmentation::UnicodeSegmentation;

    fn test_corrector() -> Corrector {
        Corrector::from_entries([
            WordEntry::new("বাংলা", 100),
            WordEntry::new("বাঙলা", 15),
            WordEntry::new("মানুষ", 90),
            WordEntry::new("মানুষগুলো", 40),
            WordEntry::new("কম্পিউটার", 80),
            WordEntry::new("কম্পিউটিং", 25),
            WordEntry::new("শিক্ষা", 70),
        ])
        .unwrap()
    }

    #[test]
    fn empty_dictionary_is_valid_and_safe() {
        let corrector = Corrector::from_words(std::iter::empty::<&str>()).unwrap();
        assert!(corrector.is_empty());
        assert_eq!(corrector.len(), 0);
        assert!(corrector.suggest("বাংলা").is_empty());
        assert!(corrector.complete_prefix("ব", 10).is_empty());
    }

    #[test]
    fn duplicate_normalized_entries_merge_their_frequencies() {
        let corrector =
            Corrector::from_entries([WordEntry::new("বাংলা", 2), WordEntry::new("বাংলা", 3)]).unwrap();
        let result = corrector.correct_with_context("বাংলা", &[], 1);
        assert_eq!(result[0].frequency, 5);
    }

    #[test]
    fn exact_word_is_first() {
        let corrector = test_corrector();
        let results = corrector.correct_with_context("বাংলা", &[], 5);
        assert_eq!(results.first().unwrap().word, "বাংলা");
        assert_eq!(results.first().unwrap().edit_cost, 0);
    }

    #[test]
    fn exact_word_stays_first_even_when_another_word_is_more_frequent() {
        let corrector =
            Corrector::from_entries([WordEntry::new("বাংলা", 1), WordEntry::new("বাঙলা", 1_000_000)])
                .unwrap();
        assert_eq!(corrector.suggest_with("বাংলা", 2)[0], "বাংলা".to_owned());
    }

    #[test]
    fn contains_uses_the_same_normalization_as_indexing() {
        let corrector = Corrector::from_words(["বাংলা"]).unwrap();
        assert!(corrector.contains("বাংলা"));
        assert!(!corrector.contains("ইংরেজি"));
    }

    #[test]
    fn prefix_completion_uses_the_same_dictionary() {
        let corrector = Corrector::from_entries([
            WordEntry::new("বাংলা", 10),
            WordEntry::new("বাংলাদেশ", 100),
            WordEntry::new("বাঙালি", 20),
            WordEntry::new("মানুষ", 50),
        ])
        .unwrap();

        assert_eq!(
            corrector.complete_prefix("বাংলা", 10),
            vec!["বাংলাদেশ".to_owned(), "বাংলা".to_owned()]
        );
        assert!(corrector.complete_prefix("", 10).is_empty());
        assert!(corrector.complete_prefix("অজানা", 10).is_empty());

        let partial_grapheme = Corrector::from_words(["কাল", "কী", "কখন"]).unwrap();
        let results = partial_grapheme.complete_prefix("ক", 10);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn finds_single_grapheme_typo() {
        let corrector = test_corrector();
        let results = corrector.suggest_with("বাংরা", 5);
        assert!(results.iter().any(|word| word == "বাংলা"));
    }

    #[test]
    fn fst_search_matches_brute_force_osa_for_short_words() {
        fn generate(words: &mut Vec<String>, prefix: &mut String, alphabet: &[&str], max: usize) {
            if !prefix.is_empty() {
                words.push(prefix.clone());
            }
            if UnicodeSegmentation::graphemes(prefix.as_str(), true).count() == max {
                return;
            }
            for token in alphabet {
                let old_len = prefix.len();
                prefix.push_str(token);
                generate(words, prefix, alphabet, max);
                prefix.truncate(old_len);
            }
        }

        let mut words = Vec::new();
        generate(&mut words, &mut String::new(), &["ক", "খ", "গ"], 3);
        words.sort();
        words.dedup();

        let thresholds = [0, 35, 65, 100, 150, 200];
        for threshold in thresholds {
            let config = CorrectionConfig {
                max_edit_cost: threshold,
                ..CorrectionConfig::default()
            };
            let corrector = Corrector::from_words(words.clone())
                .unwrap()
                .with_config(config);

            for query_word in &words {
                let query: Vec<String> = UnicodeSegmentation::graphemes(query_word.as_str(), true)
                    .map(str::to_owned)
                    .collect();
                let expected: BTreeSet<String> = words
                    .iter()
                    .filter(|candidate| {
                        let candidate_graphemes: Vec<String> =
                            UnicodeSegmentation::graphemes(candidate.as_str(), true)
                                .map(str::to_owned)
                                .collect();
                        weighted_osa_distance(&query, &candidate_graphemes, &EditCosts::default())
                            <= threshold
                    })
                    .cloned()
                    .collect();
                let actual: BTreeSet<String> = corrector
                    .correct_with_context(query_word, &[], words.len())
                    .into_iter()
                    .map(|candidate| candidate.word)
                    .collect();

                assert_eq!(
                    actual, expected,
                    "query={query_word}, threshold={threshold}"
                );
            }
        }
    }

    #[test]
    fn applies_symmetric_bengali_confusion_costs() {
        let costs = EditCosts::bengali();
        let query = vec!["শ".to_owned()];
        let candidate = vec!["ষ".to_owned()];
        assert_eq!(weighted_osa_distance(&query, &candidate, &costs), 35);
    }

    #[test]
    fn confusion_applies_to_graphemes_with_the_same_vowel_mark() {
        let costs = EditCosts::bengali();
        let query = vec!["শি".to_owned()];
        let candidate = vec!["ষি".to_owned()];
        assert_eq!(weighted_osa_distance(&query, &candidate, &costs), 35);
    }

    #[test]
    fn automaton_search_uses_bengali_base_confusions_inside_graphemes() {
        let config = CorrectionConfig {
            max_edit_cost: 35,
            ..CorrectionConfig::default()
        };
        let corrector = Corrector::from_words(["শি"]).unwrap().with_config(config);
        assert_eq!(corrector.suggest_with("ষি", 1), vec!["শি".to_owned()]);
    }

    #[test]
    fn dependent_vowel_signs_can_be_weighted() {
        let costs = EditCosts::bengali();
        let query = vec!["কি".to_owned()];
        let candidate = vec!["কী".to_owned()];
        assert_eq!(weighted_osa_distance(&query, &candidate, &costs), 60);
    }

    #[test]
    fn handles_adjacent_transposition() {
        let config = CorrectionConfig {
            max_edit_cost: 65,
            ..CorrectionConfig::default()
        };
        let corrector = Corrector::from_words(["কখগ"]).unwrap().with_config(config);
        let results = corrector.suggest_with("খকগ", 5);
        assert!(results.iter().any(|word| word == "কখগ"));
    }

    #[test]
    fn normalizes_canonically_equivalent_input() {
        let corrector = Corrector::from_words(["বাংলা"]).unwrap();
        let nfd = "বাংলা".nfd().into_owned();
        assert_eq!(corrector.suggest_with(&nfd, 1), vec!["বাংলা".to_owned()]);
    }

    #[test]
    fn morphology_forms_are_searchable() {
        let morphology = Morphology::new().with_rule(SuffixRule::append("গুলো"));
        let corrector =
            Corrector::from_entries_with_morphology([WordEntry::new("মানুষ", 90)], &morphology)
                .unwrap();
        let results = corrector.suggest_with("মানুষগুল", 5);
        assert!(results.iter().any(|word| word == "মানুষগুলো"));
    }

    #[test]
    fn empty_input_has_no_candidates() {
        let corrector = test_corrector();
        assert!(corrector.suggest("").is_empty());
    }

    fn roman_corrector() -> Corrector {
        Corrector::from_entries([
            WordEntry::new("সোনার", 60),
            WordEntry::new("উনার", 40),
            WordEntry::new("সবার", 200),
            WordEntry::new("আনার", 70),
            WordEntry::new("বাংলা", 100),
        ])
        .unwrap()
    }

    fn graphemes(word: &str) -> Vec<String> {
        UnicodeSegmentation::graphemes(word, true)
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn suggest_input_reports_real_edit_costs_for_roman_hits() {
        let corrector = roman_corrector();
        let query = SuggestionInput::with_roman("sonar", "সনার", InputMode::Avro);
        let results = corrector.suggest_input(&query, 10);
        assert!(results.iter().any(|s| s.roman_cost.is_some()));
        for suggestion in &results {
            assert_eq!(
                suggestion.edit_cost,
                weighted_osa_distance(
                    &graphemes("সনার"),
                    &graphemes(&suggestion.word),
                    &EditCosts::default()
                ),
                "{}",
                suggestion.word
            );
        }
    }

    #[test]
    fn suggest_input_roman_cost_is_a_penalty() {
        let query = SuggestionInput::with_roman("sonar", "সনার", InputMode::Avro);
        let with_weight = |roman_weight| {
            roman_corrector()
                .with_config(CorrectionConfig {
                    roman_weight,
                    ..CorrectionConfig::default()
                })
                .suggest_input(&query, 10)
        };
        let light = with_weight(1.0);
        let heavy = with_weight(2.0);
        for suggestion in &heavy {
            let baseline = light.iter().find(|s| s.word == suggestion.word).unwrap();
            assert!(
                suggestion.score >= baseline.score,
                "{}: a larger roman_weight must not improve the score",
                suggestion.word
            );
        }
        let lower_is_better = heavy
            .windows(2)
            .filter(|pair| pair[0].edit_cost == pair[1].edit_cost)
            .filter(|pair| pair[0].frequency == pair[1].frequency)
            .all(|pair| pair[0].roman_cost <= pair[1].roman_cost);
        assert!(lower_is_better);
    }

    #[test]
    fn suggest_input_without_roman_matches_bangla_api() {
        let corrector = roman_corrector();
        for query in [
            SuggestionInput::bangla_only("সনার"),
            SuggestionInput::with_roman("sonar", "সনার", InputMode::Bangla),
        ] {
            let words: Vec<String> = corrector
                .suggest_input(&query, 5)
                .into_iter()
                .map(|s| s.word)
                .collect();
            assert_eq!(words, corrector.suggest_with("সনার", 5));
        }
    }
}
