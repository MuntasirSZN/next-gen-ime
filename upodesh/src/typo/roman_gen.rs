// Offline Roman-form generation from machine-generated pattern tables.
//
// The tables (`data/avro-patterns.json`, `data/khipro-patterns.json`) are
// derived from okkhor/riti rules — no hand-written transliteration guesses.
// A Bangla word's plausible Roman forms are produced by tiling the word
// with transliterate chunks: a chunk appearing in a pattern's
// `transliterate` list can be spelled by any Roman key whose list contains
// it. Phonetic grounding: same-IPA spelling variants are cheap confusions
// (see `data/ipa-alignments.md`).
//
// Shared verbatim by `build.rs` (compile-time index) and `typo.rs`
// (`from_entries` runtime construction) via `include!`.

pub(crate) const MAX_FORMS_PER_WORD: usize = 2;

pub(crate) struct PatternSet {
    /// Transliterate prefix trie: byte edges, terminal Roman keys.
    nodes: Vec<TriNode>,
}

struct TriNode {
    next: Vec<(u8, u32)>,
    keys: Vec<String>,
}

pub(crate) fn load_patterns(bytes: &[u8]) -> PatternSet {
    let empty = || PatternSet {
        nodes: vec![TriNode {
            next: Vec::new(),
            keys: Vec::new(),
        }],
    };
    let Ok(root) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return empty();
    };
    let Some(obj) = root.as_object() else {
        return empty();
    };
    let mut rev_map: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (roman, block) in obj {
        let Some(list) = block.get("transliterate").and_then(|v| v.as_array()) else {
            continue;
        };
        let key = roman.to_ascii_lowercase();
        for entry in list.iter().filter_map(|v| v.as_str()) {
            if entry.is_empty() {
                continue;
            }
            rev_map
                .entry(entry.to_owned())
                .or_default()
                .push(key.clone());
        }
    }
    let mut nodes: Vec<TriNode> = vec![TriNode {
        next: Vec::new(),
        keys: Vec::new(),
    }];
    for (transliterate, keys) in &rev_map {
        let mut node = 0usize;
        for &b in transliterate.as_bytes() {
            let pos = nodes[node].next.iter().position(|(label, _)| *label == b);
            match pos {
                Some(p) => node = nodes[node].next[p].1 as usize,
                None => {
                    nodes.push(TriNode {
                        next: Vec::new(),
                        keys: Vec::new(),
                    });
                    let child = (nodes.len() - 1) as u32;
                    if let Some(n) = nodes.get_mut(node) {
                        n.next.push((b, child));
                    }
                    node = child as usize;
                }
            }
        }
        if let Some(n) = nodes.get_mut(node) {
            n.keys.extend(keys.iter().cloned());
        }
    }
    PatternSet { nodes }
}

impl PatternSet {
    /// Roman forms for one Bangla word by chunk tiling.
    pub(crate) fn forms(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let mut dp: Vec<Vec<String>> = vec![Vec::new(); chars.len() + 1];
        dp[0] = vec![String::new()];
        for pi in 0..chars.len() {
            if dp[pi].is_empty() {
                continue;
            }
            let mut byte_pos = 0usize;
            for c in chars[..pi].iter() {
                byte_pos += c.len_utf8();
            }
            let suffix = &word[byte_pos..];
            let mut node = 0usize;
            let mut blen = 0usize;
            for &b in suffix.as_bytes() {
                let pos = self.nodes[node]
                    .next
                    .iter()
                    .position(|(label, _)| *label == b);
                let Some(p) = pos else { break };
                node = self.nodes[node].next[p].1 as usize;
                // Walk every byte; only record matches on char boundaries,
                // since chunk keys are whole UTF-8 strings.
                blen += 1;
                if !word.is_char_boundary(byte_pos + blen) {
                    continue;
                }
                if self.nodes[node].keys.is_empty() {
                    continue;
                }
                let mut acc = 0usize;
                let mut target = pi;
                while target < chars.len() {
                    acc += chars[target].len_utf8();
                    if acc == blen {
                        break;
                    }
                    if acc > blen {
                        target = usize::MAX;
                        break;
                    }
                    target += 1;
                }
                if target == usize::MAX || target >= chars.len() {
                    continue;
                }
                let next_idx = target + 1;
                for base in dp[pi].clone() {
                    for key in &self.nodes[node].keys {
                        if dp[next_idx].len() >= MAX_FORMS_PER_WORD {
                            break;
                        }
                        let mut form = base.clone();
                        form.push_str(key);
                        if !dp[next_idx].contains(&form) {
                            dp[next_idx].push(form);
                        }
                    }
                    if dp[next_idx].len() >= MAX_FORMS_PER_WORD {
                        break;
                    }
                }
            }
        }
        dp[chars.len()]
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
}
