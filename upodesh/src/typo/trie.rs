use std::cmp::Reverse;

/// Compressed sparse-row grapheme trie.
///
/// Each node stores its outgoing edges contiguously in `eid`/`ech` between
/// `start[node]` and `start[node] + cnt[node]`. Edge IDs are grapheme-table
/// indices; child order is by descending subtree max frequency (then ID) so
/// depth-first searches visit promising subtrees early and can prune on the
/// retained heap bound.
pub(crate) struct Trie {
    pub(crate) start: Vec<u32>,
    pub(crate) cnt: Vec<u32>,
    pub(crate) term: Vec<u64>,
    pub(crate) mx: Vec<u64>,
    pub(crate) eid: Vec<u16>,
    pub(crate) ech: Vec<u32>,
}

const NONE: u32 = u32::MAX;

struct TempNode {
    head: u32,
    tail: u32,
    cnt: u32,
    term: u64,
    mx: u64,
    parent: u32,
}

struct TempEdge {
    id: u16,
    child: u32,
    next: u32,
}

/// Incremental builder for lexicographically sorted grapheme paths. Calling
/// [`TrieBuilder::add`] once per path in sorted order reuses one stack and
/// one previous-path buffer, avoiding materializing every path as a heap
/// allocation.
pub(crate) struct TrieBuilder {
    nodes: Vec<TempNode>,
    edges: Vec<TempEdge>,
    stack: Vec<u32>,
    prev_path: Vec<u16>,
}

impl TrieBuilder {
    pub(crate) fn new() -> Self {
        let mut nodes = Vec::with_capacity(4096);
        nodes.push(TempNode {
            head: NONE,
            tail: NONE,
            cnt: 0,
            term: 0,
            mx: 0,
            parent: NONE,
        });
        Self {
            nodes,
            edges: Vec::new(),
            stack: vec![0],
            prev_path: Vec::new(),
        }
    }

    pub(crate) fn add(&mut self, path: &[u16], freq: u64) {
        let mut lcp = 0usize;
        let limit = self.prev_path.len().min(path.len());
        while lcp < limit && self.prev_path[lcp] == path[lcp] {
            lcp += 1;
        }
        self.stack.truncate(lcp + 1);
        self.prev_path.truncate(lcp);
        for &id in &path[lcp..] {
            let Some(&parent) = self.stack.last() else {
                return;
            };
            let child = self.nodes.len() as u32;
            self.nodes.push(TempNode {
                head: NONE,
                tail: NONE,
                cnt: 0,
                term: 0,
                mx: 0,
                parent,
            });
            let edge_idx = self.edges.len() as u32;
            self.edges.push(TempEdge {
                id,
                child,
                next: NONE,
            });
            let p = parent as usize;
            if self.nodes[p].head == NONE {
                self.nodes[p].head = edge_idx;
            } else {
                let tail = self.nodes[p].tail as usize;
                self.edges[tail].next = edge_idx;
            }
            self.nodes[p].tail = edge_idx;
            self.nodes[p].cnt += 1;
            self.stack.push(child);
            self.prev_path.push(id);
        }
        let Some(&node) = self.stack.last() else {
            return;
        };
        self.nodes[node as usize].term = self.nodes[node as usize].term.saturating_add(freq);
        self.nodes[node as usize].mx = self.nodes[node as usize].mx.max(freq);
    }

    pub(crate) fn finish(self) -> Trie {
        Self::finish_parts(self.nodes, self.edges)
    }

    fn finish_parts(mut nodes: Vec<TempNode>, edges: Vec<TempEdge>) -> Trie {
        for idx in (1..nodes.len()).rev() {
            let parent = nodes[idx].parent as usize;
            let child_mx = nodes[idx].mx;
            if child_mx > nodes[parent].mx {
                nodes[parent].mx = child_mx;
            }
        }
        let n = nodes.len();
        let total = edges.len();
        let mut start = vec![0u32; n];
        let mut cnt = vec![0u32; n];
        let mut eid = vec![0u16; total];
        let mut ech = vec![0u32; total];
        let mut offset = 0usize;
        for (i, node) in nodes.iter().enumerate() {
            start[i] = offset as u32;
            cnt[i] = node.cnt;
            let mut e = node.head;
            while e != NONE {
                let edge = &edges[e as usize];
                eid[offset] = edge.id;
                ech[offset] = edge.child;
                offset += 1;
                e = edge.next;
            }
        }
        let mut trie = Trie {
            start,
            cnt,
            term: nodes.iter().map(|node| node.term).collect(),
            mx: nodes.iter().map(|node| node.mx).collect(),
            eid,
            ech,
        };
        // Reorder each node's edges by descending subtree maximum.
        let mut order: Vec<u32> = Vec::new();
        let mut old_eid: Vec<u16> = Vec::new();
        let mut old_ech: Vec<u32> = Vec::new();
        for node in 0..trie.start.len() {
            let s = trie.start[node] as usize;
            let c = trie.cnt[node] as usize;
            if c <= 1 {
                continue;
            }
            let end = s + c;
            order.clear();
            order.extend(0..c as u32);
            order.sort_unstable_by(|&a, &b| {
                let ca = trie.ech[s + a as usize];
                let cb = trie.ech[s + b as usize];
                Reverse(trie.mx[ca as usize])
                    .cmp(&Reverse(trie.mx[cb as usize]))
                    .then_with(|| trie.eid[s + a as usize].cmp(&trie.eid[s + b as usize]))
            });
            old_eid.clear();
            old_eid.extend_from_slice(&trie.eid[s..end]);
            old_ech.clear();
            old_ech.extend_from_slice(&trie.ech[s..end]);
            for (slot, &src) in order.iter().enumerate() {
                trie.eid[s + slot] = old_eid[src as usize];
                trie.ech[s + slot] = old_ech[src as usize];
            }
        }
        trie
    }
}

impl Trie {
    pub(crate) fn from_sorted_paths(paths: &mut [(Vec<u16>, u64)]) -> Self {
        let mut b = TrieBuilder::new();
        for (path, freq) in paths.iter() {
            b.add(path, *freq);
        }
        b.finish()
    }

    #[inline]
    pub(crate) fn child(&self, node: u32, id: u16) -> Option<u32> {
        let s = self.start[node as usize] as usize;
        let c = self.cnt[node as usize] as usize;
        for i in 0..c {
            if self.eid[s + i] == id {
                return Some(self.ech[s + i]);
            }
        }
        None
    }

    #[inline]
    pub(crate) fn walk(&self, path: &[u16]) -> Option<u32> {
        let mut node = 0u32;
        for &id in path {
            node = self.child(node, id)?;
        }
        Some(node)
    }

    #[inline]
    pub(crate) fn children(&self, node: u32) -> (usize, usize) {
        let s = self.start[node as usize] as usize;
        let c = self.cnt[node as usize] as usize;
        (s, c)
    }
}
