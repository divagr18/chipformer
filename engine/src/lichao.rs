//! Exact "max over lines at an integer point" cache for 2D hardmax heads.
//!
//! A head with query (q, 1) and keys (kx, ky) scores position s as
//! kx_s * q + ky_s: each key is a line and attention picks the highest line
//! at q. A Li Chao tree over the integer query domain answers that in
//! O(log range) per insert and per query, with exact i128 arithmetic.
//! The algorithm mirrors chipformer/dsl/evaluate.py:LiChao.

#[derive(Clone, Copy)]
struct Node {
    m: i64,
    b: i64,
    id: u32,
    left: u32, // 0 = none (node 0 is the root)
    right: u32,
}

pub struct LiChao {
    lo: i64,
    hi: i64,
    nodes: Vec<Node>,
    /// On equal scores prefer the lower id (argmax semantics). Attention heads never tie.
    low_id_wins: bool,
}

#[inline]
fn f(m: i64, b: i64, x: i64) -> i128 {
    m as i128 * x as i128 + b as i128
}

impl LiChao {
    pub fn new(lo: i64, hi: i64) -> Self {
        LiChao { lo, hi, nodes: Vec::new(), low_id_wins: false }
    }

    pub fn with_low_id_wins(lo: i64, hi: i64) -> Self {
        LiChao { lo, hi, nodes: Vec::new(), low_id_wins: true }
    }

    #[inline]
    fn better(&self, a: &Node, b: &Node, x: i64) -> bool {
        let (fa, fb) = (f(a.m, a.b, x), f(b.m, b.b, x));
        fa > fb || (self.low_id_wins && fa == fb && a.id < b.id)
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    pub fn insert(&mut self, m: i64, b: i64, id: u32) {
        let mut new = Node { m, b, id, left: 0, right: 0 };
        if self.nodes.is_empty() {
            self.nodes.push(new);
            return;
        }
        let (mut idx, mut lo, mut hi) = (0usize, self.lo, self.hi);
        loop {
            let cur = self.nodes[idx];
            let mid = (lo + hi).div_euclid(2);
            let left = self.better(&new, &cur, lo);
            let center = self.better(&new, &cur, mid);
            if center {
                let n = &mut self.nodes[idx];
                (n.m, n.b, n.id, new.m, new.b, new.id) = (new.m, new.b, new.id, cur.m, cur.b, cur.id);
            }
            if lo == hi {
                return;
            }
            let go_left = left != center;
            let child = if go_left { self.nodes[idx].left } else { self.nodes[idx].right };
            if child == 0 {
                self.nodes.push(Node { left: 0, right: 0, ..new });
                let c = (self.nodes.len() - 1) as u32;
                if go_left {
                    self.nodes[idx].left = c;
                } else {
                    self.nodes[idx].right = c;
                }
                return;
            }
            idx = child as usize;
            if go_left {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
    }

    pub fn query(&self, x: i64) -> u32 {
        assert!(self.lo <= x && x <= self.hi, "query {x} outside [{}, {}]", self.lo, self.hi);
        let (mut idx, mut lo, mut hi) = (0usize, self.lo, self.hi);
        let mut best: Option<(i128, u32)> = None;
        loop {
            let n = self.nodes[idx];
            let v = f(n.m, n.b, x);
            if best.is_none_or(|(bv, bid)| v > bv || (self.low_id_wins && v == bv && n.id < bid)) {
                best = Some((v, n.id));
            }
            if lo == hi {
                break;
            }
            let mid = (lo + hi).div_euclid(2);
            let child = if x <= mid { n.left } else { n.right };
            if child == 0 {
                break;
            }
            idx = child as usize;
            if x <= mid {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        best.expect("query on empty cache").1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_brute_force() {
        let mut rng = 12345u64;
        let mut next = |r: i64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng % (2 * r as u64)) as i64 - r
        };
        let mut lc = LiChao::new(-50, 50);
        let mut lines = vec![];
        for i in 0..500 {
            let (m, b) = (next(1000), next(1_000_000));
            lc.insert(m, b, i);
            lines.push((m, b));
            for _ in 0..5 {
                let x = next(50);
                let best = lines.iter().map(|&(m, b)| f(m, b, x)).max().unwrap();
                let (m, b) = lines[lc.query(x) as usize];
                assert_eq!(f(m, b, x), best);
            }
        }
    }
}
