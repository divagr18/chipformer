//! Exact algebraic shortcuts that exploit the structure of hand-built weights.
//!
//! None of these change any result (every quantity is an exactly representable
//! integer or dyadic rational, so evaluation order does not matter):
//!
//! - constant dims: a residual dim that the embedding sets to the same value for
//!   every token and no layer ever writes (the "one" dim) is folded into biases
//! - shared forms: each distinct weighted sum feeding an FFN unit is computed
//!   once per layer, not once per unit
//! - piecewise-linear groups: units `c * ReLU(s + b)` that share the input form s
//!   and write one output dim sum to `s * C(s) + B(s)`, where C and B are prefix
//!   sums over the units sorted by threshold -b: a binary search instead of a
//!   pass over every unit
//! - line argmax: if every unembedding row depends on one non-constant dim x,
//!   each logit is a line in x and argmax is a max-of-lines query

use crate::lichao::LiChao;
use crate::model::Csr;
use std::collections::HashMap;

/// Value of each residual dim that is constant at every position (None otherwise).
pub fn constant_dims<'a>(
    embed: &[f64],
    vocab: usize,
    d: usize,
    pos_poly_t: &Csr,
    written: impl Iterator<Item = &'a Csr>,
) -> Vec<Option<f64>> {
    let mut consts: Vec<Option<f64>> = (0..d)
        .map(|c| {
            let v = embed[c];
            (0..vocab).all(|t| embed[t * d + c] == v).then_some(v)
        })
        .collect();
    for &i in &pos_poly_t.idx {
        consts[i as usize] = None;
    }
    for m in written {
        for &i in &m.idx {
            consts[i as usize] = None;
        }
    }
    consts
}

/// Split a sparse row into its non-constant part and the constant contribution.
fn split(row: (&[u32], &[f64]), consts: &[Option<f64>]) -> (Vec<u32>, Vec<f64>, f64) {
    let (mut idx, mut val, mut bias) = (vec![], vec![], 0.0);
    for (&i, &v) in row.0.iter().zip(row.1) {
        match consts[i as usize] {
            Some(c) => bias += v * c,
            None => {
                idx.push(i);
                val.push(v);
            }
        }
    }
    (idx, val, bias)
}

const NO_FORM: u32 = u32::MAX;

struct Unit {
    gate: u32,
    gate_bias: f64,
    up: u32, // NO_FORM: the up projection is the constant up_bias
    up_bias: f64,
}

struct Group {
    form: u32,
    out: u32,
    thresholds: Vec<f64>, // sorted ascending; unit i is active when s > thresholds[i]
    c_prefix: Vec<f64>,   // sum of c over the first k units
    cb_prefix: Vec<f64>,  // sum of c*b over the first k units
}

pub struct FastFfn {
    forms: Csr,
    units: Vec<Unit>,
    down: Csr, // one row per general unit
    groups: Vec<Group>,
}

impl FastFfn {
    pub fn build(gate: &Csr, up: &Csr, down_t: &Csr, consts: &[Option<f64>]) -> FastFfn {
        let mut form_ids: HashMap<(Vec<u32>, Vec<u64>), u32> = HashMap::new();
        let mut forms: Vec<(Vec<u32>, Vec<f64>)> = vec![];
        let mut intern = |idx: Vec<u32>, val: Vec<f64>| -> u32 {
            let key = (idx.clone(), val.iter().map(|v| v.to_bits()).collect());
            *form_ids.entry(key).or_insert_with(|| {
                forms.push((idx, val));
                (forms.len() - 1) as u32
            })
        };
        let mut groups: HashMap<(u32, u32), Vec<(f64, f64, f64)>> = HashMap::new();
        let (mut units, mut down_rows) = (vec![], vec![]);
        for u in 0..gate.rows() {
            let (gi, gv, gb) = split(gate.row(u), consts);
            let (ui, uv, ub) = split(up.row(u), consts);
            let (di, dv) = down_t.row(u);
            if dv.iter().all(|&v| v == 0.0) {
                continue;
            }
            let g = intern(gi, gv);
            if ui.is_empty() && di.len() == 1 {
                // c * ReLU(s + b) with c = down coefficient * constant up value
                let c = dv[0] * ub;
                if c != 0.0 {
                    groups.entry((g, di[0])).or_default().push((-gb, c, gb));
                }
                continue;
            }
            let up_form = if ui.is_empty() { NO_FORM } else { intern(ui, uv) };
            units.push(Unit { gate: g, gate_bias: gb, up: up_form, up_bias: ub });
            down_rows.push((di.to_vec(), dv.to_vec()));
        }
        let mut groups: Vec<Group> = groups
            .into_iter()
            .map(|((form, out), mut us)| {
                us.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let (mut c_prefix, mut cb_prefix) = (vec![0.0], vec![0.0]);
                for &(_, c, b) in &us {
                    c_prefix.push(c_prefix.last().unwrap() + c);
                    cb_prefix.push(cb_prefix.last().unwrap() + c * b);
                }
                Group { form, out, thresholds: us.iter().map(|u| u.0).collect(), c_prefix, cb_prefix }
            })
            .collect();
        groups.sort_by_key(|g| (g.form, g.out));
        FastFfn { forms: Csr::from_rows(&forms), units, down: Csr::from_rows(&down_rows), groups }
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        (self.forms.rows(), self.groups.len(), self.units.len())
    }

    /// out += FFN(r)
    pub fn eval(&self, r: &[f64], buf: &mut Vec<f64>, out: &mut [f64]) {
        buf.clear();
        buf.extend((0..self.forms.rows()).map(|f| self.forms.dot(f, r)));
        for g in &self.groups {
            let s = buf[g.form as usize];
            let k = g.thresholds.partition_point(|&t| t < s);
            out[g.out as usize] += s * g.c_prefix[k] + g.cb_prefix[k];
        }
        for (i, u) in self.units.iter().enumerate() {
            let g = buf[u.gate as usize] + u.gate_bias;
            if g > 0.0 {
                let a = if u.up == NO_FORM { 0.0 } else { buf[u.up as usize] } + u.up_bias;
                let h = g * a;
                if h != 0.0 {
                    self.down.axpy(i, h, out);
                }
            }
        }
    }
}

const LINE_DOMAIN: i64 = 1 << 40;

/// argmax over logits that are all lines in one residual dim.
pub struct LineArgmax {
    dim: usize,
    tree: LiChao,
}

impl LineArgmax {
    pub fn build(unembed: &Csr, consts: &[Option<f64>]) -> Option<LineArgmax> {
        let mut dim: Option<usize> = None;
        let mut tree = LiChao::with_low_id_wins(-LINE_DOMAIN, LINE_DOMAIN);
        let exact = |x: f64| x.fract() == 0.0 && x.abs() < 4.0e15;
        for v in 0..unembed.rows() {
            let (idx, val, bias) = split(unembed.row(v), consts);
            let slope = match idx.as_slice() {
                [] => 0.0,
                [i] if dim.is_none_or(|d| d == *i as usize) => {
                    dim = Some(*i as usize);
                    val[0]
                }
                _ => return None,
            };
            if !exact(slope) || !exact(bias) {
                return None;
            }
            tree.insert(slope as i64, bias as i64, v as u32);
        }
        dim.map(|dim| LineArgmax { dim, tree })
    }

    /// None when the query is outside the exact fast path (the caller falls back to a scan).
    #[inline]
    pub fn argmax(&self, r: &[f64]) -> Option<u32> {
        let x = r[self.dim];
        (x.fract() == 0.0 && x.abs() <= LINE_DOMAIN as f64).then(|| self.tree.query(x as i64))
    }
}
