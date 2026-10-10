//! A generic decoder-only transformer with hardmax 2D attention heads.
//!
//! Loads model.safetensors + config.json written by chipformer/compile.py and
//! runs it one position at a time. Nothing here knows about CHIP-8: the
//! weights are plain matrices, stored sparsely because hand-built weights are
//! >99% zeros. Attention uses an exact line-max cache per head, which is the
//! hardmax (BETA -> infinity) limit of softmax attention; with integer scores
//! it gives the same result as softmax at the configured BETA.

use crate::fast::{FastFfn, LineArgmax, constant_dims};
use crate::lichao::LiChao;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::path::Path;

/// Rows of a sparse matrix (CSR).
pub struct Csr {
    pub(crate) ptr: Vec<u32>,
    pub(crate) idx: Vec<u32>,
    pub(crate) val: Vec<f64>,
}

impl Csr {
    pub(crate) fn row(&self, r: usize) -> (&[u32], &[f64]) {
        let (a, b) = (self.ptr[r] as usize, self.ptr[r + 1] as usize);
        (&self.idx[a..b], &self.val[a..b])
    }

    pub(crate) fn from_rows(rows: &[(Vec<u32>, Vec<f64>)]) -> Self {
        let (mut ptr, mut idx, mut val) = (vec![0u32], vec![], vec![]);
        for (i, v) in rows {
            idx.extend_from_slice(i);
            val.extend_from_slice(v);
            ptr.push(idx.len() as u32);
        }
        Csr { ptr, idx, val }
    }

    fn from_dense(data: &[f64], rows: usize, cols: usize) -> Self {
        let (mut ptr, mut idx, mut val) = (vec![0u32], vec![], vec![]);
        for r in 0..rows {
            for c in 0..cols {
                let v = data[r * cols + c];
                if v != 0.0 {
                    idx.push(c as u32);
                    val.push(v);
                }
            }
            ptr.push(idx.len() as u32);
        }
        Csr { ptr, idx, val }
    }

    fn transpose_of_dense(data: &[f64], rows: usize, cols: usize) -> Self {
        let t: Vec<f64> = (0..cols * rows).map(|i| data[(i % rows) * cols + i / rows]).collect();
        Csr::from_dense(&t, cols, rows)
    }

    pub fn rows(&self) -> usize {
        self.ptr.len() - 1
    }

    pub fn nnz(&self) -> usize {
        self.idx.len()
    }

    #[inline]
    pub(crate) fn dot(&self, row: usize, x: &[f64]) -> f64 {
        let (a, b) = (self.ptr[row] as usize, self.ptr[row + 1] as usize);
        let mut s = 0.0;
        for k in a..b {
            s += self.val[k] * x[self.idx[k] as usize];
        }
        s
    }

    /// y[idx] += scale * row  (row of a transposed matrix = one column of the original)
    #[inline]
    pub(crate) fn axpy(&self, row: usize, scale: f64, y: &mut [f64]) {
        let (a, b) = (self.ptr[row] as usize, self.ptr[row + 1] as usize);
        for k in a..b {
            y[self.idx[k] as usize] += scale * self.val[k];
        }
    }
}

struct Head {
    wq: Csr, // 2 x d: row 0 = q, row 1 must evaluate to exactly 1
    wk: Csr, // 2 x d: kx, ky
    wv: Csr, // n_v x d
    wo_t: Csr, // n_v x d (transpose of d x n_v)
    lo: i64,
    hi: i64,
    tree: usize,  // KV cache slot for keys (shared by heads with identical keys)
    store: usize, // slot for value vectors (shared by heads with identical keys and values)
}

struct Layer {
    heads: Vec<Head>,
    gate: Csr,
    up: Csr,
    down_t: Csr, // units x d
    ffn: Option<FastFfn>, // exact shortcut for gate/up/down_t (see fast.rs)
}

pub struct Model {
    pub config: Value,
    pub d: usize,
    pub vocab: usize,
    embed: Csr,
    pos_poly_t: Csr, // 2 x d
    layers: Vec<Layer>,
    unembed: Csr,
    unembed_fast: Option<LineArgmax>,
    trees: Vec<(i64, i64)>, // query domain of each key cache
    stores: Vec<usize>,     // value width of each value store
}

/// Give every head a key cache and a value store, sharing them between heads whose key
/// (or key and value) projections are identical, provided no layer in between writes any dim
/// those projections read, so the keys/values computed at the first such head are the same
/// ones every later head would compute.
fn assign_caches(layers: &mut [Layer], share: bool) -> (Vec<(i64, i64)>, Vec<usize>) {
    use std::collections::{HashMap, HashSet};
    type Sig = Vec<(Vec<u32>, Vec<u64>)>;
    let sig = |m: &Csr| -> Sig {
        (0..m.rows()).map(|r| {
            let (i, v) = m.row(r);
            (i.to_vec(), v.iter().map(|x| x.to_bits()).collect())
        }).collect()
    };
    let written: Vec<HashSet<u32>> = layers
        .iter()
        .map(|l| l.heads.iter().flat_map(|h| h.wo_t.idx.iter()).chain(l.down_t.idx.iter()).copied().collect())
        .collect();
    let untouched = |from: usize, to: usize, m: &Csr| (from..to).all(|l| m.idx.iter().all(|i| !written[l].contains(i)));
    let (mut trees, mut stores) = (vec![], vec![]);
    let mut tree_of: HashMap<(Sig, i64, i64), (usize, usize)> = HashMap::new();
    let mut store_of: HashMap<(usize, Sig), (usize, usize)> = HashMap::new();
    for l in 0..layers.len() {
        for h in 0..layers[l].heads.len() {
            let head = &layers[l].heads[h];
            let key = (sig(&head.wk), head.lo, head.hi);
            let tree = match tree_of.get(&key) {
                Some(&(t, from)) if share && untouched(from, l, &head.wk) => t,
                _ => {
                    trees.push((head.lo, head.hi));
                    tree_of.insert(key, (trees.len() - 1, l));
                    trees.len() - 1
                }
            };
            let vkey = (tree, sig(&head.wv));
            let store = match store_of.get(&vkey) {
                Some(&(s, from)) if share && untouched(from, l, &head.wv) => s,
                _ => {
                    stores.push(head.wv.rows());
                    store_of.insert(vkey, (stores.len() - 1, l));
                    stores.len() - 1
                }
            };
            let head = &mut layers[l].heads[h];
            (head.tree, head.store) = (tree, store);
        }
    }
    (trees, stores)
}

fn tensor(st: &SafeTensors, name: &str) -> (Vec<f64>, Vec<usize>) {
    let t = st.tensor(name).unwrap_or_else(|e| panic!("missing tensor {name}: {e}"));
    assert_eq!(t.dtype(), Dtype::F64, "{name} must be float64");
    let data = t.data().chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    (data, t.shape().to_vec())
}

fn csr(st: &SafeTensors, name: &str) -> Csr {
    let (data, shape) = tensor(st, name);
    Csr::from_dense(&data, shape[0], shape[1])
}

fn csr_t(st: &SafeTensors, name: &str) -> Csr {
    let (data, shape) = tensor(st, name);
    Csr::transpose_of_dense(&data, shape[0], shape[1])
}

impl Model {
    pub fn load(dir: &Path) -> Model {
        let config: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).expect("read config.json"),
        )
        .expect("parse config.json");
        let bytes = std::fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
        let st = SafeTensors::deserialize(&bytes).expect("parse safetensors");
        let d = config["d_model"].as_u64().unwrap() as usize;
        let vocab = config["vocab_size"].as_u64().unwrap() as usize;
        let layers = config["layers"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(l, spec)| {
                let n_heads = spec["heads"].as_u64().unwrap() as usize;
                let heads = (0..n_heads)
                    .map(|h| {
                        let p = format!("layers.{l}.attn.{h}");
                        let r = &spec["q_ranges"][h];
                        Head {
                            wq: csr(&st, &format!("{p}.wq")),
                            wk: csr(&st, &format!("{p}.wk")),
                            wv: csr(&st, &format!("{p}.wv")),
                            wo_t: csr_t(&st, &format!("{p}.wo")),
                            lo: r[0].as_i64().unwrap(),
                            hi: r[1].as_i64().unwrap(),
                            tree: 0,
                            store: 0,
                        }
                    })
                    .collect();
                let p = format!("layers.{l}.ffn");
                Layer {
                    heads,
                    gate: csr(&st, &format!("{p}.w_gate")),
                    up: csr(&st, &format!("{p}.w_up")),
                    down_t: csr_t(&st, &format!("{p}.w_down")),
                    ffn: None,
                }
            })
            .collect::<Vec<Layer>>();
        let (embed_dense, _) = tensor(&st, "embed.token");
        let pos_poly_t = csr_t(&st, "embed.pos_poly");
        let written = layers
            .iter()
            .flat_map(|l| l.heads.iter().map(|h| &h.wo_t).chain(std::iter::once(&l.down_t)));
        let consts = constant_dims(&embed_dense, vocab, d, &pos_poly_t, written);
        // CHIPFORMER_NO_FAST=1 runs the plain sparse matrices (for checking the shortcuts)
        let fast = std::env::var_os("CHIPFORMER_NO_FAST").is_none();
        let mut layers = layers;
        let (trees, stores) = assign_caches(&mut layers, fast);
        if fast {
            for l in &mut layers {
                l.ffn = Some(FastFfn::build(&l.gate, &l.up, &l.down_t, &consts));
            }
        }
        let unembed = csr(&st, "unembed");
        let unembed_fast = if fast { LineArgmax::build(&unembed, &consts) } else { None };
        Model {
            d,
            vocab,
            embed: Csr::from_dense(&embed_dense, vocab, d),
            pos_poly_t,
            layers,
            unembed,
            unembed_fast,
            trees,
            stores,
            config,
        }
    }

    /// (distinct key caches, distinct value stores) after sharing
    pub fn cache_counts(&self) -> (usize, usize) {
        (self.trees.len(), self.stores.len())
    }

    pub fn nnz(&self) -> usize {
        let mut n = self.embed.nnz() + self.pos_poly_t.nnz() + self.unembed.nnz();
        for l in &self.layers {
            n += l.gate.nnz() + l.up.nnz() + l.down_t.nnz();
            for h in &l.heads {
                n += h.wq.nnz() + h.wk.nnz() + h.wv.nnz() + h.wo_t.nnz();
            }
        }
        n
    }

    pub fn n_heads(&self) -> usize {
        self.layers.iter().map(|l| l.heads.len()).sum()
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
}

pub struct State<'m> {
    model: &'m Model,
    trees: Vec<LiChao>,        // line-max structure over every position's key
    tree_len: Vec<usize>,      // positions inserted so far
    values: Vec<Vec<f64>>,     // every position's value vector, flattened
    store_len: Vec<usize>,
    pub pos: usize,
    r: Vec<f64>,
    attn: Vec<f64>,
    v_buf: Vec<f64>,
    /// nanoseconds spent in attention, FFN and unembedding (when CHIPFORMER_PROFILE is set)
    pub prof: Option<[u128; 3]>,
}

#[inline]
fn exact_int(x: f64, what: &str) -> i64 {
    assert!(x.fract() == 0.0 && x.abs() < 9.0e18, "{what} is not an exact integer: {x}");
    x as i64
}

impl<'m> State<'m> {
    pub fn new(model: &'m Model) -> Self {
        let prof = std::env::var_os("CHIPFORMER_PROFILE").map(|_| [0; 3]);
        State {
            model,
            trees: model.trees.iter().map(|&(lo, hi)| LiChao::new(lo, hi)).collect(),
            tree_len: vec![0; model.trees.len()],
            values: vec![vec![]; model.stores.len()],
            store_len: vec![0; model.stores.len()],
            pos: 0,
            r: vec![0.0; model.d],
            attn: vec![0.0; model.d],
            v_buf: vec![],
            prof,
        }
    }

    /// Forget the context (positions restart at 0).
    pub fn reset(&mut self) {
        self.trees.iter_mut().for_each(LiChao::clear);
        self.values.iter_mut().for_each(Vec::clear);
        self.tree_len.iter_mut().for_each(|n| *n = 0);
        self.store_len.iter_mut().for_each(|n| *n = 0);
        self.pos = 0;
    }

    /// Process one position; returns the argmax next token.
    pub fn feed(&mut self, token: u32) -> u32 {
        let m = self.model;
        let t = self.pos as f64;
        let r = &mut self.r;
        r.iter_mut().for_each(|x| *x = 0.0);
        m.embed.axpy(token as usize, 1.0, r);
        m.pos_poly_t.axpy(0, t, r);
        m.pos_poly_t.axpy(1, t * t, r);

        let mut clock = self.prof.map(|_| std::time::Instant::now());
        let mut lap = |slot: usize, prof: &mut Option<[u128; 3]>| {
            if let (Some(p), Some(c)) = (prof.as_mut(), clock.as_mut()) {
                p[slot] += c.elapsed().as_nanos();
                *c = std::time::Instant::now();
            }
        };
        for layer in &m.layers {
            self.attn.iter_mut().for_each(|x| *x = 0.0);
            for h in &layer.heads {
                // the first head using a shared cache inserts this position's key / values
                if self.tree_len[h.tree] == self.pos {
                    let kx = exact_int(h.wk.dot(0, r), "key x");
                    let ky = exact_int(h.wk.dot(1, r), "key y");
                    self.trees[h.tree].insert(kx, ky, self.pos as u32);
                    self.tree_len[h.tree] += 1;
                }
                let n_v = h.wv.rows();
                if self.store_len[h.store] == self.pos {
                    let store = &mut self.values[h.store];
                    for i in 0..n_v {
                        store.push(h.wv.dot(i, r));
                    }
                    self.store_len[h.store] += 1;
                }
                assert!(h.wq.dot(1, r) == 1.0, "query y must be 1");
                let q = exact_int(h.wq.dot(0, r), "query");
                let src = self.trees[h.tree].query(q) as usize;
                self.v_buf.clear();
                self.v_buf.extend_from_slice(&self.values[h.store][src * n_v..(src + 1) * n_v]);
                for (i, &v) in self.v_buf.iter().enumerate() {
                    if v != 0.0 {
                        h.wo_t.axpy(i, v, &mut self.attn);
                    }
                }
            }
            for (x, a) in r.iter_mut().zip(&self.attn) {
                *x += a;
            }
            lap(0, &mut self.prof);
            // ReGLU: r += W_down (ReLU(W_gate r) * (W_up r))
            self.attn.iter_mut().for_each(|x| *x = 0.0);
            if let Some(ffn) = &layer.ffn {
                ffn.eval(r, &mut self.v_buf, &mut self.attn);
            } else {
                for u in 0..layer.gate.rows() {
                    let g = layer.gate.dot(u, r);
                    if g > 0.0 {
                        let hval = g * layer.up.dot(u, r);
                        if hval != 0.0 {
                            layer.down_t.axpy(u, hval, &mut self.attn);
                        }
                    }
                }
            }
            for (x, a) in r.iter_mut().zip(&self.attn) {
                *x += a;
            }
            lap(1, &mut self.prof);
        }
        self.pos += 1;

        if let Some(tok) = m.unembed_fast.as_ref().and_then(|f| f.argmax(r)) {
            lap(2, &mut self.prof);
            return tok;
        }
        let mut best = (f64::NEG_INFINITY, 0u32);
        for v in 0..m.vocab {
            let logit = m.unembed.dot(v, r);
            if logit > best.0 {
                best = (logit, v as u32);
            }
        }
        lap(2, &mut self.prof);
        best.1
    }
}
