//! A generic decoder-only transformer with hardmax 2D attention heads.
//!
//! Loads the weights written by chipformer/compile.py (the compact sparse
//! model.chipformer, or model.safetensors + config.json) and runs them one
//! position at a time. Nothing here knows about CHIP-8: the
//! weights are plain matrices, stored sparsely because hand-built weights are
//! >99% zeros. Attention uses an exact line-max cache per head, which is the
//! hardmax (BETA -> infinity) limit of softmax attention; with integer scores
//! it gives the same result as softmax at the configured BETA.

use crate::fast::{FastFfn, LineArgmax, constant_dims};
use crate::lichao::LiChao;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// A sparse tensor as (row, col, value) triplets.
pub struct Triplets {
    pub shape: (usize, usize),
    pub entries: Vec<(u32, u32, f64)>,
}

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

    /// Rows of the tensor, or of its transpose.
    fn from_triplets(t: &Triplets, transpose: bool) -> Self {
        let mut e: Vec<(u32, u32, f64)> = if transpose {
            t.entries.iter().map(|&(r, c, v)| (c, r, v)).collect()
        } else {
            t.entries.clone()
        };
        e.retain(|x| x.2 != 0.0);
        e.sort_by_key(|&(r, c, _)| (r, c));
        let rows = if transpose { t.shape.1 } else { t.shape.0 };
        let mut ptr = vec![0u32; rows + 1];
        for &(r, _, _) in &e {
            ptr[r as usize + 1] += 1;
        }
        for r in 0..rows {
            ptr[r + 1] += ptr[r];
        }
        Csr { ptr, idx: e.iter().map(|x| x.1).collect(), val: e.iter().map(|x| x.2).collect() }
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

/// Parse a model.chipformer file (see chipformer/compile.py:save_sparse).
fn read_sparse(bytes: &[u8]) -> Result<(Value, HashMap<String, Triplets>), String> {
    let u32_at = |o: usize| -> Result<u32, String> {
        bytes.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).ok_or("truncated".into())
    };
    if bytes.get(..4) != Some(b"CHFM".as_slice()) || u32_at(4)? != 1 {
        return Err("not a chipformer v1 sparse model".into());
    }
    let hlen = u32_at(8)? as usize;
    let header: Value = serde_json::from_slice(bytes.get(12..12 + hlen).ok_or("truncated header")?)
        .map_err(|e| format!("bad header: {e}"))?;
    let mut off = 12 + hlen;
    let mut tensors = HashMap::new();
    for t in header["tensors"].as_array().ok_or("no tensor index")? {
        let n = t["nnz"].as_u64().ok_or("bad nnz")? as usize;
        let shape = (t["shape"][0].as_u64().ok_or("bad shape")? as usize,
                     t["shape"][1].as_u64().ok_or("bad shape")? as usize);
        let body = bytes.get(off..off + 16 * n).ok_or("truncated tensor data")?;
        let word = |k: usize| u32::from_le_bytes(body[4 * k..4 * k + 4].try_into().unwrap());
        let entries = (0..n)
            .map(|k| {
                let v = f64::from_le_bytes(body[8 * n + 8 * k..8 * n + 8 * k + 8].try_into().unwrap());
                (word(k), word(n + k), v)
            })
            .collect();
        off += 16 * n;
        tensors.insert(t["name"].as_str().ok_or("bad name")?.to_string(), Triplets { shape, entries });
    }
    Ok((header["config"].clone(), tensors))
}

#[cfg(feature = "safetensors")]
fn read_safetensors(dir: &Path) -> Result<(Value, HashMap<String, Triplets>), String> {
    use safetensors::{Dtype, SafeTensors};
    let config: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
    )
    .map_err(|e| format!("config.json: {e}"))?;
    let bytes = std::fs::read(dir.join("model.safetensors")).map_err(|e| format!("model.safetensors: {e}"))?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("safetensors: {e}"))?;
    let mut tensors = HashMap::new();
    for (name, t) in st.tensors() {
        if t.dtype() != Dtype::F64 {
            return Err(format!("{name} must be float64"));
        }
        let shape = (t.shape()[0], t.shape()[1..].iter().product());
        let entries = t
            .data()
            .chunks_exact(8)
            .enumerate()
            .filter_map(|(i, b)| {
                let v = f64::from_le_bytes(b.try_into().unwrap());
                (v != 0.0).then_some(((i / shape.1) as u32, (i % shape.1) as u32, v))
            })
            .collect();
        tensors.insert(name, Triplets { shape, entries });
    }
    Ok((config, tensors))
}

impl Model {
    /// Load model.chipformer from `dir` if present, otherwise model.safetensors + config.json.
    pub fn load(dir: &Path) -> Model {
        let sparse = dir.join("model.chipformer");
        let loaded = if sparse.exists() {
            std::fs::read(&sparse).map_err(|e| e.to_string()).and_then(|b| read_sparse(&b))
        } else {
            #[cfg(feature = "safetensors")]
            {
                read_safetensors(dir)
            }
            #[cfg(not(feature = "safetensors"))]
            {
                Err("no model.chipformer (built without safetensors support)".to_string())
            }
        };
        let (config, tensors) = loaded.unwrap_or_else(|e| panic!("loading {}: {e}", dir.display()));
        Model::build(config, &tensors).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Load a model.chipformer file from memory (used by the WebAssembly build).
    pub fn from_sparse_bytes(bytes: &[u8]) -> Result<Model, String> {
        let (config, tensors) = read_sparse(bytes)?;
        Model::build(config, &tensors)
    }

    fn build(config: Value, tensors: &HashMap<String, Triplets>) -> Result<Model, String> {
        let get = |name: &str| tensors.get(name).ok_or_else(|| format!("missing tensor {name}"));
        let csr = |name: &str| get(name).map(|t| Csr::from_triplets(t, false));
        let csr_t = |name: &str| get(name).map(|t| Csr::from_triplets(t, true));
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
                        Ok(Head {
                            wq: csr(&format!("{p}.wq"))?,
                            wk: csr(&format!("{p}.wk"))?,
                            wv: csr(&format!("{p}.wv"))?,
                            wo_t: csr_t(&format!("{p}.wo"))?,
                            lo: r[0].as_i64().unwrap(),
                            hi: r[1].as_i64().unwrap(),
                            tree: 0,
                            store: 0,
                        })
                    })
                    .collect::<Result<Vec<Head>, String>>()?;
                let p = format!("layers.{l}.ffn");
                Ok(Layer {
                    heads,
                    gate: csr(&format!("{p}.w_gate"))?,
                    up: csr(&format!("{p}.w_up"))?,
                    down_t: csr_t(&format!("{p}.w_down"))?,
                    ffn: None,
                })
            })
            .collect::<Result<Vec<Layer>, String>>()?;
        let embed = csr("embed.token")?;
        let pos_poly_t = csr_t("embed.pos_poly")?;
        let written = layers
            .iter()
            .flat_map(|l| l.heads.iter().map(|h| &h.wo_t).chain(std::iter::once(&l.down_t)));
        let consts = constant_dims(&embed, d, &pos_poly_t, written);
        // CHIPFORMER_NO_FAST=1 runs the plain sparse matrices (for checking the shortcuts)
        let fast = std::env::var_os("CHIPFORMER_NO_FAST").is_none();
        let mut layers = layers;
        let (trees, stores) = assign_caches(&mut layers, fast);
        if fast {
            for l in &mut layers {
                l.ffn = Some(FastFfn::build(&l.gate, &l.up, &l.down_t, &consts));
            }
        }
        let unembed = csr("unembed")?;
        let unembed_fast = if fast { LineArgmax::build(&unembed, &consts) } else { None };
        Ok(Model {
            d,
            vocab,
            embed,
            pos_poly_t,
            layers,
            unembed,
            unembed_fast,
            trees,
            stores,
            config,
        })
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
