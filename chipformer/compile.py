"""Compile a Graph into the weight tensors of a plain transformer.

Architecture (no CHIP-8 knowledge; see chipformer/dense.py):

    r_t   = E[token_t] + W_pos @ [t, t^2]             (polynomial position embedding)
    for each layer:
        r += sum_h  Wo_h  V_h[argmax-softmax_s (Wq_h r_t) . (Wk_h r_s)]     (s <= t)
        r += W_down (ReLU(W_gate r) * (W_up r))                           (ReGLU)
    logits = U r

Residual dimensions are the graph's dims, one to one. Each attention head
has 2D queries/keys; the second query coordinate is the constant-one dim, so
scores are ``q * kx + ky``. Softmax runs at temperature 1/BETA; since scores
are integers that differ by at least 1, BETA = 1000 makes every attention row
exactly one-hot in fp64.

The unembedding row for token v is ``2v * W_next - v^2 * e_one``, so
``logit_v = y^2 - (y - v)^2`` with y the "next" output: argmax is exactly y.
"""

import json
from pathlib import Path

import numpy as np

from chipformer import spec
from chipformer.dsl.graph import FFN, ONE, Expr, Graph, Lookup

BETA = 1000.0


def _row(e: Expr, d: int) -> np.ndarray:
    r = np.zeros(d)
    for k, v in e.terms.items():
        r[k] = float(v)
    return r


def compile_graph(g: Graph, vocab_size: int = spec.VOCAB_SIZE,
                  output: str = "next") -> tuple[dict[str, np.ndarray], dict]:
    d = len(g.dims)
    w: dict[str, np.ndarray] = {}

    emb = np.zeros((vocab_size, d))
    emb[:, ONE] = 1.0
    for dim, fn in g.token_fns.items():
        emb[:, dim] = [float(fn(t)) for t in range(vocab_size)]
    w["embed.token"] = emb

    pos_names = {"pos": 1, "pos^2": 2}
    wpos = np.zeros((d, 2))
    for dim, fn in g.pos_fns.items():
        name = g.dims[dim].name
        if name not in pos_names:
            raise ValueError(f"position feature {name!r} is not t or t^2")
        wpos[dim, pos_names[name] - 1] = 1.0
    w["embed.pos_poly"] = wpos

    n_layers = g.n_layers()
    layers = []
    for layer in range(n_layers):
        heads = [n for n in g.nodes if isinstance(n, Lookup) and n.level == 2 * layer + 1]
        ffns = [n for n in g.nodes if isinstance(n, FFN) and n.level == 2 * layer + 2]
        for h, node in enumerate(heads):
            p = f"layers.{layer}.attn.{h}"
            w[f"{p}.wq"] = np.stack([_row(node.q, d), _row(g.one, d)])
            w[f"{p}.wk"] = np.stack([_row(node.kx, d), _row(node.ky, d)])
            w[f"{p}.wv"] = np.stack([_row(v, d) for v in node.values])
            wo = np.zeros((d, len(node.out)))
            for i, dim in enumerate(node.out):
                wo[dim, i] = 1.0
            w[f"{p}.wo"] = wo
        units = [(node.out, c, a, b) for node in ffns for c, a, b in node.terms]
        gate = np.zeros((len(units), d))
        up = np.zeros((len(units), d))
        down = np.zeros((d, len(units)))
        for u, (out, c, a, b) in enumerate(units):
            gate[u] = _row(b, d)
            up[u] = _row(a, d)
            down[out, u] = float(c)
        w[f"layers.{layer}.ffn.w_gate"] = gate
        w[f"layers.{layer}.ffn.w_up"] = up
        w[f"layers.{layer}.ffn.w_down"] = down
        layers.append({"heads": len(heads), "ffn_units": len(units),
                       "head_names": [n.name for n in heads],
                       "q_ranges": [list(n.q_range) for n in heads]})

    y = _row(g.outputs[output], d)
    v = np.arange(vocab_size, dtype=np.float64)
    unembed = 2.0 * v[:, None] * y[None, :]
    unembed[:, ONE] -= v * v
    w["unembed"] = unembed

    config = {
        "machine": machine_spec(),
        "d_model": d, "n_layers": n_layers, "vocab_size": vocab_size, "beta": BETA,
        "pos_features": ["t", "t^2"], "layers": layers,
        "params": int(sum(a.size for a in w.values())),
        "nonzero_params": int(sum(np.count_nonzero(a) for a in w.values())),
    }
    return w, config


def machine_spec() -> dict:
    """Constants a host needs to boot the model and talk to it (from chipformer.spec)."""
    names = ["VAL_BASE", "ADDR_BASE", "STEP", "TICK", "BOOT", "NUM_CELLS", "NULL", "CLS", "PC",
             "RNG", "KEYS", "DISPLAY", "DISPLAY_W", "DISPLAY_H", "PROGRAM_START", "FONT_BASE",
             "RNG_SEED", "QUIRKS"]
    out = {n: getattr(spec, n) for n in names}
    out["FONT"] = list(spec.FONT)
    out["QUIRK_NAMES"] = list(spec.QUIRK_NAMES)
    out["PROFILES"] = {k: list(v) for k, v in spec.PROFILES.items()}
    return out


def save(w: dict[str, np.ndarray], config: dict, out_dir: Path) -> None:
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in w.items()}, out_dir / "model.safetensors")
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))


def load(model_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    from safetensors.numpy import load_file

    return load_file(model_dir / "model.safetensors"), json.loads(
        (model_dir / "config.json").read_text())


SPARSE_MAGIC = b"CHFM"


def save_sparse(w: dict[str, np.ndarray], config: dict, path: Path) -> None:
    """The same weights as a compact sparse file (for the browser).

    Layout: b"CHFM", u32 version (1), u32 header length, UTF-8 JSON header
    {"config": ..., "tensors": [{"name", "shape", "nnz"}]}, then for each tensor
    in header order: nnz u32 rows, nnz u32 cols, nnz f64 values (little-endian,
    row-major order). Exactly the nonzero entries of the dense tensors.
    """
    tensors, blobs = [], []
    for name, a in w.items():
        a2 = a.reshape(a.shape[0], -1)
        rows, cols = np.nonzero(a2)
        tensors.append({"name": name, "shape": list(a2.shape), "nnz": len(rows)})
        blobs += [rows.astype("<u4").tobytes(), cols.astype("<u4").tobytes(),
                  a2[rows, cols].astype("<f8").tobytes()]
    header = json.dumps({"config": config, "tensors": tensors}).encode()
    with open(path, "wb") as f:
        f.write(SPARSE_MAGIC + np.array([1, len(header)], dtype="<u4").tobytes() + header)
        f.writelines(blobs)


def load_sparse(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    data = path.read_bytes()
    assert data[:4] == SPARSE_MAGIC, "not a chipformer sparse model"
    _, hlen = np.frombuffer(data[4:12], dtype="<u4")
    header = json.loads(data[12:12 + hlen])
    off, w = 12 + int(hlen), {}
    for t in header["tensors"]:
        n = t["nnz"]
        rows = np.frombuffer(data, "<u4", n, off)
        cols = np.frombuffer(data, "<u4", n, off + 4 * n)
        vals = np.frombuffer(data, "<f8", n, off + 8 * n)
        off += 16 * n
        a = np.zeros(t["shape"])
        a[rows, cols] = vals
        w[t["name"]] = a
    return w, header["config"]


def main() -> None:
    import argparse

    from chipformer.chip8 import build

    ap = argparse.ArgumentParser(description="Build the CHIP-8 transformer weights")
    ap.add_argument("--out", type=Path, default=Path("model"))
    args = ap.parse_args()
    w, config = compile_graph(build())
    save(w, config, args.out)
    save_sparse(w, config, args.out / "model.chipformer")
    print(f"{args.out}: {config['n_layers']} layers, d_model {config['d_model']}, "
          f"{sum(lay['heads'] for lay in config['layers'])} heads, "
          f"{sum(lay['ffn_units'] for lay in config['layers'])} FFN units, "
          f"{config['params']:,} params ({config['nonzero_params']:,} nonzero)")


if __name__ == "__main__":
    main()
