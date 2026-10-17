"""Circuit graph for the Qwen3.5 target: SwiGLU units and Gated DeltaNet heads.

Like ``chipformer.dsl.graph``, every value is a scalar dim of the residual stream and linear
forms (``Expr``) are free. The node types are the two sublayers of a Qwen3.5 linear-attention
layer, described closely enough that the emitter only has to scale weights:

``FFN``   a sum of units ``coef * a * relu(b)``, emitted as ``silu(K b) * (a / K)``. silu equals
          relu on every value the circuit produces: 0, >= 18 (sigmoid rounds to 1) or <= -104
          (sigmoid underflows to 0).
``Head``  one Gated DeltaNet key head with its value head(s), channel by channel: q/k/v
          channels are linear forms with conv taps over the last 4 tokens followed by silu;
          keys must be one-hot (an exact overwrite of one state row) or zero, queries
          multi-hot or zero; ``a`` picks the per-token decay (keep / halve / clear); ``z``
          gates each output column. Rows carry a ballast in their ``ref`` columns that
          dominates the per-head output RMSNorm, so reads come back exactly rescaled.

Values live in IR units: a residual value x reaches every sublayer as exactly x * 2^-r because
chip tokens carry a large ballast (see spike.py), and the emitter folds 2^r into the weights.
The constant ONE is that ballast. On text tokens every chip dim, ONE included, is 0, so every
node outputs exactly 0 there.

Exactness is checked numerically: ``evaluate`` runs a token sequence through the graph and
asserts that every intermediate the hardware materializes is a bf16 value, that silu inputs are
in its exact range, and that keys, queries and decays have the allowed shapes.
"""

from dataclasses import dataclass, field

import numpy as np

from chipformer.dsl.graph import ONE, Expr, Num, as_expr

K = 128  # silu gate scale: silu(K n) = K relu(n) for integers n
A_REF = 2**24  # ballast value of a state row (two columns)
A_BIT = 2**7  # value of a set bit in a state row: (A_BIT / A_REF)^2 * 126 is absorbed
REF_COLS = (0, 1)
DECAY = {0.0: 1.0, 110.0: 0.5, 238.0: 0.0}  # a -> exp(g) with dt_bias = -110, A_log = 0
HD = 128  # key/value head dim


@dataclass
class Dim:
    name: str
    kind: str  # "one" | "token" | "ffn" | "head"
    level: int


@dataclass
class FFN:
    name: str
    terms: list[tuple[Num, Expr, Expr]]  # coef * a * relu(b)
    out: int
    level: int
    into: bool = False  # out is an existing dim that this node adds to


@dataclass
class Chan:
    expr: Expr
    taps: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # x_t, x_{t-1}, ...


@dataclass
class Head:
    name: str
    q: dict[int, Chan]  # key-dim channel -> query
    k: dict[int, Chan]
    v: dict[int, Chan]  # value column -> value (columns REF_COLS are the ballast)
    a: Expr  # decay selector, see DECAY
    z: dict[int, Expr]  # value column -> output gate silu(z)
    out: list[tuple[int, int, float]]  # (value column, residual dim, weight)
    level: int
    n_cols: int = HD
    tiny: float = 0.0  # outputs below this may round (the counter's 2^-j for j >> JMAX)


@dataclass
class Target:
    """Per-layer capacity of the host model."""

    key_heads: int = 16
    value_heads: int = 16
    intermediate: int = 4608  # the merged 4B has 9216; 0.8B's 3584 needs 18 layers since v14

    @property
    def cols(self) -> int:
        return HD  # value columns a head uses (one value head; the output norm is per head)


@dataclass
class Graph:
    target: Target = field(default_factory=Target)

    def __post_init__(self) -> None:
        self.dims: list[Dim] = [Dim("one", "one", 0)]
        self.token_fns: dict[int, object] = {}
        self.nodes: list[FFN | Head] = []
        self.one = Expr({ONE: 1})
        self.units: dict[int, int] = {}  # sublayer -> FFN units used
        self.heads: dict[int, int] = {}  # sublayer -> key heads used

    # --- dims -----------------------------------------------------------------

    def token_feature(self, name: str, fn) -> Expr:
        d = self._dim(name, "token", 0)
        self.token_fns[d] = fn
        return Expr({d: 1})

    def _dim(self, name: str, kind: str, level: int) -> int:
        self.dims.append(Dim(name, kind, level))
        return len(self.dims) - 1

    def level(self, *exprs: Expr) -> int:
        return max((self.dims[d].level for e in exprs for d in e.terms), default=0)

    # --- placement ------------------------------------------------------------

    def _place(self, lvl: int, odd: bool, need: int, used: dict[int, int], cap: int) -> int:
        sub = lvl + 1
        while (sub % 2 == 1) != odd or used.get(sub, 0) + need > cap:
            sub += 1
        used[sub] = used.get(sub, 0) + need
        return sub

    # --- nodes ----------------------------------------------------------------

    def ffn(self, terms, name: str = "ffn", into: Expr | None = None, min_level: int = 0) -> Expr:
        terms = [(c, as_expr(a), as_expr(b)) for c, a, b in terms if c != 0]
        lvl = max(self.level(*(e for _, a, b in terms for e in (a, b))), min_level)
        sub = self._place(lvl, False, len(terms), self.units, self.target.intermediate)
        if into is not None:
            (d,) = into.terms
            self.dims[d].level = max(self.dims[d].level, sub)
        else:
            d = self._dim(name, "ffn", sub)
        self.nodes.append(FFN(name, terms, d, sub, into is not None))
        return Expr({d: 1})

    def head(self, name: str, q: dict, k: dict, v: dict, a: "Expr | Num", z: dict,
             outs: list[tuple[int, float]], min_level: int = 0, tiny: float = 0.0) -> list[Expr]:
        """Add a head; ``outs`` lists (value column, weight) pairs, one new dim each."""
        q, k, v = ({i: c if isinstance(c, Chan) else Chan(as_expr(c)) for i, c in ch.items()}
                   for ch in (q, k, v))
        a = as_expr(a)
        z = {i: as_expr(e) for i, e in z.items()}
        exprs = [c.expr for ch in (q, k, v) for c in ch.values()] + [a, *z.values()]
        lvl = max(self.level(*exprs), min_level)
        sub = self._place(lvl, True, 1, self.heads, self.target.key_heads)
        dims = [self._dim(f"{name}[{col}]", "head", sub) for col, _ in outs]
        self.nodes.append(Head(name, q, k, v, a, z,
                               [(col, d, w) for (col, w), d in zip(outs, dims)], sub,
                               self.target.cols, tiny))
        return [Expr({d: 1}) for d in dims]

    def n_layers(self) -> int:
        top = max((n.level for n in self.nodes), default=0)
        return (top + 1) // 2

    def stats(self) -> dict[str, int]:
        return {
            "layers": self.n_layers(),
            "dims": len(self.dims),
            "heads": sum(isinstance(n, Head) for n in self.nodes),
            "ffn_units": sum(len(n.terms) for n in self.nodes if isinstance(n, FFN)),
            "max_units": max(self.units.values(), default=0),
            "max_heads": max(self.heads.values(), default=0),
        }


# --- exactness checks -------------------------------------------------------------
#
# The evaluator computes every node twice: ideally (exact float64) and as the bf16 hardware does
# (every materialized tensor rounded to bf16, sums accumulated in fp32). The two must agree on
# every node output; fp32 hardware has no rounding and computes the ideal values.


class ExactnessError(AssertionError):
    pass


def bf16(x: np.ndarray) -> np.ndarray:
    """Round to the nearest bf16 value (ties to even), returned as float64."""
    f = np.asarray(x, dtype=np.float32).copy()
    u = f.view(np.uint32)
    u += 0x7FFF + ((u >> 16) & 1)
    u &= 0xFFFF0000
    return f.astype(np.float64)


def is_bf16(x: np.ndarray) -> np.ndarray:
    return (bf16(x) == x) & np.isfinite(x)


def require(ok, what: str, x) -> None:
    ok, x = np.asarray(ok), np.asarray(x)
    if not np.all(ok):
        t = np.unravel_index(int(np.argmin(ok)), ok.shape)
        raise ExactnessError(f"{what}: bad value {x[t]!r} at {t}")


def silu_ok(x: np.ndarray) -> np.ndarray:
    """silu(x) == relu(x) exactly (fp32 and bf16)."""
    return (x == 0) | (x >= 18) | (x <= -104)


# --- evaluation ---------------------------------------------------------------------


def ev(e: Expr, m: np.ndarray) -> np.ndarray:
    if not e.terms:
        return np.zeros(m.shape[0])
    idx = np.fromiter(e.terms.keys(), dtype=np.int64, count=len(e.terms))
    coef = np.array([float(c) for c in e.terms.values()])
    return m[:, idx] @ coef


def conv(c: Chan, m: np.ndarray, check: bool) -> np.ndarray:
    """silu(conv(projection)) as computed in bf16; must equal the exact value."""
    x = ev(c.expr, m)
    if check:
        require(is_bf16(x), "projection", x)
    out = np.zeros_like(x)
    for lag, w in enumerate(c.taps):
        if w:
            out[lag:] += w * x[: len(x) - lag]
    if check:
        require(is_bf16(out) & silu_ok(out), "conv output (silu input)", out)
    return np.maximum(out, 0.0)


def evaluate(g: Graph, tokens: list[int], check: bool = True) -> np.ndarray:
    """Residual stream after every node: (n_tokens, n_dims), exact in float64.

    Text tokens (ids outside the chip vocabulary) can be passed as -1: all chip dims are 0.
    """
    n = len(tokens)
    m = np.zeros((n, len(g.dims)))
    toks = np.asarray(tokens, dtype=np.int64)
    chip = toks >= 0
    m[:, ONE] = chip
    uniq, inv = np.unique(toks, return_inverse=True)
    for d, fn in g.token_fns.items():
        col = np.array([float(fn(int(u))) if u >= 0 else 0.0 for u in uniq])[inv]
        if check:
            require(is_bf16(col), f"token feature {g.dims[d].name}", col)
        m[:, d] = col
    for node in g.nodes:
        try:
            (eval_ffn if isinstance(node, FFN) else eval_head)(node, m, chip, check)
        except ExactnessError as e:
            raise ExactnessError(f"{node.name}: {e}") from None
    return m


def eval_ffn(node: FFN, m: np.ndarray, chip: np.ndarray, check: bool) -> None:
    ideal = np.zeros(m.shape[0])
    hw = np.zeros(m.shape[0])
    for c, a, b in node.terms:
        bv, av = ev(b, m), ev(a, m)
        ideal += float(c) * av * np.maximum(bv, 0.0)
        if check:
            gate = bf16(K * bv)
            require(silu_ok(gate), "gate outside silu's exact range", gate)
            up = bf16(av / K)
            hw += float(c) * bf16(np.maximum(gate, 0.0) * up)
    if check:
        hw = bf16(hw)
        require(hw == ideal, "bf16 output differs from exact", np.stack([hw, ideal], 1)[:, 0])
    if node.into:
        ideal = m[:, node.out] + ideal
        if check:
            require(is_bf16(ideal), "residual add", ideal)
    m[:, node.out] = ideal
def eval_head(node: Head, m: np.ndarray, chip: np.ndarray, check: bool) -> None:
    n = m.shape[0]
    q = np.stack([conv(c, m, check) for c in node.q.values()], 1) if node.q else np.zeros((n, 0))
    k = np.stack([conv(c, m, check) for c in node.k.values()], 1) if node.k else np.zeros((n, 0))
    qi, ki = list(node.q), list(node.k)
    v = np.zeros((n, node.n_cols))
    for col, c in node.v.items():
        v[:, col] = conv(c, m, check)
    a = ev(node.a, m)
    z = {col: ev(e, m) for col, e in node.z.items()}
    if check:
        require(is_bf16(a) & np.isin(a, list(DECAY)), "decay selector a", a)
        for col, zz in z.items():
            require(is_bf16(zz) & silu_ok(zz), f"gate z[{col}]", zz)
        nz = (k > 0).sum(1)
        require(nz <= 1, "key must be one-hot", nz)
        require(~((nz > 0) & ~chip), "key on a text token", nz)
        for t in np.nonzero((q > 0).sum(1) > 1)[0]:
            vals = q[t][q[t] > 0]
            require(vals == vals[0], "query must be multi-hot", vals)
        written = v[nz > 0]
        if len(written):
            data = np.delete(written, REF_COLS, 1)
            require(np.isin(data, [0.0, A_BIT]), "state row data column", data)
            ref = written[:, list(REF_COLS)]
            require((ref == 0) | (ref == A_REF), "state row ballast column", ref)
    state: dict[int, np.ndarray] = {}
    out = np.zeros((n, node.n_cols))
    decay = np.vectorize(DECAY.get)(a)
    for t in range(n):
        if decay[t] == 0.0:
            state.clear()
        elif decay[t] == 0.5:
            state = {s: row * 0.5 for s, row in state.items()}
        if k.shape[1] and k[t].any():
            state[ki[int(np.argmax(k[t]))]] = v[t].copy()
        if q.shape[1] and q[t].any():
            o = sum((state[s] for s, w in zip(qi, q[t]) if w > 0 and s in state),
                    np.zeros(node.n_cols))
            ref = (o[list(REF_COLS)] ** 2).sum()
            if ref == 0:
                if check:
                    require(o == 0, "read without ballast", o)
                continue
            if check:
                data = (np.delete(o, REF_COLS) ** 2).sum()
                require(np.array([data <= ref * 2.0**-26]), "data not absorbed by ballast",
                        np.array([data / ref]))
            out[t] = o / np.sqrt(ref / node.n_cols)
    for col, d, w in node.out:
        zz = z.get(col, np.zeros(n))
        val = out[:, col] * np.maximum(zz, 0.0)
        if check:
            small = np.abs(out[:, col]) < node.tiny
            require(is_bf16(out[:, col]) | small, f"normed column {col}", out[:, col])
            require(is_bf16(val * w) | small, f"head output column {col}", val * w)
        m[:, d] = val * w
