"""Computation graph whose nodes map 1:1 onto parts of a transformer.

Every value is a scalar *dimension* of the residual stream, at every position.
Dimensions come from:

  - the embedding: the constant ``one``, any function of the input token, and
    any function of the position (a learned absolute position table)
  - ``ffn``: a sum of ReGLU units ``coef * a * ReLU(b)``, with a and b linear in
    existing dims (one dim per node, many hidden units per node)
  - ``lookup``: a hardmax attention head with 2D query/key. At position t it
    selects the unique s <= t maximizing ``q_t * kx_s + ky_s`` and copies the
    value expressions evaluated at s. The query is ``(q, 1)`` and the key is
    ``(kx, ky)``, so every head is a "max over lines at q" query, which
    allows an exact O(log n) cache.

Linear combinations are free: they fold into the weight matrices of whatever
reads them, so ``Expr`` is a linear form over dims rather than a node.

Each node is placed at the earliest legal sublayer (ASAP): attention sublayers
are odd (2L-1), FFN sublayers are even (2L), and a node may only read dims
produced at strictly earlier sublayers. Level 0 is the embedding.
"""

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

Num = int | Fraction
ONE = 0  # dim id of the constant-one embedding feature


class Expr:
    """A linear form ``sum(coef * dim)``; the constant term lives on dim ONE."""

    __slots__ = ("terms",)

    def __init__(self, terms: dict[int, Num] | None = None):
        self.terms = {k: v for k, v in (terms or {}).items() if v != 0}

    def __add__(self, other: "Expr | Num") -> "Expr":
        other = as_expr(other)
        t = dict(self.terms)
        for k, v in other.terms.items():
            t[k] = t.get(k, 0) + v
        return Expr(t)

    __radd__ = __add__

    def __neg__(self) -> "Expr":
        return Expr({k: -v for k, v in self.terms.items()})

    def __sub__(self, other: "Expr | Num") -> "Expr":
        return self + (-as_expr(other))

    def __rsub__(self, other: "Expr | Num") -> "Expr":
        return as_expr(other) - self

    def __mul__(self, c: Num) -> "Expr":
        if not isinstance(c, (int, Fraction)):
            raise TypeError("Expr can only be scaled by a constant (use ffn/gate for products)")
        return Expr({k: v * c for k, v in self.terms.items()})

    __rmul__ = __mul__

    def __truediv__(self, c: int) -> "Expr":
        return self * Fraction(1, c)

    def __repr__(self) -> str:
        return " + ".join(f"{v}*d{k}" for k, v in self.terms.items()) or "0"


def as_expr(x: "Expr | Num") -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, (int, Fraction)):
        return Expr({ONE: x})
    raise TypeError(f"cannot convert {type(x).__name__} to Expr")


@dataclass
class Dim:
    name: str
    kind: str  # "one" | "token" | "pos" | "ffn" | "attn"
    level: int


@dataclass
class FFN:
    name: str
    terms: list[tuple[Num, Expr, Expr]]  # coef * a * ReLU(b)
    out: int
    level: int


@dataclass
class Lookup:
    name: str
    q: Expr
    kx: Expr
    ky: Expr
    values: list[Expr]
    out: list[int]
    q_range: tuple[int, int]  # inclusive bounds on q, needed by the exact cache
    level: int


class Graph:
    def __init__(self) -> None:
        self.dims: list[Dim] = [Dim("one", "one", 0)]
        self.token_fns: dict[int, Callable[[int], Num]] = {}
        self.pos_fns: dict[int, Callable[[int], Num]] = {}
        self.nodes: list[FFN | Lookup] = []
        self.outputs: dict[str, Expr] = {}
        self.one = Expr({ONE: 1})

    # --- embedding features --------------------------------------------------

    def token_feature(self, name: str, fn: Callable[[int], Num]) -> Expr:
        d = self._new_dim(name, "token", 0)
        self.token_fns[d] = fn
        return Expr({d: 1})

    def pos_feature(self, name: str, fn: Callable[[int], Num]) -> Expr:
        d = self._new_dim(name, "pos", 0)
        self.pos_fns[d] = fn
        return Expr({d: 1})

    # --- nodes ---------------------------------------------------------------

    def ffn(self, terms: list[tuple[Num, "Expr | Num", "Expr | Num"]], name: str = "ffn") -> Expr:
        terms = [(c, as_expr(a), as_expr(b)) for c, a, b in terms if c != 0]
        lvl = self.level(*(e for _, a, b in terms for e in (a, b)))
        sub = lvl + 1 if lvl % 2 == 1 else lvl + 2
        d = self._new_dim(name, "ffn", sub)
        self.nodes.append(FFN(name, terms, d, sub))
        return Expr({d: 1})

    def lookup(self, q: "Expr | Num", kx: "Expr | Num", ky: "Expr | Num",
               values: list["Expr | Num"], q_range: tuple[int, int],
               name: str = "lookup") -> list[Expr]:
        q, kx, ky = as_expr(q), as_expr(kx), as_expr(ky)
        values = [as_expr(v) for v in values]
        lvl = self.level(q, kx, ky, *values)
        sub = lvl + 1 if lvl % 2 == 0 else lvl + 2
        outs = [self._new_dim(f"{name}[{i}]", "attn", sub) for i in range(len(values))]
        self.nodes.append(Lookup(name, q, kx, ky, values, outs, q_range, sub))
        return [Expr({d: 1}) for d in outs]

    def output(self, name: str, e: "Expr | Num") -> None:
        self.outputs[name] = as_expr(e)

    # --- introspection --------------------------------------------------------

    def level(self, *exprs: Expr) -> int:
        return max((self.dims[d].level for e in exprs for d in e.terms), default=0)

    def n_layers(self) -> int:
        top = max((n.level for n in self.nodes), default=0)
        return (top + 1) // 2

    def stats(self) -> dict[str, int]:
        ffns = [n for n in self.nodes if isinstance(n, FFN)]
        heads = [n for n in self.nodes if isinstance(n, Lookup)]
        return {
            "layers": self.n_layers(),
            "d_model": len(self.dims),
            "heads": len(heads),
            "ffn_units": sum(len(n.terms) for n in ffns),
        }

    def _new_dim(self, name: str, kind: str, level: int) -> int:
        self.dims.append(Dim(name, kind, level))
        return len(self.dims) - 1
