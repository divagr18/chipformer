"""Exact gadgets for the q35 IR.

MLP gadgets are lists of unit terms ``(coef, a, b)`` (``coef * a * relu(b)``) so several can share
one node. They keep every positive gate value a bf16 integer (|b| <= 256), which makes silu an
exact relu; the negative side may be any size, since silu(<= -104) is exactly 0.

Values are represented as
  bits      list of 0/1 Exprs, LSB first (an integer is their weighted sum, free)
  one-hot   list of 0/1 Exprs, exactly one set (bits of the index are linear, free)
  int       an Expr holding a small integer (|value| <= 256 inside a gate)
"""

from fractions import Fraction

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import A_BIT, A_REF, REF_COLS, Chan, Graph, K

Terms = list[tuple]


def cat(term_lists) -> Terms:
    """Concatenate unit-term lists."""
    return [t for ts in term_lists for t in ts]


def from_bits(bits: list) -> Expr:
    return sum((b * (1 << i) for i, b in enumerate(bits)), Expr())


def onehot_bits(oh: list, n: int) -> list:
    """Bits of the index of a one-hot."""
    return [sum((e for v, e in enumerate(oh) if v >> i & 1), Expr()) for i in range(n)]


def onehot_int(oh: list) -> Expr:
    return sum((v * e for v, e in enumerate(oh)), Expr())


class Circ(Graph):
    # --- unit terms -----------------------------------------------------------

    def t_const(self, c) -> Terms:
        return [(c, self.one, self.one)]

    def t_and(self, lits, coef=1, a=None) -> Terms:
        """coef * a * AND(lits) for 0/1 literals.

        The gate is 4 * (sum - n + 1/2): +2 when all are set, <= -2 otherwise, so a false AND
        is exactly 0 even when the literals carry fp32 noise (silu(-256) == -0).
        """
        a = self.one if a is None else a
        return [(Fraction(coef) / 2, a, 4 * sum(lits, Expr()) - (4 * len(lits) - 2))]

    def t_ge(self, z, theta, lo, hi, coef=1, lits=()) -> Terms:
        """coef * [z >= theta] (AND lits) for an integer z in [lo, hi]."""
        if theta > hi:
            return []
        n = len(lits)
        if theta <= lo and not n:
            return self.t_const(coef)
        if hi - theta + 1 <= 256:
            m = 1 << (hi - lo).bit_length()  # > hi - lo, and a bf16 weight
            zz = z + m * (sum(lits, Expr()) - n) if n else z
            return [(coef, self.one, zz - theta + 1), (-coef, self.one, zz - theta)]
        assert theta - lo <= 256, (theta, lo, hi)
        base = self.t_and(lits, coef) if n else self.t_const(coef)
        return base + self.t_lt(z, theta, lo, hi, -coef, lits)

    def t_lt(self, z, theta, lo, hi, coef=1, lits=()) -> Terms:
        """coef * [z < theta] (AND lits) for an integer z in [lo, hi]."""
        if theta <= lo:
            return []
        n = len(lits)
        if theta > hi and not n:
            return self.t_const(coef)
        if theta - lo <= 256:
            m = 1 << (hi - lo).bit_length()
            zz = z - m * (sum(lits, Expr()) - n) if n else z
            return [(coef, self.one, theta - zz), (-coef, self.one, theta - zz - 1)]
        assert hi - theta + 1 <= 256, (theta, lo, hi)
        base = self.t_and(lits, coef) if n else self.t_const(coef)
        return base + self.t_ge(z, theta, lo, hi, -coef, lits)

    def t_in(self, z, a, b, lo, hi, coef=1, lits=()) -> Terms:
        """coef * [a <= z < b] (AND lits)."""
        a, b = max(a, lo), min(b, hi + 1)
        if a >= b:
            return []
        if a == lo:
            return self.t_lt(z, b, lo, hi, coef, lits)
        if b == hi + 1:
            return self.t_ge(z, a, lo, hi, coef, lits)
        if lits:
            return self.t_ge(z, a, lo, hi, coef, lits) + self.t_ge(z, b, lo, hi, -coef, lits)
        lt_cost = (b - lo, a - lo)
        if max(lt_cost) <= 256:
            return self.t_lt(z, b, lo, hi, coef) + self.t_lt(z, a, lo, hi, -coef)
        return self.t_ge(z, a, lo, hi, coef) + self.t_ge(z, b, lo, hi, -coef)

    def t_eq(self, z, v, lo, hi, coef=1, lits=()) -> Terms:
        return self.t_in(z, v, v + 1, lo, hi, coef, lits)

    # --- nodes ----------------------------------------------------------------

    def node(self, terms: Terms, name: str) -> Expr:
        if not terms:
            return Expr()
        return self.ffn(terms, name)

    def and_(self, lits, name: str) -> Expr:
        return self.node(self.t_and(lits), name)

    def onehot_of_bits(self, bits: list, name: str, n: int | None = None) -> list:
        """One-hot of the integer with these bits (values 0..2^len-1, or 0..n-1)."""
        size = n or 1 << len(bits)
        out = []
        for v in range(size):
            lits = [b if v >> i & 1 else self.one - b for i, b in enumerate(bits)]
            out.append(self.and_(lits, f"{name}=={v}"))
        return out

    def nib_onehot(self, z, lo, hi, name: str, div: int = 1) -> list:
        """One-hot of (z // div) mod 16 for an integer z in [lo, hi] (lo >= 0)."""
        out = []
        for x in range(16):
            terms = []
            k0 = lo // (16 * div)
            for k in range(k0, hi // (16 * div) + 1):
                a = div * (16 * k + x)
                terms += self.t_in(z, a, a + div, lo, hi)
            out.append(self.node(terms, f"{name}[{x}]"))
        return out

    def select_bits(self, cases: list[tuple[list, list]], n: int, name: str) -> list:
        """Bits of sum(cond * value) for one-hot conditions; values are bit lists or ints."""
        out = []
        for i in range(n):
            terms = []
            for lits, val in cases:
                if isinstance(val, int):
                    if val >> i & 1:
                        terms += self.t_and(lits)
                elif i < len(val):
                    b = val[i]
                    if isinstance(b, int):
                        if b:
                            terms += self.t_and(lits)
                    else:
                        terms += self.t_and([*lits, b])
            out.append(self.node(terms, f"{name}.b{i}"))
        return out

    # --- 12-bit arithmetic ----------------------------------------------------

    def add12(self, a_lo, a_hi_oh: list, d, d_max: int, name: str, a_lo_max: int = 255):
        """One-hot nibbles of (A + d) mod 4096, where A = a_lo + 256 * index(a_hi_oh)."""
        lo_hi = a_lo_max + d_max
        L = a_lo + d
        n0 = self.nib_onehot(L, 0, lo_hi, f"{name}.n0")
        n1 = self.nib_onehot(L, 0, lo_hi, f"{name}.n1", div=16)
        n2 = []
        for x in range(16):
            terms = (self.t_lt(L, 256, 0, lo_hi, lits=[a_hi_oh[x]])
                     + self.t_ge(L, 256, 0, lo_hi, lits=[a_hi_oh[(x - 1) % 16]]))
            n2.append(self.node(terms, f"{name}.n2[{x}]"))
        return n0, n1, n2

    # --- heads ----------------------------------------------------------------

    def slot_head(self, name, keys: dict, queries: dict, data: dict, ref, a=0,
                  outs=None, gate=None, tiny: float = 0.0) -> list:
        """A head whose rows are 0/1 data columns plus a ballast.

        keys: slot -> Chan or pre-activation; queries likewise; data: column (>= 2) -> Chan or
        Expr giving the *bit* (0/1) to store; ref: Chan or Expr, 1 where the row written carries
        the ballast. Returns one dim per data column, holding the read bit.
        """
        def chan(c, scale):
            if isinstance(c, Chan):
                return Chan(c.expr * scale, c.taps)
            return Chan(c * scale)

        def sel(c):
            """Keys and queries: K * (2c - 1), so c = 1 gives K and c <= 0 gives <= -K. A
            false entry is then exactly 0 after silu even with fp32 noise in c, which matters:
            a leak into another row would be rescaled to full size by the output norm."""
            if isinstance(c, Chan):
                return Chan(c.expr * (2 * K) - self.one * K, c.taps)
            return Chan(c * (2 * K) - self.one * K)

        v = {col: chan(c, A_BIT) for col, c in data.items()}
        for col in REF_COLS:
            v[col] = chan(ref, A_REF)
        gate = self.one if gate is None else gate
        cols = sorted(data) if outs is None else outs
        z = {col: K * gate for col in cols}
        return self.head(name, {s: sel(c) for s, c in queries.items()},
                         {s: sel(c) for s, c in keys.items()}, v, a, z,
                         [(col, 2.0**7) for col in cols], tiny=tiny)


def sel(c, e):
    """Key / query pre-activation K * (2e - 1) for a 0/1 (or <= 0 when false) form e."""
    if isinstance(e, Chan):
        return Chan(e.expr * (2 * K) - c.one * K, e.taps)
    return Chan(e * (2 * K) - c.one * K)


def chan(e, scale):
    if isinstance(e, Chan):
        return Chan(e.expr * scale, e.taps)
    return Chan(e * scale)


def thead(c, name, keys, queries, vheads, tiny=0.0):
    """A head of 0/1 tables: one dict per value head with data {column: 0/1 form or Chan},
    ref (1 where a written row carries the ballast), optional gate (0/1: its outputs), beta
    (write gate) and into (dims to add the outputs onto). Returns each value head's outputs."""
    specs = []
    for x in vheads:
        v = {col: chan(e, A_BIT) for col, e in x["data"].items()}
        for col in REF_COLS:
            v[col] = chan(x["ref"], A_REF)
        gate = c.one if x.get("gate") is None else x["gate"]
        cols = sorted(x["data"])
        specs.append({"v": v, "a": x.get("a", 0), "z": {col: K * gate for col in cols},
                      "outs": [(col, 2.0**7) for col in cols], "beta": x.get("beta"),
                      "into": x.get("into")})
    first, rest = specs[0], specs[1:]
    outs = c.head(name, {s: sel(c, e) for s, e in queries.items()},
                  {s: sel(c, e) for s, e in keys.items()}, first["v"], first["a"], first["z"],
                  first["outs"], tiny=tiny, beta=first["beta"], into=first["into"], vx=rest)
    res, i = [], 0
    for x in specs:
        n = len(x["outs"])
        res.append(outs[i:i + n])
        i += n
    return res
