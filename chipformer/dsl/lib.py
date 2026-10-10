"""Exact integer primitives built from ReGLU units and 2D hardmax lookups.

All inputs are assumed to be integers within the stated ranges; outputs are
then exact. Key identities (x integer):

    step(x)  = [x >= 0]  = ReLU(x + 1) - ReLU(x)
    hat(x)   = [x == 0]  = ReLU(x + 1) - 2 ReLU(x) + ReLU(x - 1)
    a * c    = a * ReLU(c)              for c in {0, 1}
    a * step(x), a * hat(x)             same identities with the gate a
"""

from chipformer.dsl.graph import Expr, Graph, Num

MAX_POS = 1 << 22  # positions are kept below this by context rollover


class Circuit(Graph):
    def __init__(self) -> None:
        super().__init__()
        self.pos = self.pos_feature("pos", lambda t: t)
        self.pos2 = self.pos_feature("pos^2", lambda t: t * t)

    # --- comparisons -----------------------------------------------------------

    def step(self, x: Expr, name: str = "step") -> Expr:
        """[x >= 0]"""
        return self.stepglu(self.one, x, name)

    def stepglu(self, a: "Expr | Num", x: Expr, name: str = "stepglu") -> Expr:
        """a * [x >= 0]"""
        return self.ffn([(1, a, x + 1), (-1, a, x)], name)

    def eq(self, x: Expr, k: "Expr | Num" = 0, name: str = "eq") -> Expr:
        """[x == k]"""
        return self.eqglu(self.one, x, k, name)

    def eqglu(self, a: "Expr | Num", x: Expr, k: "Expr | Num" = 0, name: str = "eqglu") -> Expr:
        """a * [x == k]"""
        d = x - k
        return self.ffn([(1, a, d + 1), (-2, a, d), (1, a, d - 1)], name)

    def ge(self, x: Expr, y: "Expr | Num", name: str = "ge") -> Expr:
        return self.step(x - y, name)

    def gate(self, a: "Expr | Num", c: Expr, name: str = "gate") -> Expr:
        """a * c for c in {0, 1}"""
        return self.ffn([(1, a, c)], name)

    def mux(self, c: Expr, a: "Expr | Num", b: "Expr | Num", name: str = "mux") -> Expr:
        """a if c else b, for c in {0, 1}"""
        return b + self.gate(a - b, c, name)

    def select(self, cases: list[tuple[Expr, "Expr | Num"]], name: str = "select") -> Expr:
        """sum(value * cond) for one-hot conds; one FFN node."""
        return self.ffn([(1, v, c) for c, v in cases], name)

    def all_of(self, *conds: "Expr | Num", name: str = "and") -> Expr:
        """Logical AND of binary conds in one FFN node."""
        return self.step(sum(conds, Expr()) - len(conds), name)

    def index(self, values: list["Expr | Num"], idx: Expr, name: str = "index") -> Expr:
        """values[idx] for integer 0 <= idx < len(values); one FFN node."""
        terms = []
        for i, v in enumerate(values):
            d = idx - i
            terms += [(1, v, d + 1), (-2, v, d), (1, v, d - 1)]
        return self.ffn(terms, name)

    def relu(self, x: Expr, name: str = "relu") -> Expr:
        return self.ffn([(1, self.one, x)], name)

    # --- integer arithmetic ----------------------------------------------------

    def floor_div(self, x: Expr, d: int, q_max: int, name: str = "div") -> Expr:
        """floor(x / d) for 0 <= x < d * (q_max + 1)"""
        terms = []
        for k in range(1, q_max + 1):
            terms += [(1, self.one, x - d * k + 1), (-1, self.one, x - d * k)]
        return self.ffn(terms, name)

    def divmod(self, x: Expr, d: int, q_max: int, name: str = "divmod") -> tuple[Expr, Expr]:
        q = self.floor_div(x, d, q_max, name)
        return q, x - d * q

    def wrap(self, x: Expr, m: int, name: str = "wrap") -> Expr:
        """x mod m for 0 <= x < 2m"""
        return x - m * self.step(x - m, name)

    def bits(self, x: Expr, n: int, name: str = "bits") -> list[Expr]:
        """Bits of 0 <= x < 2**n, LSB first, all computed in one FFN sublayer."""
        out = []
        for i in range(n):
            terms = []
            for m in range(1 << (n - 1 - i)):
                on, off = (2 * m + 1) << i, (2 * m + 2) << i
                terms += [(1, self.one, x - on + 1), (-1, self.one, x - on)]
                if off < (1 << n):
                    terms += [(-1, self.one, x - off + 1), (1, self.one, x - off)]
            out.append(self.ffn(terms, f"{name}{i}"))
        return out

    @staticmethod
    def from_bits(bits: list[Expr]) -> Expr:
        return sum((b * (1 << i) for i, b in enumerate(bits)), Expr())

    def band_bits(self, xb: list[Expr], yb: list[Expr], name: str = "and") -> Expr:
        return self.ffn([(1 << i, x, y) for i, (x, y) in enumerate(zip(xb, yb))], name)

    def band(self, x: Expr, y: Expr, n: int = 8, name: str = "and") -> Expr:
        return self.band_bits(self.bits(x, n, f"{name}.x"), self.bits(y, n, f"{name}.y"), name)

    def bxor(self, x: Expr, y: Expr, n: int = 8, name: str = "xor") -> Expr:
        return x + y - 2 * self.band(x, y, n, name)

    def bor(self, x: Expr, y: Expr, n: int = 8, name: str = "or") -> Expr:
        return x + y - self.band(x, y, n, name)

    # --- lookups -----------------------------------------------------------------

    def prev(self, values: list[Expr], name: str = "prev") -> list[Expr]:
        """Values at position t-1 (at t=0: position 0).

        Score 2s(t-1) - s^2 = (t-1)^2 - (s-(t-1))^2 peaks exactly at s = t-1.
        """
        return self.lookup(self.pos - 1, 2 * self.pos, -self.pos2, values,
                           (-1, MAX_POS), name)

    def latest(self, flag: Expr, values: list[Expr], name: str = "latest") -> list[Expr]:
        """Values at the latest position s <= t with flag(s) = 1 (flag in {0, 1})."""
        big = 2 * MAX_POS
        return self.lookup(0, 0, self.pos + big * (flag - 1), values, (0, 0), name)

    def latest_keyed(self, q: Expr, key: Expr, key_sq: Expr, is_entry: Expr,
                     values: list[Expr], n_keys: int, name: str = "keyed") -> list[Expr]:
        """Values at the latest entry whose integer key equals q.

        Score T(2*key*q - key^2) + s = T(q^2 - (key-q)^2) + s: the key match
        dominates, recency breaks ties. Non-entries are pushed below every
        entry. Returns ``values + [found_key]``; if no entry has key q, the
        nearest key wins, so callers compare found_key with q.
        """
        t = MAX_POS
        big = 4 * t * n_keys * n_keys
        return self.lookup(q, 2 * t * key, -t * key_sq + self.pos + big * (is_entry - 1),
                           values + [key], (0, n_keys - 1), name)
