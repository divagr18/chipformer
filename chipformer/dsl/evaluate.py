"""Exact autoregressive evaluation of a Graph.

Runs position by position, like a transformer with a KV cache: each lookup
head keeps the key (a line ``kx * q + ky``) and value vector of every past
position. Arithmetic uses Python ints/Fractions, so results are exact.

Two cache implementations:
  - "lichao": a Li Chao tree per head, O(log range) insert and query
  - "brute":  a scan over all positions that also asserts the argmax is unique
              (softmax would average ties, so a tie means a broken circuit)
"""

from fractions import Fraction

from chipformer.dsl.graph import FFN, ONE, Expr, Graph, Lookup, Num

EXACT_LIMIT = 1 << 53  # every quantity must stay exactly representable in fp64


class TieError(AssertionError):
    pass


class LiChao:
    """Max over lines at integer points of [lo, hi]; nodes are created lazily."""

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi
        self.nodes: dict[int, tuple[Num, Num, int]] = {}

    def insert(self, m: Num, b: Num, ident: int) -> None:
        new = (m, b, ident)
        idx, lo, hi = 1, self.lo, self.hi
        while True:
            cur = self.nodes.get(idx)
            if cur is None:
                self.nodes[idx] = new
                return
            mid = (lo + hi) // 2
            left = new[0] * lo + new[1] > cur[0] * lo + cur[1]
            center = new[0] * mid + new[1] > cur[0] * mid + cur[1]
            if center:
                self.nodes[idx], new = new, cur
            if lo == hi:
                return
            if left != center:
                idx, hi = 2 * idx, mid
            else:
                idx, lo = 2 * idx + 1, mid + 1

    def query(self, x: int) -> int:
        if not self.lo <= x <= self.hi:
            raise ValueError(f"query {x} outside [{self.lo}, {self.hi}]")
        idx, lo, hi = 1, self.lo, self.hi
        best_v, best_id = None, -1
        while idx in self.nodes:
            m, b, ident = self.nodes[idx]
            v = m * x + b
            if best_v is None or v > best_v:
                best_v, best_id = v, ident
            if lo == hi:
                break
            mid = (lo + hi) // 2
            if x <= mid:
                idx, hi = 2 * idx, mid
            else:
                idx, lo = 2 * idx + 1, mid + 1
        return best_id


class BruteCache:
    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi
        self.lines: list[tuple[Num, Num]] = []

    def insert(self, m: Num, b: Num, ident: int) -> None:
        assert ident == len(self.lines)
        self.lines.append((m, b))

    def query(self, x: int) -> int:
        if not self.lo <= x <= self.hi:
            raise ValueError(f"query {x} outside [{self.lo}, {self.hi}]")
        scores = [m * x + b for m, b in self.lines]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        if len(winners) > 1:
            raise TieError(f"tie between positions {winners[:4]} at q={x}")
        return winners[0]


def ev(e: Expr, vals: list[Num]) -> Num:
    return sum(c * vals[d] for d, c in e.terms.items())


def relu(x: Num) -> Num:
    return max(0, x)


class Evaluator:
    def __init__(self, graph: Graph, mode: str = "lichao"):
        self.g = graph
        cache_cls = {"lichao": LiChao, "brute": BruteCache}[mode]
        self.heads = [n for n in graph.nodes if isinstance(n, Lookup)]
        self.caches = {id(h): cache_cls(*h.q_range) for h in self.heads}
        self.head_values: dict[int, list[list[Num]]] = {id(h): [] for h in self.heads}
        self.pos = 0
        self.rows: list[list[Num]] | None = None  # optionally keep every residual row

    def keep_rows(self) -> "Evaluator":
        self.rows = []
        return self

    def feed(self, token: int) -> list[Num]:
        """Process one position; returns its final residual row."""
        g, t = self.g, self.pos
        vals: list[Num] = [0] * len(g.dims)
        vals[ONE] = 1
        for d, fn in g.token_fns.items():
            vals[d] = fn(token)
        for d, fn in g.pos_fns.items():
            vals[d] = fn(t)
        for node in g.nodes:
            if isinstance(node, FFN):
                vals[node.out] = sum(c * ev(a, vals) * relu(ev(b, vals)) for c, a, b in node.terms)
            else:
                cache = self.caches[id(node)]
                store = self.head_values[id(node)]
                kx, ky, q = ev(node.kx, vals), ev(node.ky, vals), ev(node.q, vals)
                if isinstance(q, Fraction):
                    if q.denominator != 1:
                        raise ValueError(f"{node.name}: non-integer query {q}")
                    q = int(q)
                lo, hi = node.q_range
                worst = abs(kx) * max(abs(lo), abs(hi)) + abs(ky)
                if worst >= EXACT_LIMIT:
                    raise OverflowError(f"{node.name}: score magnitude {worst:.3g} >= 2^53")
                cache.insert(kx, ky, t)
                store.append([ev(v, vals) for v in node.values])
                src = store[cache.query(q)]
                for d, v in zip(node.out, src):
                    vals[d] = v
        self.pos += 1
        if self.rows is not None:
            self.rows.append(vals)
        return vals

    def output(self, name: str, row: list[Num]) -> Num:
        return ev(self.g.outputs[name], row)
