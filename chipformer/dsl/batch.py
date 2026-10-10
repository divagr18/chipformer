"""Teacher-forced evaluation of a Graph over a whole token sequence at once.

Equivalent to feeding the tokens one by one through ``Evaluator``, but each
node is computed as a column over all positions with numpy. Values are
float64, which is exact here because every quantity is a dyadic rational
below 2^53 (lookups assert the bound).
"""

import numpy as np

from chipformer.dsl.evaluate import EXACT_LIMIT, LiChao, TieError
from chipformer.dsl.graph import FFN, ONE, Expr, Graph


def ev_batch(e: Expr, m: np.ndarray) -> np.ndarray:
    if not e.terms:
        return np.zeros(m.shape[0])
    idx = np.fromiter(e.terms.keys(), dtype=np.int64, count=len(e.terms))
    coef = np.array([float(v) for v in e.terms.values()])
    return m[:, idx] @ coef


def prefix_argmax(score: np.ndarray) -> np.ndarray:
    """For each t, the unique s <= t maximizing score[s]."""
    run = np.maximum.accumulate(score)
    prev = np.concatenate([[-np.inf], run[:-1]])
    if np.any(score == prev):
        t = int(np.argmax(score == prev))
        raise TieError(f"tie at position {t}")
    new = score > prev
    return np.maximum.accumulate(np.where(new, np.arange(len(score)), 0))


def evaluate_batch(g: Graph, tokens: list[int]) -> np.ndarray:
    """Residual stream after the last sublayer: an (n_positions, n_dims) array."""
    n = len(tokens)
    m = np.zeros((n, len(g.dims)))
    m[:, ONE] = 1.0
    toks = np.asarray(tokens, dtype=np.int64)
    uniq, inv = np.unique(toks, return_inverse=True)
    for d, fn in g.token_fns.items():
        m[:, d] = np.array([float(fn(int(u))) for u in uniq])[inv]
    for d, fn in g.pos_fns.items():
        m[:, d] = [float(fn(t)) for t in range(n)]

    for node in g.nodes:
        if isinstance(node, FFN):
            acc = np.zeros(n)
            for c, a, b in node.terms:
                acc += float(c) * ev_batch(a, m) * np.maximum(ev_batch(b, m), 0.0)
            m[:, node.out] = acc
            continue

        kx, ky, q = ev_batch(node.kx, m), ev_batch(node.ky, m), ev_batch(node.q, m)
        lo, hi = node.q_range
        worst = np.abs(kx).max() * max(abs(lo), abs(hi)) + np.abs(ky).max()
        if worst >= EXACT_LIMIT:
            raise OverflowError(f"{node.name}: score magnitude {worst:.3g} >= 2^53")
        if np.any(q != np.round(q)) or q.min() < lo or q.max() > hi:
            raise ValueError(f"{node.name}: query outside integer range [{lo}, {hi}]")
        if lo == hi:
            src = prefix_argmax(kx * lo + ky)
        else:
            src = np.empty(n, dtype=np.int64)
            tree = LiChao(lo, hi)
            for t, (a, b, x) in enumerate(zip(kx.astype(np.int64).tolist(),
                                              ky.astype(np.int64).tolist(),
                                              q.astype(np.int64).tolist())):
                tree.insert(a, b, t)
                src[t] = tree.query(x)
        for d, v in zip(node.out, node.values):
            m[:, d] = ev_batch(v, m)[src]
    return m


def output_batch(g: Graph, name: str, m: np.ndarray) -> np.ndarray:
    return ev_batch(g.outputs[name], m)
