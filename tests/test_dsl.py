import random

import pytest

from chipformer import spec
from chipformer.dsl.batch import evaluate_batch, output_batch
from chipformer.dsl.evaluate import Evaluator, LiChao, TieError
from chipformer.dsl.lib import Circuit
from chipformer.ref.emu import apply_writes
from chipformer.trace import Machine


def run_unary(build, xs):
    """Evaluate out = build(c, x) with x fed as the token id; returns outputs."""
    c = Circuit()
    x = c.token_feature("x", lambda tok: tok)
    c.output("o", build(c, x))
    return output_batch(c, "o", evaluate_batch(c, list(xs))).tolist()


def run_binary(build, n=256):
    c = Circuit()
    x = c.token_feature("x", lambda tok: tok // n)
    y = c.token_feature("y", lambda tok: tok % n)
    c.output("o", build(c, x, y))
    out = output_batch(c, "o", evaluate_batch(c, list(range(n * n))))
    return {(a, b): out[a * n + b] for a in range(n) for b in range(n)}


@pytest.mark.parametrize("name,build,ref,rng", [
    ("step", lambda c, x: c.step(x - 100), lambda v: int(v >= 100), range(300)),
    ("eq", lambda c, x: c.eq(x, 77), lambda v: int(v == 77), range(300)),
    ("eqglu", lambda c, x: c.eqglu(x * 3 + 1, x, 9), lambda v: 28 * (v == 9), range(300)),
    ("div16", lambda c, x: c.floor_div(x, 16, 15), lambda v: v // 16, range(256)),
    ("mod16", lambda c, x: c.divmod(x, 16, 15)[1], lambda v: v % 16, range(256)),
    ("wrap", lambda c, x: c.wrap(x, 256), lambda v: v % 256, range(512)),
    ("bits", lambda c, x: c.from_bits(c.bits(x, 8)), lambda v: v, range(256)),
    ("bit5", lambda c, x: c.bits(x, 8)[5], lambda v: (v >> 5) & 1, range(256)),
    ("bits12", lambda c, x: c.from_bits(c.bits(x, 12)), lambda v: v, range(4096)),
])
def test_unary(name, build, ref, rng):
    assert run_unary(build, rng) == [ref(v) for v in rng]


@pytest.mark.parametrize("name,build,ref", [
    ("and", lambda c, x, y: c.band(x, y), lambda a, b: a & b),
    ("or", lambda c, x, y: c.bor(x, y), lambda a, b: a | b),
    ("xor", lambda c, x, y: c.bxor(x, y), lambda a, b: a ^ b),
    ("add8", lambda c, x, y: c.wrap(x + y, 256), lambda a, b: (a + b) & 0xFF),
    ("sub8", lambda c, x, y: c.wrap(x - y + 256, 256), lambda a, b: (a - b) & 0xFF),
    ("ge", lambda c, x, y: c.ge(x, y), lambda a, b: int(a >= b)),
    ("mux", lambda c, x, y: c.mux(c.step(x - 128), y, 255 - y), lambda a, b: b if a >= 128 else 255 - b),
])
def test_binary_exhaustive(name, build, ref):
    got = run_binary(build)
    assert all(v == ref(a, b) for (a, b), v in got.items())


def test_layers_are_asap():
    c = Circuit()
    x = c.token_feature("x", lambda t: t)
    s = c.step(x)  # FFN of layer 1
    (p,) = c.prev([s])  # attention of layer 2
    c.step(p)  # FFN of layer 2
    assert [n.level for n in c.nodes] == [2, 3, 4] and c.n_layers() == 2


def test_lichao_matches_brute():
    rng = random.Random(0)
    lc = LiChao(-50, 50)
    lines = []
    for i in range(300):
        m, b = rng.randrange(-1000, 1000), rng.randrange(-10**6, 10**6)
        lc.insert(m, b, i)
        lines.append((m, b))
        for x in rng.sample(range(-50, 51), 5):
            best = max(m * x + b for m, b in lines)
            j = lc.query(x)
            assert lines[j][0] * x + lines[j][1] == best


def memory_circuit():
    """Latest-write memory over a canonical trace, queried at a per-position address."""
    c = Circuit()
    is_val = c.token_feature("is_val", lambda t: int(spec.is_val(t)))
    val = c.token_feature("val", lambda t: t - spec.VAL_BASE if spec.is_val(t) else 0)
    addr = c.token_feature("addr", lambda t: t - spec.ADDR_BASE if spec.is_addr(t) else 0)
    addr2 = c.token_feature("addr^2", lambda t: (t - spec.ADDR_BASE) ** 2 if spec.is_addr(t) else 0)
    query = c.pos_feature("query", lambda t: (t * 2654435761) % 64 + spec.V)  # pseudo-random reg
    a, a2 = c.prev([addr, addr2])
    v, found = c.latest_keyed(query, a, a2, is_val, [val], spec.NUM_CELLS)
    c.output("read", c.eqglu(v, found - query))
    c.output("query", query)
    return c


def random_machine(steps: int, seed: int = 3) -> Machine:
    rng = random.Random(seed)
    m = Machine(bytes(rng.randrange(256) for _ in range(400)))
    for i in range(steps):
        if i % 10 == 0:
            m.tick()
        m.step()
    return m


def expected_reads(tokens, queries):
    cells = [0] * spec.NUM_CELLS
    pending = None
    out = []
    for tok, q in zip(tokens, queries):
        if spec.is_addr(tok):
            pending = tok - spec.ADDR_BASE
        elif spec.is_val(tok):
            apply_writes(cells, [(pending, tok - spec.VAL_BASE)])
        out.append(cells[int(q)])
    return out


@pytest.mark.parametrize("mode", ["lichao", "brute"])
def test_memory_reads_incremental(mode):
    m = random_machine(60)
    c = memory_circuit()
    e = Evaluator(c, mode)
    rows = [e.feed(tok) for tok in m.tokens]
    queries = [e.output("query", r) for r in rows]
    assert [e.output("read", r) for r in rows] == expected_reads(m.tokens, queries)


def test_memory_reads_batch_long():
    m = random_machine(3000)
    c = memory_circuit()
    res = evaluate_batch(c, m.tokens)
    queries = output_batch(c, "query", res)
    assert output_batch(c, "read", res).tolist() == expected_reads(m.tokens, queries)


def test_batch_equals_incremental():
    m = random_machine(40, seed=9)
    c = memory_circuit()
    e = Evaluator(c)
    rows = [e.feed(tok) for tok in m.tokens]
    assert evaluate_batch(c, m.tokens).tolist() == [[float(v) for v in r] for r in rows]


def test_ties_are_detected():
    c = Circuit()
    (v,) = c.latest(c.one * 0 + 1, [c.pos])
    c.output("o", v)
    e = Evaluator(c, "brute")
    e.feed(0)
    c2 = Circuit()
    c2.lookup(0, 0, 0, [c2.pos], (0, 0))  # every key scores 0: a tie
    e2 = Evaluator(c2, "brute")
    e2.feed(0)
    with pytest.raises(TieError):
        e2.feed(0)
