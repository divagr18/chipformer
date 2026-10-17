"""Overlay: circuit dims that live only on chip tokens share Qwen's dims in a merged checkpoint.

The chip layers run first, and while they do every text token carries its Qwen embedding in dims
[0, D0); on chip tokens those dims are 0. So a circuit dim that is 0 at every text position can
use a Qwen dim as its residual slot (emit.Layout.over): a chip token sees the circuit's value
there, a text token sees q, a Qwen embedding value. That stays exact when

1. nothing writes into an overlaid slot at a text position, so Qwen's input is bit-identical:
   every unit and head column that writes an overlaid dim is switched off on text tokens;
2. whatever reads an overlaid slot at a text position is switched off there, or cannot matter:
   - an FFN unit: its gate gets -KILL * is_text, so silu is exactly 0 whatever q is;
   - a head that writes no key on text tokens: its key channels that read the overlay get the
     same switch, so the key stays 0 (no state write), and every output gate z of a column
     that is 0 on text does too (a query of q then reads nothing out);
3. nothing that matters on text tokens reads an overlaid dim at all: units and head columns that
   are nonzero there, keys written there (the capture), decays, and channels with conv lag taps
   (which carry a text token's projection into the next token).

The handovers and the exit clear of overlaid slots are gated by mu (chip tokens) instead of ONE
(emit.emit_layers), so a text token keeps q.

What is 0 or switched off on text tokens is measured on ``corpus()``, mixed traces of Qwen text,
inline expressions, tool calls and prints. ``apply`` writes the switches into the graph, so the
circuit's own IR follows them: a trace where a switched-off unit should have fired shows up as
an IR mismatch against the reference like any other circuit bug.
"""

import random
from dataclasses import dataclass, field

import numpy as np

from chipformer.dsl.graph import ONE, Expr
from chipformer.q35 import bridge as B
from chipformer.q35 import ir
from chipformer.q35.ir import FFN, K, Chan, Graph, Head, conv, ev

KILL = 2**20  # -KILL * is_text switches a gate, key or output gate off on text tokens


@dataclass
class Plan:
    dims: set[int] = field(default_factory=set)  # dims that may use Qwen's dims
    terms: set[tuple[int, int]] = field(default_factory=set)  # (node, term): FFN units
    keys: set[tuple[int, int]] = field(default_factory=set)  # (node, key channel)
    gates: set[tuple[int, int]] = field(default_factory=set)  # (node, value column): z
    q_bound: float = 0.0  # the largest |q| every switched-off value still absorbs

    def summary(self) -> str:
        return (f"overlay: {len(self.dims)} dims, {len(self.terms)} units, {len(self.keys)} key "
                f"channels and {len(self.gates)} output gates switched off on text, |q| <= "
                f"{self.q_bound:.4g}")


# --- the corpus ------------------------------------------------------------------------------

FORMS = {"add": [[B.PLUS]], "sub": [[B.MINUS], [B.MINUS2]],
         "mul": [[B.TIMES], [B.BSL, B.TIMESW, B.SPACE]],
         "div": [[B.DIV], [B.BSL, B.FN_DIV, B.SPACE]]}
LEADS = [[B.SPACE], [B.DOLLAR], [B.SDOLLAR, B.SPACE], [B.DDOLLAR], [B.SDDOLLAR], [B.NL],
         [B.WS], [B.LPAREN], [B.OTHER], [B.EQ, B.SPACE]]


def _digits(rng: random.Random, n: int, commas: bool) -> list[int]:
    s = str(rng.randrange(10 ** (n - 1), 10**n))
    out = []
    for i, ch in enumerate(s):
        if commas and i and (len(s) - i) % 3 == 0:
            out.append(B.COMMA)
        out.append(B.HEX[int(ch)])
    return out


def _expr(rng: random.Random) -> list[int]:
    op = rng.choice(list(FORMS))
    commas = rng.random() < 0.3
    a = _digits(rng, rng.randint(1, 7), commas)
    b = _digits(rng, rng.randint(1, 5), commas)
    gap = [B.SPACE] if rng.random() < 0.5 else []
    return rng.choice(LEADS) + a + rng.choice(FORMS[op]) + gap + b + [B.EQ]


def _number(rng: random.Random, n: int) -> str:
    return str(rng.randrange(10 ** (n - 1), 10**n))


def corpus(seed: int = 7, docs: int = 24) -> list[list[int]]:
    """Token sequences (chip tokens and text classes) of mixed sessions."""
    from chipformer.q35.lib import bcd

    rng = random.Random(seed)
    text = [t for t in range(B.TEXT_BASE, B.TEXT_BASE + B.N_TEXT) if t != B.TC_CLOSE]
    common = [B.HEX[d] for d in range(10)] * 3 + [B.SPACE] * 6 + [B.OTHER] * 6 + text
    out = []
    for _ in range(docs):
        bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
        bm.text(rng.randint(1, 4))
        for _ in range(rng.randint(3, 8)):
            r = rng.random()
            if r < 0.45:
                bm.qwen(_expr(rng) + [rng.choice(common) for _ in range(rng.randint(1, 6))])
            elif r < 0.7:
                bm.qwen([rng.choice(common) for _ in range(rng.randint(5, 25))])
            else:
                op = rng.choice(list(FORMS))
                bm.tool_call(op, _number(rng, rng.randint(1, 6)), _number(rng, rng.randint(1, 4)))
                while not bm.halted:
                    bm.step()
                bm.text(rng.randint(1, 4))
        out.append(bm.tokens)
    return out


# --- measuring and planning --------------------------------------------------------------------


def _reads(e: Expr) -> set[int]:
    return {d for d in e.terms if d != ONE}


def _text_mask(tokens: list[int]) -> np.ndarray:
    return np.array([B.is_text(t) for t in tokens])


@dataclass
class Activity:
    """What is nonzero at text positions, over a corpus."""

    zero: np.ndarray  # dim -> 0 at every text position
    terms: dict = field(default_factory=dict)  # (node, term) -> coef * a * relu(b) != 0 somewhere
    gate_max: dict = field(default_factory=dict)  # (node, term) -> max b
    keys: dict = field(default_factory=dict)  # node -> a key written at some text position
    key_max: dict = field(default_factory=dict)  # (node, channel) -> max conv output
    cols: dict = field(default_factory=dict)  # (node, column) -> output nonzero somewhere
    z_max: dict = field(default_factory=dict)  # (node, column) -> max z


def activity(g: Graph, traces: list[list[int]]) -> Activity:
    act = Activity(np.ones(len(g.dims), bool))

    def upd(d, k, v, f=max):
        d[k] = f(d[k], v) if k in d else v

    for toks in traces:
        txt = _text_mask(toks)
        m = ir.evaluate(g, toks, check=False)
        act.zero &= ~(m[txt] != 0).any(0)
        for i, n in enumerate(g.nodes):
            if isinstance(n, FFN):
                for j, (coef, a, b) in enumerate(n.terms):
                    bv = ev(b, m)[txt]
                    v = float(coef) * ev(a, m)[txt] * np.maximum(bv, 0)
                    upd(act.terms, (i, j), bool((v != 0).any()), lambda x, y: x or y)
                    upd(act.gate_max, (i, j), float(bv.max(initial=-np.inf)))
                continue
            wrote = False
            for ch_i, ch in n.k.items():
                x = conv(ch, m, False)[txt]
                wrote |= bool((x > 0).any())
                upd(act.key_max, (i, ch_i), float(ev(ch.expr, m)[txt].max(initial=-np.inf)))
            upd(act.keys, i, wrote, lambda x, y: x or y)
            for x_, vh in enumerate(n.vheads()):
                for col, d, _ in vh.out:
                    # (a dim several heads add into is judged by its total)
                    upd(act.cols, (i, x_, col), bool((m[txt, d] != 0).any()), lambda x, y: x or y)
                    if col in vh.z:
                        upd(act.z_max, (i, x_, col),
                            float(ev(vh.z[col], m)[txt].max(initial=-np.inf)))
    return act


def _vh_reads(vh) -> set[int]:
    """What a value head's write gate and decay read: they act at text positions too."""
    return _reads(vh.a) | (set() if vh.beta is None else _reads(vh.beta))


def needed(g: Graph, act: Activity, keep: set[int]) -> set[int]:
    """Dims whose value at text positions matters: the outputs the lm_head reads, what heads
    write into their state there (decays always; keys and values of heads that write keys on
    text), what conv lag taps carry to the next token, and everything those depend on."""
    need = set(keep) | _reads(g.is_text) | _reads(g.mu)
    for i, n in enumerate(g.nodes):
        if isinstance(n, Head):
            vhs = n.vheads()
            vchans = [ch for vh in vhs for ch in vh.v.values()]
            need |= set().union(*(_vh_reads(vh) for vh in vhs))
            chans = list(n.q.values()) + list(n.k.values()) + vchans
            need |= {d for ch in chans if any(ch.taps[1:]) for d in _reads(ch.expr)}
            if act.keys[i]:
                need |= {d for ch in list(n.k.values()) + vchans for d in _reads(ch.expr)}
    for n in sorted(g.nodes, key=lambda n: -n.level):  # reads come from lower levels
        if isinstance(n, FFN):
            if n.out in need:
                need |= {d for _, a, b in n.terms for d in _reads(a) | _reads(b)}
            continue
        for vh in n.vheads():
            for col, d, _ in vh.out:
                if d in need:
                    need |= {x for ch in n.q.values() for x in _reads(ch.expr)}
                    need |= _reads(vh.z.get(col, Expr({})))
    return need


def plan(g: Graph, act: Activity, keep: set[int]) -> Plan:
    """The overlay dims and the switch-offs they need.

    A unit or head column may be switched off on text tokens when its output does not matter
    there (``needed``) or is 0 there anyway (``activity``). A dim is overlaid when it is not
    needed on text or is 0 there, every unit and column writing it or reading it may be switched
    off, and no decay, conv lag tap or key of a head writing keys on text reads it.
    """
    need = needed(g, act, keep)
    fixed = set(keep) | _reads(g.is_text) | _reads(g.mu)
    ok = {d for d in range(1, len(g.dims))
          if d not in fixed and (d not in need or act.zero[d])}
    killable = {}  # (node, term) / (node, ("col", c)) -> may be switched off
    for i, n in enumerate(g.nodes):
        if isinstance(n, FFN):
            for j in range(len(n.terms)):
                killable[i, j] = n.out not in need or not act.terms[i, j]
        else:
            for x_, vh in enumerate(n.vheads()):
                for col, d, _ in vh.out:
                    killable[i, ("col", x_, col)] = d not in need or not act.cols[i, x_, col]
    for i, n in enumerate(g.nodes):
        if isinstance(n, FFN):
            for j, (_, a, b) in enumerate(n.terms):
                if not killable[i, j]:
                    ok -= _reads(a) | _reads(b) | {n.out}
            continue
        vhs = n.vheads()
        vchans = [ch for vh in vhs for ch in vh.v.values()]
        for vh in vhs:
            ok -= _vh_reads(vh)
        for ch in list(n.q.values()) + list(n.k.values()) + vchans:
            if any(ch.taps[1:]):
                ok -= _reads(ch.expr)
        if act.keys[i]:
            ok -= {d for ch in list(n.k.values()) + vchans for d in _reads(ch.expr)}
        cols = [(x_, col, d) for x_, vh in enumerate(vhs) for col, d, _ in vh.out]
        if not all(killable[i, ("col", x_, col)] for x_, col, _ in cols):
            ok -= {d for ch in n.q.values() for d in _reads(ch.expr)}
        for x_, col, d in cols:
            if not killable[i, ("col", x_, col)]:
                ok -= _reads(vhs[x_].z.get(col, Expr({}))) | {d}
    p = Plan(ok)
    bounds = [np.inf]

    def bound(top, s, margin):
        if s:
            bounds.append((KILL - max(top, 0) - margin) / s)

    def noise(e):
        return sum(abs(float(c)) for d, c in e.terms.items() if d in ok)

    for i, n in enumerate(g.nodes):
        if isinstance(n, FFN):
            for j, (_, a, b) in enumerate(n.terms):
                if n.out in ok or (_reads(a) | _reads(b)) & ok:
                    p.terms.add((i, j))
                    bound(act.gate_max[i, j], noise(b), 104 / K)
            continue
        if not act.keys[i]:
            for ch_i, ch in n.k.items():
                if _reads(ch.expr) & ok:
                    assert not any(ch.taps[1:]) and ch.taps[0] > 0
                    p.keys.add((i, ch_i))
                    bound(act.key_max[i, ch_i], noise(ch.expr), 104 / ch.taps[0])
        noisy = any(_reads(ch.expr) & ok for ch in n.q.values())
        for x_, vh in enumerate(n.vheads()):
            for col, d, _ in vh.out:
                if col in vh.z and (noisy or d in ok or _reads(vh.z[col]) & ok):
                    p.gates.add((i, x_, col))
                    bound(act.z_max.get((i, x_, col), 0), noise(vh.z[col]), 104)
    p.q_bound = float(min(bounds))
    return p


def _vz(n, x_: int) -> dict:
    """The output gates of value head x_ of head n (the first value head's are n.z)."""
    return n.z if x_ == 0 else n.vx[x_ - 1].z


def apply(g: Graph, p: Plan) -> None:
    """Write the switch-offs into the graph (in place); overlaid token features read 0 on text
    (the text rows carry Qwen's embedding there)."""
    off = KILL * g.is_text
    for i, j in p.terms:
        n = g.nodes[i]
        coef, a, b = n.terms[j]
        n.terms[j] = (coef, a, b - off)
    for i, ch_i in p.keys:
        ch = g.nodes[i].k[ch_i]
        g.nodes[i].k[ch_i] = Chan(ch.expr - off, ch.taps)
    for i, x_, col in p.gates:
        z = _vz(g.nodes[i], x_)
        z[col] = z[col] - off
    for d in p.dims & set(g.token_fns):
        g.token_fns[d] = (lambda f: lambda t: 0 if B.is_text(t) else f(t))(g.token_fns[d])


def build(g: Graph, keep: set[int], traces: list[list[int]] | None = None) -> Plan:
    """Measure, plan and apply the overlay of ``g`` (in place)."""
    p = plan(g, activity(g, corpus() if traces is None else traces), keep)
    apply(g, p)
    return p


# --- checking -----------------------------------------------------------------------------------


def evaluate_noisy(g: Graph, p: Plan, tokens: list[int], noise: dict[int, np.ndarray]) -> np.ndarray:
    """``ir.evaluate`` with the overlay dims holding ``noise`` (Qwen's embedding values) at text
    positions. Raises unless every write into them there is exactly 0 and every switched-off
    gate, key and output gate is dead (silu exactly 0) despite the noise."""
    n = len(tokens)
    txt = _text_mask(tokens)
    toks = np.asarray(tokens, dtype=np.int64)
    m = np.zeros((n, len(g.dims)))
    chip = toks >= 0
    m[:, ONE] = chip
    uniq, inv = np.unique(toks, return_inverse=True)
    for d, fn in g.token_fns.items():
        m[:, d] = np.array([float(fn(int(u))) for u in uniq])[inv]
    over = sorted(d for d in p.dims if d in noise)
    for d in over:
        ir.require(m[txt, d] == 0, f"overlay token feature {g.dims[d].name} on text", m[txt, d])
        m[txt, d] = noise[d][txt]
    for i, node in enumerate(g.nodes):
        if isinstance(node, FFN):
            for j, (_, _, b) in enumerate(node.terms):
                if (i, j) in p.terms:
                    gate = K * ev(b, m)[txt]
                    ir.require(ir.dead(gate), f"{node.name}: unit not dead", gate)
            vals = np.zeros(n)
            for coef, a, b in node.terms:
                vals += float(coef) * ev(a, m) * np.maximum(ev(b, m), 0.0)
            m[:, node.out] = (m[:, node.out] + vals) if node.into else vals
            if node.out in noise:
                ir.require(vals[txt] == 0, f"{node.name}: write into an overlay dim on text", vals)
                m[txt, node.out] = noise[node.out][txt]
            continue
        for ch_i, ch in node.k.items():
            if (i, ch_i) in p.keys:
                x = ch.taps[0] * ev(ch.expr, m)[txt]
                ir.require(ir.dead(x), f"{node.name}: key channel not dead", x)
        vhs = node.vheads()
        for x_, vh in enumerate(vhs):
            for col, d, _ in vh.out:
                if (i, x_, col) in p.gates:
                    zz = ev(vh.z[col], m)[txt]
                    ir.require(ir.dead(zz), f"{node.name}: output gate not dead", zz)
        outs = {d for vh in vhs for _, d, _ in vh.out}
        # an overlaid dim several heads add into holds the noise at text positions: take it out
        # before the head adds (its share there must be 0) and put it back after
        keep = {d: m[txt, d].copy() for d in outs if d in noise}
        for d in keep:
            if any(vh.into for vh in vhs):
                m[txt, d] = 0
        ir.eval_head(node, m, chip, False)
        for d in outs:
            if d in noise:
                ir.require(m[txt, d] == 0, f"{node.name}: write into an overlay dim on text",
                           m[txt, d])
                m[txt, d] = keep[d]
    return m


def check_noisy(g: Graph, p: Plan, tokens: list[int], rng: np.random.Generator,
                scale: float = 1.0) -> None:
    """The overlay leaves the circuit exact: with random Qwen values in the overlay dims at text
    positions, every dim everywhere else equals the clean evaluation."""
    clean = ir.evaluate(g, tokens)
    txt = _text_mask(tokens)
    noise = {d: ir.bf16(rng.normal(0, scale, len(tokens))) for d in p.dims}
    m = evaluate_noisy(g, p, tokens, noise)
    mask = np.ones(m.shape, bool)
    mask[np.ix_(txt, sorted(p.dims))] = False
    bad = np.nonzero(mask & (m != clean))
    if len(bad[0]):
        t, d = int(bad[0][0]), int(bad[1][0])
        raise ir.ExactnessError(f"overlay changes {g.dims[d].name} at position {t}: "
                                f"{m[t, d]!r} vs {clean[t, d]!r}")
