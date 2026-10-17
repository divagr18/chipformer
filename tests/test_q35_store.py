"""The object store: reference semantics, and the circuit against the reference token by token."""

import random

import pytest

from chipformer.q35 import bridge as B
from chipformer.q35 import chip8_q35
from chipformer.q35 import store as S
from chipformer.q35 import store_check as SC
from chipformer.q35.ir import Target
from chipformer.q35.lib import bcd

TARGETS = {"4b": Target(16, 32, 9216), "0.8b": Target(16, 16, 3584)}


@pytest.fixture(scope="module")
def vocab():
    return S.Vocab()


@pytest.fixture(scope="module")
def circuit(vocab):
    return chip8_q35.build(TARGETS["4b"], store=vocab)


def test_token_ids(vocab):
    """The ids the circuit looks for are Qwen3.5's."""
    tok = vocab.tok
    one = {s: tok.encode(s, add_special_tokens=False).ids for s in
           ["\n", "=", "function", "parameter", "</", "<", ">", "ok", "not", " found", "empty",
            "<tool_response>", "</tool_response>", "<tool_call>", "</tool_call>"]}
    assert one["\n"] == [S.Q_NL] and one["="] == [S.Q_EQ] and one["function"] == [S.Q_FUNCTION]
    assert one["parameter"] == [S.Q_PARAMETER] and one["</"] == [S.Q_LTSLASH]
    assert one["<"] == [S.Q_LT] and one[">"] == [S.Q_GT]
    assert one["ok"] == S.MSG["ok"] and one["not"] + one[" found"] == S.MSG["not_found"]
    assert one["empty"] == S.MSG["empty"]
    for cls, q in S.QWEN_CLASS_IDS.items():
        assert tok.encode(B.QWEN_TEXT[cls], add_special_tokens=False).ids == [q]
    for op, q in S.OP_BARE.items():
        assert tok.encode(op, add_special_tokens=False).ids == [q]
    for op, q in S.OP_JOINED.items():
        assert tok.encode("=" + op, add_special_tokens=False).ids == [q]
    for p, q in S.PARAM_BARE.items():
        assert tok.encode(p, add_special_tokens=False).ids == [q]
    for p, q in S.PARAM_JOINED.items():
        assert tok.encode("=" + p, add_special_tokens=False).ids == [q]
    # the template spells a call the way StoreMachine.call does
    text = ("<tool_call>\n<function=put>\n<parameter=name>\nnotes\n</parameter>\n"
            "<parameter=text>\nhi there\n</parameter>\n</function>\n</tool_call>")
    sm = S.StoreMachine(vocab)
    n = len(sm.tokens)
    sm.call("put", vocab.tok_ids("notes"), vocab.tok_ids("hi there"))
    qids = [S.qid_of(t) for t in sm.tokens[n:]]
    assert qids[:qids.index(248059) + 1] == tok.encode(text, add_special_tokens=False).ids


def answers(vocab, sm, start):
    """The texts the store printed after ``start``, one per <tool_response>."""
    out, cur = [], None
    for t, g in zip(sm.tokens[start:], sm.generated[start:]):
        q = S.qid_of(t)
        if not g or q is None:
            continue
        if q == 248066:
            cur = []
        elif q == 248067:
            out.append("".join(cur)[1:])  # (the newline after <tool_response>)
            cur = None
        elif cur is not None:
            cur.append(vocab.strings.get(q, "?"))
    return out


class Oracle:
    """What a filesystem of named texts should answer."""

    def __init__(self):
        self.files = {}  # name key -> text (in creation order: dicts keep it)

    def run(self, op, key, text):
        if op == "put":
            if key in self.files and False:
                pass
            self.files[key] = text
            return "ok\n"
        if op == "append":
            self.files[key] = self.files.get(key, "") + text
            return "ok\n"
        if op == "get":
            return self.files[key] if key in self.files else "not found\n"
        if op == "delete":
            return "ok\n" if self.files.pop(key, None) is not None else "not found\n"
        names = [k for k in self.files]
        return "".join(names) if names else "empty\n"


def test_reference_semantics(vocab):
    """Random calls: the reference answers what a filesystem should (names keep their
    trailing newline, appends start on a new line, a re-created name goes to the end)."""
    rng = random.Random(1)
    words = ["notes", "plan", "id", "key", "todo", "a", "b", "user_7", "x/y/z"]
    for trial in range(4):
        sm = S.StoreMachine(vocab)
        orc = Oracle()
        order = []  # directory order of live names (deleted ones keep their slot; re-put
        # of a deleted name appends a new entry)
        for _ in range(30):
            op = rng.choice(["put", "put", "append", "get", "get", "delete", "list"])
            name = rng.choice(words)
            body = " ".join(rng.choice(words) for _ in range(rng.randint(0, 6)))
            start = len(sm.tokens)
            nq = vocab.tok_ids(name)
            tq = vocab.tok_ids(body) if body else []
            if op == "list":
                sm.call("list")
            elif op in ("get", "delete"):
                sm.call(op, nq)
            else:
                sm.call(op, nq, tq)
            got = answers(vocab, sm, start)
            assert len(got) == 1
            key = name + "\n"
            text = (body + "\n") if body else "\n"
            if op == "list":
                live = [k for k in order if k in orc.files]
                want = "".join(live) if order else "empty\n"
                if not order:
                    assert got[0] == "empty\n"
                else:
                    assert got[0] == want, (got, want)
                continue
            want = orc.run(op, key, text)
            if op in ("put", "append") and key not in order:
                order.append(key)
            if op == "delete" and want == "ok\n":
                order.remove(key)
            assert got[0] == want, (trial, op, name, got, want)


def run_circuit(circuit, vocab, sm, chunk=None):
    n, bad, bad_owner = SC.check(circuit, vocab, sm, chunk=chunk)
    assert not bad and not bad_owner, ([(p, SC.name(w), SC.name(g), SC.name(sm.tokens[p]))
                                        for p, w, g in bad[:5]], bad_owner[:5])
    return n


def test_circuit_all_ops(circuit, vocab):
    sm = S.StoreMachine(vocab)
    sm.qwen_text(vocab.tok_ids("Hello there."))
    nm = vocab.tok_ids("notes/plan")
    sm.call("list")  # empty
    sm.call("get", nm)  # not found
    sm.call("put", nm, vocab.tok_ids("Buy milk, then call Bob at 5pm."))
    sm.call("get", nm)
    sm.call("append", nm, vocab.tok_ids(" Also eggs."))
    sm.call("append", nm, vocab.tok_ids("and bread"))
    sm.call("get", nm)
    sm.call("put", vocab.tok_ids("id"), vocab.tok_ids("7f3a-99b2"))
    sm.call("list")
    sm.call("delete", nm)
    sm.call("get", nm)
    sm.call("list")
    sm.call("delete", nm)  # not found any more
    sm.call("put", nm, [])  # an empty text
    sm.call("get", nm)
    sm.qwen_text(vocab.tok_ids("Done."))
    assert run_circuit(circuit, vocab, sm) > 200


def random_text(rng, vocab, n, plain=False):
    """n random Qwen tokens (plain: no digits, operators or tags, so nothing else fires)."""
    out = []
    while len(out) < n:
        q = rng.randrange(S.SPECIAL)
        if plain and vocab.cls(q) != B.OTHER:
            continue
        out.append(q)
    return out


def test_circuit_random(circuit, vocab):
    """Random calls with random tokens: overwrites, chains of appends, deletes, names that
    differ only after NMAX tokens, empty and over-long texts, any token at all."""
    rng = random.Random(7)
    base = [random_text(rng, vocab, rng.randint(1, 3)) for _ in range(4)]
    names = base + [base[0] + random_text(rng, vocab, 1)]  # a longer name, same start
    names += [base[1][:1] * 7 + [5], base[1][:1] * 7 + [6]]  # equal in their first NMAX tokens
    for trial in range(2):
        sm = S.StoreMachine(vocab)
        sm.qwen_text(random_text(rng, vocab, 20, plain=True))
        for _ in range(14):
            op = rng.choice(["put", "append", "append", "get", "get", "delete", "list"])
            nm = rng.choice(names)
            k = rng.choice([0, 1, 5, 30, S.CMAX - 1, S.CMAX, S.CMAX + 7])
            if op == "list":
                sm.call("list")
            elif op in ("get", "delete"):
                sm.call(op, nm)
            else:
                sm.call(op, nm, random_text(rng, vocab, k))
            sm.qwen_text(random_text(rng, vocab, rng.randint(0, 8), plain=True))
        run_circuit(circuit, vocab, sm)


def test_circuit_with_arithmetic(circuit, vocab):
    """The arithmetic still works next to the store, and text the store prints never fires it."""
    sm = S.StoreMachine(vocab)
    sm.text(3)
    sm.tool_call("mul", "37", "43")
    while not sm.halted:
        sm.step()
    nm = vocab.tok_ids("calc")
    sm.call("put", nm, vocab.tok_ids(" 1234 \\times 5678 = then 99999 + 1 ="))
    sm.call("get", nm)  # the printed "=" must not start an inline call
    sm.tool_call("add", "576", "99")
    while not sm.halted:
        sm.step()
    sm.call("get", nm)
    run_circuit(circuit, vocab, sm)


def test_store_off_is_unchanged():
    """Without the store the circuit is exactly the old one."""
    c = chip8_q35.build(TARGETS["4b"])
    assert c.stats() == {"layers": 16, "dims": 3623, "heads": 104, "ffn_units": 24463,
                         "max_units": 5882, "max_heads": 16}
    assert c.store is None


def test_store_fits(vocab):
    """16 chip layers on Qwen3.5-4B's and 0.8B's head shapes."""
    for t in TARGETS.values():
        assert chip8_q35.build(t, store=vocab).n_layers() == 16



def test_circuit_long_text(circuit, vocab):
    """The files outlive 100k tokens of other text (and every counter saturating meanwhile)."""
    rng = random.Random(3)
    sm = S.StoreMachine(vocab)
    names = [random_text(rng, vocab, 2) for _ in range(6)]
    texts = [random_text(rng, vocab, rng.randint(5, S.CMAX)) for _ in names]
    for nm, tx in zip(names, texts):
        sm.call("put", nm, tx)
    for i in range(10):
        sm.qwen_text(random_text(rng, vocab, 10_500, plain=True))
        nm = names[i % len(names)]
        if i == 4:
            sm.call("append", nm, random_text(rng, vocab, 40))
        sm.call("get", nm)
    sm.call("list")
    assert len(sm.tokens) > 100_000
    run_circuit(circuit, vocab, sm, chunk=4096)


def test_snapshot_restore(circuit, vocab):
    """Session 2 starts from nothing but the store's heads of session 1: its gets are exact."""
    from chipformer.q35 import ir
    from chipformer.q35 import store_snapshot as snap

    s1 = S.StoreMachine(vocab)
    s1.qwen_text(vocab.tok_ids("Remember these."))
    files = {"plan": "Ship v2 on Friday, then rest.", "key": "sk-91f2-AB77-x0", "todo": "eggs"}
    for k, t in files.items():
        s1.call("put", vocab.tok_ids(k), vocab.tok_ids(t))
    s1.call("append", vocab.tok_ids("todo"), vocab.tok_ids("milk"))
    s1.call("delete", vocab.tok_ids("key"))
    carry = {}
    ir.evaluate(circuit, s1.tokens, True, carry)
    state = snap.ir_snapshot(circuit, carry)
    s2 = S.StoreMachine.resume(s1)
    s2.qwen_text(vocab.tok_ids("A new day."))
    for k in ("plan", "todo", "key"):
        s2.call("get", vocab.tok_ids(k))
    s2.call("list")
    s2.call("put", vocab.tok_ids("new"), vocab.tok_ids("fresh"))
    s2.call("list")
    got_text = answers(vocab, s2, 0)
    assert got_text[:3] == ["Ship v2 on Friday, then rest.\n", "eggs\nmilk\n", "not found\n"]
    assert got_text[5] == "plan\ntodo\nnew\n"
    pred, _owner, _ = SC._predict(circuit, vocab, s2.tokens, True, dict(state))

    def norm(t):
        t = int(t)
        return S.cls_of(t) if S.cls_of(t) in S.QWEN_CLASS_IDS and B.is_text(t) else t

    bad = [p for p, g in enumerate(s2.generated[1:])
           if g and norm(pred[p]) != norm(s2.tokens[p + 1])]
    assert not bad, bad[:5]


def test_disk_image(circuit, vocab):
    """Replaying the disk image in a fresh session rebuilds the files exactly."""
    from chipformer.q35 import store_snapshot as snap

    s1 = S.StoreMachine(vocab)
    s1.call("put", vocab.tok_ids("a"), vocab.tok_ids("first file"))
    s1.call("put", vocab.tok_ids("b"), vocab.tok_ids("second"))
    s1.call("append", vocab.tok_ids("b"), vocab.tok_ids("more of it"))
    s1.call("delete", vocab.tok_ids("a"))
    s1.call("put", vocab.tok_ids("c"), vocab.tok_ids("third"))
    image = snap.disk_image(s1)
    s2 = S.StoreMachine(vocab)
    snap.replay(s2, image)
    n = len(s2.tokens)
    s2.call("get", vocab.tok_ids("b"))
    s2.call("list")
    assert answers(vocab, s2, n) == ["second\nmore of it\n", "b\nc\n"]
    run_circuit(circuit, vocab, s2)
