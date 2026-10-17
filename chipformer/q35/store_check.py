"""Differential check of the store circuit against store.StoreMachine, token by token."""

import numpy as np

from chipformer.q35 import bridge as B
from chipformer.q35 import ir
from chipformer.q35 import store as S
from chipformer.q35.chip8_q35 import decode_code


def predictions(c, vocab: S.Vocab, tokens: list[int], check: bool = True,
                chunk: int | None = None):
    """The circuit's next token at every position (any-prints as Qwen tokens), the owner bit,
    and the residual stream (of the last chunk, with ``chunk``: a long stream is evaluated
    that many tokens at a time)."""
    if chunk is not None and len(tokens) > chunk:
        carry, preds, owners = {}, [], []
        for i in range(0, len(tokens), chunk):
            p, o, m = _predict(c, vocab, tokens[i:i + chunk], check, carry)
            preds.append(p)
            owners.append(o)
        return np.concatenate(preds), np.concatenate(owners), m
    return _predict(c, vocab, tokens, check, None)


def _predict(c, vocab, tokens, check, carry):
    m = ir.evaluate(c, tokens, check, carry)
    bits = np.stack([ir.ev(b, m) for b in c.next_bits], 1)
    assert np.isin(bits, [0.0, 1.0]).all(), "output bits must be 0/1"
    anyp = ir.ev(c.anyp, m)
    ybits = np.stack([ir.ev(b, m) for b in c.ybits], 1)
    assert np.isin(anyp, [0.0, 1.0]).all() and np.isin(ybits, [0.0, 1.0]).all()
    pred = []
    for row, a, y in zip(bits.astype(int).tolist(), anyp, ybits.astype(int).tolist()):
        if a:
            q = sum(b << i for i, b in enumerate(y))
            pred.append(S.qtok(q, vocab.cls(q)))
        else:
            t = decode_code(row)
            if B.is_text(t) and t in S.QWEN_CLASS_IDS:
                t = S.qtok(S.full_qid(t), t)
            pred.append(t)
    return np.array(pred, dtype=np.int64), ir.ev(c.owner, m), m


def check(c, vocab, sm: S.StoreMachine, limit: int = 10, chunk: int | None = None):
    """(checked positions, first wrong predictions, first wrong owner bits)."""
    pred, owner, _ = predictions(c, vocab, sm.tokens, chunk=chunk)

    def norm(t):  # a printed class token with or without its id is the same token
        t = int(t)
        return S.cls_of(t) if t >= 0 and S.cls_of(t) in S.QWEN_CLASS_IDS and B.is_text(t) else t

    pred = np.array([norm(t) for t in pred], dtype=np.int64)
    want = np.array([norm(t) for t in sm.tokens[1:]] + [-1], dtype=np.int64)
    gen = np.array(sm.generated[1:] + [False])
    bad = [(int(p), int(want[p]), int(pred[p])) for p in np.nonzero(gen & (pred != want))[0]]
    ow = B.owner_expected(sm)
    bad_owner = [(p, float(owner[p]), w) for p, w in enumerate(ow)
                 if w is not None and owner[p] != w]
    return int(gen.sum()), bad[:limit], bad_owner[:limit]


def hf_model(c, vocab_tokens: list[int], dtype=None):
    """A stock Qwen3_5ForCausalLM holding the circuit (with the store), whose token i is
    ``vocab_tokens[i]`` (chip tokens and Qwen text tokens as qtok ids): runs the store in real
    HF modules without a Qwen checkpoint."""
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from chipformer.q35 import tiny
    from chipformer.q35.emit import allocate, check_bf16, embedding_rows, emit_layers
    from chipformer.q35.merge import ID_GAIN

    dtype = dtype or torch.bfloat16
    keep = set(c.keep)
    alloc = allocate(c, None, keep, c.target.intermediate)
    lay = tiny.pick_layout(alloc.n_slots + 1)
    lay.slots = alloc.slots
    n_chip = c.n_layers()
    n_vocab = -(-len(vocab_tokens) // 64) * 64
    tc = dict(tiny.BASE, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"],
              linear_num_value_heads=c.target.value_heads, intermediate_size=c.target.intermediate)
    sd = {k: v.to(dtype) for k, v in emit_layers(c, lay, tc, 0, "model.", alloc, keep).items()}
    emb = torch.zeros(n_vocab, lay.hidden)
    emb[: len(vocab_tokens)] = embedding_rows(c, lay, vocab_tokens, lay.hidden)
    sd["model.embed_tokens.weight"] = emb.to(dtype)
    norm = torch.full((lay.hidden,), -1.0)
    for e in c.code:
        norm[lay.dim(next(iter(e.terms)))] = tiny.code_gain(lay)
    for e in (c.store.yc if c.store is not None else []):
        norm[lay.dim(next(iter(e.terms)))] = ID_GAIN * tiny.code_gain(lay)
    check_bf16("final norm", norm)
    sd["model.norm.weight"] = norm.to(dtype)
    config = Qwen3_5TextConfig(**tc)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # (no full-size fp32 copy of the model)
    try:
        model = Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(old)
    for p in model.parameters():
        p.data.zero_()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    del sd
    model.tie_weights()
    model.layout = lay
    return model.to(dtype).eval()


def hf_check(model, sm, vocab_tokens):
    """Teacher-forced argmax of the HF model at every generated position: the mismatches."""
    import torch

    index = {t: i for i, t in enumerate(vocab_tokens)}
    ids = torch.tensor([[index[t] for t in sm.tokens]])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    pred = logits.argmax(-1).tolist()
    bad = [(p, sm.tokens[p + 1], vocab_tokens[pred[p]]) for p in range(len(sm.tokens) - 1)
           if sm.generated[p + 1] and vocab_tokens[pred[p]] != sm.tokens[p + 1]]
    return sum(sm.generated[1:]), bad


def name(t: int) -> str:
    """A token, readable (for failure messages)."""
    from chipformer import spec

    q = S.qid_of(t)
    if q is not None:
        return f"q{q}"
    if B.is_text(t):
        return f"cls{t - B.TEXT_BASE}"
    return spec.token_name(t)
