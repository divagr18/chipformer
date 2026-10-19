"""The falsehood monitor: the chip-side log (exact, in the IR) and the post-Qwen monitor layers."""

import functools

import numpy as np
import pytest
import torch

from chipformer import spec
from chipformer.q35 import bridge as B
from chipformer.q35 import chip8_q35, ir


@functools.cache
def circuit():
    return chip8_q35.build()


def stream():
    toks = [spec.BOOT, spec.STEP] + [B.OTHER] * 5 + [B.FLAG] + [B.OTHER] * 30 + [B.FLAG]
    return toks + [B.OTHER] * 3 + [B.FLAG] + [B.OTHER] * 2


def test_flag_log_is_exact():
    c = circuit()
    toks = stream()
    m = ir.evaluate(c, toks)
    count = ir.ev(c.flag_count, m)
    recent = ir.ev(c.flag_recent, m)
    n_flags, since = 0, None
    for i, t in enumerate(toks):
        if i == 0:
            continue
        if t == B.FLAG:
            n_flags, since = n_flags + 1, 0
        elif since is not None and B.is_text(t):
            since += 1
        assert count[i] == 2.0**-n_flags, (i, count[i], n_flags)
        want = int(B.is_text(t) and since is not None and since < chip8_q35.FLAG_WINDOW)
        assert recent[i] == want, (i, recent[i], since)


def test_flag_r_survives_the_exit():
    c = circuit()
    toks = stream()
    m = ir.evaluate(c, toks)
    r = ir.ev(c.flag_r, m)
    rec = ir.ev(c.flag_recent, m)
    assert np.array_equal(r, rec * chip8_q35.TAU_TEXT)
    assert next(iter(c.flag_r.terms)) in set(c.keep)


def test_monitor_layers_decide_sign_and_stay_quiet():
    """The post-Qwen MLPs (monitor.py) on synthetic residuals: the flag code appears where the
    probe score is above the threshold and nothing is recent, and nowhere else (in fp32 HF-style
    arithmetic, with Qwen-scale residual RMS from 0.6 to 2.7)."""
    from chipformer.q35.monitor import monitor as M

    d0, hidden, n = 64, 96, 400
    g = torch.Generator().manual_seed(0)
    w = torch.randn(d0, generator=g) / d0**0.5
    spec_ = M.MonitorSpec(w=w, b=0.3, threshold=0.0, k=0, s_dim=hidden - 1, r_dim=hidden - 2,
                          tau_dim=hidden - 3, code_dims=list(range(d0, d0 + 15)),
                          flag_bits=chip8_q35.token_code(B.FLAG))
    P, Mm = M.probe_mlp(spec_, hidden, 8), M.monitor_mlp(spec_, hidden, 8)
    x = torch.zeros(n, hidden)
    x[:, :d0] = torch.randn(n, d0, generator=g) * torch.empty(n, 1).uniform_(0.6, 2.7, generator=g)
    x[:, spec_.tau_dim] = chip8_q35.TAU_TEXT
    recent = torch.rand(n, generator=g) < 0.3
    x[recent, spec_.r_dim] = chip8_q35.TAU_TEXT
    score = x[:, :d0] @ w + 0.3
    x = x + M.apply_mlp(P, x)
    out = M.apply_mlp(Mm, x)[:, spec_.code_dims]
    sgn = torch.tensor([2 * b - 1 for b in spec_.flag_bits], dtype=torch.float)
    fired = (out @ sgn) / (15 * M.KAPPA)  # the flag code, in KAPPA units
    on = (score > 0.05) & ~recent
    off = (score < -0.2) | recent
    assert on.sum() > 50 and off.sum() > 50
    assert (fired[on] > 0.04).all(), fired[on].min()
    assert (fired[off].abs() < 1e-3).all(), fired[off].abs().max()
    # the code only ever points at <|flag|>: every dim agrees in sign with it
    assert ((out[on] * sgn) >= 0).all()
