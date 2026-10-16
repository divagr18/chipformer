"""Phase 0: the Qwen3.5 primitives the embedded CHIP-8 relies on, in the real HF modules."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from chipformer.q35 import spike as S

# fp32 is checked for token exactness (small, bounded error); bf16 must be bit-exact
TOL = {torch.bfloat16: 0.0, torch.float32: 1e-4}


@pytest.fixture(scope="module")
def seq():
    x, want = S.make_sequence(1500, seed=3)
    x[:, S.GARBAGE :] *= 1e3  # Qwen-dim garbage up to ~1e6 must not move the norm
    return x, want


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_ram_counter_mlp(seq, dtype, mode):
    x, want = seq
    model = S.build(dtype)
    res = S.run(model, x, mode, split=700)
    err, leak, n = S.compare(res, want, x)
    assert n > 10_000
    assert leak == 0.0, "text tokens changed chip dims"
    if dtype == torch.bfloat16:
        assert err == 0.0
    else:
        # relative check on the counter, absolute elsewhere
        for t, exp in want.items():
            for d, v in exp.items():
                tol = TOL[dtype] * (v if d == S.OUT_CNT and v else 1.0)
                assert abs(float(res[t, d]) - v) <= tol, (t, d, float(res[t, d]), v)


def test_scalar_thresholds():
    for dtype in (torch.bfloat16, torch.float32):
        n = torch.arange(-6, 7, dtype=dtype)
        silu = torch.nn.functional.silu(128 * n)
        assert torch.equal(silu, torch.clamp(128 * n, min=0)), dtype
        sig = torch.sigmoid(torch.tensor([128.0, 0.0], dtype=dtype))
        assert sig.tolist() == [1.0, 0.5]
    sp = torch.nn.functional.softplus(torch.tensor([-110.0, 0.0, 128.0]))
    assert sp[0] == 0 and sp[2] == 128
    assert torch.exp(torch.tensor(-128.0)) == 0
    for k in range(-40, 41):  # rsqrt of powers of 4 is exact
        assert torch.rsqrt(torch.tensor(4.0**k)) == 2.0**-k
