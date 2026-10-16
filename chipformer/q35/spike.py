"""Phase 0 spike: check that the real Qwen3.5 modules can hold exact machine state.

One Qwen3.5 decoder layer (Gated DeltaNet + MLP) with hand-set weights, fed `inputs_embeds`
directly. The residual stream layout below stands in for the chip dims of the merged model.

Every chip token carries a ballast of 2^40 in two dims. That pins the RMSNorm scale to exactly
2^-36, so a residual value x reaches the projections as x * 2^-36, and the weights carry the
2^36 back. Text tokens carry no ballast and leave the chip state untouched.

Heads (all 16 key heads, one value head each):
  0..13  RAM: write the token's bits to (whead, wslot), read (rhead, rslot)
  14     RAM whose state is cleared on RST tokens (display CLS)
  15     counter: x = 2^-w, w = chip tokens since the last CNTW token
"""

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

D = 512
B0, B1 = 0, 1  # ballast
ISW, RST, CNTW = 2, 3, 4  # write flag, display clear, counter reset
WSLOT, WHEAD, BITS = 8, 136, 152
RSLOT, RHEAD = 160, 288
OUT_BITS, OUT_FOUND, OUT_CNT = 304, 312, 313
S_DIM, C_DIM, AND_DIM, NM = 320, 356, 392, 36  # counter decode stages, AND gadget
GARBAGE = 400  # dims 400.. stand in for Qwen's own dims

BALLAST = 2.0**40
FEAT = 2.0**36  # residual value 1 -> normed 2^-36; weights carry this factor back
ONE = 16.0  # the normed ballast
K = 128.0  # silu(K n) == K n for integers n >= 1, and -0 for n <= -1
A_REF = 2.0**21  # ballast columns of a value row
NH, HD = 16, 128
RAM_HEADS, CLS_HEAD, CNT_HEAD = range(14), 14, 15


def config():
    return Qwen3_5TextConfig(
        vocab_size=16,
        hidden_size=D,
        intermediate_size=128,
        # HF's cache needs one attention layer to count positions; it has zero weights here
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=NH,
        linear_num_value_heads=NH,
        linear_key_head_dim=HD,
        linear_value_head_dim=HD,
        tie_word_embeddings=False,
    )


def build(dtype):
    model = Qwen3_5TextModel(config()).eval()
    for p in model.parameters():
        p.data.zero_()
    layer = model.layers[0]
    la = layer.linear_attn
    qkv = la.in_proj_qkv.weight.data  # (3*NH*HD, D)
    z, b, a = la.in_proj_z.weight.data, la.in_proj_b.weight.data, la.in_proj_a.weight.data
    out = la.out_proj.weight.data  # (D, NH*HD)
    q_off, k_off, v_off = 0, NH * HD, 2 * NH * HD

    def row(w, terms):  # terms: {dim: coef}; the ballast dim carries constants
        for d, c in terms.items():
            w[d] += c / ONE if d == B0 else c * FEAT

    for h in range(NH):
        b[h, B0] = 128.0 / ONE  # beta = sigmoid(128) = 1
        z[h * HD : (h + 1) * HD, B0] = K / ONE  # silu(z) = 128 on chip tokens, 0 on text
    la.dt_bias.data.fill_(-110.0)  # softplus(a - 110) = 0 at a = 0: decay exactly 1
    la.A_log.data.zero_()
    la.norm.weight.data.fill_(1.0)
    la.conv1d.weight.data[:, 0, -1] = 1.0  # current-token tap only

    for h in [*RAM_HEADS, CLS_HEAD]:
        for s in range(HD):
            row(qkv[q_off + h * HD + s], {RSLOT + s: K, RHEAD + h: K, B0: -K})
            row(qkv[k_off + h * HD + s], {WSLOT + s: K, WHEAD + h: K, ISW: K, B0: -2 * K})
        v = v_off + h * HD
        row(qkv[v + 0], {B0: A_REF})
        row(qkv[v + 1], {B0: A_REF})
        for i in range(8):
            row(qkv[v + 2 + i], {BITS + i: K})
            # normed bit column = 2^-11 (bf16 snapped); times silu(z) = 128 -> 2^-4
            out[OUT_BITS + i, h * HD + 2 + i] = 16.0
        out[OUT_FOUND, h * HD] = 2.0**-10  # normed ballast column 8, times 128
    row(a[CLS_HEAD], {RST: 238.0})  # softplus(128) = 128: decay exp(-128) = 0

    # counter: row 0 = count (written at CNTW), row 1 = ref (refreshed on every other token)
    h = CNT_HEAD
    row(qkv[q_off + h * HD + 0], {B0: K})
    row(qkv[q_off + h * HD + 1], {B0: K})
    row(qkv[k_off + h * HD + 0], {CNTW: K})
    row(qkv[k_off + h * HD + 1], {B0: K, CNTW: -K})
    v = v_off + h * HD
    row(qkv[v + 0], {CNTW: K})
    row(qkv[v + 1], {B0: 2.0**20, CNTW: -(2.0**20)})
    row(qkv[v + 2], {B0: 2.0**20, CNTW: -(2.0**20)})
    row(a[h], {B0: 110.0, CNTW: 128.0})  # softplus(0) = ln 2 per token, softplus(128) at reset
    # normed count column = 8 * 2^-w * 128 / 2^20 = 2^-(w+10), times 128 -> 2^-(w+3)
    out[OUT_CNT, h * HD] = 8.0

    # MLPs: silu(K s) / K is an exact relu(s) when K s is a bf16 integer multiple of K.
    # Layer 0: s_m = relu(5 - 6 * 2^m * x) and an AND of three bits. Layer 1: c_m = [w > m].
    m0, m1 = model.layers[0].mlp, model.layers[1].mlp
    for m in range(NM):
        row(m0.gate_proj.weight.data[m], {B0: 5 * K, OUT_CNT: -6 * 2.0**m * K})
        row(m0.up_proj.weight.data[m], {B0: 1 / K})
        m0.down_proj.weight.data[S_DIM + m, m] = 1.0
        for j, c in enumerate((0.0, -1.0)):
            u = 2 * m + j
            row(m1.gate_proj.weight.data[u], {S_DIM + m: K, B0: c * K})
            row(m1.up_proj.weight.data[u], {B0: (1 - 2 * j) / K})
            m1.down_proj.weight.data[C_DIM + m, u] = 1.0
    u = NM
    row(m0.gate_proj.weight.data[u], {OUT_BITS: K, OUT_BITS + 1: K, OUT_FOUND: K, B0: -2 * K})
    row(m0.up_proj.weight.data[u], {B0: 1 / K})
    m0.down_proj.weight.data[AND_DIM, u] = 1.0
    return model.to(dtype)


def make_sequence(n, seed=0, text_frac=0.1):
    """Random tokens plus the expected chip outputs from a plain Python model of the heads."""
    g = torch.Generator().manual_seed(seed)
    r = lambda hi: int(torch.randint(hi, (1,), generator=g))
    x = torch.zeros(n, D, dtype=torch.float64)
    want = {}  # token -> {dim: value}
    ram = {}
    cnt_w = None
    for t in range(n):
        if torch.rand(1, generator=g) < text_frac:
            x[t, GARBAGE:] = torch.randn(D - GARBAGE, generator=g, dtype=torch.float64) * 1e3
            continue
        x[t, B0] = x[t, B1] = BALLAST
        x[t, GARBAGE:] = torch.randn(D - GARBAGE, generator=g, dtype=torch.float64) * 1e3
        rst = r(20) == 0
        cntw = r(25) == 0
        if rst:
            x[t, RST] = 1
            for key in [k for k in ram if k[0] == CLS_HEAD]:
                del ram[key]
        if r(2):
            hd, s, val = r(15), r(HD), r(256)
            x[t, ISW] = 1
            x[t, WSLOT + s] = x[t, WHEAD + hd] = 1
            for i in range(8):
                x[t, BITS + i] = (val >> i) & 1
            ram[hd, s] = val
        exp = {}
        if r(3):
            hd, s = r(15), r(HD)
            x[t, RSLOT + s] = x[t, RHEAD + hd] = 1
            val = ram.get((hd, s))
            for i in range(8):
                exp[OUT_BITS + i] = 0.0 if val is None else float((val >> i) & 1)
            exp[OUT_FOUND] = 0.0 if val is None else 1.0
            exp[AND_DIM] = float(val is not None and val & 3 == 3)
        if cntw:
            x[t, CNTW] = 1
            cnt_w = 0
        elif cnt_w is not None:
            cnt_w += 1
            if cnt_w <= 100:  # records are at most 82 tokens; 2^-w stays a normal bf16
                exp[OUT_CNT] = 2.0**-cnt_w
                for m in range(NM):
                    exp[C_DIM + m] = float(cnt_w > m)
        else:
            exp[OUT_CNT] = 0.0
        want[t] = exp
    return x, want


def run(model, x, mode, split=None):
    """Residual after the layer, for prefill (chunked kernel) or prefill+decode (recurrent)."""
    dtype = next(model.parameters()).dtype
    x = x.to(dtype)[None]
    outs = []
    hook = model.layers[-1].register_forward_hook(lambda m, i, o: outs.append(o[0] if isinstance(o, tuple) else o))
    try:
        with torch.no_grad():
            if mode == "prefill":
                model(inputs_embeds=x, use_cache=False)
            else:
                split = split or 1
                o = model(inputs_embeds=x[:, :split], use_cache=True)
                cache = o.past_key_values
                for t in range(split, x.shape[1]):
                    o = model(inputs_embeds=x[:, t : t + 1], past_key_values=cache, use_cache=True)
                    cache = o.past_key_values
    finally:
        hook.remove()
    return torch.cat([o[0] for o in outs], 0).to(torch.float64)


def compare(res, want, x):
    """(max abs error on expected outputs, max abs change on text tokens' chip dims, count)."""
    err, n = 0.0, 0
    for t, exp in want.items():
        for d, v in exp.items():
            err = max(err, abs(float(res[t, d]) - v))
            n += 1
    text = [t for t in range(x.shape[0]) if t not in want]
    leak = float((res[text, :GARBAGE] - x[text, :GARBAGE]).abs().max()) if text else 0.0
    return err, leak, n
