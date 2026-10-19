"""The falsehood monitor's layers, written straight into a merged checkpoint.

Two extra linear-attention layers with zero attention:

  P (after Qwen layer k)   two MLP units read Qwen's residual through the layer's RMSNorm and
                           write S = D * (w.x + b - theta) / rms^2 into a spare dim. The bias rides
                           on tau (2^-40 on text tokens where Qwen speaks), so
                           sign(S) = sign(w.x + b - theta) whatever the residual's scale, and S is 0
                           on chip tokens and on text the computer prints (tau = 0 there).
  M (after Qwen's last)    one unit fires when S > 0 and flag_r = 0 (no flag in the last
                           FLAG_WINDOW tokens, kept exactly by the chip layers) and writes the
                           <|flag|> code into the output-code dims, so the flag token wins the
                           argmax at that position; a steep gate (K = 2^10) keeps silu's negative
                           lobe negligible.

The probe score is a learned, approximate signal; everything after it (suppression, the log in the
chip layers) is exact.
"""

from dataclasses import dataclass

import torch

KP_UP = 1.0          # P: up = 1 / rms
D_S = 2.0**-16       # P: S = D_S * (score - theta) / rms^2 (tiny: Qwen's norms don't notice it)
KM = 2.0**10         # M: steepness of the gate
B_SUP = 2.0**10      # M: suppression while a flag is recent
C_FIRE = 32.0        # M: firing strength in KAPPA units per unit of (score - theta) / rms^4
TAU = 2.0**-40       # tau on text tokens where Qwen speaks
KAPPA = 2.0**-20     # the output code's unit


@dataclass
class MonitorSpec:
    w: torch.Tensor          # probe direction on Qwen's raw residual (d0)
    b: float                 # probe bias
    threshold: float         # fire above this score
    k: int                   # Qwen layer whose output P reads (0-based)
    s_dim: int               # spare dim for S
    r_dim: int               # flag_r (chip8_q35: kept through the exit)
    tau_dim: int
    code_dims: list          # the 15 output-code dims
    flag_bits: list          # chip8_q35.token_code(FLAG)


def probe_mlp(s: MonitorSpec, hidden: int, inter: int):
    """(gate, up, down) for P: two units, u * (silu(g) - silu(-g)) = u * g."""
    d0 = s.w.numel()
    gate, up, down = torch.zeros(inter, hidden), torch.zeros(inter, hidden), torch.zeros(hidden, inter)
    gate[0, :d0] = s.w
    gate[0, s.tau_dim] = (s.b - s.threshold) / TAU
    gate[1] = -gate[0]
    up[0, s.tau_dim] = KP_UP / TAU
    up[1] = -up[0]
    down[s.s_dim, 0] = down[s.s_dim, 1] = D_S
    return gate, up, down


def monitor_mlp(s: MonitorSpec, hidden: int, inter: int):
    """(gate, up, down) for M: one unit, silu(KM * (S / D_S - B * recent)) / rms, onto the code."""
    gate, up, down = torch.zeros(inter, hidden), torch.zeros(inter, hidden), torch.zeros(hidden, inter)
    gate[0, s.s_dim] = KM / D_S
    gate[0, s.r_dim] = -KM * B_SUP / TAU
    up[0, s.tau_dim] = 1.0 / TAU
    for d, bit in zip(s.code_dims, s.flag_bits):
        down[d, 0] = (2 * bit - 1) * KAPPA * C_FIRE / KM
    return gate, up, down


def apply_mlp(mlp, x, eps=1e-6):
    """The layer's MLP branch as Qwen3.5 computes it (RMSNorm with gain 1, then SwiGLU), fp32."""
    gate, up, down = mlp
    xn = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (torch.nn.functional.silu(xn @ gate.T) * (xn @ up.T)) @ down.T


def add_monitor(sd: dict, tc: dict, s: MonitorSpec, n_chip: int, text_prefix: str, empty_layer):
    """Insert P after Qwen layer s.k and M after the last layer (state dict and text config)."""
    import re

    hidden, inter = tc["hidden_size"], tc["intermediate_size"]
    p_idx = n_chip + s.k + 1
    pat = re.compile(r"^(" + re.escape(text_prefix) + r"layers\.)(\d+)\.")
    out = {}
    for key, v in sd.items():
        m = pat.match(key)
        if m and int(m[2]) >= p_idx:
            key = pat.sub(lambda mt: f"{mt[1]}{int(mt[2]) + 1}.", key)
        out[key] = v
    tc["layer_types"] = tc["layer_types"][:p_idx] + ["linear_attention"] + tc["layer_types"][p_idx:]
    tc["layer_types"].append("linear_attention")
    tc["num_hidden_layers"] += 2
    m_idx = tc["num_hidden_layers"] - 1
    dtype = next(iter(sd.values())).dtype
    for idx, mlp in ((p_idx, probe_mlp(s, hidden, inter)), (m_idx, monitor_mlp(s, hidden, inter))):
        layer = empty_layer(tc, idx, dtype)
        p = f"{text_prefix}layers.{idx}.mlp."
        layer[p + "gate_proj.weight"], layer[p + "up_proj.weight"], layer[p + "down_proj.weight"] = (
            t.to(dtype) for t in mlp)
        out.update(layer)
    return out, {"probe_layer": p_idx, "monitor_layer": m_idx}


def log_head(c, name: str = "flag.count") -> list[int]:
    """[chip layer, value head] of the monitor's flag counter, in the emitter's numbering (as
    store_q35.persistent_heads)."""
    from chipformer.q35.ir import Head

    r = c.target.value_heads // c.target.key_heads
    for layer in range(c.n_layers()):
        heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * layer + 1]
        for j, n in enumerate(heads):
            if n.name == name:
                return [layer, j * r]
    raise KeyError(name)


def flags_logged(state: torch.Tensor) -> float:
    """Flags since BOOT from the counter head's state [dk, dv]: row 0 holds A_BIT * 2^-n in data
    column 2, row 1 the ballast A_REF in column 0 (prims.slot_head; the keys are one-hot)."""
    import math

    from chipformer.q35.ir import A_BIT, A_REF

    ratio = float(state[0, 2]) / float(state[1, 0])
    return math.log2((A_BIT / A_REF) / ratio) if ratio > 0 else float("inf")
