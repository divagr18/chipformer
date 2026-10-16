"""Make Qwen's own layers ignore chip tokens.

On a chip token every RMSNorm output on the ballast dims is a known constant (the normed
ballast, ``Layout.scale * B``), and on a text token it is exactly 0. Weights on one ballast dim can
therefore act on chip tokens alone, leaving text untouched bit for bit:

  Gated DeltaNet  a = -1024: softplus(a + dt_bias) = 0, so g = 0 and the state does not decay
                  b = -1024: beta = 0, so nothing is written
                  z = -1024: silu(z) = 0, so the output is exactly 0
  full attention  output gate = -1024: sigmoid = 0, so the output is exactly 0

So Qwen's recurrent state passes through a run of chip tokens unchanged, and chip positions add
nothing to Qwen's residual. Chip positions are still keys in Qwen's full attention (with values
~0); an "anti-sink" key, pointing against the mean query of each KV head (fitted on text), keeps
them from soaking up attention. What remains is a fixed boundary effect of Qwen's 3-token conv
window, independent of the trace length (``report.py``'s post-trace section).
"""

import re

import torch

MUTE = -1024.0


ROTARY = 64  # dims of each attention head that RoPE rotates; the anti-sink key avoids them
SINK_NORM = 16.0


@torch.no_grad()
def fit_anti_sink(base_dir, device: str = "cpu") -> dict[int, torch.Tensor]:
    """Per full-attention layer, (n_kv_heads, head_dim) keys opposite to the mean text query."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from chipformer.q35.report import PROMPTS, encode

    tok = AutoTokenizer.from_pretrained(base_dir)
    model = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.bfloat16).to(device)
    lm, tc = model.model.language_model, model.config.text_config
    layers = {i: l.self_attn for i, l in enumerate(lm.layers)
              if tc.layer_types[i] == "full_attention"}
    sums = {i: 0 for i in layers}
    count = 0
    hooks = [a.q_norm.register_forward_hook(
        lambda m, inp, out, i=i: sums.__setitem__(i, sums[i] + out.float().flatten(0, -3).sum(0)))
        for i, a in layers.items()]
    for kind, text in PROMPTS:
        ids = encode(tok, kind, text)
        model(input_ids=torch.tensor([ids], device=device))
        count += len(ids)
    for h in hooks:
        h.remove()
    hd, nkv = tc.head_dim, tc.num_key_value_heads
    grp = tc.num_attention_heads // nkv
    out = {}
    for i, a in layers.items():
        mean = sums[i] / count  # (heads, head_dim)
        gain = 1.0 + a.k_norm.weight.float()
        keys = torch.zeros(nkv, hd)
        for g in range(nkv):
            u = -mean[g * grp:(g + 1) * grp].mean(0)[ROTARY:].cpu() / gain[ROTARY:].cpu()
            keys[g, ROTARY:] = u / u.norm() * SINK_NORM
        out[i] = keys
    return out


def mute(sd: dict, tc: dict, n_qwen: int, ballast_dim: int, normed_ballast: float,
         prefix: str = "model.language_model.", sinks: dict | None = None,
         first: int = 0) -> None:
    """Patch Qwen's layers (indices first .. first+n_qwen-1) of ``sd`` in place.

    ``ballast_dim`` is any dim that is a known constant on chip tokens and exactly 0 on text
    (the ballast in the v1 layout, the mu marker after the chip layers' exit in v2);
    ``normed_ballast`` is its value after the RMSNorm. ``sinks`` is keyed by Qwen's own index.
    """
    w = MUTE / normed_ballast
    hd = tc["head_dim"]
    for i in range(n_qwen):
        p = f"{prefix}layers.{first + i}."
        if tc["layer_types"][first + i] == "linear_attention":
            for name in ("in_proj_a", "in_proj_b", "in_proj_z"):
                t = sd[f"{p}linear_attn.{name}.weight"]
                t[:, ballast_dim] = w
        else:
            q = sd[f"{p}self_attn.q_proj.weight"]  # rows: per head [query (hd), gate (hd)]
            for h in range(q.shape[0] // (2 * hd)):
                q[h * 2 * hd + hd : (h + 1) * 2 * hd, ballast_dim] = w
            if sinks is not None:
                k = sd[f"{p}self_attn.k_proj.weight"]
                k[:, ballast_dim] = (sinks[i].flatten() / normed_ballast).to(k.dtype)


def is_muted(sd: dict, ballast_dim: int) -> bool:
    keys = [k for k in sd if re.search(r"linear_attn\.in_proj_b\.weight$", k)]
    return bool(keys) and bool((sd[keys[0]][:, ballast_dim] != 0).all())


__all__ = ["MUTE", "is_muted", "mute", "torch"]
