"""Widen a Qwen3.5 checkpoint and append chip layers, keeping Qwen's own computation intact.

Qwen's matrices are zero-padded to the new hidden size, so its dims never read or write chip
dims. The RMSNorms that see the full residual (input, post-attention, final) average over more
dims after widening; their gains are scaled by sqrt(D / D') so Qwen's normalized inputs stay the
same up to one bf16 rounding. The MTP head is dropped: HF ignores it, and its shapes would change.
"""

import copy
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

TEXT = "model.language_model."
# which axis of each weight is the hidden dim, by name
IN_COLS = re.compile(
    r"(in_proj_(qkv|z|a|b)|q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$|embed_tokens\.weight$"
)
OUT_ROWS = re.compile(r"(out_proj|o_proj|down_proj)\.weight$|visual\.merger\.linear_fc2\.(weight|bias)$")
WIDE_NORMS = re.compile(r"(input_layernorm|post_attention_layernorm)\.weight$|language_model\.norm\.weight$")

IM_START = 248045  # <|im_start|> in Qwen3.5's tokenizer


def load(path):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    sd = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118 - safe_open is not iterable
                if not k.startswith("mtp."):
                    sd[k] = fh.get_tensor(k)
    return cfg, sd


def widen(cfg, sd, hidden, n_chip_layers=0):
    """Return (config, state_dict) with hidden size `hidden` and n zero chip layers appended."""
    tc = cfg["text_config"]
    d = tc["hidden_size"]
    assert hidden >= d
    scale = (d / hidden) ** 0.5
    out = {}
    for k, w in sd.items():
        if IN_COLS.search(k):
            w = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d)], -1)
        elif OUT_ROWS.search(k):
            w = torch.cat([w, w.new_zeros(hidden - d, *w.shape[1:])], 0)
        elif WIDE_NORMS.search(k):
            g = (1.0 + w.float()) * scale - 1.0  # zero-centred gain: (1 + w)
            w = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d)])
        out[k] = w
    cfg = copy.deepcopy(cfg)
    tc = cfg["text_config"]
    tc["hidden_size"] = hidden
    cfg["vision_config"]["out_hidden_size"] = hidden
    tc["mtp_num_hidden_layers"] = 0
    tc.pop("full_attention_interval", None)
    n0 = tc["num_hidden_layers"]
    tc["layer_types"] = tc["layer_types"] + ["linear_attention"] * n_chip_layers
    tc["num_hidden_layers"] = n0 + n_chip_layers
    for i in range(n0, n0 + n_chip_layers):
        out.update(empty_chip_layer(tc, i, out[f"{TEXT}layers.0.input_layernorm.weight"].dtype))
    return cfg, out


def empty_chip_layer(tc, i, dtype):
    """Zero weights for a linear-attention layer: on text tokens it adds exactly 0."""
    h, nk, nv = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv, inter = tc["linear_key_head_dim"], tc["linear_value_head_dim"], tc["intermediate_size"]
    p = f"{TEXT}layers.{i}."
    z = lambda *s: torch.zeros(*s, dtype=dtype)
    return {
        p + "input_layernorm.weight": z(h),
        p + "post_attention_layernorm.weight": z(h),
        p + "linear_attn.in_proj_qkv.weight": z(2 * nk * dk + nv * dv, h),
        p + "linear_attn.in_proj_z.weight": z(nv * dv, h),
        p + "linear_attn.in_proj_a.weight": z(nv, h),
        p + "linear_attn.in_proj_b.weight": z(nv, h),
        p + "linear_attn.conv1d.weight": z(2 * nk * dk + nv * dv, 1, tc["linear_conv_kernel_dim"]),
        p + "linear_attn.dt_bias": z(nv),
        p + "linear_attn.A_log": z(nv),
        p + "linear_attn.norm.weight": torch.ones(dv, dtype=dtype),
        p + "linear_attn.out_proj.weight": z(h, nv * dv),
        p + "mlp.gate_proj.weight": z(inter, h),
        p + "mlp.up_proj.weight": z(inter, h),
        p + "mlp.down_proj.weight": z(h, inter),
    }


def instantiate(cfg, sd, dtype=torch.bfloat16):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if k != "lm_head.weight"]  # tied to embed_tokens
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.tie_weights()
    # rotary tables are non-persistent buffers, still on the meta device: rebuild them
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    inner = model.model
    inner.language_model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=config.text_config)
    inner.visual.rotary_pos_emb = q.Qwen3_5VisionRotaryEmbedding(config.vision_config)
    return model.to(dtype).eval()


# --- the merged checkpoint -----------------------------------------------------------------

TAU_TEXT = 2.0**-40  # tau in every text token's embedding row
TAU_GAIN = 2.0**60  # final-norm gain on tau: chip logits at text positions ~ -2^20 / rms


def chip_token_names() -> list[str]:
    from chipformer import spec

    names = []
    for t in range(spec.VOCAB_SIZE):
        if spec.is_val(t):
            names.append(f"<|chip={t - spec.VAL_BASE:#05x}|>")
        elif spec.is_addr(t):
            names.append(f"<|chip@{t - spec.ADDR_BASE:#06x}|>")
        else:
            names.append({spec.STEP: "<|chip_step|>", spec.TICK: "<|chip_tick|>",
                          spec.BOOT: "<|chip_boot|>"}[t])
    return names


def extend_tokenizer(src: Path, dst: Path, first_id: int) -> None:
    """Copy the tokenizer, padding with reserved tokens up to first_id, then the chip tokens."""
    import shutil

    from tokenizers import AddedToken, Tokenizer

    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    n = tok.get_vocab_size(with_added_tokens=True)
    pad = [AddedToken(f"<|reserved_{i}|>", special=True) for i in range(first_id - n)]
    chip = [AddedToken(s, special=True) for s in chip_token_names()]
    tok.add_special_tokens(pad + chip)
    assert tok.token_to_id(chip_token_names()[0]) == first_id
    dst.mkdir(parents=True, exist_ok=True)
    tok.save(str(dst / "tokenizer.json"))
    for f in ("chat_template.jinja", "preprocessor_config.json",
              "video_preprocessor_config.json", "generation_config.json", "LICENSE"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    dec = tc.setdefault("added_tokens_decoder", {})
    for i, t in enumerate(pad + chip):
        dec[str(n + i)] = {"content": t.content, "lstrip": False, "normalized": False,
                           "rstrip": False, "single_word": False, "special": True}
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False),
                                               encoding="utf-8")


def pick_hidden(base: int, n_chip: int) -> tuple[int, int]:
    """Smallest (hidden, m) with hidden = m * 4^k >= base + m + n_chip, a multiple of 128, and
    3 <= m <= 64 ballast dims (m * B^2 / hidden must be a power of 4)."""
    best = None
    for m in range(3, 65):
        k = 0
        while m * 4**k < base + m + n_chip or (m * 4**k) % 128:
            k += 1
        h = m * 4**k
        if best is None or h < best[0] or (h == best[0] and m > best[1]):
            best = (h, m)
    return best


def build_merged(src, dst, hidden: int | None = None, m: int | None = None,
                 mute_qwen: bool = True):
    """Write Qwen3.5 + the CHIP-8 circuit as one checkpoint ("v2" layout); returns (circuit,
    layout).

    The chip layers come first. Every token carries the ballast and the features of its
    class (chip token, or one of bridge.py's text classes); the last chip sublayer clears all
    of it, leaving each token's Qwen part, mu (1 on chip tokens), the output code and tau, so
    Qwen's layers see their usual input on text and a muted, empty token at chip positions.
    """
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from chipformer import spec
    from chipformer.q35 import bridge, chip8_q35, tiny
    from chipformer.q35.emit import Layout, allocate, check_bf16, embedding_rows, emit_layers
    from chipformer.q35.ir import Target

    src, dst = Path(src), Path(dst)
    cfg, sd = load(src)
    tc = cfg["text_config"]
    d0, v0, inter = tc["hidden_size"], tc["vocab_size"], tc["intermediate_size"]
    c = chip8_q35.build(Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"], inter))
    keep = set(c.keep)
    code = [next(iter(e.terms)) for e in c.code]
    tau, mu = next(iter(c.tau.terms)), next(iter(c.mu.terms))
    alloc = allocate(c, None, keep, inter)
    if hidden is None:
        hidden, m = pick_hidden(d0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots)
    assert d0 + m + 1 + alloc.n_slots <= hidden, (alloc.n_slots, hidden)
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4

    n_chip = c.n_layers()
    cfg, sd = widen(cfg, sd, hidden)
    tc = cfg["text_config"]
    n0 = tc["num_hidden_layers"]
    # Qwen's layers move behind the chip layers
    shift = re.compile(r"^(model\.language_model\.layers\.)(\d+)\.")
    sd = {shift.sub(lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.", k): v for k, v in sd.items()}
    tc["layer_types"] = ["linear_attention"] * n_chip + tc["layer_types"]
    tc["num_hidden_layers"] = n_chip + n0
    vocab = -(-(v0 + spec.VOCAB_SIZE) // 64) * 64
    tc["vocab_size"] = vocab

    if mute_qwen:
        from chipformer.q35.mute import fit_anti_sink, mute

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sinks = fit_anti_sink(src, dev)
        mute(sd, tc, n0, lay.dim(mu), hidden**0.5, sinks=sinks, first=n_chip)
    layers = emit_layers(c, lay, tc, 0, TEXT, alloc, keep)
    sd.update({k: v.to(torch.bfloat16) for k, v in layers.items()})
    del layers

    # embeddings: Qwen's rows keep their Qwen part and get their text class's chip features
    tok = AutoTokenizer.from_pretrained(src)
    text_ids = {}
    for cls, string in bridge.QWEN_TEXT.items():
        ids = tok.encode(string, add_special_tokens=False)
        assert len(ids) == 1, (string, ids)
        text_ids[cls] = ids[0]
    classes = list(range(bridge.TEXT_BASE, bridge.TEXT_BASE + bridge.N_TEXT))
    rows = dict(zip(classes, embedding_rows(c, lay, classes, hidden).to(torch.bfloat16)))
    full = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    full[:v0, :d0] = sd[TEXT + "embed_tokens.weight"][:, :d0]
    full[:v0, d0:] = rows[bridge.OTHER][d0:]
    for cls, i in text_ids.items():
        full[i, d0:] = rows[cls][d0:]
    for cls, strings in bridge.EXTRA_TEXT.items():
        for string in strings:
            (i,) = tok.encode(string, add_special_tokens=False)
            full[i, d0:] = rows[cls][d0:]
    chip = embedding_rows(c, lay, list(range(spec.VOCAB_SIZE)), hidden)
    full[v0 : v0 + spec.VOCAB_SIZE] = chip.to(torch.bfloat16)
    sd[TEXT + "embed_tokens.weight"] = full
    # image tokens come from the vision merger: give them the OTHER class too
    bias = "model.visual.merger.linear_fc2.bias"
    if bias in sd:
        sd[bias] = sd[bias].clone()
        sd[bias][d0:] = rows[bridge.OTHER][d0:].to(sd[bias].dtype)

    norm = sd[TEXT + "norm.weight"].float()
    norm[d0:] = -1.0  # chip dims are invisible to the lm_head ...
    for d in code:  # ... except the output code and tau
        norm[lay.dim(d)] = tiny.code_gain(lay)
    norm[lay.dim(tau)] = TAU_GAIN
    check_bf16("final norm", norm[d0:])
    sd[TEXT + "norm.weight"] = norm.to(torch.bfloat16)

    dst.mkdir(parents=True, exist_ok=True)
    shard, size, shards = {}, 0, []
    for k in sorted(sd):
        shard[k] = sd[k].contiguous()
        size += sd[k].numel() * 2
        if size > 2**31:
            shards.append(shard)
            shard, size = {}, 0
    shards.append(shard)
    index = {"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())}, "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    extend_tokenizer(src, dst, v0)
    gen_path = dst / "generation_config.json"
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else {}
    if "eos_token_id" not in gen:
        eos = cfg["text_config"].get("eos_token_id", 248044)
        gen.update(eos_token_id=[248046, eos], pad_token_id=eos)
    # After the circuit prints a tool response inside the assistant turn, Qwen sometimes opens a
    # new turn (<|im_start|>user ...) instead of going on. Qwen never writes <|im_start|> in its
    # own output, so the merged model bans it; the circuit never predicts it either.
    gen["bad_words_ids"] = [[IM_START]]
    gen_path.write_text(json.dumps(gen, indent=2))
    meta = {"layout": "v2", "chip_token_offset": v0, "chip_vocab": spec.VOCAB_SIZE,
            "chip_layers": n_chip, "qwen_layers_from": n_chip, "hidden": hidden,
            "chip_dims_offset": d0, "ballast_dims": m, "circuit": c.stats(),
            "slots": alloc.n_slots, "muted": mute_qwen,
            "text_ids": {str(k): v for k, v in text_ids.items()}}
    (dst / "chipformer.json").write_text(json.dumps(meta, indent=2))
    return c, lay


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Merge the CHIP-8 circuit into a Qwen3.5 checkpoint")
    ap.add_argument("src", help="Qwen3.5 checkpoint directory")
    ap.add_argument("--out", default="dist/qwen3.5-0.8b-chip8")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--ballast", type=int, default=None)
    ap.add_argument("--no-mute", action="store_true", help="leave Qwen's layers active on chip tokens")
    args = ap.parse_args()
    c, lay = build_merged(args.src, args.out, args.hidden, args.ballast, not args.no_mute)
    print(f"{args.out}: hidden {lay.hidden}, {c.n_layers()} chip layers, {c.stats()}")


if __name__ == "__main__":
    main()
