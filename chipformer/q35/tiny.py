"""A standalone Qwen3.5 model that holds only the chip circuit.

Same architecture class and DeltaNet head shapes as Qwen3.5-0.8B, so the circuit that runs here
merges into the real model unchanged. HF's cache needs one full-attention layer to count
positions; it is appended last with zero weights, so it adds nothing.
"""


import torch

from chipformer.q35.emit import Layout, allocate, check_bf16, embedding_rows, emit_layers
from chipformer.q35.ir import Graph

BASE = {  # Qwen3.5-0.8B text config, minus the sizes we pick
    "hidden_act": "silu",
    "intermediate_size": 3584,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "attn_output_gate": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                        "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                        "mrope_interleaved": True},
    "max_position_embeddings": 262144,
    "tie_word_embeddings": True,
}


def pick_layout(n_dims: int) -> Layout:
    """Smallest hidden = m * 4^k (a multiple of 64) that fits m ballast dims and n_dims."""
    best = None
    for m in range(3, 16):  # >= 3 ballast dims: constants are split over them
        k = 0
        while m * 4**k < m + n_dims or (m * 4**k) % 64:
            k += 1
        h = m * 4**k
        if best is None or h < best.hidden:
            best = Layout(hidden=h, m=m)
    return best


CODE_GAIN = 2.0**20  # final-norm gain on the output code (the exit leaves KAPPA * +-1 there)


def code_gain(lay: Layout | None = None) -> float:
    return CODE_GAIN


def build(g: Graph, vocab: list[int], code_dims: list[int], dtype=torch.bfloat16,
          clearable: set[int] | None = None, share: bool = False):
    """Qwen3_5ForCausalLM whose token i is chip token vocab[i]; lm_head reads ``code_dims``.

    With ``clearable`` (dims whose slots may be handed over), dims share residual slots.
    """
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    alloc = None
    keep = set(getattr(g, "keep", code_dims))
    if share:
        alloc = allocate(g, clearable, keep, BASE["intermediate_size"])
        lay = pick_layout(alloc.n_slots + 1)
        lay.slots = alloc.slots
    else:
        lay = pick_layout(len(g.dims))
    n_chip = g.n_layers()
    n_vocab = -(-len(vocab) // 64) * 64
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=n_vocab, num_hidden_layers=n_chip + 1,
              layer_types=["linear_attention"] * n_chip + ["full_attention"])
    sd = emit_layers(g, lay, tc, 0, "model.", alloc, keep if hasattr(g, "keep") else None)
    emb = torch.zeros(n_vocab, lay.hidden)
    emb[: len(vocab)] = embedding_rows(g, lay, vocab, lay.hidden)
    sd["model.embed_tokens.weight"] = emb
    norm = torch.full((lay.hidden,), -1.0)  # (1 + w) = 0: lm_head sees only the code dims
    for d in code_dims:
        norm[lay.dim(d)] = code_gain(lay)  # gain 1 + w, close enough to a power of 2
    check_bf16("final norm", norm)
    sd["model.norm.weight"] = norm
    config = Qwen3_5TextConfig(**tc)
    model = Qwen3_5ForCausalLM(config)
    for p in model.parameters():
        p.data.zero_()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    assert all(k.startswith(f"model.layers.{n_chip}.") or k == "lm_head.weight" for k in missing), \
        [k for k in missing if not k.startswith(f"model.layers.{n_chip}.")][:5]
    model.tie_weights()
    model.layout = lay
    return model.to(dtype).eval()


def sample_traces() -> list[list[int]]:
    """Short traces covering every opcode and quirk, for deciding which dims are clearable."""
    from chipformer.check import run_machine
    from chipformer.fuzz import random_program, random_quirks

    out = []
    for seed in range(6):
        m = run_machine(random_program(seed), 300, ipf=7, seed=seed, key_rate=0.3,
                        quirks=random_quirks(seed))
        out.append(m.tokens)
    return out


def build_chip8(dtype=torch.bfloat16, share_slots: bool = True):
    from chipformer.q35 import bridge, chip8_q35

    c = chip8_q35.build()
    vocab = list(range(bridge.TEXT_BASE + bridge.N_TEXT))
    model = build(c, vocab, [next(iter(e.terms)) for e in c.code], dtype=dtype,
                  share=share_slots)
    return c, model


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Save CHIP-8 as a standalone Qwen3.5 checkpoint")
    ap.add_argument("--out", default="dist/q35-tiny")
    args = ap.parse_args()
    c, model = build_chip8()
    model.save_pretrained(args.out)
    print(f"{args.out}: {c.stats()}, hidden {model.layout.hidden}")


if __name__ == "__main__":
    main()
