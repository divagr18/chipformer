"""Firmware for rv-tiny: an ELF becomes a state snapshot in the model's HF cache, then the model
runs the program by plain greedy decoding.

    python -m chipformer.rv.firmware dist/rv-tiny program.elf [--max-new 4000]

The snapshot is the circuit's head states (ROM rows, RAM words, registers, PC, counters; see
core.snapshot), written into the linear-attention layers' recurrent states after a one-token
PAD prefix (a zero embedding: it leaves every state and conv window at 0). The run then starts
at GO. Everything after that is the model's own greedy decoding (``generate``).
"""

import argparse

import torch

from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S


def load_cache(model, c, machine):
    """A DynamicCache holding the PAD prefix and the machine's snapshot."""
    dev = next(model.parameters()).device
    core.snapshot(c, machine)
    out = model(input_ids=torch.tensor([[S.VOCAB]], device=dev), use_cache=True)
    hf.inject(c, out.past_key_values)
    return out.past_key_values


def run(model, c, machine, max_new: int = 4000) -> list[int]:
    """The generated tokens after GO (stopping at HALT)."""
    dev = next(model.parameters()).device
    cache = load_cache(model, c, machine)
    ids = torch.tensor([[S.VOCAB, S.tok("GO")]], device=dev)
    with torch.no_grad():
        out = model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=max_new,
                             do_sample=False, eos_token_id=S.tok("HALT"), pad_token_id=S.VOCAB)
    return out[0, 2:].tolist()


def output_text(tokens: list[int]) -> str:
    """The program's putchar output in a generated trace."""
    return bytes(p for t in tokens if t < S.VOCAB and (f := S.family(t))[0] == "BYTE"
                 for p in [f[1]]).decode("latin-1")


def main():
    from transformers import Qwen3_5ForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("elf")
    ap.add_argument("--max-new", type=int, default=4000)
    args = ap.parse_args()
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    c = core.build()
    toks = run(model, c, ref.Machine.from_program(elf.program(args.elf)), args.max_new)
    print(output_text(toks), end="")
    print(f"[{len(toks)} tokens, last {S.name(toks[-1]) if toks[-1] < S.VOCAB else toks[-1]}]")


if __name__ == "__main__":
    main()
