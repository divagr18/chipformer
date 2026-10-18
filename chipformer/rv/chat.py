"""Talk to a merged rv1 checkpoint: Qwen writes, the RV32 machine runs inside the same model.

    python -m chipformer.rv.chat dist/q4-rv1 --firmware tests/data/rv/c-calc.elf \\
        --ask "What is 123456789 * 987654321?"

Decoding is greedy and transparent to chip tokens (as BatchDecoder's fp + tc): a chip token
runs through the chip layers only, so Qwen's layers, caches and RoPE positions see the text
alone; a text token runs through every layer. At every position the next token is the
circuit's when its output code is set (the owner), else Qwen's argmax over its own vocabulary.
The firmware (an ELF, parked at its first exit ecall) is a state snapshot written into the chip
layers' recurrent states after the prompt's prefill (rv/firmware.py, rv/hf.py).

``--check`` replays the whole stream through the IR evaluator and the reference machine: every
circuit-owned token must equal the IR's prediction and the reference's next token.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.rv import bridge as BR
from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S

CALC_TOOL = {"type": "function", "function": {
    "name": "calc",
    "description": "Evaluate an integer arithmetic expression exactly (+, -, *, parentheses; "
                   "numbers of any size). Returns the exact value.",
    "parameters": {"type": "object", "properties": {
        "expression": {"type": "string", "description": "e.g. 123456789*987654321-42"}},
        "required": ["expression"]}}}


class RvChat:
    def __init__(self, path, device: str | None = None) -> None:
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        from chipformer.q35.fast import exact_matmuls
        from chipformer.q35.ir import Target

        exact_matmuls()
        self.path = Path(path)
        self.meta = json.loads((self.path / "chipformer.json").read_text())
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3_5ForCausalLM.from_pretrained(self.path, dtype=torch.bfloat16)
        self.model = self.model.to(self.dev).eval()
        self.lm = self.model.model
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.T = BR.Text(self.path / "tokenizer.json")
        cfg = self.model.config
        self.c = core.build(Target(cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                                   cfg.intermediate_size), self.T)
        self.n_chip, self.v0 = self.meta["chip_layers"], self.meta["chip_token_offset"]
        self.code_dims = torch.tensor(self.meta["code_dims"], device=self.dev)
        self.eos = {248046, 248044}

    # --- one position ----------------------------------------------------------------------
    @torch.no_grad()
    def _layers(self, ids, cache, pos: int | None):
        """Run one token through the chip layers (pos None) or the whole stack (text at RoPE
        position pos); returns (residual after the chip layers, logits or None)."""
        h = self.lm.embed_tokens(torch.tensor([[ids]], device=self.dev))
        pe = tp = None
        if pos is not None:
            p4 = torch.full((4, 1, 1), pos, device=self.dev, dtype=torch.long)
            tp = p4[0]
            pe = self.lm.rotary_emb(h, p4[1:])
        hc = None
        n = self.n_chip if pos is None else len(self.lm.layers)
        for i in range(n):
            h = self.lm.layers[i](h, position_embeddings=pe, attention_mask=None,
                                  position_ids=tp, past_key_values=cache)
            if i == self.n_chip - 1:
                hc = h
        logits = None if pos is None else self.model.lm_head(self.lm.norm(h))[0, -1]
        return hc[0, -1], logits

    def circuit_token(self, hc) -> int | None:
        """The model id the circuit predicts at a position, None where Qwen predicts."""
        code = hc[self.code_dims].float()
        if not bool((code != 0).any()):
            return None
        bits = (code > 0).long().tolist()
        nf = core.NFAM
        fid = sum(b << i for i, b in enumerate(bits[:nf]))
        p = sum(b << i for i, b in enumerate(bits[nf:]))
        t = core.decode_code(fid, p, self.T)
        if t < 0:
            raise RuntimeError(f"the circuit predicts no token (family {fid}, payload {p})")
        return t - S.TEXT_TOKEN if S.is_text(t) else self.v0 + t

    # --- a conversation --------------------------------------------------------------------
    def prompt_ids(self, messages, tools=(CALC_TOOL,)) -> list[int]:
        out = self.tok.apply_chat_template(messages, tools=list(tools) or None,
                                           add_generation_prompt=True, enable_thinking=False,
                                           tokenize=True)
        if not isinstance(out, list):
            out = out["input_ids"]
        return [int(x) for x in out]

    @torch.no_grad()
    def generate(self, ids: list[int], machine: "ref.Machine", max_new: int = 4000,
                 max_chip: int = 60000, check_head: bool = False, verbose: bool = False,
                 lock: "BR.Session | None" = None) -> dict:
        """Greedy decoding after the prompt ``ids`` with the parked ``machine`` loaded.
        ``lock`` (a Session on a copy of the machine): stop at the first token that differs from
        the reference's."""
        cache = None
        out = self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=True)
        cache = out.past_key_values
        core.snapshot(self.c, machine)
        hf.inject(self.c, cache)
        logits = out.logits[0, -1]
        pos = len(ids)  # the next text token's RoPE position
        gen, n_text, n_chip_tok = [], 0, 0
        stream_hc = []
        # the prompt's last position: Qwen predicts (nothing captured yet)
        nxt = int(torch.argmax(logits[: self.v0]))
        t0 = time.time()
        mismatch = 0
        expect, lock_error = [], None
        while n_text < max_new and n_chip_tok < max_chip:
            gen.append(nxt)
            if lock is not None:  # check against the reference as we go
                st = nxt - self.v0 if nxt >= self.v0 else BR.text_token(nxt)
                if expect:
                    e = expect.pop(0)
                    if st != e:
                        lock_error = {"at": len(gen) - 1, "got": S.name(st) if st < S.VOCAB
                                      else self.tok.decode([nxt]),
                                      "want": S.name(e) if e < S.VOCAB
                                      else self.tok.decode([e - S.TEXT_TOKEN]),
                                      "recent": [S.name(t - self.v0) if t >= self.v0 else
                                                 self.tok.decode([t]) for t in gen[-12:]]}
                        break
                elif st < S.VOCAB:
                    lock_error = {"at": len(gen) - 1, "got": S.name(st), "want": "text"}
                    break
                else:
                    k = len(lock.toks)
                    lock.qwen([nxt])
                    expect = list(lock.toks[k + 1:])
            if nxt in self.eos:
                break
            if verbose and len(gen) % 500 == 0:
                tail = [S.name(t - self.v0) if t >= self.v0 else repr(self.tok.decode([t]))
                        for t in gen[-8:]]
                print(f"  {len(gen)} tokens ({n_text} text, {n_chip_tok} chip) "
                      f"{time.time() - t0:.0f}s: {' '.join(tail)}", flush=True)
            if nxt >= self.v0:  # a chip token
                n_chip_tok += 1
                hc, _ = self._layers(nxt, cache, None)
                nxt = self.circuit_token(hc)
                if nxt is None:
                    raise RuntimeError("the circuit gave up a chip position")
            else:
                n_text += 1
                hc, logits = self._layers(nxt, cache, pos)
                pos += 1
                ct = self.circuit_token(hc)
                q = int(torch.argmax(logits))
                if ct is not None and check_head and q != ct:
                    mismatch += 1
                nxt = ct if ct is not None else int(torch.argmax(logits[: self.v0]))
            if check_head:
                stream_hc.append(hc[self.code_dims].float().cpu().numpy())
        dt = time.time() - t0
        return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok,
                "seconds": dt, "head_mismatch": mismatch, "lock_error": lock_error}

    def to_stream(self, ids: list[int]) -> list[int]:
        """Model ids -> the circuit's token stream (chip tokens, TEXT_TOKEN | qwen id)."""
        return [i - self.v0 if i >= self.v0 else BR.text_token(i) for i in ids]

    def text_of(self, ids: list[int]) -> str:
        return self.tok.decode([i for i in ids if i < self.v0], skip_special_tokens=False)


def replay(chat: RvChat, prompt: list[int], gen: list[int], firmware) -> dict:
    """Check a generated stream: the reference machine (fed Qwen's own tokens) must produce the
    same chip tokens and echoes, and the IR circuit must predict every circuit-owned token."""
    from chipformer.q35 import ir

    m = ref.Machine.from_program(elf.program(firmware))
    BR.park(m)
    ses = BR.Session(chat.T, m)
    stream = chat.to_stream(prompt + gen)
    # the prompt is prefilled history (its system prompt shows the tool-call format, so it holds
    # <tool_call> .. </tool_call> tokens: the circuit owns those </tool_call> positions, but no
    # run follows); then the reference is fed Qwen's tokens and appends the runs itself
    ses.history(prompt)
    skip = {x for x in range(len(prompt)) if prompt[x] == chat.T.TCC}
    i = 0
    pos = len(prompt)
    while pos < len(stream):
        t = stream[pos]
        if not S.is_text(t):
            raise AssertionError(f"chip token {S.name(t)} at {pos} where the reference has none")
        k = len(ses.toks)
        ses.qwen([t - S.TEXT_TOKEN])
        pos += 1
        added = ses.toks[k + 1:]
        if added:  # a run: the stream must continue with exactly the reference's tokens
            got = stream[pos:pos + len(added)]
            if got != added:
                j = next((x for x in range(min(len(got), len(added))) if got[x] != added[x]),
                         min(len(got), len(added)))
                return {"ok": False, "where": pos + j, "got": got[j:j + 5], "want": added[j:j + 5]}
            pos += len(added)
        i += 1
    m2 = ref.Machine.from_program(elf.program(firmware))
    BR.park(m2)
    core.snapshot(chat.c, m2)
    carry, mats = {}, []
    for a in range(0, len(stream), 2048):
        mats.append(ir.evaluate(chat.c, stream[a:a + 2048], carry=carry))
    mat = np.concatenate(mats)
    pred = core.predictions(chat.c, mat)
    own = np.rint(ir.ev(chat.c.owner, mat)).astype(int)
    bad_own = int(sum(own[x] != int(ses.own[x]) for x in range(len(stream))))
    bad = int(sum(1 for x in range(len(stream) - 1)
                  if ses.own[x] and x not in skip and pred[x] != stream[x + 1]))
    return {"ok": bad == 0 and bad_own == 0, "positions": len(stream),
            "owned": int(sum(ses.own)) - len(skip), "wrong_predictions": bad,
            "owner_mismatch": bad_own, "runs": len(ses.runs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--firmware", default="tests/data/rv/c-calc.elf")
    ap.add_argument("--ask", default="Use the calc tool: what is 123456789 * 987654321?")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    chat = RvChat(args.model)
    m = ref.Machine.from_program(elf.program(args.firmware))
    BR.park(m)
    ids = chat.prompt_ids([{"role": "user", "content": args.ask}])
    m2 = ref.Machine.from_program(elf.program(args.firmware))
    BR.park(m2)
    lock = BR.Session(chat.T, m2)
    lock.history(ids)
    r = chat.generate(ids, m, args.max_new, check_head=True, verbose=True, lock=lock)
    print(chat.text_of(r["tokens"]))
    print(f"[{r['text_tokens']} text + {r['chip_tokens']} chip tokens in {r['seconds']:.1f}s, "
          f"head mismatches {r['head_mismatch']}, lock error {r['lock_error']}]")
    if args.check:
        print(replay(chat, ids, r["tokens"], args.firmware))


if __name__ == "__main__":
    main()
