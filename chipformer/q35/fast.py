"""Fast greedy decoding of a q35 checkpoint with stock modules: a static cache plus one
compiled (CUDA-graph) single-token step that also takes the argmax on the GPU.

Pure decoding: every token the machine writes is the model's own greedy prediction; the host
(chipformer.session.Session) only injects the boot prompt, TICKs and key writes, exactly as with
any other model. Long sessions are bounded by ``max_len``; past it the session re-boots from its
mirror of the cells (the trace's own rollover record), which also resets the cache.

    runner = Runner(model, max_len=65536)
    s = runner.session(rom, quirks)
    s.tick(); s.step(); ...
"""

import torch

from chipformer.session import Session
from chipformer.trace import boot_tokens


class Runner:
    def __init__(self, model, max_len: int = 65536, token_offset: int = 0, n_chip: int = 8708,
                 compile: bool = True):
        from transformers import StaticCache

        self.model = model
        self.off = token_offset
        self.n_chip = n_chip
        self.max_len = max_len
        self.dev = next(model.parameters()).device
        cfg = getattr(model.config, "text_config", model.config)
        self.cache = StaticCache(config=cfg, max_cache_len=max_len)
        # Run the text stack directly (a multimodal wrapper would recompute vision-aware rope
        # indices every step) and score only the chip rows of the tied output head: in a machine
        # record the predicted token is always a chip token (logit ~240 vs ~0 for text), so the
        # greedy choice is the same as over the full vocabulary.
        inner = model.model
        self.lm = getattr(inner, "language_model", inner)
        self.head = model.lm_head.weight[token_offset:token_offset + n_chip]
        self.pos = 0
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.cpos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self._step = torch.compile(self._forward, mode="reduce-overhead", fullgraph=False) \
            if compile else self._forward

    def _forward(self, tok, cpos):
        h = self.lm(input_ids=tok, past_key_values=self.cache, cache_position=cpos,
                    use_cache=True).last_hidden_state[0, -1]
        return (self.head @ h).argmax(-1)

    @torch.no_grad()
    def prefill(self, toks: list[int]) -> int:
        x = torch.tensor([[t + self.off for t in toks]], device=self.dev)
        cpos = torch.arange(self.pos, self.pos + len(toks), device=self.dev)
        h = self.lm(input_ids=x, past_key_values=self.cache, cache_position=cpos,
                    use_cache=True).last_hidden_state[0, -1]
        self.pos += len(toks)
        return int((self.head @ h).argmax())

    @torch.no_grad()
    def feed(self, token: int) -> int:
        self.tok.fill_(token + self.off)
        self.cpos.fill_(self.pos)
        nxt = self._step(self.tok, self.cpos)
        self.pos += 1
        return int(nxt)

    def reset(self) -> None:
        self.cache.reset()
        self.pos = 0

    def session(self, rom: bytes, quirks=None) -> Session:
        """A Session whose boot prompt goes through one prefill; rolls over before max_len."""
        runner = self
        boot: list[int] = []

        class Fast(Session):
            def _generate_record(self):
                if runner.pos > runner.max_len - 200:  # room for the longest record
                    runner.reset()
                    dump = boot_tokens(self.cells)
                    self.tokens += dump
                    self.pred = runner.prefill(dump)
                return super()._generate_record()

        collecting = [True]

        def feed(tok: int) -> int:
            if collecting[0]:
                boot.append(tok)
                return -1
            return runner.feed(tok)

        s = Fast(feed, rom, quirks)
        collecting[0] = False
        s.pred = runner.prefill(boot)
        return s
