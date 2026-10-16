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

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from chipformer.session import Session
from chipformer.trace import boot_tokens


# --- weight blocks ---------------------------------------------------------------------------
#
# Most of a merged checkpoint's weight bytes are exact zeros: the chip layers use a few hundred
# of their 3584+ input dims and a fraction of their units, Qwen's widened matrices never touch
# the chip dims (bar one mute column), and the output head reads no chip dim past the output
# code. Decoding at batch 1 is bound by reading weights, so reading only each matrix's nonzero
# block is the largest exact speedup there is. The checkpoint itself stays as it is; this only
# changes which kernels run.


def _hull(mask: torch.Tensor) -> tuple[int, int]:
    idx = mask.nonzero().flatten()
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)


class SparseLinear(torch.nn.Module):
    """``F.linear(x, W)`` for a weight that is zero outside the block ``W[r0:r1, c0:c1]``;
    reads only the block. Equal to the dense product up to the order of the accumulation over
    the block's columns, so exactly equal wherever that sum is exact (all chip arithmetic)."""

    def __init__(self, w: torch.Tensor, rows=None, cols=None, block=None):
        super().__init__()
        nz = w != 0 if rows is None or cols is None else None
        self.out_features, self.in_features = w.shape
        self.r0, self.r1 = rows or _hull(nz.any(1))
        self.c0, self.c1 = cols or _hull(nz.any(0))
        if block is None:
            block = torch.nn.Parameter(w[self.r0:self.r1, self.c0:self.c1].contiguous(),
                                       requires_grad=False)
        self.weight = block

    def forward(self, x):
        if self.r1 == self.r0 or self.c1 == self.c0:
            return x.new_zeros(*x.shape[:-1], self.out_features)
        y = F.linear(x[..., self.c0:self.c1], self.weight)
        if self.r0 or self.r1 != self.out_features:
            y = F.pad(y, (self.r0, self.out_features - self.r1))
        return y

    def extra_repr(self):
        return (f"{self.out_features}x{self.in_features} -> rows {self.r0}:{self.r1}, "
                f"cols {self.c0}:{self.c1}")


class SparseEmbedding(torch.nn.Module):
    """Embedding whose table is zero outside columns ``c0:c1`` (shared with the output head)."""

    def __init__(self, block: torch.nn.Parameter, c0: int, dim: int):
        super().__init__()
        self.weight, self.c0, self.dim = block, c0, dim

    def forward(self, ids):
        e = F.embedding(ids, self.weight)
        return F.pad(e, (self.c0, self.dim - self.c0 - self.weight.shape[1]))


def _n_chip_layers(model) -> int:
    meta = Path(getattr(model.config, "_name_or_path", "") or ".") / "chipformer.json"
    if meta.exists():
        return json.loads(meta.read_text())["chip_layers"]
    cfg = getattr(model.config, "text_config", model.config)
    return cfg.num_hidden_layers  # a chip-only checkpoint (tiny.py)


@torch.no_grad()
def compact(model, qwen: bool = True, n_chip: int | None = None) -> dict:
    """Replace the text stack's linear maps by their nonzero blocks, in place.

    ``qwen=False`` touches only the chip layers, whose arithmetic is exact, so every logit stays
    bit-identical. ``qwen=True`` also compacts Qwen's layers, the embedding and the output head:
    the dropped terms are exact zeros, but the shorter sums may round differently in Qwen's own
    (inexact) text arithmetic, like any change of GEMM kernel. Returns weight counts.
    """
    inner = model.model
    lm = getattr(inner, "language_model", inner)
    n_chip = _n_chip_layers(model) if n_chip is None else n_chip
    before = after = 0
    for i, layer in enumerate(lm.layers):
        if i >= n_chip and not qwen:
            continue
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear) and child.bias is None:
                    s = SparseLinear(child.weight)
                    before += child.weight.numel()
                    after += s.weight.numel()
                    setattr(parent, name, s)
    if qwen and isinstance(model.lm_head, torch.nn.Linear):
        w = model.lm_head.weight
        emb = lm.embed_tokens.weight
        c0, c1 = _hull((w != 0).any(0))
        if w.data_ptr() == emb.data_ptr():
            ce0, ce1 = _hull((emb != 0).any(0))
            c0, c1 = min(c0, ce0), max(c1, ce1)
        block = torch.nn.Parameter(w[:, c0:c1].contiguous(), requires_grad=False)
        before += w.numel()
        after += block.numel()
        if w.data_ptr() == emb.data_ptr():
            lm.embed_tokens = SparseEmbedding(block, c0, w.shape[1])
        model.lm_head = SparseLinear(w, rows=(0, w.shape[0]), cols=(c0, c1), block=block)
    torch.cuda.empty_cache()
    return {"dense_weights": before, "compact_weights": after}


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
        lm_head = model.lm_head
        self.head = lm_head.weight[token_offset:token_offset + n_chip]
        self.cols = (slice(lm_head.c0, lm_head.c1) if isinstance(lm_head, SparseLinear)
                     else slice(None))
        self.pos = 0
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.cpos = torch.zeros(1, dtype=torch.long, device=self.dev)
        self._step = torch.compile(self._forward, mode="reduce-overhead", fullgraph=False) \
            if compile else self._forward

    def _forward(self, tok, cpos):
        h = self.lm(input_ids=tok, past_key_values=self.cache, cache_position=cpos,
                    use_cache=True).last_hidden_state[0, -1]
        return (self.head @ h[self.cols]).argmax(-1)

    @torch.no_grad()
    def prefill(self, toks: list[int]) -> int:
        x = torch.tensor([[t + self.off for t in toks]], device=self.dev)
        cpos = torch.arange(self.pos, self.pos + len(toks), device=self.dev)
        h = self.lm(input_ids=x, past_key_values=self.cache, cache_position=cpos,
                    use_cache=True).last_hidden_state[0, -1]
        self.pos += len(toks)
        return int((self.head @ h[self.cols]).argmax())

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
