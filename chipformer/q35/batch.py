"""Batched greedy decoding of q35 checkpoints with stock modules, token-identical to decoding
each prompt on its own.

    outs = generate_batch(model, tok, prompts, max_new=4500, batch=16)

How it stays equal to ``model.generate(do_sample=False)`` on one prompt:

- **Prefill alone.** Each prompt is prefilled on its own (batch 1, no padding) through the same
  modules ``generate`` uses, so its states and first token are the batch-1 ones. The states are
  then copied into one row of the batched cache. There is no padding anywhere, so nothing
  depends on ``apply_mask_to_padding_states`` or on DeltaNet chunk alignment, and the circuit
  never sees a pad token.
- **Per-row positions.** Every row keeps its own position: full attention writes key/value ``t``
  of row ``b`` at slot ``t`` (like the batch-1 static cache), RoPE gets the row's own position,
  and the mask of row ``b`` admits slots ``<= pos[b]``. Gated DeltaNet rows are independent.
- **Continuous batching.** A row that emits an end token (or reaches ``max_new``) is refilled
  with the next prompt at once, so the batch stays full while the queue lasts. When the queue
  is empty and few rows are left, the live rows are compacted into a smaller batch.

What can still differ from batch 1 is floating-point rounding in Qwen's own text arithmetic
(GEMM kernels for M = 1 and M = B accumulate in different orders); the circuit's arithmetic is
exact either way. ``dist/spikes/b1_batch.py`` checks the outputs against ``generate``.

The decode step (embedding, all layers, the full-vocabulary head and the argmax) is one
``torch.compile`` graph with CUDA graphs, per batch size.
"""

from collections import deque

import torch


def _text_model(model):
    inner = model.model
    return getattr(inner, "language_model", inner)


def _mark(t):
    torch._dynamo.mark_static_address(t)
    return t


class _Rows:
    """Batched cache: linear-attention layers (stock ``LinearAttentionLayer``), and full-attention
    layers whose ``update`` writes each row at its own position."""

    def __init__(self, cfg, small, batch: int, max_len: int, pos: torch.Tensor):
        from transformers.cache_utils import Cache, LinearAttentionLayer, StaticLayer

        rows = torch.arange(batch, device=pos.device)

        class RowKV(StaticLayer):
            def update(self, key_states, value_states, *args, **kwargs):
                # decode only: one new position per row
                self.keys[rows, :, pos] = key_states[:, :, 0]
                self.values[rows, :, pos] = value_states[:, :, 0]
                return self.keys, self.values

        layers = []
        for kind, s in zip(cfg.layer_types, small.layers):
            if kind == "linear_attention":
                la = LinearAttentionLayer()
                conv, rec = s.conv_states[0], s.recurrent_states[0]
                la.dtype, la.device = conv.dtype, conv.device
                la.conv_states[0] = _mark(conv.new_zeros(batch, *conv.shape[1:]))
                la.recurrent_states[0] = _mark(rec.new_zeros(batch, *rec.shape[1:]))
                la.conv_kernel_size[0] = conv.shape[-1]
                la.is_conv_states_initialized[0] = True
                la.is_recurrent_states_initialized[0] = True
                la.has_previous_state[0] = True
                layers.append(la)
            else:
                kv = RowKV(max_cache_len=max_len)
                kv.dtype, kv.device = s.keys.dtype, s.keys.device
                kv.batch_size, kv.num_heads = batch, s.keys.shape[1]
                kv.k_head_dim, kv.v_head_dim = s.keys.shape[-1], s.values.shape[-1]
                kv.keys = _mark(s.keys.new_zeros(batch, s.keys.shape[1], max_len, s.keys.shape[-1]))
                kv.values = _mark(s.values.new_zeros(batch, s.values.shape[1], max_len,
                                                     s.values.shape[-1]))
                kv.cumulative_length = kv.cumulative_length.to(pos.device)
                kv.is_initialized = True
                layers.append(kv)
        self.cache = Cache(layers=layers)
        self.kinds = list(cfg.layer_types)

    @torch.no_grad()
    def load_row(self, b: int, small, n: int) -> None:
        """Copy a batch-1 prefill cache holding ``n`` positions into row ``b``."""
        for kind, big, s in zip(self.kinds, self.cache.layers, small.layers):
            if kind == "linear_attention":
                big.conv_states[0][b].copy_(s.conv_states[0][0])
                big.recurrent_states[0][b].copy_(s.recurrent_states[0][0])
            else:
                big.keys[b, :, :n].copy_(s.keys[0, :, :n])
                big.values[b, :, :n].copy_(s.values[0, :, :n])

    @torch.no_grad()
    def take(self, idx: torch.Tensor, into: "_Rows") -> None:
        """Copy rows ``idx`` of this cache into rows 0.. of ``into`` (a smaller batch)."""
        for big, small in zip(self.cache.layers, into.cache.layers):
            if hasattr(big, "keys") and big.keys is not None:
                small.keys[: len(idx)].copy_(big.keys[idx])
                small.values[: len(idx)].copy_(big.values[idx])
            else:
                small.conv_states[0][: len(idx)].copy_(big.conv_states[0][idx])
                small.recurrent_states[0][: len(idx)].copy_(big.recurrent_states[0][idx])


class BatchDecoder:
    """Greedy decoding of many prompts through one model, ``batch`` rows at a time.

    ``sizes`` are the batch sizes with a compiled step (the largest is the full batch); when the
    queue is empty and the live rows fit a smaller size, they are compacted into it.
    """

    def __init__(self, model, batch: int = 16, max_len: int = 8192, eos=None,
                 compile: bool = True, sizes=None, mode: str = "reduce-overhead",
                 fullgraph: bool = True):
        from transformers import StaticCache

        self.model = model
        self.lm = _text_model(model)
        self.head = model.lm_head  # an nn.Linear, or fast.SparseLinear after fast.compact
        self.cfg = getattr(model.config, "text_config", model.config)
        self.dev = model.lm_head.weight.device
        self.max_len = max_len
        if eos is None:
            eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        self.eos = set(eos)
        self.small = StaticCache(config=self.cfg, max_cache_len=max_len)
        self.sizes = sorted(set(sizes or [batch]) | {batch})
        self.batch = batch
        self.compile = compile
        self.mode, self.fullgraph = mode, fullgraph
        self.levels = {}  # batch size -> (rows cache, tok, pos, step fn)
        self.slots = torch.arange(max_len, device=self.dev)

    # --- one batch size ---------------------------------------------------------------------

    def _level(self, n: int):
        if n not in self.levels:
            pos = _mark(torch.zeros(n, dtype=torch.long, device=self.dev))
            tok = _mark(torch.zeros(n, 1, dtype=torch.long, device=self.dev))
            rows = _Rows(self.cfg, self.small, n, self.max_len, pos)
            cache, slots, lm, head = rows.cache, self.slots, self.lm, self.head

            def step(tok, pos):
                mask = (slots[None, :] <= pos[:, None])[:, None, None, :]
                h = lm(input_ids=tok, position_ids=pos[:, None], past_key_values=cache,
                       attention_mask={"full_attention": mask, "linear_attention": None},
                       use_cache=True).last_hidden_state[:, -1]
                return head(h).argmax(-1)

            if self.compile:
                step = torch.compile(step, mode=self.mode, fullgraph=self.fullgraph)
            self.levels[n] = (rows, tok, pos, step)
        return self.levels[n]

    @torch.no_grad()
    def prefill(self, ids: list[int]) -> int:
        """Batch-1 prefill into ``self.small`` (as ``generate`` does it); returns the first token."""
        self.small.reset()
        x = torch.tensor([ids], device=self.dev)
        n = x.shape[1]
        h = self.lm(input_ids=x, position_ids=torch.arange(n, device=self.dev)[None],
                    attention_mask=torch.ones_like(x), past_key_values=self.small,
                    use_cache=True).last_hidden_state[:, -1:]
        return int(self.head(h)[0, -1].argmax())

    @torch.no_grad()
    def warmup(self, prompt: list[int], steps: int = 3) -> None:
        """Compile (and record the CUDA graphs of) every batch size's step ahead of time."""
        self.prefill(prompt)
        for n in self.sizes:
            _rows, tok, pos, step = self._level(n)
            tok.zero_()
            pos.zero_()
            for _ in range(steps):
                step(tok, pos)
        torch.cuda.synchronize()

    # --- the loop -----------------------------------------------------------------------------

    @torch.no_grad()
    def run(self, prompts: list[list[int]], max_new: int, on_done=None) -> list[list[int]]:
        """Greedy continuations (each ending with its end token, or ``max_new`` long)."""
        for p in prompts:
            assert len(p) + max_new <= self.max_len, (len(p), max_new, self.max_len)
        out: list[list[int] | None] = [None] * len(prompts)
        queue = deque(range(len(prompts)))
        if not queue:
            return []
        # the batched cache copies the shapes of a prefilled batch-1 cache
        first = self.prefill(prompts[queue[0]])
        n = self.batch
        rows, tok, pos, step = self._level(n)
        live: list[int | None] = [None] * n  # prompt index per row
        gen: list[list[int]] = [[] for _ in range(n)]

        def finish(b):
            i = live[b]
            out[i] = gen[b]
            live[b] = None
            if on_done is not None:
                on_done(i, gen[b])

        def admit(b, first=None):
            while queue:
                i = queue.popleft()
                t = self.prefill(prompts[i]) if first is None else first
                first = None
                live[b], gen[b] = i, [t]
                if t in self.eos or max_new <= 1:
                    finish(b)
                    continue
                rows.load_row(b, self.small, len(prompts[i]))
                tok[b, 0] = t
                pos[b] = len(prompts[i])
                return
            tok[b, 0] = 0  # an idle row decodes garbage at slot 0 until it is reused
            pos[b] = 0

        admit(0, first)
        for b in range(1, n):
            admit(b)
        while any(i is not None for i in live):
            nxt = step(tok, pos)
            tok.copy_(nxt[:, None])
            pos.add_(1)
            toks = nxt.tolist()
            for b, t in enumerate(toks):
                if live[b] is None:
                    continue
                gen[b].append(t)
                if t in self.eos or len(gen[b]) >= max_new:
                    finish(b)
                    admit(b)
            if not queue:
                alive = [b for b in range(n) if live[b] is not None]
                smaller = [s for s in self.sizes if s < n and s >= len(alive)]
                if alive and smaller:
                    m = smaller[0]
                    rows2, tok2, pos2, step2 = self._level(m)
                    idx = torch.tensor(alive, device=self.dev)
                    rows.take(idx, rows2)
                    tok2[: len(alive)].copy_(tok[idx])
                    pos2[: len(alive)].copy_(pos[idx])
                    pos2[len(alive):] = 0
                    tok2[len(alive):] = 0
                    live = [live[b] for b in alive] + [None] * (m - len(alive))
                    gen = [gen[b] for b in alive] + [[] for _ in range(m - len(alive))]
                    n, rows, tok, pos, step = m, rows2, tok2, pos2, step2
        return out


_DECODERS: dict = {}


def generate_batch(model, tok, prompts: list[list[int]], max_new: int, batch: int | None = None,
                   max_len: int | None = None, **kw) -> list[list[int]]:
    """Greedy continuations of ``prompts`` (token id lists), equal to running each alone.

    ``tok`` is unused (end tokens come from ``model.generation_config``); it is accepted so the
    call reads like the usual ``generate`` helpers. Decoders (and their compiled steps) are
    reused across calls with the same model, batch and ``max_len``.
    """
    del tok
    batch = batch or min(16, len(prompts))
    if max_len is None:
        need = max(len(p) for p in prompts) + max_new
        max_len = -(-need // 1024) * 1024
    key = (id(model), batch, max_len, tuple(sorted(kw.items())))
    if key not in _DECODERS:
        sizes = kw.pop("sizes", None) or sorted({batch, *(s for s in (1, 2, 4, 8, 16, 32)
                                                          if s < batch)})
        _DECODERS[key] = BatchDecoder(model, batch=batch, max_len=max_len, sizes=sizes, **kw)
    return _DECODERS[key].run(prompts, max_new)
