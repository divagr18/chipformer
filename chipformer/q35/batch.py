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
exact either way (``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`` must be
False, see ``fast.exact_matmuls``).

Decoding modes (each step is a CUDA graph per batch size, captured once):

- ``compile=True`` (default): Inductor kernels. Fastest; at batch 1 it matches
  ``generate(cache_implementation="static")`` (which also compiles) only while the graph is the
  same as HF's (no compaction), since any change shifts Inductor's fusion and Qwen's rounding.
- ``compile=False, kv_len="generate"``: eager kernels, attending over exactly generate's static
  cache length (``len(prompt) + max_new``; the attention result depends on it). At batch 1 this
  is bit-identical to ``generate(cache_implementation="static", disable_compile=True)``, also
  with ``fast.compact`` and ``skip_qwen`` (dist/spikes/b22_eager.py).

Options: ``skip_qwen`` (chip-token inputs run only the chip layers; Qwen's layers are exact
no-ops there), ``freeze_chip_positions`` (RoPE counts only non-chip tokens), ``run(hook=...)``
(token forcing / custom stops), generation_config ``bad_words_ids`` are honored. Prompts are
prefilled ahead on a second stream (``prefetch``) while the batch decodes.
"""

import time
from collections import deque

import torch


def _text_model(model):
    inner = model.model
    return getattr(inner, "language_model", inner)


def _mark(t):
    torch._dynamo.mark_static_address(t)
    return t


def _fold_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kw):
    """HF's SDPA attention, except that a single-token query attends without ``repeat_kv``:
    the GQA group goes into the query axis (heads ``j*G .. j*G+G-1`` share KV head ``j``), so
    the cache is not copied G times per layer and step. Bit-identical to the repeat_kv path
    (dist/spikes/b14_fold.py); prompts (more than one query) take HF's path unchanged."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    b, h, q, d = query.shape
    kv = key.shape[1]
    if q != 1 or h == kv or dropout:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kw)
    out = torch.nn.functional.scaled_dot_product_attention(
        query.view(b, kv, h // kv, d), key, value, attn_mask=attention_mask, scale=scaling)
    return out.reshape(b, h, 1, d).transpose(1, 2).contiguous(), None


def _use_fold(model) -> None:
    from transformers import AttentionInterface

    AttentionInterface.register("q35_fold", _fold_sdpa)
    cfg = getattr(model.config, "text_config", model.config)
    if cfg._attn_implementation in ("sdpa", "q35_fold"):
        cfg._attn_implementation = "q35_fold"


class _Rows:
    """Batched cache: linear-attention layers (stock ``LinearAttentionLayer``), and full-attention
    layers whose ``update`` writes each row at its own position."""

    def __init__(self, cfg, small, batch: int, max_len: int, pos: torch.Tensor,
                 kvpos: torch.Tensor | None = None):
        from transformers.cache_utils import Cache, LinearAttentionLayer, StaticLayer

        self.view = max_len  # attention reads slots [0, view): a compile-time constant
        owner = self
        slot = pos if kvpos is None else kvpos  # KV slot per row (transparent: text count)

        class RowKV(StaticLayer):
            def update(self, key_states, value_states, *args, **kwargs):
                # decode only: one new position per row
                rows = torch.arange(slot.shape[0], device=slot.device)
                self.keys[rows, :, slot] = key_states[:, :, 0]
                self.values[rows, :, slot] = value_states[:, :, 0]
                return self.keys[:, :, :owner.view], self.values[:, :, :owner.view]

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


class _Prefetcher:
    """Prefills the queued prompts ahead of time on a second CUDA stream, from a worker thread,
    into a few spare batch-1 caches; the decode loop then only copies a ready cache into a row.
    Same modules and kernels as ``BatchDecoder.prefill``, so the states are the same bits; the
    GPU overlaps the (launch-bound) prefill with the decode steps."""

    def __init__(self, dec: "BatchDecoder", prompts, order, depth: int, lens=None):
        import threading

        from transformers import StaticCache

        self.dec, self.prompts, self.lens = dec, prompts, lens
        self.order = list(order)
        self.ready: dict[int, tuple] = {}
        self.cv = threading.Condition()
        self.stream = torch.cuda.Stream()
        while len(dec.spare) < depth:
            dec.spare.append(StaticCache(config=dec.cfg, max_cache_len=dec.max_len))
        self.free = [(c, None) for c in dec.spare[:depth]]  # (cache, event: free after it)
        self.stop = False
        self.error = None
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _work(self):
        try:
            with torch.no_grad(), torch.cuda.stream(self.stream):
                for i in self.order:
                    with self.cv:
                        while not self.free and not self.stop:
                            self.cv.wait()
                        if self.stop:
                            return
                        cache, ev = self.free.pop()
                    if ev is not None:
                        self.stream.wait_event(ev)  # the row copy out of it is done
                    if self.lens is not None:  # a cache of generate's length for this prompt
                        from transformers import StaticCache

                        cache = StaticCache(config=self.dec.cfg, max_cache_len=self.lens[i])
                    t0 = time.perf_counter()
                    t = self.dec._prefill_into(cache, self.prompts[i])
                    done = torch.cuda.Event()
                    done.record(self.stream)
                    with self.cv:
                        self.dec.stats["prefills"] += 1
                        self.dec.stats["prefill_s"] += time.perf_counter() - t0
                        self.ready[i] = (cache, t, done)
                        self.cv.notify_all()
        except BaseException as e:  # noqa: BLE001 - surfaced in get()
            with self.cv:
                self.error = e
                self.cv.notify_all()

    def get(self, i: int):
        """(cache, first token) for prompt ``i``; call ``release`` after copying from it."""
        with self.cv:
            while i not in self.ready:
                if self.error is not None:
                    raise self.error
                self.cv.wait()
            cache, t, done = self.ready.pop(i)
        torch.cuda.current_stream().wait_event(done)
        return cache, t

    def release(self, cache):
        ev = torch.cuda.Event()
        ev.record()
        with self.cv:
            self.free.append((cache, ev))
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()
        self.thread.join()


class BatchDecoder:
    """Greedy decoding of many prompts through one model, ``batch`` rows at a time.

    ``sizes`` are the batch sizes with a compiled step (the largest is the full batch); when the
    queue is empty and the live rows fit a smaller size, they are compacted into it.
    """

    def __init__(self, model, batch: int = 16, max_len: int = 8192, eos=None,
                 compile: bool = True, sizes=None, mode: str = "graphs",
                 fullgraph: bool = True, skip_qwen: bool = False, chip_per_full: int = 3,
                 fold_gqa: bool = False, view: int = 0, prefetch: int = 2,
                 freeze_chip_positions: bool = False, kv_len: str | None = None,
                 transparent_chip: bool = False):
        from transformers import StaticCache

        from chipformer.q35.fast import _n_chip_layers, chip_meta, exact_matmuls

        exact_matmuls()  # batch sizes change cuBLAS's kernel choice; keep the circuit exact
        if fold_gqa:
            _use_fold(model)

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
        # generate's bad_words_ids / suppress_tokens (single tokens): never predicted
        gc = model.generation_config
        ban = set(getattr(gc, "suppress_tokens", None) or [])
        for w in getattr(gc, "bad_words_ids", None) or []:
            assert len(w) == 1, f"multi-token bad word {w} is not supported"
            ban.add(w[0])
        self.banned = torch.tensor(sorted(ban), dtype=torch.long, device=self.dev) if ban else None
        self.small = StaticCache(config=self.cfg, max_cache_len=max_len)
        self.sizes = sorted(set(sizes or [batch]) | {batch})
        self.batch = batch
        self.compile = compile
        self.mode, self.fullgraph = mode, fullgraph
        self.levels = {}  # batch size -> (rows cache, tok, pos, step, chip step, chip keys)
        self.ready = set()
        self.pool, self.graphs = None, []
        # attention views: KV slots read per step, by the longest live row
        self.views = sorted({*range(view, max_len, view), max_len}) if view else [max_len]
        self.view_graphs = {}  # batch size -> {view: [text step, chip step]}
        self.raw = {}  # batch size -> compiled (uncaptured) steps
        self.prefetch = prefetch  # prompts prefilled ahead, on a second stream
        self.spare = []  # their batch-1 caches
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        self.slots = torch.arange(max_len, device=self.dev)
        self.chip_per_full = chip_per_full
        # kv_len="generate": attention reads exactly generate's static cache length
        # (prompt + max_new) in the prefill and every step. The attention kernel's result
        # depends on that length, so this is what makes batch 1 bit-identical to
        # generate(cache_implementation="static", disable_compile=True). Eager kernels only.
        self.kv_exact = kv_len == "generate"
        assert not (self.kv_exact and compile), "kv_len='generate' needs compile=False"
        meta = chip_meta(model)
        self.rpos = {}  # batch size -> RoPE position buffer (``pos`` itself unless frozen)
        self.freeze = (freeze_chip_positions or transparent_chip) and meta is not None
        if meta is not None:
            self.chip_lo = meta["chip_token_offset"]
            self.chip_hi = self.chip_lo + meta["chip_vocab"]
        # transparent_chip (implies freeze_chip_positions): chip tokens are invisible to Qwen's
        # layers. A chip-token input leaves their conv windows as they were (the stock model
        # shifts in a zero), and Qwen's full attention keeps its KV by text position: a chip
        # token's key goes to the slot the next text token overwrites, so text keys are
        # contiguous and no query ever reads a chip key. Qwen's decode steps then compute what
        # they would on the same text with the chip tokens deleted (dist/spikes/b26_*.py).
        self.transparent = transparent_chip and meta is not None
        if meta is not None:
            self.n_chip = _n_chip_layers(model)
        self.skip_qwen = skip_qwen and meta is not None
        if self.skip_qwen:
            self.n_chip, self.chip_dims = _n_chip_layers(model), meta["chip_dims_offset"]
            self.chip_lo = meta["chip_token_offset"]
            self.chip_hi = self.chip_lo + meta["chip_vocab"]
        self.stats = {"view_sum": 0, "chip_steps": 0, "steps": 0, "row_steps": 0, "tokens": 0, "prefills": 0, "prefill_s": 0.0}

    # --- one batch size ---------------------------------------------------------------------

    def _level(self, n: int):
        if n not in self.levels:
            pos = _mark(torch.zeros(n, dtype=torch.long, device=self.dev))
            tok = _mark(torch.zeros(n, 1, dtype=torch.long, device=self.dev))
            # RoPE position per row: the slot, or with freeze_chip_positions the number of
            # non-chip tokens before it (chip layers have no RoPE; Qwen sees contiguous text)
            rpos = _mark(torch.zeros(n, dtype=torch.long, device=self.dev)) if self.freeze else pos
            self.rpos[n] = rpos
            transparent = self.transparent
            kvpos = rpos if transparent else pos  # the KV slot each row writes
            rows = _Rows(self.cfg, self.small, n, self.max_len, pos, kvpos)
            cache, slots, lm, head = rows.cache, self.slots, self.lm, self.head
            if transparent:
                kinds = self.cfg.layer_types
                qlin = [cache.layers[i] for i in range(self.n_chip, len(kinds))
                        if kinds[i] == "linear_attention"]
                lo, hi = self.chip_lo, self.chip_hi

            def step(tok, pos):
                # slots past the view are masked anyway: reading fewer of them is bit-identical
                # (fully masked key blocks add exact zeros; dist/spikes/b14_fold.py)
                mask = slots[None, :rows.view] <= kvpos[:, None]
                if transparent:
                    ischip = (tok[:, 0] >= lo) & (tok[:, 0] < hi)
                    saved = [L.conv_states[0].clone() for L in qlin]
                h = lm(input_ids=tok, position_ids=rpos[:, None], past_key_values=cache,
                       attention_mask={"full_attention": mask[:, None, None, :],
                                       "linear_attention": None},
                       use_cache=True).last_hidden_state[:, -1]
                if transparent:  # chip inputs: Qwen's conv windows stay as they were
                    for L, c in zip(qlin, saved):
                        L.conv_states[0].copy_(torch.where(ischip[:, None, None], c,
                                                           L.conv_states[0]))
                return self._argmax(head(h))

            chip = kconst = None
            act = _mark(torch.ones(n, dtype=torch.bool, device=self.dev))
            if self.skip_qwen:
                chip, kconst = self._chip_step(rows, n, act, kvpos)
            if self.compile:
                # "graphs": inductor kernels, captured below as one plain CUDA graph per step
                # kind (no cudagraph-tree bookkeeping between the two kinds of step)
                mode = None if self.mode == "graphs" else self.mode
                step = torch.compile(step, mode=mode, fullgraph=self.fullgraph)
                if chip is not None:
                    chip = torch.compile(chip, mode=mode, fullgraph=self.fullgraph)
            self.levels[n] = (rows, tok, pos, step, chip, kconst, act)
        return self.levels[n]

    def _chip_step(self, rows, n, act, kvpos):
        """The step for rows whose input is a chip token. Qwen's layers are exact no-ops there
        (muted: residual unchanged, DeltaNet state unchanged, conv window shifts in a zero,
        full attention stores a value of 0 under a fixed key; dist/spikes/b5_skip.py checks all
        of it bitwise), so they are replaced by those cache writes, and the head reads only the
        chip dims (Qwen's dims of the hidden state are exactly 0)."""
        from chipformer.q35.fast import SparseLinear

        lm, cache, kinds = self.lm, rows.cache, self.cfg.layer_types
        n_chip, d0 = self.n_chip, self.chip_dims
        attn = [cache.layers[i] for i in range(n_chip, len(kinds)) if kinds[i] == "full_attention"]
        lin = [cache.layers[i] for i in range(n_chip, len(kinds))
               if kinds[i] == "linear_attention"]
        kconst = [_mark(L.keys.new_zeros(L.keys.shape[1], L.keys.shape[-1])) for L in attn]
        w = self.head.weight
        c0 = self.head.c0 if isinstance(self.head, SparseLinear) else 0
        c1 = c0 + w.shape[1]
        if not isinstance(self.head, SparseLinear):
            from chipformer.q35.fast import _hull, nonzero_cols

            c1 = _hull(nonzero_cols(w))[1]
        block = w[:, d0 - c0:c1 - c0].contiguous()
        chip_layers = lm.layers[:n_chip]
        chip_lin = [cache.layers[i] for i in range(n_chip) if kinds[i] == "linear_attention"]
        transparent = self.transparent
        def chip(tok, pos):
            # rows with act False (text rows waiting for a full step) come out untouched;
            # ``act`` is the level's buffer, all True unless the caller sets it
            saved = [(L.conv_states[0].clone(), L.recurrent_states[0].clone()) for L in chip_lin]
            h = lm.embed_tokens(tok)
            for layer in chip_layers:
                h = layer(h, position_embeddings=None, past_key_values=cache, use_cache=True)
            a3 = act[:, None, None]
            for L, (c, r) in zip(chip_lin, saved):
                L.conv_states[0].copy_(torch.where(a3, L.conv_states[0], c))
                L.recurrent_states[0].copy_(torch.where(act[:, None, None, None],
                                                        L.recurrent_states[0], r))
            idx = torch.arange(pos.shape[0], device=pos.device)
            for L, k in zip(attn, kconst):
                L.keys[idx, :, kvpos] = torch.where(a3, k, L.keys[idx, :, kvpos])
                L.values[idx, :, kvpos] = torch.where(a3, 0.0, L.values[idx, :, kvpos])
            if not transparent:  # transparent: Qwen's conv windows stay as they were
                for L in lin:
                    c = L.conv_states[0]
                    c.copy_(torch.where(a3, torch.nn.functional.pad(c[..., 1:], (0, 1)), c))
            h = lm.norm(h)[:, -1, d0:c1]
            return self._argmax(torch.nn.functional.linear(h, block))

        return chip, kconst

    def _argmax(self, logits):
        if self.banned is not None:  # as generate's logits processors do: score -inf
            logits = logits.index_fill(-1, self.banned, float("-inf"))
        return logits.argmax(-1)

    @torch.no_grad()
    def prefill(self, ids: list[int]) -> int:
        """Batch-1 prefill into ``self.small`` (as ``generate`` does it); returns the first token."""
        t0 = time.perf_counter()
        t = self._prefill_into(self.small, ids)
        self.stats["prefills"] += 1
        self.stats["prefill_s"] += time.perf_counter() - t0
        return t

    def _rope_positions(self, ids: list[int]) -> torch.Tensor:
        if not self.freeze:
            return torch.arange(len(ids), device=self.dev)
        lo, hi = self.chip_lo, self.chip_hi
        p, out = 0, []
        for t in ids:
            out.append(p)
            p += not lo <= t < hi
        return torch.tensor(out, device=self.dev)

    def _is_text(self, tok):
        return (tok < self.chip_lo) | (tok >= self.chip_hi)

    def chip_mask(self, ids: list[int]) -> torch.Tensor:
        """[len(ids)] bool on the device: which tokens are chip tokens."""
        x = torch.tensor(ids, device=self.dev)
        return (x >= self.chip_lo) & (x < self.chip_hi)

    def mark_prompt(self, n: int, b: int, ids: list[int]) -> None:
        """Call after loading a prefilled prompt into row ``b`` of the batch-``n`` level. With
        transparent_chip it moves the row's full-attention KV to text positions (the prefill
        wrote one slot per token); a no-op otherwise."""
        if not self.transparent:
            return
        text = (~self.chip_mask(ids)).nonzero().flatten()
        if len(text) == len(ids):
            return
        for L in self.levels[n][0].cache.layers:
            if getattr(L, "keys", None) is not None:
                L.keys[b, :, :len(text)] = L.keys[b, :, text]
                L.values[b, :, :len(text)] = L.values[b, :, text]

    def _prefill_into(self, cache, ids: list[int]) -> int:
        cache.reset()
        x = torch.tensor([ids], device=self.dev)
        mask = torch.ones_like(x)
        if self.transparent:
            ch = self.chip_mask(ids)
            if bool(ch.any()):
                # chip keys are invisible (a chip query keeps its own). A leading run of chip
                # tokens (the BIOS) also matches Qwen's zero-padded conv window exactly; chip
                # tokens later in a prompt still shift zeros into the conv windows.
                n = len(ids)
                size = next(L.max_cache_len for L in cache.layers if hasattr(L, "max_cache_len"))
                q = torch.arange(n, device=self.dev)[:, None]
                k = torch.arange(size, device=self.dev)[None, :]
                vis = (k <= q) & (~torch.nn.functional.pad(ch, (0, size - n))[None, :] | (k == q))
                mask = {"full_attention": vis[None, None], "linear_attention": None}
        h = self.lm(input_ids=x, position_ids=self._rope_positions(ids)[None],
                    attention_mask=mask, past_key_values=cache,
                    use_cache=True).last_hidden_state[:, -1:]
        return int(self._argmax(self.head(h)[0, -1]))

    @torch.no_grad()
    def _prepare(self, n: int):
        """The level of batch size ``n``, compiled and warmed; for the chip step, its fixed
        keys are taken from a real full step on a chip-token input."""
        level = self._level(n)
        if n in self.ready:
            return level
        rows, tok, pos, step, chip, kconst, _act = level
        tok.zero_()
        pos.zero_()
        for _ in range(3):
            step(tok, pos)
        if chip is not None:
            from chipformer.ref.emu import Chip8
            from chipformer.trace import boot_tokens

            boot = [t + self.chip_lo for t in boot_tokens(Chip8(bytes([0x12, 0x00])).cells)]
            k = len(boot) - 1
            self.prefill(boot[:k])
            rows.load_row(0, self.small, k)
            tok[0, 0], pos[0] = boot[k], k
            step(tok, pos)
            kinds = self.cfg.layer_types
            attn = [rows.cache.layers[i] for i in range(self.n_chip, len(kinds))
                    if kinds[i] == "full_attention"]
            for L, kc in zip(attn, kconst):
                kc.copy_(L.keys[0, :, int((self.rpos[n] if self.transparent else pos)[0])])
            tok.fill_(boot[k])
            pos.zero_()
            for _ in range(3):
                chip(tok, pos)
            tok.zero_()
            pos.zero_()
        if self.mode == "graphs":  # compiled or eager kernels, captured as CUDA graphs
            # the compiled functions own buffers their kernels read (e.g. inductor's copies of
            # closure tensors): they must outlive the graphs, or replays read freed memory
            self.raw[n] = (step, chip)
            graphs = {}
            for view in self.views:
                rows.view = view
                tok.zero_()
                pos.zero_()
                for _ in range(2):
                    step(tok, pos)
                g = [self._capture(step, tok, pos), None]
                if chip is not None:  # the chip step reads no keys: one graph serves all views
                    if not graphs:
                        tok.fill_(self.chip_lo)
                        for _ in range(2):
                            chip(tok, pos)
                        g[1] = self._capture(chip, tok, pos)
                    else:
                        g[1] = next(iter(graphs.values()))[1]
                graphs[view] = g
            rows.view = self.max_len
            tok.zero_()
            pos.zero_()
            self.view_graphs[n] = graphs
            level = self.levels[n] = (rows, tok, pos, *graphs[self.max_len], kconst, _act)
        torch.cuda.synchronize()
        self.ready.add(n)
        return level

    def _view_graph(self, n: int, v: int):
        """[text step, chip step] graphs of batch size ``n`` whose attention reads ``v`` slots."""
        graphs = self.view_graphs[n]
        if v not in graphs:
            rows, tok, pos = self.levels[n][:3]
            rows.view = v
            g = self._capture(self.raw[n][0], tok, pos, warm=False)
            rows.view = self.max_len
            graphs[v] = [g, graphs[self.max_len][1]]
        return graphs[v]

    def _capture(self, fn, tok, pos, warm: bool = True):
        """``fn(tok, pos)`` as a CUDA graph over the level's static buffers; the returned
        callable takes the same (ignored: always those buffers) arguments. ``warm=False``
        records without running it first (safe mid-run: recording runs no kernel)."""
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warm up on the capture stream (lazy cuBLAS state)
            for _ in range(2 if warm else 0):
                fn(tok, pos)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        # thread_local: the prefetch thread may be running a prefill on its own stream
        with torch.cuda.graph(g, pool=self.pool, stream=s, capture_error_mode="thread_local"):
            out = fn(tok, pos)
        self.pool = g.pool()
        self.graphs.append(g)

        def replay(tok_=None, pos_=None):
            g.replay()
            return out

        return replay

    @torch.no_grad()
    def warmup(self, prompt: list[int] | None = None) -> None:
        """Compile (and record the CUDA graphs of) every batch size's steps ahead of time."""
        if prompt is not None:
            self.prefill(prompt)
        for n in self.sizes:
            self._prepare(n)

    # --- the loop -----------------------------------------------------------------------------

    @torch.no_grad()
    def run(self, prompts: list[list[int]], max_new: int, on_done=None,
            hook=None) -> list[list[int]]:
        """Greedy continuations (each ending with its end token, or ``max_new`` long).

        ``hook(i, t) -> (t_used, stop)`` sees every token of prompt ``i`` as it is produced: it
        may replace it (token forcing: ``t_used`` is what is kept and fed back) or end the row
        (``stop``). With a hook the loop reads every step before queueing the next one.
        ``on_done(i, tokens)`` is called as each prompt finishes."""
        for p in prompts:
            assert len(p) + max_new <= self.max_len, (len(p), max_new, self.max_len)
        if not prompts:
            return []
        if not self.small.layers[0].is_conv_states_initialized[0]:
            self.prefill(prompts[0])  # the batched cache copies a prefilled cache's shapes
        self._prepare(self.batch)
        lens = [len(p) + max_new for p in prompts] if self.kv_exact else None
        # record every graph now (all batch sizes, and generate-length views): capturing
        # mid-run would race the prefetch thread's allocations
        for m in self.sizes:
            self._prepare(m)
            for v in sorted(set(lens or [])) if m in self.view_graphs else []:
                self._view_graph(m, min(v, self.max_len))
        pre = _Prefetcher(self, prompts, range(len(prompts)), max(1, self.prefetch), lens)
        try:
            return self._run(prompts, max_new, on_done, pre, hook)
        finally:
            pre.close()

    def _run(self, prompts, max_new, on_done, pre, hook=None):
        out: list[list[int] | None] = [None] * len(prompts)
        queue = deque(range(len(prompts)))
        n = self.batch
        rows, tok, pos, step, chip, _, act = self._prepare(n)
        rpos = self.rpos[n]
        live: list[int | None] = [None] * n  # prompt index per row
        gen: list[list[int]] = [[] for _ in range(n)]
        plen = [0] * n  # prompt length per row

        def finish(b):
            i = live[b]
            out[i] = gen[b]
            live[b] = None
            if on_done is not None:
                on_done(i, gen[b])

        def admit(b):
            while queue:
                i = queue.popleft()
                cache, t = pre.get(i)
                stop = False
                if hook is not None:
                    t, stop = hook(i, t)
                live[b], gen[b] = i, [t]
                if stop or t in self.eos or max_new <= 1:
                    pre.release(cache)
                    finish(b)
                    continue
                rows.load_row(b, cache, len(prompts[i]))
                pre.release(cache)
                tok[b, 0] = t
                pos[b] = len(prompts[i])
                plen[b] = len(prompts[i])
                if self.freeze:
                    rpos[b] = sum(not self.chip_lo <= x < self.chip_hi for x in prompts[i])
                self.mark_prompt(n, b, prompts[i])
                return
            tok[b, 0] = 0  # an idle row decodes garbage at slot 0 until it is reused
            pos[b] = 0

        for b in range(n):
            admit(b)
        # The host runs one step ahead: step t+1 is queued before the tokens of step t are read,
        # so the GPU never waits for Python. A row that finishes at step t has its (stale) step
        # t+1 token dropped; a row refilled then starts with step t+2.
        host = [torch.zeros(self.batch, dtype=torch.long, pin_memory=True) for _ in range(2)]
        skip = [0] * n
        flip = 0

        lo, hi = (self.chip_lo, self.chip_hi) if chip is not None else (0, 0)
        budget = 0  # chip steps left before the waiting text rows get a full step

        def launch():
            """One step. With skip_qwen, rows whose input is a chip token take the cheap chip
            step while text rows wait (up to ``chip_per_full`` chip steps per full step); a full
            step advances every row (it is exact for chip rows too)."""
            nonlocal flip, budget
            on = None
            if chip is not None:
                on = [live[b] is not None and lo <= gen[b][-1] < hi for b in range(n)]
                text = any(live[b] is not None and not on[b] for b in range(n))
                if any(on) and (not text or budget > 0):
                    budget -= 1
                else:
                    on = None
                    budget = self.chip_per_full
            step_fn, chip_fn = step, chip
            if n in self.view_graphs and self.kv_exact:  # generate's cache length
                v = max((plen[b] + max_new for b in range(n) if live[b] is not None), default=0)
                step_fn, chip_fn = self._view_graph(n, min(max(v, 1), self.max_len))
            elif n in self.view_graphs:  # the smallest attention view covering every live row
                need = 1 + max((plen[b] + len(gen[b]) for b in range(n) if live[b] is not None),
                               default=0)
                v = next(v for v in self.views if v >= need)
                step_fn, chip_fn = self.view_graphs[n][v]
                self.stats["view_sum"] += v
            if on is not None:
                act.copy_(torch.tensor(on), non_blocking=True)
                nxt = chip_fn(tok, pos)
                tok.copy_(torch.where(act[:, None], nxt[:, None], tok))
                pos.add_(act)
                self.stats["chip_steps"] += 1
            else:
                nxt = step_fn(tok, pos)
                if self.freeze:  # the input was text: the next RoPE position moves on
                    rpos.add_(self._is_text(tok[:, 0]))
                tok.copy_(nxt[:, None])
                pos.add_(1)
            pos.clamp_(max=self.max_len - 1)  # idle rows count up; live rows never get there
            buf = host[flip][:n]
            flip ^= 1
            buf.copy_(nxt, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            self.stats["steps"] += 1
            self.stats["row_steps"] += n
            return ev, buf, on

        def collect(res, inflight: bool):
            ev, buf, on = res
            ev.synchronize()
            for b, t in enumerate(buf.tolist()):
                if skip[b]:
                    skip[b] -= 1
                    continue
                if live[b] is None or (on is not None and not on[b]):
                    continue
                stop = False
                if hook is not None:
                    t2, stop = hook(live[b], t)
                    if t2 != t:  # forced: feed the forced token instead
                        tok[b, 0] = t2
                        t = t2
                gen[b].append(t)
                self.stats["tokens"] += 1
                if stop or t in self.eos or len(gen[b]) >= max_new:
                    finish(b)
                    skip[b] = int(inflight)
                    admit(b)

        pending = None
        while any(i is not None for i in live):
            cur = launch()
            if chip is not None or hook is not None:  # the next step depends on them: no lag
                collect(cur, False)
                cur = None
            elif pending is not None:
                collect(pending, True)
            pending = cur
            if not queue:
                alive = [b for b in range(n) if live[b] is not None]
                smaller = [s for s in self.sizes if s < n and s >= len(alive)]
                if smaller:
                    if pending is not None:
                        collect(pending, False)
                    pending = None
                    alive = [b for b in range(n) if live[b] is not None]
                    if not alive:
                        break
                    m = smaller[0]
                    rows2, tok2, pos2, step2, chip2, _, act2 = self._prepare(m)
                    idx = torch.tensor(alive, device=self.dev)
                    rows.take(idx, rows2)
                    tok2[: len(alive)].copy_(tok[idx])
                    if self.freeze:
                        self.rpos[m][: len(alive)].copy_(rpos[idx])
                        self.rpos[m][len(alive):] = 0
                    pos2[: len(alive)].copy_(pos[idx])
                    pos2[len(alive):] = 0
                    tok2[len(alive):] = 0
                    live = [live[b] for b in alive] + [None] * (m - len(alive))
                    gen = [gen[b] for b in alive] + [[] for _ in range(m - len(alive))]
                    plen = [plen[b] for b in alive] + [0] * (m - len(alive))
                    skip = [0] * m
                    n, rows, tok, pos, step, chip, act = m, rows2, tok2, pos2, step2, chip2, act2
                    rpos = self.rpos[m]
        if pending is not None:
            collect(pending, False)
        return out


_DECODERS: dict = {}


def generate_batch(model, tok, prompts: list[list[int]], max_new: int, batch: int | None = None,
                   max_len: int | None = None, **kw) -> list[list[int]]:
    """Greedy continuations of ``prompts`` (token id lists), equal to running each alone.

    ``tok`` is unused (end tokens come from ``model.generation_config``); it is accepted so the
    call reads like the usual ``generate`` helpers. The decoder (and its compiled steps) is
    reused by later calls with the same model, batch, ``max_len`` and options; a call with
    different ones replaces it (one decoder per model, to bound memory).
    """
    del tok
    if not prompts:
        return []
    batch = batch or min(16, len(prompts))
    if max_len is None:
        need = max(len(p) for p in prompts) + max_new
        max_len = -(-need // 1024) * 1024
    sizes = kw.pop("sizes", None) or sorted({batch, *(s for s in (1, 2, 4, 8, 16, 32)
                                                      if s < batch)})
    key = (batch, max_len, tuple(sizes), tuple(sorted(kw.items())))
    old = _DECODERS.get(id(model))
    if old is None or old[0] != key:
        _DECODERS.pop(id(model), None)
        torch.cuda.empty_cache()
        _DECODERS[id(model)] = (key, BatchDecoder(model, batch=batch, max_len=max_len,
                                                  sizes=sizes, **kw))
    return _DECODERS[id(model)][1].run(prompts, max_new)


def generate_fast(model, prompts: list[list[int]], max_new: int, batch: int | None = None,
                  compact_qwen: bool = False, skip_qwen: bool = True, **kw) -> list[list[int]]:
    """Fast greedy decoding of a merged q35 checkpoint: the drop-in for

        [model.generate(torch.tensor([p]), do_sample=False, max_new_tokens=max_new)[0, len(p):]
         .tolist() for p in prompts]

    Each continuation ends with its end token (generation_config.eos_token_id) or is
    ``max_new`` long. On first use it compacts the chip layers in place (``fast.compact``: only
    their nonzero weight blocks are read, every logit stays bit-identical), and with
    ``skip_qwen`` chip-token inputs run only the chip layers (Qwen's layers are exact no-ops
    there). ``compact_qwen=True`` also compacts Qwen's layers and the head: faster at text
    steps, but Qwen's sums round differently (1 of 12 demo answers changed its final period). Prompts are
    decoded ``batch`` at a time with continuous batching (``batch=1`` is the latency setting).
    The compacted model still works with ``generate`` and plain forwards.
    """
    from chipformer.q35.fast import compact

    done = getattr(model, "_q35_compact", None)
    if done is None:
        compact(model, qwen=compact_qwen)
        model._q35_compact = compact_qwen
    elif compact_qwen and not done:
        compact(model, qwen=True)
        model._q35_compact = True
    return generate_batch(model, None, prompts, max_new, batch=batch, skip_qwen=skip_qwen, **kw)
