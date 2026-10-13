"""chipformer: a decoder-only transformer whose weights were set by hand to execute CHIP-8.

The network is ordinary: token embedding plus a position embedding from [t, t^2], causal
softmax attention heads with residual connections, ReGLU feed-forward layers and a linear
unembedding, all in float64, with no normalization. Nothing about CHIP-8 is in this file
except the host loop at the bottom (ChipformerGame), which only feeds the boot prompt,
injects TICK and key tokens, and reads the screen back from the model's own writes.

    model = AutoModelForCausalLM.from_pretrained("divagr1925/chipformer", trust_remote_code=True,
                                                 dtype=torch.float64)
    game = model.play(open("tetris.ch8", "rb").read(), quirks="shift,loadstore,clip,jump", ipf=15)
    for _ in range(60):
        game.frame()
    print(game.ascii())
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_chipformer import ChipformerConfig

SPARSE_MIN = 100_000  # weights; smaller matrices stay dense


class ChipformerCache:
    """Keys and values per layer, shaped (batch, heads, positions, width), plus the length.

    Buffers grow by doubling, so appending one token costs O(1) amortized.
    """

    def __init__(self):
        self.kv: dict[int, list[torch.Tensor]] = {}
        self.length = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def append(self, li: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store a layer's new keys/values; returns all keys and values so far."""
        start, n = self.length, k.shape[-2]
        if li not in self.kv:
            cap = max(1024, 2 * n)
            self.kv[li] = [x.new_zeros(*x.shape[:-2], cap, x.shape[-1]) for x in (k, v)]
        buf = self.kv[li]
        if start + n > buf[0].shape[-2]:
            cap = max(2 * buf[0].shape[-2], start + n)
            for j, old in enumerate(buf):
                buf[j] = old.new_zeros(*old.shape[:-2], cap, old.shape[-1])
                buf[j][..., :start, :] = old[..., :start, :]
        buf[0][..., start:start + n, :] = k
        buf[1][..., start:start + n, :] = v
        return buf[0][..., :start + n, :], buf[1][..., :start + n, :]


class ChipformerHead(nn.Module):
    def __init__(self, d: int, qk: int, v: int):
        super().__init__()
        self.wq = nn.Parameter(torch.zeros(qk, d, dtype=torch.float64))
        self.wk = nn.Parameter(torch.zeros(qk, d, dtype=torch.float64))
        self.wv = nn.Parameter(torch.zeros(v, d, dtype=torch.float64))
        self.wo = nn.Parameter(torch.zeros(d, v, dtype=torch.float64))


class ChipformerFFN(nn.Module):
    """ReGLU: w_down (relu(w_gate x) * (w_up x))."""

    def __init__(self, d: int, units: int):
        super().__init__()
        self.w_gate = nn.Parameter(torch.zeros(units, d, dtype=torch.float64))
        self.w_up = nn.Parameter(torch.zeros(units, d, dtype=torch.float64))
        self.w_down = nn.Parameter(torch.zeros(d, units, dtype=torch.float64))


class ChipformerLayer(nn.Module):
    def __init__(self, d: int, spec: dict):
        super().__init__()
        self.attn = nn.ModuleList(ChipformerHead(d, qk, v) for qk, v in spec["heads"])
        self.ffn = ChipformerFFN(d, spec["ffn_units"])


class ChipformerEmbedding(nn.Module):
    def __init__(self, vocab: int, d: int):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(vocab, d, dtype=torch.float64))
        self.pos_poly = nn.Parameter(torch.zeros(d, 2, dtype=torch.float64))


class ChipformerPreTrainedModel(PreTrainedModel):
    config_class = ChipformerConfig
    base_model_prefix = "chipformer"
    _supports_cache_class = False

    def _init_weights(self, module):
        pass  # every weight comes from the checkpoint; there is nothing to initialize


class ChipformerForCausalLM(ChipformerPreTrainedModel):
    def __init__(self, config: ChipformerConfig):
        super().__init__(config)
        d = config.d_model
        self.embed = ChipformerEmbedding(config.vocab_size, d)
        self.layers = nn.ModuleList(ChipformerLayer(d, s) for s in config.layers)
        self.unembed = nn.Parameter(torch.zeros(config.vocab_size, d, dtype=torch.float64))
        self._csr: dict[str, torch.Tensor] | None = None
        self._fused: list[tuple[torch.Tensor, ...]] | None = None
        self.post_init()

    def sparse(self, enable: bool = True) -> "ChipformerForCausalLM":
        """Multiply by sparse (CSR) copies of the feed-forward and unembedding matrices.

        Only 0.4% of the weights are nonzero, and the dense feed-forward matrices are most of
        the 259 MB that every decoded token otherwise reads. The arithmetic is exact either
        way (the circuits only produce values float64 represents exactly), so the outputs
        are identical.
        """
        if not enable:
            self._csr = None
            return self
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "sparse CSR support is in beta"
            # small matrices are faster dense; only the big ones are worth a sparse kernel
            self._csr = {name: p.detach().to_sparse_csr() for name, p in self.named_parameters()
                         if (".ffn." in name or name == "unembed") and p.numel() >= SPARSE_MIN}
        return self

    def _lin(self, x: torch.Tensor, w: torch.Tensor, name: str) -> torch.Tensor:
        """x @ w.T, through the sparse copy when enabled."""
        m = None if self._csr is None else self._csr.get(name)
        if m is None:
            return x @ w.T
        y = m @ x.reshape(-1, x.shape[-1]).T
        return y.T.reshape(*x.shape[:-1], m.shape[0])

    def _heads(self, li: int) -> tuple[torch.Tensor, ...] | None:
        """A layer's heads stacked into (heads, width, d) tensors, zero-padded to one width.

        Padding adds exact zeros, so this is the same computation as running each head alone.
        Built on first use; call ``refresh()`` after changing the weights in place.
        """
        if self._fused is None:
            self._fused = []
            for layer in self.layers:
                if not len(layer.attn):
                    self._fused.append(None)
                    continue
                qk = max(h.wq.shape[0] for h in layer.attn)
                vd = max(h.wv.shape[0] for h in layer.attn)

                def stack(ws, width):
                    return torch.stack([nn.functional.pad(w.detach(), (0, 0, 0, width - w.shape[0]))
                                        for w in ws])
                self._fused.append((
                    stack([h.wq for h in layer.attn], qk), stack([h.wk for h in layer.attn], qk),
                    stack([h.wv for h in layer.attn], vd),
                    stack([h.wo.T for h in layer.attn], vd).transpose(-1, -2)))
        return self._fused[li]

    def refresh(self) -> None:
        """Rebuild the stacked/sparse copies after editing weights in place."""
        self._fused = None
        if self._csr is not None:
            self.sparse()

    def get_input_embeddings(self):
        return None

    def _attend(self, q, k, v, start: int) -> torch.Tensor:
        """Causal softmax attention; query i sits at position start + i, key j at position j."""
        s = q @ k.transpose(-1, -2)
        n, total = s.shape[-2], s.shape[-1]
        if n > 1:
            qpos = torch.arange(start, start + n, device=s.device)[:, None]
            s = s.masked_fill(torch.arange(total, device=s.device)[None, :] > qpos, float("-inf"))
        s = s - s.amax(-1, keepdim=True)
        p = torch.exp(self.config.beta * s)
        return (p / p.sum(-1, keepdim=True)) @ v

    def forward(self, input_ids: torch.LongTensor, past_key_values: ChipformerCache | None = None,
                use_cache: bool | None = None, labels: torch.LongTensor | None = None,
                logits_to_keep: int = 0, return_dict: bool | None = None, **kwargs):
        """input_ids: (batch, n). Positions continue from the cache length (absolute index)."""
        cache = past_key_values if past_key_values is not None else ChipformerCache()
        start = cache.length
        n = input_ids.shape[-1]
        if start + n > self.config.max_position:
            raise ValueError(f"position {start + n} exceeds max_position {self.config.max_position}; "
                             "restart the context from a state dump (see ChipformerGame.rollover)")
        t = torch.arange(start, start + n, dtype=torch.float64, device=input_ids.device)
        r = self.embed.token[input_ids] + torch.stack([t, t * t], 1) @ self.embed.pos_poly.T
        for li, layer in enumerate(self.layers):
            heads = self._heads(li)
            if heads is not None:  # some layers are feed-forward only
                wq, wk, wv, wo = heads
                x = r.unsqueeze(-3)  # (batch, 1, n, d): broadcast over heads
                k, v = cache.append(li, x @ wk.transpose(-1, -2), x @ wv.transpose(-1, -2))
                out = self._attend(x @ wq.transpose(-1, -2), k, v, start)  # (batch, heads, n, width)
                r = r + (out @ wo.transpose(-1, -2)).sum(-3)
            f, p = layer.ffn, f"layers.{li}.ffn"
            hidden = torch.relu(self._lin(r, f.w_gate, f"{p}.w_gate")) * self._lin(r, f.w_up, f"{p}.w_up")
            r = r + self._lin(hidden, f.w_down, f"{p}.w_down")
        logits = self._lin(r if not logits_to_keep else r[..., -logits_to_keep:, :], self.unembed, "unembed")
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                                               labels[..., 1:].reshape(-1))
        cache.length = start + n
        return CausalLMOutputWithPast(loss=loss, logits=logits,
                                      past_key_values=None if use_cache is False else cache)

    def play(self, rom: "bytes | str | Path", quirks="octo", ipf: int = 10,
             keep_tokens: bool = False, sparse: bool = True) -> "ChipformerGame":
        """Boot a CHIP-8 ROM; call ``.frame()`` on the result once per 60 Hz frame.

        ``sparse`` switches on the exact sparse matmuls (see ``sparse()``); pass False to run
        the dense weights as loaded.
        """
        if sparse and self._csr is None:
            self.sparse()
        elif not sparse:
            self.sparse(False)
        return ChipformerGame(self, rom, quirks=quirks, ipf=ipf, keep_tokens=keep_tokens)


# --- host loop -----------------------------------------------------------------------------

MAX_RECORD = 2 * 40 + 2  # longer than any valid record


@dataclass
class _Machine:
    """Token layout and boot constants, read from config.machine."""

    VAL_BASE: int
    ADDR_BASE: int
    STEP: int
    TICK: int
    BOOT: int
    NUM_CELLS: int
    NULL: int
    CLS: int
    PC: int
    RNG: int
    KEYS: int
    DISPLAY: int
    DISPLAY_W: int
    DISPLAY_H: int
    PROGRAM_START: int
    FONT_BASE: int
    RNG_SEED: int
    QUIRKS: int
    FONT: list
    QUIRK_NAMES: list
    PROFILES: dict


class ChipformerGame:
    """Runs a CHIP-8 program on the model, the way the Rust/wasm engine does.

    The host never computes machine semantics. It (1) feeds the boot prompt
    ``BOOT <state dump> STEP``, (2) injects ``TICK`` and key writes ``ADDR k, VAL v, STEP``
    right after a STEP, and (3) keeps a mirror of the cells from the model's own writes so it
    can show the screen. Every other token is the model's greedy prediction.
    """

    def __init__(self, model: ChipformerForCausalLM, rom, quirks="octo", ipf: int = 10,
                 keep_tokens: bool = False):
        if isinstance(rom, (str, Path)):
            rom = Path(rom).read_bytes()
        self.model = model
        self.m = _Machine(**model.config.machine)
        self.ipf = ipf
        self.keep_tokens = keep_tokens
        self.tokens: list[int] = []
        self.instructions = 0
        self.rollovers = 0
        self.fed = 0  # tokens fed since boot, across rollovers
        m = self.m
        if len(rom) > 0x1000 - m.PROGRAM_START:
            raise ValueError(f"ROM too large: {len(rom)} bytes")
        cells = [0] * m.NUM_CELLS
        cells[m.FONT_BASE:m.FONT_BASE + len(m.FONT)] = m.FONT
        cells[m.PROGRAM_START:m.PROGRAM_START + len(rom)] = list(rom)
        cells[m.PC] = m.PROGRAM_START
        cells[m.RNG] = m.RNG_SEED
        for a, v in self._quirk_cells(quirks).items():
            cells[a] = v
        self.cells = cells
        self._new_context()

    def _quirk_cells(self, quirks) -> dict[int, int]:
        m = self.m
        if quirks is None:
            on = []
        elif isinstance(quirks, str):
            on = m.PROFILES[quirks] if quirks in m.PROFILES else [q for q in quirks.split(",") if q]
        else:
            on = list(quirks)
        bad = set(on) - set(m.QUIRK_NAMES)
        if bad:
            raise ValueError(f"unknown quirks {sorted(bad)}; known: {m.QUIRK_NAMES} or {list(m.PROFILES)}")
        return {m.QUIRKS + i: int(q in on) for i, q in enumerate(m.QUIRK_NAMES)}

    def boot_tokens(self) -> list[int]:
        """BOOT, the CLS write, every other nonzero cell in address order, STEP."""
        m, c = self.m, self.cells
        out = [m.BOOT, m.ADDR_BASE + m.CLS, m.VAL_BASE + c[m.CLS]]
        for a, v in enumerate(c):
            if a not in (m.CLS, m.NULL) and v:
                out += [m.ADDR_BASE + a, m.VAL_BASE + v]
        return out + [m.STEP]

    def _new_context(self) -> None:
        self.cache = ChipformerCache()
        self._feed(self.boot_tokens())

    @torch.inference_mode()
    def _feed(self, toks: list[int]) -> None:
        ids = torch.tensor([toks], dtype=torch.long, device=self.model.device)
        out = self.model(ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.cache = out.past_key_values
        self.pred = int(out.logits[0, -1].argmax())
        self.fed += len(toks)
        if self.keep_tokens:
            self.tokens += toks

    @property
    def position(self) -> int:
        return self.cache.length

    def _apply(self, writes) -> None:
        m = self.m
        for a, v in writes:
            if a == m.CLS:
                n = m.DISPLAY_W * m.DISPLAY_H // 8
                self.cells[m.DISPLAY:m.DISPLAY + n] = [0] * n
            self.cells[a] = v

    def _record(self) -> list[tuple[int, int]]:
        """Let the model write until it emits STEP."""
        m, writes = self.m, []
        for _ in range(MAX_RECORD):
            tok = self.pred
            if tok == m.STEP:
                self._feed([tok])
                self._apply(writes)
                return writes
            if not m.ADDR_BASE <= tok < m.STEP:
                raise RuntimeError(f"model emitted {tok} where ADDR/STEP was expected at {self.position}")
            self._feed([tok])
            val = self.pred
            if not m.VAL_BASE <= val < m.ADDR_BASE:
                raise RuntimeError(f"model emitted {val} after an ADDR at {self.position}")
            self._feed([val])
            writes.append((tok - m.ADDR_BASE, val - m.VAL_BASE))
        raise RuntimeError("record never ended")

    def step(self) -> list[tuple[int, int]]:
        """Execute one instruction (the model writes its effects)."""
        self.instructions += 1
        return self._record()

    def tick(self) -> list[tuple[int, int]]:
        """60 Hz timer tick: inject TICK, the model writes DT and ST."""
        self._feed([self.m.TICK])
        return self._record()

    def set_key(self, key: int, down: bool) -> None:
        m = self.m
        a = m.KEYS + key
        if bool(self.cells[a]) == down:
            return
        self._feed([m.ADDR_BASE + a, m.VAL_BASE + int(down), m.STEP])
        self.cells[a] = int(down)

    def rollover(self) -> None:
        """Restart the context from a state dump of the mirror (keeps positions bounded)."""
        self._new_context()
        self.rollovers += 1

    def frame(self, keys=()) -> None:
        """One 60 Hz frame: key changes, one TICK, then ``ipf`` instructions.

        ``keys``: the CHIP-8 keys (0..15) held this frame, or a 16-bit mask.
        """
        held = {k for k in range(16) if keys >> k & 1} if isinstance(keys, int) else set(keys)
        for k in range(16):
            self.set_key(k, k in held)
        self.tick()
        for _ in range(self.ipf):
            self.step()
        if self.position > self.model.config.rollover_at:
            self.rollover()

    def pixels(self) -> list[list[int]]:
        m = self.m
        w8 = m.DISPLAY_W // 8
        return [[self.cells[m.DISPLAY + y * w8 + x // 8] >> (7 - x % 8) & 1 for x in range(m.DISPLAY_W)]
                for y in range(m.DISPLAY_H)]

    def ascii(self) -> str:
        """The 64x32 screen as 16 lines of half-block characters."""
        p = self.pixels()
        chars = {(0, 0): " ", (1, 0): "▀", (0, 1): "▄", (1, 1): "█"}
        return "\n".join("".join(chars[p[y][x], p[y + 1][x]] for x in range(len(p[0])))
                         for y in range(0, len(p), 2))
