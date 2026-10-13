"""Configuration for chipformer: a transformer whose weights were set by hand to execute CHIP-8."""

from transformers import PretrainedConfig


class ChipformerConfig(PretrainedConfig):
    """Shape of the network plus the machine constants a host needs to talk to it.

    ``layers`` has one entry per layer: ``{"heads": [[qk_dim, v_dim], ...], "ffn_units": n,
    "head_names": [...]}``. Heads have their own (tiny) widths, so they are not one fused
    tensor. ``machine`` holds the token layout (ADDR_BASE, STEP, TICK, ...), the font and
    the quirk profiles; see docs/trace_format.md in the GitHub repo.
    """

    model_type = "chipformer"

    def __init__(self, d_model: int = 382, vocab_size: int = 8708, beta: float = 1000.0,
                 layers: list | None = None, machine: dict | None = None,
                 max_position: int = 1 << 22, rollover_at: int = 250_000, **kwargs):
        self.d_model = d_model
        self.hidden_size = d_model
        self.vocab_size = vocab_size
        self.beta = beta
        self.layers = layers or []
        self.n_layers = len(self.layers)
        self.num_hidden_layers = len(self.layers)
        self.machine = machine or {}
        self.max_position = max_position  # the circuits assume positions stay below this
        self.rollover_at = rollover_at  # the host restarts the context from a state dump here
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
