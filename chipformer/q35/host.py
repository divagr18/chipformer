"""Drive a q35 checkpoint with chipformer.session.Session (prefill + cached decode).

``Feeder.feed(token)`` processes one position through the HF model with its cache and returns the
greedy next token. Prompts can be pushed in one prefill with ``prefill``; the host's TICK and key
injections then go through the normal decode path.
"""

import torch

from chipformer.session import Session


class Feeder:
    def __init__(self, model, token_ids=None):
        """``token_ids`` maps a chip token to the model's vocabulary (identity for tiny models)."""
        self.model = model
        self.ids = token_ids
        self.chip = None if token_ids is None else torch.tensor(token_ids)
        self.cache = None
        self.pending: list[int] = []
        self.n = 0

    def _id(self, t: int) -> int:
        return t if self.ids is None else self.ids[t]

    def _run(self, toks: list[int]) -> int:
        x = torch.tensor([[self._id(t) for t in toks]], device=self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.n += len(toks)
        logits = out.logits[0, -1].float()
        if self.chip is not None:
            return int(logits[self.chip].argmax())
        return int(logits.argmax())

    def feed(self, token: int) -> int:
        return self._run([token])

    def reset(self) -> None:
        self.cache = None
        self.n = 0


def session(model, rom: bytes, quirks=None, token_ids=None) -> Session:
    """A Session whose boot prompt goes through one prefill, then decodes token by token."""
    f = Feeder(model, token_ids)
    boot: list[int] = []

    def feed(tok: int) -> int:
        if boot is not None and collecting[0]:
            boot.append(tok)
            return -1
        return f.feed(tok)

    collecting = [True]
    s = Session(feed, rom, quirks)
    collecting[0] = False
    s.pred = f._run(boot)
    s.feeder = f
    return s
