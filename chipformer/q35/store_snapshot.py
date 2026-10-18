"""Durability for the object store: carry its files from one session to the next.

Two ways:

``snapshot`` / ``restore``  the store lives in Gated DeltaNet state of the chip layers, so a
    snapshot is just those heads' recurrent states (a few MB however much is stored). ``restore``
    returns a fresh stock ``DynamicCache`` in which only these heads are filled; prefill a new
    conversation with it (``model(..., past_key_values=cache)`` or ``generate``) and Qwen starts
    from an empty context while its computer still has the files.

``disk_image``  a token record that rebuilds the store when it is prefilled at the start of a
    session: one ``put`` (and ``append``s) per live file with the circuit's answers.

The store's persistent heads are listed in the checkpoint's chipformer.json
(``store.persistent_heads``: [chip layer, value head]).
"""

import json
from pathlib import Path

import torch

from chipformer.q35 import store as S


def persistent(model_dir) -> list[list[int]]:
    meta = json.loads((Path(model_dir) / "chipformer.json").read_text())
    return meta["store"]["persistent_heads"]


def snapshot(cache, heads: list[list[int]], row: int = 0) -> dict:
    """The store's state in ``cache`` (batch row ``row``): {"heads", "states"}."""
    states = [cache.layers[layer].recurrent_states[0][row, vh].detach().clone().cpu()
              for layer, vh in heads]
    return {"heads": heads, "states": torch.stack(states)}


def restore(model, snap: dict, batch: int = 1):
    """A fresh DynamicCache holding only the snapshot's heads (every other state at 0)."""
    from transformers import DynamicCache

    cfg = model.config.get_text_config(decoder=True)
    cache = DynamicCache(config=model.config)
    dev = next(model.parameters()).device
    nk, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv, kern = cfg.linear_key_head_dim, cfg.linear_value_head_dim, cfg.linear_conv_kernel_dim
    conv_dim = 2 * nk * dk + nv * dv
    by_layer: dict[int, list] = {}
    for (layer, vh), s in zip(snap["heads"], snap["states"]):
        by_layer.setdefault(layer, []).append((vh, s))
    for layer, items in by_layer.items():
        L = cache.layers[layer]
        rec = torch.zeros(batch, nv, dk, dv, dtype=items[0][1].dtype, device=dev)
        for vh, s in items:
            rec[:, vh] = s.to(dev)
        L.recurrent_states[0] = rec
        L.conv_states[0] = torch.zeros(batch, conv_dim, kern, dtype=model.dtype, device=dev)
        L.conv_kernel_size[0] = kern
        L.is_conv_states_initialized[0] = True
        L.is_recurrent_states_initialized[0] = True
        L.has_previous_state[0] = True
        L.dtype, L.device = model.dtype, dev
    return cache


def save(snap: dict, path) -> None:
    torch.save(snap, path)


def load(path) -> dict:
    return torch.load(path, weights_only=True)


# --- the IR view of the same thing (tests) -----------------------------------------------


def ir_snapshot(c, carry: dict) -> dict:
    """The persistent heads' states from an IR evaluation's ``carry`` (store_check)."""
    from chipformer.q35.store_q35 import is_persistent

    return {k: v for k, v in carry.items()
            if k[1] == "S" and is_persistent(c.nodes[k[0]].name)}


# --- the disk image ----------------------------------------------------------------------


def disk_image(sm: S.StoreMachine) -> list[tuple]:
    """The calls that rebuild ``sm``'s live files, in directory order: (op, name, text)."""
    calls = []
    for k in sorted(sm.dev.dir):
        e = sm.dev.dir[k]
        if not e.alive:
            continue
        name = list(e.name[:min(e.length, S.NMAX)])
        g, first = sm.dev.dir_h[k], True
        while True:
            n, nxt, has = sm.dev.meta.get(g, (0, 0, 0))
            body = [sm.dev.content.get((g, i), 0) for i in range(n)]
            # the stored chunk ends with the template's newline (its call adds it back)
            if body and body[-1] == S.Q_NL:
                body = body[:-1]
            calls.append(("put" if first else "append", name[:-1] if name and name[-1] == S.Q_NL
                          else name, body))
            first = False
            if not has:
                break
            g = nxt
    return calls


def replay(sm: S.StoreMachine, calls: list[tuple]) -> None:
    """Prefill a disk image: the calls and the circuit's answers become part of the trace."""
    for op, name, text in calls:
        sm.call(op, name, text)
