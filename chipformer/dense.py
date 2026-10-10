"""A plain decoder-only transformer forward pass. Knows nothing about CHIP-8.

Standard pieces only: token embedding plus position embedding, causal
multi-head softmax attention with residual connections, ReGLU feed-forward
layers, and a linear unembedding. Everything is fp64.
"""

import numpy as np


def softmax_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, beta: float) -> np.ndarray:
    n = q.shape[0]
    scores = q @ k.T
    scores[np.triu_indices(n, 1)] = -np.inf  # causal mask
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(beta * scores)
    p /= p.sum(axis=1, keepdims=True)
    return p @ v


def forward(w: dict[str, np.ndarray], config: dict, tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (final residual stream, logits), both of shape (n_positions, ...)."""
    t = np.arange(len(tokens), dtype=np.float64)
    r = w["embed.token"][np.asarray(tokens)] + np.stack([t, t * t], 1) @ w["embed.pos_poly"].T
    for layer, spec in enumerate(config["layers"]):
        attn = np.zeros_like(r)
        for h in range(spec["heads"]):
            p = f"layers.{layer}.attn.{h}"
            out = softmax_attention(r @ w[f"{p}.wq"].T, r @ w[f"{p}.wk"].T, r @ w[f"{p}.wv"].T,
                                    config["beta"])
            attn += out @ w[f"{p}.wo"].T
        r = r + attn
        p = f"layers.{layer}.ffn"
        hidden = np.maximum(r @ w[f"{p}.w_gate"].T, 0.0) * (r @ w[f"{p}.w_up"].T)
        r = r + hidden @ w[f"{p}.w_down"].T
    return r, r @ w["unembed"].T
