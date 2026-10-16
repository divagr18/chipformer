"""CHIP-8 inside a stock Qwen3.5 checkpoint: bf16-exact circuits built from Gated DeltaNet and SwiGLU."""

try:  # the chip is exact only without bf16 split-K reductions, which cuBLAS may pick for some shapes
    import torch

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
except ImportError:
    pass
