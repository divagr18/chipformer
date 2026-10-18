"""Export a merged checkpoint as a text-only stock ``Qwen3_5ForCausalLM`` (no vision tower).

Qwen3.5 ships as ``Qwen3_5ForConditionalGeneration`` (text plus a vision tower); the computer
lives in the text model, so a text-only export drops the vision weights (0.34B params at 4B)
and loads with ``AutoModelForCausalLM`` / ``Qwen3_5ForCausalLM``. Text in, text out is
unchanged: the language model's weights are copied as they are.

    python -m chipformer.q35.textonly MERGED OUT
"""

import json
import shutil
import sys
from pathlib import Path

PREFIX = "model.language_model."


def export(src, dst) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    tc = dict(cfg["text_config"])
    tc["architectures"] = ["Qwen3_5ForCausalLM"]
    tc["model_type"] = "qwen3_5_text"
    tc["tie_word_embeddings"] = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", True))
    for k in ("torch_dtype", "dtype", "transformers_version"):
        if k in cfg and k not in tc:
            tc[k] = cfg[k]
    (dst / "config.json").write_text(json.dumps(tc, indent=2))
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    keep = {k: f for k, f in index.items() if k.startswith(PREFIX)}
    files = sorted(set(keep.values()))
    weight_map, total = {}, 0
    for i, f in enumerate(files):
        out = {}
        with safe_open(src / f, "pt") as s:
            for k in s.keys():  # noqa: SIM118 - a safetensors handle, not a dict
                if k in keep:
                    out["model." + k[len(PREFIX):]] = s.get_tensor(k)
        name = f"model-{i + 1:05d}-of-{len(files):05d}.safetensors"
        save_file(out, str(dst / name), metadata={"format": "pt"})
        for k, t in out.items():
            weight_map[k] = name
            total += t.numel() * t.element_size()
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    for p in src.iterdir():  # tokenizer, generation config, chat template, chipformer.json
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (
                "config.json", "model.safetensors.index.json", "preprocessor_config.json",
                "video_preprocessor_config.json"):
            shutil.copy(p, dst / p.name)


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
