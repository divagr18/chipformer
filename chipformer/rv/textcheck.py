"""Text fidelity of a merged rv1 checkpoint against its base Qwen3.5.

    python -m chipformer.rv.textcheck BASE MERGED [--out dist/rv-text.json]

On report.PROMPTS: base greedy continuations (32 tokens), then teacher-forced next-token
distributions over Qwen's vocabulary on prompt + continuation. KL(base || merged) is compared
with the noise floor KL(base sdpa || base eager attention). Also: the merged model's own greedy
continuations (plain HF generate) and the merged tokenizer's encodings against the base's.
"""

import argparse
import gc
import json

import torch

from chipformer.q35.report import PROMPTS, encode

N_NEW = 32
SAMPLES = ["Hello, world! 123456789 * 987654321 = ?", "  indented\n\tcode()  # comment",
           "naïve café — 東京 🙂", "<tool_call>\n<function=calc>\n<parameter=expression>\n1+2\n"
           "</parameter>\n</function>\n</tool_call>"]


@torch.no_grad()
def logprobs(model, seqs, nq, dev):
    out = []
    for g in seqs:
        lg = model(input_ids=torch.tensor([g], device=dev)).logits[0, :, :nq].float()
        out.append(torch.log_softmax(lg, -1).half().cpu())
    return out


def kl(a, b):
    return [float((p.float().exp() * (p.float() - q.float())).sum(-1).mean()) for p, q in zip(a, b)]


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoTokenizer, Qwen3_5ForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("merged")
    ap.add_argument("--out", default="dist/rv-text.json")
    args = ap.parse_args()
    dev = "cuda"
    tb = AutoTokenizer.from_pretrained(args.base)
    tm = AutoTokenizer.from_pretrained(args.merged)
    tok_same = [tb(s).input_ids == tm(s).input_ids for s in SAMPLES]
    prompts = [encode(tb, k, t) for k, t in PROMPTS]
    nq = len(tb)

    base = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16).to(dev)
    seqs = []
    with torch.no_grad():
        for ids in prompts:
            g = base.generate(torch.tensor([ids], device=dev), max_new_tokens=N_NEW,
                              do_sample=False)
            seqs.append(g[0].tolist())
    lb = logprobs(base, seqs, nq, dev)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    eager = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16,
                                                        attn_implementation="eager").to(dev)
    le = logprobs(eager, seqs, nq, dev)
    del eager
    gc.collect()
    torch.cuda.empty_cache()
    merged = Qwen3_5ForCausalLM.from_pretrained(args.merged, dtype=torch.bfloat16).to(dev)
    lm = logprobs(merged, seqs, nq, dev)
    same_greedy = []
    with torch.no_grad():
        for ids, g in zip(prompts, seqs):
            h = merged.generate(torch.tensor([ids], device=dev), max_new_tokens=N_NEW,
                                do_sample=False)
            same_greedy.append(h[0].tolist() == g)
    k_m, k_e = kl(lb, lm), kl(lb, le)
    top1 = [float((p.argmax(-1) == q.argmax(-1)).float().mean()) for p, q in zip(lb, lm)]
    res = {"kl_merged_mean": sum(k_m) / len(k_m), "kl_noise_floor_mean": sum(k_e) / len(k_e),
           "kl_merged": k_m, "kl_noise_floor": k_e, "top1_agree": top1,
           "greedy_identical": same_greedy, "tokenizer_identical": tok_same}
    print(json.dumps({k: v for k, v in res.items() if k.endswith("mean")
                      or k in ("greedy_identical", "tokenizer_identical")}, indent=1))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
