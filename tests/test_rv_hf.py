"""The RV32 core in stock Qwen3_5ForCausalLM modules (bf16, the torch reference kernels): with the
state snapshot injected into the HF cache, the residual stream equals the IR evaluator's on every
IR dim at every position, for a chunked prefill and for token-by-token decoding."""

import pytest

from chipformer.rv import core, hf, ref
from chipformer.rv.programs import program


@pytest.fixture(scope="module")
def circ():
    return core.build()


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_signext_in_hf(circ, mode):
    p = program("signext")
    core.snapshot(circ, ref.Machine.from_program(p))
    toks = ref.Machine.from_program(p).run()
    model, lay, ids = hf.build_model(circ, toks)
    assert hf.check(circ, model, lay, ids, toks, mode) == len(toks)


def test_call_prefill_in_hf(circ):
    p = program("call")
    core.snapshot(circ, ref.Machine.from_program(p))
    toks = ref.Machine.from_program(p).run()
    model, lay, ids = hf.build_model(circ, toks)
    assert hf.check(circ, model, lay, ids, toks, "prefill") == len(toks)
