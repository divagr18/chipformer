"""Random RV32IM programs (chipformer/rv/fuzz.py): fault-free in the reference, exact in the IR
(teacher-forced), and generated free-running by the executor (CPU here; the 10^6-instruction
runs are on a GPU: python -m chipformer.rv.fuzz)."""

import pytest

from chipformer.q35 import ir
from chipformer.rv import core, fuzz, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble


@pytest.fixture(scope="module")
def circ():
    return core.build()


@pytest.mark.parametrize("seed", range(3))
def test_fuzz_ir(circ, seed):
    prog = assemble(fuzz.gen_program(1000 + seed, block=40, iters=3))
    m = ref.Machine.from_program(prog)
    core.snapshot(circ, m)
    toks = ref.Machine.from_program(prog).run()
    assert m.fault is None
    mat = ir.evaluate(circ, toks, check=True, kernel=seed == 0)
    pred = core.predictions(circ, mat)
    bad = [i for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
    assert not bad, (bad[0], S.name(toks[bad[0] + 1]))


def test_fuzz_free_running_cpu(circ):
    progs = [assemble(fuzz.gen_program(2000 + s, block=30, iters=2)) for s in range(2)]
    res = fuzz.free_run(circ, progs, batch=2, device="cpu", compile_=False)
    assert res["instructions"] > 100
