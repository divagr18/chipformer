"""The Rust engine vs the circuit, and its free-running traces vs the reference.

Needs `cargo build --release` in engine/ and a compiled model in model/
(`uv run python -m chipformer.compile`).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from chipformer.check import run_machine, verify_trace
from chipformer.chip8 import build
from chipformer.dsl.batch import evaluate_batch, output_batch
from chipformer.fuzz import random_program, random_quirks

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "engine" / "target" / "release" / ("engine.exe" if (ROOT / "engine" / "target" / "release" / "engine.exe").exists() else "engine")
MODEL = ROOT / "model"

pytestmark = pytest.mark.skipif(not (EXE.exists() and (MODEL / "model.safetensors").exists()),
                                reason="engine or model not built")


def test_predict_matches_circuit(tmp_path):
    m = run_machine(random_program(21), 800, seed=21, key_rate=0.5)
    np.array(m.tokens, dtype="<u4").tofile(tmp_path / "t.u32")
    subprocess.run([str(EXE), "predict", str(MODEL), str(tmp_path / "t.u32"), str(tmp_path / "p.u32")],
                   check=True, capture_output=True)
    pred = np.fromfile(tmp_path / "p.u32", dtype="<u4").astype(float)
    g = build()
    assert np.array_equal(pred, output_batch(g, "next", evaluate_batch(g, m.tokens)))


@pytest.mark.parametrize("rom", ["fuzz", "pong"])
def test_free_running_with_rollover(tmp_path, rom):
    path, quirks = ROOT / "roms" / "pong.ch8", "chip48"
    if rom == "fuzz":
        path, quirks = tmp_path / "fuzz.ch8", ",".join(random_quirks(5))
        path.write_bytes(random_program(5))
    elif not path.exists():
        pytest.skip("roms/pong.ch8 not provided")
    subprocess.run([str(EXE), "run", str(MODEL), str(path), "--instr", "4000", "--seed", "9",
                    "--quirks", quirks,
                    "--rollover", "8000", "--dump", str(tmp_path / "run.u32")],
                   check=True, capture_output=True)
    n, err = verify_trace(np.fromfile(tmp_path / "run.u32", dtype="<u4").tolist())
    assert err is None, err
    assert n > 4000
