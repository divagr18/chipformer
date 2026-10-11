"""Featured games: the Rust engine plays each one free-running, verified record by record.

Needs `cargo build --release` in engine/ and `uv run python -m chipformer.compile`.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from chipformer.check import verify_trace

ROOT = Path(__file__).resolve().parent.parent
RELEASE = ROOT / "engine" / "target" / "release"
EXE = RELEASE / "engine.exe" if (RELEASE / "engine.exe").exists() else RELEASE / "engine"
MANIFEST = ROOT / "games" / "manifest.json"
FEATURED = sorted((g for g in json.loads(MANIFEST.read_text(encoding="utf-8"))["games"]
                   if "featured" in g), key=lambda g: g["featured"]) if MANIFEST.exists() else []

pytestmark = pytest.mark.skipif(not (EXE.exists() and (ROOT / "model").exists()),
                                reason="engine or model not built")


@pytest.mark.parametrize("game", FEATURED, ids=lambda g: g["id"])
def test_featured_game(tmp_path, game):
    subprocess.run([str(EXE), "run", str(ROOT / "model"), str(ROOT / "games" / game["file"]),
                    "--instr", "5000", "--ipf", str(game["ipf"]), "--quirks", ",".join(game["quirks"]),
                    "--seed", "3", "--dump", str(tmp_path / "t.u32")], check=True, capture_output=True)
    n, err = verify_trace(np.fromfile(tmp_path / "t.u32", dtype="<u4").tolist())
    assert err is None, err
    assert n > 5000
