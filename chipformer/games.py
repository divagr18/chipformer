"""Game library: fetch ROMs and metadata, build the manifest, validate on the engine.

    uv run python -m chipformer.games fetch      # download sources into games/sources/
    uv run python -m chipformer.games build      # games/manifest.json + games/roms/
    uv run python -m chipformer.games validate   # run every game on the engine, verify traces

Sources:
  - CHIP-8 Archive (github.com/JohnEarnest/chip8Archive): CC0 games with Octo quirk flags
  - classic games (github.com/kripod/chip8-roms, the Revival Studios pack: "can be freely
    distributed in its original form")
  - CHIP-8 database (github.com/chip-8/chip-8-database): title, authors, platform, tickrate
    and keys for known ROMs, keyed by SHA-1
"""

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / "games"
SOURCES = GAMES / "sources"

ARCHIVE = "https://raw.githubusercontent.com/JohnEarnest/chip8Archive/master"
DATABASE = "https://raw.githubusercontent.com/chip-8/chip-8-database/master/database"
CLASSICS_API = "https://api.github.com/repos/kripod/chip8-roms/contents/games"
CLASSICS_RAW = "https://raw.githubusercontent.com/kripod/chip8-roms/master/games"

LICENSES = {
    "archive": "CC0",
    "classics": "freely distributable (Revival Studios CHIP-8 pack)",
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "chipformer"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch() -> None:
    """Download every source into games/sources/ (data only; nothing is executed)."""
    (SOURCES / "database").mkdir(parents=True, exist_ok=True)
    for name in ["programs.json", "sha1-hashes.json", "platforms.json", "quirks.json"]:
        (SOURCES / "database" / name).write_bytes(_get(f"{DATABASE}/{name}"))

    arch = SOURCES / "archive"
    (arch / "roms").mkdir(parents=True, exist_ok=True)
    for name in ["programs.json", "authors.json"]:
        (arch / name).write_bytes(_get(f"{ARCHIVE}/{name}"))
    programs = json.loads((arch / "programs.json").read_text(encoding="utf-8"))
    for key, p in programs.items():
        if p.get("platform", "chip8") == "chip8":
            (arch / "roms" / f"{key}.ch8").write_bytes(_get(f"{ARCHIVE}/roms/{key}.ch8"))

    cls = SOURCES / "classics"
    cls.mkdir(parents=True, exist_ok=True)
    listing = json.loads(_get(CLASSICS_API))
    for entry in listing:
        if entry["name"].lower().endswith(".ch8"):
            url = f"{CLASSICS_RAW}/{urllib.parse.quote(entry['name'])}"
            (cls / entry["name"]).write_bytes(_get(url))
    print(f"archive: {len(list((arch / 'roms').glob('*.ch8')))} ROMs, "
          f"classics: {len(list(cls.glob('*.ch8')))} ROMs, database: "
          f"{len(json.loads((SOURCES / 'database' / 'programs.json').read_text(encoding='utf-8')))} programs")


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# CHIP-8 database platforms we can run (lores CHIP-8 instruction set), best first
RUNNABLE_PLATFORMS = ["modernChip8", "originalChip8", "hybridVIP", "chip48", "superchip1",
                      "superchip"]
# Octo's keyboard convention: WASD = keypad 5/7/8/9, E = keypad 6
OCTO_KEYS = {"up": 5, "left": 7, "down": 8, "right": 9, "a": 6}

# Games shown in the demo, in display order, with key roles where the database has none
# (from each game's own documentation) and a one-line control hint. Role names follow the
# CHIP-8 database: up/down/left/right/a/b, player2Up/player2Down.
FEATURED = {
    "tetris": ({}, "Left/Right move, Down drops faster, Space rotates"),
    "space-invaders": ({}, "Left/Right move, Space fires (press Space to start)"),
    "brix": ({"left": 4, "right": 6}, "Left/Right move the paddle"),
    "blinky": ({"up": 6, "down": 3, "left": 7, "right": 8}, "Arrow keys steer through the maze"),
    "pong": ({"up": 1, "down": 4, "player2Up": 12, "player2Down": 13},
             "Left paddle A/D (or W/S), right paddle Left/Right (or Up/Down)"),
    "super-pong": ({}, "WASD moves the paddles, Space (E) resets"),
    "ufo": ({"left": 4, "a": 5, "right": 6}, "Left/Space/Right fire left, up, right"),
    "missile-command": ({}, "Space fires"),
    "br8kout": ({}, "Left/Right move the paddle"),
    "cave-explorer": ({}, "Arrow keys move"),
    "outlaw": ({}, "Arrow keys move, Space shoots"),
    "flight-runner": ({}, "Up/Down steer"),
    "down8": ({}, "Left/Right move"),
    "dino-run": ({}, "Up or Space jumps"),
}

# Per-game play tuning. "ipf" overrides the platform's default speed. "tap" lists key roles
# delivered as one short pulse per physical press: games that act on every pass of their loop
# while a key is held (Tetris rotates once per pass) otherwise repeat a press several times.
# Measured on Tetris: at 30 instructions/frame a 150 ms press rotates twice; at 15 it rotates
# once even for a 200 ms hold.
#
# "game_over" lets the cabinet end games that never end themselves: Fran Dachille's Tetris
# (1991, HP48) has no game-over detection (its manual says "ENTER - restart"), so once the
# well is full it spawns pieces into the stack forever. The host declares the game over
# when any pixel in the given screen rows/columns stays lit for `seconds`. For Tetris, pieces
# spawn in rows 2-5; rows 6-7 are only crossed by falling pieces (at most 2.4 s in simulated
# play) until the stack reaches them. This is the host reading the screen, not the model.
TUNING = {
    "tetris": {"ipf": 15, "tap": ["a"],
               "game_over": {"rows": [6, 7], "cols": [27, 35], "seconds": 5,
                             "message": "The well is full"}},
}
TAP_FRAMES = 3  # length of a tap pulse, in 60 Hz frames

# Ids for classics whose titles collide: the well-known original gets the plain name.
CLASSIC_IDS = {
    "Pong [Paul Vervalin, 1990].ch8": "pong",
    "Pong (1 player).ch8": "pong-1-player",
}


def _slug(title: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in title.lower()).strip("-").replace("--", "-")


def _short(text: str | None, limit: int = 280) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _db_quirks(platform: dict) -> tuple[list[str], list[str]]:
    """Map a CHIP-8 database platform's quirks onto our flags; returns (flags, notes)."""
    q = platform["quirks"]
    flags, notes = [], []
    if q.get("shift"):
        flags.append("shift")
    if q.get("memoryLeaveIUnchanged") or q.get("memoryIncrementByX"):
        flags.append("loadstore")
        if q.get("memoryIncrementByX"):
            notes.append("platform increments I by X after FX55/FX65; run with I unchanged")
    if not q.get("wrap"):
        flags.append("clip")
    if q.get("jump"):
        flags.append("jump")
    if q.get("logic"):
        flags.append("logic")
    if q.get("vblank"):
        notes.append("platform waits for vblank on draw; not modelled")
    return flags, notes


def build() -> list[dict]:
    """Assemble games/manifest.json and games/roms/ from the downloaded sources."""
    db = json.loads((SOURCES / "database" / "programs.json").read_text(encoding="utf-8"))
    hashes = json.loads((SOURCES / "database" / "sha1-hashes.json").read_text(encoding="utf-8"))
    platforms = {p["id"]: p for p in json.loads(
        (SOURCES / "database" / "platforms.json").read_text(encoding="utf-8"))}
    archive = json.loads((SOURCES / "archive" / "programs.json").read_text(encoding="utf-8"))
    authors = json.loads((SOURCES / "archive" / "authors.json").read_text(encoding="utf-8"))
    old = {g["id"]: g for g in load_manifest()} if (GAMES / "manifest.json").exists() else {}

    games: list[dict] = []
    seen: set[str] = set()

    def add(entry: dict, rom: bytes, gid: str | None = None) -> None:
        base = gid = gid or _slug(entry["title"])
        n = 2
        while gid in seen:
            gid, n = f"{base}-{n}", n + 1
        seen.add(gid)
        entry = {"id": gid, **entry, "file": f"roms/{gid}.ch8", "sha1": sha1(rom)}
        for keep in ("status", "result"):  # validation results survive rebuilds
            if keep in old.get(gid, {}):
                entry[keep] = old[gid][keep]
        if gid in FEATURED:
            keys, hint = FEATURED[gid]
            entry["keys"] = {**entry["keys"], **keys}
            entry["controls"] = hint
            entry["featured"] = list(FEATURED).index(gid)
        tuning = TUNING.get(gid, {})
        entry["ipf"] = tuning.get("ipf", entry["ipf"])
        entry["tap"] = tuning.get("tap", [])
        if "game_over" in tuning:
            entry["game_over"] = tuning["game_over"]
        (GAMES / "roms").mkdir(parents=True, exist_ok=True)
        (GAMES / "roms" / f"{gid}.ch8").write_bytes(rom)
        games.append(entry)

    def db_entry(rom: bytes):
        i = hashes.get(sha1(rom))
        if i is None:
            return None, None
        prog = db[int(i)]
        return prog, prog["roms"][sha1(rom)]

    for key, p in archive.items():
        if p.get("platform", "chip8") != "chip8":
            continue
        rom = (SOURCES / "archive" / "roms" / f"{key}.ch8").read_bytes()
        opts = p.get("options", {})
        flags = [name for name, opt in [("shift", "shiftQuirks"), ("loadstore", "loadStoreQuirks"),
                                        ("clip", "clipQuirks"), ("logic", "logicQuirks"),
                                        ("jump", "jumpQuirks"), ("vforder", "vfOrderQuirks")]
                 if opts.get(opt) in (True, 1, "true")]
        _, dbrom = db_entry(rom)
        add({
            "title": p["title"],
            "authors": [authors.get(a, {}).get("name", a) if isinstance(authors.get(a), dict) else a
                        for a in p.get("authors", [])],
            "year": (p.get("release") or "")[:4] or None,
            "description": _short(p.get("desc")),
            "source": f"{ARCHIVE.replace('raw.githubusercontent.com', 'github.com')}/roms/{key}.ch8",
            "license": LICENSES["archive"],
            "platform": "octo",
            "quirks": flags,
            "ipf": int(opts.get("tickrate", 20)),
            "keys": (dbrom or {}).get("keys") or OCTO_KEYS,
            "notes": [],
        }, rom)

    for path in sorted((SOURCES / "classics").glob("*.ch8")):
        if "(alt)" in path.name:
            continue
        rom = path.read_bytes()
        prog, dbrom = db_entry(rom)
        name = path.stem.split(" [")[0].split(" (")[0]
        plats = [pl for pl in (dbrom or {}).get("platforms", []) if pl in RUNNABLE_PLATFORMS]
        plat = plats[0] if plats else "originalChip8"
        flags, notes = _db_quirks(platforms[plat])
        if not plats:
            notes.append("not in the CHIP-8 database as a lores CHIP-8 game; COSMAC quirks assumed")
        add({
            "title": prog["title"] if prog else name,
            "authors": (prog or {}).get("authors", []),
            "year": (prog or {}).get("release"),
            "description": _short((dbrom or {}).get("description") or (prog or {}).get("description")),
            "source": f"https://github.com/kripod/chip8-roms/blob/master/games/{urllib.parse.quote(path.name)}",
            "license": LICENSES["classics"],
            "platform": plat,
            "quirks": flags,
            "ipf": int((dbrom or {}).get("tickrate") or platforms[plat].get("defaultTickrate", 15)),
            "keys": (dbrom or {}).get("keys") or {},
            "notes": notes,
        }, rom, CLASSIC_IDS.get(path.name))

    missing = set(FEATURED) - {g["id"] for g in games}
    if missing:
        raise ValueError(f"featured games not found: {sorted(missing)}")
    (GAMES / "manifest.json").write_text(json.dumps({"games": games}, indent=1), encoding="utf-8")
    write_attribution(games)
    print(f"manifest: {len(games)} games, {len(FEATURED)} featured")
    return games


def write_attribution(games: list[dict], path: Path | None = None) -> None:
    lines = [
        "# Game credits",
        "",
        "ROMs in `roms/`. CHIP-8 Archive games are CC0. The classics come from the Revival",
        "Studios CHIP-8 pack, which \"can be freely distributed in its original form\"; each",
        "game belongs to its author.",
        "",
        "| Game | Authors | Year | License | Source |",
        "|---|---|---|---|---|",
    ]
    for g in sorted(games, key=lambda g: g["title"].lower()):
        lines.append(f"| {g['title']} | {', '.join(g['authors']) or 'unknown'} | {g.get('year') or ''}"
                     f" | {g['license']} | [link]({g['source']}) |")
    (path or GAMES / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_manifest() -> list[dict]:
    return json.loads((GAMES / "manifest.json").read_text(encoding="utf-8"))["games"]


# opcodes outside lores CHIP-8 (SCHIP / XO-CHIP); executing one means the game won't run right
def foreign_opcode(op: int) -> str | None:
    hi, n, nn = op >> 12, op & 0xF, op & 0xFF
    if hi == 0 and op not in (0x00E0, 0x00EE):
        return f"{op:04X} (machine code / SCHIP)"
    if hi == 0xD and n == 0:
        return "DXY0 (SCHIP 16x16 sprite)"
    if hi == 0x5 and n != 0:
        return f"5XY{n:X} (XO-CHIP)"
    if hi == 0xF and nn in (0x00, 0x01, 0x02, 0x30, 0x3A, 0x75, 0x85):
        return f"FX{nn:02X} (SCHIP/XO-CHIP)"
    return None


def validate(ids: list[str] | None = None, instr: int = 20000) -> None:
    """Reference profile + free-running engine run + trace verification per game."""
    import os
    import random
    import subprocess

    import numpy as np

    from chipformer.check import verify_trace
    from chipformer.ref.emu import fetch as fetch_op
    from chipformer.trace import Machine

    exe = ROOT / "engine" / "target" / "release" / ("engine.exe" if os.name == "nt" else "engine")
    games = load_manifest()
    tmp = Path(os.environ.get("TEMP", "/tmp")) / "chipformer-validate"
    tmp.mkdir(exist_ok=True)
    shots = GAMES / "shots"
    shots.mkdir(exist_ok=True)
    for g in games:
        if ids and g["id"] not in ids:
            continue
        rom = (GAMES / g["file"]).read_bytes()
        quirks = ",".join(g["quirks"])
        # 1. reference: executed opcodes, tokens per instruction
        rng = random.Random(0)
        m = Machine(rom, tuple(g["quirks"]))
        foreign: dict[str, int] = {}
        for i in range(instr):
            if i % g["ipf"] == 0:
                m.tick()
                if rng.random() < 0.2:
                    m.set_key(rng.randrange(16), rng.random() < 0.5)
            f = foreign_opcode(fetch_op(m.emu.cells))
            if f:
                foreign[f] = foreign.get(f, 0) + 1
            m.step()
        tok_per_instr = (len(m.tokens) - m.generated.index(True)) / instr
        need = tok_per_instr * g["ipf"] * 60
        # 2. the transformer, free-running, verified record by record
        dump = tmp / f"{g['id']}.u32"
        r = subprocess.run([str(exe), "run", str(ROOT / "model"), str(GAMES / g["file"]),
                            "--instr", str(instr), "--ipf", str(g["ipf"]), "--quirks", quirks,
                            "--dump", str(dump)], check=False, capture_output=True, text=True)
        if r.returncode != 0:
            verified, err = 0, "engine error: " + r.stderr.strip().splitlines()[-1]
        else:
            toks = np.fromfile(dump, dtype="<u4").tolist()
            verified, err = verify_trace(toks)
            save_screenshot(toks, shots / f"{g['id']}.png")
        status = "unsupported" if foreign else ("ok" if err is None else "mismatch")
        g["status"] = status
        g["result"] = {"verified_records": verified, "error": err, "foreign_opcodes": foreign,
                       "tokens_per_instruction": round(tok_per_instr, 2),
                       "realtime_tok_per_s": round(need)}
        print(f"{g['id']:32} {status:11} {verified:7,} records  {need:8,.0f} tok/s needed"
              f"  {', '.join(foreign) if foreign else ''}{err or ''}", flush=True)
    (GAMES / "manifest.json").write_text(json.dumps({"games": games}, indent=1), encoding="utf-8")


def save_screenshot(tokens: list[int], path: Path, scale: int = 6) -> None:
    """The final screen of a (verified) trace, rendered from the model's own writes."""
    from PIL import Image

    from chipformer import spec
    from chipformer.ref.emu import apply_writes, display_pixels
    from chipformer.trace import parse

    cells = [0] * spec.NUM_CELLS
    for rec in parse(tokens, allow_reboot=True):
        if rec.kind == "boot":
            cells = [0] * spec.NUM_CELLS
        apply_writes(cells, rec.writes)
    img = Image.new("RGB", (spec.DISPLAY_W, spec.DISPLAY_H))
    img.putdata([(221, 245, 158) if p else (25, 34, 31) for row in display_pixels(cells) for p in row])
    img.resize((spec.DISPLAY_W * scale, spec.DISPLAY_H * scale), Image.NEAREST).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fetch", "build", "validate"])
    ap.add_argument("ids", nargs="*", help="validate only these game ids")
    ap.add_argument("--instr", type=int, default=20000)
    args = ap.parse_args()
    if args.command == "fetch":
        fetch()
    elif args.command == "build":
        build()
    else:
        validate(args.ids or None, args.instr)


if __name__ == "__main__":
    main()
