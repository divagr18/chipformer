# Quirks

CHIP-8 interpreters disagree on a few behaviours, and games depend on them. In chipformer the
quirks are **machine state, not separate models**: six flag cells (`spec.QUIRKS`,
`0x1017–0x101C`) are set in the boot prompt and read by the network at every instruction. One
fixed set of weights runs every profile.

The flags mean the same as Octo's flags of the same name, so CHIP-8 Archive metadata maps over
directly. All flags off is Octo's default.

| Cell | Flag | On | Off |
|---|---|---|---|
| `0x1017` | `shift` | 8XY6/8XYE shift VX in place | VX ← VY shifted |
| `0x1018` | `loadstore` | FX55/FX65 leave I unchanged | I += X + 1 |
| `0x1019` | `clip` | sprites clip at the screen edges | sprites wrap around |
| `0x101A` | `logic` | 8XY1/2/3 reset VF to 0 | VF untouched |
| `0x101B` | `jump` | BXNN jumps to XNN + VX | BNNN jumps to NNN + V0 |
| `0x101C` | `vforder` | 8XY4–E write VX after VF (the result wins when X = F) | VF written last (the flag wins) |

Named profiles (`spec.PROFILES`, `--quirks <name>`, or a comma-separated flag list):

| Profile | Flags | For |
|---|---|---|
| `octo` | none | Octo / CHIP-8 Archive defaults |
| `chip48` | shift, loadstore, clip | HP48-era games, e.g. Pong (1990) |
| `cosmac` | logic, clip | the original COSMAC VIP interpreter |

Fixed behaviour in every profile:
- sprites start at (VX % 64, VY % 32)
- FX0A completes while any key is held (lowest key wins)
- the stack pointer wraps modulo 16
- there is no display wait. Octo declares a `vBlankQuirks` flag but never reads it.

The Timendus quirks ROM (`5-quirks.ch8`) reports whichever profile is loaded.
