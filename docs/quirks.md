# Quirk profile

We implement the CHIP-48 / "modern" behaviour that most 1990s games, Pong included, were
written for:

| Quirk | Our behaviour | COSMAC original |
|---|---|---|
| vF reset (8XY1/2/3) | VF untouched | VF ← 0 |
| Memory (FX55/FX65) | I unchanged | I += X+1 |
| Display wait | none | wait for vblank |
| Clipping | sprites clip at the edges (start position wraps) | same |
| Shifting (8XY6/8XYE) | shift VX in place | VX ← VY shifted |
| Jumping (BNNN) | NNN + V0 | same |
| FX0A | completes while any key is held, lowest key wins | waits for release |
| Stack | SP wraps modulo 16 | – |

The Timendus quirks ROM (`5-quirks.ch8`, CHIP-8 profile) correctly flags vF reset, memory,
display wait and shifting as deviating from COSMAC. That is expected.
