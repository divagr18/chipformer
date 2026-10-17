# q35: more built-in skills (plan)

Status: design and estimates only, written against v15 (5efa9e6) with v16 (signed division, `mod`,
`gcd`) in progress. Every token count here is either measured on the reference trace
(`BridgeMachine`, v15 BIOS) or derived from those measurements with the per-record costs below.
Circuit costs are estimates, using what v14 division actually cost as the yardstick.

## 0. Starting point (measured)

**What the computer does today.** It does exact integer add, sub, mul and div:
- Operands are up to 24 digits each (`MAX_DIGITS`, `MAX_ARG`).
- A print holds at most 58 digits (`MAX_PRINT`), and a record holds at most 64 writes (`JMAX`).
- div prints the quotient, its terminating decimal, or 10 decimals then "..." (`DEC = 10`).
- Inline triggers are "A op B =" with exactly one operator. A "." breaks the window, so
  "2.5 × 3 =" is left alone.

**Trace cost per call.** The v15 reference counts the tokens after the CALL record, including the
print. The median of 20 random cases:

| call | tokens | | call | tokens |
|---|---|---|---|---|
| add 4+3 digits | 30 | | mul 8x8 | 206 |
| add 12+12 | 55 | | mul 12x12 | 394 |
| sub 8-5 | 42 | | div 4/4 | 526 |
| mul 2x2 | 44 | | div 6/2 | 570 |
| mul 4x3 | 68 | | div 12/4 | 751 |

- **Per-record costs** (used for every estimate below):
  - A plain CHIP-8 instruction costs about 5 tokens.
  - FX55/FX65 over n registers, or a macro record with n writes, costs about 2n + 3.
  - The CALL record costs about 2 tokens per argument digit, plus about 10.
- **Division overhead.** Division spends about 58 plain records (about 300 tokens) in BIOS setup
  and its strip/print loop. That loop, not BDIV, is division's largest overhead.
- **Speed.** Every estimate below is in tokens. At an assumed 50 tok/s per stream for the merged
  4B, 1k trace tokens is about 20 s. With `tc`, trace length affects only speed and context
  budget, never Qwen's text.

**Circuit occupancy (v15, 16 layers = 32 sublayers).**
- 104 of 256 head slots and 24,463 of 57,344 FFN units at intermediate 3584, with 3,623 dims.
- Full MLP sublayers: 2, 12 and 14 (3584 each). Sublayers 16 and 20 are about 89% full.
  Sublayers 22-32 are nearly empty, but they are deep.
- Full attention sublayers: 3, 5, 21 and 23 (16 heads each).
- At the 4B, intermediate is 9216, so MLP room is no issue there. The binding costs are:
  - **Hidden width.** About 1,580 chip dims sit on top of Qwen's 2,560, giving hidden 4224.
    v14 division alone took it from 3840 to 4224.
  - **Depth.** BDIV only fit after moving its quotient digit to the record's first write.
- Macro opcodes: 5XY1-5XY8 are used (5XY0 is SE), so 9, A, B, C, D, E and F are free. The Y
  field gives variants, as 5X13 BMUL0 does.
- ROM: the BIOS occupies 0x200-0x2AE (174 bytes).
  - Free space: about 850 bytes at 0x2AE-0x5FF, and about 1.8 KB above 0x900.
  - Data: A, B, LEN and R sit at 0x600-0x6BE, and W runs from 0x6BF to about 0x6E2.
  - Macro jump targets (T = 0x200 + 2·XY) must sit below 0x400.

**What Qwen actually writes.**
- Calculations Qwen wrote before an "=" (classified by `scratchpad/skills/qwenwrites.py`):

| GSM-Hard, 200 problems | base + nudge | v14 inline + tc |
|---|---|---|
| "... =" calculations | 1891 | 1803 |
| integer, one operator (today's trigger) | 61.8% | 60.1% |
| decimal operand, one operator | 8.9% | 9.9% |
| integer chains (2+ operators) | 3.9% | 4.1% |
| containing `\frac` | 4.8% | 3.9% |
| containing parentheses | 4.2% | 4.9% |
| problems with any decimal | 85 | 95 |
| problems with `\frac` | 71 | 66 |
| problems with a negative operand | 26 | 20 |
| `\sqrt`, `\log`, `^`, `!`, `\binom`, `mod` | ≤ 9 problems each | |

- **Gold calculator annotations.**
  - GSM8K test (4282 annotations):
    - 4.9% have a decimal input (10.8% of problems).
    - 13.1% are multi-operator (35% of problems).
    - 0.7% are inexact divisions.
    - Powers and percents: none.
  - GSM-Hard solutions:
    - 22.6% contain decimal literals, 23% have non-integer answers and 11.1% have negative answers.
    - `%` appears in 12.5%.
- **DeepMind modules, fraction of all 10k questions:**

| module | decimal operands | negative operands | chains with parentheses | fraction answers |
|---|---|---|---|---|
| mul_big (extrapolate) | 75% | 74% | – | 0% |
| add_or_sub_big | 74% | 74% | – | 0% |
| div_big | 0% | 74% | – | 45% |
| mul_div_multiple_longer | 0% | 99% | all | 0% |
| mixed_longer | 0% | 99% | all | 0% |
| mixed (interpolate) | 0% | 96% | all | 49% |

So the integer-only computer can be used on about a quarter of DM's mul_big and add_or_sub_big,
which is why the `dmxc_` clean subsets exist. Decimals, negatives and chains are worth far more
than logs or roots on every benchmark we run today. Roots, logs and powers matter for MATH,
SciBench and synthetic sets.

## 1. The architecture decision: an expression tape, not one trigger per op

Today each op needs its own text class, scanner state, capture columns, CALL-entry logic and
digit ports. These capture costs are in the circuit; the arithmetic itself lives in the BIOS.
Doing that for 20 more ops would cost circuit width per op. It would also never cover chains,
parentheses or mixed notation.

**Proposal: two levels.**
1. **The fast path (keep).** Single binary add/sub/mul/div with the existing triggers, extended
   to decimal and signed operands (§3, P1). These are 60-70% of Qwen's calculations and the
   cheapest in tokens.
2. **The expression tape (new, P2).** Every other calculation goes through it.
   - **Capture.** While a candidate window is open, each text token's class code (6-7 bits) is
     stored in a new DeltaNet "tape" head, keyed by its index in the window (≤ 64 tokens). This is
     the same mechanism as today's digit port, which keys digits by (argument, digit index), but
     keyed by token index. No CALL record copies the expression, so the 64-write limit doesn't
     bind.
   - **Fire.** The window fires at " =" or "=" if it contains at least one operator or function
     and passes the fire rules in §5.
   - **Hand-off.** The CALL record then writes only the tape length, MODE and PC = `eval`.
   - **Read.** A new instruction, **BTAPE 5XYD: VX = tape[VY]**, lets the BIOS read the tape one
     token per record (about 5 tokens each).
   - **Evaluate.** A BIOS parser (shunting-yard, ≤ 8 operands, paren depth ≤ 4) evaluates with
     exact rational bignums. It calls the existing routines as subroutines; they get RET-ending
     entry variants.
   - **Print.** An integer, a terminating decimal, a reduced fraction or a 10-decimal truncation
     (§4).
   - **New ops become BIOS-only.** sqrt, ^, log, mod, gcd and ! need only text classes for
     their spellings plus BIOS code. They add zero circuit heads.
   - **Tool mode.** It gets one tool, `calc(expression)`. Its parameter text goes through the same
     tape. ("=calc" tokenizes as "=", "calc", so it needs one FN class.)

**The trade-off.**
- **Tokens.** Parsing costs about 25-40 trace tokens per input token, so a 20-token expression
  adds about 0.5-0.8k tokens before any arithmetic. That's why single ops keep the fast path.
- **Coverage.** The tape covers chains, parentheses, signs, `\frac`, `^` and functions in one
  circuit change. It lets Qwen write a whole DM expression once and get the exact answer. In
  inline mode those modules currently fail by running out of context while Qwen evaluates by hand.
- **Verdict.** Build the tape before any new op, and route every new skill through it.

## 2. New shared macros (promote hot loops; each is one record)

| opcode | name | semantics | uses | est. circuit cost |
|---|---|---|---|---|
| 5XYB | BMOV | A[0..n) = R[z..z+n), LA = n, with z = R's leading zeros (Y bit: into B) | every loop that feeds a result back (powers, factorial, gcd, Newton) | 1-2 heads, 0.5-1k units, 30-60 dims |
| 5XY9 | BSDIV | in place A = A div VX for 2 ≤ VX ≤ 25, MSB first, VF = remainder (r·10 + d ≤ 249 stays bf16-exact) | base conversion, small-prime trial division, halving, mod small | 1-2 heads, 1-1.5k units, ~50 dims |
| 5XYA | BSMAD | in place A = A·VX + VY for VX ≤ 16, VY ≤ 15 (9·16 + 15 = 159 < 256) | Horner (from base k), n! and nCr with small factors, ×10^k alignment | 1-2 heads, 1-1.5k units, ~50 dims |
| 5XYC | BSQRT (Y: add-back) | one square-root digit, modelled on BDIV: window -= (D·10 + q)·q with q estimated from the window's top digits over D's (D = 2·root so far); P[t] = q; flags like BDIV | sqrt | 3-5 heads, 2-3k units, 150-250 dims (BDIV+BADB took about 380 dims) |
| 5XYD | BTAPE | VX = tape[VY] | the evaluator | tape head 1-2 heads + 0.3k units, ~30 dims |
| 5XYE | BSHAD | in place W += W >> VX (Briggs step), VF = [W ≥ 10^m] | log, exp, a^x | 1-2 heads, ~1k units, ~50 dims |
| (print) | PRINT variants | MODE bit 16 prints nibbles 10-15 as a-f (hex); a register gives a "." position, so decimals print without shifting memory | hex output, every decimal result | 50-150 units |

**Strategy.** Write each skill first as a BIOS program on the existing macros, which costs zero
circuit width. Prove it exact against the Python reference. Measure where its tokens go, then
promote only the loops that dominate. BMOV comes first, because every iterative skill copies R
back into A.

## 3. Skill table

Trace estimates are for typical sizes. "BIOS" means no circuit change beyond the text classes.

| skill | how Qwen writes it (measured Qwen3.5 tokens) | output | algorithm | BIOS / macro | trace tokens | circuit | unblocks |
|---|---|---|---|---|---|---|---|
| **decimal operands** (add/sub/mul/div) | `' ','3','.','7','5',' \\','times',' ','2','.','4',' ='`; "1,234.56" keeps its commas | exact decimal, trailing zeros stripped ("7.5"), "-" sign; div keeps its 10-decimal "..." form | capture counts fraction digits fa, fb per operand (PERIOD becomes neutral between digits; a second "." breaks the window). The BIOS aligns: add/sub pad the shorter fraction with zeros (one FX55); mul puts the point at fa+fb; div computes A·10^fb / (B·10^fa) | capture change + BIOS; later the print-with-"." variant | +60-90 over the integer op (add 12.5+3.75 ≈ 110; mul 4x3 decimal ≈ 130) | 2-3 heads, 0.5-1k units, 30-60 dims (fraction counters, CALL writes LEN+3/4) | DM mul_big and add_or_sub_big full modules (75% decimal), GSM-Hard (~10% of calculations), NUPA Float tasks, FinQA |
| **negative operands** (all ops) | `' -','1','2'` (" -" is MINUS, the binary-minus class: unary when p = 0 or after an operator), `' (-','1','2',')'` (" (-" and "(-" are fused tokens: new classes; ")" becomes neutral) | " -digits" (MODE bit 8, already printed by readout) | sign bits sa, sb in MODE. mul/div: sign = sa xor sb. add/sub: compare magnitudes with BSUB's branch and route to add, sub or BRSUB (exists) | capture + BIOS | +20-40 | ~1 head, 0.3k units, ~15 dims | DM (74-99% negative operands), GSM-Hard (11% negative answers), BBH |
| **expression evaluator** | chains `350,000 + 17,500 + 42,000 =`, `' (','2',' +',' ','3',')',' \\','times',...`, `\frac{1785}{13201915} =` (`'}{'` and `'}'` are tokens), DM-style `(-1)/(-1)*(-10)/(-4)*...` | integer; exact decimal if any operand is decimal; reduced "p/q" for an inexact all-integer expression; else 10 decimals + "..." | tape → shunting-yard → rational bignums. Reduce by gcd only when the final p/q isn't an integer (try BDIV first) | tape + BTAPE + BIOS (~300-400 bytes) | parse 25-40/token; 5 operands of ≤ 6 digits ≈ 1.5-3k; DM mixed_longer (~25 tokens, 8 operands) ≈ 3-6k | ~3-5 heads, 1-2k units, 60-100 dims (tape, trigger generalisation, ~25-30 new text classes) | DM mixed, mixed_longer, mul_div_multiple(_longer), add_sub_multiple(_longer), div_big fractions (45%); BBH mat2; GSM chains (4%) and `\frac` (4-5%) |
| **integer powers** a^n | `' ','2','^{','1','0','}',' ='`, `'^'` (61), `' ^'`, `'**'` (332); `'^{-'` is fused (negative exponents → rationals) | integer up to 58 digits, else scientific "1.2676506002e30" (digits, ".", "e" are all single-spelling classes) | n ≤ 16: repeated BMUL + BMOV. Larger n: square and multiply | BIOS, BMOV | 7^20 (17 digits) ≈ 1.3k; 2^64 ≈ 1.0-1.5k; 3^100 (48 digits) ≈ 4k | classes only (+BMOV) | MATH counting/number theory, synthetic, SciBench |
| **factorial, nCr, nPr** | `' ','5','!',' ='`, `'\\','bin','om','{',...,'}{',...`; "C(10, 3)" is ambiguous (skip) | integer / sci | n!: BSMAD per factor ≤ 16, then BMUL. nCr: multiplicative with exact BSDIV by i ≤ 25 (above 25, BDIV) | BIOS, BSMAD/BSDIV | 20! ≈ 0.8k; C(50,25) ≈ 1.8k | classes only (+macros) | MATH counting & probability |
| **sqrt** | `' \\','sqrt','{',...,'}',' ='` (with `$\\` fused when after "$"); `'sqrt','('` | integer if a perfect square, else 10 decimals + "..." | digit by digit (pairs of digits, like long division). The BIOS prototype estimates the digit with 8-bit register maths, then checks with BMUL + BSUB; BSQRT later | BIOS first, then BSQRT | 6-digit radicand → 13 digits: BIOS ≈ 4-5k, with BSQRT ≈ 1.2k | 0, then BSQRT | MATH, SciBench, synthetic; "√77449 ≈ 278" style prime checks in GSM-Hard |
| **n-th roots** | `'\\','sqrt','[','3',']{'...`, `'^{','1','/','3','}'` | integer if exact, else 10 decimals + "..." | exact integer roots: digit-by-digit search with a power check. General: exp10(log10(x)/n) | BIOS | perfect cubes ≈ 2k; general ≈ log + exp (§ logs) | 0 | SciBench, MATH (rare) |
| **log10, ln, log2, log_b** | `' \\','log','_{','1','0','}',...`, `' \\','log','_','2',...`, `' \\','ln',' ',...`, `'\\','log','('` | 10 decimals + "..." (truncated; integer when exact, e.g. log2 8 = 3, found by an exact-power pre-check) | Briggs pseudo-division: normalise x into [1, 10) (shift count = characteristic). For k = 0..12, multiply x by (1 + 10^-k) while ≤ 10, subtracting log10(1 + 10^-k) from a 14-digit accumulator. ROM table: 13 constants × 14 digits ≈ 182 bytes. ln = log10 × 2.30258509299404 (one BMUL); log2 ≈ × 3.32192809488736; log_b a = log a / log b | BIOS first; BSHAD makes it practical | BIOS ≈ 14k (about 58 inner steps × about 240); with BSHAD plus a table-load BMOV variant ≈ 3.5k; log_b ≈ 2 logs + div | 0, then BSHAD | SciBench (chemistry pH, decibels, half-lives), synthetic vs mpmath |
| **exp, a^x (real x), e^x** | `' e','^{','0','.','5','}'` (" e" is a word token: a class only after a math opener), `' \\','exp','('` | 10 decimals + "..." | inverse Briggs: subtract table constants from y, multiply by (1 + 10^-k). a^x = exp10(x·log10 a) | BIOS, then BSHAD | ≈ the same as log (3.5-14k); a^x ≈ 2× | 0 | SciBench, finance (compound interest with non-integer periods) |
| **mod, floor-div** | `' ','1','7',' \\','mod',...`, `' \\','b','mod'` ('b' is the hex-digit class: allow BSL, b, mod), `' mod'` (1410), `'\\','p','mod','{'` (congruence, no "="; skip), `' //'` | integer remainder / quotient | div's BDIV loop stopped at t = LA; print the W window or P (v16) | BIOS (v16) | ≈ (LA−LB+1)·17 + 100: 12/4 digits ≈ 250 | classes only | DM numbers__div_remainder, MATH number theory, GSM-Hard (12.5% use %) |
| **gcd, lcm** | `' \\','gcd','('`, `' gcd','('`, `'\\','text','{l','cm','}('` (lcm is awkward; tool form) | integer | Euclid via BDIV remainders + BMOV (v16); lcm = a / gcd · b | BIOS (v16) + BMOV | 12-digit pair ≈ 23 steps × 120 ≈ 2.8k; lcm +0.8k | classes only | DM numbers__gcd, numbers__lcm, fraction reduction in the evaluator |
| **modular exponentiation** | tool `powmod(a, e, m)`; inline is rare | integer | square and multiply with a mod each step | BIOS | 10-digit m, 30-bit e ≈ 35k (too slow inline; tool only) | 0 | MATH number theory (rare) |
| **primality, factorization** | "Is 25439 prime?" is prose (tool `is_prime`, `factor`); `=is`,`prime` and `=f`,`actor` fuse in tool names | "yes"/"no" is awkward (output classes); print the smallest factor, or the factor list "2, 3, 3, 7" (COMMA exists) | trial division: BSDIV by primes ≤ 25 (one record each), then BDIV by a ROM table of primes < 1000 (336 bytes). Complete for n < 10^6 | BIOS, BSDIV | composite: early exit, usually < 1k; prime near 10^6: ≈ 20k | 0 (+BSDIV) | DM numbers__is_prime, numbers__list_prime_factors (check sizes after download) |
| **base conversion** | `' ','1','0','1','1','_','2'`, `' (','1','0','1','1',')_','2'` (')_' fused), `'_{','1','6','}'`, `' 0','x','1','F'` ('x' and uppercase A-F aren't classes); DM uses prose "(base 11)" → tool `to_base(n, k)` / `from_base(s, k)` | digits in base k, with a-f printed through the hex print bit (MODE 16) | to base k: BSDIV passes (one record per output digit). From base k: Horner with BSMAD | BIOS + BSDIV/BSMAD + hex print | 12-digit decimal → binary ≈ 1.4k, → hex ≈ 0.4k; 40-bit binary → decimal ≈ 1.2k | +60 units (print) and ~8 input classes ('x', 'A'-'F' as input-only, '_', '_{', ')_') | DM numbers__base_conversion, arithmetic__add_or_sub_in_base |
| **percent** | `' ','1','5','%',' of',...` (" of" is a word: allowed only straight after "%"), `'\\','%'`, `' %'`; Qwen mostly writes 0.15 × 80 itself | exact decimal | x% → x/100 (decimal point shift, free), "of" → × | BIOS in the evaluator | ≈ decimal mul | 3 classes | FinQA, GSM (Qwen rarely needs it) |
| **rounding** | prose ("round to 2 dp"); tool `round(x, n)` | decimal | truncate at n plus round-half-even/up on the guard digit (a carry via BADD) | BIOS | ≈ 100 | 0 | DM numbers__round_number, FinQA |
| **comparison, min/max** | prose, tool `max(...)` | the chosen number | BSUB's branch | BIOS | ≈ 50 per comparison | 0 | DM comparison__* (Qwen already does these well: low value) |
| **trig (sin, cos, tan, atan)** | `' \\','sin','('`, `'^\\','circ'` (degrees) | 10 decimals + "..." | CORDIC with a ROM atan table (≈ 13 × 14 digits), degrees → radians by one BMUL | BIOS (+BSHAD) | ≈ 10-20k | 0 | SciBench physics only. **Low priority.** |

**Ambiguities to settle when building.**
- **The DIV class must split.** "/" is today's DIV class with extra spellings " /" and "÷". To
  print reduced fractions "p/q", "/" must become a single-spelling printable class and the others
  move to an input-only DIV2. This is the same fix MINUS/MINUS2 needed.
- **Scientific notation.** Printing "× 10^{n}" would need printable "^{" and "}" classes, and
  " ×" has several spellings. So print "1.2676506002e30" instead: digits, "." and "e" are all
  single-spelling already.
- **Precision of irrational results.** These use 14-15 internal digits for 10 printed digits.
  The printed truncation is wrong only when the true digits 11-14 are all 9s or all 0s within
  the error bound, with probability about 10^-3. The routine can detect that case and add guard
  digits (one more Briggs level).

## 4. Output formats and graders

| result | printed | GSM grader (numeric) | DM grader (exact string after our numeric parse) |
|---|---|---|---|
| integer | " 382", " -382" | ✓ | ✓ |
| exact decimal | " 31572525473.687842" (no trailing zeros) | ✓ | ✓ |
| inexact division | " 0.0001352076..." | ✓ approximately | ✗ when the gold is "357/2640383": the evaluator prints " 357/2640383" for all-integer input |
| irrational (sqrt, log) | " 1.4142135623..." | Qwen rounds | DM has no such module; SciBench uses a relative tolerance |
| huge integer | " 1.2676506002e30" | ✓ if parsed | not needed |

Check every printed format against the scorer in `evals.py` (`is_answer`, `to_number`) before a
benchmark run. The v14 selftest bug (truncation vs rounding) was exactly this.

## 5. Build order (value per effort) and acceptance tests

Every phase is a new checkpoint version and has to pass the same gates:
1. **Reference.** The Python reference (`macros.py` and the BIOS on `BridgeMachine`) is exact on
   ≥ 2,000 random cases against `fractions`/`decimal`/`mpmath`. The cases cover edge cases: 0,
   ±, the longest operands, leading and trailing zeros, exact and inexact results.
2. **Circuit.** IR evaluator == reference on every new routine's traces plus the existing fuzz
   (`tests/test_q35_*`). The tiny bf16 build is token-exact, and all q35 tests pass.
3. **GPU.** The evals selftest on the merged 4B gets 0 wrong across every skill's tool and inline
   forms, with text KL vs base within v13-v15's range (3e-4 to 2e-3).
4. **Benchmark.** Paired McNemar vs base_nudge and vs the previous version on the target sets.
   No module may regress beyond noise, and the machine must be exact on every fire.

| phase | contents | effort | benchmark that should move |
|---|---|---|---|
| **P1 (v17)** | decimal and negative operands for the fast path (on top of v16 signed div); print-with-"." variant | 3-4 days | DM dmx_mul_big and dmx_add_or_sub_big **full** modules (today only the 25% clean subset is usable), GSM-Hard, NUPA Float tasks |
| **P2 (v18)** | tape head, BTAPE, generalised trigger (§5 rules), BIOS shunting-yard evaluator with rationals, RET-ending routine variants, BMOV, tool `calc(expression)`, DIV/DIV2 split | 6-10 days | DM mixed, mixed_longer, mul_div_multiple_longer, add_sub_multiple_longer, div_big (fraction golds), BBH mat2 |
| **P3 (v19)** | `^`, `!`, `\binom`, `%`, rounding in the evaluator; BSMAD | 2-3 days | synthetic powers grid, MATH counting & probability |
| **P4 (v20)** | `mod`, `//`, `gcd`, `lcm` in the evaluator (from v16); base conversion with BSDIV and the hex print bit; primality and factorization below 10^6 (tool forms) | 4-6 days | DM numbers__gcd, lcm, div_remainder, base_conversion, is_prime, list_prime_factors; MATH number theory |
| **P5 (v21)** | sqrt (BIOS, then BSQRT), n-th roots, log/ln/log2/log_b and exp/a^x (BIOS Briggs, then BSHAD) | 6-10 days | synthetic sqrt/log vs mpmath, SciBench |
| P6 (optional) | trig by CORDIC | 4 days | SciBench physics |

**Trigger rules for the tape (P2).** These carry over the lessons from v8-v14:
- **Where the window starts.** On a safe opener (space, newline, "$", "$$", "(", " (-", a chip
  token) followed by a number, "(", "-", `\frac`, `\sqrt`, `\log` or `\ln`. So "x^2 + 1 =" and
  "f(3) =" never fire.
- **What the window may contain.** Digits, ".", a comma between digits, spaces, + − × \times
  \cdot * / ÷ \div, ( ), ^ ^{ ^{- { } }{ **, \frac \dfrac \sqrt [ ], mod \bmod //, !, %, \log
  \ln _ _{, and $ openers. Anything else breaks the window. The tape holds at most 64 tokens;
  past that it doesn't fire.
- **When it fires.** At "=" or " =", with at least one operator or function and balanced
  brackets. Plus one of:
  - total digits ≥ the fast-path thresholds (`INLINE_MIN_DIGITS`), or
  - a "hard" construct: sqrt, log, ln, ^ with exponent ≥ 3, !, binom, mod with ≥ 4 digits, or
    ≥ 3 operands.
- **Never an empty print.** The v12 lesson: an empty print makes Qwen loop. Every fire prints a
  value. Cases the BIOS can't handle print the best exact prefix it can, or "e" notation; they
  never print nothing.
- **"=" followed by Qwen's own number.** As today, the scanner restarts after "=".

## 6. The circuit budget

| | heads | units | live dims | depth risk |
|---|---|---|---|---|
| P1 capture (fraction counters, signs, classes) | +2-3 | +0.5-1k | +30-60 | low: capture runs on text tokens, off the macro chain |
| P2 tape + BTAPE + trigger + BMOV | +3-5 | +1.5-3k | +90-160 | low-medium: BTAPE is a plain register write; BMOV reads the R bank like BMUL |
| P3/P4 BSMAD, BSDIV, hex print | +2-4 | +2-3k | +100 | medium: on the macro chain, but simpler than BDIV (one carry, no estimate) |
| P5 BSQRT, BSHAD | +4-7 | +3-4k | +200-300 | **high**: BSQRT is a BDIV-class datapath on the critical path |
| **total** | **+11-19 (→ 115-123 of 256)** | **+7-11k (→ 32-36k)** | **+420-620** | |

**What that means per model.**
- **4B.** MLP room is ample at 9216. Hidden grows by about 420-620 dims, from 4224 to about
  4650-4850 (the `--hidden`/`--ballast` power-of-4 rule picks the exact value). That's about 10-15%
  more bytes per token, so decode is about 10-15% slower. P1-P4 alone add about 220-320 dims,
  about 5-8%.
- **Tiny and 0.8B (intermediate 3584).**
  - Sublayers 2, 12 and 14 are already full, and 16 and 20 are about 89% full.
  - P1-P4 should fit by spreading into the empty late sublayers (22-32).
  - P5's BSQRT will likely need a 17th or 18th chip layer. Alternatively the tiny checkpoint can
    drop BSQRT and BSHAD and keep the BIOS-only sqrt and log, which are slower but exact.
- **Depth.** BDIV fit only after moving its quotient digit to the record's first write (§0), so
  depth needs watching for each macro on the chain. Build BSQRT on BDIV's estimate and window
  logic rather than as a separate datapath.
- **JMAX and digit limits.**
  - Up to 29 digits per operand fits as is (the CALL record writes LA + LB + 4 ≤ 64).
  - Beyond that, JMAX would have to rise from 64 to 96. The 2^-j counter stays in bf16's normal
    range (2^-97). Cost: about +32 thermometer nodes plus every j-enumerating term, about
    1.5-3k units and 40 dims.
  - Results over 58 digits print in "e" notation. Exact printing past 58 digits would need a
    print-and-continue variant that keeps the circuit as owner.
- **ROM.** The evaluator (about 400 bytes), the Briggs routines and table (about 400), the prime
  table (336) and the base/power/factorial routines (about 300) total about 1.4 KB. That fits in
  the free space below 0x600 and above 0x900. Routines that are macro jump targets stay below
  0x400.

## 7. Benchmarks per skill

| skill | benchmark | local? |
|---|---|---|
| decimals, negatives | DM `arithmetic__mul_big`, `add_or_sub_big` (full, not clean), interpolate `add_or_sub`, `mul`; GSM-Hard | yes: DM arithmetic modules on q35-tc, q35-eval, aws-q35 (`~/data/dm`); GSM via HF |
| | NUPA Float/Fraction/ScientificNotation tasks (per the Number Cookbook paper; also its floordiv, mod, max/min, sig_fig, to_scient tasks) | **no**: only `int_tasks.json` (the 4 Integer tasks) is cached; re-download the full `test.json` |
| | FinQA test (programs: divide, multiply, subtract, add, exp, greater) | **no**: the HF copies are script datasets and no longer load; get the GitHub JSON (czyssrs/FinQA) |
| evaluator | DM `mixed`, `mixed_longer`, `mul_div_multiple(_longer)`, `add_sub_multiple(_longer)`, `div_big`; BBH `multistep_arithmetic_two` | yes |
| powers, factorial, binom | MATH counting_and_probability, number_theory, prealgebra (EleutherAI/hendrycks_math); synthetic a^n grid | **no** (HF download) / generate |
| mod, gcd, lcm, primes, bases | DM `numbers__gcd`, `numbers__lcm`, `numbers__div_remainder`, `numbers__is_prime`, `numbers__is_factor`, `numbers__list_prime_factors`, `numbers__base_conversion`, `arithmetic__add_or_sub_in_base`; MATH number_theory | **no**: the boxes extracted only `arithmetic__*`; download `mathematics_dataset-v1.0.tar.gz` (~2.2 GB) and extract `numbers__*`, `comparison__*`, `measurement__*` |
| roots, logs, exp | synthetic sqrt/log/exp/a^x vs mpmath (10-decimal truncation exact match); SciBench (relative tolerance) | generate / **no** (HF) |
| rounding, comparison, units | DM `numbers__round_number`, `comparison__*`, `measurement__conversion` | **no** (same tarball) |
| no-harm | GSM8K, SVAMP/ASDiv (small numbers), MMLU, IFEval, text KL | GSM8K yes; others HF |

The most persuasive synthetic charts are accuracy against operand size: n×m digit products,
a^n, sqrt and log to 10 decimals. Base Qwen collapses as sizes grow, while the computer stays
at 100% up to its digit limits.

## 8. Risks

1. **Qwen not handing over the calculation (the division lesson).**
   - On DM div_big, Qwen did long division by hand and 21 of 100 problems ran out of context. It
     wrote "a/b =" for the whole division only part of the time.
   - New notations will have the same problem, worse for the rare ones (`\log`, `\sqrt`), which
     Qwen tends to approximate ("≈ 278") rather than compute.
   - **Mitigations:**
     - Nudge prompts that list the exact notations.
     - Measure the hand-over rate per skill, i.e. the fraction of gold calculations that Qwen
       writes in a firing form.
     - Later, when weight changes are allowed, the LoRA from the roadmap.
2. **Trigger false positives.**
   - Watch for:
     - "%" as percent vs mod: take percent; mod only via "mod".
     - "e" as Euler's number vs a word: a class only after a math opener.
     - "^" in "x^2": the window must start at a number.
     - Version strings like "3.5.1": a second "." breaks the window.
     - Prices like "$2.50": treat "$" as the opener, with the decimal number inside.
     - "!" in prose: fire only as "digits! =".
   - **Accept:** false positives below 2% on GSM8K/MMLU text, and no change in text KL.
3. **Output-format mismatches.**
   - Fraction vs decimal (§4), "e" notation, and the trailing "." closer.
   - The truncated "..." form: graders that round can disagree in the last digit, and Qwen may
     round it itself.
   - Run every format through `is_answer` before benchmark runs.
4. **Trace length vs context.**
   - BIOS-only log/exp (about 14k tokens), primality (up to 20k) and modpow (35k) exceed today's
     merged budget. `max_len` is budget + 1024 + 4096, so these traces end as `context` stops.
   - **Options:**
     - Promote the hot loops to macros: log ≈ 3.5k, sqrt ≈ 1.2k.
     - Give tool-only forms with a larger budget.
     - Use the speed work's KV dropping for chip tokens, which is exact for the chip.
   - **Do this:** set a per-skill token cap in the reference (assert in tests), and check it
     against the eval budgets.
5. **Width creep slows decode by 10-15%** at the 4B (§6). Promote only loops whose share of trace
   tokens on the target benchmarks is measured, and keep BIOS-only versions where the speed doesn't
   matter.
6. **Text classes.** Qwen generates its own tokenization of tool names: "=add", "=m" "ul", "=p"
   "ow", "=s" "qrt", "=g" "cd", "=l" "cm", "=f" "actor" "ial". Its generated spellings, not
   the tokenizer's split, decide the classes. Measure them from generated calls, as was done for
   FN_ADD/FN_MUL. Prefer tool names that stay one token after "=": "=root", "=log", "=mod",
   "=calc", "=power", "=exp", "=round".
7. **8-bit BIOS maths.** Digit estimates (sqrt, Briggs normalisation) work on values up to 999
   in 8-bit registers. Use the 10H + L split as the circuit does, or two-register arithmetic, and
   fuzz the boundaries.

## 9. P2 as built (v18) and where its trace tokens go

**What v18 has** (branch `q35-v18-expr`):
- The tape capture in the circuit: one head of up to 48 codes plus window counters, about 10 heads in all.
- The expression CALL record, and two copy macros:
  - BLD (5X0B / 5X1B): A or B = mem[I..I+VX), with its length.
  - BSTA (500C): mem[I..] = A.
- A BIOS evaluator in `lib/eval.asm` (the general shunting-yard over unreduced fractions) and
  `lib/chain.asm` (chain mode).
- The circuit is still 16 chip layers at hidden 4352 for the 4B (1,640 slots), and 3456 for the 0.8B.
- The BIOS is about 3 KB, so the boot record is about 4.5k tokens.

**Measured trace tokens per expression.** Median of 60 DM questions; the BIOS runs on the reference machine, and the circuit matches it.

| module | v18 first (8b5957d) | chain mode (e5dcd6e) |
|---|---|---|
| mul_div_multiple_longer | 13.5k | 9.3k |
| add_sub_multiple_longer | 11.9k | 7.5k |
| mixed_longer | 14.7k | 17.7k (chain mode, then a restart) |

**Where they go.** Take one mul_div question: `(-1)/(-1)*(-10)/(-4)*70/(-7)*28/(-4)*(6/1)/((-1848)/(-44))`. That is 40 tape cells and 16 numbers, giving 2,097 records and 10.2k tokens. A plain CHIP-8 record costs about 5 tokens: STEP, ADDR PC, VAL PC, and usually one more write.

| part | records | per unit |
|---|---|---|
| reading the tape (fetch, dispatch, digits, fused-token stubs) | ~430 | ~10 per cell |
| ending a number (length, leading zeros, limits, its sign or role bits) | ~300 | ~19 per number |
| adding it to an accumulator (B = number, A = accumulator, macro, R to A, lengths, spill or reload) | ~430 | ~27 per number, plus the macro's own writes |
| operators (unary or binary, deciding the mode, the next operand's bits) | ~390 | ~17 per operator |
| brackets | ~200 | ~10 per bracket |
| setup, the final print or div | ~350 | |

So each number costs about 70 records, about 300-350 tokens, before any arithmetic. The general
evaluator costs about 2x this, mostly in the operator stack, value headers and copies. Plain
CHIP-8 can't get much lower: hand-tuning the general evaluator gained under 10%.

**What circuit-level macros would save** (estimated from the counts above):

1. **An operand-load macro, BNUM.**
   - What it does: from tape index VX, take the digit run (stopping at the first non-digit code), skip its leading zeros, and write B = the digits and LB = their count. Then VX = the run's end and VF = [the run is "0"].
   - What it replaces: the ~19 records of ending a number, plus the ~3 of loading B.
   - Cost: one record of 2n + 5 tokens.
   - Circuit: a run-length scan over the tape. It is a thermometer over "is a digit" of tape[VX + j], so it costs one RAM-bank read per write, like BLD, plus a "first non-digit" prefix chain. That chain is the hard part: a prefix-AND over up to 24 cells. It is about 2-3 heads and 1-2k units, and the depth risk is medium.
   - Saves about 20 records per number, about 30% of chain mode.
2. **An accumulate macro, BACC (Y bit: multiply or add).**
   - What it does: ACC[k] = ACC[k] op B in place, where k = VY's bit 0, with its length.
   - What it replaces: today's facc sequence of A = accumulator, macro, R to A, lengths, and the spill or reload, about 27 records.
   - Cost: the macro's own digit writes plus about 3 records.
   - Circuit: it reuses BMUL/BADD's datapath, with the accumulator as a third digit bank in place of A. A 24 × 24 product in one record exceeds JMAX, so it would keep BMUL's row loop.
   - Saves about 20 records per number.
3. **A tape-dispatch instruction (5XYD as in §1, plus a jump).**
   - What it does: VX = tape[VY], VY += 1, then PC = TABLE + 2 · VX in one record.
   - What it replaces: the 6-record fetch and dispatch per cell.
   - Saves about 5 records per cell (40 cells: about 200 records).
4. **No PC write for a sequential instruction.**
   - Today a record whose PC stays has no PC write. The other way round, the default would be PC + 2, and staying macros (BMUL rows, BTRIM, BDIV) would write their PC.
   - Saves 2 tokens on most records, about 35% of every BIOS trace, games included.
   - It is also the deepest change: the circuit has to derive PC from the last PC write plus 2 × the records since it. That needs a counter head and a 12-bit add on the fetch path, which is the critical path.

With 1-3, a mul_div chain like the one above would come to about 16 × (2n + 5 + 10) + 40 × 5 + 22 × 12 + 350 records. That is about 1.1k records, about 3-4k tokens with the macro writes. With 4 as well, about 2.5k. For comparison, the expression's own text is about 40 Qwen tokens; if Qwen worked it out step by step, it would write about 500-1,500 tokens.

**Known limits of v18.**
- A tool response has no "-", so a negative `calc` result prints "...".
- Fractions print as 10 decimals and "...", not as reduced p/q. That needs the v16 gcd.
- A decimal result followed by more maths (". 0.625. \times 8 =") reads as a number with two
  "." and prints "...". An integer result goes on fine: "408. × 2 =" prints 816.
- Mixed expressions pay for the chain-mode attempt before the general evaluator.
