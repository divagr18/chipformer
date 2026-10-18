// The chipformer digit file (custom-0 / custom-1, see chipformer/rv/ref.py): 8 registers of 128
// decimal digits, index 0 least significant. Row ops take every operand from one word:
//   BN_OPS(n, off, m, d, s, ovw) = d | s << 3 | ovw << 6 | n << 7 | off << 13 | m << 20
//   bn_mac: for i < n: v = (ovw ? 0 : D[i+off]) + S[i] * m + c; D[i+off] = v % 10; returns c
//   bn_msb: for i < n: v = D[i+off] - S[i] * m - b;          D[i+off] = v % 10; returns b
//   bn_ld:  for i < n: D[i+off] = min(end[-1-i] & 15, 9)      (ASCII digits ending at end - 1)
//   bn_get: returns D[off]
//   bn_scan: the length of the ASCII digit run at s
// Printing uses a sled of BPRD instructions per register (sled.S): one token per digit.
#pragma once
#include <stdint.h>

#define BN_OPS(n, off, m, d, s, ovw)                                                        \
    ((uint32_t)(d) | (uint32_t)(s) << 3 | (uint32_t)(ovw) << 6 | (uint32_t)(n) << 7 |         \
     (uint32_t)(off) << 13 | (uint32_t)(m) << 20)
#define BN_OFF(off) ((uint32_t)(off) << 13)
#define BN_M(m) ((uint32_t)(m) << 20)

static inline uint32_t bn_mac(uint32_t ops) {
    uint32_t r;
    __asm__ volatile(".insn r 0x0B, 0, 0, %0, x0, %1" : "=r"(r) : "r"(ops) : "memory");
    return r;
}

static inline uint32_t bn_msb(uint32_t ops) {
    uint32_t r;
    __asm__ volatile(".insn r 0x0B, 1, 0, %0, x0, %1" : "=r"(r) : "r"(ops) : "memory");
    return r;
}

static inline void bn_ld(const char *end, uint32_t ops) {
    __asm__ volatile(".insn r 0x0B, 2, 0, x0, %0, %1" : : "r"(end), "r"(ops) : "memory");
}

static inline uint32_t bn_get(uint32_t ops) {
    uint32_t r;
    __asm__ volatile(".insn r 0x0B, 3, 0, %0, x0, %1" : "=r"(r) : "r"(ops) : "memory");
    return r;
}

// the number of ASCII digits at s, s + 1, .. (at most 63)
static inline uint32_t bn_scan(const char *s) {
    uint32_t r;
    __asm__ volatile(".insn r 0x0B, 4, 0, %0, %1, x0" : "=r"(r) : "r"(s) : "memory");
    return r;
}

extern const uint32_t bn_sleds[8];  // sled.S: the first BPRD of each register's sled

// print the top `len` digits of register r (len <= 128), most significant first
static inline void bn_print(int r, int len) {
    void (*f)(void) = (void (*)(void))(bn_sleds[r] + 4 * (128 - len));
    f();
}
