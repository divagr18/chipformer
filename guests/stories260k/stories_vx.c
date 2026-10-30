// stories260K on the RV core with the VX vector unit (chipformer/rv/vx.py): stories6.c's model and
// numerics, with the loops as vector ops:
// - matvecs: VMAT over whole int4 rows (the activations in VA, the row rescale fused); the
//   classifier's argmax is VMAT's argmax mode
// - quantize: VQ into VA (with the sum of squares for 1/rms) or into the int8 K / V caches
// - SiLU * b, residual adds, RoPE and the embedding row: VEW; the max |v| trackers replace TRACK
// - attention: q per head by VRED + VQ, scores and p.V by VMATR over the caches, softmax by VEXP
// Streams are at most 64 tokens, so long vectors are split into ops of 32 words / 64 rows.
#include <stdint.h>
#include "rt.h"
#include "weights_vx.h"

#ifndef NTOK
#define NTOK 8
#endif
#define W 32       // cache slots: slot 0 holds position 0, slots 1..31 a ring of the latest
#define OFFQ 192   // VA: xq at 0..175, the two heads' queries at OFFQ, probabilities at OFFP
#define OFFP 208

#define VOP(f3, cfg, a, b) \
    __asm__ volatile(".insn r 0x5B, " #f3 ", 0, %0, %1, %2" ::"r"(cfg), "r"(a), "r"(b) : "memory")
#define VMAT(cfg, out, f) VOP(0, cfg, out, f)
#define VMATR(cfg, out, s) VOP(1, cfg, out, s)
#define VQ(cfg, src, r) VOP(2, cfg, src, r)
#define VEW(cfg, a, p) VOP(3, cfg, a, p)
#define VEXP(cfg, s, c) VOP(4, cfg, s, c)
#define VRED(cfg, a) VOP(6, cfg, a, 0)
#define P(x) ((int32_t)(uintptr_t)(x))

static inline void vset(const void *p, int32_t w) {
    __asm__ volatile(".insn r 0x5B, 7, 0, zero, %0, %1" ::"r"(P(p)), "r"(w) : "memory");
}
static inline int32_t vget(int32_t sel) {
    int32_t r;
    __asm__ volatile(".insn r 0x5B, 5, 0, %0, %1, zero" : "=r"(r) : "r"(sel) : "memory");
    return r;
}

static int32_t x[DIM], xb[DIM], q[DIM], kk[KVD], vv[KVD], hb[HID], hb2[HID];
static int32_t sc[W];
static int8_t kq[NL][W][KVD] __attribute__((aligned(4)));  // K, int8 (scale KMAX / 127)
static int8_t vt[NL][KVD][W] __attribute__((aligned(4)));  // V transposed

// VEW over n words in ops of 32 (op: 0 add, 1 glu, 2 rope, 3 emb); returns the max |result|
static int32_t vew(int op, int32_t *a, const void *b, int32_t p, int n, int hs) {
    vset(b, 0);
    for (int i = 0; i < n; i += 32) {
        int k = n - i < 32 ? n - i : 32;
        VEW(k | op << 8 | (i ? 1 : 0) << 11 | hs << 12, a + i, p);
    }
    return vget(0);
}

// v (Q16) -> VA[off..] = round(v * 127 / amax); returns the sum of squares of the codes
static int32_t quantize(const int32_t *v, int n, int32_t amax, int off) {
    if (amax < 256) amax = 256;
    int32_t r = 0x7F000000 / amax;
    for (int i = 0; i < n; i += 128) {
        int k = n - i < 128 ? n - i : 128;
        VQ(k | (off + i) << 10 | (i ? 1 : 0) << 26, v + i, r);
    }
    return vget(1);
}

// F = 2^36 / sqrt(ss) for the vector just quantized (stories6.c's norm_f)
static int32_t norm_f(int32_t ss) {
    uint32_t m = (uint32_t)ss << 10, r = 0, b = 1u << 30;
    while (b > m) b >>= 2;
    while (b) {
        if (m >= r + b) { m -= r + b; r = (r >> 1) + b; } else r >>= 1;
        b >>= 2;
    }
    return (int32_t)((0x80000000u / r) << 10);
}

static int32_t plain_f(int32_t amax) { return (int32_t)(((uint32_t)amax << 9) / 127 << 8); }

// out[r] (Q16) = mulh(W_r . VA[0:8k] * S[r], f), rows in ops of 32
static void matmul(int32_t *out, int wb, const int32_t *S, int k, int rows, int32_t f) {
    vset(S, wb);
    for (int r = 0; r < rows; r += 32) {
        int n = rows - r < 32 ? rows - r : 32;
        VMAT(n | k << 7, out + r, f);
    }
}

// the two heads of KV group g over the n filled slots
static void attend(int l, int g, int n) {
    for (int hh = 0; hh < 2; hh++) {
        int32_t *qv = q + (2 * g + hh) * HS;
        VRED(HS, qv);
        int32_t amax = vget(0);
        if (amax < 256) amax = 256;
        VQ(HS | OFFQ << 10, qv, 0x7F000000 / amax);
        int32_t ca = (int32_t)(((((int64_t)amax * KMAX[l]) >> 10) * 47075) >> 16);
        vset(&kq[l][0][g * HS], 0);  // scores: rows t of the K cache, the signed max for VEXP
        VMATR(n | HS << 7 | OFFQ << 13 | (KVD / 4) << 21 | 1 << 30, sc, 1);
        vset(P8, 0);
        VEXP(n | OFFP << 8, sc, ca);
        int32_t sum = vget(1);
        int32_t r = (VMAX[l] << 12) / (127 * sum);
        vset(&vt[l][g * HS][0], 0);  // o (Q16) = sum p v8 * VMAX / 127 / sum p
        VMATR(HS | n << 7 | OFFP << 13 | (W / 4) << 21 | 1 << 29, xb + (2 * g + hh) * HS, r);
    }
}

static int forward(int tok, int pos) {
    int32_t vmax = vew(3, x, Q_emb + tok * DIM, S_emb[tok], DIM, 0);
    for (int l = 0; l < NL; l++) {
        int32_t f = norm_f(quantize(x, DIM, vmax, 0));
        matmul(q, WB_wq[l], S_wq + l * DIM, DIM / 8, DIM, f);
        matmul(kk, WB_wk[l], S_wk + l * KVD, DIM / 8, KVD, f);
        matmul(vv, WB_wv[l], S_wv + l * KVD, DIM / 8, KVD, f);
        vew(2, q, 0, P(ROPE + pos * HS), DIM, HS);
        vew(2, kk, 0, P(ROPE + pos * HS), KVD, HS);
        int slot = pos == 0 ? 0 : 1 + (pos - 1) % (W - 1), n = pos + 1 < W ? pos + 1 : W;
        vset(&kq[l][slot][0], 0);
        VQ(KVD | 1 << 8, kk, 0x7F000000 / KMAX[l]);
        int32_t vr = 0x7F000000 / VMAX[l];
        for (int i = 0; i < KVD; i += 16) {
            vset(&vt[l][i][slot], 0);
            VQ(16 | 2 << 8 | W << 18, vv + i, vr);
        }
        for (int g = 0; g < NKV; g++) attend(l, g, n);
        VRED(DIM, xb);
        vmax = vget(0);
        quantize(xb, DIM, vmax, 0);
        matmul(q, WB_wo[l], S_wo + l * DIM, DIM / 8, DIM, plain_f(vmax < 256 ? 256 : vmax));
        vmax = vew(0, x, q, 0, DIM, 0);
        f = norm_f(quantize(x, DIM, vmax, 0));
        matmul(hb, WB_w1[l], S_w1 + l * HID, DIM / 8, HID, f);
        matmul(hb2, WB_w3[l], S_w3 + l * HID, DIM / 8, HID, f);
        vmax = vew(1, hb, hb2, P(SILU), HID, 0);
        quantize(hb, HID, vmax, 0);
        matmul(q, WB_w2[l], S_w2 + l * DIM, HIDW / 8, DIM, plain_f(vmax < 256 ? 256 : vmax));
        vmax = vew(0, x, q, 0, DIM, 0);
    }
    // the classifier: the final norm only rescales the logits, so argmax skips it
    quantize(x, DIM, vmax, 0);
    vset(S_cls, WB_CLS);
    for (int r = 0; r < VOCAB; r += 64) VMAT(64 | (DIM / 8) << 7 | 1 << 20 | (r ? 1 : 0) << 22, 0, 1 << 30);
    return vget(2);
}

int main(void) {
    int tok = 1;
    for (int pos = 0; pos < NTOK && pos < MAXPOS; pos++) {
        int nxt = forward(tok, pos);
        int a = VOFF[nxt], b = VOFF[nxt + 1];
        if (tok == 1 && VSTR[a] == ' ') a++;
        for (int i = a; i < b; i++) rv_putchar((char)VSTR[i]);
        tok = nxt;
    }
    rv_putchar('\n');
    return 0;
}
