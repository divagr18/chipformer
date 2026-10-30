// stories260K on the RV core with the VX vector unit (chipformer/rv/vx.py): stories6.c's model with
// the loops as vector ops sized for the circuit (no multiply wider than 24 x 16 bits):
// - matvecs: VMAT over int4 rows (64 lanes a token), out = acc * S_r; the input's scale F is folded
//   into whatever uses the output (a quantize factor, GLU's input scale, the residual add's factor)
// - quantize: VQ, a 16-bit window of each value times a 15-bit factor; the program picks the window
//   from the producer's max |v| (VB tokens keep it) and the factor from the scale
// - SiLU * b, residual adds, RoPE and the embedding row: VEW
// - attention: q per head by VRED + VQ, scores and p.V by VMATR over the caches, softmax by VEXP
// - 1/rms from VRED's sum of squares of the quantized vector, softmax sums by VRED
// Streams are at most 64 tokens, so long vectors are split into ops.
#include <stdint.h>
#include "rt.h"
#include "weights_vx.h"

#ifndef NTOK
#define NTOK 8
#endif
#define W 32       // cache slots: slot 0 holds position 0, slots 1..31 a ring of the latest
#define OFFQ 192   // VA: xq at 0..175, the head's query at OFFQ, probabilities at OFFP
#define OFFP 208

#define VOP(f3, cfg, a, b) \
    __asm__ volatile(".insn r 0x5B, " #f3 ", 0, %0, %1, %2" ::"r"(cfg), "r"(a), "r"(b) : "memory")
#define VMAT(cfg, out) VOP(0, cfg, out, 0)
#define VMATR(cfg, out, s) VOP(1, cfg, out, s)
#define VQ(cfg, src, r) VOP(2, cfg, src, r)
#define VEW(cfg, a, p) VOP(3, cfg, a, p)
#define VEXP(cfg, s, c) VOP(4, cfg, s, c)
#define VRED(cfg, a) VOP(6, cfg, a, 0)
#define P(x) ((int32_t)(uintptr_t)(x))

static inline void vset(const void *p, int32_t w, const void *q) {
    __asm__ volatile(".insn r 0x5B, 7, 0, %0, %1, %2" ::"r"(P(q)), "r"(P(p)), "r"(w) : "memory");
}
static inline int32_t vget(int32_t sel) {
    int32_t r;
    __asm__ volatile(".insn r 0x5B, 5, 0, %0, %1, zero" : "=r"(r) : "r"(sel) : "memory");
    return r;
}
static inline uint32_t mulhu(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) >> 32); }
static inline int32_t mulh(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * b) >> 32); }

static const uint8_t BL8[256] = {
#define B4(n) n, n, n, n
#define B16(n) B4(n), B4(n), B4(n), B4(n)
    0, 1, 2, 2, B4(3), B4(4), B4(4), B16(5), B16(6), B16(6), B16(7), B16(7), B16(7), B16(7),
    B16(8), B16(8), B16(8), B16(8), B16(8), B16(8), B16(8), B16(8)};
static int bitlen(uint32_t u) {
    if (u >> 16) return u >> 24 ? 24 + BL8[u >> 24] : 16 + BL8[u >> 16];
    return u >> 8 ? 8 + BL8[u >> 8] : BL8[u];
}
// the window of the latest tracker (VGET 3: the bit length of its max)
static int vwin(void) { int b = vget(3); return b > 15 ? b - 15 : 0; }
static int32_t vmax_bl(int *bl) { *bl = vget(3); return vget(0); }

// a factor f as r * 2^sr with r < 2^15; returns sr
static int fwin(uint32_t f, int32_t *r) {
    int b = bitlen(f), sr = b > 15 ? b - 15 : 0;
    *r = (int32_t)(f >> sr);
    return sr;
}

static int32_t x[DIM], xb[DIM], q[DIM], kk[KVD], vv[KVD], hb[HID], hb2[HID];
static int32_t sc[W];
static int8_t kq[NL][W][KVD] __attribute__((aligned(4)));  // K, int8 (scale KMAX / 127)
static int8_t vt[NL][KVD][W] __attribute__((aligned(4)));  // V transposed
static uint32_t KQ[NL], VQF[NL], KC[NL];  // 127 * 2^40 / KMAX, / VMAX; KMAX * 47075 >> 10

// VEW over n words in ops of 32; cfg carries everything but n and keep (max |result| tracked)
static void vew(int32_t cfg, int32_t *a, int32_t p, int n) {
    for (int i = 0; i < n; i += 32) {
        int k = n - i < 32 ? n - i : 32;
        VEW(cfg | k | (i ? 1 : 0) << 11, a + i, p);
    }
}
#define EW(op, sh, so, shb) ((op) << 8 | (sh) << 16 | ((so) & 63) << 21 | (shb) << 27)

// v -> VA[off..] = round(v * 127 / amax) (VQ: a 16-bit window of v, times r / 2^15); bl: the
// bit length of amax
static void quantize(const int32_t *v, int n, int32_t amax, int bl, int off) {
    if (amax < 256) amax = 256, bl = 9;
    int sh = bl - 11;
    if (sh < 0) sh = 0;
    int32_t r = (127 << 15) / (amax >> sh);
    for (int i = 0; i < n; i += 128) {
        int k = n - i < 128 ? n - i : 128;
        VQ(k | (off + i) << 10 | sh << 26, v + i, r);
    }
}

// the window and factor of a quantize by m * 2^sh / 2^25 (m = mulhu(F, 127 * 2^40 / max)):
// returns sh, with *r in [2^11, 2^12) where m allows
static int cache_q(uint32_t m, int32_t *r) {
    int sh = 37 - bitlen(m);
    if (sh < 0) sh = 0;
    if (sh > 31) sh = 31;
    *r = (int32_t)(sh >= 25 ? m << (sh - 25) : m >> (25 - sh));
    return sh;
}

// F = 2^36 / sqrt(sum of squares of the quantized vector) (stories6.c's norm_f)
static int32_t norm_f(int n) {
    VRED(n | 1 << 9, 0);
    uint32_t m = (uint32_t)vget(1) << 10;  // r = floor(sqrt(m)): Newton from 2^ceil(bits / 2)
    uint32_t r = 1u << ((bitlen(m) + 1) >> 1), t;
    while ((t = (r + m / r) >> 1) < r) r = t;
    return (int32_t)((0x80000000u / r) << 10);
}

static int32_t plain_f(int32_t amax) { return (int32_t)(((uint32_t)amax << 9) / 127 << 8); }

// out[r] = W_r . VA[0:8k] * S[r] (rows in ops of 32, or 8 for k > 8); returns the window of out
static int32_t matmul(int32_t *out, int wb, const int32_t *S, int k, int rows) {
    vset(S, wb, 0);
    int per = k > 8 ? 8 : 32;
    for (int r = 0; r < rows; r += per) {
        int n = rows - r < per ? rows - r : per;
        VMAT(n | k << 7 | (r ? 1 : 0) << 22, out + r);
    }
    return vwin();
}

// x += mulh(o, f) for the rows o of a matvec (sh: their window)
static void residual(const int32_t *o, int sh, uint32_t f) {
    int32_t r;
    int sr = fwin(f, &r);
    vset(o, 0, 0);
    vew(EW(0, sh, sh + sr - 32, 0), x, r, DIM);
}

// the two heads of KV group g over the n filled slots; q's true values are mulh(q, f)
static void attend(int l, int g, int n, int32_t f) {
    for (int hh = 0; hh < 2; hh++) {
        int32_t *qv = q + (2 * g + hh) * HS;
        VRED(HS, qv);
        int bl;
        int32_t araw = vmax_bl(&bl), amax = mulh(araw, f);
        if (amax < 256) {  // stories6.c floors the query's max at 256
            amax = 256;
            araw = (int32_t)((256ull << 32) / (uint32_t)f);
            bl = bitlen((uint32_t)araw);
        }
        quantize(qv, HS, araw, bl, OFFQ);
        // ca = amax * KC >> 16 (stories6.c's exp-table scale), as its top 16 bits ca >> k1
        uint32_t lo = (uint32_t)amax * KC[l], hi = mulhu((uint32_t)amax, KC[l]);
        int32_t ca = (int32_t)(hi << 16 | lo >> 16), c, ksh;
        if (hi >> 15) {  // ca >= 2^31: stories6.c's int32 ca is negative there, a hard max
            c = 65535, ksh = 0;
        } else {
            int k1 = bitlen((uint32_t)ca) - 16;
            if (k1 < 1) k1 = 1;
            c = ca >> k1, ksh = 32 - k1;
        }
        vset(&kq[l][0][g * HS], 0, 0);  // scores: rows t of the K cache, the signed max for VEXP
        VMATR(n | HS << 7 | OFFQ << 13 | (KVD / 4) << 21 | 1 << 30, sc, 1);
        vset(P8, 0, 0);
        VEXP(n | OFFP << 8 | ksh << 16, sc, c);
        VRED(n | 2 << 9 | OFFP << 11, 0);
        int32_t sum = vget(1);
        int32_t r = (VMAX[l] << 12) / (127 * sum);
        vset(&vt[l][g * HS][0], 0, 0);  // o (Q16) = sum p v8 * VMAX / 127 / sum p
        VMATR(HS | n << 7 | OFFP << 13 | (W / 4) << 21 | 1 << 29, xb + (2 * g + hh) * HS, r);
    }
}

static int forward(int tok, int pos) {
    int bl;
    vset(Q_emb + tok * DIM, 0, 0);
    vew(EW(3, 0, 0, 0), x, S_emb[tok], DIM);
    int32_t vmax = vmax_bl(&bl);
    for (int l = 0; l < NL; l++) {
        quantize(x, DIM, vmax, bl, 0);
        int32_t f = norm_f(DIM);
        int aq = matmul(q, WB_wq[l], S_wq + l * DIM, DIM / 8, DIM);
        int ak = matmul(kk, WB_wk[l], S_wk + l * KVD, DIM / 8, KVD);
        matmul(vv, WB_wv[l], S_wv + l * KVD, DIM / 8, KVD);
        int32_t rp = P(ROPE + pos * HS);
        vew(EW(2, aq, 0, 0) | HS << 12, q, rp, DIM);
        vew(EW(2, ak, 0, 0) | HS << 12, kk, rp, KVD);
        int slot = pos == 0 ? 0 : 1 + (pos - 1) % (W - 1), n = pos + 1 < W ? pos + 1 : W;
        int32_t r;
        int sh = cache_q(mulhu((uint32_t)f, KQ[l]), &r);
        vset(&kq[l][slot][0], 0, 0);
        VQ(KVD | 1 << 8 | sh << 26, kk, r);
        sh = cache_q(mulhu((uint32_t)f, VQF[l]), &r);
        for (int i = 0; i < KVD; i += 16) {
            vset(&vt[l][i][slot], 0, 0);
            VQ(16 | 2 << 8 | W << 18 | sh << 26, vv + i, r);
        }
        for (int g = 0; g < NKV; g++) attend(l, g, n, f);
        VRED(DIM, xb);
        vmax = vmax_bl(&bl);
        if (vmax < 256) vmax = 256, bl = 9;
        quantize(xb, DIM, vmax, bl, 0);
        int ao = matmul(q, WB_wo[l], S_wo + l * DIM, DIM / 8, DIM);
        residual(q, ao, (uint32_t)plain_f(vmax));
        vmax = vmax_bl(&bl);
        quantize(x, DIM, vmax, bl, 0);
        f = norm_f(DIM);
        int sha = matmul(hb, WB_w1[l], S_w1 + l * HID, DIM / 8, HID);
        int shb = matmul(hb2, WB_w3[l], S_w3 + l * HID, DIM / 8, HID);
        int32_t ra;
        int sra = fwin((uint32_t)f, &ra);
        vset(SILU, 0, hb2);  // hb = silu(true a) * (raw b >> shb) >> 8: 2^(36 - shb) / F of the true
        vew(EW(1, sha, sha + sra - 32, shb), hb, ra, HID);
        vmax = vmax_bl(&bl);
        if (vmax < 256) vmax = 256, bl = 9;
        quantize(hb, HID, vmax, bl, 0);
        int a2 = matmul(q, WB_w2[l], S_w2 + l * DIM, HIDW / 8, DIM);
        uint32_t fp = mulhu((uint32_t)plain_f(vmax), (uint32_t)f);  // * F / 2^(36 - shb)
        fp = shb >= 4 ? fp << (shb - 4) : fp >> (4 - shb);
        residual(q, a2, fp);
        vmax = vmax_bl(&bl);
    }
    // the classifier: the final norm only rescales the logits, so argmax skips it
    quantize(x, DIM, vmax, bl, 0);
    vset(S_cls, WB_CLS, 0);
    for (int r = 0; r < VOCAB; r += 64) VMAT(64 | (DIM / 8) << 7 | 1 << 20 | (r ? 1 : 0) << 22, 0);
    return vget(2);
}

int main(void) {
    for (int l = 0; l < NL; l++) {
        KQ[l] = (uint32_t)((127ull << 40) / (uint32_t)KMAX[l]);
        VQF[l] = (uint32_t)((127ull << 40) / (uint32_t)VMAX[l]);
        KC[l] = (uint32_t)(((uint64_t)KMAX[l] * 47075) >> 10);
    }
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
