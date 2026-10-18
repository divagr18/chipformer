// Word-at-a-time memcpy/memmove/memset/memcmp/strlen/strcmp/strchr for RV32 (overrides picolibc's byte loops).
// Build with -O2 -fno-builtin -fno-tree-loop-distribute-patterns so gcc does not turn loops back into calls.
#include <stddef.h>
#include <stdint.h>

typedef uint32_t __attribute__((may_alias)) w32;

void *memset(void *d, int c, size_t n) {
    uint8_t *p = d; uint32_t v = (uint8_t)c; v |= v << 8; v |= v << 16;
    while (n && ((uintptr_t)p & 3)) { *p++ = (uint8_t)c; n--; }
    while (n >= 16) { ((w32 *)p)[0] = v; ((w32 *)p)[1] = v; ((w32 *)p)[2] = v; ((w32 *)p)[3] = v; p += 16; n -= 16; }
    while (n >= 4) { *(w32 *)p = v; p += 4; n -= 4; }
    while (n) { *p++ = (uint8_t)c; n--; }
    return d;
}

void *memcpy(void *restrict d, const void *restrict s, size_t n) {
    uint8_t *dp = d; const uint8_t *sp = s;
    if ((((uintptr_t)dp ^ (uintptr_t)sp) & 3) == 0) {
        while (n && ((uintptr_t)dp & 3)) { *dp++ = *sp++; n--; }
        while (n >= 16) { w32 a = ((const w32 *)sp)[0], b = ((const w32 *)sp)[1], c = ((const w32 *)sp)[2], e = ((const w32 *)sp)[3];
            ((w32 *)dp)[0] = a; ((w32 *)dp)[1] = b; ((w32 *)dp)[2] = c; ((w32 *)dp)[3] = e; dp += 16; sp += 16; n -= 16; }
        while (n >= 4) { *(w32 *)dp = *(const w32 *)sp; dp += 4; sp += 4; n -= 4; }
    }
    while (n) { *dp++ = *sp++; n--; }
    return d;
}

void *memmove(void *d, const void *s, size_t n) {
    uint8_t *dp = d; const uint8_t *sp = s;
    if (dp == sp || n == 0) return d;
    if (dp < sp || dp >= sp + n) return memcpy(d, s, n);
    dp += n; sp += n;
    if ((((uintptr_t)dp ^ (uintptr_t)sp) & 3) == 0) {
        while (n && ((uintptr_t)dp & 3)) { *--dp = *--sp; n--; }
        while (n >= 4) { dp -= 4; sp -= 4; *(w32 *)dp = *(const w32 *)sp; n -= 4; }
    }
    while (n) { *--dp = *--sp; n--; }
    return d;
}

int memcmp(const void *a, const void *b, size_t n) {
    const uint8_t *p = a, *q = b;
    if ((((uintptr_t)p | (uintptr_t)q) & 3) == 0) {
        while (n >= 4 && *(const w32 *)p == *(const w32 *)q) { p += 4; q += 4; n -= 4; }
    }
    while (n) { if (*p != *q) return *p - *q; p++; q++; n--; }
    return 0;
}

#define HASZERO(v) (((v) - 0x01010101u) & ~(v) & 0x80808080u)
size_t strlen(const char *s) {
    const char *p = s;
    while ((uintptr_t)p & 3) { if (!*p) return p - s; p++; }
    while (!HASZERO(*(const w32 *)p)) p += 4;
    while (*p) p++;
    return p - s;
}

int strcmp(const char *a, const char *b) {
    if ((((uintptr_t)a | (uintptr_t)b) & 3) == 0) {
        while (1) { uint32_t x = *(const w32 *)a, y = *(const w32 *)b; if (x != y || HASZERO(x)) break; a += 4; b += 4; }
    }
    while (*a && *a == *b) { a++; b++; }
    return (uint8_t)*a - (uint8_t)*b;
}

char *strchr(const char *s, int c) {
    for (;; s++) { if (*s == (char)c) return (char *)s; if (!*s) return NULL; }
}
