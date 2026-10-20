// calc: exact decimal integer expressions on the chipformer digit file (bn.h).
//
// Input: the NUL-terminated expression in calc_in (RAM 0x100; the host writes it into the
// snapshot). Grammar (Python's precedence and semantics):
//   expr = term {(+|-) term}      term = factor {(*|/|//|%) factor}
//   factor = (-|+) factor | power  power = atom [(**|^) factor]
//   atom = digits | ( expr ) | gcd ( expr , expr )
// / is exact (rationals: + - * / on reduced fractions p/q), // floor division, % the matching
// modulo (the divisor's sign), ** / ^ a power with a small non-negative exponent; // % ** gcd
// need integers. Decimal operands (+ - * only) print with a point. Output: the value ("p/q" for
// a fraction) and "\n", or "error\n" (division by zero, a value or product over MAXD digits, a
// syntax error).
//
// A value is one word: register | negative << 8 | length << 16; a register's digits at and above
// the value's length are 0. Register 7 holds the constant 1 (quotient digits, +1).
#include <stdint.h>
#include "bn.h"
#include "rt.h"

char calc_in[256] __attribute__((section(".calcin")));

#define MAXD 62
#define ONE 7

typedef uint32_t num;
#define REG(a) ((a) & 7)
#define NEG(a) ((a) >> 8 & 1)
#define LEN(a) ((a) >> 16)
#define NUM(r, neg, len) ((uint32_t)(r) | (uint32_t)(neg) << 8 | (uint32_t)(len) << 16)

static uint32_t dirty[8];  // per register: digits that may be nonzero lie below this
static uint32_t used;     // allocated registers
static int err;
static char one_ch = '1';

static int dig(num a, int i) { return bn_get(BN_OPS(0, i, 0, REG(a), 0, 0)); }

static void wrote(num a, int n) {
    if (n > dirty[REG(a)]) dirty[REG(a)] = n;
}

static num alloc(int len) {
    for (int r = 0; r < ONE; r++)
        if (!(used >> r & 1)) {
            used |= 1u << r;
            for (int o = 0; o < dirty[r]; o += 63)  // zero what an earlier value left
                bn_mac(BN_OPS(dirty[r] - o < 63 ? dirty[r] - o : 63, o, 0, r, r, 1));
            dirty[r] = 0;
            return NUM(r, 0, len);
        }
    err = 2;  // out of registers: "too complex"
    return NUM(0, 0, len);
}

static void release(num a) { used &= ~(1u << REG(a)); }

static int is_zero(num a) { return LEN(a) == 1 && !dig(a, 0); }

static num trim(num a) {  // drop leading zero digits; zero is never negative
    int n = LEN(a);
    while (n > 1 && !dig(a, n - 1)) n--;
    a = (a & 0xFFFF) | (uint32_t)n << 16;
    return n == 1 && !dig(a, 0) ? a & ~0x100u : a;
}

static int cmp_mag(num a, num b) {
    if (LEN(a) != LEN(b)) return LEN(a) < LEN(b) ? -1 : 1;
    for (int i = LEN(a) - 1; i >= 0; i--) {
        int x = dig(a, i), y = dig(b, i);
        if (x != y) return x < y ? -1 : 1;
    }
    return 0;
}

static num add_mag(num a, num b) {  // |a| + |b| into a's register (a's sign)
    int n = (LEN(a) > LEN(b) ? LEN(a) : LEN(b)) + 1;
    if (n > MAXD) err = 1;
    bn_mac(BN_OPS(n, 0, 1, REG(a), REG(b), 0));
    wrote(a, n);
    if (!dig(a, n - 1)) n--;
    return (a & 0xFFFF) | (uint32_t)n << 16;
}

static num sub_mag(num a, num b) {  // |a| - |b| into a's register (|a| >= |b|)
    bn_msb(BN_OPS(LEN(a), 0, 1, REG(a), REG(b), 0));
    return trim(a);
}

__attribute__((noinline)) static num add(num a, num b) {  // a + b; consumes both
    num s;
    if (NEG(a) == NEG(b)) {
        s = add_mag(a, b);
        release(b);
    } else if (cmp_mag(a, b) >= 0) {
        s = sub_mag(a, b);
        release(b);
    } else {
        s = sub_mag(b, a);
        release(a);
    }
    return s;
}

__attribute__((noinline)) static num mul(num a, num b) {  // a * b in a new register; keeps both
    if (LEN(a) < LEN(b)) {
        num t = a;
        a = b;
        b = t;
    }
    int la = LEN(a), lb = LEN(b);
    num c = alloc(la + lb);
    if (la + lb > MAXD) {
        err = 1;
        return c;
    }
    uint32_t base = BN_OPS(la + 1, 0, 0, REG(c), REG(a), 0);
    for (int j = 0; j < lb; j++) {
        uint32_t d = dig(b, j);
        if (d) bn_mac(base | BN_OFF(j) | BN_M(d));
    }
    wrote(c, la + lb);
    return trim(c | (NEG(a) ^ NEG(b)) << 8);
}

typedef struct {
    num q, r;
} qr;

__attribute__((noinline)) static qr divmod(num a, num b) {  // |a| = q |b| + r: r in a's register, q new, b kept
    if (is_zero(b)) err = 1;
    if (err || cmp_mag(a, b) < 0) return (qr){alloc(1), a & ~0x100u};
    int la = LEN(a), lb = LEN(b), ra = REG(a);
    num q = alloc(la - lb + 1);
    int b1 = dig(b, lb - 1), den = lb >= 2 ? b1 * 10 + dig(b, lb - 2) : b1;
    uint32_t msb = BN_OPS(lb + 1, 0, 0, ra, REG(b), 0), mac = BN_OPS(lb + 1, 0, 1, ra, REG(b), 0);
    uint32_t setq = BN_OPS(1, 0, 0, REG(q), ONE, 1);
    for (int t = la - lb; t >= 0; t--) {
        int top = t + lb;  // the window a[t .. top]: its value / |b| is the digit (<= 9)
        int w = dig(a, top) * 10 + dig(a, top - 1);
        if (lb >= 2) w = w * 10 + dig(a, top - 2);
        int qd = (lb >= 3 ? w + 1 : w) / den;  // never below the digit, at most 2 above
        if (qd > 9) qd = 9;
        if (qd) {
            uint32_t o = BN_OFF(t);
            if (bn_msb(msb | o | BN_M(qd)))  // negative: add b back (at most twice; bounded
                do qd--;                      // anyway, should a register be corrupt)
                while (!bn_mac(mac | o) && qd > 0);
            if (qd) bn_mac(setq | o | BN_M(qd));
        }
    }
    wrote(q, la - lb + 1);
    return (qr){trim(q), trim(NUM(ra, 0, lb))};
}

__attribute__((noinline)) static num divide(num a, num b, int want_mod) {  // floor division / modulo; consumes both
    int sa = NEG(a), sb = NEG(b);
    qr d = divmod(a, b);
    if (sa != sb && !is_zero(d.r)) {
        d.q = add_mag(d.q, NUM(ONE, 0, 1));
        num r2 = sub_mag(b & ~0x100u, d.r);  // |b| - r, in b's register
        release(d.r);
        d.r = r2;
    } else {
        release(b);
    }
    release(want_mod ? d.q : d.r);
    num out = want_mod ? d.r : d.q;
    out = (out & ~0x100u) | (uint32_t)(want_mod ? sb : sa ^ sb) << 8;
    return is_zero(out) ? out & ~0x100u : out;
}

__attribute__((noinline)) static num gcd(num a, num b) {
    a &= ~0x100u;
    b &= ~0x100u;
    while (!err && !is_zero(b)) {
        qr d = divmod(a, b);
        release(d.q);
        a = b;
        b = d.r;
    }
    release(b);
    return a;
}

__attribute__((noinline)) static num power(num a, num e) {
    int x = 0;
    if (NEG(e) || LEN(e) > 3) err = 1;
    for (int i = LEN(e) - 1; i >= 0 && !err; i--) x = x * 10 + dig(e, i);
    release(e);
    num res = alloc(1);
    bn_mac(BN_OPS(1, 0, 1, REG(res), ONE, 1));
    wrote(res, 1);
    while (x && !err) {
        if (x & 1) {
            num t = mul(res, a);
            release(res);
            res = t;
        }
        x >>= 1;
        if (x) {
            num t = mul(a, a);
            release(a);
            a = t;
        }
    }
    release(a);
    return res;
}


// --- the expression: an operator-precedence (shunting-yard) loop, no recursion ---------------
// ops: + - (1), * / // % (2: 'd', 'f'), unary minus 'n' (3), ^ (4, right associative); '(' and 'g' (gcd's
// open bracket) mark groups

static num vals[16];
static char ops[16];
void _exit(int code) __attribute__((noreturn));

static int prec(int op) {  // 'f': // (floor), 'd': / (exact)
    return op == '^' ? 4 : op == 'n' ? 3 : op == '*' || op == 'f' || op == 'd' || op == '%' ? 2 : 1;
}

// Rationals: / is exact, so a value may be a fraction vals[i] / vden[i] (hasden[i]; the
// denominator positive and > 1, the fraction reduced). + - * / take fractions; // % ** gcd and
// decimal operands need integers.
static num vden[16];
static int hasden[16];

static num copy_of(num a) {
    num c = alloc(LEN(a));
    bn_mac(BN_OPS(LEN(a), 0, 1, REG(c), REG(a), 1));
    wrote(c, LEN(a));
    return NUM(REG(c), NEG(a), LEN(a));
}

__attribute__((noinline)) static void reduce(int i) {  // vals[i] / vden[i] in lowest terms
    num n = vals[i], d = vden[i];
    hasden[i] = 0;
    if (err) return;
    if (is_zero(n)) {
        release(d);
        return;
    }
    int neg = NEG(n);
    num g = gcd(copy_of(n), copy_of(d));
    qr x = divmod(n & ~0x100u, g);
    release(x.r);
    qr y = divmod(d, g);
    release(y.r);
    release(g);
    vals[i] = x.q | (uint32_t)neg << 8;
    if (LEN(y.q) == 1 && dig(y.q, 0) == 1) {
        release(y.q);
    } else {
        vden[i] = y.q;
        hasden[i] = 1;
    }
}

// (a / da) op (b / db) for op in + - * / (da, db absent where fa, fb are 0): the result's
// numerator and denominator into slot i, then reduced. Each operand register is released as
// soon as its last product is made (7 registers: a/b + c/d peaks at 5).
__attribute__((noinline)) static void frac_op(int op, int i, num a, num da, int fa, num b, num db,
                                              int fb) {
    num n, d;
    if (op == '*') {
        n = mul(a, b);
        release(a);
        release(b);
        if (fa && fb) {
            d = mul(da, db);
            release(da);
            release(db);
        } else {
            d = fa ? da : db;
        }
    } else if (op == 'd') {  // (a db) / (da b)
        if (is_zero(b)) err = 1;
        if (fb) {
            n = mul(a, db);
            release(a);
            release(db);
        } else {
            n = a;
        }
        if (fa) {
            d = mul(da, b);
            release(da);
            release(b);
        } else {
            d = b;
        }
        if (NEG(d)) {
            d &= ~0x100u;
            if (!is_zero(n)) n ^= 0x100u;
        }
    } else {  // + -: (a db +- b da) / (da db)
        num t1 = a, t2 = b;
        if (fb) {
            t1 = mul(a, db);
            release(a);
        }
        if (fa) {
            t2 = mul(b, da);
            release(b);
        }
        if (fa && fb) {
            d = mul(da, db);
            release(da);
            release(db);
        } else {
            d = fa ? da : db;
        }
        n = add(t1, op == '-' && !is_zero(t2) ? t2 ^ 0x100u : t2);
    }
    vals[i] = n;
    vden[i] = d;
    reduce(i);
}

// decimals: a value is the integer vals[i] scaled by 10^-vsc[i] (+ - * keep it exact; the other
// operators need integers)
static int vsc[16];

static num shift(num a, int k) {  // a * 10^k in a new register; consumes a
    int n = LEN(a) + k;
    if (n > MAXD) {
        err = 1;
        return a;
    }
    num c = alloc(n);
    bn_mac(BN_OPS(LEN(a), k, 1, REG(c), REG(a), 1));
    wrote(c, n);
    release(a);
    return NUM(REG(c), NEG(a), n);
}

static int apply(int op, int nv) {  // returns the new value count
    if (op == 'n') {
        num v = vals[nv - 1];
        vals[nv - 1] = is_zero(v) ? v : v ^ 0x100u;
        return nv;
    }
    if (nv < 2) {
        err = 1;
        return nv;
    }
    int sb = vsc[nv - 1], sa = vsc[nv - 2];
    int fb = hasden[nv - 1], fa = hasden[nv - 2];
    num b = vals[--nv], a = vals[nv - 1], r;
    int rational = op == 'd' || ((fa || fb) && (op == '+' || op == '-' || op == '*'));
    if ((fa || fb) && !rational) err = 1;  // // % ** gcd need integers
    if (rational && (sa || sb)) err = 1;   // fractions and decimals do not mix
    if (err) {
        r = a;
    } else if (rational) {
        frac_op(op, nv - 1, a, vden[nv - 1], fa, b, vden[nv], fb);
        vsc[nv - 1] = 0;
        return nv;
    } else if (op == '+' || op == '-') {
        if (sa < sb) a = shift(a, sb - sa), sa = sb;
        if (sb < sa) b = shift(b, sa - sb);
        r = add(a, op == '-' && !is_zero(b) ? b ^ 0x100u : b);
    } else if (op == '*') {
        r = mul(a, b);
        release(a);
        release(b);
        sa += sb;
    } else if (sa || sb) {
        err = 1;
        r = a;
    } else if (op == '^') {
        r = power(a, b);
    } else if (op == 'g') {
        r = gcd(a, b);
    } else {
        r = divide(a, b, op == '%');
    }
    hasden[nv - 1] = 0;
    vals[nv - 1] = r;
    vsc[nv - 1] = sa;
    return nv;
}

static int calc_eval(void) {  // prints the value (or "error") and a newline; 0 on success
    used = 0;
    err = 0;
    for (int r = 0; r < ONE; r++) {  // every call starts clean: no digits left by the last one
        bn_mac(BN_OPS(63, 0, 0, r, r, 1));
        dirty[r] = 0;
    }
    bn_ld(&one_ch + 1, BN_OPS(1, 0, 0, ONE, 0, 0));
    const char *s = calc_in;
    int operand = 1, nv = 0, no = 0;  // operand: an operand (or a prefix) comes next
    // the fast path: "A op B" (two plain numbers, one operator)
    int na = bn_scan(s);
    if (na && na <= MAXD) {
        const char *o = s + na;
        int c = o[0], two = (c == '*' || c == '/') && o[1] == c;
        int op = c == '*' && two ? '^' : c == '/' ? (two ? 'f' : 'd') : c;
        const char *t = o + 1 + two;
        int nb = bn_scan(t);
        if (nb && nb <= MAXD && !t[nb] &&
            (op == '+' || op == '-' || op == '*' || op == 'f' || op == 'd' || op == '%' ||
             op == '^')) {
            bn_ld(o, BN_OPS(na, 0, 0, 0, 0, 0));
            bn_ld(t + nb, BN_OPS(nb, 0, 0, 1, 0, 0));
            used = 3;
            dirty[0] = na;
            dirty[1] = nb;
            vals[0] = dig(NUM(0, 0, na), na - 1) ? NUM(0, 0, na) : trim(NUM(0, 0, na));
            vals[1] = dig(NUM(1, 0, nb), nb - 1) ? NUM(1, 0, nb) : trim(NUM(1, 0, nb));
            vsc[0] = vsc[1] = 0;
            hasden[0] = hasden[1] = 0;
            nv = apply(op, 2);
            if (err) goto fail;
            goto print;
        }
    }
    for (;;) {
        int c = *s;
        if (c == ' ') {
            s++;
            continue;
        }
        if (nv >= 15 || no >= 15) err = 1;
        if (err) break;
        if (operand) {
            if ((unsigned)(c - '0') < 10) {
                int n = bn_scan(s);  // (63: a longer run, over MAXD)
                const char *end = s + n;
                int frac = 0;
                if (*end == '.' && (unsigned)(end[1] - '0') < 10) {  // a decimal: digits . digits
                    static char dbuf[MAXD + 2];
                    frac = bn_scan(end + 1);
                    if (n + frac > MAXD) err = 1;
                    if (err) break;
                    for (int i = 0; i < n; i++) dbuf[i] = s[i];
                    for (int i = 0; i < frac; i++) dbuf[n + i] = end[1 + i];
                    s = end + 1 + frac;
                    n += frac;
                    end = dbuf + n;
                } else {
                    s = end;
                }
                num a = alloc(n);
                if (n > MAXD) err = 1;
                bn_ld(end, BN_OPS(n, 0, 0, REG(a), 0, 0));
                wrote(a, n);
                vsc[nv] = frac;
                hasden[nv] = 0;
                vals[nv++] = dig(a, n - 1) ? a : trim(a);
                operand = 0;
            } else if (c == '(') {
                ops[no++] = '(';
                s++;
            } else if (c == '-' || c == '+') {
                if (c == '-') ops[no++] = 'n';
                s++;
            } else if (c == 'g' && s[1] == 'c' && s[2] == 'd' && s[3] == '(') {
                ops[no++] = 'g';
                s += 4;
            } else {
                err = 1;
            }
            continue;
        }
        if (c == ')' || c == ',') {
            while (no && ops[no - 1] != '(' && ops[no - 1] != 'g') nv = apply(ops[--no], nv);
            if (!no) {
                err = 1;
            } else if (c == ')') {
                if (ops[--no] == 'g') nv = apply('g', nv);
            } else if (ops[no - 1] != 'g') {
                err = 1;
            } else {
                operand = 1;
            }
            s++;
            continue;
        }
        if (!c) break;
        int op = c, len = 1;
        if (c == '*' && s[1] == '*') {
            op = '^';
            len = 2;
        } else if (c == '/') {
            op = s[1] == '/' ? 'f' : 'd';
            len = s[1] == '/' ? 2 : 1;
        } else if (c != '+' && c != '-' && c != '*' && c != '/' && c != '%' && c != '^') {
            err = 1;
            continue;
        }
        int pr = prec(op);
        while (no && ops[no - 1] != '(' && ops[no - 1] != 'g' &&
               (prec(ops[no - 1]) > pr || (prec(ops[no - 1]) == pr && op != '^')))
            nv = apply(ops[--no], nv);
        ops[no++] = op;
        s += len;
        operand = 1;
    }
    while (!err && no) {
        int op = ops[--no];
        if (op == '(' || op == 'g')
            err = 1;
        else
            nv = apply(op, nv);
    }
    if (err || nv != 1 || operand) {
    fail:
        for (const char *m = err == 2 ? "error (too complex: split it)\n" : "error\n"; *m; m++)
            rv_putchar(*m);
        return 1;
    }
print:;
    num v = vals[0];
    int sc = vsc[0], n = LEN(v);
    if (NEG(v)) rv_putchar('-');
    if (!sc) {
        bn_print(REG(v), n);
    } else {  // digits with the point, no trailing zeros after it (0.5, 12, -0.004)
        int z = 0;
        while (z < sc && !dig(v, z)) z++;
        if (n <= sc) rv_putchar('0');
        for (int i = n > sc ? n - 1 : sc - 1; i >= z; i--) {
            if (i == sc - 1) rv_putchar('.');
            rv_putchar('0' + (i < n ? dig(v, i) : 0));
        }
    }
    if (hasden[0]) {  // a reduced fraction: num/den
        rv_putchar('/');
        bn_print(REG(vden[0]), LEN(vden[0]));
    }
    rv_putchar('\n');
    return 0;
}

#ifdef BRIDGE
// The P3 bridge: each run (GO, after </tool_call>) reads the tool call's text from the text
// buffer (the call's i-th token at 0x8000 + 16 i), takes its first parameter's value (what
// follows the second '>' up to the next '<', newlines dropped) into calc_in, prints the tool
// response and parks (exit ecall) until the next call:
//   \n<tool_response>\nVALUE\n</tool_response>\n<think>\n\n</think>\n   (+ the HALT echo \n)
#define TEXT ((const volatile unsigned char *)0x8000)
#define ROWS 80
enum { TOOL_RESPONSE = 256, TOOL_RESPONSE_END, THINK, THINK_END };

static void put(int c) { rv_ecall2(1, c, 0, 0); }

static int gather(void) {
    int n = 0, gt = 0, i;
    for (i = 1; i <= ROWS; i++) {
        const volatile unsigned char *r = TEXT + 16 * i;
        int c = r[0];
        if (!c) break;
        for (int k = 0; k < 16 && c; c = ++k < 16 ? r[k] : 0) {
            if (gt < 2) {
                gt += c == '>';
            } else if (c == '<') {
                calc_in[n] = 0;
                return n;
            } else if (c != '\n' && n < (int)sizeof calc_in - 1) {
                calc_in[n++] = (char)c;
            }
        }
    }
    calc_in[n] = 0;
    if (gt < 2) return -1;
    return i > ROWS ? -2 : n;  // -2: no closing '<' within the capture window (truncated)
}

int main(void) {
    for (;;) {
        rv_ecall2(93, 0, 0, 0);  // park: HALT; the next GO resumes here
        int n = gather();
        put('\n');
        put(TOOL_RESPONSE);
        put('\n');
        if (n < 0) {
            for (const char *m = n == -2 ? "error (too long: split it)\n" : "error\n"; *m; m++)
                put(*m);
        } else {
            calc_eval();
        }
        put(TOOL_RESPONSE_END);
        put('\n');
        put(THINK);
        put('\n');
        put('\n');
        put(THINK_END);
        put('\n');
    }
}
#else
int main(void) { return calc_eval(); }
#endif
