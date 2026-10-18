// calc: exact decimal integer expressions on the chipformer digit file (bn.h).
//
// Input: the NUL-terminated expression in calc_in (RAM 0x100; the host writes it into the
// snapshot). Grammar (Python's precedence and semantics):
//   expr = term {(+|-) term}      term = factor {(*|/|//|%) factor}
//   factor = (-|+) factor | power  power = atom [(**|^) factor]
//   atom = digits | ( expr ) | gcd ( expr , expr )
// / and // are floor division, % the matching modulo (the divisor's sign), ** / ^ a power with
// a small non-negative exponent. Output: the value in decimal and "\n", or "error\n" (division by
// zero, a value over MAXD digits, a syntax error).
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
    err = 1;
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
            if (bn_msb(msb | o | BN_M(qd)))  // negative: add b back
                do qd--;
                while (!bn_mac(mac | o));
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
// ops: + - (1), * / % (2), unary minus 'n' (3), ^ (4, right associative); '(' and 'g' (gcd's
// open bracket) mark groups

static num vals[16];
static char ops[16];
void _exit(int code) __attribute__((noreturn));

static int prec(int op) {
    return op == '^' ? 4 : op == 'n' ? 3 : op == '*' || op == '/' || op == '%' ? 2 : 1;
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
    num b = vals[--nv], a = vals[nv - 1], r;
    if (op == '+' || op == '-') {
        r = add(a, op == '-' && !is_zero(b) ? b ^ 0x100u : b);
    } else if (op == '*') {
        r = mul(a, b);
        release(a);
        release(b);
    } else if (op == '^') {
        r = power(a, b);
    } else if (op == 'g') {
        r = gcd(a, b);
    } else {
        r = divide(a, b, op == '%');
    }
    vals[nv - 1] = r;
    return nv;
}

int main(void) {
    bn_ld(&one_ch + 1, BN_OPS(1, 0, 0, ONE, 0, 0));
    const char *s = calc_in;
    int operand = 1, nv = 0, no = 0;  // operand: an operand (or a prefix) comes next
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
                s += n;
                num a = alloc(n);
                if (n > MAXD) err = 1;
                bn_ld(s, BN_OPS(n, 0, 0, REG(a), 0, 0));
                wrote(a, n);
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
        } else if (c == '/' && s[1] == '/') {
            len = 2;
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
        for (const char *m = "error\n"; *m; m++) rv_putchar(*m);
        _exit(1);
    }
    num v = vals[0];
    if (NEG(v)) rv_putchar('-');
    bn_print(REG(v), LEN(v));
    rv_putchar('\n');
    _exit(0);
}
