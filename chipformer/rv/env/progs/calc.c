// calc: the P3 firmware. Each run (GO, after </tool_call>) reads the tool call's text from the
// text buffer, evaluates the expression of its first parameter exactly (integers of any size:
// + - * and parentheses, unary minus, spaces and thousands commas ignored), prints the tool
// response and parks (exit ecall) until the next call.
//
//   \n<tool_response>\n<value>\n</tool_response>\n<think>\n\n</think>\n   (+ the HALT echo \n)
#include "rt.h"

#define TEXT ((const volatile unsigned char *)0x8000)  // row i at TEXT + 16 i, i = 1 .. 96
#define ROWS 96
#define MAXD 400  // digits of a number

enum { TOOL_RESPONSE = 256, TOOL_RESPONSE_END, THINK, THINK_END };

static void put(int c) { rv_ecall2(1, c, 0, 0); }
static void park(void) { rv_ecall2(93, 0, 0, 0); }

static char text[ROWS * 16 + 1];

static int gather(void) {
    int n = 0;
    for (int i = 1; i <= ROWS; i++) {
        const volatile unsigned char *r = TEXT + 16 * i;
        if (!r[0]) break;
        for (int k = 0; k < 16 && r[k]; k++) text[n++] = (char)r[k];
    }
    text[n] = 0;
    return n;
}

typedef struct {
    int neg, n;  // n digits, little-endian; n = 0 is zero
    unsigned char d[MAXD];
} big;

static const unsigned char DIV10[100] = {
    0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7,7,7,
    8,8,8,8,8,8,8,8,8,8, 9,9,9,9,9,9,9,9,9,9};

static int err;
static const char *s;
static big pool[24];
static int top;

static big *fresh(void) {
    if (top >= 24) { err = 1; top = 0; }
    big *b = &pool[top++];
    b->neg = 0;
    b->n = 0;
    return b;
}

static void trim(big *a) {
    while (a->n && !a->d[a->n - 1]) a->n--;
    if (!a->n) a->neg = 0;
}

static int cmpmag(const big *a, const big *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (int i = a->n - 1; i >= 0; i--)
        if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
    return 0;
}

static void addmag(big *r, const big *a, const big *b) {
    int n = a->n > b->n ? a->n : b->n, c = 0;
    if (n + 1 > MAXD) { err = 1; return; }
    for (int i = 0; i < n; i++) {
        int t = (i < a->n ? a->d[i] : 0) + (i < b->n ? b->d[i] : 0) + c;
        c = t >= 10;
        r->d[i] = (unsigned char)(t - 10 * c);
    }
    r->d[n] = (unsigned char)c;
    r->n = n + 1;
    trim(r);
}

static void submag(big *r, const big *a, const big *b) {  // |a| >= |b|
    int c = 0;
    for (int i = 0; i < a->n; i++) {
        int t = a->d[i] - (i < b->n ? b->d[i] : 0) - c;
        c = t < 0;
        r->d[i] = (unsigned char)(t + 10 * c);
    }
    r->n = a->n;
    trim(r);
}

static big *add(const big *a, const big *b, int negb) {
    big *r = fresh();
    int bn = b->neg ^ negb;
    if (a->neg == bn) {
        addmag(r, a, b);
        r->neg = a->neg;
    } else if (cmpmag(a, b) >= 0) {
        submag(r, a, b);
        r->neg = a->neg;
    } else {
        submag(r, b, a);
        r->neg = bn;
    }
    trim(r);
    return r;
}

static big *mul(const big *a, const big *b) {
    big *r = fresh();
    int n = a->n + b->n;
    if (n > MAXD) { err = 1; return r; }
    for (int i = 0; i < n; i++) r->d[i] = 0;
    for (int i = 0; i < a->n; i++) {
        int c = 0, ai = a->d[i];
        if (!ai) continue;
        for (int j = 0; j < b->n; j++) {
            int t = r->d[i + j] + ai * b->d[j] + c;
            c = DIV10[t];
            r->d[i + j] = (unsigned char)(t - 10 * c);
        }
        for (int k = i + b->n; c; k++) {
            int t = r->d[k] + c;
            c = t >= 10;
            r->d[k] = (unsigned char)(t - 10 * c);
        }
    }
    r->n = n;
    r->neg = a->neg ^ b->neg;
    trim(r);
    return r;
}

static void skip(void) {
    while (*s == ' ' || *s == '\t' || *s == '\n') s++;
}

static big *expr(void);

static big *factor(void) {
    skip();
    if (*s == '-') { s++; big *a = factor(); a->neg = a->n ? !a->neg : 0; return a; }
    if (*s == '+') { s++; return factor(); }
    if (*s == '(') {
        s++;
        big *a = expr();
        skip();
        if (*s == ')') s++; else err = 1;
        return a;
    }
    if (*s < '0' || *s > '9') { err = 1; return fresh(); }
    char tmp[MAXD];
    int n = 0;
    while ((*s >= '0' && *s <= '9') || *s == ',' || *s == '_') {
        if (*s != ',' && *s != '_') {
            if (n >= MAXD) { err = 1; break; }
            tmp[n++] = *s;
        }
        s++;
    }
    big *a = fresh();
    for (int i = 0; i < n; i++) a->d[i] = (unsigned char)(tmp[n - 1 - i] - '0');
    a->n = n;
    trim(a);
    return a;
}

static int is_times(void) {
    if (*s == '*' || *s == 'x') return 1;
    if ((unsigned char)s[0] == 0xC3 && (unsigned char)s[1] == 0x97) { s++; return 1; }  // ×
    return 0;
}

static big *term(void) {
    big *a = factor();
    for (;;) {
        skip();
        if (!is_times()) return a;
        s++;
        a = mul(a, factor());
    }
}

static big *expr(void) {
    big *a = term();
    for (;;) {
        skip();
        if (*s == '+') { s++; a = add(a, term(), 0); }
        else if (*s == '-') { s++; a = add(a, term(), 1); }
        else return a;
    }
}

static const char *find(const char *h, const char *n) {
    for (; *h; h++) {
        int i = 0;
        while (n[i] && h[i] == n[i]) i++;
        if (!n[i]) return h;
    }
    return 0;
}

static void puts_(const char *p) {
    while (*p) put((unsigned char)*p++);
}

static void handle(void) {
    gather();
    err = 0;
    top = 0;
    big *v = 0;
    const char *p = find(text, "<parameter=");
    if (p && (p = find(p, ">"))) {
        s = p + 1;
        v = expr();
        skip();
        if (*s != '<' && *s != 0) err = 1;
    } else {
        err = 1;
    }
    put('\n');
    put(TOOL_RESPONSE);
    put('\n');
    if (err || !v) {
        puts_("error");
    } else if (!v->n) {
        put('0');
    } else {
        if (v->neg) put('-');
        for (int i = v->n - 1; i >= 0; i--) put('0' + v->d[i]);
    }
    put('\n');
    put(TOOL_RESPONSE_END);
    put('\n');
    put(THINK);
    put('\n');
    put('\n');
    put(THINK_END);
    put('\n');
}

int main(void) {
    for (;;) {
        park();
        handle();
    }
}

// freestanding: no libc (gcc may still call these)
void exit(int code) {
    for (;;) rv_ecall2(93, code, 0, 0);
}

void *memset(void *d, int c, unsigned long n) {
    unsigned char *p = d;
    while (n--) *p++ = (unsigned char)c;
    return d;
}

void *memcpy(void *d, const void *s_, unsigned long n) {
    unsigned char *p = d;
    const unsigned char *q = s_;
    while (n--) *p++ = *q++;
    return d;
}
