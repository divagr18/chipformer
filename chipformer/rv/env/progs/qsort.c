#include <stdio.h>
#include <stdlib.h>
#define N 1000
static int a[N];
static unsigned s = 12345;

static int rnd(void) {
    s = s * 1103515245u + 12345u;
    return (s >> 8) & 0xffffff;
}

static int cmp(const void *x, const void *y) {
    int p = *(const int *)x, q = *(const int *)y;
    return (p > q) - (p < q);
}

int main(void) {
    for (int i = 0; i < N; i++) a[i] = rnd();
    qsort(a, N, sizeof a[0], cmp);
    long err = 0;
    unsigned h = 0;
    for (int i = 0; i < N; i++) {
        if (i && a[i - 1] > a[i]) err++;
        h = h * 31u + (unsigned)a[i];
    }
    printf("qsort 1000: err=%ld min=%d max=%d hash=%08x\n", err, a[0], a[N - 1], h);
    return 0;
}
