#include <stdio.h>
static int g[81];
static long steps;

static int ok(int p, int v) {
    int r = p / 9, c = p % 9, br = r / 3 * 3, bc = c / 3 * 3;
    for (int i = 0; i < 9; i++) {
        if (g[r * 9 + i] == v || g[i * 9 + c] == v) return 0;
        if (g[(br + i / 3) * 9 + bc + i % 3] == v) return 0;
    }
    return 1;
}

static int solve(int p) {
    while (p < 81 && g[p]) p++;
    if (p == 81) return 1;
    for (int v = 1; v <= 9; v++) {
        steps++;
        if (ok(p, v)) {
            g[p] = v;
            if (solve(p + 1)) return 1;
            g[p] = 0;
        }
    }
    return 0;
}

/* the Wikipedia example puzzle */
static const char puz[] =
    "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";

int main(void) {
    for (int i = 0; i < 81; i++) g[i] = puz[i] == '.' ? 0 : puz[i] - '0';
    int s = solve(0);
    printf("solved=%d steps=%ld row0=", s, steps);
    for (int i = 0; i < 9; i++) putchar('0' + g[i]);
    putchar('\n');
    return 0;
}
