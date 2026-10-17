static int a[64];

int main(void) {
    unsigned x = 12345;
    for (int i = 0; i < 64; i++) {
        x = x * 1103515245u + 12345u; /* libgcc __mulsi3 on rv32i */
        a[i] = (int)(x >> 8) - (1 << 22);
    }
    for (int i = 1; i < 64; i++) {
        int v = a[i], j = i - 1;
        while (j >= 0 && a[j] > v) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = v;
    }
    unsigned h = 0;
    for (int i = 0; i < 64; i++) {
        if (i && a[i - 1] > a[i]) return 255;
        h = (h << 5) - h + (unsigned)a[i];
    }
    return (int)h;
}
