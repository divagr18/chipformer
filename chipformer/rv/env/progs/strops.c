static char buf[64];
static const char src[] = "risc-v inside qwen, exactly";

static int len(const char *s) {
    int n = 0;
    while (s[n]) n++;
    return n;
}

int main(void) {
    int n = len(src);
    for (int i = 0; i < n; i++) buf[i] = src[n - 1 - i];
    buf[n] = 0;
    unsigned short h[8] = {0};
    for (int i = 0; i < n; i++) h[i & 7] = (unsigned short)((h[i & 7] << 3) - h[i & 7] + (unsigned char)buf[i]);
    int s = 0;
    for (int i = 0; i < 8; i++) s += (short)h[i];
    return s + n;
}
