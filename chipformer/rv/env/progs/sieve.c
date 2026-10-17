static unsigned char comp[2000];

int main(void) {
    int count = 0;
    for (int i = 2; i < 2000; i++) {
        if (comp[i]) continue;
        count++;
        for (int j = i + i; j < 2000; j += i) comp[j] = 1;
    }
    return count; /* 303 */
}
