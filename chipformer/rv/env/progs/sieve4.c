#include <stdio.h>
#define N 10000
static unsigned char comp[N];

int main(void) {
    int count = 0;
    for (int i = 2; i < N; i++) {
        if (!comp[i]) {
            count++;
            for (int j = i * i; j < N; j += i) comp[j] = 1;
        }
    }
    printf("primes below %d: %d\n", N, count);
    return 0;
}
