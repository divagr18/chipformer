#include <stdio.h>

static int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

int main(void) {
    volatile int n = 25;
    printf("fib(25)=%d\n", fib(n));
    return 0;
}
