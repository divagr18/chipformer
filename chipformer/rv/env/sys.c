// picolibc stdio and exit on the chipformer RV32 machine.
#include <stdio.h>
#include <stdlib.h>
#include "rt.h"

static int rv_putc(char c, FILE *f) {
    (void)f;
    rv_putchar(c);
    return (unsigned char)c;
}

static FILE __stdio = FDEV_SETUP_STREAM(rv_putc, NULL, NULL, _FDEV_SETUP_WRITE);
FILE *const stdin = &__stdio;
FILE *const stdout = &__stdio;
FILE *const stderr = &__stdio;

void _exit(int code) {
    for (;;) rv_ecall2(93, code, 0, 0);
}
