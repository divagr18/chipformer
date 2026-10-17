/* mini-gmp: parse two 20-digit numbers, multiply them, print the 40-digit product. */
#include <stdio.h>
#include "mini-gmp.h"
static char buf[64];

int main(void) {
    mpz_t a, b, c;
    mpz_init(a);
    mpz_init(b);
    mpz_init(c);
    mpz_set_str(a, "12345678901234567890", 10);
    mpz_set_str(b, "98765432109876543211", 10);
    mpz_mul(c, a, b);
    mpz_get_str(buf, 10, c);
    printf("mul=%s\n", buf);
    return 0;
}
