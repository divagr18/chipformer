#include <stdio.h>
static const char msg[] = "The quick brown fox jumps over the lazy dog";

int main(void) {
    unsigned crc = 0xFFFFFFFFu;
    for (const char *p = msg; *p; p++) {
        crc ^= (unsigned char)*p;
        for (int k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & -(crc & 1u));
    }
    printf("crc32=%08x\n", ~crc); /* 414fa339 */
    return 0;
}
