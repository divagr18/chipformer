// System calls of the chipformer RV32 machine: putchar is ecall a7 = 1, exit is a7 = 93.
// Built with -DQEMU, putchar becomes write(1, &c, 1) (a7 = 64) for qemu-riscv32 user mode.
#pragma once

static inline long rv_ecall2(long n, long a0, long a1, long a2) {
    register long r_a0 __asm__("a0") = a0, r_a1 __asm__("a1") = a1, r_a2 __asm__("a2") = a2;
    register long r_a7 __asm__("a7") = n;
    __asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a1), "r"(r_a2), "r"(r_a7) : "memory");
    return r_a0;
}

static inline void rv_putchar(char c) {
#ifdef QEMU
    rv_ecall2(64, 1, (long)&c, 1);
#else
    rv_ecall2(1, (unsigned char)c, 0, 0);
#endif
}

static inline void rv_mark(int id) { (void)id; }
