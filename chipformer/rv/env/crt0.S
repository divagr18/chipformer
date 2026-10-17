# Minimal C runtime: the host (or qemu) sets sp, and the snapshot holds .data and a zeroed .bss
# (qemu loads them from the ELF); tp points at the TLS block; main's return value goes to exit
# (ecall a7 = 93).
    .section .text.init
    .globl _start
_start:
    .weak __tls_base
    la tp, __tls_base
    call main
    call exit
1:  j 1b
