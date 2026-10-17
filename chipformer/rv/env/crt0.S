# Minimal C runtime: the host (or qemu) sets sp; main's return value becomes the exit code.
    .section .text.init
    .globl _start
_start:
    call main
    li a7, 93
    ecall
