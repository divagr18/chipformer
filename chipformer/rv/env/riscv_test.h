// riscv-tests environment for the chipformer RV32 machine: no CSRs, no traps; a test ends with
// the Linux-style exit ecall (a7 = 93, a0 = 0 on pass, the failing TESTNUM otherwise), which
// qemu-riscv32 (user mode) and the circuit both understand.
#ifndef _ENV_CHIPFORMER_RV_H
#define _ENV_CHIPFORMER_RV_H

#define RVTEST_RV64U .macro init; .endm
#define RVTEST_RV32U .macro init; .endm
#define TESTNUM gp
#define RVTEST_CODE_BEGIN .text; .globl _start; _start: init;
#define RVTEST_CODE_END unimp
#define RVTEST_PASS li a0, 0; li a7, 93; ecall
#define RVTEST_FAIL mv a0, TESTNUM; li a7, 93; ecall
#define RVTEST_DATA_BEGIN .data; .align 4; .global begin_signature; begin_signature:
#define RVTEST_DATA_END .align 4; .global end_signature; end_signature:
#define EXTRA_DATA

#endif
