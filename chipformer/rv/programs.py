"""Small RV32I test programs (assembly) for the reference and the circuit."""

from chipformer.rv.asm import assemble

LOOP = """
_start: li a0, 0
        li t0, 10
loop:   addi a0, a0, 3
        addi t0, t0, -1
        bnez t0, loop
        li a7, 93
        ecall
"""

MEMCPY = """
_start: la a0, dst
        la a1, src
        li a2, 13
bytes:  lbu t0, 0(a1)
        sb t0, 0(a0)
        addi a0, a0, 1
        addi a1, a1, 1
        addi a2, a2, -1
        bnez a2, bytes
        la a0, wdst
        la a1, src
        li a2, 3
words:  lw t0, 0(a1)
        sw t0, 0(a0)
        addi a0, a0, 4
        addi a1, a1, 4
        addi a2, a2, -1
        bnez a2, words
        la t1, dst
        lw a0, 4(t1)
        lw a1, 8(t1)
        li a7, 93
        ecall
.data
src:    .word 0x64636261, 0x68676665, 0xF00DBEEF, 0x80
dst:    .space 16
wdst:   .space 12
"""

FIB = """
_start: li a0, 25
        li t0, 0
        li t1, 1
fib:    beqz a0, done
        add t2, t0, t1
        mv t0, t1
        mv t1, t2
        addi a0, a0, -1
        j fib
done:   mv a0, t0
        li a7, 93
        ecall
"""

CALL = """
_start: li a0, 7
        call fact
        mv s1, a0
        li a0, 12
        call sum
        li a7, 93
        ecall
fact:   addi sp, sp, -16
        sw ra, 12(sp)
        sw s0, 8(sp)
        mv s0, a0
        li t0, 2
        blt a0, t0, fact1
        addi a0, a0, -1
        call fact
        mv t1, a0
        li a0, 0
mloop:  beqz s0, mdone       # a0 = t1 * s0 by repeated addition (no M in P0)
        add a0, a0, t1
        addi s0, s0, -1
        j mloop
mdone:  j fact2
fact1:  li a0, 1
fact2:  lw ra, 12(sp)
        lw s0, 8(sp)
        addi sp, sp, 16
        ret
sum:    addi sp, sp, -16      # recursive sum 1..a0
        sw ra, 12(sp)
        sw a0, 8(sp)
        beqz a0, sum0
        addi a0, a0, -1
        call sum
        lw t0, 8(sp)
        add a0, a0, t0
        j sum1
sum0:   li a0, 0
sum1:   lw ra, 12(sp)
        addi sp, sp, 16
        ret
"""

SIGNEXT = """
_start: la t0, vals
        lb a0, 0(t0)
        lbu a1, 0(t0)
        lh a2, 2(t0)
        lhu a3, 2(t0)
        lb a4, 5(t0)
        lh a5, 6(t0)
        li t1, -2
        sh t1, 8(t0)
        sb t1, 11(t0)
        lw a6, 8(t0)
        lui a7, 0xABCDE
        srai s2, a7, 7
        srli s3, a7, 7
        slli s4, a7, 3
        slt s5, a7, zero
        sltu s6, a7, zero
        sltiu s7, a0, -1
        xori s8, a1, 0x7FF
        ori s9, a0, 0x123
        andi s10, a2, -256
        sub s11, a2, a7
        li a7, 93
        ecall
.data
vals:   .word 0x8F7F80FF, 0x7F7F0080, 0x11223344
"""

ALL = {"loop": LOOP, "memcpy": MEMCPY, "fib": FIB, "call": CALL, "signext": SIGNEXT}


def program(name: str):
    return assemble(ALL[name])
