/* The same layout at addresses qemu-riscv32 (user mode) can map. */
ENTRY(_start)
SECTIONS
{
  . = 0x100000;
  .text : { *(.text.init) *(.text .text.*) }
  . = 0x200000;
  .data : { *(.rodata .rodata.* .srodata .srodata.*) *(.data .data.* .sdata .sdata.*) }
  .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) }
  _end = .;
}
