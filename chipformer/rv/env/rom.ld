/* The chipformer RV32 address map: RAM at 0 (data / rodata / bss from 0x100, stack below
   0x4000), code in ROM at 0x10000. */
ENTRY(_start)
MEMORY
{
  RAM (rw) : ORIGIN = 0x100, LENGTH = 0x3F00
  ROM (rx) : ORIGIN = 0x10000, LENGTH = 64K
}
SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > ROM
  .data : { *(.rodata .rodata.* .srodata .srodata.*) *(.data .data.* .sdata .sdata.*) } > RAM
  .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) } > RAM
  _end = .;
}
