/* The chipformer RV32 address map (P1): RAM at 0 (data / rodata / TLS / bss / heap from 0x100,
   the stack below 0x8000), code in ROM at 0x10000 (128 KB). crt0 points tp at __tls_base (the
   single thread's TLS block: .tdata then .tbss). */
ENTRY(_start)
MEMORY
{
  RAM (rw) : ORIGIN = 0x100, LENGTH = 0x7F00
  ROM (rx) : ORIGIN = 0x10000, LENGTH = 128K
}
SECTIONS
{
  .text : { *(.text.init) *(.text .text.*) } > ROM
  .data : { *(.rodata .rodata.* .srodata .srodata.*) *(.data .data.* .sdata .sdata.*) } > RAM
  .tdata : { __tls_base = .; *(.tdata .tdata.*) } > RAM
  .tbss : { *(.tbss .tbss.*) } > RAM
  .bss : {
    . = . + SIZEOF(.tbss);  /* (.tbss takes no address space of its own) */
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __heap_start = .;
  } > RAM
  __heap_end = 0x7000;  /* 4 KB of stack above the heap */
  _end = __heap_start;
}
