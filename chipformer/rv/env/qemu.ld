/* The same layout at addresses qemu-riscv32 (user mode) can map (the heap inside .bss, so it is
   mapped too). */
ENTRY(_start)
SECTIONS
{
  . = 0x100000;
  .text : { *(.text.init) *(.text .text.*) }
  . = 0x180000;
  .rodata : {
    *(.rodata .rodata.* .srodata .srodata.*)
    *(.init_array .init_array.* .fini_array .fini_array.* .eh_frame .eh_frame_hdr)
  }
  . = 0x200000;
  .data : { *(.data .data.* .sdata .sdata.*) }
  .tdata : { __tls_base = .; *(.tdata .tdata.*) }
  .tbss : { *(.tbss .tbss.*) }
  .bss : {
    . = . + SIZEOF(.tbss);
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __heap_start = .;
    . = . + 0x10000;
    __heap_end = .;
  }
  _end = .;
}
