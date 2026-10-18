# MicroPython for the chipformer RV32 machine (RV32IM + Zba, ROM / DROM / RAM map, putchar and
# exit ecalls), based on ports/minimal. Built inside a MicroPython tree by build_mpy.sh:
#   make CONFIG=basic SCRIPT=/abs/script.py MAP=rom|qemu BUILD=... [HOT=...] [COPT=...]
CONFIG ?= basic
MAP ?= rom
STACK_KB ?= 6
BUILD ?= build-$(CONFIG)-$(MAP)
include ../../py/mkenv.mk
QSTR_DEFS = qstrdefsport.h
MICROPY_ROM_TEXT_COMPRESSION ?= 1
include $(TOP)/py/py.mk
CROSS_COMPILE = riscv64-unknown-elf-
ENV ?= $(abspath ../../../env)
PICO = /usr/lib/picolibc/riscv64-unknown-elf
MARCH ?= rv32im_zba
ARCH = -march=$(MARCH) -mabi=ilp32
INC += -I. -I$(TOP) -I$(BUILD) -I$(ENV) -isystem $(PICO)/include
CFLAGS += $(INC) -Wall -std=gnu99 $(ARCH) $(COPT) -DRV_CFG_$(CONFIG)=1 -DHEAP_KB=0
CFLAGS += -DSTACK_BYTES=$(shell echo $$(( $(STACK_KB) * 1024 - 512 ))) $(if $(filter qemu,$(MAP)),-DQEMU)
CFLAGS += $(OPTFLAGS) -DNDEBUG -fdata-sections -ffunction-sections
OPTFLAGS ?= -Os
CSUPEROPT = $(OPTFLAGS)
AFLAGS = $(ARCH)
LIBGCC = $(shell $(CC) $(ARCH) -print-libgcc-file-name)
LIBS = -Wl,--start-group $(PICO)/lib/release/rv32im/ilp32/libc.a \
  $(PICO)/lib/release/rv32im/ilp32/libm.a $(LIBGCC) -Wl,--end-group
LD = $(CC)
LDFLAGS = $(ARCH) -nostdlib -nostartfiles -static -Wl,--gc-sections -Wl,-Map=$(BUILD)/firmware.map \
  -T$(BUILD)/mpy.ld
SRC_C = main.c shared/runtime/gchelper_native.c shared/runtime/stdout_helpers.c
SRC_S = shared/runtime/gchelper_rv32i.s
HOT ?=
HOTFLAGS ?= -O3
OBJ += $(PY_CORE_O)
OBJ += $(addprefix $(BUILD)/, $(SRC_C:.c=.o))
OBJ += $(addprefix $(BUILD)/, $(SRC_S:.s=.o))
OBJ += $(BUILD)/crt0.o $(BUILD)/fastlib.o $(BUILD)/sys.o
$(foreach f,$(HOT),$(eval $(BUILD)/$(f).o: CFLAGS += $(HOTFLAGS)))
$(BUILD)/main.o: $(BUILD)/script.h
$(BUILD)/script.h: $(SCRIPT)
	$(Q)mkdir -p $(BUILD)
	$(Q)python3 -c "import sys; s = open(sys.argv[1], 'rb').read(); print('static const char script[] = {' + ','.join(str(b) for b in s) + ',0};')" $< > $@
$(BUILD)/crt0.o: $(ENV)/crt0.S
	$(Q)mkdir -p $(BUILD)
	$(Q)$(CC) $(ARCH) -c -o $@ $<
$(BUILD)/sys.o: $(ENV)/sys.c
	$(Q)mkdir -p $(BUILD)
	$(Q)$(CC) $(ARCH) -O2 -I$(ENV) -isystem $(PICO)/include $(if $(filter qemu,$(MAP)),-DQEMU) -c -o $@ $<
$(BUILD)/fastlib.o: fastlib.c
	$(Q)mkdir -p $(BUILD)
	$(Q)$(CC) $(ARCH) -O2 -fno-builtin -fno-tree-loop-distribute-patterns -c -o $@ $<
$(BUILD)/mpy.ld: $(ENV)/$(MAP).ld
	$(Q)mkdir -p $(BUILD)
	$(Q)sed 's/__heap_end = 0x7000;.*/__heap_end = $(shell printf 0x%x $$(( 0x8000 - $(STACK_KB) * 1024 )));/' $< > $@
all: $(BUILD)/firmware.elf
$(BUILD)/firmware.elf: $(OBJ) $(BUILD)/mpy.ld
	$(ECHO) "LINK $@"
	$(Q)$(LD) $(LDFLAGS) -o $@ $(OBJ) $(LIBS)
	$(Q)$(SIZE) -A $@ | egrep '^\.(text|rodata|data|bss|tdata|tbss)|Total'
include $(TOP)/py/mkrules.mk
