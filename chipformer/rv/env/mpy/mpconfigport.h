#include <stdint.h>
#if defined(RV_CFG_min)
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)
#define MICROPY_FLOAT_IMPL       (MICROPY_FLOAT_IMPL_NONE)
#elif defined(RV_CFG_basic)
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_BASIC_FEATURES)
#define MICROPY_FLOAT_IMPL       (MICROPY_FLOAT_IMPL_DOUBLE)
#elif defined(RV_CFG_extra)
#ifndef MICROPY_PY_UCTYPES
#define MICROPY_PY_UCTYPES       (0)
#endif
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_EXTRA_FEATURES)
#define MICROPY_FLOAT_IMPL       (MICROPY_FLOAT_IMPL_DOUBLE)
#endif
#ifdef RV_NOCOMP
#ifndef MICROPY_ENABLE_COMPILER
#define MICROPY_ENABLE_COMPILER           (0)
#endif
#else
#ifndef MICROPY_ENABLE_COMPILER
#define MICROPY_ENABLE_COMPILER           (1)
#endif
#endif
#ifndef MICROPY_ENABLE_GC
#define MICROPY_ENABLE_GC                 (1)
#endif
#ifndef MICROPY_HELPER_REPL
#define MICROPY_HELPER_REPL               (1)
#endif
#ifndef MICROPY_LONGINT_IMPL
#define MICROPY_LONGINT_IMPL              (MICROPY_LONGINT_IMPL_MPZ)
#endif
#ifndef MICROPY_ENABLE_EXTERNAL_IMPORT
#define MICROPY_ENABLE_EXTERNAL_IMPORT    (0)
#endif
#ifndef MICROPY_ALLOC_PATH_MAX
#define MICROPY_ALLOC_PATH_MAX            (256)
#endif
#ifndef MICROPY_ALLOC_PARSE_CHUNK_INIT
#define MICROPY_ALLOC_PARSE_CHUNK_INIT    (16)
#endif
#ifndef MICROPY_PY_SYS_MODULES
#define MICROPY_PY_SYS_MODULES            (0)
#endif
#ifndef MICROPY_PY_SYS_EXIT
#define MICROPY_PY_SYS_EXIT               (0)
#endif
#ifndef MICROPY_PY_SYS_PATH
#define MICROPY_PY_SYS_PATH               (0)
#endif
#ifndef MICROPY_PY_SYS_ARGV
#define MICROPY_PY_SYS_ARGV               (0)
#endif
#ifndef MICROPY_PY_SYS_STDFILES
#define MICROPY_PY_SYS_STDFILES           (0)
#endif
#ifndef MICROPY_PY_OS
#define MICROPY_PY_OS                     (0)
#endif
#ifndef MICROPY_PY_TIME
#define MICROPY_PY_TIME                   (0)
#endif
#ifndef MICROPY_VFS
#define MICROPY_VFS                       (0)
#endif
#ifndef MICROPY_READER_VFS
#define MICROPY_READER_VFS                (0)
#endif
#ifndef MICROPY_PY_IO
#define MICROPY_PY_IO                     (0)
#endif
#ifndef MICROPY_PY_SELECT
#define MICROPY_PY_SELECT                 (0)
#endif
#ifndef MICROPY_PY_MACHINE
#define MICROPY_PY_MACHINE                (0)
#endif
#ifndef MICROPY_PY_RANDOM
#define MICROPY_PY_RANDOM                 (0)
#endif
#ifndef MICROPY_GCREGS_SETJMP
#define MICROPY_GCREGS_SETJMP             (0)
#endif
#define MP_SSIZE_MAX                      (0x7fffffff)
#ifndef MICROPY_PERSISTENT_CODE_LOAD
#define MICROPY_PERSISTENT_CODE_LOAD      (1)
#endif
typedef long mp_off_t;
#include <alloca.h>
#define MICROPY_HW_BOARD_NAME "chipformer"
#define MICROPY_HW_MCU_NAME "rv32im_zba"
#define MICROPY_HEAP_SIZE (HEAP_KB * 1024)
#define MP_STATE_PORT MP_STATE_VM
