// MicroPython on the chipformer RV32 machine: boot, run the script compiled into the firmware
// (script.h, in the DROM), print through putchar ecalls, exit with 0 (or 1 on an exception).
#include <stdint.h>
#include <string.h>
#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/cstack.h"
#include "py/mperrno.h"
#include "shared/runtime/gchelper.h"
#include "rt.h"
#include "script.h"  // static const char script[]

extern char __heap_start[], __heap_end[];

static int run_script(const char *src, size_t len) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, src, len, 0);
        qstr source_name = lex->source_name;
        mp_parse_tree_t parse_tree = mp_parse(lex, MP_PARSE_FILE_INPUT);
        mp_obj_t module_fun = mp_compile(&parse_tree, source_name, false);
        mp_call_function_0(module_fun);
        nlr_pop();
        return 0;
    }
    mp_obj_print_exception(&mp_plat_print, (mp_obj_t)nlr.ret_val);
    return 1;
}

int main(void) {
    mp_cstack_init_with_sp_here(STACK_BYTES);
    gc_init(__heap_start, __heap_end);
    mp_init();
    int rc = run_script(script, sizeof(script) - 1);
    mp_deinit();
    return rc;
}

void gc_collect(void) {
    gc_collect_start();
    gc_helper_collect_regs_and_stack();
    gc_collect_end();
}

int mp_hal_stdin_rx_chr(void) { return 4; }

mp_uint_t mp_hal_stdout_tx_strn(const char *str, mp_uint_t len) {
    for (mp_uint_t i = 0; i < len; i++) rv_putchar(str[i]);
    return len;
}

mp_lexer_t *mp_lexer_new_from_file(qstr filename) { mp_raise_OSError(MP_ENOENT); }
mp_import_stat_t mp_import_stat(const char *path) { return MP_IMPORT_STAT_NO_EXIST; }
void nlr_jump_fail(void *val) { for (;;) rv_ecall2(93, 3, 0, 0); }
void MP_NORETURN __fatal_error(const char *msg) { for (;;) rv_ecall2(93, 4, 0, 0); }
