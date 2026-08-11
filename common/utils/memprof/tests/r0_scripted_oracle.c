/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "r0_raw_emit.h"
#include "r0_scripted_backend.h"

#include <errno.h>
#include <stdint.h>

#ifndef OAI_MEMPROF_R0_ORACLE_MODE
#error "OAI_MEMPROF_R0_ORACLE_MODE must be 0 (A00) or 1 (A01)"
#elif OAI_MEMPROF_R0_ORACLE_MODE == 0
#define OAI_MEMPROF_R0_MODE_NAME "A00"
#define OAI_MEMPROF_R0_MALLOC __real_malloc
#define OAI_MEMPROF_R0_CALLOC __real_calloc
#define OAI_MEMPROF_R0_REALLOC __real_realloc
#define OAI_MEMPROF_R0_FREE __real_free
#elif OAI_MEMPROF_R0_ORACLE_MODE == 1
#define OAI_MEMPROF_R0_MODE_NAME "A01"
#define OAI_MEMPROF_R0_MALLOC __wrap_malloc
#define OAI_MEMPROF_R0_CALLOC __wrap_calloc
#define OAI_MEMPROF_R0_REALLOC __wrap_realloc
#define OAI_MEMPROF_R0_FREE __wrap_free
#else
#error "unsupported OAI_MEMPROF_R0_ORACLE_MODE"
#endif

enum {
  OAI_MEMPROF_R0_EVALUATOR_COUNT = 30,
  OAI_MEMPROF_R0_ERRNO_BASE = 1000,
};

_Static_assert(sizeof(uintptr_t) == 8, "the frozen R0 scripted oracle requires a 64-bit uintptr_t");

static void *p1;
static void *p2;
static void *p3;
static void *p4;
static void *p5;
static void *p6;
static void *p7;
static void *p8;

static uint32_t evaluator_calls;
static uint32_t evaluator_faults;
static unsigned char evaluator_seen[OAI_MEMPROF_R0_EVALUATOR_COUNT + 1];
static int caller_errno_in;

static int transaction_errno(uint32_t transaction)
{
  return OAI_MEMPROF_R0_ERRNO_BASE + (int)transaction;
}

static size_t evaluate_size(uint32_t transaction, const char *phase, const char *operand, uint32_t evaluator, size_t value)
{
  ++evaluator_calls;
  if (evaluator == 0 || evaluator > OAI_MEMPROF_R0_EVALUATOR_COUNT) {
    ++evaluator_faults;
  } else if (evaluator_seen[evaluator]++ != 0) {
    ++evaluator_faults;
  }
  oai_memprof_r0_emit_eval(transaction, phase, operand, evaluator, "SIZE", (uintptr_t)value);
  return value;
}

static void *evaluate_pointer(uint32_t transaction, const char *phase, const char *operand, uint32_t evaluator, void *value)
{
  ++evaluator_calls;
  if (evaluator == 0 || evaluator > OAI_MEMPROF_R0_EVALUATOR_COUNT) {
    ++evaluator_faults;
  } else if (evaluator_seen[evaluator]++ != 0) {
    ++evaluator_faults;
  }
  oai_memprof_r0_emit_eval(transaction, phase, operand, evaluator, "PTR", (uintptr_t)value);
  return value;
}

static void begin_transaction(uint32_t transaction)
{
  oai_memprof_r0_set_transaction(transaction);
  errno = transaction_errno(transaction);
  caller_errno_in = errno;
}

static void record_caller(uint32_t transaction,
                          const char *phase,
                          const char *api,
                          uintptr_t arg0,
                          uintptr_t arg1,
                          const void *result)
{
  const int errno_out = errno;
  oai_memprof_r0_emit_caller(transaction, phase, api, arg0, arg1, result, caller_errno_in, errno_out);
}

__attribute__((constructor(101))) static void oracle_begin(void)
{
  oai_memprof_r0_backend_begin(OAI_MEMPROF_R0_MODE_NAME);
}

__attribute__((constructor(200))) static void constructor_transactions(void)
{
  begin_transaction(1);
  p1 = OAI_MEMPROF_R0_MALLOC(evaluate_size(1, "CTOR", "size", 1, 16));
  record_caller(1, "CTOR", "malloc", 16, 0, p1);

  begin_transaction(2);
  p2 = OAI_MEMPROF_R0_CALLOC(evaluate_size(2, "CTOR", "count", 2, 2), evaluate_size(2, "CTOR", "size", 3, 8));
  record_caller(2, "CTOR", "calloc", 2, 8, p2);
}

int main(void)
{
  void *result;

  begin_transaction(3);
  p3 = OAI_MEMPROF_R0_MALLOC(evaluate_size(3, "MAIN", "size", 4, 0));
  record_caller(3, "MAIN", "malloc", 0, 0, p3);

  begin_transaction(4);
  result = OAI_MEMPROF_R0_MALLOC(evaluate_size(4, "MAIN", "size", 5, 64));
  record_caller(4, "MAIN", "malloc", 64, 0, result);

  begin_transaction(5);
  p4 = OAI_MEMPROF_R0_CALLOC(evaluate_size(5, "MAIN", "count", 6, 0), evaluate_size(5, "MAIN", "size", 7, 4));
  record_caller(5, "MAIN", "calloc", 0, 4, p4);

  begin_transaction(6);
  p5 = OAI_MEMPROF_R0_CALLOC(evaluate_size(6, "MAIN", "count", 8, 3), evaluate_size(6, "MAIN", "size", 9, 0));
  record_caller(6, "MAIN", "calloc", 3, 0, p5);

  begin_transaction(7);
  result = OAI_MEMPROF_R0_CALLOC(evaluate_size(7, "MAIN", "count", 10, 4), evaluate_size(7, "MAIN", "size", 11, 8));
  record_caller(7, "MAIN", "calloc", 4, 8, result);

  begin_transaction(8);
  result = OAI_MEMPROF_R0_CALLOC(evaluate_size(8, "MAIN", "count", 12, SIZE_MAX), evaluate_size(8, "MAIN", "size", 13, 2));
  record_caller(8, "MAIN", "calloc", SIZE_MAX, 2, result);

  begin_transaction(9);
  p6 = OAI_MEMPROF_R0_REALLOC(evaluate_pointer(9, "MAIN", "pointer", 14, NULL), evaluate_size(9, "MAIN", "size", 15, 24));
  record_caller(9, "MAIN", "realloc", (uintptr_t)NULL, 24, p6);

  begin_transaction(10);
  result = OAI_MEMPROF_R0_REALLOC(evaluate_pointer(10, "MAIN", "pointer", 16, p1), evaluate_size(10, "MAIN", "size", 17, 32));
  record_caller(10, "MAIN", "realloc", (uintptr_t)p1, 32, result);
  p1 = result;

  begin_transaction(11);
  result = OAI_MEMPROF_R0_REALLOC(evaluate_pointer(11, "MAIN", "pointer", 18, p2), evaluate_size(11, "MAIN", "size", 19, 48));
  record_caller(11, "MAIN", "realloc", (uintptr_t)p2, 48, result);
  p2 = NULL;
  p7 = result;

  begin_transaction(12);
  result = OAI_MEMPROF_R0_REALLOC(evaluate_pointer(12, "MAIN", "pointer", 20, p7), evaluate_size(12, "MAIN", "size", 21, 96));
  record_caller(12, "MAIN", "realloc", (uintptr_t)p7, 96, result);

  begin_transaction(13);
  result = OAI_MEMPROF_R0_REALLOC(evaluate_pointer(13, "MAIN", "pointer", 22, p3), evaluate_size(13, "MAIN", "size", 23, 0));
  record_caller(13, "MAIN", "realloc", (uintptr_t)p3, 0, result);
  p3 = NULL;

  begin_transaction(14);
  p8 = OAI_MEMPROF_R0_REALLOC(evaluate_pointer(14, "MAIN", "pointer", 24, NULL), evaluate_size(14, "MAIN", "size", 25, 0));
  record_caller(14, "MAIN", "realloc", (uintptr_t)NULL, 0, p8);

  begin_transaction(15);
  OAI_MEMPROF_R0_FREE(evaluate_pointer(15, "MAIN", "pointer", 26, NULL));
  record_caller(15, "MAIN", "free", (uintptr_t)NULL, 0, NULL);

  begin_transaction(16);
  OAI_MEMPROF_R0_FREE(evaluate_pointer(16, "MAIN", "pointer", 27, p1));
  record_caller(16, "MAIN", "free", (uintptr_t)p1, 0, NULL);
  p1 = NULL;

  begin_transaction(17);
  OAI_MEMPROF_R0_FREE(evaluate_pointer(17, "MAIN", "pointer", 28, p4));
  record_caller(17, "MAIN", "free", (uintptr_t)p4, 0, NULL);
  p4 = NULL;

  begin_transaction(18);
  OAI_MEMPROF_R0_FREE(p5);
  record_caller(18, "MAIN", "free", (uintptr_t)p5, 0, NULL);
  p5 = NULL;

  begin_transaction(19);
  OAI_MEMPROF_R0_FREE(p6);
  record_caller(19, "MAIN", "free", (uintptr_t)p6, 0, NULL);
  p6 = NULL;

  begin_transaction(20);
  result = OAI_MEMPROF_R0_REALLOC(evaluate_pointer(20, "MAIN", "pointer", 29, p8), evaluate_size(20, "MAIN", "size", 30, 7));
  record_caller(20, "MAIN", "realloc", (uintptr_t)p8, 7, result);
  p8 = result;

  return 0;
}

__attribute__((destructor(200))) static void destructor_transactions(void)
{
  begin_transaction(21);
  OAI_MEMPROF_R0_FREE(p7);
  record_caller(21, "DTOR", "free", (uintptr_t)p7, 0, NULL);
  p7 = NULL;

  begin_transaction(22);
  OAI_MEMPROF_R0_FREE(p8);
  record_caller(22, "DTOR", "free", (uintptr_t)p8, 0, NULL);
  p8 = NULL;
}

__attribute__((destructor(101))) static void oracle_end(void)
{
  for (uint32_t evaluator = 1; evaluator <= OAI_MEMPROF_R0_EVALUATOR_COUNT; ++evaluator) {
    if (evaluator_seen[evaluator] != 1)
      ++evaluator_faults;
  }
  oai_memprof_r0_emit_summary(oai_memprof_r0_real_calls(),
                              evaluator_calls,
                              oai_memprof_r0_live_allocations(),
                              oai_memprof_r0_context_probes(),
                              evaluator_faults);
}
