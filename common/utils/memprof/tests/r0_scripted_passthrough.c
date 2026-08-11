/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "r0_scripted_backend.h"

#include <errno.h>

enum oai_memprof_r0_mutation_e {
  OAI_MEMPROF_R0_MUTATION_NONE = 0,
  OAI_MEMPROF_R0_MUTATION_DUPLICATE_REAL = 1,
  OAI_MEMPROF_R0_MUTATION_OPERAND = 2,
  OAI_MEMPROF_R0_MUTATION_ERRNO = 3,
  OAI_MEMPROF_R0_MUTATION_RESULT = 4,
  OAI_MEMPROF_R0_MUTATION_SUPPRESS_FREE_NULL = 5,
  OAI_MEMPROF_R0_MUTATION_CONTEXT = 6,
};

#ifndef OAI_MEMPROF_R0_MUTATION
#define OAI_MEMPROF_R0_MUTATION OAI_MEMPROF_R0_MUTATION_NONE
#endif

void *__wrap_malloc(size_t size)
{
  const uint32_t transaction = oai_memprof_r0_current_transaction();

  if (OAI_MEMPROF_R0_MUTATION == OAI_MEMPROF_R0_MUTATION_CONTEXT && transaction == 1)
    oai_memprof_r0_context_probe();

  void *result = __real_malloc(size);
  if (OAI_MEMPROF_R0_MUTATION == OAI_MEMPROF_R0_MUTATION_DUPLICATE_REAL && transaction == 1)
    (void)__real_malloc(size);
  if (OAI_MEMPROF_R0_MUTATION == OAI_MEMPROF_R0_MUTATION_ERRNO && transaction == 1)
    errno = EIO;
  if (OAI_MEMPROF_R0_MUTATION == OAI_MEMPROF_R0_MUTATION_RESULT && transaction == 1)
    result = NULL;
  return result;
}

void *__wrap_calloc(size_t count, size_t size)
{
  if (OAI_MEMPROF_R0_MUTATION == OAI_MEMPROF_R0_MUTATION_OPERAND && oai_memprof_r0_current_transaction() == 2)
    ++count;
  return __real_calloc(count, size);
}

void *__wrap_realloc(void *pointer, size_t size)
{
  return __real_realloc(pointer, size);
}

void __wrap_free(void *pointer)
{
  if (OAI_MEMPROF_R0_MUTATION == OAI_MEMPROF_R0_MUTATION_SUPPRESS_FREE_NULL && oai_memprof_r0_current_transaction() == 15)
    return;
  __real_free(pointer);
}
