/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "oai_memprof_wrap_internal.h"

void *__real_malloc(size_t size);
void *__wrap_malloc(size_t size);

OAI_MEMPROF_WRAPPER_ATTRIBUTES void *__wrap_malloc(size_t size)
{
  const uint64_t control = oai_memprof_control_load_v1();
  (void)control;
  return __real_malloc(size);
}
