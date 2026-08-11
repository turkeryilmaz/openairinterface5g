/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "oai_memprof_wrap_internal.h"

void __real_free(void *pointer);
void __wrap_free(void *pointer);

OAI_MEMPROF_WRAPPER_ATTRIBUTES void __wrap_free(void *pointer)
{
  const uint64_t control = oai_memprof_control_load_v1();
  (void)control;
  __real_free(pointer);
}
