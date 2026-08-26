/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "oai_memprof_active_wrap_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

void *__real_pvalloc(size_t size);
void *__wrap_pvalloc(size_t size);

OAI_MEMPROF_WRAPPER_ATTRIBUTES void *__wrap_pvalloc(size_t size)
{
  const uint64_t control = oai_memprof_active_control_load_v1();
  if (oai_memprof_active_control_state_v1(control) != OAI_MEMPROF_CORE_ACTIVE)
    return __real_pvalloc(size);

  oai_memprof_core_ticket_t ticket = {0};
  const bool admitted = oai_memprof_active_runtime_begin_v1(10, (uint64_t)size, true, &ticket);
  void *result = __real_pvalloc(size);
  if (!admitted)
    return result;

  const int result_errno = errno;
  uint32_t flags = OAI_MEMPROF_FLAG_ADDRESS_AFTER_VALID | OAI_MEMPROF_FLAG_ARG0_VALID | OAI_MEMPROF_FLAG_RESULT_ERRNO;
  if (size == 0)
    flags |= OAI_MEMPROF_FLAG_ZERO_SIZE_REQUEST;
  if (result != NULL)
    flags |= OAI_MEMPROF_FLAG_SUCCESSOR_CREATED;
  else if (size != 0)
    flags |= OAI_MEMPROF_FLAG_OPERATION_FAILED;
  const oai_memprof_core_payload_t payload = {
      .address_after = (uint64_t)(uintptr_t)result,
      .arg0 = (uint64_t)size,
      .flags = flags,
      .result_code = result_errno,
      .api_id = 10,
      .event_kind = 1,
  };
  (void)oai_memprof_active_runtime_end_v1(&ticket, &payload);
  errno = result_errno;
  return result;
}
