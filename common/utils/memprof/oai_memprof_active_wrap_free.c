/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "oai_memprof_active_wrap_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

void __real_free(void *pointer);
void __wrap_free(void *pointer);

OAI_MEMPROF_WRAPPER_ATTRIBUTES void __wrap_free(void *pointer)
{
  const uint64_t control = oai_memprof_active_control_load_v1();
  if (oai_memprof_active_control_state_v1(control) != OAI_MEMPROF_CORE_ACTIVE) {
    __real_free(pointer);
    return;
  }

  const uint64_t address_before = (uint64_t)(uintptr_t)pointer;
  oai_memprof_core_ticket_t ticket = {0};
  const bool admitted = oai_memprof_active_runtime_begin_v1(4, 0, false, &ticket);
  if (admitted && pointer != NULL && oai_memprof_active_control_mode_v1(control) == OAI_MEMPROF_CORE_SAMPLED)
    (void)oai_memprof_active_runtime_sample_predecessor_v1(&ticket, address_before);
  __real_free(pointer);
  if (!admitted)
    return;

  const int result_errno = errno;
  uint32_t flags = OAI_MEMPROF_FLAG_ADDRESS_BEFORE_VALID | OAI_MEMPROF_FLAG_RESULT_ERRNO;
  if (pointer != NULL)
    flags |= OAI_MEMPROF_FLAG_PREDECESSOR_ENDED;
  const oai_memprof_core_payload_t payload = {
      .address_before = address_before,
      .flags = flags,
      .result_code = result_errno,
      .api_id = 4,
      .event_kind = 3,
  };
  (void)oai_memprof_active_runtime_end_v1(&ticket, &payload);
  errno = result_errno;
}
