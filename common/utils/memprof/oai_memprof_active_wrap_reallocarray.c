/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "oai_memprof_active_wrap_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

void *__real_reallocarray(void *pointer, size_t count, size_t size);
void *__wrap_reallocarray(void *pointer, size_t count, size_t size);

OAI_MEMPROF_WRAPPER_ATTRIBUTES void *__wrap_reallocarray(void *pointer, size_t count, size_t size)
{
  const uint64_t control = oai_memprof_active_control_load_v1();
  if (oai_memprof_active_control_state_v1(control) != OAI_MEMPROF_CORE_ACTIVE)
    return __real_reallocarray(pointer, count, size);

  const bool overflow = count != 0 && size > SIZE_MAX / count;
  const uint64_t product = overflow ? 0 : (uint64_t)(count * size);
  const uint64_t address_before = (uint64_t)(uintptr_t)pointer;
  oai_memprof_core_ticket_t ticket = {0};
  const bool admitted = oai_memprof_active_runtime_begin_v1(5, product, !overflow, &ticket);
  if (admitted && pointer != NULL && oai_memprof_active_control_mode_v1(control) == OAI_MEMPROF_CORE_SAMPLED)
    (void)oai_memprof_active_runtime_sample_predecessor_v1(&ticket, address_before);
  void *result = __real_reallocarray(pointer, count, size);
  if (!admitted)
    return result;

  const int result_errno = errno;
  uint32_t flags = OAI_MEMPROF_FLAG_ADDRESS_BEFORE_VALID | OAI_MEMPROF_FLAG_ADDRESS_AFTER_VALID | OAI_MEMPROF_FLAG_ARG1_VALID
                   | OAI_MEMPROF_FLAG_ARG2_VALID | OAI_MEMPROF_FLAG_RESULT_ERRNO;
  if (overflow) {
    flags |= OAI_MEMPROF_FLAG_REALLOCARRAY_PRODUCT_OVERFLOW | OAI_MEMPROF_FLAG_OPERATION_FAILED;
  } else {
    flags |= OAI_MEMPROF_FLAG_ARG0_VALID;
    if (product == 0)
      flags |= OAI_MEMPROF_FLAG_ZERO_SIZE_REQUEST;
    if (result != NULL)
      flags |= OAI_MEMPROF_FLAG_SUCCESSOR_CREATED;

    if (pointer != NULL) {
      if (result != NULL) {
        flags |= OAI_MEMPROF_FLAG_PREDECESSOR_ENDED;
      } else if (product != 0) {
        flags |= OAI_MEMPROF_FLAG_OPERATION_FAILED;
      } else if (oai_memprof_active_runtime_realloc_zero_policy_v1() == 1) {
        flags |= OAI_MEMPROF_FLAG_PREDECESSOR_ENDED;
      } else {
        flags |= OAI_MEMPROF_FLAG_OPERATION_FAILED;
      }
    } else if (result == NULL && product != 0) {
      flags |= OAI_MEMPROF_FLAG_OPERATION_FAILED;
    }
  }

  const oai_memprof_core_payload_t payload = {
      .address_before = address_before,
      .address_after = (uint64_t)(uintptr_t)result,
      .arg0 = product,
      .arg1 = (uint64_t)count,
      .arg2 = (uint64_t)size,
      .flags = flags,
      .result_code = result_errno,
      .api_id = 5,
      .event_kind = 2,
  };
  (void)oai_memprof_active_runtime_end_v1(&ticket, &payload);
  errno = result_errno;
  return result;
}
