/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_ACTIVE_RUNTIME_ABI_H
#define OAI_MEMPROF_ACTIVE_RUNTIME_ABI_H

#include "oai_memprof_active_core.h"

#include <stdatomic.h>
#include <stdint.h>

#if defined(__cplusplus)
#error "the memory-lifetime profiler ACTIVE runtime ABI is C-only"
#endif

#define OAI_MEMPROF_ACTIVE_RUNTIME_ABI_VERSION UINT32_C(1)
#define OAI_MEMPROF_ACTIVE_CONTROL_CACHE_LINE_BYTES 64
#define OAI_MEMPROF_ACTIVE_CONTROL_STATE_MASK UINT64_C(0xff)
#define OAI_MEMPROF_ACTIVE_CONTROL_MODE_SHIFT 8U
#define OAI_MEMPROF_ACTIVE_CONTROL_GENERATION_SHIFT 16U

#if defined(OAI_MEMPROF_ACTIVE_RUNTIME_BUILD)
#define OAI_MEMPROF_ACTIVE_VISIBILITY __attribute__((visibility("protected")))
#else
#define OAI_MEMPROF_ACTIVE_VISIBILITY __attribute__((visibility("default")))
#endif

extern OAI_MEMPROF_ACTIVE_VISIBILITY _Atomic(uint64_t) oai_memprof_active_control_v1;

typedef struct oai_memprof_active_runtime_config_s {
  oai_memprof_core_config_t core;
  uint16_t realloc_zero_policy_id;
} oai_memprof_active_runtime_config_t;

static inline __attribute__((always_inline, no_instrument_function)) uint64_t oai_memprof_active_control_load_v1(void)
{
  return atomic_load_explicit(&oai_memprof_active_control_v1, memory_order_seq_cst);
}

static inline __attribute__((always_inline, no_instrument_function)) uint8_t oai_memprof_active_control_state_v1(uint64_t control)
{
  return (uint8_t)(control & OAI_MEMPROF_ACTIVE_CONTROL_STATE_MASK);
}

static inline __attribute__((always_inline, no_instrument_function)) uint8_t oai_memprof_active_control_mode_v1(uint64_t control)
{
  return (uint8_t)((control >> OAI_MEMPROF_ACTIVE_CONTROL_MODE_SHIFT) & UINT64_C(0xff));
}

OAI_MEMPROF_ACTIVE_VISIBILITY oai_memprof_core_status_t
oai_memprof_active_runtime_bootstrap_v1(const oai_memprof_active_runtime_config_t *config);
OAI_MEMPROF_ACTIVE_VISIBILITY oai_memprof_core_status_t oai_memprof_active_runtime_activate_v1(void);
OAI_MEMPROF_ACTIVE_VISIBILITY oai_memprof_core_status_t oai_memprof_active_runtime_seal_v1(uint64_t timeout_ns);
OAI_MEMPROF_ACTIVE_VISIBILITY oai_memprof_core_status_t oai_memprof_active_runtime_drain_v1(oai_memprof_core_sink_t sink,
                                                                                            void *context);
OAI_MEMPROF_ACTIVE_VISIBILITY oai_memprof_core_status_t oai_memprof_active_runtime_complete_v1(void);
OAI_MEMPROF_ACTIVE_VISIBILITY oai_memprof_core_status_t
oai_memprof_active_runtime_snapshot_v1(oai_memprof_core_snapshot_t *snapshot);
OAI_MEMPROF_ACTIVE_VISIBILITY oai_memprof_core_status_t
oai_memprof_active_runtime_thread_info_v1(uint32_t slot_index, oai_memprof_core_thread_info_t *info);
OAI_MEMPROF_ACTIVE_VISIBILITY bool oai_memprof_active_runtime_begin_v1(uint16_t api_id,
                                                                       uint64_t requested_bytes,
                                                                       bool requested_bytes_valid,
                                                                       oai_memprof_core_ticket_t *ticket);
OAI_MEMPROF_ACTIVE_VISIBILITY bool oai_memprof_active_runtime_sample_predecessor_v1(oai_memprof_core_ticket_t *ticket,
                                                                                    uint64_t address);
OAI_MEMPROF_ACTIVE_VISIBILITY oai_memprof_core_status_t
oai_memprof_active_runtime_end_v1(oai_memprof_core_ticket_t *ticket, const oai_memprof_core_payload_t *payload);
OAI_MEMPROF_ACTIVE_VISIBILITY uint16_t oai_memprof_active_runtime_realloc_zero_policy_v1(void);

#undef OAI_MEMPROF_ACTIVE_VISIBILITY

#endif /* OAI_MEMPROF_ACTIVE_RUNTIME_ABI_H */
