/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define OAI_MEMPROF_ACTIVE_RUNTIME_BUILD 1
#include "oai_memprof_active_runtime_abi.h"

#include <stdatomic.h>

#define OAI_MEMPROF_ACTIVE_MAX_GENERATION ((UINT64_C(1) << 48) - UINT64_C(1))

_Alignas(OAI_MEMPROF_ACTIVE_CONTROL_CACHE_LINE_BYTES)
    __attribute__((visibility("protected"))) _Atomic(uint64_t) oai_memprof_active_control_v1 = 0;

static oai_memprof_core_t *runtime_core;
static uint64_t runtime_generation;
static uint8_t runtime_mode;
static uint16_t runtime_realloc_zero_policy;

static uint64_t pack_control(uint8_t state)
{
  return (runtime_generation << OAI_MEMPROF_ACTIVE_CONTROL_GENERATION_SHIFT)
         | ((uint64_t)runtime_mode << OAI_MEMPROF_ACTIVE_CONTROL_MODE_SHIFT) | state;
}

oai_memprof_core_status_t oai_memprof_active_runtime_bootstrap_v1(const oai_memprof_active_runtime_config_t *config)
{
  if (config == NULL)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  if (runtime_core != NULL || atomic_load_explicit(&oai_memprof_active_control_v1, memory_order_seq_cst) != 0)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  if (config->realloc_zero_policy_id != 1 && config->realloc_zero_policy_id != 2)
    return OAI_MEMPROF_CORE_INVALID_CONFIGURATION;
  if (config->core.process_generation == 0 || config->core.process_generation > OAI_MEMPROF_ACTIVE_MAX_GENERATION)
    return OAI_MEMPROF_CORE_INVALID_CONFIGURATION;

  oai_memprof_core_t *core = NULL;
  const oai_memprof_core_status_t status = oai_memprof_core_bootstrap(&config->core, &core);
  if (status != OAI_MEMPROF_CORE_OK)
    return status;
  runtime_core = core;
  runtime_generation = config->core.process_generation;
  runtime_mode = config->core.mode_id;
  runtime_realloc_zero_policy = config->realloc_zero_policy_id;
  return OAI_MEMPROF_CORE_OK;
}

oai_memprof_core_status_t oai_memprof_active_runtime_activate_v1(void)
{
  if (runtime_core == NULL)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  const oai_memprof_core_status_t status = oai_memprof_core_activate(runtime_core);
  if (status != OAI_MEMPROF_CORE_OK)
    return status;
  atomic_store_explicit(&oai_memprof_active_control_v1, pack_control(OAI_MEMPROF_CORE_ACTIVE), memory_order_seq_cst);
  return OAI_MEMPROF_CORE_OK;
}

oai_memprof_core_status_t oai_memprof_active_runtime_seal_v1(uint64_t timeout_ns)
{
  if (runtime_core == NULL)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  uint64_t expected = pack_control(OAI_MEMPROF_CORE_ACTIVE);
  if (!atomic_compare_exchange_strong_explicit(&oai_memprof_active_control_v1,
                                               &expected,
                                               pack_control(OAI_MEMPROF_CORE_DRAINING),
                                               memory_order_seq_cst,
                                               memory_order_seq_cst))
    return OAI_MEMPROF_CORE_INVALID_STATE;
  return oai_memprof_core_seal(runtime_core, timeout_ns);
}

oai_memprof_core_status_t oai_memprof_active_runtime_drain_v1(oai_memprof_core_sink_t sink, void *context)
{
  if (runtime_core == NULL)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  return oai_memprof_core_drain(runtime_core, sink, context);
}

oai_memprof_core_status_t oai_memprof_active_runtime_complete_v1(void)
{
  if (runtime_core == NULL)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  const oai_memprof_core_status_t status = oai_memprof_core_complete(runtime_core);
  if (status != OAI_MEMPROF_CORE_OK)
    return status;
  atomic_store_explicit(&oai_memprof_active_control_v1, pack_control(OAI_MEMPROF_CORE_COMPLETE), memory_order_seq_cst);
  return OAI_MEMPROF_CORE_OK;
}

oai_memprof_core_status_t oai_memprof_active_runtime_snapshot_v1(oai_memprof_core_snapshot_t *snapshot)
{
  if (runtime_core == NULL)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  return oai_memprof_core_snapshot(runtime_core, snapshot);
}

oai_memprof_core_status_t oai_memprof_active_runtime_thread_info_v1(uint32_t slot_index, oai_memprof_core_thread_info_t *info)
{
  if (runtime_core == NULL)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  return oai_memprof_core_thread_info(runtime_core, slot_index, info);
}

bool oai_memprof_active_runtime_begin_v1(uint16_t api_id,
                                         uint64_t requested_bytes,
                                         bool requested_bytes_valid,
                                         oai_memprof_core_ticket_t *ticket)
{
  return runtime_core != NULL && oai_memprof_core_begin(runtime_core, api_id, requested_bytes, requested_bytes_valid, ticket);
}

bool oai_memprof_active_runtime_sample_predecessor_v1(oai_memprof_core_ticket_t *ticket, uint64_t address)
{
  return runtime_core != NULL && oai_memprof_core_sample_predecessor(ticket, address);
}

oai_memprof_core_status_t oai_memprof_active_runtime_end_v1(oai_memprof_core_ticket_t *ticket,
                                                            const oai_memprof_core_payload_t *payload)
{
  if (runtime_core == NULL)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  return oai_memprof_core_end(ticket, payload);
}

uint16_t oai_memprof_active_runtime_realloc_zero_policy_v1(void)
{
  return runtime_realloc_zero_policy;
}
