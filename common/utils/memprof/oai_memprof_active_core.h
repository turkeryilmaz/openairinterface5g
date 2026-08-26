/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_ACTIVE_CORE_H
#define OAI_MEMPROF_ACTIVE_CORE_H

#include "oai_memprof_wire.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define OAI_MEMPROF_CORE_API_SLOT_COUNT UINT32_C(32)
#define OAI_MEMPROF_CORE_ADMITTED_API_COUNT UINT16_C(12)

#ifdef __cplusplus
extern "C" {
#endif

enum oai_memprof_core_state {
  OAI_MEMPROF_CORE_PRESENT_OFF = 0,
  OAI_MEMPROF_CORE_BOOTSTRAP = 1,
  OAI_MEMPROF_CORE_ACTIVE = 2,
  OAI_MEMPROF_CORE_DRAINING = 3,
  OAI_MEMPROF_CORE_COMPLETE = 4,
};

enum oai_memprof_core_mode {
  OAI_MEMPROF_CORE_COUNTERS = 2,
  OAI_MEMPROF_CORE_SAMPLED = 3,
  OAI_MEMPROF_CORE_EXACT_EVENTS = 4,
};

enum oai_memprof_core_flag {
  OAI_MEMPROF_CORE_COUNTER_ENTER_VALID = UINT32_C(1) << 5,
  OAI_MEMPROF_CORE_COUNTER_EXIT_VALID = UINT32_C(1) << 6,
  OAI_MEMPROF_CORE_CPU_ENTER_VALID = UINT32_C(1) << 7,
  OAI_MEMPROF_CORE_CPU_EXIT_VALID = UINT32_C(1) << 8,
  OAI_MEMPROF_CORE_PREDECESSOR_MATCH_VALID = UINT32_C(1) << 14,
  OAI_MEMPROF_CORE_PREDECESSOR_SELECTED = UINT32_C(1) << 15,
  OAI_MEMPROF_CORE_SUCCESSOR_SELECTED = UINT32_C(1) << 16,
  OAI_MEMPROF_CORE_CROSS_THREAD_ENDPOINT = UINT32_C(1) << 17,
  OAI_MEMPROF_CORE_BOUNDARY_STRADDLING = UINT32_C(1) << 18,
};

typedef enum oai_memprof_core_status_e {
  OAI_MEMPROF_CORE_OK = 0,
  OAI_MEMPROF_CORE_INVALID_ARGUMENT,
  OAI_MEMPROF_CORE_INVALID_CONFIGURATION,
  OAI_MEMPROF_CORE_NO_MEMORY,
  OAI_MEMPROF_CORE_SYSTEM_ERROR,
  OAI_MEMPROF_CORE_INVALID_STATE,
  OAI_MEMPROF_CORE_SEAL_TIMEOUT,
  OAI_MEMPROF_CORE_SINK_ERROR,
} oai_memprof_core_status_t;

typedef struct oai_memprof_core_s oai_memprof_core_t;

typedef struct oai_memprof_core_config_s {
  uint64_t process_generation;
  uint64_t table_entries;
  uint64_t sample_seed;
  uint64_t sample_threshold;
  uint32_t max_threads;
  uint32_t ring_records;
  uint32_t table_probes;
  uint8_t mode_id;
} oai_memprof_core_config_t;

typedef struct oai_memprof_core_ticket_s {
  oai_memprof_core_t *core;
  uint64_t generation;
  uint64_t thread_sequence;
  uint64_t counter_enter;
  uint64_t requested_bytes;
  uint64_t predecessor_address;
  uint64_t predecessor_sequence;
  uint64_t predecessor_requested_bytes;
  uint64_t predecessor_birth_counter;
  uint64_t predecessor_tag;
  size_t predecessor_slot;
  uint32_t slot_index;
  uint32_t thread_index;
  uint32_t predecessor_thread_index;
  uint16_t cpu_enter;
  uint16_t api_id;
  bool counter_enter_valid;
  bool cpu_enter_valid;
  bool requested_bytes_valid;
  bool predecessor_requested_bytes_valid;
  bool successor_selected;
  bool predecessor_match;
  bool admitted;
} oai_memprof_core_ticket_t;

typedef struct oai_memprof_core_payload_s {
  uint64_t address_before;
  uint64_t address_after;
  uint64_t arg0;
  uint64_t arg1;
  uint64_t arg2;
  uint32_t context_id;
  uint32_t callsite_id;
  uint32_t flags;
  int32_t result_code;
  uint16_t api_id;
  uint16_t event_kind;
} oai_memprof_core_payload_t;

typedef struct oai_memprof_core_snapshot_s {
  uint64_t process_generation;
  uint64_t reservations;
  uint64_t ready_threads;
  uint64_t registration_capacity_failures;
  uint64_t unregistered_active_thread_failures;
  uint64_t diagnostic_saturation_transitions;
  uint32_t registration_diagnostic_saturated_mask;
  uint64_t recursion_bypasses;
  uint64_t ring_full_losses;
  uint64_t admitted_transactions;
  uint64_t completed_transactions;
  uint64_t emitted_events;
  uint64_t requested_bytes;
  uint64_t table_entries;
  uint64_t sample_seed;
  uint64_t sample_threshold;
  uint32_t table_probes;
  uint32_t table_shards;
  uint8_t state;
  uint8_t mode_id;
} oai_memprof_core_snapshot_t;

typedef struct oai_memprof_core_thread_info_s {
  uint64_t process_generation;
  uint64_t registration_ordinal;
  uint64_t thread_sequence;
  uint64_t api_attempts[OAI_MEMPROF_CORE_API_SLOT_COUNT];
  uint64_t requested_bytes;
  uint64_t completed_transactions;
  uint64_t recursion_bypasses;
  uint64_t ring_full_losses;
  uint64_t size_unknowns;
  uint64_t sample_insertion_failures;
  uint64_t sample_lookup_failures;
  uint64_t sample_probe_exhaustions;
  uint64_t sample_pairing_failures;
  uint64_t counter_invalids;
  uint32_t thread_index;
  uint32_t diagnostic_saturated_mask;
} oai_memprof_core_thread_info_t;

typedef bool (*oai_memprof_core_sink_t)(void *context, const oai_memprof_event_v1_t *event);

oai_memprof_core_status_t oai_memprof_core_bootstrap(const oai_memprof_core_config_t *config, oai_memprof_core_t **core);
oai_memprof_core_status_t oai_memprof_core_activate(oai_memprof_core_t *core);
bool oai_memprof_core_selection_value(uint64_t process_generation,
                                      uint32_t thread_index,
                                      uint64_t thread_sequence,
                                      uint64_t sample_seed,
                                      uint64_t *value);
bool oai_memprof_core_begin(oai_memprof_core_t *core,
                            uint16_t api_id,
                            uint64_t requested_bytes,
                            bool requested_bytes_valid,
                            oai_memprof_core_ticket_t *ticket);
bool oai_memprof_core_sample_predecessor(oai_memprof_core_ticket_t *ticket, uint64_t address);
oai_memprof_core_status_t oai_memprof_core_end(oai_memprof_core_ticket_t *ticket, const oai_memprof_core_payload_t *payload);
oai_memprof_core_status_t oai_memprof_core_seal(oai_memprof_core_t *core, uint64_t timeout_ns);
oai_memprof_core_status_t oai_memprof_core_drain(oai_memprof_core_t *core, oai_memprof_core_sink_t sink, void *context);
oai_memprof_core_status_t oai_memprof_core_complete(oai_memprof_core_t *core);
oai_memprof_core_status_t oai_memprof_core_snapshot(const oai_memprof_core_t *core, oai_memprof_core_snapshot_t *snapshot);
oai_memprof_core_status_t oai_memprof_core_thread_info(const oai_memprof_core_t *core,
                                                       uint32_t slot_index,
                                                       oai_memprof_core_thread_info_t *info);
uint64_t oai_memprof_core_control(const oai_memprof_core_t *core);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_ACTIVE_CORE_H */
