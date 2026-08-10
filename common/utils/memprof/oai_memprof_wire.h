/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_WIRE_H
#define OAI_MEMPROF_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Schema-v1 disk event layout. These offsets describe bytes on disk, not the
 * layout of oai_memprof_event_v1_t in native memory.
 */
enum oai_memprof_event_v1_wire_layout {
  OAI_MEMPROF_EVENT_V1_THREAD_SEQUENCE_OFFSET = 0,
  OAI_MEMPROF_EVENT_V1_COUNTER_ENTER_OFFSET = 8,
  OAI_MEMPROF_EVENT_V1_COUNTER_EXIT_OFFSET = 16,
  OAI_MEMPROF_EVENT_V1_ADDRESS_BEFORE_OFFSET = 24,
  OAI_MEMPROF_EVENT_V1_ADDRESS_AFTER_OFFSET = 32,
  OAI_MEMPROF_EVENT_V1_ARG0_OFFSET = 40,
  OAI_MEMPROF_EVENT_V1_ARG1_OFFSET = 48,
  OAI_MEMPROF_EVENT_V1_ARG2_OFFSET = 56,
  OAI_MEMPROF_EVENT_V1_CONTEXT_ID_OFFSET = 64,
  OAI_MEMPROF_EVENT_V1_CALLSITE_ID_OFFSET = 68,
  OAI_MEMPROF_EVENT_V1_THREAD_INDEX_OFFSET = 72,
  OAI_MEMPROF_EVENT_V1_FLAGS_OFFSET = 76,
  OAI_MEMPROF_EVENT_V1_RESULT_CODE_OFFSET = 80,
  OAI_MEMPROF_EVENT_V1_API_ID_OFFSET = 84,
  OAI_MEMPROF_EVENT_V1_EVENT_KIND_OFFSET = 86,
  OAI_MEMPROF_EVENT_V1_CPU_ENTER_OFFSET = 88,
  OAI_MEMPROF_EVENT_V1_CPU_EXIT_OFFSET = 90,
  OAI_MEMPROF_EVENT_V1_RESERVED_ZERO_OFFSET = 92,
  OAI_MEMPROF_EVENT_V1_WIRE_SIZE = 96,
};

/*
 * Host representation of one completed schema-v1 wrapper transaction. It is
 * deliberately not packed and must never be copied directly to or from disk.
 * The reserved wire field is omitted because it has no host-side meaning.
 */
typedef struct oai_memprof_event_v1_s {
  uint64_t thread_sequence;
  uint64_t counter_enter;
  uint64_t counter_exit;
  uint64_t address_before;
  uint64_t address_after;
  uint64_t arg0;
  uint64_t arg1;
  uint64_t arg2;
  uint32_t context_id;
  uint32_t callsite_id;
  uint32_t thread_index;
  uint32_t flags;
  int32_t result_code;
  uint16_t api_id;
  uint16_t event_kind;
  uint16_t cpu_enter;
  uint16_t cpu_exit;
} oai_memprof_event_v1_t;

typedef enum oai_memprof_wire_status_e {
  OAI_MEMPROF_WIRE_OK = 0,
  OAI_MEMPROF_WIRE_NULL_ARGUMENT,
  OAI_MEMPROF_WIRE_WRONG_SIZE,
  OAI_MEMPROF_WIRE_NONZERO_RESERVED,
} oai_memprof_wire_status_t;

/*
 * Encode exactly one 96-byte little-endian record. The destination is left
 * unchanged on error, and bytes 92..95 are always written as zero on success.
 * API IDs, event kinds, and flags are preserved verbatim; their meanings and
 * valid combinations belong to the authoritative catalogs, not this codec.
 */
oai_memprof_wire_status_t oai_memprof_event_v1_encode(const oai_memprof_event_v1_t *event, uint8_t *wire, size_t wire_size);

/*
 * Decode exactly one 96-byte little-endian record. The output is left
 * unchanged on error. A nonzero reserved field is a structural schema error.
 * Catalog validation is intentionally separate from structural decoding.
 */
oai_memprof_wire_status_t oai_memprof_event_v1_decode(oai_memprof_event_v1_t *event, const uint8_t *wire, size_t wire_size);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_WIRE_H */
