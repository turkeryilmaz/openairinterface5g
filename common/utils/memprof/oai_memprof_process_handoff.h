/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_PROCESS_HANDOFF_H
#define OAI_MEMPROF_PROCESS_HANDOFF_H

#include "oai_memprof_stream_writer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES UINT32_C(1152)
#define OAI_MEMPROF_PROCESS_HANDOFF_V1_THREAD_BYTES UINT32_C(448)
#define OAI_MEMPROF_PROCESS_HANDOFF_V1_DIGEST_BYTES UINT32_C(32)
#define OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_BOOTSTRAP_BYTES UINT64_C(65536)
#define OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_MAPS_BYTES UINT64_C(16777216)
#define OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_THREADS UINT32_C(65534)
#define OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_RING_RECORDS UINT32_C(1048576)
#define OAI_MEMPROF_PROCESS_HANDOFF_V1_DIAGNOSTIC_COUNT UINT32_C(10)

/* Wire diagnostic order maps exactly to reason IDs 1,16,17,18,32,48,49,50,51,64. */
enum oai_memprof_process_handoff_v1_diagnostic_index {
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_RING_FULL = 0,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_RECURSION_BYPASS = 1,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_INTERNAL_BYPASS = 2,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_UNSUPPORTED_DOMAIN = 3,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SIZE_UNKNOWN = 4,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_INSERTION = 5,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_LOOKUP = 6,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_PROBE = 7,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_PAIRING = 8,
  OAI_MEMPROF_HANDOFF_DIAGNOSTIC_COUNTER_INVALID = 9,
};

typedef enum oai_memprof_process_handoff_status_e {
  OAI_MEMPROF_PROCESS_HANDOFF_OK = 0,
  OAI_MEMPROF_PROCESS_HANDOFF_INVALID_ARGUMENT,
  OAI_MEMPROF_PROCESS_HANDOFF_INVALID_CONFIGURATION,
  OAI_MEMPROF_PROCESS_HANDOFF_NO_MEMORY,
  OAI_MEMPROF_PROCESS_HANDOFF_WRONG_SIZE,
  OAI_MEMPROF_PROCESS_HANDOFF_BAD_MAGIC,
  OAI_MEMPROF_PROCESS_HANDOFF_BAD_CHECKSUM,
  OAI_MEMPROF_PROCESS_HANDOFF_UNSUPPORTED_VERSION,
  OAI_MEMPROF_PROCESS_HANDOFF_NONZERO_RESERVED,
  OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION,
  OAI_MEMPROF_PROCESS_HANDOFF_INTEGER_OVERFLOW,
  OAI_MEMPROF_PROCESS_HANDOFF_CODEC_ERROR,
} oai_memprof_process_handoff_status_t;

typedef struct oai_memprof_process_handoff_thread_v1_s {
  oai_memprof_core_thread_info_t runtime;
  uint64_t diagnostic_values[OAI_MEMPROF_PROCESS_HANDOFF_V1_DIAGNOSTIC_COUNT];
  uint32_t diagnostic_saturated_mask;
} oai_memprof_process_handoff_thread_v1_t;

typedef struct oai_memprof_process_handoff_v1_s {
  oai_memprof_container_v1_opening_header_t opening_header;
  oai_memprof_stream_writer_result_t writer;
  oai_memprof_clock_sample_v1_t opening_sample;
  const uint8_t *bootstrap_bytes;
  size_t bootstrap_size;
  const uint8_t *maps_bytes;
  size_t maps_size;
  const oai_memprof_process_handoff_thread_v1_t *threads;
  size_t thread_count;
  uint32_t ring_records;
  uint32_t flush_records;
  uint64_t flush_interval_ns;
  uint16_t realloc_zero_policy_id;
  uint64_t unregistered_active_thread_failures;
  uint64_t writer_io_or_finalization_failures;
  uint64_t diagnostic_saturation_transitions;
  /* Bits 0 and 1 bind reasons 2 and 3; every other bit is reserved. */
  uint32_t registration_diagnostic_saturated_mask;
  uint8_t bootstrap_sha256[32];
  uint8_t maps_sha256[32];
  uint8_t opening_header_sha256[32];
  /* SHA-256 of the exact closed stream prefix reported by writer.stream_bytes. */
  uint8_t prefix_sha256[32];
  uint8_t handoff_sha256[32];
} oai_memprof_process_handoff_v1_t;

/* Return the exact required wire size without mutating output on failure. */
oai_memprof_process_handoff_status_t oai_memprof_process_handoff_v1_size(size_t bootstrap_size,
                                                                         size_t maps_size,
                                                                         size_t thread_count,
                                                                         size_t *wire_size);

/* Encode a canonical self-hashed handoff. Output is unchanged on validation error. */
oai_memprof_process_handoff_status_t oai_memprof_process_handoff_v1_encode(const oai_memprof_process_handoff_v1_t *handoff,
                                                                           uint8_t *wire,
                                                                           size_t wire_size);

/*
 * Decode and verify a canonical handoff. The immutable byte sections in the
 * result point into wire and remain valid only while wire remains unchanged.
 * Result and caller thread storage are unchanged on every error.
 */
oai_memprof_process_handoff_status_t oai_memprof_process_handoff_v1_decode(oai_memprof_process_handoff_v1_t *handoff,
                                                                           oai_memprof_process_handoff_thread_v1_t *threads,
                                                                           size_t thread_capacity,
                                                                           const uint8_t *wire,
                                                                           size_t wire_size);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_PROCESS_HANDOFF_H */
