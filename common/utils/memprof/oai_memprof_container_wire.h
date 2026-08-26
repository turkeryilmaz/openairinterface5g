/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_CONTAINER_WIRE_H
#define OAI_MEMPROF_CONTAINER_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Schema-v1 container sizes. These are wire sizes, never native structure
 * sizes. Native structures in this header are deliberately unpacked.
 */
enum oai_memprof_container_v1_wire_size {
  OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE = 512,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE = 32,
  OAI_MEMPROF_CONTAINER_V1_EVENT_RECORD_SIZE = 96,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE = 256,
  OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE = 32,
  OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE = 32,
  OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE = 64,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE = 256,
};

enum oai_memprof_container_v1_opening_header_offset {
  OAI_MEMPROF_CONTAINER_V1_OPENING_MAGIC_OFFSET = 0,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CONTAINER_MAJOR_OFFSET = 8,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CONTAINER_MINOR_OFFSET = 10,
  OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_BYTES_OFFSET = 12,
  OAI_MEMPROF_CONTAINER_V1_OPENING_EVENT_RECORD_BYTES_OFFSET = 14,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CHUNK_HEADER_BYTES_OFFSET = 16,
  OAI_MEMPROF_CONTAINER_V1_OPENING_MINIMUM_READER_MINOR_OFFSET = 18,
  OAI_MEMPROF_CONTAINER_V1_OPENING_REQUIRED_FEATURES_OFFSET = 20,
  OAI_MEMPROF_CONTAINER_V1_OPENING_ENDIAN_MARKER_OFFSET = 24,
  OAI_MEMPROF_CONTAINER_V1_OPENING_PAGE_SIZE_BYTES_OFFSET = 28,
  OAI_MEMPROF_CONTAINER_V1_OPENING_POINTER_WIDTH_BYTES_OFFSET = 32,
  OAI_MEMPROF_CONTAINER_V1_OPENING_SCOPE_KIND_OFFSET = 33,
  OAI_MEMPROF_CONTAINER_V1_OPENING_ROLE_KIND_OFFSET = 34,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CLOCK_KIND_OFFSET = 35,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_KIND_OFFSET = 36,
  OAI_MEMPROF_CONTAINER_V1_OPENING_RESERVED_ZERO_0_OFFSET = 38,
  OAI_MEMPROF_CONTAINER_V1_OPENING_PROCESS_GENERATION_OFFSET = 40,
  OAI_MEMPROF_CONTAINER_V1_OPENING_COUNTER_FREQUENCY_NUMERATOR_OFFSET = 48,
  OAI_MEMPROF_CONTAINER_V1_OPENING_COUNTER_FREQUENCY_DENOMINATOR_OFFSET = 56,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_ERROR_BOUND_NS_OFFSET = 64,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_SPAN_NS_OFFSET = 72,
  OAI_MEMPROF_CONTAINER_V1_OPENING_START_COUNTER_OFFSET = 80,
  OAI_MEMPROF_CONTAINER_V1_OPENING_START_MONOTONIC_RAW_NS_OFFSET = 88,
  OAI_MEMPROF_CONTAINER_V1_OPENING_START_REALTIME_UNIX_NS_OFFSET = 96,
  OAI_MEMPROF_CONTAINER_V1_OPENING_PID_OFFSET = 104,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CONFIGURED_THREAD_CAPACITY_OFFSET = 108,
  OAI_MEMPROF_CONTAINER_V1_OPENING_RUN_UUID_OFFSET = 112,
  OAI_MEMPROF_CONTAINER_V1_OPENING_PROCESS_UUID_OFFSET = 128,
  OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_KIND_OFFSET = 144,
  OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_ALGORITHM_OFFSET = 146,
  OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_LENGTH_OFFSET = 148,
  OAI_MEMPROF_CONTAINER_V1_OPENING_RESERVED_ZERO_1_OFFSET = 150,
  OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_VALUE_OFFSET = 152,
  OAI_MEMPROF_CONTAINER_V1_OPENING_PRIMARY_BINARY_SHA256_OFFSET = 184,
  OAI_MEMPROF_CONTAINER_V1_OPENING_SCHEMA_BUNDLE_DEFINITION_SHA256_OFFSET = 216,
  OAI_MEMPROF_CONTAINER_V1_OPENING_API_CATALOG_DEFINITION_SHA256_OFFSET = 248,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CALLSITE_CATALOG_DEFINITION_SHA256_OFFSET = 280,
  OAI_MEMPROF_CONTAINER_V1_OPENING_CONFIGURATION_INSTANCE_SHA256_OFFSET = 312,
  OAI_MEMPROF_CONTAINER_V1_OPENING_PRIMARY_BUILD_ID_SHA256_OFFSET = 344,
  OAI_MEMPROF_CONTAINER_V1_OPENING_RESERVED_ZERO_2_OFFSET = 376,
  OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_CRC32C_OFFSET = 508,
};

enum oai_memprof_container_v1_chunk_header_offset {
  OAI_MEMPROF_CONTAINER_V1_CHUNK_MAGIC_OFFSET = 0,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_MAJOR_OFFSET = 4,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_MINOR_OFFSET = 5,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_BYTES_OFFSET = 6,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_SEQUENCE_OFFSET = 8,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_RECORD_COUNT_OFFSET = 16,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_PAYLOAD_BYTES_OFFSET = 20,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_PAYLOAD_CRC32C_OFFSET = 24,
  OAI_MEMPROF_CONTAINER_V1_CHUNK_FLAGS_OFFSET = 28,
};

enum oai_memprof_container_v1_trailer_header_offset {
  OAI_MEMPROF_CONTAINER_V1_TRAILER_MAGIC_OFFSET = 0,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_SCHEMA_MAJOR_OFFSET = 16,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_SCHEMA_MINOR_OFFSET = 18,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_FIXED_HEADER_BYTES_OFFSET = 20,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_BODY_BYTES_OFFSET = 24,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_PROCESS_GENERATION_OFFSET = 32,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_SCOPE_KIND_OFFSET = 40,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_LIFECYCLE_STATE_OFFSET = 42,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_PAYLOAD_WRITER_STATE_OFFSET = 44,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_FINALIZATION_STAGE_OFFSET = 46,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_TERMINAL_FLAGS_OFFSET = 48,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_CHUNK_COUNT_OFFSET = 56,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_RECORD_COUNT_OFFSET = 64,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_PAYLOAD_BYTES_OFFSET = 72,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_FIRST_CHUNK_OFFSET = 80,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_CHUNKS_END_OFFSET = 88,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_GENERATION_OFFSET = 96,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_START_COUNTER_OFFSET = 104,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_CUTOFF_BEFORE_COUNTER_OFFSET = 112,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_CUTOFF_AFTER_COUNTER_OFFSET = 120,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_QUIESCENCE_COMPLETE_COUNTER_OFFSET = 128,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_COUNTER_OFFSET = 136,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_START_MONOTONIC_RAW_NS_OFFSET = 144,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_MONOTONIC_RAW_NS_OFFSET = 152,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_REALTIME_UNIX_NS_OFFSET = 160,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_ENTRY_COUNT_OFFSET = 168,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_ENTRY_BYTES_OFFSET = 172,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_ENTRY_COUNT_OFFSET = 176,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_ENTRY_BYTES_OFFSET = 180,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_ENTRY_COUNT_OFFSET = 184,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_ENTRY_BYTES_OFFSET = 188,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_TABLE_OFFSET_OFFSET = 192,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_TABLE_OFFSET_OFFSET = 200,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_TABLE_OFFSET_OFFSET = 208,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_TERMINAL_REASON_CODE_OFFSET = 216,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_RESERVED_ZERO_0_OFFSET = 220,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_LOSS_SUM_OFFSET = 224,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_BYPASS_SUM_OFFSET = 232,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_SATURATED_COUNTER_INSTANCES_OFFSET = 240,
  OAI_MEMPROF_CONTAINER_V1_TRAILER_RESERVED_ZERO_1_OFFSET = 248,
};

enum oai_memprof_container_v1_event_total_entry_offset {
  OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_EVENT_KIND_OFFSET = 0,
  OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_API_ID_OFFSET = 2,
  OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_RESERVED_ZERO_0_OFFSET = 4,
  OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_RECORD_COUNT_OFFSET = 8,
  OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_RESERVED_ZERO_1_OFFSET = 16,
};

enum oai_memprof_container_v1_diagnostic_total_entry_offset {
  OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_REASON_ID_OFFSET = 0,
  OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_CLASS_FLAGS_OFFSET = 2,
  OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SUMMARY_FLAGS_OFFSET = 4,
  OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SATURATING_TOTAL_OFFSET = 8,
  OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_NONZERO_COUNTER_INSTANCES_OFFSET = 16,
  OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SATURATED_COUNTER_INSTANCES_OFFSET = 20,
  OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_RESERVED_ZERO_OFFSET = 24,
};

enum oai_memprof_container_v1_object_entry_offset {
  OAI_MEMPROF_CONTAINER_V1_OBJECT_KIND_OFFSET = 0,
  OAI_MEMPROF_CONTAINER_V1_OBJECT_FORMAT_ID_OFFSET = 2,
  OAI_MEMPROF_CONTAINER_V1_OBJECT_FLAGS_OFFSET = 4,
  OAI_MEMPROF_CONTAINER_V1_OBJECT_SCHEMA_REVISION_OFFSET = 8,
  OAI_MEMPROF_CONTAINER_V1_OBJECT_RESERVED_ZERO_OFFSET = 12,
  OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_COUNT_OFFSET = 16,
  OAI_MEMPROF_CONTAINER_V1_OBJECT_BYTE_COUNT_OFFSET = 24,
  OAI_MEMPROF_CONTAINER_V1_OBJECT_SHA256_OFFSET = 32,
};

enum oai_memprof_container_v1_footer_offset {
  OAI_MEMPROF_CONTAINER_V1_FOOTER_MAGIC_OFFSET = 0,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_SCHEMA_MAJOR_OFFSET = 16,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_SCHEMA_MINOR_OFFSET = 18,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_BYTES_OFFSET = 20,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_FLAGS_OFFSET = 24,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_OFFSET_OFFSET = 32,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_BODY_BYTES_OFFSET = 40,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_STREAM_BYTES_OFFSET = 48,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_PREFIX_BYTES_OFFSET = 56,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_HEADER_BYTES_OFFSET = 64,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_CHUNK_COUNT_OFFSET = 72,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_RECORD_COUNT_OFFSET = 80,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_RESERVED_ZERO_0_OFFSET = 88,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_PREFIX_SHA256_OFFSET = 96,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_BODY_SHA256_OFFSET = 128,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_OPENING_HEADER_SHA256_OFFSET = 160,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_RESERVED_ZERO_1_OFFSET = 192,
  OAI_MEMPROF_CONTAINER_V1_FOOTER_FOOTER_SHA256_OFFSET = 224,
};

#define OAI_MEMPROF_CONTAINER_V1_REQUIRED_FEATURES UINT32_C(0x0000000f)
#define OAI_MEMPROF_CONTAINER_V1_ENDIAN_MARKER UINT32_C(0x01020304)
#define OAI_MEMPROF_CONTAINER_V1_MAX_CHUNK_RECORD_COUNT UINT32_C(44739242)
#define OAI_MEMPROF_CONTAINER_V1_MAX_EVENT_ENTRIES UINT32_C(16384)
#define OAI_MEMPROF_CONTAINER_V1_MAX_DIAGNOSTIC_ENTRIES UINT32_C(4096)
#define OAI_MEMPROF_CONTAINER_V1_MAX_OBJECT_ENTRIES UINT32_C(64)
#define OAI_MEMPROF_CONTAINER_V1_MAX_TRAILER_BODY_BYTES UINT64_C(1048576)
#define OAI_MEMPROF_CONTAINER_V1_FOOTER_FLAGS UINT64_C(0x000000000000000f)
#define OAI_MEMPROF_CONTAINER_V1_TERMINAL_FLAGS_MASK UINT64_C(0x000000000001ffff)
#define OAI_MEMPROF_CONTAINER_V1_COMPLETE_REQUIRED_FLAGS UINT64_C(0x0000000000000fff)
#define OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_CLASS_FLAGS_MASK UINT16_C(0x03ff)
#define OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_SUMMARY_FLAGS_MASK UINT32_C(0x00000003)
#define OAI_MEMPROF_CONTAINER_V1_OBJECT_FLAGS_MASK UINT32_C(0x0000001f)
#define OAI_MEMPROF_CONTAINER_V1_MAX_OBJECT_ENTRY_COUNT UINT64_C(16777216)
#define OAI_MEMPROF_CONTAINER_V1_MAX_OBJECT_BYTE_COUNT UINT64_C(268435456)

enum oai_memprof_container_v1_scope_kind {
  OAI_MEMPROF_CONTAINER_V1_SCOPE_MEASUREMENT_INTERVAL = 1,
  OAI_MEMPROF_CONTAINER_V1_SCOPE_PROCESS_LIFETIME = 2,
};

enum oai_memprof_container_v1_role_kind {
  OAI_MEMPROF_CONTAINER_V1_ROLE_GNB = 1,
  OAI_MEMPROF_CONTAINER_V1_ROLE_NR_UE = 2,
};

enum oai_memprof_container_v1_clock_kind {
  OAI_MEMPROF_CONTAINER_V1_CLOCK_X86_TSC = 1,
  OAI_MEMPROF_CONTAINER_V1_CLOCK_AARCH64_CNTVCT_EL0 = 2,
};

enum oai_memprof_container_v1_calibration_kind {
  OAI_MEMPROF_CONTAINER_V1_CALIBRATION_EXACT_RATE = 1,
  OAI_MEMPROF_CONTAINER_V1_CALIBRATION_MEASURED_AFFINE = 2,
};

enum oai_memprof_container_v1_source_object_kind {
  OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_COMMIT = 1,
};

enum oai_memprof_container_v1_source_object_algorithm {
  OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_SHA1 = 1,
  OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_SHA256 = 2,
};

enum oai_memprof_container_v1_terminal_lifecycle_state {
  OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_COMPLETE = 5,
  OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED = 6,
  OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_INCOMPLETE = 7,
};

enum oai_memprof_container_v1_payload_writer_state {
  /* Values 1..4, 7, and 8 are named by the lifecycle model but forbidden in a trusted terminal body. */
  OAI_MEMPROF_CONTAINER_V1_WRITER_NOT_STARTED = 1,
  OAI_MEMPROF_CONTAINER_V1_WRITER_OPEN = 2,
  OAI_MEMPROF_CONTAINER_V1_WRITER_DRAINING = 3,
  OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_SYNCED = 4,
  OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED = 5,
  OAI_MEMPROF_CONTAINER_V1_WRITER_IO_FAILED_AT_SAFE_BOUNDARY_AND_CLOSED_VERIFIED = 6,
  OAI_MEMPROF_CONTAINER_V1_WRITER_ABORTED_BEFORE_OPEN = 7,
  OAI_MEMPROF_CONTAINER_V1_WRITER_CLOSE_UNVERIFIED = 8,
};

enum oai_memprof_container_v1_terminal_reason_code {
  OAI_MEMPROF_CONTAINER_V1_REASON_NONE = 0,
  OAI_MEMPROF_CONTAINER_V1_REASON_QUIESCENCE_TIMEOUT = 1,
  OAI_MEMPROF_CONTAINER_V1_REASON_RING_DRAIN_FAILED = 2,
  OAI_MEMPROF_CONTAINER_V1_REASON_CATALOG_FREEZE_FAILED = 3,
  OAI_MEMPROF_CONTAINER_V1_REASON_DIAGNOSTICS_FREEZE_FAILED = 4,
  OAI_MEMPROF_CONTAINER_V1_REASON_PAYLOAD_SYNC_FAILED_AT_SAFE_BOUNDARY = 5,
  OAI_MEMPROF_CONTAINER_V1_REASON_PAYLOAD_IO_FAILED_AT_SAFE_BOUNDARY = 6,
  OAI_MEMPROF_CONTAINER_V1_REASON_COUNTER_OR_TIME_INVALID = 7,
  OAI_MEMPROF_CONTAINER_V1_REASON_OPERATOR_CANCELLED = 8,
  OAI_MEMPROF_CONTAINER_V1_REASON_UNSUPPORTED_SCOPE = 9,
};

enum oai_memprof_container_v1_finalization_stage {
  OAI_MEMPROF_CONTAINER_V1_FINALIZATION_ACTIVE_ONLY = 0,
  OAI_MEMPROF_CONTAINER_V1_FINALIZATION_ADMISSION_SEALED = 1,
  OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRODUCERS_QUIESCED = 2,
  OAI_MEMPROF_CONTAINER_V1_FINALIZATION_RINGS_DRAINED_AND_CALLSITES_INTERNED = 3,
  OAI_MEMPROF_CONTAINER_V1_FINALIZATION_CATALOGS_FROZEN = 4,
  OAI_MEMPROF_CONTAINER_V1_FINALIZATION_DIAGNOSTICS_FROZEN = 5,
  OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRE_SYNC_TERMINAL_MATERIAL_FROZEN = 6,
};

typedef enum oai_memprof_container_v1_status_e {
  OAI_MEMPROF_CONTAINER_V1_OK = 0,
  OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT,
  OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE,
  OAI_MEMPROF_CONTAINER_V1_BAD_MAGIC,
  OAI_MEMPROF_CONTAINER_V1_BAD_CHECKSUM,
  OAI_MEMPROF_CONTAINER_V1_UNSUPPORTED_VERSION,
  OAI_MEMPROF_CONTAINER_V1_INVALID_FIXED_FIELD,
  OAI_MEMPROF_CONTAINER_V1_NONZERO_RESERVED,
  OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM,
  OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE,
  OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION,
  OAI_MEMPROF_CONTAINER_V1_INTEGER_OVERFLOW,
  OAI_MEMPROF_CONTAINER_V1_PAYLOAD_SIZE_MISMATCH,
} oai_memprof_container_v1_status_t;

typedef struct oai_memprof_container_v1_opening_header_s {
  uint32_t page_size_bytes;
  uint8_t scope_kind;
  uint8_t role_kind;
  uint8_t clock_kind;
  uint16_t calibration_kind;
  uint64_t process_generation;
  uint64_t counter_frequency_numerator;
  uint64_t counter_frequency_denominator;
  uint64_t calibration_error_bound_ns;
  uint64_t calibration_span_ns;
  uint64_t start_counter;
  uint64_t start_monotonic_raw_ns;
  uint64_t start_realtime_unix_ns;
  uint32_t pid;
  uint32_t configured_thread_capacity;
  uint8_t run_uuid[16];
  uint8_t process_uuid[16];
  uint16_t source_object_kind;
  uint16_t source_object_algorithm;
  uint16_t source_object_length;
  uint8_t source_object_value[32];
  uint8_t primary_binary_sha256[32];
  uint8_t schema_bundle_definition_sha256[32];
  uint8_t api_catalog_definition_sha256[32];
  uint8_t callsite_catalog_definition_sha256[32];
  uint8_t configuration_instance_sha256[32];
  uint8_t primary_build_id_sha256[32];
} oai_memprof_container_v1_opening_header_t;

typedef struct oai_memprof_container_v1_chunk_header_s {
  uint64_t writer_chunk_sequence;
  uint32_t record_count;
} oai_memprof_container_v1_chunk_header_t;

typedef struct oai_memprof_container_v1_trailer_header_s {
  uint64_t trailer_body_bytes;
  uint64_t process_generation;
  uint16_t scope_kind;
  uint16_t lifecycle_state;
  uint16_t payload_writer_state;
  uint16_t finalization_stage;
  uint64_t terminal_flags;
  uint64_t chunk_count;
  uint64_t record_count;
  uint64_t payload_bytes;
  uint64_t first_chunk_offset;
  uint64_t chunks_end_offset;
  uint64_t active_generation;
  uint64_t active_start_counter;
  uint64_t cutoff_before_counter;
  uint64_t cutoff_after_counter;
  uint64_t quiescence_complete_counter;
  uint64_t final_counter;
  uint64_t active_start_monotonic_raw_ns;
  uint64_t final_monotonic_raw_ns;
  uint64_t final_realtime_unix_ns;
  uint32_t event_entry_count;
  uint32_t diagnostic_entry_count;
  uint32_t object_entry_count;
  uint64_t event_table_offset;
  uint64_t diagnostic_table_offset;
  uint64_t object_table_offset;
  uint32_t terminal_reason_code;
  uint64_t diagnostic_loss_sum;
  uint64_t diagnostic_bypass_sum;
  uint64_t saturated_counter_instances;
} oai_memprof_container_v1_trailer_header_t;

typedef struct oai_memprof_container_v1_event_total_entry_s {
  uint16_t event_kind;
  uint16_t api_id;
  uint64_t record_count;
} oai_memprof_container_v1_event_total_entry_t;

typedef struct oai_memprof_container_v1_diagnostic_total_entry_s {
  uint16_t reason_id;
  uint16_t class_flags;
  uint32_t summary_flags;
  uint64_t saturating_total;
  uint32_t nonzero_counter_instances;
  uint32_t saturated_counter_instances;
} oai_memprof_container_v1_diagnostic_total_entry_t;

typedef struct oai_memprof_container_v1_object_entry_s {
  uint16_t object_kind;
  uint16_t format_id;
  uint32_t object_flags;
  uint32_t schema_revision;
  uint64_t entry_count;
  uint64_t byte_count;
  uint8_t sha256[32];
} oai_memprof_container_v1_object_entry_t;

typedef struct oai_memprof_container_v1_footer_s {
  uint64_t trailer_offset;
  uint64_t trailer_body_bytes;
  uint64_t stream_bytes;
  uint64_t prefix_bytes;
  uint64_t chunk_count;
  uint64_t record_count;
  uint8_t prefix_sha256[32];
  uint8_t trailer_body_sha256[32];
  uint8_t opening_header_sha256[32];
  uint8_t footer_sha256[32];
} oai_memprof_container_v1_footer_t;

/*
 * CRC-32C/Castagnoli with the schema-v1 parameters. NULL data is accepted
 * only when data_size is zero. crc32c is left unchanged on error.
 */
oai_memprof_container_v1_status_t oai_memprof_container_v1_crc32c(const uint8_t *data, size_t data_size, uint32_t *crc32c);

/*
 * Dependency-free SHA-256 for exact container and external-object bindings.
 * NULL data is accepted only when data_size is zero. The 32-byte digest is
 * transactional and remains unchanged on error.
 */
oai_memprof_container_v1_status_t oai_memprof_container_v1_sha256(const uint8_t *data, size_t data_size, uint8_t digest[32]);

/*
 * Every encoder and decoder below is transactional: its caller-owned output
 * remains byte-for-byte unchanged on error. Exact wire sizes are required.
 *
 * The codec enforces fixed byte grammar, reserved zeros, bounded arithmetic,
 * locally decidable cross-field relations, and the CRC-32C fields. It does not
 * interpret event/catalog IDs or JSON. SHA-256 arrays are otherwise opaque
 * values whose whole-stream comparison belongs to the later verifier.
 *
 * Footer encoding is the exception: footer_sha256 is either all-zero (unset)
 * or the exact SHA-256 of canonical encoded footer bytes [0,224). A mismatched
 * nonzero input returns BAD_CHECKSUM. The encoder always emits the calculated
 * digest. Footer decoding verifies that digest after magic and before version
 * or fixed-field validation.
 */
oai_memprof_container_v1_status_t oai_memprof_container_v1_opening_header_encode(
    const oai_memprof_container_v1_opening_header_t *header,
    uint8_t *wire,
    size_t wire_size);
oai_memprof_container_v1_status_t oai_memprof_container_v1_opening_header_decode(oai_memprof_container_v1_opening_header_t *header,
                                                                                 const uint8_t *wire,
                                                                                 size_t wire_size);

/* The payload is required, exactly sized, and covered by the stored CRC-32C. */
oai_memprof_container_v1_status_t oai_memprof_container_v1_chunk_header_encode(
    const oai_memprof_container_v1_chunk_header_t *header,
    const uint8_t *payload,
    size_t payload_size,
    uint8_t *wire,
    size_t wire_size);
oai_memprof_container_v1_status_t oai_memprof_container_v1_chunk_header_decode(oai_memprof_container_v1_chunk_header_t *header,
                                                                               const uint8_t *wire,
                                                                               size_t wire_size,
                                                                               const uint8_t *payload,
                                                                               size_t payload_size);

oai_memprof_container_v1_status_t oai_memprof_container_v1_trailer_header_encode(
    const oai_memprof_container_v1_trailer_header_t *header,
    uint8_t *wire,
    size_t wire_size);
oai_memprof_container_v1_status_t oai_memprof_container_v1_trailer_header_decode(oai_memprof_container_v1_trailer_header_t *header,
                                                                                 const uint8_t *wire,
                                                                                 size_t wire_size);

oai_memprof_container_v1_status_t oai_memprof_container_v1_event_total_entry_encode(
    const oai_memprof_container_v1_event_total_entry_t *entry,
    uint8_t *wire,
    size_t wire_size);
oai_memprof_container_v1_status_t oai_memprof_container_v1_event_total_entry_decode(
    oai_memprof_container_v1_event_total_entry_t *entry,
    const uint8_t *wire,
    size_t wire_size);

oai_memprof_container_v1_status_t oai_memprof_container_v1_diagnostic_total_entry_encode(
    const oai_memprof_container_v1_diagnostic_total_entry_t *entry,
    uint8_t *wire,
    size_t wire_size);
oai_memprof_container_v1_status_t oai_memprof_container_v1_diagnostic_total_entry_decode(
    oai_memprof_container_v1_diagnostic_total_entry_t *entry,
    const uint8_t *wire,
    size_t wire_size);

oai_memprof_container_v1_status_t oai_memprof_container_v1_object_entry_encode(const oai_memprof_container_v1_object_entry_t *entry,
                                                                               uint8_t *wire,
                                                                               size_t wire_size);
oai_memprof_container_v1_status_t oai_memprof_container_v1_object_entry_decode(oai_memprof_container_v1_object_entry_t *entry,
                                                                               const uint8_t *wire,
                                                                               size_t wire_size);

oai_memprof_container_v1_status_t oai_memprof_container_v1_footer_encode(const oai_memprof_container_v1_footer_t *footer,
                                                                         uint8_t *wire,
                                                                         size_t wire_size);
oai_memprof_container_v1_status_t oai_memprof_container_v1_footer_decode(oai_memprof_container_v1_footer_t *footer,
                                                                         const uint8_t *wire,
                                                                         size_t wire_size);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_CONTAINER_WIRE_H */
