/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_STREAM_FINALIZER_H
#define OAI_MEMPROF_STREAM_FINALIZER_H

#include "oai_memprof_stream_writer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum oai_memprof_stream_finalizer_status_e {
  OAI_MEMPROF_STREAM_FINALIZER_OK = 0,
  OAI_MEMPROF_STREAM_FINALIZER_INVALID_ARGUMENT,
  OAI_MEMPROF_STREAM_FINALIZER_INVALID_CONFIGURATION,
  OAI_MEMPROF_STREAM_FINALIZER_PREFOOTER_INVALID,
  OAI_MEMPROF_STREAM_FINALIZER_IDENTITY_MISMATCH,
  OAI_MEMPROF_STREAM_FINALIZER_NO_MEMORY,
  OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR,
  OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR,
  OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR,
  OAI_MEMPROF_STREAM_FINALIZER_RUNTIME_ERROR,
  OAI_MEMPROF_STREAM_FINALIZER_STREAM_LIMIT,
  OAI_MEMPROF_STREAM_FINALIZER_AUTHENTICATION_MISMATCH,
} oai_memprof_stream_finalizer_status_t;

typedef struct oai_memprof_stream_finalizer_config_s {
  int directory_fd;
  const char *file_name;
  oai_memprof_stream_writer_result_t prefooter;
  oai_memprof_container_v1_trailer_header_t trailer_header;
  const oai_memprof_container_v1_event_total_entry_t *event_entries;
  size_t event_entry_count;
  const oai_memprof_container_v1_diagnostic_total_entry_t *diagnostic_entries;
  size_t diagnostic_entry_count;
  const oai_memprof_container_v1_object_entry_t *object_entries;
  size_t object_entry_count;
  /*
   * Required by offline finalization. These are the independently
   * authenticated SHA-256 values for the exact writer-reported prefix and
   * its opening header. The live entry point intentionally ignores them.
   */
  const uint8_t *authenticated_prefix_sha256;
  const uint8_t *authenticated_opening_header_sha256;
} oai_memprof_stream_finalizer_config_t;

typedef struct oai_memprof_stream_finalizer_result_s {
  oai_memprof_stream_finalizer_status_t status;
  oai_memprof_core_status_t runtime_status;
  uint64_t stream_bytes;
  uint64_t appended_bytes;
  uint64_t file_device;
  uint64_t file_inode;
  int32_t system_errno;
  bool stream_verified;
  bool runtime_complete;
} oai_memprof_stream_finalizer_result_t;

/*
 * Finalize one writer-produced DRAINING pre-footer in place. The function
 * reopens only the exact single-link regular file reported by the writer,
 * revalidates the opening/chunks/events and terminal tables before appending,
 * writes the canonical trailer/footer with short-write completion, fsyncs and
 * rereads the complete stream, and publishes runtime COMPLETE only after the
 * path and bytes are durably verified. Any failure retains the observed file
 * bytes and leaves the runtime non-COMPLETE.
 */
oai_memprof_stream_finalizer_status_t oai_memprof_stream_finalize_v1(const oai_memprof_stream_finalizer_config_t *config,
                                                                     oai_memprof_stream_finalizer_result_t *result);

/*
 * Complete a retained writer result after the producing process has stopped.
 * The byte, inode, table, append, synchronization, and whole-stream checks are
 * identical to the live entry point. No process-local ACTIVE runtime state is
 * read or changed; runtime_complete therefore remains false even when the
 * durable stream has lifecycle COMPLETE. The caller must obtain the pre-footer
 * result from an independently authenticated handoff.
 */
oai_memprof_stream_finalizer_status_t oai_memprof_stream_finalize_offline_v1(const oai_memprof_stream_finalizer_config_t *config,
                                                                             oai_memprof_stream_finalizer_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_STREAM_FINALIZER_H */
