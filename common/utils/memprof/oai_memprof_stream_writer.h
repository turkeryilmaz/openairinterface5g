/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_STREAM_WRITER_H
#define OAI_MEMPROF_STREAM_WRITER_H

#include "oai_memprof_active_runtime_abi.h"
#include "oai_memprof_container_wire.h"
#include "oai_memprof_clock.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OAI_MEMPROF_STREAM_WRITER_MAX_FLUSH_RECORDS UINT32_C(65536)
#define OAI_MEMPROF_STREAM_WRITER_CLOCK_BRACKET_NS UINT64_C(1000000)

typedef enum oai_memprof_stream_writer_status_e {
  OAI_MEMPROF_STREAM_WRITER_OK = 0,
  OAI_MEMPROF_STREAM_WRITER_INVALID_ARGUMENT,
  OAI_MEMPROF_STREAM_WRITER_INVALID_CONFIGURATION,
  OAI_MEMPROF_STREAM_WRITER_NO_MEMORY,
  OAI_MEMPROF_STREAM_WRITER_SYSTEM_ERROR,
  OAI_MEMPROF_STREAM_WRITER_CODEC_ERROR,
  OAI_MEMPROF_STREAM_WRITER_RUNTIME_ERROR,
  OAI_MEMPROF_STREAM_WRITER_THREAD_ERROR,
  OAI_MEMPROF_STREAM_WRITER_INVALID_STATE,
  OAI_MEMPROF_STREAM_WRITER_IO_ERROR,
  OAI_MEMPROF_STREAM_WRITER_STREAM_LIMIT,
  OAI_MEMPROF_STREAM_WRITER_CLOCK_ERROR,
} oai_memprof_stream_writer_status_t;

typedef struct oai_memprof_stream_writer_s oai_memprof_stream_writer_t;

typedef struct oai_memprof_stream_writer_config_s {
  int directory_fd;
  const char *file_name;
  oai_memprof_active_runtime_config_t runtime;
  oai_memprof_container_v1_opening_header_t opening_header;
  uint32_t flush_records;
  uint64_t flush_interval_ns;
} oai_memprof_stream_writer_config_t;

typedef struct oai_memprof_stream_writer_result_s {
  oai_memprof_stream_writer_status_t status;
  oai_memprof_core_status_t runtime_status;
  oai_memprof_core_snapshot_t runtime_snapshot;
  oai_memprof_clock_status_t clock_status;
  oai_memprof_clock_info_v1_t clock_info;
  oai_memprof_clock_sample_v1_t seal_before_sample;
  oai_memprof_clock_sample_v1_t seal_after_sample;
  oai_memprof_clock_sample_v1_t drain_complete_sample;
  oai_memprof_clock_sample_v1_t final_sample;
  uint64_t chunk_count;
  uint64_t record_count;
  uint64_t payload_bytes;
  uint64_t stream_bytes;
  uint64_t file_device;
  uint64_t file_inode;
  int32_t system_errno;
  bool prefooter_closed;
} oai_memprof_stream_writer_result_t;

/*
 * Start creates a new regular stream relative to a borrowed directory FD with
 * O_EXCL, writes and validates the frozen 512-byte opening header, bootstraps
 * the ACTIVE runtime, starts one
 * asynchronous consumer, and only then publishes ACTIVE. The output pointer is
 * unchanged on error. A partially created stream is retained on failure.
 */
oai_memprof_stream_writer_status_t oai_memprof_stream_writer_start_v1(const oai_memprof_stream_writer_config_t *config,
                                                                      oai_memprof_stream_writer_t **writer);

/*
 * Finish seals producers, joins the sole consumer, performs the final drain,
 * writes any last chunk, synchronizes and closes the pre-footer stream, and
 * leaves the runtime DRAINING. COMPLETE is reserved for the later in-process
 * catalog/trailer/footer finalizer. The writer becomes invalid after this call.
 * Result is always populated for a valid writer.
 */
oai_memprof_stream_writer_status_t oai_memprof_stream_writer_finish_v1(oai_memprof_stream_writer_t *writer,
                                                                       uint64_t seal_timeout_ns,
                                                                       oai_memprof_stream_writer_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_STREAM_WRITER_H */
