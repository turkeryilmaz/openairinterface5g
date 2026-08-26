/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_PROCESS_SESSION_H
#define OAI_MEMPROF_PROCESS_SESSION_H

#include "oai_memprof_process_handoff.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OAI_MEMPROF_PROCESS_SESSION_V1_MAX_FILE_NAME_BYTES UINT32_C(127)

typedef enum oai_memprof_process_session_status_e {
  OAI_MEMPROF_PROCESS_SESSION_OK = 0,
  OAI_MEMPROF_PROCESS_SESSION_INVALID_ARGUMENT,
  OAI_MEMPROF_PROCESS_SESSION_INVALID_CONFIGURATION,
  OAI_MEMPROF_PROCESS_SESSION_NO_MEMORY,
  OAI_MEMPROF_PROCESS_SESSION_SYSTEM_ERROR,
  OAI_MEMPROF_PROCESS_SESSION_CLOCK_ERROR,
  OAI_MEMPROF_PROCESS_SESSION_WRITER_ERROR,
  OAI_MEMPROF_PROCESS_SESSION_RUNTIME_ERROR,
  OAI_MEMPROF_PROCESS_SESSION_HANDOFF_ERROR,
  OAI_MEMPROF_PROCESS_SESSION_IO_ERROR,
  OAI_MEMPROF_PROCESS_SESSION_INVALID_STATE,
} oai_memprof_process_session_status_t;

typedef struct oai_memprof_process_session_s oai_memprof_process_session_t;

typedef struct oai_memprof_process_session_config_s {
  int directory_fd;
  const char *stream_file_name;
  const char *handoff_file_name;
  const uint8_t *configuration_bytes;
  size_t configuration_size;
  oai_memprof_active_runtime_config_t runtime;
  oai_memprof_container_v1_opening_header_t opening_header;
  uint32_t flush_records;
  uint64_t flush_interval_ns;
} oai_memprof_process_session_config_t;

typedef struct oai_memprof_process_session_result_s {
  oai_memprof_process_session_status_t status;
  oai_memprof_stream_writer_result_t writer;
  uint64_t handoff_bytes;
  uint64_t handoff_device;
  uint64_t handoff_inode;
  int32_t system_errno;
  bool handoff_published;
} oai_memprof_process_session_result_t;

/*
 * Start snapshots canonical configuration and /proc/self/maps before ACTIVE,
 * binds dynamic process/clock fields into the opening header, and starts the
 * sole asynchronous writer. The output pointer is unchanged on every error.
 */
oai_memprof_process_session_status_t oai_memprof_process_session_start_v1(const oai_memprof_process_session_config_t *config,
                                                                          oai_memprof_process_session_t **session);

/*
 * Finish seals and joins the writer, snapshots every READY producer, and only
 * after a successful, closed pre-footer publishes one self-authenticating
 * handoff with O_EXCL. The session becomes invalid after this call. Result is
 * always populated for a valid session.
 */
oai_memprof_process_session_status_t oai_memprof_process_session_finish_v1(oai_memprof_process_session_t *session,
                                                                           uint64_t seal_timeout_ns,
                                                                           oai_memprof_process_session_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_PROCESS_SESSION_H */
