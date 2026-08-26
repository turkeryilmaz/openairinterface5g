/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_SOFTMODEM_SESSION_H
#define OAI_MEMPROF_SOFTMODEM_SESSION_H

#include "oai_memprof_process_session.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum oai_memprof_softmodem_session_status_e {
  OAI_MEMPROF_SOFTMODEM_SESSION_DISABLED = 0,
  OAI_MEMPROF_SOFTMODEM_SESSION_OK,
  OAI_MEMPROF_SOFTMODEM_SESSION_ALREADY_FINISHED,
  OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_ENVIRONMENT,
  OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_STATE,
  OAI_MEMPROF_SOFTMODEM_SESSION_IO_ERROR,
  OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_CONFIGURATION,
  OAI_MEMPROF_SOFTMODEM_SESSION_PROCESS_ERROR,
} oai_memprof_softmodem_session_status_t;

enum oai_memprof_softmodem_role {
  OAI_MEMPROF_SOFTMODEM_ROLE_GNB = 1,
  OAI_MEMPROF_SOFTMODEM_ROLE_NR_UE = 2,
};

/*
 * If the complete OAI_MEMPROF_SESSION_* environment is absent this is an
 * inert present-disabled call.  If any member is present, the enable member
 * must be exactly "1" and every other member must be present and valid.
 * The launcher is responsible for deriving the scalar members from the
 * canonical effective-configuration object.  Handoff and offline verification
 * independently reconcile those scalars with the object bytes.
 */
oai_memprof_softmodem_session_status_t oai_memprof_softmodem_session_start_v1(uint16_t expected_role_kind);

/*
 * Seal and publish the process handoff once.  Repeated completion is benign so
 * normal cleanup and a non-assert exit path may race without double-finishing.
 */
oai_memprof_softmodem_session_status_t oai_memprof_softmodem_session_finish_v1(oai_memprof_process_session_result_t *result);

const char *oai_memprof_softmodem_session_status_name_v1(oai_memprof_softmodem_session_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_SOFTMODEM_SESSION_H */
