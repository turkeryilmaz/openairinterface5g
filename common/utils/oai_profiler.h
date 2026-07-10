/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_PROFILER_H
#define OAI_PROFILER_H

#include <stdbool.h>
#include <stdint.h>

#include "time_meas.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OAI_PROFILE_MAX_THREADS 256

typedef enum {
  OAI_PROFILE_EVENT_UNSPEC = 0,
  OAI_PROFILE_EVENT_UE_SLOT_LOOP,
  OAI_PROFILE_EVENT_UE_RF_READ,
  OAI_PROFILE_EVENT_UE_RF_READ_DRIFT,
  OAI_PROFILE_EVENT_UE_SCOPE_COPY,
  OAI_PROFILE_EVENT_UE_TIMING_COMPUTE,
  OAI_PROFILE_EVENT_UE_DL_PREPROCESS,
  OAI_PROFILE_EVENT_UE_DL_PROCESSING,
  OAI_PROFILE_EVENT_UE_DL_ACTOR_DISPATCH,
  OAI_PROFILE_EVENT_UE_NTN_CONFIG_APPLY,
  OAI_PROFILE_EVENT_UE_TX_SCHEDULE,
  OAI_PROFILE_EVENT_UE_TX_SLOT,
  OAI_PROFILE_EVENT_UE_TX_UL_INDICATION,
  OAI_PROFILE_EVENT_UE_TX_BARRIER_WAIT,
  OAI_PROFILE_EVENT_UE_TX_PHY_PROCEDURES,
  OAI_PROFILE_EVENT_UE_TX_RU_WRITE,
  OAI_PROFILE_EVENT_UE_RF_WRITE,
  OAI_PROFILE_EVENT_UE_TX_DEADLINE_MISS,
  OAI_PROFILE_EVENT_GNB_SLOT_INDICATION,
  OAI_PROFILE_EVENT_GNB_RX_TRIGGER,
  OAI_PROFILE_EVENT_GNB_PHY_TX,
  OAI_PROFILE_EVENT_GNB_RU_TX,
  OAI_PROFILE_EVENT_GNB_L1_TX_JOB,
  OAI_PROFILE_EVENT_GNB_L1_RX_JOB,
  OAI_PROFILE_EVENT_GNB_PRACH_QUEUE_DRAIN,
  OAI_PROFILE_EVENT_GNB_PHASE_COMP,
  OAI_PROFILE_EVENT_GNB_PHY_UESPEC_RX,
  OAI_PROFILE_EVENT_GNB_UL_INDICATION,
  OAI_PROFILE_EVENT_GNB_RF_READ,
  OAI_PROFILE_EVENT_GNB_RF_READ_ALIGN,
  OAI_PROFILE_EVENT_GNB_RF_WRITE,
  OAI_PROFILE_EVENT_MAX
} oai_profile_event_id_t;

extern volatile int oai_profiler_enabled;

void oai_profiler_init(const char *process_name,
                       int argc,
                       char **argv,
                       bool enable_from_cli,
                       const char *profile_dir,
                       uint32_t buffer_records,
                       uint32_t flush_us);
void oai_profiler_shutdown(void);
const char *oai_profiler_event_name(oai_profile_event_id_t event_id);
void oai_profiler_record_setting(const char *key, const char *value, const char *source);
void oai_profiler_record_setting_int(const char *key, int64_t value, const char *source);
void oai_profiler_register_thread(void);

static inline uint64_t oai_profiler_start(void)
{
  return oai_profiler_enabled ? rdtsc_oai() : 0;
}

void oai_profiler_record_duration(oai_profile_event_id_t event_id,
                                  uint64_t start_tick,
                                  int frame,
                                  int slot,
                                  int64_t aux0,
                                  int64_t aux1,
                                  int64_t aux2,
                                  int64_t aux3,
                                  uint32_t flags);
void oai_profiler_record_instant(oai_profile_event_id_t event_id,
                                 int frame,
                                 int slot,
                                 int64_t aux0,
                                 int64_t aux1,
                                 int64_t aux2,
                                 int64_t aux3,
                                 uint32_t flags);

#define OAI_PROFILE_START(var) uint64_t var = oai_profiler_start()
#define OAI_PROFILE_STOP(event_id, start_tick, frame, slot, aux0, aux1, aux2, aux3, flags) \
  oai_profiler_record_duration((event_id), (start_tick), (frame), (slot), (aux0), (aux1), (aux2), (aux3), (flags))
#define OAI_PROFILE_MARK(event_id, frame, slot, aux0, aux1, aux2, aux3, flags) \
  oai_profiler_record_instant((event_id), (frame), (slot), (aux0), (aux1), (aux2), (aux3), (flags))

#ifdef __cplusplus
}
#endif

#endif /* OAI_PROFILER_H */
