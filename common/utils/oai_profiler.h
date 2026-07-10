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

#define OAI_PROFILE_SCHEMA_VERSION 2U
#define OAI_PROFILE_MAX_THREADS 256
#define OAI_PROFILE_MAX_NESTING_DEPTH 64U
#define OAI_PROFILE_ABSOLUTE_SLOT_UNKNOWN INT64_C(-1)

typedef enum {
  OAI_PROFILE_EVENT_KIND_UNKNOWN = 0,
  OAI_PROFILE_EVENT_KIND_DURATION,
  OAI_PROFILE_EVENT_KIND_INSTANT,
} oai_profile_event_kind_t;

typedef enum {
  OAI_PROFILE_DETAIL_BOUNDARY = 1,
  OAI_PROFILE_DETAIL_STAGE,
  OAI_PROFILE_DETAIL_KERNEL,
} oai_profile_detail_t;

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

typedef struct {
  int64_t absolute_slot;
  uint64_t correlation_id;
  uint64_t parent_id;
} oai_profile_context_t;

typedef struct {
  uint64_t start_tick;
  uint64_t span_id;
  uint64_t parent_id;
  uint64_t correlation_id;
  int64_t absolute_slot;
  int32_t cpu_start;
  uint16_t depth;
  uint16_t thread_index;
  uint8_t stack_registered;
  uint8_t reserved[3];
} oai_profile_span_t;

typedef struct {
  const char *name;
  const char *role;
  const char *subsystem;
  const char *event_class;
  oai_profile_event_kind_t default_kind;
  oai_profile_detail_t detail;
  const char *aux_name[4];
  const char *aux_unit[4];
  const char *flags_name;
} oai_profile_event_descriptor_t;

extern int oai_profiler_enabled;

static inline bool oai_profiler_is_enabled(void)
{
  return __atomic_load_n(&oai_profiler_enabled, __ATOMIC_ACQUIRE) != 0;
}

void oai_profiler_init(const char *process_name,
                       int argc,
                       char **argv,
                       bool enable_from_cli,
                       const char *profile_dir,
                       uint32_t buffer_records,
                       uint32_t flush_us);
void oai_profiler_shutdown(void);
const char *oai_profiler_event_name(oai_profile_event_id_t event_id);
const oai_profile_event_descriptor_t *oai_profiler_event_descriptor(oai_profile_event_id_t event_id);
const char *oai_profiler_event_kind_name(oai_profile_event_kind_t kind);
void oai_profiler_record_setting(const char *key, const char *value, const char *source);
void oai_profiler_record_setting_int(const char *key, int64_t value, const char *source);
void oai_profiler_register_thread(void);
void oai_profiler_set_context(oai_profile_context_t context);
oai_profile_context_t oai_profiler_get_context(void);
void oai_profiler_clear_context(void);
uint64_t oai_profiler_next_correlation_id(void);
oai_profile_span_t oai_profiler_span_start_enabled(void);

static inline oai_profile_span_t oai_profiler_span_start(void)
{
  oai_profile_span_t span = {0};
  if (oai_profiler_is_enabled())
    span = oai_profiler_span_start_enabled();
  return span;
}

static inline uint64_t oai_profiler_start(void)
{
  return oai_profiler_is_enabled() ? rdtsc_oai() : 0;
}

void oai_profiler_record_span(oai_profile_event_id_t event_id,
                              oai_profile_span_t span,
                              int frame,
                              int slot,
                              int64_t aux0,
                              int64_t aux1,
                              int64_t aux2,
                              int64_t aux3,
                              uint32_t flags);
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

#define OAI_PROFILE_START(var) oai_profile_span_t var = oai_profiler_span_start()
#define OAI_PROFILE_STOP(event_id, span, frame, slot, aux0, aux1, aux2, aux3, flags) \
  oai_profiler_record_span((event_id), (span), (frame), (slot), (aux0), (aux1), (aux2), (aux3), (flags))
#define OAI_PROFILE_MARK(event_id, frame, slot, aux0, aux1, aux2, aux3, flags) \
  oai_profiler_record_instant((event_id), (frame), (slot), (aux0), (aux1), (aux2), (aux3), (flags))

#ifdef __cplusplus
}
#endif

#endif /* OAI_PROFILER_H */
