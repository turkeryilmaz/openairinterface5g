/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_PROFILER_SYSTEM_H
#define OAI_PROFILER_SYSTEM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OAI_PROFILE_THREAD_METRIC_SCHEDSTAT (1U << 0)
#define OAI_PROFILE_THREAD_METRIC_STAT (1U << 1)
#define OAI_PROFILE_THREAD_METRIC_STATUS (1U << 2)
#define OAI_PROFILE_THREAD_METRIC_CPU_FREQUENCY (1U << 3)
#define OAI_PROFILE_THREAD_METRIC_CORE_MASK \
  (OAI_PROFILE_THREAD_METRIC_SCHEDSTAT | OAI_PROFILE_THREAD_METRIC_STAT | OAI_PROFILE_THREAD_METRIC_STATUS)

typedef struct {
  uint32_t valid_mask;
  char state;
  int32_t processor;
  int32_t priority;
  int32_t nice;
  uint32_t rt_priority;
  uint32_t policy;
  int64_t cpu_frequency_khz;
  uint64_t runtime_ns;
  uint64_t runqueue_wait_ns;
  uint64_t timeslices;
  uint64_t minor_faults;
  uint64_t major_faults;
  uint64_t user_ticks;
  uint64_t system_ticks;
  uint64_t voluntary_context_switches;
  uint64_t involuntary_context_switches;
} oai_profile_thread_metrics_snapshot_t;

typedef struct {
  bool previous_valid;
  uint64_t previous_monotonic_ns;
  oai_profile_thread_metrics_snapshot_t previous;
} oai_profile_thread_metrics_state_t;

typedef struct {
  oai_profile_thread_metrics_snapshot_t current;
  uint64_t interval_ns;
  uint64_t delta_runtime_ns;
  uint64_t delta_runqueue_wait_ns;
  uint64_t delta_timeslices;
  uint64_t delta_minor_faults;
  uint64_t delta_major_faults;
  uint64_t delta_user_ticks;
  uint64_t delta_system_ticks;
  uint64_t delta_voluntary_context_switches;
  uint64_t delta_involuntary_context_switches;
  bool delta_valid;
  bool cpu_changed_since_previous;
  char status[40];
  int error_code;
} oai_profile_thread_metrics_observation_t;

void oai_profile_read_thread_metrics(pid_t tid,
                                     uint64_t monotonic_ns,
                                     oai_profile_thread_metrics_state_t *state,
                                     oai_profile_thread_metrics_observation_t *observation);

#define OAI_PROFILE_KERNEL_ACTIVITY_INTERRUPTS (1U << 0)
#define OAI_PROFILE_KERNEL_ACTIVITY_CONTEXT_SWITCHES (1U << 1)
#define OAI_PROFILE_KERNEL_ACTIVITY_PROCESSES (1U << 2)
#define OAI_PROFILE_KERNEL_ACTIVITY_RUNNING (1U << 3)
#define OAI_PROFILE_KERNEL_ACTIVITY_BLOCKED (1U << 4)
#define OAI_PROFILE_KERNEL_ACTIVITY_SOFTIRQS (1U << 5)
#define OAI_PROFILE_KERNEL_ACTIVITY_ALL_MASK ((1U << 6) - 1U)
#define OAI_PROFILE_SOFTIRQ_CLASSES 10U

typedef struct {
  uint32_t valid_mask;
  uint64_t interrupts;
  uint64_t context_switches;
  uint64_t processes_created;
  uint64_t processes_running;
  uint64_t processes_blocked;
  uint64_t softirqs;
  uint64_t softirq_classes[OAI_PROFILE_SOFTIRQ_CLASSES];
} oai_profile_kernel_activity_snapshot_t;

typedef struct {
  bool previous_valid;
  uint64_t previous_monotonic_ns;
  oai_profile_kernel_activity_snapshot_t previous;
} oai_profile_kernel_activity_state_t;

typedef struct {
  oai_profile_kernel_activity_snapshot_t current;
  uint64_t interval_ns;
  uint64_t delta_interrupts;
  uint64_t delta_context_switches;
  uint64_t delta_processes_created;
  uint64_t delta_softirqs;
  uint64_t delta_softirq_classes[OAI_PROFILE_SOFTIRQ_CLASSES];
  bool delta_valid;
  char status[40];
  int error_code;
} oai_profile_kernel_activity_observation_t;

void oai_profile_read_kernel_activity(uint64_t monotonic_ns,
                                      oai_profile_kernel_activity_state_t *state,
                                      oai_profile_kernel_activity_observation_t *observation);
const char *oai_profile_softirq_class_name(size_t index);

typedef struct oai_profile_activity_state_s oai_profile_activity_state_t;

typedef struct {
  const char *source;
  const char *label;
  const char *description;
  int32_t cpu;
  uint64_t raw_count;
  uint64_t delta_count;
  uint64_t interval_ns;
  bool delta_valid;
  bool radio_relevant;
  const char *status;
} oai_profile_activity_observation_t;

typedef void (*oai_profile_activity_callback_t)(const oai_profile_activity_observation_t *observation, void *opaque);

typedef struct {
  uint32_t cpu_count;
  uint32_t rows;
  uint32_t parse_errors;
  int error_code;
  char status[40];
} oai_profile_activity_result_t;

oai_profile_activity_state_t *oai_profile_activity_state_create(void);
void oai_profile_activity_state_destroy(oai_profile_activity_state_t *state);
oai_profile_activity_result_t oai_profile_collect_interrupts(oai_profile_activity_state_t *state,
                                                             uint64_t monotonic_ns,
                                                             oai_profile_activity_callback_t callback,
                                                             void *opaque);
oai_profile_activity_result_t oai_profile_collect_softirqs(oai_profile_activity_state_t *state,
                                                           uint64_t monotonic_ns,
                                                           oai_profile_activity_callback_t callback,
                                                           void *opaque);

#ifdef __cplusplus
}
#endif

#endif /* OAI_PROFILER_SYSTEM_H */
