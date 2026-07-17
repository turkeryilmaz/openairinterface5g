/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE
#include "common/utils/oai_profiler_system.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef struct {
  uint64_t rows;
  uint64_t valid_deltas;
  uint64_t radio_relevant;
} callback_stats_t;

static uint64_t monotonic_raw_ns(void)
{
  struct timespec value = {0};
  assert(clock_gettime(CLOCK_MONOTONIC_RAW, &value) == 0);
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static void count_activity(const oai_profile_activity_observation_t *observation, void *opaque)
{
  callback_stats_t *stats = opaque;
  assert(observation != NULL);
  assert(observation->source != NULL);
  assert(observation->label != NULL);
  assert(observation->status != NULL);
  stats->rows++;
  stats->valid_deltas += observation->delta_valid;
  stats->radio_relevant += observation->radio_relevant;
}

static void test_thread_metrics(void)
{
  const pid_t tid = (pid_t)syscall(SYS_gettid);
  oai_profile_thread_metrics_state_t state = {0};
  oai_profile_thread_metrics_observation_t first = {0};
  oai_profile_read_thread_metrics(tid, monotonic_raw_ns(), &state, &first);
  assert((first.current.valid_mask & OAI_PROFILE_THREAD_METRIC_CORE_MASK) == OAI_PROFILE_THREAD_METRIC_CORE_MASK);
  assert(strcmp(first.status, "warmup") == 0);
  assert(!first.delta_valid);

  volatile uint64_t sink = 0;
  for (uint64_t i = 0; i < UINT64_C(100000); i++)
    sink += i;
  (void)sink;

  oai_profile_thread_metrics_observation_t second = {0};
  oai_profile_read_thread_metrics(tid, monotonic_raw_ns(), &state, &second);
  assert(second.delta_valid);
  assert(second.interval_ns > 0);
  assert(strcmp(second.status, "ok") == 0);

  oai_profile_thread_metrics_observation_t missing = {0};
  oai_profile_thread_metrics_state_t missing_state = {0};
  oai_profile_read_thread_metrics(INT_MAX, monotonic_raw_ns(), &missing_state, &missing);
  assert(!missing.delta_valid);
  assert(strcmp(missing.status, "thread_unavailable") == 0 || strcmp(missing.status, "partial") == 0);
}

static void test_kernel_activity(void)
{
  oai_profile_kernel_activity_state_t state = {0};
  oai_profile_kernel_activity_observation_t first = {0};
  oai_profile_read_kernel_activity(monotonic_raw_ns(), &state, &first);
  assert(first.current.valid_mask == OAI_PROFILE_KERNEL_ACTIVITY_ALL_MASK);
  assert(strcmp(first.status, "warmup") == 0);

  oai_profile_kernel_activity_observation_t second = {0};
  oai_profile_read_kernel_activity(monotonic_raw_ns(), &state, &second);
  assert(second.delta_valid);
  assert(second.interval_ns > 0);
  assert(strcmp(second.status, "ok") == 0);
}

static void test_irq_activity(void)
{
  oai_profile_activity_state_t *state = oai_profile_activity_state_create();
  assert(state != NULL);
  callback_stats_t stats = {0};
  const uint64_t first_time = monotonic_raw_ns();
  oai_profile_activity_result_t softirq = oai_profile_collect_softirqs(state, first_time, count_activity, &stats);
  if (strcmp(softirq.status, "unavailable") != 0) {
    assert(softirq.cpu_count > 0);
    assert(softirq.rows > 0);
  }
  const uint64_t rows_after_first = stats.rows;
  softirq = oai_profile_collect_softirqs(state, monotonic_raw_ns(), count_activity, &stats);
  if (strcmp(softirq.status, "unavailable") != 0) {
    assert(stats.rows > rows_after_first);
    assert(stats.valid_deltas > 0);
  }

  oai_profile_activity_result_t hardirq = oai_profile_collect_interrupts(state, monotonic_raw_ns(), count_activity, &stats);
  if (strcmp(hardirq.status, "unavailable") != 0) {
    assert(hardirq.cpu_count > 0);
    assert(hardirq.rows > 0 || hardirq.parse_errors > 0);
  }
  oai_profile_activity_state_destroy(state);
}

int main(void)
{
  test_thread_metrics();
  test_kernel_activity();
  test_irq_activity();
  return 0;
}
