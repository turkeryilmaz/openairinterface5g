/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE
#include "common/utils/oai_profiler_system.h"

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define FIXTURE_OBSERVATION_CAPACITY 64

typedef struct {
  uint64_t rows;
  uint64_t valid_deltas;
  uint64_t radio_relevant;
} callback_stats_t;

typedef struct {
  char source[16];
  char label[48];
  char description[128];
  int32_t cpu;
  uint64_t raw_count;
  uint64_t delta_count;
  uint64_t interval_ns;
  bool delta_valid;
} fixture_observation_t;

typedef struct {
  fixture_observation_t observations[FIXTURE_OBSERVATION_CAPACITY];
  size_t count;
} fixture_observations_t;

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

static void capture_activity(const oai_profile_activity_observation_t *observation, void *opaque)
{
  fixture_observations_t *captured = opaque;
  assert(observation != NULL);
  assert(captured->count < FIXTURE_OBSERVATION_CAPACITY);
  fixture_observation_t *stored = &captured->observations[captured->count++];
  snprintf(stored->source, sizeof(stored->source), "%s", observation->source);
  snprintf(stored->label, sizeof(stored->label), "%s", observation->label);
  snprintf(stored->description, sizeof(stored->description), "%s", observation->description);
  stored->cpu = observation->cpu;
  stored->raw_count = observation->raw_count;
  stored->delta_count = observation->delta_count;
  stored->interval_ns = observation->interval_ns;
  stored->delta_valid = observation->delta_valid;
}

static oai_profile_activity_result_t collect_fixture(const char *contents,
                                                     oai_profile_activity_state_t *state,
                                                     uint64_t monotonic_ns,
                                                     fixture_observations_t *captured)
{
  char path[] = "/tmp/oai-profiler-activity-XXXXXX";
  const int fd = mkstemp(path);
  assert(fd >= 0);
  FILE *file = fdopen(fd, "w");
  assert(file != NULL);
  const size_t length = strlen(contents);
  assert(fwrite(contents, 1, length, file) == length);
  assert(fclose(file) == 0);
  const oai_profile_activity_result_t result =
      oai_profile_collect_interrupts_path(path, state, monotonic_ns, capture_activity, captured);
  assert(unlink(path) == 0);
  return result;
}

static void assert_fixture_result(oai_profile_activity_result_t result,
                                  uint32_t cpu_count,
                                  uint32_t rows,
                                  uint32_t parse_errors,
                                  int error_code,
                                  const char *status)
{
  assert(result.cpu_count == cpu_count);
  assert(result.rows == rows);
  assert(result.parse_errors == parse_errors);
  assert(result.error_code == error_code);
  assert(strcmp(result.status, status) == 0);
}

static void test_x86_interrupt_fixture(void)
{
  static const char first[] =
      "       CPU0 CPU1 CPU2 CPU3\n"
      "149: 10 20 30 40 IR-PCI-MSI 524288-edge xhci_hcd\n"
      "eRr: 9\n"
      "mIs: 2 \t\n"
      "ERR: 1 2 3 4 Per-CPU error counters\n";
  static const char second[] =
      "       CPU0 CPU1 CPU2 CPU3\n"
      "149: 11 22 33 44 IR-PCI-MSI 524288-edge xhci_hcd\n"
      "ERR: 10\n"
      "MIS: 3\n"
      "ERR: 2 4 6 8 Per-CPU error counters\n";
  oai_profile_activity_state_t *state = oai_profile_activity_state_create();
  assert(state != NULL);
  fixture_observations_t captured = {0};

  oai_profile_activity_result_t result = collect_fixture(first, state, 1000, &captured);
  assert_fixture_result(result, 4, 8, 0, 0, "ok");
  assert(captured.count == 8);
  for (size_t i = 0; i < 4; i++) {
    assert(strcmp(captured.observations[i].label, "149") == 0);
    assert(captured.observations[i].cpu == (int32_t)i);
    assert(captured.observations[i].raw_count == 10 * (i + 1));
    assert(!captured.observations[i].delta_valid);
  }
  for (size_t i = 4; i < 8; i++) {
    assert(strcmp(captured.observations[i].label, "ERR") == 0);
    assert(captured.observations[i].cpu == (int32_t)(i - 4));
    assert(captured.observations[i].raw_count == i - 3);
  }

  result = collect_fixture(second, state, 2000, &captured);
  assert_fixture_result(result, 4, 8, 0, 0, "ok");
  assert(captured.count == 16);
  for (size_t i = 8; i < 16; i++) {
    assert(captured.observations[i].delta_valid);
    assert(captured.observations[i].interval_ns == 1000);
  }
  assert(captured.observations[8].delta_count == 1);
  assert(captured.observations[11].delta_count == 4);
  assert(captured.observations[12].delta_count == 1);
  assert(captured.observations[15].delta_count == 4);
  oai_profile_activity_state_destroy(state);
}

static void test_arm_interrupt_fixture(void)
{
  static const char fixture[] =
      "       CPU0 CPU1 CPU2 CPU3\n"
      " 27: 1 2 3 4 GICv3 30 Level arch_timer\n"
      "IPI0: 5 6 7 8 Rescheduling interrupts\n"
      "err: 0\n"
      "MIS: 0\n";
  oai_profile_activity_state_t *state = oai_profile_activity_state_create();
  assert(state != NULL);
  fixture_observations_t captured = {0};
  const oai_profile_activity_result_t result = collect_fixture(fixture, state, 1000, &captured);
  assert_fixture_result(result, 4, 8, 0, 0, "ok");
  assert(captured.count == 8);
  assert(strcmp(captured.observations[0].label, "27") == 0);
  assert(strcmp(captured.observations[0].description, "GICv3 30 Level arch_timer") == 0);
  assert(strcmp(captured.observations[4].label, "IPI0") == 0);
  assert(strcmp(captured.observations[4].description, "Rescheduling interrupts") == 0);
  oai_profile_activity_state_destroy(state);
}

static void test_interrupt_malformed_fixture(void)
{
  static const char fixture[] =
      "       CPU0 CPU1 CPU2 CPU3\n"
      " 27: 1 2 3 4 GICv3 30 Level arch_timer\n"
      " 28: 1 2 3\n"
      " 29: 1 two 3 4 malformed\n"
      "MIS: 7 unexpected-description\n";
  oai_profile_activity_state_t *state = oai_profile_activity_state_create();
  assert(state != NULL);
  fixture_observations_t captured = {0};
  const oai_profile_activity_result_t result = collect_fixture(fixture, state, 1000, &captured);
  assert_fixture_result(result, 4, 4, 3, EPROTO, "partial");
  assert(captured.count == 4);
  for (size_t i = 0; i < captured.count; i++)
    assert(strcmp(captured.observations[i].label, "27") == 0);
  oai_profile_activity_state_destroy(state);
}

static void test_single_cpu_interrupt_scalar(void)
{
  static const char fixture[] =
      "       CPU0\n"
      "ERR: 7\n";
  oai_profile_activity_state_t *state = oai_profile_activity_state_create();
  assert(state != NULL);
  fixture_observations_t captured = {0};
  const oai_profile_activity_result_t result = collect_fixture(fixture, state, 1000, &captured);
  assert_fixture_result(result, 1, 1, 0, 0, "ok");
  assert(captured.count == 1);
  assert(strcmp(captured.observations[0].label, "ERR") == 0);
  assert(captured.observations[0].raw_count == 7);
  oai_profile_activity_state_destroy(state);
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
  callback_stats_t softirq_stats = {0};
  const uint64_t first_time = monotonic_raw_ns();
  oai_profile_activity_result_t softirq = oai_profile_collect_softirqs(state, first_time, count_activity, &softirq_stats);
  if (strcmp(softirq.status, "unavailable") != 0) {
    assert(softirq.cpu_count > 0);
    assert(softirq.rows > 0);
  }
  const uint64_t softirq_rows_after_first = softirq_stats.rows;
  softirq = oai_profile_collect_softirqs(state, monotonic_raw_ns(), count_activity, &softirq_stats);
  if (strcmp(softirq.status, "unavailable") != 0) {
    assert(softirq_stats.rows > softirq_rows_after_first);
    assert(softirq_stats.valid_deltas > 0);
  }

  callback_stats_t hardirq_stats = {0};
  oai_profile_activity_result_t hardirq = oai_profile_collect_interrupts(state, monotonic_raw_ns(), count_activity, &hardirq_stats);
  if (strcmp(hardirq.status, "unavailable") != 0) {
    assert(hardirq.cpu_count > 0);
    assert(hardirq.rows > 0 || hardirq.parse_errors > 0);
    if (hardirq.parse_errors > 0)
      assert(hardirq.error_code == EPROTO);
    const uint64_t hardirq_rows_after_first = hardirq_stats.rows;
    hardirq = oai_profile_collect_interrupts(state, monotonic_raw_ns(), count_activity, &hardirq_stats);
    assert(hardirq_stats.rows > hardirq_rows_after_first);
    assert(hardirq_stats.valid_deltas > 0);
  }
  oai_profile_activity_state_destroy(state);
}

int main(void)
{
  test_x86_interrupt_fixture();
  test_arm_interrupt_fixture();
  test_interrupt_malformed_fixture();
  test_single_cpu_interrupt_scalar();
  test_thread_metrics();
  test_kernel_activity();
  test_irq_activity();
  return 0;
}
