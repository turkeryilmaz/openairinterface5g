/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE
#include "common/utils/oai_profiler.h"

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
  oai_profile_work_t work;
} worker_argument_t;

#define LIFECYCLE_PRODUCER_THREADS 4
#define LIFECYCLE_MIN_ITERATIONS UINT64_C(1000)
#define LIFECYCLE_MIN_SETTING_CALLS UINT64_C(4)

typedef struct {
  uint64_t iterations;
} lifecycle_producer_argument_t;

typedef struct {
  pthread_barrier_t *barrier;
  uint64_t *completed;
} lifecycle_shutdown_argument_t;

static const char *const csv_files[] = {
    "events.csv",
    "event_catalog.csv",
    "sync.csv",
    "drops.csv",
    "settings.csv",
    "host_metrics.csv",
    "pmu_catalog.csv",
    "pmu_availability.csv",
    "pmu_samples.csv",
    "pmu_read_overhead.csv",
    "profiler_primitive_overhead.csv",
    "clock_catalog.csv",
    "system_catalog.csv",
    "thread_metrics.csv",
    "kernel_activity.csv",
    "interrupts.csv",
    "softirqs.csv",
    "system_read_overhead.csv",
    "external_sources.csv",
};

static void consume_cpu(void)
{
  volatile uint64_t value = 0;
  for (uint64_t i = 0; i < UINT64_C(100000); i++)
    value += i;
  (void)value;
}

static void *worker_body(void *opaque)
{
  worker_argument_t *argument = opaque;
  assert(pthread_setname_np(pthread_self(), "profile_test") == 0);
  oai_profiler_register_thread();
  const oai_profile_context_t previous = oai_profiler_enter_work(argument->work);
  OAI_PROFILE_START(worker_span);
  consume_cpu();
  OAI_PROFILE_STOP(OAI_PROFILE_EVENT_UE_DL_PROCESSING, worker_span, 10, 4, 10, 5, 1, 0, 0);
  OAI_PROFILE_MARK(OAI_PROFILE_EVENT_UE_TX_DEADLINE_MISS, 10, 4, 100, 120, 20, 0, 0);
  oai_profiler_leave_work(previous);
  return NULL;
}
static void wait_at_barrier(pthread_barrier_t *barrier)
{
  const int result = pthread_barrier_wait(barrier);
  assert(result == 0 || result == PTHREAD_BARRIER_SERIAL_THREAD);
}

static void *lifecycle_producer_body(void *opaque)
{
  lifecycle_producer_argument_t *argument = opaque;
  assert(pthread_setname_np(pthread_self(), "lifecycle_prod") == 0);
  oai_profiler_register_thread();
  while (oai_profiler_is_enabled()) {
    const uint64_t iteration = __atomic_add_fetch(&argument->iterations, 1, __ATOMIC_RELAXED);
    OAI_PROFILE_START(span);
    OAI_PROFILE_STOP(OAI_PROFILE_EVENT_UE_DL_PROCESSING, span, 11, 5, (int64_t)iteration, 0, 0, 0, 0);
    OAI_PROFILE_MARK(OAI_PROFILE_EVENT_UE_TX_DEADLINE_MISS, 11, 5, (int64_t)iteration, 0, 0, 0, 0);
  }
  return NULL;
}

static void *lifecycle_setting_body(void *opaque)
{
  uint64_t *calls = opaque;
  assert(pthread_setname_np(pthread_self(), "lifecycle_set") == 0);
  while (oai_profiler_is_enabled()) {
    const uint64_t call = __atomic_load_n(calls, __ATOMIC_RELAXED) + 1;
    oai_profiler_record_setting_int("test.concurrent_setting", (int64_t)call, "lifecycle-test");
    __atomic_fetch_add(calls, 1, __ATOMIC_RELAXED);
  }
  return NULL;
}

static void *lifecycle_shutdown_body(void *opaque)
{
  lifecycle_shutdown_argument_t *argument = opaque;
  wait_at_barrier(argument->barrier);
  oai_profiler_shutdown();
  __atomic_fetch_add(argument->completed, 1, __ATOMIC_RELAXED);
  return NULL;
}

static void wait_for_lifecycle_activity(lifecycle_producer_argument_t *arguments, uint64_t *setting_calls)
{
  for (int attempt = 0; attempt < 5000; attempt++) {
    uint64_t producer_iterations = 0;
    for (size_t i = 0; i < LIFECYCLE_PRODUCER_THREADS; i++)
      producer_iterations += __atomic_load_n(&arguments[i].iterations, __ATOMIC_RELAXED);
    if (producer_iterations >= LIFECYCLE_MIN_ITERATIONS
        && __atomic_load_n(setting_calls, __ATOMIC_RELAXED) >= LIFECYCLE_MIN_SETTING_CALLS)
      return;
    usleep(1000);
  }
  assert(false);
}

static size_t csv_column_count(const char *line)
{
  size_t columns = 1;
  bool quoted = false;
  for (const char *cursor = line; *cursor != '\0' && *cursor != '\n' && *cursor != '\r'; cursor++) {
    if (*cursor == '"') {
      if (quoted && cursor[1] == '"')
        cursor++;
      else
        quoted = !quoted;
    } else if (*cursor == ',' && !quoted) {
      columns++;
    }
  }
  assert(!quoted);
  return columns;
}

static size_t validate_csv(const char *directory, const char *name, size_t minimum_data_rows)
{
  char path[4096];
  assert(snprintf(path, sizeof(path), "%s/%s", directory, name) < (int)sizeof(path));
  FILE *file = fopen(path, "r");
  if (file == NULL)
    fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
  assert(file != NULL);
  char *line = NULL;
  size_t line_capacity = 0;
  assert(getline(&line, &line_capacity, file) > 0);
  const size_t columns = csv_column_count(line);
  assert(columns > 1);
  size_t rows = 0;
  while (getline(&line, &line_capacity, file) > 0) {
    assert(csv_column_count(line) == columns);
    rows++;
  }
  assert(!ferror(file));
  free(line);
  fclose(file);
  assert(rows >= minimum_data_rows);
  return rows;
}

static bool file_contains(const char *path, const char *text)
{
  FILE *file = fopen(path, "r");
  assert(file != NULL);
  char *line = NULL;
  size_t capacity = 0;
  bool found = false;
  while (getline(&line, &capacity, file) > 0) {
    if (strstr(line, text) != NULL) {
      found = true;
      break;
    }
  }
  free(line);
  fclose(file);
  return found;
}

static void remove_test_output(const char *output)
{
  char path[4096];
  for (size_t i = 0; i < sizeof(csv_files) / sizeof(csv_files[0]); i++) {
    assert(snprintf(path, sizeof(path), "%s/%s", output, csv_files[i]) < (int)sizeof(path));
    assert(unlink(path) == 0);
  }
  assert(snprintf(path, sizeof(path), "%s/metadata.txt", output) < (int)sizeof(path));
  assert(unlink(path) == 0);
  assert(rmdir(output) == 0);
}

static void remove_test_root(const char *root)
{
  char path[4096];
  assert(snprintf(path, sizeof(path), "%s/configs/gNB", root) < (int)sizeof(path));
  assert(rmdir(path) == 0);
  assert(snprintf(path, sizeof(path), "%s/configs/nrUE", root) < (int)sizeof(path));
  assert(rmdir(path) == 0);
  assert(snprintf(path, sizeof(path), "%s/configs", root) < (int)sizeof(path));
  assert(rmdir(path) == 0);
  assert(rmdir(root) == 0);
}

int main(void)
{
  uint64_t previous_tick = oai_profiler_read_tick();
  for (size_t i = 0; i < 10000; i++) {
    const uint64_t current_tick = oai_profiler_read_tick();
    assert(current_tick >= previous_tick);
    previous_tick = current_tick;
  }
  assert(pthread_setname_np(pthread_self(), "profile_main") == 0);
  char root[] = "/tmp/oai-profiler-archive-XXXXXX";
  assert(mkdtemp(root) != NULL);
  char output[4096];
  assert(snprintf(output, sizeof(output), "%s/run", root) < (int)sizeof(output));
  assert(setenv("OAI_PROFILE_ROOT", root, 1) == 0);
  assert(setenv("OAI_PROFILE_EXPERIMENT_ID", "archive-test", 1) == 0);
  assert(setenv("OAI_PROFILE_CAMPAIGN_ID", "unit-test", 1) == 0);
  assert(setenv("OAI_PROFILE_VARIANT", "in-process", 1) == 0);
  assert(setenv("OAI_PROFILE_TRIAL", "1", 1) == 0);
  assert(setenv("OAI_PROFILE_HOST_METRICS_US", "100000", 1) == 0);
  assert(setenv("OAI_PROFILE_CALIBRATION_SAMPLES", "16", 1) == 0);
  assert(setenv("OAI_PROFILE_CALIBRATION_WARMUP", "2", 1) == 0);

  char program[] = "test-nr-uesoftmodem";
  char imsi_option[] = "--uicc0.imsi";
  char imsi_value[] = "001010000000001";
  char *argv[] = {program, imsi_option, imsi_value, NULL};
  oai_profiler_init(program, 3, argv, true, output, 1024, 1000, "off", 100000);
  assert(oai_profiler_is_enabled());
  oai_profiler_register_thread();
  oai_profiler_record_duration(OAI_PROFILE_EVENT_UE_DL_PROCESSING, UINT64_MAX, -1, -1, 0, 0, 0, 0, 0);
  const uint64_t correlation_id = oai_profiler_next_correlation_id();
  oai_profiler_set_context((oai_profile_context_t){
      .absolute_slot = 204,
      .correlation_id = correlation_id,
      .parent_id = 0,
  });
  OAI_PROFILE_START(root_span);
  worker_argument_t argument = {
      .work = oai_profiler_capture_work(204),
  };
  pthread_t worker;
  assert(pthread_create(&worker, NULL, worker_body, &argument) == 0);
  consume_cpu();
  assert(pthread_join(worker, NULL) == 0);
  OAI_PROFILE_STOP(OAI_PROFILE_EVENT_UE_SLOT_LOOP, root_span, 10, 4, 6, 512, 512, 0, 0);
  OAI_PROFILE_MARK(OAI_PROFILE_EVENT_UE_TX_DEADLINE_COMPUTE,
                   10,
                   5,
                   100000,
                   107680,
                   7680,
                   1999000000,
                   OAI_PROFILE_UE_TX_DEADLINE_FLAG_VALID);
  OAI_PROFILE_MARK(OAI_PROFILE_EVENT_UE_TX_DEADLINE_CHECK,
                   10,
                   5,
                   2000010000,
                   2000000000,
                   10000,
                   0,
                   OAI_PROFILE_UE_TX_DEADLINE_FLAG_VALID | OAI_PROFILE_UE_TX_DEADLINE_FLAG_MISSED);
  oai_profiler_record_setting("test.setting", "enabled", "unit-test");
  lifecycle_producer_argument_t producer_arguments[LIFECYCLE_PRODUCER_THREADS] = {0};
  pthread_t producer_threads[LIFECYCLE_PRODUCER_THREADS];
  for (size_t i = 0; i < LIFECYCLE_PRODUCER_THREADS; i++)
    assert(pthread_create(&producer_threads[i], NULL, lifecycle_producer_body, &producer_arguments[i]) == 0);

  uint64_t setting_calls = 0;
  pthread_t setting_thread;
  assert(pthread_create(&setting_thread, NULL, lifecycle_setting_body, &setting_calls) == 0);
  wait_for_lifecycle_activity(producer_arguments, &setting_calls);

  pthread_barrier_t shutdown_barrier;
  assert(pthread_barrier_init(&shutdown_barrier, NULL, 3) == 0);
  uint64_t shutdown_completed = 0;
  lifecycle_shutdown_argument_t shutdown_argument = {
      .barrier = &shutdown_barrier,
      .completed = &shutdown_completed,
  };
  pthread_t shutdown_threads[2];
  for (size_t i = 0; i < 2; i++)
    assert(pthread_create(&shutdown_threads[i], NULL, lifecycle_shutdown_body, &shutdown_argument) == 0);
  wait_at_barrier(&shutdown_barrier);

  for (size_t i = 0; i < 2; i++)
    assert(pthread_join(shutdown_threads[i], NULL) == 0);
  for (size_t i = 0; i < LIFECYCLE_PRODUCER_THREADS; i++)
    assert(pthread_join(producer_threads[i], NULL) == 0);
  assert(pthread_join(setting_thread, NULL) == 0);
  assert(pthread_barrier_destroy(&shutdown_barrier) == 0);
  assert(__atomic_load_n(&shutdown_completed, __ATOMIC_RELAXED) == 2);
  assert(!oai_profiler_is_enabled());

  for (size_t i = 0; i < sizeof(csv_files) / sizeof(csv_files[0]); i++) {
    size_t minimum_rows = 0;
    if (strcmp(csv_files[i], "events.csv") == 0)
      minimum_rows = 3;
    else if (strcmp(csv_files[i], "host_metrics.csv") == 0)
      minimum_rows = 1;
    else if (strcmp(csv_files[i], "thread_metrics.csv") == 0)
      minimum_rows = 2;
    else if (strcmp(csv_files[i], "kernel_activity.csv") == 0)
      minimum_rows = 16;
    else if (strcmp(csv_files[i], "system_read_overhead.csv") == 0)
      minimum_rows = 4;
    else if (strcmp(csv_files[i], "profiler_primitive_overhead.csv") == 0)
      minimum_rows = 109;
    const size_t rows = validate_csv(output, csv_files[i], minimum_rows);
    if (strcmp(csv_files[i], "event_catalog.csv") == 0)
      assert(rows == (size_t)OAI_PROFILE_EVENT_MAX - 1);
    else if (strcmp(csv_files[i], "profiler_primitive_overhead.csv") == 0)
      assert(rows == 109);
  }
  char event_catalog_path[4096];
  assert(snprintf(event_catalog_path, sizeof(event_catalog_path), "%s/event_catalog.csv", output)
         < (int)sizeof(event_catalog_path));
  assert(file_contains(event_catalog_path, "USRP_RX_RECV"));
  assert(file_contains(event_catalog_path, "USRP_TX_ASYNC_EVENT"));
  assert(file_contains(event_catalog_path, "UE_PDSCH_WORKSPACE_ALLOCATION"));
  assert(file_contains(event_catalog_path, "UE_PDCCH_SCOPE_COPY"));
  assert(file_contains(event_catalog_path, "PROFILER_PRIMITIVE_CALIBRATION"));
  assert(file_contains(event_catalog_path, "UE_TX_DEADLINE_COMPUTE"));
  assert(file_contains(event_catalog_path, "UE_TX_DEADLINE_CHECK"));
  char settings_path[4096];
  assert(snprintf(settings_path, sizeof(settings_path), "%s/settings.csv", output) < (int)sizeof(settings_path));
  assert(file_contains(settings_path, "test.concurrent_setting"));
  char drops_path[4096];
  assert(snprintf(drops_path, sizeof(drops_path), "%s/drops.csv", output) < (int)sizeof(drops_path));
  assert(file_contains(drops_path, ",profile_main,0,0,0,1\n"));
  char host_metrics_path[4096];
  assert(snprintf(host_metrics_path, sizeof(host_metrics_path), "%s/host_metrics.csv", output) < (int)sizeof(host_metrics_path));
  assert(file_contains(host_metrics_path,
                       "end_monotonic_raw_ns,end_tick,writer_cpu_end,writer_cpu_migrated,"
                       "acquisition_duration_monotonic_raw_ns,acquisition_duration_tick,acquisition_duration_us,"
                       "status,getloadavg_count,getrusage_status,error_mask\n"));
  assert(file_contains(host_metrics_path, ",ok,3,ok,0\n"));
  char system_catalog_path[4096];
  assert(snprintf(system_catalog_path, sizeof(system_catalog_path), "%s/system_catalog.csv", output)
         < (int)sizeof(system_catalog_path));
  assert(file_contains(system_catalog_path, "host_metrics,acquisition_duration,us,"));
  assert(file_contains(system_catalog_path, "host_metrics,error_mask,bitmask,"));
  char metadata_path[4096];
  assert(snprintf(metadata_path, sizeof(metadata_path), "%s/metadata.txt", output) < (int)sizeof(metadata_path));
  assert(file_contains(metadata_path, "schema_version=2"));
  assert(file_contains(metadata_path, "experiment_id=archive-test"));
  assert(file_contains(metadata_path, "campaign_id=unit-test"));
  assert(file_contains(metadata_path, "calibration_samples=16"));
  assert(file_contains(metadata_path, "calibration_warmup=2"));
  assert(file_contains(metadata_path, "cmdline=test-nr-uesoftmodem --uicc0.imsi <redacted>"));
  assert(!file_contains(metadata_path, imsi_value));
  assert(file_contains(metadata_path, "duration_monotonic_raw_ns="));
  assert(file_contains(metadata_path, "duration_clock=CLOCK_MONOTONIC_RAW"));
  assert(file_contains(metadata_path, "realtime_clock_regressed=0"));
  assert(file_contains(metadata_path, "monotonic_raw_clock_regressed=0"));
  assert(file_contains(metadata_path, "clean_shutdown=1"));

  char second_output[4096];
  assert(snprintf(second_output, sizeof(second_output), "%s/run-second", root) < (int)sizeof(second_output));
  assert(setenv("OAI_PROFILE_EXPERIMENT_ID", "archive-test-second", 1) == 0);
  oai_profiler_init(program, 3, argv, true, second_output, 1024, 1000, "off", 100000);
  assert(oai_profiler_is_enabled());
  oai_profiler_register_thread();
  oai_profiler_set_context((oai_profile_context_t){
      .absolute_slot = 205,
      .correlation_id = oai_profiler_next_correlation_id(),
      .parent_id = 0,
  });
  OAI_PROFILE_MARK(OAI_PROFILE_EVENT_UE_TX_DEADLINE_MISS, 10, 5, 1, 0, 0, 0, 0);
  oai_profiler_shutdown();
  assert(!oai_profiler_is_enabled());

  for (size_t i = 0; i < sizeof(csv_files) / sizeof(csv_files[0]); i++) {
    const size_t minimum_rows = strcmp(csv_files[i], "events.csv") == 0 ? 1 : 0;
    const size_t rows = validate_csv(second_output, csv_files[i], minimum_rows);
    if (strcmp(csv_files[i], "event_catalog.csv") == 0)
      assert(rows == (size_t)OAI_PROFILE_EVENT_MAX - 1);
    else if (strcmp(csv_files[i], "profiler_primitive_overhead.csv") == 0)
      assert(rows == 109);
  }
  assert(snprintf(metadata_path, sizeof(metadata_path), "%s/metadata.txt", second_output) < (int)sizeof(metadata_path));
  assert(file_contains(metadata_path, "experiment_id=archive-test-second"));
  assert(file_contains(metadata_path, "clean_shutdown=1"));

  remove_test_output(output);
  remove_test_output(second_output);
  remove_test_root(root);
  return 0;
}
