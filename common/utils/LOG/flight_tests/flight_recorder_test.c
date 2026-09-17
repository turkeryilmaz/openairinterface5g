/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*
 * Standalone native validation for common/utils/LOG/flight_recorder.c.
 * Build this runner with -DFLIGHT_RECORDER_TESTING so its test-only writer
 * hooks are available. All generated directories are supplied by argv[1].
 */

#define _GNU_SOURCE

#include "flight_recorder.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define TEST_PATH_BYTES 512
#define TEST_LINE_BYTES 1024

extern void flight_recorder_test_set_write_limit(int byte_limit);
extern int flight_recorder_test_writer_policy(void);
extern void flight_recorder_test_set_writer_paused(bool paused);
extern void flight_recorder_test_set_emit_after_check_paused(bool paused);
extern bool flight_recorder_test_emit_after_check_reached(void);
extern void flight_recorder_test_set_writer_stop_empty_paused(bool paused);
extern bool flight_recorder_test_writer_stop_empty_reached(void);
extern void flight_recorder_test_set_shutdown_before_cas_paused(bool paused);
extern bool flight_recorder_test_shutdown_before_cas_reached(void);

#define CHECK(condition, ...)                         \
  do {                                                \
    if (!(condition)) {                               \
      fprintf(stderr, "%s:%d: ", __func__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                   \
      fputc('\n', stderr);                            \
      return false;                                   \
    }                                                 \
  } while (0)

typedef struct {
  int id;
  int count;
} ordered_worker_t;

static atomic_bool g_active_producer_stop = ATOMIC_VAR_INIT(false);

static void sleep_milliseconds(long milliseconds)
{
  const struct timespec pause = {
      .tv_sec = milliseconds / 1000,
      .tv_nsec = (milliseconds % 1000) * 1000000L,
  };
  nanosleep(&pause, NULL);
}

static int recorder_file_paths(const char *directory, char paths[FLIGHT_RECORDER_MAX_FILES][TEST_PATH_BYTES])
{
  DIR *stream = opendir(directory);
  if (stream == NULL)
    return -1;

  int count = 0;
  struct dirent *entry;
  while ((entry = readdir(stream)) != NULL) {
    if (strncmp(entry->d_name, "oai-flight-recorder-", strlen("oai-flight-recorder-")) != 0)
      continue;
    if (strstr(entry->d_name, ".ndjson") == NULL)
      continue;
    if (count >= (int)FLIGHT_RECORDER_MAX_FILES) {
      closedir(stream);
      return -1;
    }
    const int result = snprintf(paths[count], TEST_PATH_BYTES, "%s/%s", directory, entry->d_name);
    if (result < 0 || result >= TEST_PATH_BYTES) {
      closedir(stream);
      return -1;
    }
    count++;
  }

  closedir(stream);
  return count;
}

static int directory_entry_count(const char *directory)
{
  DIR *stream = opendir(directory);
  if (stream == NULL)
    return -1;

  int count = 0;
  struct dirent *entry;
  while ((entry = readdir(stream)) != NULL) {
    if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
      count++;
  }
  closedir(stream);
  return count;
}

static bool json_unsigned(const char *line, const char *field, uint64_t *value)
{
  char marker[64];
  const int marker_length = snprintf(marker, sizeof(marker), "\"%s\":", field);
  if (marker_length < 0 || marker_length >= (int)sizeof(marker))
    return false;

  const char *start = strstr(line, marker);
  if (start == NULL)
    return false;
  start += marker_length;
  if (*start == '-')
    return false;

  char *end = NULL;
  errno = 0;
  const unsigned long long parsed = strtoull(start, &end, 10);
  if (errno != 0 || end == start)
    return false;
  *value = (uint64_t)parsed;
  return true;
}

static bool read_footer_field(const char *directory, const char *field, uint64_t *value)
{
  char paths[FLIGHT_RECORDER_MAX_FILES][TEST_PATH_BYTES];
  const int file_count = recorder_file_paths(directory, paths);
  if (file_count < 0)
    return false;

  for (int index = 0; index < file_count; ++index) {
    FILE *file = fopen(paths[index], "r");
    if (file == NULL)
      return false;
    char line[TEST_LINE_BYTES];
    while (fgets(line, sizeof(line), file) != NULL) {
      if (strstr(line, "\"kind\":\"capture_footer\"") != NULL && json_unsigned(line, field, value)) {
        fclose(file);
        return true;
      }
    }
    fclose(file);
  }
  return false;
}

static bool has_line(const char *directory, const char *needle)
{
  char paths[FLIGHT_RECORDER_MAX_FILES][TEST_PATH_BYTES];
  const int file_count = recorder_file_paths(directory, paths);
  if (file_count < 0)
    return false;

  for (int index = 0; index < file_count; ++index) {
    FILE *file = fopen(paths[index], "r");
    if (file == NULL)
      return false;
    char line[TEST_LINE_BYTES];
    while (fgets(line, sizeof(line), file) != NULL) {
      if (strstr(line, needle) != NULL) {
        fclose(file);
        return true;
      }
    }
    fclose(file);
  }
  return false;
}

static void *ordered_worker(void *argument)
{
  const ordered_worker_t *worker = argument;
  for (int index = 0; index < worker->count; ++index)
    flight_recorder_emit(FLIGHT_EVENT_UE_SYNC, worker->id, index, 3, 4, 5, 6);
  return NULL;
}

static void *single_emit_worker(void *unused)
{
  (void)unused;
  flight_recorder_emit(FLIGHT_EVENT_GNB_SLOT, 1, 2, 3, 4, 5, 6);
  return NULL;
}

static void *active_producer(void *unused)
{
  (void)unused;
  while (!atomic_load_explicit(&g_active_producer_stop, memory_order_acquire)) {
    flight_recorder_emit(FLIGHT_EVENT_RADIO_RX, 1, 2, 3, 4, 5, 6);
    sleep_milliseconds(1);
  }
  return NULL;
}

static void *paused_after_check_producer(void *unused)
{
  (void)unused;
  flight_recorder_emit(FLIGHT_EVENT_LIFECYCLE, 777777, 2, 3, 4, 5, 6);
  return NULL;
}

static void *shutdown_worker(void *unused)
{
  (void)unused;
  flight_recorder_shutdown();
  return NULL;
}

static bool wait_for_test_hook(bool (*reached)(void))
{
  for (int attempt = 0; attempt < 1000; ++attempt) {
    if (reached())
      return true;
    sleep_milliseconds(1);
  }
  return false;
}

static bool test_disabled(const char *directory)
{
  unsetenv("OAI_FLIGHT_RECORDER_DIR");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_init();
  CHECK(!flight_recorder_enabled(), "disabled recorder reported enabled");
  flight_recorder_emit(FLIGHT_EVENT_UE_SYNC, 1, 2, 3, 4, 5, 6);
  flight_recorder_shutdown();
  flight_recorder_shutdown();
  CHECK(directory_entry_count(directory) == 0, "disabled recorder created output");
  return true;
}

static bool test_invalid_path(const char *directory)
{
  char invalid_path[TEST_PATH_BYTES];
  CHECK(snprintf(invalid_path, sizeof(invalid_path), "%s/not-a-directory", directory) > 0, "invalid path overflow");
  const int descriptor = open(invalid_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  CHECK(descriptor >= 0, "could not create invalid-path fixture: %d", errno);
  close(descriptor);

  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", invalid_path, 1) == 0, "setenv failed");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_init();
  CHECK(!flight_recorder_enabled(), "regular-file output path enabled recorder");
  flight_recorder_emit(FLIGHT_EVENT_UE_SYNC, 1, 2, 3, 4, 5, 6);
  flight_recorder_shutdown();
  char paths[FLIGHT_RECORDER_MAX_FILES][TEST_PATH_BYTES];
  CHECK(recorder_file_paths(directory, paths) == 0, "invalid path created recorder file");
  return true;
}

static bool test_ordering_timestamps_and_short_writes(const char *directory)
{
  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", directory, 1) == 0, "setenv output failed");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_test_set_write_limit(7);
  flight_recorder_init();
  CHECK(flight_recorder_enabled(), "recorder did not enable");
  sleep_milliseconds(5);
  CHECK(flight_recorder_test_writer_policy() == SCHED_OTHER, "writer was not SCHED_OTHER");

  enum { worker_count = 4, records_per_worker = 200 };
  pthread_t workers[worker_count];
  ordered_worker_t arguments[worker_count];
  for (int index = 0; index < worker_count; ++index) {
    arguments[index] = (ordered_worker_t){.id = index, .count = records_per_worker};
    CHECK(pthread_create(&workers[index], NULL, ordered_worker, &arguments[index]) == 0, "pthread_create failed");
  }
  for (int index = 0; index < worker_count; ++index)
    CHECK(pthread_join(workers[index], NULL) == 0, "pthread_join failed");

  flight_recorder_shutdown();
  flight_recorder_shutdown();

  char paths[FLIGHT_RECORDER_MAX_FILES][TEST_PATH_BYTES];
  const int file_count = recorder_file_paths(directory, paths);
  CHECK(file_count == 1, "expected one output file, got %d", file_count);
  struct stat status;
  CHECK(stat(paths[0], &status) == 0, "stat failed: %d", errno);
  CHECK((status.st_mode & 0777) == 0600, "output mode was %o", status.st_mode & 0777);

  FILE *file = fopen(paths[0], "r");
  CHECK(file != NULL, "could not read output");
  int events = 0;
  int last_b[FLIGHT_RECORDER_MAX_THREAD_RINGS];
  bool ring_seen[FLIGHT_RECORDER_MAX_THREAD_RINGS];
  memset(last_b, 0, sizeof(last_b));
  memset(ring_seen, 0, sizeof(ring_seen));
  bool clock_seen = false;
  bool footer_seen = false;
  char line[TEST_LINE_BYTES];
  while (fgets(line, sizeof(line), file) != NULL) {
    if (strstr(line, "\"kind\":\"clock_correlation\"") != NULL) {
      uint64_t available;
      CHECK(json_unsigned(line, "available", &available) && available == 1, "clock correlation unavailable");
      CHECK(strstr(line, "\"bracket_uncertainty_ns\":") != NULL, "clock bracket missing");
      clock_seen = true;
    }
    if (strstr(line, "\"kind\":\"capture_footer\"") != NULL)
      footer_seen = true;
    if (strstr(line, "\"kind\":\"event\"") == NULL)
      continue;

    uint64_t ring;
    uint64_t sequence;
    uint64_t mono;
    uint64_t realtime;
    uint64_t event;
    uint64_t b;
    CHECK(json_unsigned(line, "ring", &ring) && ring < FLIGHT_RECORDER_MAX_THREAD_RINGS, "invalid ring");
    CHECK(json_unsigned(line, "sequence", &sequence) && sequence > 0, "missing sequence");
    CHECK(json_unsigned(line, "mono_ns", &mono) && mono > 0, "missing monotonic timestamp");
    CHECK(json_unsigned(line, "realtime_ns", &realtime) && realtime > 0, "missing realtime timestamp");
    CHECK(json_unsigned(line, "event", &event) && event == FLIGHT_EVENT_UE_SYNC, "wrong event");
    CHECK(json_unsigned(line, "b", &b) && b < records_per_worker, "invalid worker sequence");
    if (ring_seen[ring])
      CHECK(b == (uint64_t)last_b[ring] + 1, "ring %" PRIu64 " record ordering broken", ring);
    else
      ring_seen[ring] = true;
    last_b[ring] = (int)b;
    events++;
  }
  fclose(file);

  CHECK(clock_seen, "missing clock correlation");
  CHECK(footer_seen, "missing clean footer");
  CHECK(events == worker_count * records_per_worker, "expected %d events, got %d", worker_count * records_per_worker, events);
  return true;
}

static bool test_saturation(const char *directory)
{
  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", directory, 1) == 0, "setenv output failed");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_test_set_writer_paused(true);
  flight_recorder_init();
  CHECK(flight_recorder_enabled(), "recorder did not enable");
  for (unsigned int index = 0; index < FLIGHT_RECORDER_RING_RECORDS + 512U; ++index)
    flight_recorder_emit(FLIGHT_EVENT_UE_AGC, index, 0, 0, 0, 0, 0);
  flight_recorder_test_set_writer_paused(false);
  flight_recorder_shutdown();

  uint64_t dropped = 0;
  CHECK(read_footer_field(directory, "dropped_ring_full", &dropped), "missing saturation footer");
  CHECK(dropped >= 512U, "expected saturation drops, got %" PRIu64, dropped);
  return true;
}

static bool test_no_slot(const char *directory)
{
  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", directory, 1) == 0, "setenv output failed");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_init();
  CHECK(flight_recorder_enabled(), "recorder did not enable");

  pthread_t workers[FLIGHT_RECORDER_MAX_THREAD_RINGS + 1U];
  for (unsigned int index = 0; index < FLIGHT_RECORDER_MAX_THREAD_RINGS + 1U; ++index)
    CHECK(pthread_create(&workers[index], NULL, single_emit_worker, NULL) == 0, "pthread_create failed");
  for (unsigned int index = 0; index < FLIGHT_RECORDER_MAX_THREAD_RINGS + 1U; ++index)
    CHECK(pthread_join(workers[index], NULL) == 0, "pthread_join failed");

  flight_recorder_shutdown();
  uint64_t no_slot_threads = 0;
  uint64_t dropped_no_slot = 0;
  CHECK(read_footer_field(directory, "no_slot_threads", &no_slot_threads), "missing no-slot thread metadata");
  CHECK(read_footer_field(directory, "dropped_no_slot", &dropped_no_slot), "missing no-slot drop metadata");
  CHECK(no_slot_threads >= 1, "no-slot thread metadata was %" PRIu64, no_slot_threads);
  CHECK(dropped_no_slot >= 1, "no-slot drop metadata was %" PRIu64, dropped_no_slot);
  return true;
}

static bool test_rotation_bound(const char *directory)
{
  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", directory, 1) == 0, "setenv output failed");
  CHECK(setenv("OAI_FLIGHT_RECORDER_MAX_BYTES", "8192", 1) == 0, "setenv limit failed");
  flight_recorder_init();
  CHECK(flight_recorder_enabled(), "recorder did not enable");
  for (int index = 0; index < 128; ++index)
    flight_recorder_emit(FLIGHT_EVENT_GNB_UE_BYTES, index, 1, 2, 3, 4, 5);
  flight_recorder_shutdown();

  char paths[FLIGHT_RECORDER_MAX_FILES][TEST_PATH_BYTES];
  const int file_count = recorder_file_paths(directory, paths);
  CHECK(file_count > 0 && file_count <= (int)FLIGHT_RECORDER_MAX_FILES, "rotation created %d files", file_count);
  off_t total_size = 0;
  for (int index = 0; index < file_count; ++index) {
    struct stat status;
    CHECK(stat(paths[index], &status) == 0, "rotation stat failed");
    CHECK(status.st_size <= 1024, "file exceeds per-file cap: %jd", (intmax_t)status.st_size);
    total_size += status.st_size;
  }
  CHECK(total_size <= 8192, "rotation exceeds total cap: %jd", (intmax_t)total_size);
  CHECK(has_line(directory, "\"overwrites_available\":1"), "rotation did not retain overwrite metadata");
  CHECK(has_line(directory, "\"kind\":\"capture_footer\""), "rotation footer missing");
  return true;
}

static bool test_shutdown_with_active_producer(const char *directory)
{
  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", directory, 1) == 0, "setenv output failed");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  atomic_store_explicit(&g_active_producer_stop, false, memory_order_release);
  flight_recorder_init();
  CHECK(flight_recorder_enabled(), "recorder did not enable");

  pthread_t worker;
  CHECK(pthread_create(&worker, NULL, active_producer, NULL) == 0, "pthread_create failed");
  sleep_milliseconds(5);
  flight_recorder_shutdown();
  CHECK(!flight_recorder_enabled(), "recorder remained enabled after shutdown");
  atomic_store_explicit(&g_active_producer_stop, true, memory_order_release);
  CHECK(pthread_join(worker, NULL) == 0, "pthread_join failed");
  flight_recorder_shutdown();
  CHECK(has_line(directory, "\"kind\":\"capture_footer\""), "active-shutdown footer missing");
  return true;
}

static bool test_final_drain_after_producer_quiescence(const char *directory)
{
  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", directory, 1) == 0, "setenv output failed");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_test_set_emit_after_check_paused(true);
  flight_recorder_test_set_writer_stop_empty_paused(true);
  flight_recorder_init();
  CHECK(flight_recorder_enabled(), "recorder did not enable");

  pthread_t producer;
  CHECK(pthread_create(&producer, NULL, paused_after_check_producer, NULL) == 0, "producer creation failed");
  CHECK(wait_for_test_hook(flight_recorder_test_emit_after_check_reached), "producer did not pause after state check");

  pthread_t stopper;
  CHECK(pthread_create(&stopper, NULL, shutdown_worker, NULL) == 0, "shutdown creation failed");
  CHECK(wait_for_test_hook(flight_recorder_test_writer_stop_empty_reached), "writer did not pause after empty shutdown drain");

  flight_recorder_test_set_emit_after_check_paused(false);
  CHECK(pthread_join(producer, NULL) == 0, "producer join failed");
  flight_recorder_test_set_writer_stop_empty_paused(false);
  CHECK(pthread_join(stopper, NULL) == 0, "shutdown join failed");
  CHECK(!flight_recorder_enabled(), "recorder remained enabled after shutdown");
  flight_recorder_shutdown();

  CHECK(has_line(directory, "\"event\":1,\"a\":777777"), "final post-quiescence record was lost");
  CHECK(has_line(directory, "\"kind\":\"capture_footer\""), "final-drain footer missing");
  return true;
}

static bool test_writer_error_races_shutdown(const char *directory)
{
  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", directory, 1) == 0, "setenv output failed");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_init();
  CHECK(flight_recorder_enabled(), "recorder did not enable");

  flight_recorder_test_set_shutdown_before_cas_paused(true);
  pthread_t stopper;
  CHECK(pthread_create(&stopper, NULL, shutdown_worker, NULL) == 0, "shutdown creation failed");
  CHECK(wait_for_test_hook(flight_recorder_test_shutdown_before_cas_reached), "shutdown did not pause before compare-exchange");

  flight_recorder_test_set_write_limit(-1);
  flight_recorder_emit(FLIGHT_EVENT_RADIO_TX, 777778, 2, 3, 4, 5, 6);
  for (int index = 0; index < 100 && flight_recorder_enabled(); ++index)
    sleep_milliseconds(1);
  CHECK(!flight_recorder_enabled(), "writer error did not publish failed state");

  flight_recorder_test_set_shutdown_before_cas_paused(false);
  CHECK(pthread_join(stopper, NULL) == 0, "shutdown join failed");
  flight_recorder_shutdown();
  CHECK(!has_line(directory, "\"kind\":\"capture_footer\""), "clean footer followed writer failure");
  return true;
}

static bool test_write_error_disables_capture(const char *directory)
{
  CHECK(setenv("OAI_FLIGHT_RECORDER_DIR", directory, 1) == 0, "setenv output failed");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_test_set_write_limit(-1);
  flight_recorder_init();
  for (int index = 0; index < 100 && flight_recorder_enabled(); ++index)
    sleep_milliseconds(1);
  CHECK(!flight_recorder_enabled(), "write error did not disable capture");
  flight_recorder_emit(FLIGHT_EVENT_RADIO_TX, 1, 2, 3, 4, 5, 6);
  flight_recorder_shutdown();
  CHECK(!has_line(directory, "\"kind\":\"capture_footer\""), "clean footer followed write failure");
  return true;
}

static bool test_sample_cadence(const char *directory)
{
  (void)directory;
  uint32_t calls = 0;
  for (unsigned period = 0; period < 3; ++period) {
    for (unsigned skipped = 0; skipped < 1023; ++skipped)
      CHECK(!flight_recorder_sample_due(&calls), "successful-read sample arrived early");
    CHECK(flight_recorder_sample_due(&calls), "missing periodic successful-read sample");
  }
  CHECK(calls == 3072, "counter does not count receive calls");
  calls = UINT32_MAX - 1;
  CHECK(!flight_recorder_sample_due(&calls), "unexpected pre-wrap sample");
  CHECK(flight_recorder_sample_due(&calls) && calls == 0, "unsigned wrap interrupted cadence");
  CHECK(!flight_recorder_sample_due(&calls), "unexpected post-wrap sample");
  return true;
}

typedef bool (*test_function_t)(const char *directory);

static bool make_case_directory(char *path, size_t path_size, const char *root, const char *name)
{
  const int result = snprintf(path, path_size, "%s/%s-XXXXXX", root, name);
  if (result < 0 || (size_t)result >= path_size)
    return false;
  return mkdtemp(path) != NULL;
}

static bool run_case(const char *root, const char *name, test_function_t function)
{
  const pid_t child = fork();
  if (child < 0) {
    fprintf(stderr, "fork failed for %s: %d\n", name, errno);
    return false;
  }
  if (child == 0) {
    char directory[TEST_PATH_BYTES];
    if (!make_case_directory(directory, sizeof(directory), root, name))
      _exit(2);
    const bool passed = function(directory);
    _exit(passed ? 0 : 1);
  }

  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "FAIL %s (status=%d)\n", name, status);
    return false;
  }
  printf("PASS %s\n", name);
  return true;
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s <validation-directory>\n", argv[0]);
    return 2;
  }

  const char *root = argv[1];
  struct stat root_status;
  if (stat(root, &root_status) != 0 || !S_ISDIR(root_status.st_mode)) {
    fprintf(stderr, "validation directory is unavailable: %s\n", root);
    return 2;
  }

  const struct {
    const char *name;
    test_function_t function;
  } cases[] = {
      {"sample-cadence", test_sample_cadence},
      {"disabled", test_disabled},
      {"invalid-path", test_invalid_path},
      {"ordering-timestamps-short-writes", test_ordering_timestamps_and_short_writes},
      {"saturation", test_saturation},
      {"no-slot", test_no_slot},
      {"rotation-bound", test_rotation_bound},
      {"active-shutdown", test_shutdown_with_active_producer},
      {"final-drain-after-quiescence", test_final_drain_after_producer_quiescence},
      {"writer-error-races-shutdown", test_writer_error_races_shutdown},
      {"write-error", test_write_error_disables_capture},
  };

  for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
    if (!run_case(root, cases[index].name, cases[index].function))
      return 1;
  }
  return 0;
}
