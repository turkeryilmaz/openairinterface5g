/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "flight_recorder.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define FLIGHT_RECORDER_FILENAME_BYTES 96U
#define FLIGHT_RECORDER_LINE_BYTES 512U
#define FLIGHT_RECORDER_MIN_TOTAL_BYTES 8192U
#define FLIGHT_RECORDER_SLEEP_NS 1000000L
#define FLIGHT_RECORDER_TIMESTAMP_UNAVAILABLE INT64_MIN

#if defined(FLIGHT_RECORDER_FORCE_INERT_TEST)
#define FLIGHT_RECORDER_LOCKFREE_ARCH 0
#elif defined(__x86_64__) || defined(__aarch64__)
#define FLIGHT_RECORDER_LOCKFREE_ARCH 1
#else
#define FLIGHT_RECORDER_LOCKFREE_ARCH 0
#endif

#if FLIGHT_RECORDER_LOCKFREE_ARCH

_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "flight recorder state atomics must be always lock-free");
_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "flight recorder 64-bit atomics must be always lock-free");

_Static_assert((FLIGHT_RECORDER_RING_RECORDS & (FLIGHT_RECORDER_RING_RECORDS - 1U)) == 0,
               "flight recorder ring capacity must be a power of two");

typedef enum {
  recorder_off = 0,
  recorder_starting,
  recorder_running,
  recorder_stopping,
  recorder_failed,
  recorder_stopped,
} recorder_state_t;

typedef struct {
  uint64_t sequence;
  int64_t mono_ns;
  int64_t realtime_ns;
  int64_t values[6];
  uint32_t event;
} flight_record_t;

typedef struct {
  atomic_uint_fast64_t write_index;
  atomic_uint_fast64_t read_index;
  atomic_uint_fast64_t dropped_full;
  flight_record_t records[FLIGHT_RECORDER_RING_RECORDS];
} __attribute__((aligned(64))) flight_ring_t;

typedef struct {
  int fd;
  uint64_t bytes;
  uint64_t generation;
  uint64_t first_sequence;
  uint64_t last_sequence;
  char name[FLIGHT_RECORDER_FILENAME_BYTES];
} recorder_file_t;

static atomic_int g_state = ATOMIC_VAR_INIT(recorder_off);
static atomic_uint_fast64_t g_active_emits = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_next_ring = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_sequence = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_dropped_no_slot = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_no_slot_threads = ATOMIC_VAR_INIT(0);
static atomic_uint_fast64_t g_invalid_timestamps = ATOMIC_VAR_INIT(0);
static atomic_bool g_writer_start_permitted = ATOMIC_VAR_INIT(false);
static pthread_mutex_t g_lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_writer;
static bool g_writer_started;
static bool g_atexit_registered;
static int g_directory_fd = -1;
static uint64_t g_capture_id;
static uint64_t g_file_limit;
static int g_current_file = -1;
static flight_ring_t g_rings[FLIGHT_RECORDER_MAX_THREAD_RINGS];
static recorder_file_t g_files[FLIGHT_RECORDER_MAX_FILES];

static _Thread_local int tls_ring = -1;

#ifdef FLIGHT_RECORDER_TESTING
static atomic_int g_test_write_limit = ATOMIC_VAR_INIT(0);
static atomic_int g_test_writer_policy = ATOMIC_VAR_INIT(-1);
static atomic_bool g_test_writer_paused = ATOMIC_VAR_INIT(false);
static atomic_bool g_test_emit_after_check_paused = ATOMIC_VAR_INIT(false);
static atomic_bool g_test_emit_after_check_reached = ATOMIC_VAR_INIT(false);
static atomic_bool g_test_writer_stop_empty_paused = ATOMIC_VAR_INIT(false);
static atomic_bool g_test_writer_stop_empty_reached = ATOMIC_VAR_INIT(false);
static atomic_bool g_test_shutdown_before_cas_paused = ATOMIC_VAR_INIT(false);
static atomic_bool g_test_shutdown_before_cas_reached = ATOMIC_VAR_INIT(false);

void flight_recorder_test_set_write_limit(int byte_limit)
{
  atomic_store_explicit(&g_test_write_limit, byte_limit, memory_order_release);
}

int flight_recorder_test_writer_policy(void)
{
  return atomic_load_explicit(&g_test_writer_policy, memory_order_acquire);
}

void flight_recorder_test_set_writer_paused(bool paused)
{
  atomic_store_explicit(&g_test_writer_paused, paused, memory_order_release);
}

void flight_recorder_test_set_emit_after_check_paused(bool paused)
{
  atomic_store_explicit(&g_test_emit_after_check_reached, false, memory_order_release);
  atomic_store_explicit(&g_test_emit_after_check_paused, paused, memory_order_release);
}

bool flight_recorder_test_emit_after_check_reached(void)
{
  return atomic_load_explicit(&g_test_emit_after_check_reached, memory_order_acquire);
}

void flight_recorder_test_set_writer_stop_empty_paused(bool paused)
{
  atomic_store_explicit(&g_test_writer_stop_empty_reached, false, memory_order_release);
  atomic_store_explicit(&g_test_writer_stop_empty_paused, paused, memory_order_release);
}

bool flight_recorder_test_writer_stop_empty_reached(void)
{
  return atomic_load_explicit(&g_test_writer_stop_empty_reached, memory_order_acquire);
}

void flight_recorder_test_set_shutdown_before_cas_paused(bool paused)
{
  atomic_store_explicit(&g_test_shutdown_before_cas_reached, false, memory_order_release);
  atomic_store_explicit(&g_test_shutdown_before_cas_paused, paused, memory_order_release);
}

bool flight_recorder_test_shutdown_before_cas_reached(void)
{
  return atomic_load_explicit(&g_test_shutdown_before_cas_reached, memory_order_acquire);
}
#endif

static void recorder_stderr(const char *message, int error_number)
{
  if (error_number != 0)
    fprintf(stderr, "flight recorder disabled: %s (errno=%d)\n", message, error_number);
  else
    fprintf(stderr, "flight recorder disabled: %s\n", message);
}

static void recorder_fail(const char *message, int error_number)
{
  atomic_store_explicit(&g_state, recorder_failed, memory_order_seq_cst);
  recorder_stderr(message, error_number);
}

static bool time_to_ns(clockid_t clock_id, int64_t *result)
{
  struct timespec timestamp;
  if (clock_gettime(clock_id, &timestamp) != 0) {
    *result = FLIGHT_RECORDER_TIMESTAMP_UNAVAILABLE;
    return false;
  }

  const int64_t max_seconds = INT64_MAX / INT64_C(1000000000);
  const int64_t max_nanoseconds = INT64_MAX % INT64_C(1000000000);
  if (timestamp.tv_sec < 0 || timestamp.tv_sec > max_seconds
      || (timestamp.tv_sec == max_seconds && timestamp.tv_nsec > max_nanoseconds)) {
    *result = FLIGHT_RECORDER_TIMESTAMP_UNAVAILABLE;
    return false;
  }

  *result = (int64_t)timestamp.tv_sec * INT64_C(1000000000) + timestamp.tv_nsec;
  return true;
}

static bool ring_put(flight_ring_t *ring, const flight_record_t *record)
{
  /*
   * This is the same acquire/release SPSC publication pattern as
   * common/utils/ds/spsc_q.c, kept static here so a producer never allocates.
   * Each TLS registration owns exactly one producer side of a ring.
   */
  const uint64_t write_index = atomic_load_explicit(&ring->write_index, memory_order_relaxed);
  const uint64_t read_index = atomic_load_explicit(&ring->read_index, memory_order_acquire);
  if (write_index - read_index >= FLIGHT_RECORDER_RING_RECORDS) {
    atomic_fetch_add_explicit(&ring->dropped_full, 1, memory_order_relaxed);
    return false;
  }

  ring->records[write_index & (FLIGHT_RECORDER_RING_RECORDS - 1U)] = *record;
  atomic_store_explicit(&ring->write_index, write_index + 1, memory_order_release);
  return true;
}

static bool ring_get(flight_ring_t *ring, flight_record_t *record)
{
  const uint64_t read_index = atomic_load_explicit(&ring->read_index, memory_order_relaxed);
  const uint64_t write_index = atomic_load_explicit(&ring->write_index, memory_order_acquire);
  if (read_index == write_index)
    return false;

  *record = ring->records[read_index & (FLIGHT_RECORDER_RING_RECORDS - 1U)];
  atomic_store_explicit(&ring->read_index, read_index + 1, memory_order_release);
  return true;
}

static ssize_t recorder_write(int fd, const void *buffer, size_t length)
{
#ifdef FLIGHT_RECORDER_TESTING
  const int byte_limit = atomic_load_explicit(&g_test_write_limit, memory_order_acquire);
  if (byte_limit < 0) {
    errno = EIO;
    return -1;
  }
  if (byte_limit > 0 && length > (size_t)byte_limit)
    length = (size_t)byte_limit;
#endif
  return write(fd, buffer, length);
}

static bool write_all(int fd, const char *buffer, size_t length)
{
  size_t written = 0;
  while (written < length) {
    const ssize_t result = recorder_write(fd, buffer + written, length - written);
    if (result > 0) {
      written += (size_t)result;
      continue;
    }
    if (result < 0 && errno == EINTR)
      continue;
    if (result == 0)
      errno = EIO;
    return false;
  }
  return true;
}

static bool format_line(char *buffer, size_t buffer_size, size_t *length, const char *format, ...)
{
  va_list arguments;
  va_start(arguments, format);
  const int result = vsnprintf(buffer, buffer_size, format, arguments);
  va_end(arguments);
  if (result < 0 || (size_t)result >= buffer_size)
    return false;
  *length = (size_t)result;
  return true;
}

static bool writer_write_raw(const char *line, size_t length)
{
  if (g_current_file < 0 || g_current_file >= (int)FLIGHT_RECORDER_MAX_FILES) {
    recorder_fail("writer has no output file", 0);
    return false;
  }

  recorder_file_t *file = &g_files[g_current_file];
  if (length > g_file_limit || file->bytes > g_file_limit - length) {
    recorder_fail("record exceeds bounded output file", EFBIG);
    return false;
  }

  if (!write_all(file->fd, line, length)) {
    recorder_fail("output write failed", errno);
    return false;
  }

  file->bytes += length;
  return true;
}

static bool writer_ensure_space(size_t length);

static bool writer_emit_clock_correlation(bool rotate_if_needed)
{
  int64_t mono_before;
  int64_t realtime;
  int64_t mono_after;
  const bool mono_before_available = time_to_ns(CLOCK_MONOTONIC, &mono_before);
  const bool realtime_available = time_to_ns(CLOCK_REALTIME, &realtime);
  const bool mono_after_available = time_to_ns(CLOCK_MONOTONIC, &mono_after);
  const bool available = mono_before_available && realtime_available && mono_after_available;
  const int64_t uncertainty =
      available && mono_after >= mono_before ? mono_after - mono_before : FLIGHT_RECORDER_TIMESTAMP_UNAVAILABLE;
  char line[FLIGHT_RECORDER_LINE_BYTES];
  size_t length;
  if (!format_line(line,
                   sizeof(line),
                   &length,
                   "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"clock_correlation\","
                   "\"available\":%u,\"mono_before_ns\":%" PRId64 ",\"realtime_ns\":%" PRId64 ",\"mono_after_ns\":%" PRId64
                   ",\"bracket_uncertainty_ns\":%" PRId64 "}\n",
                   FLIGHT_RECORDER_SCHEMA_VERSION,
                   available ? 1U : 0U,
                   mono_before,
                   realtime,
                   mono_after,
                   uncertainty)) {
    recorder_fail("clock correlation formatting failed", 0);
    return false;
  }

  if (rotate_if_needed && !writer_ensure_space(length))
    return false;
  return writer_write_raw(line, length);
}

static bool writer_write_file_header(unsigned int slot, uint64_t previous_first, uint64_t previous_last)
{
  char line[FLIGHT_RECORDER_LINE_BYTES];
  size_t length;
  const recorder_file_t *file = &g_files[slot];
  const unsigned int was_overwritten = previous_last != 0 ? 1U : 0U;
  if (!format_line(line,
                   sizeof(line),
                   &length,
                   "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"file_begin\","
                   "\"capture_available\":1,\"file_slot\":%u,\"file_generation\":%" PRIu64 ",\"file_limit_bytes\":%" PRIu64
                   ",\"overwrites_available\":%u,"
                   "\"overwrites_sequence_first\":%" PRIu64 ",\"overwrites_sequence_last\":%" PRIu64
                   ",\"ring_capacity\":%u,\"max_rings\":%u,\"payload_truncated\":0}\n",
                   FLIGHT_RECORDER_SCHEMA_VERSION,
                   slot,
                   file->generation,
                   g_file_limit,
                   was_overwritten,
                   previous_first,
                   previous_last,
                   FLIGHT_RECORDER_RING_RECORDS,
                   FLIGHT_RECORDER_MAX_THREAD_RINGS)) {
    recorder_fail("file header formatting failed", 0);
    return false;
  }

  if (!writer_write_raw(line, length))
    return false;
  return writer_emit_clock_correlation(false);
}

static bool writer_open_file(unsigned int slot)
{
  if (slot >= FLIGHT_RECORDER_MAX_FILES) {
    recorder_fail("invalid output file slot", EINVAL);
    return false;
  }

  recorder_file_t *file = &g_files[slot];
  const uint64_t previous_first = file->first_sequence;
  const uint64_t previous_last = file->last_sequence;

  if (file->fd < 0) {
    const int fd = openat(g_directory_fd, file->name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
      recorder_fail("output file creation failed", errno);
      return false;
    }
    if (fchmod(fd, 0600) != 0) {
      const int saved_errno = errno;
      close(fd);
      recorder_fail("output file permissions failed", saved_errno);
      return false;
    }
    file->fd = fd;
  } else {
    if (ftruncate(file->fd, 0) != 0 || lseek(file->fd, 0, SEEK_SET) < 0) {
      recorder_fail("output rotation failed", errno);
      return false;
    }
  }

  file->bytes = 0;
  file->generation++;
  file->first_sequence = 0;
  file->last_sequence = 0;
  g_current_file = (int)slot;
  return writer_write_file_header(slot, previous_first, previous_last);
}

static bool writer_rotate(void)
{
  const unsigned int next = g_current_file < 0 ? 0U : ((unsigned int)g_current_file + 1U) % FLIGHT_RECORDER_MAX_FILES;
  return writer_open_file(next);
}

static bool writer_ensure_space(size_t length)
{
  if (g_current_file < 0 || g_current_file >= (int)FLIGHT_RECORDER_MAX_FILES) {
    recorder_fail("writer has no output file", 0);
    return false;
  }
  if (length > g_file_limit) {
    recorder_fail("record exceeds bounded output file", EFBIG);
    return false;
  }
  if (g_files[g_current_file].bytes > g_file_limit - length)
    return writer_rotate();
  return true;
}

static bool writer_write_event(unsigned int ring_id, const flight_record_t *record)
{
  char line[FLIGHT_RECORDER_LINE_BYTES];
  size_t length;
  if (!format_line(line,
                   sizeof(line),
                   &length,
                   "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"event\","
                   "\"sequence\":%" PRIu64 ",\"ring\":%u,\"mono_ns\":%" PRId64 ",\"realtime_ns\":%" PRId64 ",\"event\":%" PRIu32
                   ",\"a\":%" PRId64 ",\"b\":%" PRId64 ",\"c\":%" PRId64 ",\"d\":%" PRId64 ",\"e\":%" PRId64 ",\"f\":%" PRId64
                   ",\"payload_truncated\":0}\n",
                   FLIGHT_RECORDER_SCHEMA_VERSION,
                   record->sequence,
                   ring_id,
                   record->mono_ns,
                   record->realtime_ns,
                   record->event,
                   record->values[0],
                   record->values[1],
                   record->values[2],
                   record->values[3],
                   record->values[4],
                   record->values[5])) {
    recorder_fail("event formatting failed", 0);
    return false;
  }

  if (g_current_file < 0 && !writer_open_file(0))
    return false;
  if (!writer_ensure_space(length))
    return false;
  if (!writer_write_raw(line, length))
    return false;

  recorder_file_t *file = &g_files[g_current_file];
  if (file->first_sequence == 0)
    file->first_sequence = record->sequence;
  file->last_sequence = record->sequence;
  return true;
}

static uint64_t dropped_full_total(void)
{
  uint64_t total = 0;
  for (unsigned int index = 0; index < FLIGHT_RECORDER_MAX_THREAD_RINGS; ++index)
    total += atomic_load_explicit(&g_rings[index].dropped_full, memory_order_relaxed);
  return total;
}

static bool writer_emit_health(void)
{
  char line[FLIGHT_RECORDER_LINE_BYTES];
  size_t length;
  if (!format_line(line,
                   sizeof(line),
                   &length,
                   "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"health\","
                   "\"capture_available\":1,\"writer_errors\":0,\"dropped_ring_full\":%" PRIu64 ",\"dropped_no_slot\":%" PRIu64
                   ",\"no_slot_threads\":%" PRIu64 ",\"invalid_timestamp_records\":%" PRIu64 ",\"payload_truncated\":0}\n",
                   FLIGHT_RECORDER_SCHEMA_VERSION,
                   dropped_full_total(),
                   atomic_load_explicit(&g_dropped_no_slot, memory_order_relaxed),
                   atomic_load_explicit(&g_no_slot_threads, memory_order_relaxed),
                   atomic_load_explicit(&g_invalid_timestamps, memory_order_relaxed))) {
    recorder_fail("health formatting failed", 0);
    return false;
  }

  if (!writer_ensure_space(length))
    return false;
  return writer_write_raw(line, length);
}

static bool writer_write_footer(void)
{
  char line[FLIGHT_RECORDER_LINE_BYTES];
  size_t length;
  const uint64_t assigned = atomic_load_explicit(&g_next_ring, memory_order_relaxed);
  const uint64_t rings_assigned = assigned > FLIGHT_RECORDER_MAX_THREAD_RINGS ? FLIGHT_RECORDER_MAX_THREAD_RINGS : assigned;
  if (!format_line(line,
                   sizeof(line),
                   &length,
                   "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"capture_footer\","
                   "\"clean\":1,\"writer_errors\":0,\"dropped_ring_full\":%" PRIu64 ",\"dropped_no_slot\":%" PRIu64
                   ",\"no_slot_threads\":%" PRIu64 ",\"invalid_timestamp_records\":%" PRIu64 ",\"payload_truncated\":0,"
                   "\"rings_assigned\":%" PRIu64 "}\n",
                   FLIGHT_RECORDER_SCHEMA_VERSION,
                   dropped_full_total(),
                   atomic_load_explicit(&g_dropped_no_slot, memory_order_relaxed),
                   atomic_load_explicit(&g_no_slot_threads, memory_order_relaxed),
                   atomic_load_explicit(&g_invalid_timestamps, memory_order_relaxed),
                   rings_assigned)) {
    recorder_fail("footer formatting failed", 0);
    return false;
  }

  if (g_current_file < 0 && !writer_open_file(0))
    return false;
  if (!writer_ensure_space(length))
    return false;
  return writer_write_raw(line, length);
}

static bool writer_drain_once(void)
{
  bool drained = false;
  for (unsigned int ring_id = 0; ring_id < FLIGHT_RECORDER_MAX_THREAD_RINGS; ++ring_id) {
    flight_record_t record;
    for (unsigned int record_count = 0; record_count < FLIGHT_RECORDER_RING_RECORDS && ring_get(&g_rings[ring_id], &record);
         ++record_count) {
      drained = true;
      if (!writer_write_event(ring_id, &record))
        return false;
    }
  }
  return drained;
}

static void writer_close_files(void)
{
  for (unsigned int index = 0; index < FLIGHT_RECORDER_MAX_FILES; ++index) {
    if (g_files[index].fd >= 0) {
      close(g_files[index].fd);
      g_files[index].fd = -1;
    }
  }
}

static bool writer_set_sched_other(void)
{
  int policy = -1;
  struct sched_param parameter;
  if (pthread_getschedparam(pthread_self(), &policy, &parameter) == 0 && policy == SCHED_OTHER) {
#ifdef FLIGHT_RECORDER_TESTING
    atomic_store_explicit(&g_test_writer_policy, policy, memory_order_release);
#endif
    return true;
  }

  memset(&parameter, 0, sizeof(parameter));
  if (pthread_setschedparam(pthread_self(), SCHED_OTHER, &parameter) != 0)
    return false;
  if (pthread_getschedparam(pthread_self(), &policy, &parameter) != 0 || policy != SCHED_OTHER)
    return false;
#ifdef FLIGHT_RECORDER_TESTING
  atomic_store_explicit(&g_test_writer_policy, policy, memory_order_release);
#endif
  return true;
}

static void *writer_main(void *unused)
{
  (void)unused;
  while (!atomic_load_explicit(&g_writer_start_permitted, memory_order_acquire)) {
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
    nanosleep(&pause, NULL);
  }
  const recorder_state_t initial_state = atomic_load_explicit(&g_state, memory_order_seq_cst);
  if (initial_state != recorder_running && initial_state != recorder_stopping)
    return NULL;

  if (!writer_set_sched_other()) {
    recorder_fail("writer could not enter SCHED_OTHER", 0);
    writer_close_files();
    return NULL;
  }

  if (!writer_open_file(0)) {
    writer_close_files();
    return NULL;
  }

#ifdef FLIGHT_RECORDER_TESTING
  while (atomic_load_explicit(&g_test_writer_paused, memory_order_acquire)) {
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
    nanosleep(&pause, NULL);
  }
#endif

  int64_t last_clock_ns = 0;
  while (atomic_load_explicit(&g_state, memory_order_seq_cst) == recorder_running) {
    int64_t now_ns;
    if (time_to_ns(CLOCK_MONOTONIC, &now_ns) && now_ns - last_clock_ns >= INT64_C(1000000000)) {
      if (!writer_emit_clock_correlation(true) || !writer_emit_health())
        break;
      last_clock_ns = now_ns;
    }
    writer_drain_once();
    if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_running)
      break;

    const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
    nanosleep(&pause, NULL);
  }

  if (atomic_load_explicit(&g_state, memory_order_seq_cst) == recorder_stopping) {
    while (true) {
      const bool drained = writer_drain_once();
#ifdef FLIGHT_RECORDER_TESTING
      if (!drained && atomic_load_explicit(&g_test_writer_stop_empty_paused, memory_order_acquire)) {
        atomic_store_explicit(&g_test_writer_stop_empty_reached, true, memory_order_release);
        while (atomic_load_explicit(&g_test_writer_stop_empty_paused, memory_order_acquire)) {
          const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
          nanosleep(&pause, NULL);
        }
      }
#endif
      if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_stopping)
        break;
      if (atomic_load_explicit(&g_active_emits, memory_order_seq_cst) == 0) {
        /* Once active is zero, no enrolled producer can publish after this final bounded pass. */
        writer_drain_once();
        break;
      }
      if (!drained) {
        const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
        nanosleep(&pause, NULL);
      }
    }
    if (atomic_load_explicit(&g_state, memory_order_seq_cst) == recorder_stopping)
      writer_write_footer();
  }

  writer_close_files();
  return NULL;
}

static bool configure_output_directory(void)
{
  const char *directory = getenv("OAI_FLIGHT_RECORDER_DIR");
  if (directory == NULL || directory[0] == '\0')
    return false;

  const char *max_bytes = getenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  uint64_t total_limit = FLIGHT_RECORDER_MAX_TOTAL_BYTES;
  if (max_bytes != NULL && max_bytes[0] != '\0') {
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(max_bytes, &end, 10);
    if (errno != 0 || end == max_bytes || *end != '\0' || parsed < FLIGHT_RECORDER_MIN_TOTAL_BYTES
        || parsed > FLIGHT_RECORDER_MAX_TOTAL_BYTES) {
      recorder_stderr("invalid OAI_FLIGHT_RECORDER_MAX_BYTES", EINVAL);
      return false;
    }
    total_limit = (uint64_t)parsed;
  }
  g_file_limit = total_limit / FLIGHT_RECORDER_MAX_FILES;

  g_directory_fd = open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (g_directory_fd < 0) {
    recorder_stderr("output directory unavailable", errno);
    return false;
  }

  struct stat directory_status;
  if (fstat(g_directory_fd, &directory_status) != 0 || !S_ISDIR(directory_status.st_mode)) {
    const int saved_errno = errno == 0 ? ENOTDIR : errno;
    close(g_directory_fd);
    g_directory_fd = -1;
    recorder_stderr("output directory validation failed", saved_errno);
    return false;
  }

  int64_t monotonic_start;
  if (!time_to_ns(CLOCK_MONOTONIC, &monotonic_start)) {
    close(g_directory_fd);
    g_directory_fd = -1;
    recorder_stderr("capture identifier clock unavailable", 0);
    return false;
  }
  g_capture_id = (uint64_t)monotonic_start;

  for (unsigned int index = 0; index < FLIGHT_RECORDER_MAX_FILES; ++index) {
    const int result = snprintf(g_files[index].name,
                                sizeof(g_files[index].name),
                                "oai-flight-recorder-%ld-%016" PRIx64 "-%u.ndjson",
                                (long)getpid(),
                                g_capture_id,
                                index);
    if (result < 0 || (size_t)result >= sizeof(g_files[index].name)) {
      close(g_directory_fd);
      g_directory_fd = -1;
      recorder_stderr("output filename formatting failed", 0);
      return false;
    }
  }

  const int first_fd = openat(g_directory_fd, g_files[0].name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (first_fd < 0) {
    const int saved_errno = errno;
    close(g_directory_fd);
    g_directory_fd = -1;
    recorder_stderr("output file creation failed", saved_errno);
    return false;
  }
  if (fchmod(first_fd, 0600) != 0) {
    const int saved_errno = errno;
    close(first_fd);
    close(g_directory_fd);
    g_directory_fd = -1;
    recorder_stderr("output file permissions failed", saved_errno);
    return false;
  }
  g_files[0].fd = first_fd;
  return true;
}

static bool start_writer(void)
{
  pthread_attr_t attributes;
  int result = pthread_attr_init(&attributes);
  if (result != 0) {
    recorder_stderr("writer attribute initialization failed", result);
    return false;
  }

  bool valid = true;
  struct sched_param parameter;
  memset(&parameter, 0, sizeof(parameter));
  if (pthread_attr_setinheritsched(&attributes, PTHREAD_EXPLICIT_SCHED) != 0
      || pthread_attr_setschedpolicy(&attributes, SCHED_OTHER) != 0 || pthread_attr_setschedparam(&attributes, &parameter) != 0)
    valid = false;

  if (valid)
    result = pthread_create(&g_writer, &attributes, writer_main, NULL);
  pthread_attr_destroy(&attributes);
  if (!valid || result != 0) {
    recorder_stderr("writer creation with SCHED_OTHER failed", valid ? result : 0);
    return false;
  }

  g_writer_started = true;
  return true;
}

void flight_recorder_init(void)
{
  pthread_mutex_lock(&g_lifecycle_lock);
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_off) {
    pthread_mutex_unlock(&g_lifecycle_lock);
    return;
  }

  const char *directory = getenv("OAI_FLIGHT_RECORDER_DIR");
  if (directory == NULL || directory[0] == '\0') {
    pthread_mutex_unlock(&g_lifecycle_lock);
    return;
  }
  if (!atomic_is_lock_free(&g_state) || !atomic_is_lock_free(&g_active_emits)) {
    recorder_stderr("producer atomics are not lock-free", 0);
    pthread_mutex_unlock(&g_lifecycle_lock);
    return;
  }

  atomic_store_explicit(&g_state, recorder_starting, memory_order_seq_cst);

  /*
   * The rings stay static for late-producer safety. Touching them here faults
   * pages in outside any real-time producer path.
   */
  memset(g_rings, 0, sizeof(g_rings));
  memset(g_files, 0, sizeof(g_files));
  for (unsigned int index = 0; index < FLIGHT_RECORDER_MAX_FILES; ++index)
    g_files[index].fd = -1;
  atomic_store_explicit(&g_active_emits, 0, memory_order_seq_cst);
  atomic_store_explicit(&g_next_ring, 0, memory_order_relaxed);
  atomic_store_explicit(&g_sequence, 0, memory_order_relaxed);
  atomic_store_explicit(&g_dropped_no_slot, 0, memory_order_relaxed);
  atomic_store_explicit(&g_no_slot_threads, 0, memory_order_relaxed);
  atomic_store_explicit(&g_invalid_timestamps, 0, memory_order_relaxed);
  atomic_store_explicit(&g_writer_start_permitted, false, memory_order_relaxed);
  g_current_file = -1;

  if (!configure_output_directory() || !start_writer()) {
    if (g_directory_fd >= 0) {
      close(g_directory_fd);
      g_directory_fd = -1;
    }
    writer_close_files();
    atomic_store_explicit(&g_state, recorder_off, memory_order_seq_cst);
    pthread_mutex_unlock(&g_lifecycle_lock);
    return;
  }

  atomic_store_explicit(&g_state, recorder_running, memory_order_seq_cst);
  atomic_store_explicit(&g_writer_start_permitted, true, memory_order_release);
  if (!g_atexit_registered) {
    if (atexit(flight_recorder_shutdown) == 0)
      g_atexit_registered = true;
    else
      recorder_stderr("atexit shutdown registration failed", 0);
  }
  pthread_mutex_unlock(&g_lifecycle_lock);
}

bool flight_recorder_enabled(void)
{
  return atomic_load_explicit(&g_state, memory_order_seq_cst) == recorder_running;
}

void flight_recorder_emit(uint32_t event, int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f)
{
  /*
   * Enrollment and stop share one seq_cst order across g_state and g_active_emits.
   * If the second state load sees running, then this enrollment RMW and that
   * load both precede the stop store in that order. A stop-phase zero observation
   * therefore cannot precede such an enrolled producer. If the stop store is
   * earlier, the second load sees stopping and the producer publishes nothing.
   * The seq_cst completion decrement also releases ring publication before the
   * writer's zero observation and its final SPSC acquire drain.
   */
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_running)
    return;

  atomic_fetch_add_explicit(&g_active_emits, 1, memory_order_seq_cst);
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_running) {
    atomic_fetch_sub_explicit(&g_active_emits, 1, memory_order_seq_cst);
    return;
  }

#ifdef FLIGHT_RECORDER_TESTING
  if (atomic_load_explicit(&g_test_emit_after_check_paused, memory_order_acquire)) {
    atomic_store_explicit(&g_test_emit_after_check_reached, true, memory_order_release);
    while (atomic_load_explicit(&g_test_emit_after_check_paused, memory_order_acquire)) {
      const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
      nanosleep(&pause, NULL);
    }
  }
#endif

  if (tls_ring == -1) {
    const uint64_t assigned = atomic_fetch_add_explicit(&g_next_ring, 1, memory_order_relaxed);
    if (assigned >= FLIGHT_RECORDER_MAX_THREAD_RINGS) {
      tls_ring = -2;
      atomic_fetch_add_explicit(&g_no_slot_threads, 1, memory_order_relaxed);
    } else {
      tls_ring = (int)assigned;
    }
  }

  if (tls_ring == -2) {
    atomic_fetch_add_explicit(&g_dropped_no_slot, 1, memory_order_relaxed);
    atomic_fetch_sub_explicit(&g_active_emits, 1, memory_order_seq_cst);
    return;
  }

  flight_record_t record = {
      .event = event,
      .values = {a, b, c, d, e, f},
  };
  const bool monotonic_valid = time_to_ns(CLOCK_MONOTONIC, &record.mono_ns);
  const bool realtime_valid = time_to_ns(CLOCK_REALTIME, &record.realtime_ns);
  if (!monotonic_valid || !realtime_valid)
    atomic_fetch_add_explicit(&g_invalid_timestamps, 1, memory_order_relaxed);
  record.sequence = atomic_fetch_add_explicit(&g_sequence, 1, memory_order_relaxed) + 1;

  ring_put(&g_rings[tls_ring], &record);
  atomic_fetch_sub_explicit(&g_active_emits, 1, memory_order_seq_cst);
}

void flight_recorder_shutdown(void)
{
  pthread_mutex_lock(&g_lifecycle_lock);
  const recorder_state_t state = atomic_load_explicit(&g_state, memory_order_seq_cst);
  if (state != recorder_running && state != recorder_failed) {
    pthread_mutex_unlock(&g_lifecycle_lock);
    return;
  }

#ifdef FLIGHT_RECORDER_TESTING
  if (state == recorder_running && atomic_load_explicit(&g_test_shutdown_before_cas_paused, memory_order_acquire)) {
    atomic_store_explicit(&g_test_shutdown_before_cas_reached, true, memory_order_release);
    while (atomic_load_explicit(&g_test_shutdown_before_cas_paused, memory_order_acquire)) {
      const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
      nanosleep(&pause, NULL);
    }
  }
#endif

  if (state == recorder_running) {
    int expected = recorder_running;
    if (!atomic_compare_exchange_strong_explicit(&g_state, &expected, recorder_stopping, memory_order_seq_cst, memory_order_seq_cst)
        && expected != recorder_failed) {
      pthread_mutex_unlock(&g_lifecycle_lock);
      return;
    }
  }

  if (g_writer_started) {
    pthread_join(g_writer, NULL);
    g_writer_started = false;
  }

  if (g_directory_fd >= 0) {
    close(g_directory_fd);
    g_directory_fd = -1;
  }
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) == recorder_stopping)
    atomic_store_explicit(&g_state, recorder_stopped, memory_order_seq_cst);
  pthread_mutex_unlock(&g_lifecycle_lock);
}

#else /* FLIGHT_RECORDER_LOCKFREE_ARCH */

void flight_recorder_init(void)
{
  const char *directory = getenv("OAI_FLIGHT_RECORDER_DIR");
  if (directory != NULL && directory[0] != '\0')
    fprintf(stderr, "flight recorder disabled: lock-free producer atomics unavailable on this architecture\n");
}

void flight_recorder_shutdown(void)
{
}

bool flight_recorder_enabled(void)
{
  return false;
}

void flight_recorder_emit(uint32_t event, int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f)
{
  (void)event;
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  (void)e;
  (void)f;
}

#endif /* FLIGHT_RECORDER_LOCKFREE_ARCH */
