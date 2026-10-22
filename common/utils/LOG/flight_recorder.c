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
#include <sys/statvfs.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define FLIGHT_RECORDER_FILENAME_BYTES 96U
#define FLIGHT_RECORDER_LINE_BYTES 512U
#define FLIGHT_RECORDER_MIN_TOTAL_BYTES 8192U
#define FLIGHT_RECORDER_SLEEP_NS 1000000L
#define FLIGHT_RECORDER_TIMESTAMP_UNAVAILABLE INT64_MIN
#define FLIGHT_RECORDER_ASYNC_REPEAT_NS INT64_C(100000000)
#define FLIGHT_RECORDER_BATCH_BYTES (64U * 1024U)
#define FLIGHT_RECORDER_RING_QUANTUM 64U
#define FLIGHT_RECORDER_CRITICAL_RESERVE 128U
#define FLIGHT_RECORDER_EVENT_BINS 256U
#define FLIGHT_RECORDER_SAMPLE_GROUPS 32U
#define FLIGHT_RECORDER_SAMPLE_PERIOD 64U
#define FLIGHT_RECORDER_SAMPLE_WARMUP 8U
#define FLIGHT_RECORDER_POWER_GROUPS 3U

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
_Static_assert(FLIGHT_RECORDER_CRITICAL_RESERVE < FLIGHT_RECORDER_RING_RECORDS, "critical reserve must leave bulk capacity");
_Static_assert(FLIGHT_SAMPLE_UE_TX_RELATIVE_POWER - FLIGHT_SAMPLE_UE_TX_POWER_REQUEST + 1 == FLIGHT_RECORDER_POWER_GROUPS,
               "power sampling groups must occupy the fixed cache range");

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
  atomic_uint_fast64_t reserved;
  atomic_uint_fast64_t capacity;
} recorder_event_loss_t;

typedef struct {
  atomic_uint_fast64_t evaluated;
  atomic_uint_fast64_t retained;
  atomic_uint_fast64_t reduced;
} recorder_sampling_t;

/* Producer-owned state shares the rings' initialization and retained lifetime.
 * The writer reads sampling counters, never these signatures. */
typedef struct {
  bool valid;
  int64_t instance;
  int64_t values[4];
  uint64_t opportunities;
} recorder_power_sampling_t;

typedef struct {
  atomic_uint_fast64_t write_index;
  atomic_uint_fast64_t read_index;
  atomic_uint_fast64_t dropped_full;
  atomic_uint_fast64_t dropped_reserved;
  atomic_uint_fast64_t dropped_capacity;
  atomic_uint_fast64_t occupancy_high_water;
  recorder_event_loss_t event_loss[FLIGHT_RECORDER_EVENT_BINS + 1U];
  recorder_sampling_t sampling[FLIGHT_RECORDER_SAMPLE_GROUPS];
  recorder_power_sampling_t power_sampling[FLIGHT_RECORDER_POWER_GROUPS][FLIGHT_UE_TX_CHANNEL_SRS];
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

/* Sole-writer state, separate from producer rings. Count only suppressed
 * records consumed from one ring, never native async callbacks or ring drops. */
typedef struct {
  bool active;
  int64_t device;
  int64_t code;
  int64_t channel;
  int64_t flags;
  int64_t window_first_ns;
  uint64_t count;
  uint64_t sequence_first;
  uint64_t sequence_last;
  int64_t mono_first_ns;
  int64_t mono_last_ns;
  int64_t device_ticks_first;
  int64_t device_ticks_last;
} async_repeat_t;

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
static uid_t g_output_uid;
static gid_t g_output_gid;
static uint64_t g_capture_id;
static uint64_t g_file_limit;
static uint64_t g_total_limit;
static uint64_t g_total_written;
static uint64_t g_next_file_number;
static uint64_t g_min_free_bytes;
static bool g_detail_full;
static int g_current_file = -1;
static flight_ring_t g_rings[FLIGHT_RECORDER_MAX_THREAD_RINGS];
static recorder_file_t g_files[FLIGHT_RECORDER_MAX_FILES];
static async_repeat_t g_async_repeats[FLIGHT_RECORDER_MAX_THREAD_RINGS];

/* Only the writer uses the batch and counters. Budget counters include bytes
 * admitted to this batch; physical counters advance on successful writes. */
static char g_write_batch[FLIGHT_RECORDER_BATCH_BYTES];
static size_t g_write_batch_used;
static bool g_write_failed;
static unsigned int g_next_drain_ring;
static struct {
  uint64_t records_drained;
  uint64_t lines_written;
  uint64_t bytes_written;
  uint64_t write_calls;
  uint64_t drain_passes;
  uint64_t drain_wall_ns;
  uint64_t drain_clock_errors;
} g_writer_stats;

static _Thread_local int tls_ring = -1;

#ifdef FLIGHT_RECORDER_TESTING
static uint64_t g_test_file_limit;
void flight_recorder_test_set_file_limit(uint64_t bytes)
{
  g_test_file_limit = bytes;
}
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

static bool record_is_bulk(const flight_record_t *record)
{
  const int64_t *v = record->values;
  /* This is a whitelist of successful numeric observations. Unknown events,
   * transition/application records and faults keep the full FIFO capacity.
   * Payload tests follow the emitting call sites, including short TX returns,
   * queue discards and unavailable/late deadline evidence. */
  switch (record->event) {
    case FLIGHT_EVENT_UE_MEASUREMENTS:
    case FLIGHT_EVENT_GNB_SLOT:
      return true;
    case FLIGHT_EVENT_GNB_UE_BYTES:
      return v[5] == 0;
    case FLIGHT_EVENT_GNB_UE_RADIO:
      return v[3] == 0 && v[4] == 0 && v[5] == 0;
    case FLIGHT_EVENT_RADIO_RX_LEVEL:
      return v[1] > 0 && v[3] != INT64_MIN && v[4] != INT64_MIN && ((uint64_t)v[5] >> 32) == 0;
    case FLIGHT_EVENT_UE_TX_POWER_REQUEST:
      return v[2] >= FLIGHT_UE_TX_CHANNEL_PUSCH && v[2] <= FLIGHT_UE_TX_CHANNEL_SRS && v[4] > 0;
    case FLIGHT_EVENT_RADIO_TX_LEVEL:
      return v[2] > 0 && v[3] > 0 && ((uint64_t)v[5] & UINT32_MAX) == 0;
    case FLIGHT_EVENT_RADIO_TX_LEVEL_STATE:
      return v[2] > 0 && v[3] == v[2] && v[4] != INT64_MIN && v[5] != INT64_MIN;
    case FLIGHT_EVENT_RADIO_TX_POWER:
    case FLIGHT_EVENT_RADIO_TX_RELATIVE_POWER:
      // c packs status (OK=0) and the boolean applied bit at bit 8.
      return v[0] != FLIGHT_UE_TX_CHANNEL_PRACH && (v[2] == 0 || v[2] == 256) && v[4] != INT64_MIN && v[5] != INT64_MIN;
    case FLIGHT_EVENT_RADIO_TX_POWER_SAMPLES:
    case FLIGHT_EVENT_RADIO_TX_RELATIVE_SAMPLES:
      return v[0] != FLIGHT_UE_TX_CHANNEL_PRACH && v[2] > 0 && v[4] != INT64_MIN;
    case FLIGHT_EVENT_RADIO_TX_RELATIVE_QUALITY:
      return v[0] != FLIGHT_UE_TX_CHANNEL_PRACH && v[2] > 0 && v[3] != INT64_MIN && v[4] != INT64_MIN && v[5] != INT64_MIN;
    case FLIGHT_EVENT_RADIO_TX_POWER_QUALITY:
      return v[0] != FLIGHT_UE_TX_CHANNEL_PRACH && v[2] != INT64_MIN && v[3] != INT64_MIN && v[4] != INT64_MIN;
    case FLIGHT_EVENT_RADIO_TX_SUBMIT:
      // Flag 1 is a routine middle-of-burst request, with no future flag bits.
      return v[3] > 0 && v[4] == 1;
    case FLIGHT_EVENT_RADIO_TX_QUEUE:
      // Preserve accumulating queue depth even before the queue is near full.
      return v[2] > 0 && v[3] >= 0 && v[3] <= 1 && (v[4] == 1 || v[4] == 2) && v[5] == 0;
    case FLIGHT_EVENT_RADIO_TX_SEND:
      // Metadata 1 is timed data without start/end-of-burst markers.
      return v[3] > 0 && v[4] == 1 && v[5] > 0;
    case FLIGHT_EVENT_RADIO_TX_RESULT:
      return v[3] > 0 && v[4] == v[3] && v[5] == 0;
    case FLIGHT_EVENT_UE_TX_SCHEDULE:
      return v[3] >= 0 && v[4] >= 0 && v[5] > 0;
    case FLIGHT_EVENT_UE_TX_DEADLINE:
      return v[3] >= 0 && v[4] != INT64_MIN && v[4] <= 0 && v[5] == 1;
    case FLIGHT_EVENT_UE_TX_ANCHOR:
      return v[2] >= 0 && v[3] >= 0 && v[4] >= 0;
    default:
      return false;
  }
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
  const uint64_t occupancy = write_index - read_index;
  const bool full = occupancy >= FLIGHT_RECORDER_RING_RECORDS;
  if (full || (occupancy >= FLIGHT_RECORDER_RING_RECORDS - FLIGHT_RECORDER_CRITICAL_RESERVE && record_is_bulk(record))) {
    atomic_fetch_add_explicit(&ring->dropped_full, 1, memory_order_relaxed);
    const unsigned int event_bin = record->event < FLIGHT_RECORDER_EVENT_BINS ? record->event : FLIGHT_RECORDER_EVENT_BINS;
    if (full) {
      atomic_fetch_add_explicit(&ring->dropped_capacity, 1, memory_order_relaxed);
      atomic_fetch_add_explicit(&ring->event_loss[event_bin].capacity, 1, memory_order_relaxed);
    } else {
      atomic_fetch_add_explicit(&ring->dropped_reserved, 1, memory_order_relaxed);
      atomic_fetch_add_explicit(&ring->event_loss[event_bin].reserved, 1, memory_order_relaxed);
    }
    return false;
  }

  if (occupancy + 1 > atomic_load_explicit(&ring->occupancy_high_water, memory_order_relaxed))
    atomic_store_explicit(&ring->occupancy_high_water, occupancy + 1, memory_order_relaxed);
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
    ++g_writer_stats.write_calls;
    const ssize_t result = recorder_write(fd, buffer + written, length - written);
    if (result > 0) {
      g_writer_stats.bytes_written += (size_t)result;
      for (size_t index = written; index < written + (size_t)result; ++index)
        g_writer_stats.lines_written += buffer[index] == '\n';
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

static bool writer_flush_batch(void)
{
  if (g_write_failed)
    return false;
  if (g_write_batch_used == 0)
    return true;
  if (!write_all(g_files[g_current_file].fd, g_write_batch, g_write_batch_used)) {
    const int saved_errno = errno;
    // A partial write cannot be retried from the batch start without duplicates.
    g_write_failed = true;
    g_write_batch_used = 0;
    recorder_fail("output write failed", saved_errno);
    return false;
  }
  g_write_batch_used = 0;
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
  if (g_write_failed)
    return false;
  if (g_current_file < 0 || g_current_file >= (int)FLIGHT_RECORDER_MAX_FILES) {
    recorder_fail("writer has no output file", 0);
    return false;
  }

  recorder_file_t *file = &g_files[g_current_file];
  if (length > g_file_limit || file->bytes > g_file_limit - length) {
    recorder_fail("record exceeds bounded output file", EFBIG);
    return false;
  }

  if (length > UINT64_MAX - g_total_written
      || (g_total_limit && (length > g_total_limit || g_total_written > g_total_limit - length))) {
    recorder_fail("recording byte limit reached; earlier files retained", EFBIG);
    return false;
  }
  if (length > sizeof(g_write_batch)) {
    recorder_fail("record exceeds writer batch", EFBIG);
    return false;
  }
  if (length > sizeof(g_write_batch) - g_write_batch_used && !writer_flush_batch())
    return false;

  memcpy(g_write_batch + g_write_batch_used, line, length);
  g_write_batch_used += length;
  file->bytes += length;
  g_total_written += length;
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

static const char *recorder_detail_mode(void)
{
  return g_detail_full ? "full" : "balanced";
}

static bool writer_emit_sampling_config(void)
{
  char line[FLIGHT_RECORDER_LINE_BYTES];
  size_t length;
  if (!format_line(line,
                   sizeof(line),
                   &length,
                   "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"sampling_config\","
                   "\"detail_mode\":\"%s\",\"selection_period\":%u,\"warmup_opportunities\":%u,\"groups\":%u,"
                   "\"key_mix_version\":1,\"payload_truncated\":0}\n",
                   FLIGHT_RECORDER_SCHEMA_VERSION,
                   recorder_detail_mode(),
                   FLIGHT_RECORDER_SAMPLE_PERIOD,
                   FLIGHT_RECORDER_SAMPLE_WARMUP,
                   FLIGHT_RECORDER_SAMPLE_GROUPS)) {
    recorder_fail("sampling config formatting failed", 0);
    return false;
  }
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
  if (!writer_emit_clock_correlation(false))
    return false;
  return slot != 0 || file->generation != 1 || writer_emit_sampling_config();
}

static bool writer_has_space(void)
{
  if (g_min_free_bytes == 0)
    return true;
  struct statvfs space;
  if (fstatvfs(g_directory_fd, &space) != 0 || space.f_frsize == 0) {
    recorder_fail("free space unavailable", errno);
    return false;
  }
  const uint64_t needed = g_min_free_bytes / space.f_frsize + (g_min_free_bytes % space.f_frsize != 0);
  if (space.f_bavail <= needed) {
    recorder_fail("free-space reserve reached; earlier files retained", ENOSPC);
    return false;
  }
  return true;
}

static bool set_output_permissions(int fd)
{
  // The supervisor gives its private recorder directory to the invoking user.
  // Native files, including rotations, must remain readable by that same user.
  if (geteuid() == 0 && fchown(fd, g_output_uid, g_output_gid) != 0)
    return false;
  return fchmod(fd, 0600) == 0;
}

static bool writer_open_file(unsigned int slot)
{
  if (slot >= FLIGHT_RECORDER_MAX_FILES) {
    recorder_fail("invalid output file slot", EINVAL);
    return false;
  }

  if (!writer_flush_batch())
    return false;
  recorder_file_t *file = &g_files[slot];
  if (!writer_has_space())
    return false;
  if (file->generation > 0) {
    if (file->fd >= 0)
      close(file->fd);
    file->fd = -1;
    int length = snprintf(file->name,
                          sizeof(file->name),
                          "oai-flight-recorder-%ld-%016" PRIx64 "-%" PRIu64 ".ndjson",
                          (long)getpid(),
                          g_capture_id,
                          g_next_file_number++);
    if (length < 0 || (size_t)length >= sizeof(file->name)) {
      recorder_fail("output filename exhausted", EOVERFLOW);
      return false;
    }
  }

  if (file->fd < 0) {
    const int fd = openat(g_directory_fd, file->name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
      recorder_fail("output file creation failed", errno);
      return false;
    }
    if (!set_output_permissions(fd)) {
      const int saved_errno = errno;
      close(fd);
      recorder_fail("output file permissions failed", saved_errno);
      return false;
    }
    file->fd = fd;
  }

  file->bytes = 0;
  file->generation++;
  file->first_sequence = 0;
  file->last_sequence = 0;
  g_current_file = (int)slot;
  return writer_write_file_header(slot, 0, 0);
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

static bool writer_flush_async_repeat(unsigned int ring_id)
{
  async_repeat_t *repeat = &g_async_repeats[ring_id];
  if (repeat->count) {
    char line[FLIGHT_RECORDER_LINE_BYTES];
    size_t length;
    if (!format_line(line,
                     sizeof(line),
                     &length,
                     "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"async_repeat_summary\","
                     "\"event\":%u,\"ring\":%u,\"device\":%" PRId64 ",\"code\":%" PRId64 ",\"channel\":%" PRId64
                     ",\"flags\":%" PRId64 ",\"repeat_count\":%" PRIu64 ",\"sequence_first\":%" PRIu64 ",\"sequence_last\":%" PRIu64
                     ",\"mono_first_ns\":%" PRId64 ",\"mono_last_ns\":%" PRId64 ",\"device_ticks_first\":%" PRId64
                     ",\"device_ticks_last\":%" PRId64 ",\"payload_truncated\":0}\n",
                     FLIGHT_RECORDER_SCHEMA_VERSION,
                     (unsigned int)FLIGHT_EVENT_RADIO_TX_ASYNC,
                     ring_id,
                     repeat->device,
                     repeat->code,
                     repeat->channel,
                     repeat->flags,
                     repeat->count,
                     repeat->sequence_first,
                     repeat->sequence_last,
                     repeat->mono_first_ns,
                     repeat->mono_last_ns,
                     repeat->device_ticks_first,
                     repeat->device_ticks_last)) {
      recorder_fail("async summary formatting failed", 0);
      return false;
    }
    if (!writer_ensure_space(length) || !writer_write_raw(line, length))
      return false;
    recorder_file_t *file = &g_files[g_current_file];
    if (file->first_sequence == 0)
      file->first_sequence = repeat->sequence_first;
    file->last_sequence = repeat->sequence_last;
  }
  *repeat = (async_repeat_t){0};
  return true;
}

static bool writer_write_record(unsigned int ring_id, const flight_record_t *record)
{
  async_repeat_t *repeat = &g_async_repeats[ring_id];
  /* Keep unknown timestamps and future uses of reserved f raw. The first
   * event in each run/window is always visible before any summary delay. */
  const bool eligible = record->event == FLIGHT_EVENT_RADIO_TX_ASYNC && record->values[5] == 0 && record->mono_ns >= 0;
  const bool same_key = eligible && repeat->device == record->values[0] && repeat->code == record->values[1]
                        && repeat->channel == record->values[2] && repeat->flags == record->values[4];
  if (repeat->active
      && (!same_key || record->mono_ns < repeat->window_first_ns
          || record->mono_ns - repeat->window_first_ns >= FLIGHT_RECORDER_ASYNC_REPEAT_NS || repeat->count == UINT64_MAX)) {
    if (!writer_flush_async_repeat(ring_id))
      return false;
  }
  if (repeat->active) {
    if (!repeat->count) {
      repeat->sequence_first = record->sequence;
      repeat->mono_first_ns = record->mono_ns;
      repeat->device_ticks_first = record->values[3];
    }
    ++repeat->count;
    repeat->sequence_last = record->sequence;
    repeat->mono_last_ns = record->mono_ns;
    repeat->device_ticks_last = record->values[3];
    return true;
  }
  if (!writer_write_event(ring_id, record))
    return false;
  if (eligible)
    *repeat = (async_repeat_t){.active = true,
                               .device = record->values[0],
                               .code = record->values[1],
                               .channel = record->values[2],
                               .flags = record->values[4],
                               .window_first_ns = record->mono_ns};
  return true;
}

static bool writer_flush_async_due(int64_t now_ns, bool force)
{
  for (unsigned int ring_id = 0; ring_id < FLIGHT_RECORDER_MAX_THREAD_RINGS; ++ring_id) {
    const async_repeat_t *repeat = &g_async_repeats[ring_id];
    if (repeat->active
        && (force || now_ns < repeat->window_first_ns || now_ns - repeat->window_first_ns >= FLIGHT_RECORDER_ASYNC_REPEAT_NS)
        && !writer_flush_async_repeat(ring_id))
      return false;
  }
  return true;
}

static uint64_t dropped_full_total(void)
{
  uint64_t total = 0;
  for (unsigned int index = 0; index < FLIGHT_RECORDER_MAX_THREAD_RINGS; ++index)
    total += atomic_load_explicit(&g_rings[index].dropped_full, memory_order_relaxed);
  return total;
}

static bool writer_emit_ring_health(int64_t snapshot_ns)
{
  const uint64_t assigned = atomic_load_explicit(&g_next_ring, memory_order_relaxed);
  const unsigned int rings = assigned > FLIGHT_RECORDER_MAX_THREAD_RINGS ? FLIGHT_RECORDER_MAX_THREAD_RINGS : assigned;
  for (unsigned int ring_id = 0; ring_id < rings; ++ring_id) {
    const flight_ring_t *ring = &g_rings[ring_id];
    char line[FLIGHT_RECORDER_LINE_BYTES];
    size_t length;
    time_to_ns(CLOCK_MONOTONIC, &snapshot_ns);
    if (!format_line(line,
                     sizeof(line),
                     &length,
                     "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"ring_health\","
                     "\"snapshot_mono_ns\":%" PRId64 ",\"ring\":%u,\"capacity\":%u,\"occupancy_high_water\":%" PRIu64
                     ",\"dropped_ring_full\":%" PRIu64 ",\"dropped_reserved\":%" PRIu64 ",\"dropped_capacity\":%" PRIu64
                     ",\"payload_truncated\":0}\n",
                     FLIGHT_RECORDER_SCHEMA_VERSION,
                     snapshot_ns,
                     ring_id,
                     FLIGHT_RECORDER_RING_RECORDS,
                     atomic_load_explicit(&ring->occupancy_high_water, memory_order_relaxed),
                     atomic_load_explicit(&ring->dropped_full, memory_order_relaxed),
                     atomic_load_explicit(&ring->dropped_reserved, memory_order_relaxed),
                     atomic_load_explicit(&ring->dropped_capacity, memory_order_relaxed))) {
      recorder_fail("ring health formatting failed", 0);
      return false;
    }
    if (!writer_ensure_space(length) || !writer_write_raw(line, length))
      return false;

    for (unsigned int event_bin = 0; event_bin <= FLIGHT_RECORDER_EVENT_BINS; ++event_bin) {
      const uint64_t reserved = atomic_load_explicit(&ring->event_loss[event_bin].reserved, memory_order_relaxed);
      const uint64_t capacity = atomic_load_explicit(&ring->event_loss[event_bin].capacity, memory_order_relaxed);
      if (reserved == 0 && capacity == 0)
        continue;
      const bool known = event_bin < FLIGHT_RECORDER_EVENT_BINS;
      time_to_ns(CLOCK_MONOTONIC, &snapshot_ns);
      if (!format_line(line,
                       sizeof(line),
                       &length,
                       "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"event_loss\","
                       "\"snapshot_mono_ns\":%" PRId64 ",\"ring\":%u,\"event\":%" PRIu32 ",\"event_known\":%u,"
                       "\"dropped_reserved\":%" PRIu64 ",\"dropped_capacity\":%" PRIu64 ",\"payload_truncated\":0}\n",
                       FLIGHT_RECORDER_SCHEMA_VERSION,
                       snapshot_ns,
                       ring_id,
                       known ? event_bin : UINT32_MAX,
                       known ? 1U : 0U,
                       reserved,
                       capacity)) {
        recorder_fail("event loss formatting failed", 0);
        return false;
      }
      if (!writer_ensure_space(length) || !writer_write_raw(line, length))
        return false;
    }
  }
  return true;
}

static bool writer_emit_writer_health(void)
{
  if (!writer_flush_batch())
    return false;
  int64_t snapshot_ns;
  time_to_ns(CLOCK_MONOTONIC, &snapshot_ns);
  char line[2U * FLIGHT_RECORDER_LINE_BYTES];
  size_t length;
  // Counters describe this snapshot, before encoding/writing its own record.
  if (!format_line(line,
                   sizeof(line),
                   &length,
                   "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"writer_health\","
                   "\"snapshot_mono_ns\":%" PRId64 ",\"records_drained\":%" PRIu64 ",\"lines_written\":%" PRIu64
                   ",\"bytes_written\":%" PRIu64 ",\"bytes_accepted\":%" PRIu64 ",\"write_calls\":%" PRIu64
                   ",\"buffered_bytes\":%zu,\"drain_passes\":%" PRIu64 ",\"drain_wall_ns\":%" PRIu64
                   ",\"drain_clock_errors\":%" PRIu64
                   ",\"batch_capacity_bytes\":%u,\"ring_quantum\":%u,\"critical_reserve_records\":%u,\"payload_truncated\":0}\n",
                   FLIGHT_RECORDER_SCHEMA_VERSION,
                   snapshot_ns,
                   g_writer_stats.records_drained,
                   g_writer_stats.lines_written,
                   g_writer_stats.bytes_written,
                   g_total_written,
                   g_writer_stats.write_calls,
                   g_write_batch_used,
                   g_writer_stats.drain_passes,
                   g_writer_stats.drain_wall_ns,
                   g_writer_stats.drain_clock_errors,
                   FLIGHT_RECORDER_BATCH_BYTES,
                   FLIGHT_RECORDER_RING_QUANTUM,
                   FLIGHT_RECORDER_CRITICAL_RESERVE)) {
    recorder_fail("writer health formatting failed", 0);
    return false;
  }
  return writer_ensure_space(length) && writer_write_raw(line, length) && writer_flush_batch();
}

static bool writer_emit_sampling_health(void)
{
  const uint64_t assigned = atomic_load_explicit(&g_next_ring, memory_order_relaxed);
  const unsigned int rings = assigned > FLIGHT_RECORDER_MAX_THREAD_RINGS ? FLIGHT_RECORDER_MAX_THREAD_RINGS : assigned;
  for (unsigned int ring_id = 0; ring_id < rings; ++ring_id) {
    for (unsigned int group = 0; group < FLIGHT_RECORDER_SAMPLE_GROUPS; ++group) {
      const recorder_sampling_t *sampling = &g_rings[ring_id].sampling[group];
      const uint64_t retained = atomic_load_explicit(&sampling->retained, memory_order_relaxed);
      const uint64_t reduced = atomic_load_explicit(&sampling->reduced, memory_order_relaxed);
      /* Derive the live evaluated count from published decisions, avoiding a
       * torn three-counter identity. At quiescence all three counters agree. */
      const uint64_t evaluated = retained + reduced;
      if (evaluated == 0)
        continue;
      int64_t snapshot_ns;
      time_to_ns(CLOCK_MONOTONIC, &snapshot_ns);
      char line[FLIGHT_RECORDER_LINE_BYTES];
      size_t length;
      if (!format_line(line,
                       sizeof(line),
                       &length,
                       "{\"schema\":\"oai.flight_recorder\",\"version\":%u,\"kind\":\"sampling_health\","
                       "\"snapshot_mono_ns\":%" PRId64 ",\"ring\":%u,\"group\":%u,\"evaluated\":%" PRIu64 ",\"retained\":%" PRIu64
                       ",\"reduced\":%" PRIu64 ",\"selection_period\":%u,\"detail_mode\":\"%s\","
                       "\"payload_truncated\":0}\n",
                       FLIGHT_RECORDER_SCHEMA_VERSION,
                       snapshot_ns,
                       ring_id,
                       group + 1U,
                       evaluated,
                       retained,
                       reduced,
                       FLIGHT_RECORDER_SAMPLE_PERIOD,
                       recorder_detail_mode())) {
        recorder_fail("sampling health formatting failed", 0);
        return false;
      }
      if (!writer_ensure_space(length) || !writer_write_raw(line, length))
        return false;
    }
  }
  return true;
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
  int64_t snapshot_ns;
  time_to_ns(CLOCK_MONOTONIC, &snapshot_ns);
  return writer_write_raw(line, length) && writer_emit_ring_health(snapshot_ns) && writer_emit_sampling_health()
         && writer_emit_writer_health();
}

static bool writer_write_footer(void)
{
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) == recorder_failed || !writer_flush_batch())
    return false;
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
  return writer_write_raw(line, length) && writer_flush_batch();
}

static bool writer_drain_once(void)
{
  int64_t start_ns;
  time_to_ns(CLOCK_MONOTONIC, &start_ns);
  ++g_writer_stats.drain_passes;
  bool drained = false;
  const unsigned int start_ring = g_next_drain_ring;
  g_next_drain_ring = (start_ring + 1U) % FLIGHT_RECORDER_MAX_THREAD_RINGS;
  for (unsigned int offset = 0; offset < FLIGHT_RECORDER_MAX_THREAD_RINGS; ++offset) {
    const unsigned int ring_id = (start_ring + offset) % FLIGHT_RECORDER_MAX_THREAD_RINGS;
    flight_record_t record;
    for (unsigned int record_count = 0; record_count < FLIGHT_RECORDER_RING_QUANTUM && ring_get(&g_rings[ring_id], &record);
         ++record_count) {
      drained = true;
      ++g_writer_stats.records_drained;
      if (!writer_write_record(ring_id, &record))
        goto failed;
    }
  }
  int64_t now_ns;
  time_to_ns(CLOCK_MONOTONIC, &now_ns);
  if (!writer_flush_async_due(now_ns, false) || !writer_flush_batch())
    goto failed;
  goto finished;
failed:
  drained = false;
finished:
  time_to_ns(CLOCK_MONOTONIC, &now_ns);
  if (start_ns >= 0 && now_ns >= start_ns)
    g_writer_stats.drain_wall_ns += (uint64_t)(now_ns - start_ns);
  else
    ++g_writer_stats.drain_clock_errors;
  return drained;
}

static void writer_close_files(void)
{
  // Preserve admitted records after a later budget/space failure, too.
  writer_flush_batch();
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
      if (!writer_has_space() || !writer_emit_clock_correlation(true) || !writer_emit_health())
        break;
      last_clock_ns = now_ns;
    }
    const bool drained = writer_drain_once();
    if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_running)
      break;

    if (!drained) {
      const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
      nanosleep(&pause, NULL);
    }
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
        /* No enrolled producer can publish after zero is observed. Drain all
         * remaining quanta, including publication after the preceding pass. */
        while (writer_drain_once() && atomic_load_explicit(&g_state, memory_order_seq_cst) == recorder_stopping) {
        }
        break;
      }
      if (!drained) {
        const struct timespec pause = {.tv_sec = 0, .tv_nsec = FLIGHT_RECORDER_SLEEP_NS};
        nanosleep(&pause, NULL);
      }
    }
    if (atomic_load_explicit(&g_state, memory_order_seq_cst) == recorder_stopping && writer_flush_async_due(0, true)
        && writer_emit_health())
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

  const char *detail = getenv("OAI_FLIGHT_RECORDER_DETAIL");
  g_detail_full = false;
  if (detail != NULL) {
    if (strcmp(detail, "full") == 0)
      g_detail_full = true;
    else if (strcmp(detail, "balanced") != 0) {
      recorder_stderr("invalid OAI_FLIGHT_RECORDER_DETAIL", EINVAL);
      return false;
    }
  }

  const char *max_bytes = getenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  uint64_t total_limit = 0;
  if (max_bytes != NULL && max_bytes[0] != '\0') {
    char *end = NULL;
    errno = 0;
    const unsigned long long parsed = strtoull(max_bytes, &end, 10);
    if (errno != 0 || end == max_bytes || *end != '\0' || (parsed != 0 && parsed < FLIGHT_RECORDER_MIN_TOTAL_BYTES)
        || parsed > INT64_MAX) {
      recorder_stderr("invalid OAI_FLIGHT_RECORDER_MAX_BYTES", EINVAL);
      return false;
    }
    total_limit = (uint64_t)parsed;
  }
  g_total_limit = total_limit;
  g_total_written = 0;
  g_next_file_number = FLIGHT_RECORDER_MAX_FILES;
  g_file_limit = total_limit && total_limit < FLIGHT_RECORDER_MAX_FILE_BYTES ? total_limit : FLIGHT_RECORDER_MAX_FILE_BYTES;
#ifdef FLIGHT_RECORDER_TESTING
  if (g_test_file_limit)
    g_file_limit = g_test_file_limit;
#endif
  g_min_free_bytes = FLIGHT_RECORDER_DEFAULT_MIN_FREE_BYTES;
  const char *reserve = getenv("OAI_FLIGHT_RECORDER_MIN_FREE_BYTES");
  if (reserve && *reserve) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(reserve, &end, 10);
    if (errno || end == reserve || *end || parsed > INT64_MAX) {
      recorder_stderr("invalid OAI_FLIGHT_RECORDER_MIN_FREE_BYTES", EINVAL);
      return false;
    }
    g_min_free_bytes = parsed;
  }

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
  g_output_uid = directory_status.st_uid;
  g_output_gid = directory_status.st_gid;

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
  if (!set_output_permissions(first_fd)) {
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
  memset(g_async_repeats, 0, sizeof(g_async_repeats));
  memset(g_write_batch, 0, sizeof(g_write_batch));
  memset(&g_writer_stats, 0, sizeof(g_writer_stats));
  g_write_batch_used = 0;
  g_write_failed = false;
  g_next_drain_ring = 0;
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

static bool producer_enter(void)
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
  atomic_fetch_add_explicit(&g_active_emits, 1, memory_order_seq_cst);
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_running) {
    atomic_fetch_sub_explicit(&g_active_emits, 1, memory_order_seq_cst);
    return false;
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
  return true;
}

static int producer_ring(void)
{
  if (tls_ring == -1) {
    const uint64_t assigned = atomic_fetch_add_explicit(&g_next_ring, 1, memory_order_relaxed);
    if (assigned >= FLIGHT_RECORDER_MAX_THREAD_RINGS) {
      tls_ring = -2;
      atomic_fetch_add_explicit(&g_no_slot_threads, 1, memory_order_relaxed);
    } else {
      tls_ring = (int)assigned;
    }
  }
  return tls_ring;
}

static bool sampling_key_selected(uint64_t key)
{
  // Fixed SplitMix64 finalizer with its input increment, independent of group.
  key += UINT64_C(0x9e3779b97f4a7c15);
  key = (key ^ (key >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  key = (key ^ (key >> 27)) * UINT64_C(0x94d049bb133111eb);
  key ^= key >> 31;
  return (key & (FLIGHT_RECORDER_SAMPLE_PERIOD - 1U)) == 0;
}

static bool sample_ring_group(flight_ring_t *ring, unsigned int group, uint64_t key, bool force)
{
  recorder_sampling_t *sampling = &ring->sampling[group - 1U];
  const uint64_t evaluated = atomic_load_explicit(&sampling->evaluated, memory_order_relaxed);
  const bool retained = force || g_detail_full || evaluated < FLIGHT_RECORDER_SAMPLE_WARMUP || sampling_key_selected(key);
  atomic_store_explicit(&sampling->evaluated, evaluated + 1, memory_order_relaxed);
  atomic_uint_fast64_t *counter = retained ? &sampling->retained : &sampling->reduced;
  atomic_store_explicit(counter, atomic_load_explicit(counter, memory_order_relaxed) + 1, memory_order_relaxed);
  return retained;
}

bool flight_recorder_sample(unsigned int group, uint64_t key, bool force)
{
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_running)
    return false;
  if (group == 0 || group > FLIGHT_RECORDER_SAMPLE_GROUPS)
    return true;
  if (!producer_enter())
    return false;

  const int ring = producer_ring();
  bool retained = true;
  if (ring >= 0)
    retained = sample_ring_group(&g_rings[ring], group, key, force);
  // No-slot decisions retain: actual emits still own per-record loss accounting.
  atomic_fetch_sub_explicit(&g_active_emits, 1, memory_order_seq_cst);
  return retained;
}

static bool power_signature_unknown(unsigned int group, int64_t instance, int64_t a, int64_t b, int64_t c, int64_t d)
{
  if (instance < 0 || a == INT64_MIN || b == INT64_MIN || c == INT64_MIN || d == INT64_MIN)
    return true;
  switch (group) {
    case FLIGHT_SAMPLE_UE_TX_POWER_REQUEST:
      return a == INT16_MIN || a == INT_MIN || b <= 0 || b > INT16_MAX;
    case FLIGHT_SAMPLE_UE_TX_CONTROL:
      return a < 0;
    case FLIGHT_SAMPLE_UE_TX_RELATIVE_POWER:
      return b > c || d < b || d > c;
    default:
      return true;
  }
}

bool flight_recorder_sample_power(unsigned int group,
                                  unsigned int channel,
                                  int64_t instance,
                                  int64_t a,
                                  int64_t b,
                                  int64_t c,
                                  int64_t d)
{
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_running)
    return false;
  if (group < FLIGHT_SAMPLE_UE_TX_POWER_REQUEST || group > FLIGHT_SAMPLE_UE_TX_RELATIVE_POWER
      || channel < FLIGHT_UE_TX_CHANNEL_PRACH || channel > FLIGHT_UE_TX_CHANNEL_SRS)
    return true;
  if (!producer_enter())
    return false;

  const int ring_id = producer_ring();
  bool retained = true;
  if (ring_id >= 0) {
    flight_ring_t *ring = &g_rings[ring_id];
    recorder_power_sampling_t *state = &ring->power_sampling[group - FLIGHT_SAMPLE_UE_TX_POWER_REQUEST][channel - 1U];
    const bool changed = !state->valid || state->instance != instance || state->values[0] != a || state->values[1] != b
                         || state->values[2] != c || state->values[3] != d;
    const bool force = changed || channel == FLIGHT_UE_TX_CHANNEL_PRACH || power_signature_unknown(group, instance, a, b, c, d);
    state->valid = true;
    state->instance = instance;
    state->values[0] = a;
    state->values[1] = b;
    state->values[2] = c;
    state->values[3] = d;
    const uint64_t key = state->opportunities++;
    retained = sample_ring_group(ring, group, key, force);
  }
  atomic_fetch_sub_explicit(&g_active_emits, 1, memory_order_seq_cst);
  return retained;
}

void flight_recorder_emit(uint32_t event, int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f)
{
  if (atomic_load_explicit(&g_state, memory_order_seq_cst) != recorder_running || !producer_enter())
    return;

  if (producer_ring() < 0) {
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

bool flight_recorder_sample(unsigned int group, uint64_t key, bool force)
{
  (void)group;
  (void)key;
  (void)force;
  return false;
}

bool flight_recorder_sample_power(unsigned int group,
                                  unsigned int channel,
                                  int64_t instance,
                                  int64_t a,
                                  int64_t b,
                                  int64_t c,
                                  int64_t d)
{
  (void)group;
  (void)channel;
  (void)instance;
  (void)a;
  (void)b;
  (void)c;
  (void)d;
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
