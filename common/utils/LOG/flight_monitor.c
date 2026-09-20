/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "flight_monitor.h"
#include "radio_health.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#if (defined(__x86_64__) || defined(__aarch64__)) && !defined(FLIGHT_MONITOR_FORCE_INERT_TEST)

#define FLIGHT_MONITOR_FIELDS_MAX 32U
#define FLIGHT_MONITOR_MESSAGE_BYTES 1024U
#define FLIGHT_RADIO_MESSAGE_BYTES 8192U
#define FLIGHT_MONITOR_INTERVAL_SECONDS 1L

typedef struct {
  const char *name;
} flight_monitor_field_descriptor_t;

static const flight_monitor_field_descriptor_t g_field_descriptors[FLIGHT_MONITOR_FIELD_COUNT] = {
    [FLIGHT_MONITOR_RX_SAMPLES] = {.name = "rx_samples"},
    [FLIGHT_MONITOR_TX_SAMPLES] = {.name = "tx_samples"},
    [FLIGHT_MONITOR_SEARCH_ATTEMPTS] = {.name = "search_attempts"},
    [FLIGHT_MONITOR_SYNC_SUCCESSES] = {.name = "sync_successes"},
    [FLIGHT_MONITOR_RRC_MESSAGES] = {.name = "rrc_messages"},
    [FLIGHT_MONITOR_NAS_MESSAGES] = {.name = "nas_messages"},
    [FLIGHT_MONITOR_RRC_STATE] = {.name = "rrc_state"},
    [FLIGHT_MONITOR_PDU_ACCEPTS] = {.name = "pdu_accepts"},
    [FLIGHT_MONITOR_PDU_ACTIVE] = {.name = "pdu_active"},
    [FLIGHT_MONITOR_NAS_REJECT] = {.name = "nas_reject"},
    [FLIGHT_MONITOR_RRC_HOLD_UNTIL_NS] = {.name = "rrc_hold_until_ns"},
    [FLIGHT_MONITOR_UE_SLOT_INPUTS] = {.name = "ue_slot_inputs"},
    [FLIGHT_MONITOR_UE_DL_COMPLETED] = {.name = "ue_dl_completed"},
    [FLIGHT_MONITOR_UE_TX_COMPLETED] = {.name = "ue_tx_completed"},
    [FLIGHT_MONITOR_DRB_CONTEXT_ACTIVE] = {.name = "drb_context_active"},
};

_Static_assert(FLIGHT_MONITOR_FIELD_COUNT <= FLIGHT_MONITOR_FIELDS_MAX, "flight monitor field capacity exceeded");

static atomic_bool g_enabled = ATOMIC_VAR_INIT(false);
static atomic_bool g_stop = ATOMIC_VAR_INIT(false);
static atomic_uint_fast64_t g_values[FLIGHT_MONITOR_FIELD_COUNT];
static atomic_bool g_observed[FLIGHT_MONITOR_FIELD_COUNT];
static atomic_uint_fast64_t g_send_drops = ATOMIC_VAR_INIT(0);

static pthread_mutex_t g_lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_wakeup_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_wakeup;
static bool g_wakeup_initialized;
static clockid_t g_wakeup_clock = CLOCK_REALTIME;
static pthread_t g_worker;
static bool g_worker_started;
static bool g_atexit_registered;
static pid_t g_owner_pid;
static int g_monitor_fd = -1;
static uint64_t g_sequence;
/* The monitor owns these; each device has an independent message sequence. */
static uint64_t g_radio_sequences[RADIO_HEALTH_MAX_DEVICES];
static uint64_t g_radio_send_drops[RADIO_HEALTH_MAX_DEVICES];

#ifdef FLIGHT_MONITOR_TESTING
static atomic_int g_test_worker_policy = ATOMIC_VAR_INIT(-1);
#endif

static bool parse_nonnegative_int(const char *text, int *value)
{
  if (text == NULL || text[0] == '\0')
    return false;

  errno = 0;
  char *end = NULL;
  const long parsed = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX)
    return false;

  *value = (int)parsed;
  return true;
}

static bool capture_parent_matches(void)
{
  int expected_parent = -1;
  return parse_nonnegative_int(getenv("_OAI_FLIGHT_CAPTURE_PARENT"), &expected_parent) && expected_parent == (int)getppid();
}

static bool monitor_fd_is_connected_unix_datagram(int fd)
{
  if (fcntl(fd, F_GETFD) == -1)
    return false;

  int socket_type = 0;
  socklen_t socket_type_length = sizeof(socket_type);
  if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &socket_type, &socket_type_length) != 0 || socket_type != SOCK_DGRAM)
    return false;

  struct sockaddr_un peer = {0};
  socklen_t peer_length = sizeof(peer);
  if (getpeername(fd, (struct sockaddr *)&peer, &peer_length) != 0)
    return false;

  return peer.sun_family == AF_UNIX;
}

static bool monitor_producer_atomics_are_lock_free(void)
{
  return atomic_is_lock_free(&g_enabled) && atomic_is_lock_free(&g_values[0]) && atomic_is_lock_free(&g_observed[0]);
}

static bool initialize_wakeup_condition(void)
{
  if (g_wakeup_initialized)
    return true;

  pthread_condattr_t attributes;
  if (pthread_condattr_init(&attributes) != 0)
    return false;

  const bool valid = pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC) == 0 && pthread_cond_init(&g_wakeup, &attributes) == 0;
  pthread_condattr_destroy(&attributes);
  if (!valid)
    return false;

  g_wakeup_clock = CLOCK_MONOTONIC;
  g_wakeup_initialized = true;
  return true;
}

static bool append_value(char *message, size_t message_size, size_t *used, bool *first, const char *name, uint64_t value)
{
  const int written = snprintf(message + *used, message_size - *used, "%s\"%s\":%" PRIu64, *first ? "" : ",", name, value);
  if (written < 0 || (size_t)written >= message_size - *used)
    return false;

  *used += written;
  *first = false;
  return true;
}

static void emit_snapshot(void)
{
  if (getpid() != g_owner_pid || g_monitor_fd < 0)
    return;

  struct timespec monotonic = {0};
  if (clock_gettime(CLOCK_MONOTONIC, &monotonic) != 0)
    return;

  char message[FLIGHT_MONITOR_MESSAGE_BYTES];
  int written = snprintf(message,
                         sizeof(message),
                         "{\"kind\":\"native_progress\",\"schema_version\":%u,\"pid\":%ld,\"sequence\":%" PRIu64
                         ",\"mono_ns\":%" PRIu64 ",\"send_drops\":%" PRIu64 ",\"values\":{",
                         FLIGHT_MONITOR_SCHEMA_VERSION,
                         (long)g_owner_pid,
                         ++g_sequence,
                         (uint64_t)monotonic.tv_sec * UINT64_C(1000000000) + (uint64_t)monotonic.tv_nsec,
                         (uint64_t)atomic_load_explicit(&g_send_drops, memory_order_relaxed));
  if (written < 0 || (size_t)written >= sizeof(message))
    return;

  size_t used = (size_t)written;
  bool first = true;
  for (unsigned int index = 0; index < FLIGHT_MONITOR_FIELD_COUNT; ++index) {
    if (!atomic_load_explicit(&g_observed[index], memory_order_relaxed))
      continue;
    if (!append_value(message,
                      sizeof(message),
                      &used,
                      &first,
                      g_field_descriptors[index].name,
                      (uint64_t)atomic_load_explicit(&g_values[index], memory_order_relaxed)))
      return;
  }

  written = snprintf(message + used, sizeof(message) - used, "}}");
  if (written < 0 || (size_t)written >= sizeof(message) - used)
    return;
  used += (size_t)written;

  const ssize_t sent = send(g_monitor_fd, message, used, MSG_DONTWAIT | MSG_NOSIGNAL);
  if (sent != (ssize_t)used)
    atomic_fetch_add_explicit(&g_send_drops, 1, memory_order_relaxed);
}

static void emit_radio_snapshots(void)
{
  if (getpid() != g_owner_pid || g_monitor_fd < 0)
    return;

  for (uint32_t slot = 0; slot < RADIO_HEALTH_MAX_DEVICES; ++slot) {
    radio_health_snapshot_t snapshot;
    struct timespec monotonic;
    if (!radio_health_snapshot(slot, &snapshot) || clock_gettime(CLOCK_MONOTONIC, &monotonic) != 0)
      continue;

    char message[FLIGHT_RADIO_MESSAGE_BYTES];
    int written =
        snprintf(message,
                 sizeof(message),
                 "{\"kind\":\"radio_health\",\"schema_version\":1,\"pid\":%ld,\"sequence\":%" PRIu64 ",\"mono_ns\":%" PRIu64
                 ",\"send_drops\":%" PRIu64 ",\"device_id\":%u,\"backend\":\"%s\",\"device_type\":%u,\"active\":%s,\"supported\":[",
                 (long)g_owner_pid,
                 ++g_radio_sequences[slot],
                 (uint64_t)monotonic.tv_sec * UINT64_C(1000000000) + (uint64_t)monotonic.tv_nsec,
                 g_radio_send_drops[slot],
                 snapshot.device_id,
                 radio_health_backend_name(snapshot.backend),
                 snapshot.device_type,
                 snapshot.lifecycle == RADIO_HEALTH_LIFECYCLE_ACTIVE ? "true" : "false");
    bool valid = written >= 0 && (size_t)written < sizeof(message);
    size_t used = valid ? (size_t)written : 0;
    bool first = true;
    for (unsigned int i = 0; valid && i < RADIO_HEALTH_METRIC_COUNT; ++i) {
      if (!radio_health_metric_supported(snapshot.capabilities, (radio_health_metric_t)i))
        continue;
      written = snprintf(message + used,
                         sizeof(message) - used,
                         "%s\"%s\"",
                         first ? "" : ",",
                         radio_health_metric_name((radio_health_metric_t)i));
      valid = written >= 0 && (size_t)written < sizeof(message) - used;
      if (valid)
        used += (size_t)written;
      first = false;
    }
    if (valid) {
      written = snprintf(message + used, sizeof(message) - used, "],\"values\":{");
      valid = written >= 0 && (size_t)written < sizeof(message) - used;
      if (valid)
        used += (size_t)written;
    }
    first = true;
    for (unsigned int i = 0; valid && i < RADIO_HEALTH_METRIC_COUNT; ++i) {
      if (!(snapshot.observed_metrics & RADIO_HEALTH_METRIC_BIT(i)))
        continue;
      valid = append_value(message,
                           sizeof(message),
                           &used,
                           &first,
                           radio_health_metric_name((radio_health_metric_t)i),
                           snapshot.values[i]);
    }
    if (valid) {
      written = snprintf(message + used, sizeof(message) - used, "}}");
      valid = written >= 0 && (size_t)written < sizeof(message) - used;
      if (valid)
        used += (size_t)written;
    }
    if (!valid || send(g_monitor_fd, message, used, MSG_DONTWAIT | MSG_NOSIGNAL) != (ssize_t)used)
      ++g_radio_send_drops[slot];
  }
}

static void wait_for_next_snapshot(void)
{
  struct timespec deadline = {0};
  if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0)
    return;
  deadline.tv_sec += FLIGHT_MONITOR_INTERVAL_SECONDS;

  pthread_mutex_lock(&g_wakeup_lock);
  if (!atomic_load_explicit(&g_stop, memory_order_acquire))
    (void)pthread_cond_timedwait(&g_wakeup, &g_wakeup_lock, &deadline);
  pthread_mutex_unlock(&g_wakeup_lock);
}

static void *flight_monitor_main(void *unused)
{
  (void)unused;

  int policy = -1;
  struct sched_param parameter = {0};
  if (pthread_getschedparam(pthread_self(), &policy, &parameter) != 0 || policy != SCHED_OTHER) {
    atomic_store_explicit(&g_enabled, false, memory_order_release);
    radio_health_set_enabled(false);
    return NULL;
  }
#ifdef FLIGHT_MONITOR_TESTING
  atomic_store_explicit(&g_test_worker_policy, policy, memory_order_release);
#endif
  (void)pthread_setname_np(pthread_self(), "flight-monitor");

  emit_snapshot();
  emit_radio_snapshots();
  while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
    wait_for_next_snapshot();
    if (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
      emit_snapshot();
      emit_radio_snapshots();
    }
  }
  emit_snapshot();
  emit_radio_snapshots();
  return NULL;
}

static bool start_monitor_thread(void)
{
  pthread_attr_t attributes;
  int result = pthread_attr_init(&attributes);
  if (result != 0)
    return false;

  struct sched_param parameter = {0};
  bool valid = pthread_attr_setinheritsched(&attributes, PTHREAD_EXPLICIT_SCHED) == 0
               && pthread_attr_setschedpolicy(&attributes, SCHED_OTHER) == 0
               && pthread_attr_setschedparam(&attributes, &parameter) == 0;
  if (valid)
    result = pthread_create(&g_worker, &attributes, flight_monitor_main, NULL);
  pthread_attr_destroy(&attributes);
  if (!valid || result != 0)
    return false;

  g_worker_started = true;
  return true;
}

void flight_monitor_init(void)
{
  const pid_t current_pid = getpid();
  if (g_owner_pid != 0 && g_owner_pid != current_pid)
    return;

  pthread_mutex_lock(&g_lifecycle_lock);
  if (g_worker_started || atomic_load_explicit(&g_enabled, memory_order_acquire)) {
    pthread_mutex_unlock(&g_lifecycle_lock);
    return;
  }

  int monitor_fd = -1;
  if (!capture_parent_matches() || !parse_nonnegative_int(getenv("_OAI_FLIGHT_MONITOR_FD"), &monitor_fd)
      || !monitor_fd_is_connected_unix_datagram(monitor_fd) || !monitor_producer_atomics_are_lock_free()
      || !initialize_wakeup_condition()) {
    pthread_mutex_unlock(&g_lifecycle_lock);
    return;
  }

  for (unsigned int index = 0; index < FLIGHT_MONITOR_FIELD_COUNT; ++index) {
    atomic_store_explicit(&g_values[index], 0, memory_order_relaxed);
    atomic_store_explicit(&g_observed[index], false, memory_order_relaxed);
  }
  atomic_store_explicit(&g_send_drops, 0, memory_order_relaxed);
  atomic_store_explicit(&g_stop, false, memory_order_release);
  g_sequence = 0;
  for (unsigned int i = 0; i < RADIO_HEALTH_MAX_DEVICES; ++i) {
    g_radio_sequences[i] = 0;
    g_radio_send_drops[i] = 0;
  }
  g_owner_pid = current_pid;
  g_monitor_fd = monitor_fd;
#ifdef FLIGHT_MONITOR_TESTING
  atomic_store_explicit(&g_test_worker_policy, -1, memory_order_release);
#endif
  atomic_store_explicit(&g_enabled, true, memory_order_release);
  radio_health_set_enabled(true);

  if (!start_monitor_thread()) {
    radio_health_set_enabled(false);
    atomic_store_explicit(&g_enabled, false, memory_order_release);
    g_monitor_fd = -1;
    g_owner_pid = 0;
    pthread_mutex_unlock(&g_lifecycle_lock);
    return;
  }

  if (!g_atexit_registered && atexit(flight_monitor_shutdown) == 0)
    g_atexit_registered = true;
  pthread_mutex_unlock(&g_lifecycle_lock);
}

void flight_monitor_shutdown(void)
{
  const pid_t current_pid = getpid();
  if (g_owner_pid == 0 || g_owner_pid != current_pid) {
    atomic_store_explicit(&g_enabled, false, memory_order_release);
    return;
  }

  pthread_mutex_lock(&g_lifecycle_lock);
  radio_health_set_enabled(false);
  atomic_store_explicit(&g_enabled, false, memory_order_release);
  if (g_worker_started) {
    atomic_store_explicit(&g_stop, true, memory_order_release);
    pthread_mutex_lock(&g_wakeup_lock);
    pthread_cond_broadcast(&g_wakeup);
    pthread_mutex_unlock(&g_wakeup_lock);
    pthread_join(g_worker, NULL);
    g_worker_started = false;
  }
  if (g_monitor_fd >= 0) {
    close(g_monitor_fd);
    g_monitor_fd = -1;
  }
  g_owner_pid = 0;
  pthread_mutex_unlock(&g_lifecycle_lock);
}

bool flight_monitor_enabled(void)
{
  return atomic_load_explicit(&g_enabled, memory_order_relaxed);
}

void flight_monitor_add(flight_monitor_field_t field, uint64_t amount)
{
  if (!atomic_load_explicit(&g_enabled, memory_order_relaxed))
    return;
  if ((unsigned int)field >= FLIGHT_MONITOR_FIELD_COUNT)
    return;

  atomic_fetch_add_explicit(&g_values[field], amount, memory_order_relaxed);
  atomic_store_explicit(&g_observed[field], true, memory_order_relaxed);
}

void flight_monitor_set(flight_monitor_field_t field, uint64_t value)
{
  if (!atomic_load_explicit(&g_enabled, memory_order_relaxed))
    return;
  if ((unsigned int)field >= FLIGHT_MONITOR_FIELD_COUNT)
    return;

  atomic_store_explicit(&g_values[field], value, memory_order_relaxed);
  atomic_store_explicit(&g_observed[field], true, memory_order_relaxed);
}

#ifdef FLIGHT_MONITOR_TESTING
uint64_t flight_monitor_test_send_drops(void)
{
  return atomic_load_explicit(&g_send_drops, memory_order_relaxed);
}

int flight_monitor_test_worker_policy(void)
{
  return atomic_load_explicit(&g_test_worker_policy, memory_order_acquire);
}

int flight_monitor_test_wakeup_clock(void)
{
  return g_wakeup_clock;
}
#endif

#else
/* Match the recorder's inert fallback on targets without its lock-free ABI. */
void flight_monitor_init(void)
{
}
void flight_monitor_shutdown(void)
{
}
bool flight_monitor_enabled(void)
{
  return false;
}
void flight_monitor_add(flight_monitor_field_t field, uint64_t amount)
{
  (void)field;
  (void)amount;
}
void flight_monitor_set(flight_monitor_field_t field, uint64_t value)
{
  (void)field;
  (void)value;
}
#endif
