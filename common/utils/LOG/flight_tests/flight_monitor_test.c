/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "flight_monitor.h"

#include <errno.h>
#include <poll.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define TEST_TIMEOUT_MS 3000

static int failures;

#define CHECK(condition, format, ...)                                            \
  do {                                                                           \
    if (!(condition)) {                                                          \
      fprintf(stderr, "%s:%d: " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
      failures++;                                                                \
    }                                                                            \
  } while (0)

static int64_t monotonic_milliseconds(void)
{
  struct timespec now = {0};
  CHECK(clock_gettime(CLOCK_MONOTONIC, &now) == 0, "CLOCK_MONOTONIC unavailable");
  return (int64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static void set_monitor_environment(int monitor_fd)
{
  char fd_text[32];
  char parent_text[32];
  snprintf(fd_text, sizeof(fd_text), "%d", monitor_fd);
  snprintf(parent_text, sizeof(parent_text), "%ld", (long)getppid());
  CHECK(setenv("_OAI_FLIGHT_MONITOR_FD", fd_text, 1) == 0, "set monitor descriptor failed");
  CHECK(setenv("_OAI_FLIGHT_CAPTURE_PARENT", parent_text, 1) == 0, "set capture parent failed");
}

static void clear_monitor_environment(void)
{
  CHECK(unsetenv("_OAI_FLIGHT_MONITOR_FD") == 0, "unset monitor descriptor failed");
  CHECK(unsetenv("_OAI_FLIGHT_CAPTURE_PARENT") == 0, "unset capture parent failed");
}

static bool receive_message(int fd, char *message, size_t message_size, int timeout_ms)
{
  struct pollfd input = {.fd = fd, .events = POLLIN};
  if (poll(&input, 1, timeout_ms) <= 0)
    return false;

  const ssize_t length = recv(fd, message, message_size - 1, 0);
  if (length < 0)
    return false;
  message[length] = '\0';
  return true;
}

static bool wait_for_message_with(int fd, const char *first, const char *second, int timeout_ms, char *message, size_t message_size)
{
  const int64_t deadline = monotonic_milliseconds() + timeout_ms;
  while (monotonic_milliseconds() < deadline) {
    const int64_t remaining = deadline - monotonic_milliseconds();
    if (!receive_message(fd, message, message_size, remaining > 200 ? 200 : (int)remaining))
      continue;
    if (strstr(message, first) != NULL && (second == NULL || strstr(message, second) != NULL))
      return true;
  }
  return false;
}

static void test_reject_packing(void)
{
  CHECK(flight_monitor_pack_nas_reject(0x12, 0x34, 0x56, 0x42, UINT32_C(0x789abcde)) == UINT64_C(0x12345642789abcde),
        "NAS rejection packing does not preserve the raw T3502 timer");
}

static void test_disabled_without_descriptor(void)
{
  clear_monitor_environment();
  flight_monitor_init();
  CHECK(!flight_monitor_enabled(), "monitor enabled without inherited descriptor");
  CHECK(flight_monitor_test_worker_policy() == -1, "monitor worker started without inherited descriptor");
  flight_monitor_add(FLIGHT_MONITOR_RX_SAMPLES, 1);
  flight_monitor_set(FLIGHT_MONITOR_RRC_STATE, 3);
  flight_monitor_shutdown();
}

static void test_emitted_counters_and_final_snapshot(void)
{
  int sockets[2] = {-1, -1};
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sockets) == 0, "socketpair failed: %s", strerror(errno));
  if (sockets[0] < 0 || sockets[1] < 0)
    return;

  set_monitor_environment(sockets[0]);
  flight_monitor_init();
  CHECK(flight_monitor_enabled(), "monitor did not enable with valid inherited descriptor");
  CHECK(flight_monitor_test_wakeup_clock() == CLOCK_MONOTONIC, "monitor wakeup condition is not monotonic");

  char message[1024];
  char pid_text[64];
  snprintf(pid_text, sizeof(pid_text), "\"pid\":%ld", (long)getpid());
  CHECK(wait_for_message_with(sockets[1], "\"kind\":\"native_progress\"", pid_text, TEST_TIMEOUT_MS, message, sizeof(message)),
        "initial native progress snapshot missing");
  CHECK(flight_monitor_test_worker_policy() == SCHED_OTHER, "monitor worker policy is not SCHED_OTHER");

  flight_monitor_add(FLIGHT_MONITOR_RX_SAMPLES, 17);
  flight_monitor_set(FLIGHT_MONITOR_RRC_STATE, 7);
  CHECK(wait_for_message_with(sockets[1], "\"rx_samples\":17", "\"rrc_state\":7", TEST_TIMEOUT_MS, message, sizeof(message)),
        "counter snapshot missing expected atomic values");

  flight_monitor_shutdown();
  CHECK(wait_for_message_with(sockets[1], "\"rx_samples\":17", "\"rrc_state\":7", TEST_TIMEOUT_MS, message, sizeof(message)),
        "final native progress snapshot missing");
  close(sockets[1]);
  clear_monitor_environment();
}

static void test_full_catalog_fits_snapshot(void)
{
  int sockets[2] = {-1, -1};
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sockets) == 0, "socketpair failed: %s", strerror(errno));
  if (sockets[0] < 0 || sockets[1] < 0)
    return;

  set_monitor_environment(sockets[0]);
  flight_monitor_init();
  char message[1024];
  CHECK(receive_message(sockets[1], message, sizeof(message), TEST_TIMEOUT_MS), "initial catalog snapshot missing");
  for (unsigned int field = 0; field < FLIGHT_MONITOR_FIELD_COUNT; ++field) {
    const uint64_t value = field == FLIGHT_MONITOR_DRB_CONTEXT_ACTIVE ? UINT64_MAX : field + 1;
    flight_monitor_set((flight_monitor_field_t)field, value);
  }

  CHECK(wait_for_message_with(sockets[1],
                              "\"drb_context_active\":18446744073709551615",
                              "\"ue_tx_completed\":14",
                              TEST_TIMEOUT_MS,
                              message,
                              sizeof(message)),
        "full native progress catalog snapshot missing");
  CHECK(strstr(message, "\"ue_slot_inputs\":12") != NULL, "slot-input counter missing from full catalog snapshot");
  CHECK(strstr(message, "\"ue_dl_completed\":13") != NULL, "DL-completion counter missing from full catalog snapshot");
  CHECK(strstr(message, "\"rx_samples\":1") != NULL, "first catalog field missing from full catalog snapshot");
  CHECK(strlen(message) < sizeof(message) - 1, "full catalog snapshot reached the fixed datagram limit");

  flight_monitor_shutdown();
  close(sockets[1]);
  clear_monitor_environment();
}

static void fill_peer_receive_queue(int sender_fd)
{
  char payload[512] = {0};
  while (send(sender_fd, payload, sizeof(payload), MSG_DONTWAIT | MSG_NOSIGNAL) >= 0)
    ;
  CHECK(errno == EAGAIN || errno == EWOULDBLOCK, "datagram queue fill failed with errno=%d", errno);
}

static void test_full_socket_does_not_block_shutdown(void)
{
  int sockets[2] = {-1, -1};
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sockets) == 0, "socketpair failed: %s", strerror(errno));
  if (sockets[0] < 0 || sockets[1] < 0)
    return;

  fill_peer_receive_queue(sockets[0]);
  set_monitor_environment(sockets[0]);
  flight_monitor_init();
  const int64_t drop_deadline = monotonic_milliseconds() + TEST_TIMEOUT_MS;
  while (flight_monitor_test_send_drops() == 0 && monotonic_milliseconds() < drop_deadline) {
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = 10000000};
    nanosleep(&pause, NULL);
  }
  CHECK(flight_monitor_test_send_drops() > 0, "full monitor socket did not report a nonblocking send drop");

  const int64_t shutdown_started = monotonic_milliseconds();
  flight_monitor_shutdown();
  CHECK(monotonic_milliseconds() - shutdown_started < 500, "monitor shutdown blocked on a full datagram socket");
  close(sockets[1]);
  clear_monitor_environment();
}

static void test_child_does_not_emit(void)
{
  int sockets[2] = {-1, -1};
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sockets) == 0, "socketpair failed: %s", strerror(errno));
  if (sockets[0] < 0 || sockets[1] < 0)
    return;

  set_monitor_environment(sockets[0]);
  flight_monitor_init();
  char message[1024];
  CHECK(receive_message(sockets[1], message, sizeof(message), TEST_TIMEOUT_MS), "parent initial snapshot missing before fork");

  const pid_t child = fork();
  CHECK(child >= 0, "fork failed: %s", strerror(errno));
  if (child == 0) {
    flight_monitor_shutdown();
    _exit(0);
  }
  if (child < 0) {
    flight_monitor_shutdown();
    close(sockets[1]);
    clear_monitor_environment();
    return;
  }

  int child_status = 0;
  CHECK(waitpid(child, &child_status, 0) == child, "waitpid failed: %s", strerror(errno));
  CHECK(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0, "child monitor shutdown failed");

  flight_monitor_add(FLIGHT_MONITOR_RX_SAMPLES, 1);
  char child_pid_text[64];
  char parent_pid_text[64];
  snprintf(child_pid_text, sizeof(child_pid_text), "\"pid\":%ld", (long)child);
  snprintf(parent_pid_text, sizeof(parent_pid_text), "\"pid\":%ld", (long)getpid());
  bool parent_snapshot_seen = false;
  const int64_t deadline = monotonic_milliseconds() + TEST_TIMEOUT_MS;
  while (monotonic_milliseconds() < deadline) {
    if (!receive_message(sockets[1], message, sizeof(message), 200))
      continue;
    CHECK(strstr(message, child_pid_text) == NULL, "forked child emitted a native progress snapshot: %s", message);
    if (strstr(message, parent_pid_text) != NULL && strstr(message, "\"rx_samples\":1") != NULL)
      parent_snapshot_seen = true;
  }
  CHECK(parent_snapshot_seen, "parent monitor did not continue after child shutdown");

  flight_monitor_shutdown();
  close(sockets[1]);
  clear_monitor_environment();
}

int main(void)
{
  test_reject_packing();
  test_disabled_without_descriptor();
  test_emitted_counters_and_final_snapshot();
  test_full_catalog_fits_snapshot();
  test_full_socket_does_not_block_shutdown();
  test_child_does_not_emit();
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
