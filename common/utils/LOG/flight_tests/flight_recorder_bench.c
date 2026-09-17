/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*
 * A small native observer-effect measurement. It intentionally measures only
 * flight_recorder_emit() with fixed numeric values; it does not model an OAI
 * call site's argument computation.
 */

#define _GNU_SOURCE

#include "flight_recorder.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void flight_recorder_test_set_writer_paused(bool paused);

static uint64_t monotonic_ns(void)
{
  struct timespec timestamp;
  if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0)
    return 0;
  return (uint64_t)timestamp.tv_sec * UINT64_C(1000000000) + (uint64_t)timestamp.tv_nsec;
}

static uint64_t measure_emits(uint64_t iterations)
{
  const uint64_t start = monotonic_ns();
  for (uint64_t index = 0; index < iterations; ++index)
    flight_recorder_emit(FLIGHT_EVENT_GNB_SLOT, 1, 2, 3, 4, 5, 6);
  const uint64_t finish = monotonic_ns();
  return finish - start;
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s <existing-output-directory>\n", argv[0]);
    return 2;
  }

  const uint64_t disabled_iterations = UINT64_C(5000000);
  const uint64_t enabled_iterations = FLIGHT_RECORDER_RING_RECORDS;

  unsetenv("OAI_FLIGHT_RECORDER_DIR");
  unsetenv("OAI_FLIGHT_RECORDER_MAX_BYTES");
  flight_recorder_init();
  const uint64_t disabled_ns = measure_emits(disabled_iterations);
  flight_recorder_shutdown();

  if (setenv("OAI_FLIGHT_RECORDER_DIR", argv[1], 1) != 0) {
    perror("setenv OAI_FLIGHT_RECORDER_DIR");
    return 2;
  }
  flight_recorder_test_set_writer_paused(true);
  flight_recorder_init();
  if (!flight_recorder_enabled()) {
    fprintf(stderr, "enabled recorder did not start\n");
    return 1;
  }
  const uint64_t enabled_ns = measure_emits(enabled_iterations);
  flight_recorder_test_set_writer_paused(false);
  flight_recorder_shutdown();

  printf("{\"disabled_iterations\":%" PRIu64 ",\"disabled_total_ns\":%" PRIu64
         ",\"disabled_ns_per_emit\":%.2f,\"enabled_iterations\":%" PRIu64 ",\"enabled_total_ns\":%" PRIu64
         ",\"enabled_ns_per_emit\":%.2f}\n",
         disabled_iterations,
         disabled_ns,
         (double)disabled_ns / (double)disabled_iterations,
         enabled_iterations,
         enabled_ns,
         (double)enabled_ns / (double)enabled_iterations);
  return 0;
}
