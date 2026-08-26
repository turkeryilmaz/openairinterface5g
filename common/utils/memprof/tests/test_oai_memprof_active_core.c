/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _POSIX_C_SOURCE 200809L

#include "common/utils/memprof/oai_memprof_active_core.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      exit(EXIT_FAILURE);                                                             \
    }                                                                                 \
  } while (0)

_Static_assert(OAI_MEMPROF_CORE_API_SLOT_COUNT == 32, "the authenticated API counter population must be 32");
_Static_assert(sizeof(((oai_memprof_core_thread_info_t *)0)->api_attempts) / sizeof(uint64_t) == OAI_MEMPROF_CORE_API_SLOT_COUNT,
               "the thread projection must expose every API counter slot");

typedef struct event_sink_s {
  oai_memprof_event_v1_t events[32];
  size_t count;
  size_t fail_at;
} event_sink_t;

static bool collect_event(void *context, const oai_memprof_event_v1_t *event)
{
  event_sink_t *sink = context;
  if (sink->count == sink->fail_at)
    return false;
  CHECK(sink->count < sizeof(sink->events) / sizeof(sink->events[0]));
  sink->events[sink->count++] = *event;
  return true;
}

static oai_memprof_core_t *new_core(uint8_t mode, uint32_t threads, uint32_t records)
{
  const oai_memprof_core_config_t config = {
      .process_generation = UINT64_C(0x12345),
      .max_threads = threads,
      .ring_records = records,
      .mode_id = mode,
  };
  oai_memprof_core_t *core = NULL;
  CHECK(oai_memprof_core_bootstrap(&config, &core) == OAI_MEMPROF_CORE_OK);
  CHECK(core != NULL);
  return core;
}

static oai_memprof_core_t *new_sampled_core(uint64_t generation,
                                            uint64_t seed,
                                            uint64_t threshold,
                                            uint64_t entries,
                                            uint32_t probes,
                                            uint32_t threads,
                                            uint32_t records)
{
  const oai_memprof_core_config_t config = {
      .process_generation = generation,
      .table_entries = entries,
      .sample_seed = seed,
      .sample_threshold = threshold,
      .max_threads = threads,
      .ring_records = records,
      .table_probes = probes,
      .mode_id = OAI_MEMPROF_CORE_SAMPLED,
  };
  oai_memprof_core_t *core = NULL;
  CHECK(oai_memprof_core_bootstrap(&config, &core) == OAI_MEMPROF_CORE_OK);
  CHECK(core != NULL);
  return core;
}

static oai_memprof_core_payload_t malloc_payload(uint64_t address, uint64_t bytes)
{
  return (oai_memprof_core_payload_t){
      .address_after = address,
      .arg0 = bytes,
      .flags = (UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 11) | (UINT32_C(1) << 24),
      .api_id = 1,
      .event_kind = 1,
  };
}

static void test_configuration_and_inactive_bypass(void)
{
  oai_memprof_core_t *core = NULL;
  const oai_memprof_core_config_t bad[] = {
      {.process_generation = 0, .max_threads = 1, .ring_records = 2, .mode_id = OAI_MEMPROF_CORE_EXACT_EVENTS},
      {.process_generation = 1, .max_threads = 0, .ring_records = 2, .mode_id = OAI_MEMPROF_CORE_EXACT_EVENTS},
      {.process_generation = 1, .max_threads = 1, .ring_records = 3, .mode_id = OAI_MEMPROF_CORE_EXACT_EVENTS},
      {.process_generation = 1, .max_threads = 1, .ring_records = 2, .mode_id = 1},
      {.process_generation = 1,
       .table_entries = 4,
       .sample_threshold = 0,
       .max_threads = 1,
       .ring_records = 2,
       .table_probes = 1,
       .mode_id = OAI_MEMPROF_CORE_SAMPLED},
      {.process_generation = 1,
       .table_entries = 4,
       .sample_threshold = 1,
       .max_threads = 1,
       .ring_records = 2,
       .table_probes = 5,
       .mode_id = OAI_MEMPROF_CORE_SAMPLED},
  };
  for (size_t index = 0; index < sizeof(bad) / sizeof(bad[0]); ++index)
    CHECK(oai_memprof_core_bootstrap(&bad[index], &core) == OAI_MEMPROF_CORE_INVALID_CONFIGURATION);

  core = new_core(OAI_MEMPROF_CORE_EXACT_EVENTS, 2, 4);
  oai_memprof_core_ticket_t ticket = {.admitted = true};
  CHECK(!oai_memprof_core_begin(core, 1, 64, true, &ticket));
  CHECK(!ticket.admitted);
  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_core_snapshot(core, &snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.state == OAI_MEMPROF_CORE_BOOTSTRAP);
  CHECK(snapshot.reservations == 0);
}

static void test_exact_event_lifecycle(void)
{
  oai_memprof_core_t *core = new_core(OAI_MEMPROF_CORE_EXACT_EVENTS, 2, 4);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_INVALID_STATE);

  oai_memprof_core_ticket_t ticket = {.admitted = true};
  CHECK(!oai_memprof_core_begin(core, OAI_MEMPROF_CORE_ADMITTED_API_COUNT + UINT16_C(1), 64, true, &ticket));
  CHECK(!ticket.admitted);
  CHECK(oai_memprof_core_begin(core, 1, 64, true, &ticket));
  oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x1000), 64);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_core_snapshot(core, &snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.ready_threads == 1);
  CHECK(snapshot.admitted_transactions == 1);
  CHECK(snapshot.completed_transactions == 1);
  CHECK(snapshot.emitted_events == 1);
  CHECK(snapshot.requested_bytes == 64);

  event_sink_t active_sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &active_sink) == OAI_MEMPROF_CORE_OK);
  CHECK(active_sink.count == 1);
  CHECK(active_sink.events[0].thread_sequence == 1);
  CHECK(active_sink.events[0].thread_index == 1);
  CHECK(active_sink.events[0].address_after == UINT64_C(0x1000));

  ticket = (oai_memprof_core_ticket_t){0};
  CHECK(oai_memprof_core_begin(core, 1, 32, true, &ticket));
  payload = malloc_payload(UINT64_C(0x1100), 32);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  CHECK(oai_memprof_core_seal(core, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  event_sink_t sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 1);
  CHECK(sink.events[0].thread_sequence == 2);
  CHECK(sink.events[0].thread_index == 1);
  CHECK(sink.events[0].address_after == UINT64_C(0x1100));
  CHECK(sink.events[0].arg0 == 32);
  CHECK(sink.events[0].api_id == 1);
  CHECK(sink.events[0].event_kind == 1);
  CHECK((sink.events[0].flags & (OAI_MEMPROF_CORE_COUNTER_ENTER_VALID | OAI_MEMPROF_CORE_COUNTER_EXIT_VALID))
        == (OAI_MEMPROF_CORE_COUNTER_ENTER_VALID | OAI_MEMPROF_CORE_COUNTER_EXIT_VALID));
  CHECK((sink.events[0].flags & OAI_MEMPROF_CORE_BOUNDARY_STRADDLING) == 0);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
  CHECK(!oai_memprof_core_begin(core, 1, 1, true, &ticket));
  CHECK(oai_memprof_core_snapshot(core, &snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.state == OAI_MEMPROF_CORE_COMPLETE);
  CHECK(snapshot.reservations == 1);
}

static void test_counter_mode_and_recursion(void)
{
  oai_memprof_core_t *core = new_core(OAI_MEMPROF_CORE_COUNTERS, 1, 2);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);
  oai_memprof_core_ticket_t outer = {0};
  oai_memprof_core_ticket_t nested = {0};
  CHECK(oai_memprof_core_begin(core, 2, 128, true, &outer));
  CHECK(!oai_memprof_core_begin(core, 2, 128, true, &nested));
  oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x2000), 128);
  payload.api_id = 2;
  CHECK(oai_memprof_core_end(&outer, &payload) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_seal(core, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  event_sink_t sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 0);
  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_core_snapshot(core, &snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.admitted_transactions == 1);
  CHECK(snapshot.completed_transactions == 1);
  CHECK(snapshot.recursion_bypasses == 1);
  CHECK(snapshot.emitted_events == 0);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
}

static void test_ring_full_and_sink_retry(void)
{
  oai_memprof_core_t *core = new_core(OAI_MEMPROF_CORE_EXACT_EVENTS, 1, 2);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);
  for (uint64_t sequence = 0; sequence < 3; ++sequence) {
    oai_memprof_core_ticket_t ticket = {0};
    CHECK(oai_memprof_core_begin(core, 1, 8, true, &ticket));
    oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x3000) + sequence, 8);
    CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  }
  CHECK(oai_memprof_core_seal(core, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_core_snapshot(core, &snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.admitted_transactions == 3);
  CHECK(snapshot.completed_transactions == 3);
  CHECK(snapshot.emitted_events == 2);
  CHECK(snapshot.ring_full_losses == 1);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_INVALID_STATE);

  event_sink_t failing = {.fail_at = 1};
  CHECK(oai_memprof_core_drain(core, collect_event, &failing) == OAI_MEMPROF_CORE_SINK_ERROR);
  CHECK(failing.count == 1);
  event_sink_t retry = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &retry) == OAI_MEMPROF_CORE_OK);
  CHECK(retry.count == 1);
  CHECK(retry.events[0].thread_sequence == 2);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
}

typedef struct capacity_worker_s {
  oai_memprof_core_t *core;
  bool admitted;
  bool retry_admitted;
} capacity_worker_t;

static void *capacity_worker(void *argument)
{
  capacity_worker_t *worker = argument;
  oai_memprof_core_ticket_t ticket = {0};
  worker->admitted = oai_memprof_core_begin(worker->core, 4, 0, false, &ticket);
  if (worker->admitted) {
    const oai_memprof_core_payload_t payload = {
        .flags = UINT32_C(1) << 24,
        .api_id = 4,
        .event_kind = 3,
    };
    CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  }
  ticket = (oai_memprof_core_ticket_t){0};
  worker->retry_admitted = oai_memprof_core_begin(worker->core, 4, 0, false, &ticket);
  CHECK(!worker->retry_admitted);
  return NULL;
}

static void test_registration_capacity(void)
{
  oai_memprof_core_t *core = new_core(OAI_MEMPROF_CORE_EXACT_EVENTS, 1, 2);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);
  oai_memprof_core_ticket_t ticket = {0};
  CHECK(oai_memprof_core_begin(core, 1, 1, true, &ticket));
  oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x4000), 1);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  capacity_worker_t worker = {.core = core};
  pthread_t thread;
  CHECK(pthread_create(&thread, NULL, capacity_worker, &worker) == 0);
  CHECK(pthread_join(thread, NULL) == 0);
  CHECK(!worker.admitted);
  CHECK(!worker.retry_admitted);
  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_core_snapshot(core, &snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.ready_threads == 1);
  CHECK(snapshot.reservations == 2);
  CHECK(snapshot.registration_capacity_failures == 2);
  CHECK(snapshot.registration_diagnostic_saturated_mask == 0);
  CHECK(snapshot.diagnostic_saturation_transitions == 0);
  CHECK(oai_memprof_core_seal(core, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  event_sink_t sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
}

typedef struct crossing_worker_s {
  oai_memprof_core_t *core;
  _Atomic(bool) admitted;
  _Atomic(bool) release;
} crossing_worker_t;

static void *crossing_producer(void *argument)
{
  crossing_worker_t *worker = argument;
  oai_memprof_core_ticket_t ticket = {0};
  CHECK(oai_memprof_core_begin(worker->core, 1, 32, true, &ticket));
  atomic_store_explicit(&worker->admitted, true, memory_order_release);
  while (!atomic_load_explicit(&worker->release, memory_order_acquire)) {
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = 10000};
    (void)nanosleep(&pause, NULL);
  }
  oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x5000), 32);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  return NULL;
}

typedef struct seal_worker_s {
  oai_memprof_core_t *core;
  oai_memprof_core_status_t status;
} seal_worker_t;

static void *seal_core(void *argument)
{
  seal_worker_t *worker = argument;
  worker->status = oai_memprof_core_seal(worker->core, UINT64_C(1000000000));
  return NULL;
}

static void test_seal_boundary_and_quiescence(void)
{
  oai_memprof_core_t *core = new_core(OAI_MEMPROF_CORE_EXACT_EVENTS, 2, 4);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);
  crossing_worker_t producer = {.core = core};
  pthread_t producer_thread;
  CHECK(pthread_create(&producer_thread, NULL, crossing_producer, &producer) == 0);
  while (!atomic_load_explicit(&producer.admitted, memory_order_acquire)) {
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = 10000};
    (void)nanosleep(&pause, NULL);
  }

  seal_worker_t sealer = {.core = core, .status = OAI_MEMPROF_CORE_INVALID_STATE};
  pthread_t seal_thread;
  CHECK(pthread_create(&seal_thread, NULL, seal_core, &sealer) == 0);
  for (unsigned attempt = 0; attempt < 100000; ++attempt) {
    if ((oai_memprof_core_control(core) & UINT64_C(0xff)) == OAI_MEMPROF_CORE_DRAINING)
      break;
    if (attempt == 99999)
      CHECK(false);
  }
  oai_memprof_core_ticket_t denied = {0};
  CHECK(!oai_memprof_core_begin(core, 1, 1, true, &denied));
  atomic_store_explicit(&producer.release, true, memory_order_release);
  CHECK(pthread_join(producer_thread, NULL) == 0);
  CHECK(pthread_join(seal_thread, NULL) == 0);
  CHECK(sealer.status == OAI_MEMPROF_CORE_OK);

  event_sink_t sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 1);
  CHECK((sink.events[0].flags & OAI_MEMPROF_CORE_BOUNDARY_STRADDLING) != 0);
  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_core_snapshot(core, &snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.reservations == 1);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
}

static void test_thread_catalog_projection(void)
{
  oai_memprof_core_t *core = new_core(OAI_MEMPROF_CORE_COUNTERS, 2, 2);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);
  oai_memprof_core_ticket_t ticket = {0};
  CHECK(oai_memprof_core_begin(core, 3, 77, true, &ticket));
  oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x6000), 77);
  payload.api_id = 3;
  payload.event_kind = 2;
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_seal(core, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  oai_memprof_core_thread_info_t info = {0};
  CHECK(oai_memprof_core_thread_info(core, 0, &info) == OAI_MEMPROF_CORE_OK);
  CHECK(info.process_generation == UINT64_C(0x12345));
  CHECK(info.registration_ordinal == 1);
  CHECK(info.thread_index == 1);
  CHECK(info.thread_sequence == 1);
  for (size_t api = 0; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api)
    CHECK(info.api_attempts[api] == (api == 2 ? UINT64_C(1) : UINT64_C(0)));
  CHECK(info.requested_bytes == 77);
  CHECK(info.completed_transactions == 1);
  event_sink_t sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
}

static bool discard_event(void *context, const oai_memprof_event_v1_t *event)
{
  (void)context;
  (void)event;
  return true;
}

static oai_memprof_core_payload_t free_payload(uint64_t address)
{
  return (oai_memprof_core_payload_t){
      .address_before = address,
      .flags = (UINT32_C(1) << 0) | (UINT32_C(1) << 12) | (UINT32_C(1) << 24),
      .api_id = 4,
      .event_kind = 3,
  };
}

static oai_memprof_core_payload_t realloc_payload(uint64_t before, uint64_t after, uint64_t bytes)
{
  return (oai_memprof_core_payload_t){
      .address_before = before,
      .address_after = after,
      .arg0 = bytes,
      .flags = (UINT32_C(1) << 0) | (UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 11) | (UINT32_C(1) << 12)
               | (UINT32_C(1) << 24),
      .api_id = 3,
      .event_kind = 2,
  };
}

static void test_selection_literals(void)
{
  const struct {
    uint64_t generation;
    uint32_t thread;
    uint64_t sequence;
    uint64_t expected;
  } vectors[] = {
      {1, 1, 1, UINT64_C(0xf881b6f2eef5f925)},
      {1, 1, 2, UINT64_C(0x54936737c945f86f)},
      {7, 31, 99, UINT64_C(0x6d0ffb5472684179)},
      {UINT64_C(0x0102030405060708), UINT32_C(0x11223344), UINT64_C(0x8899aabbccddeeff), UINT64_C(0x89a052bd3f9a4f4a)},
  };
  for (size_t index = 0; index < sizeof(vectors) / sizeof(vectors[0]); ++index) {
    uint64_t value = 0;
    CHECK(oai_memprof_core_selection_value(vectors[index].generation, vectors[index].thread, vectors[index].sequence, 0, &value));
    CHECK(value == vectors[index].expected);
  }
  uint64_t value = 0;
  CHECK(oai_memprof_core_selection_value(1, 1, 1, UINT64_C(0x0123456789abcdef), &value));
  CHECK(value == (UINT64_C(0xf881b6f2eef5f925) ^ UINT64_C(0x0123456789abcdef)));
  CHECK(!oai_memprof_core_selection_value(0, 1, 1, 0, &value));
  CHECK(!oai_memprof_core_selection_value(1, UINT32_MAX, 1, 0, &value));
  CHECK(!oai_memprof_core_selection_value(1, 1, 0, 0, &value));
  CHECK(!oai_memprof_core_selection_value(1, 1, 1, 0, NULL));
}

static void test_sampled_selection_and_same_thread_release(void)
{
  const uint64_t threshold = UINT64_C(0x54936737c945f870);
  oai_memprof_core_t *core = new_sampled_core(1, 0, threshold, 8, 1, 1, 8);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);

  oai_memprof_core_ticket_t ticket = {0};
  CHECK(oai_memprof_core_begin(core, 1, 16, true, &ticket));
  CHECK(!ticket.successor_selected);
  oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x1000), 16);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  ticket = (oai_memprof_core_ticket_t){0};
  CHECK(oai_memprof_core_begin(core, 1, 64, true, &ticket));
  CHECK(ticket.successor_selected);
  payload = malloc_payload(UINT64_C(0x2000), 64);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  ticket = (oai_memprof_core_ticket_t){0};
  CHECK(oai_memprof_core_begin(core, 4, 0, false, &ticket));
  CHECK(oai_memprof_core_sample_predecessor(&ticket, UINT64_C(0x2000)));
  payload = free_payload(UINT64_C(0x2000));
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  ticket = (oai_memprof_core_ticket_t){0};
  CHECK(oai_memprof_core_begin(core, 4, 0, false, &ticket));
  CHECK(!oai_memprof_core_sample_predecessor(&ticket, UINT64_C(0x1000)));
  payload = free_payload(UINT64_C(0x1000));
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  CHECK(oai_memprof_core_seal(core, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  event_sink_t sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 2);
  CHECK(sink.events[0].thread_sequence == 2);
  CHECK((sink.events[0].flags & OAI_MEMPROF_CORE_SUCCESSOR_SELECTED) != 0);
  CHECK((sink.events[0].flags
         & (OAI_MEMPROF_CORE_PREDECESSOR_MATCH_VALID | OAI_MEMPROF_CORE_PREDECESSOR_SELECTED
            | OAI_MEMPROF_CORE_CROSS_THREAD_ENDPOINT))
        == 0);
  CHECK(sink.events[1].thread_sequence == 3);
  CHECK((sink.events[1].flags & (OAI_MEMPROF_CORE_PREDECESSOR_MATCH_VALID | OAI_MEMPROF_CORE_PREDECESSOR_SELECTED))
        == (OAI_MEMPROF_CORE_PREDECESSOR_MATCH_VALID | OAI_MEMPROF_CORE_PREDECESSOR_SELECTED));
  CHECK((sink.events[1].flags & (OAI_MEMPROF_CORE_SUCCESSOR_SELECTED | OAI_MEMPROF_CORE_CROSS_THREAD_ENDPOINT)) == 0);
  CHECK(sink.events[1].arg0 == 64 && sink.events[1].arg1 == 1 && sink.events[1].arg2 == 2);

  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_core_snapshot(core, &snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.admitted_transactions == 4 && snapshot.completed_transactions == 4 && snapshot.emitted_events == 2);
  CHECK(snapshot.table_entries == 8 && snapshot.table_probes == 1 && snapshot.table_shards == 8);
  CHECK(snapshot.sample_seed == 0 && snapshot.sample_threshold == threshold);
  oai_memprof_core_thread_info_t info = {0};
  CHECK(oai_memprof_core_thread_info(core, 0, &info) == OAI_MEMPROF_CORE_OK);
  CHECK(info.sample_insertion_failures == 0 && info.sample_lookup_failures == 0 && info.sample_probe_exhaustions == 0
        && info.sample_pairing_failures == 0);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
}

static void test_sampled_same_address_realloc_identity(void)
{
  oai_memprof_core_t *core = new_sampled_core(7, 0, UINT64_MAX, 8, 1, 1, 8);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);
  oai_memprof_core_ticket_t ticket = {0};
  CHECK(oai_memprof_core_begin(core, 1, 32, true, &ticket) && ticket.successor_selected);
  oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x3000), 32);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  ticket = (oai_memprof_core_ticket_t){0};
  CHECK(oai_memprof_core_begin(core, 3, 48, true, &ticket) && ticket.successor_selected);
  CHECK(oai_memprof_core_sample_predecessor(&ticket, UINT64_C(0x3000)));
  payload = realloc_payload(UINT64_C(0x3000), UINT64_C(0x3000), 48);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  ticket = (oai_memprof_core_ticket_t){0};
  CHECK(oai_memprof_core_begin(core, 4, 0, false, &ticket));
  CHECK(oai_memprof_core_sample_predecessor(&ticket, UINT64_C(0x3000)));
  CHECK(ticket.predecessor_thread_index == 1 && ticket.predecessor_sequence == 2 && ticket.predecessor_requested_bytes == 48);
  payload = free_payload(UINT64_C(0x3000));
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);

  CHECK(oai_memprof_core_seal(core, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  event_sink_t sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 3);
  CHECK((sink.events[1].flags & (OAI_MEMPROF_CORE_PREDECESSOR_SELECTED | OAI_MEMPROF_CORE_SUCCESSOR_SELECTED))
        == (OAI_MEMPROF_CORE_PREDECESSOR_SELECTED | OAI_MEMPROF_CORE_SUCCESSOR_SELECTED));
  CHECK(sink.events[1].arg1 == 1 && sink.events[1].arg2 == 1);
  CHECK(sink.events[2].arg0 == 48 && sink.events[2].arg1 == 1 && sink.events[2].arg2 == 2);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
}

typedef struct sampled_cross_worker_s {
  oai_memprof_core_t *core;
  bool matched;
} sampled_cross_worker_t;

static void *sampled_cross_free(void *argument)
{
  sampled_cross_worker_t *worker = argument;
  oai_memprof_core_ticket_t ticket = {0};
  CHECK(oai_memprof_core_begin(worker->core, 4, 0, false, &ticket));
  worker->matched = oai_memprof_core_sample_predecessor(&ticket, UINT64_C(0x4000));
  oai_memprof_core_payload_t payload = free_payload(UINT64_C(0x4000));
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  return NULL;
}

static void test_sampled_cross_thread_release(void)
{
  oai_memprof_core_t *core = new_sampled_core(7, 0, UINT64_MAX, 8, 1, 2, 8);
  CHECK(oai_memprof_core_activate(core) == OAI_MEMPROF_CORE_OK);
  oai_memprof_core_ticket_t ticket = {0};
  CHECK(oai_memprof_core_begin(core, 1, 80, true, &ticket) && ticket.successor_selected);
  oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x4000), 80);
  CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  sampled_cross_worker_t worker = {.core = core};
  pthread_t thread;
  CHECK(pthread_create(&thread, NULL, sampled_cross_free, &worker) == 0);
  CHECK(pthread_join(thread, NULL) == 0);
  CHECK(worker.matched);
  CHECK(oai_memprof_core_seal(core, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  event_sink_t sink = {.fail_at = SIZE_MAX};
  CHECK(oai_memprof_core_drain(core, collect_event, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 2);
  CHECK(sink.events[1].thread_index == 2 && sink.events[1].arg1 == 1 && sink.events[1].arg2 == 1);
  CHECK((sink.events[1].flags & OAI_MEMPROF_CORE_CROSS_THREAD_ENDPOINT) != 0);
  CHECK(oai_memprof_core_complete(core) == OAI_MEMPROF_CORE_OK);
}

static void test_sampled_bounded_failure_diagnostics(void)
{
  oai_memprof_core_t *capacity = new_sampled_core(7, 0, UINT64_MAX, 1, 1, 1, 8);
  CHECK(oai_memprof_core_activate(capacity) == OAI_MEMPROF_CORE_OK);
  for (uint64_t address = UINT64_C(0x5000); address <= UINT64_C(0x5100); address += UINT64_C(0x100)) {
    oai_memprof_core_ticket_t ticket = {0};
    CHECK(oai_memprof_core_begin(capacity, 1, 8, true, &ticket) && ticket.successor_selected);
    oai_memprof_core_payload_t payload = malloc_payload(address, 8);
    CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  }
  oai_memprof_core_thread_info_t info = {0};
  CHECK(oai_memprof_core_thread_info(capacity, 0, &info) == OAI_MEMPROF_CORE_OK);
  CHECK(info.sample_insertion_failures == 1 && info.sample_probe_exhaustions == 0 && info.sample_pairing_failures == 0);
  CHECK(oai_memprof_core_seal(capacity, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_drain(capacity, discard_event, NULL) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_complete(capacity) == OAI_MEMPROF_CORE_OK);

  oai_memprof_core_t *duplicate = new_sampled_core(7, 0, UINT64_MAX, 8, 1, 1, 8);
  CHECK(oai_memprof_core_activate(duplicate) == OAI_MEMPROF_CORE_OK);
  for (size_t attempt = 0; attempt < 2; ++attempt) {
    oai_memprof_core_ticket_t ticket = {0};
    CHECK(oai_memprof_core_begin(duplicate, 1, 8, true, &ticket) && ticket.successor_selected);
    oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x6000), 8);
    CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  }
  CHECK(oai_memprof_core_thread_info(duplicate, 0, &info) == OAI_MEMPROF_CORE_OK);
  CHECK(info.sample_pairing_failures == 1 && info.sample_insertion_failures == 0 && info.sample_probe_exhaustions == 0);
  CHECK(oai_memprof_core_seal(duplicate, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_drain(duplicate, discard_event, NULL) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_complete(duplicate) == OAI_MEMPROF_CORE_OK);

  oai_memprof_core_t *probe = new_sampled_core(7, 0, UINT64_MAX, 512, 1, 1, 512);
  CHECK(oai_memprof_core_activate(probe) == OAI_MEMPROF_CORE_OK);
  bool exhausted = false;
  for (uint64_t attempt = 0; attempt < 400 && !exhausted; ++attempt) {
    oai_memprof_core_ticket_t ticket = {0};
    CHECK(oai_memprof_core_begin(probe, 1, 8, true, &ticket));
    oai_memprof_core_payload_t payload = malloc_payload(UINT64_C(0x100000) + attempt * UINT64_C(0x100), 8);
    CHECK(oai_memprof_core_end(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
    CHECK(oai_memprof_core_thread_info(probe, 0, &info) == OAI_MEMPROF_CORE_OK);
    exhausted = info.sample_probe_exhaustions != 0;
  }
  CHECK(exhausted);
  CHECK(info.sample_insertion_failures == 0 && info.sample_lookup_failures == 0 && info.sample_pairing_failures == 0);
  CHECK(oai_memprof_core_seal(probe, UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_drain(probe, discard_event, NULL) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_core_complete(probe) == OAI_MEMPROF_CORE_OK);
}

int main(void)
{
  test_configuration_and_inactive_bypass();
  test_selection_literals();
  test_sampled_selection_and_same_thread_release();
  test_sampled_same_address_realloc_identity();
  test_sampled_cross_thread_release();
  test_sampled_bounded_failure_diagnostics();
  test_exact_event_lifecycle();
  test_counter_mode_and_recursion();
  test_ring_full_and_sink_retry();
  test_registration_capacity();
  test_seal_boundary_and_quiescence();
  test_thread_catalog_projection();
  puts("active producer core tests passed");
  return EXIT_SUCCESS;
}
