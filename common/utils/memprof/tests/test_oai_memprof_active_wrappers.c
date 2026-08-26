/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "common/utils/memprof/oai_memprof_active_runtime_abi.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      exit(EXIT_FAILURE);                                                             \
    }                                                                                 \
  } while (0)

void *__wrap_malloc(size_t size);
void *__wrap_calloc(size_t count, size_t size);
void *__wrap_realloc(void *pointer, size_t size);
void __wrap_free(void *pointer);
void *__wrap_reallocarray(void *pointer, size_t count, size_t size);
void *__wrap_aligned_alloc(size_t alignment, size_t size);
int __wrap_posix_memalign(void **memptr, size_t alignment, size_t size);
void *__wrap_memalign(size_t alignment, size_t size);
void *__wrap_valloc(size_t size);
void *__wrap_pvalloc(size_t size);
char *__wrap_strdup(const char *source);
char *__wrap_strndup(const char *source, size_t size);

static unsigned malloc_calls;
static unsigned calloc_calls;
static unsigned realloc_calls;
static unsigned free_calls;
static unsigned reallocarray_calls;
static unsigned aligned_alloc_calls;
static unsigned posix_memalign_calls;
static unsigned memalign_calls;
static unsigned valloc_calls;
static unsigned pvalloc_calls;
static unsigned strdup_calls;
static unsigned strndup_calls;

void *__real_malloc(size_t size)
{
  (void)size;
  ++malloc_calls;
  errno = 40 + (int)malloc_calls;
  return (void *)(uintptr_t)(UINT64_C(0x1000) + malloc_calls * UINT64_C(0x100));
}

void *__real_calloc(size_t count, size_t size)
{
  ++calloc_calls;
  errno = 50 + (int)calloc_calls;
  if (count != 0 && size > SIZE_MAX / count)
    return NULL;
  return (void *)(uintptr_t)(UINT64_C(0x2000) + calloc_calls * UINT64_C(0x100));
}

void *__real_realloc(void *pointer, size_t size)
{
  (void)pointer;
  ++realloc_calls;
  errno = 60 + (int)realloc_calls;
  if (size == 0)
    return NULL;
  return (void *)(uintptr_t)(UINT64_C(0x3000) + realloc_calls * UINT64_C(0x100));
}

void __real_free(void *pointer)
{
  (void)pointer;
  ++free_calls;
  errno = 70 + (int)free_calls;
}

void *__real_reallocarray(void *pointer, size_t count, size_t size)
{
  (void)pointer;
  ++reallocarray_calls;
  errno = 80 + (int)reallocarray_calls;
  if (count != 0 && size > SIZE_MAX / count)
    return NULL;
  return (void *)(uintptr_t)(UINT64_C(0x4000) + reallocarray_calls * UINT64_C(0x100));
}

void *__real_aligned_alloc(size_t alignment, size_t size)
{
  (void)alignment;
  (void)size;
  ++aligned_alloc_calls;
  errno = 90 + (int)aligned_alloc_calls;
  return (void *)(uintptr_t)(UINT64_C(0x5000) + aligned_alloc_calls * UINT64_C(0x100));
}

int __real_posix_memalign(void **memptr, size_t alignment, size_t size)
{
  (void)size;
  ++posix_memalign_calls;
  errno = 100 + (int)posix_memalign_calls;
  if (alignment == 3)
    return 22;
  *memptr = (void *)(uintptr_t)(UINT64_C(0x6000) + posix_memalign_calls * UINT64_C(0x100));
  return 0;
}

void *__real_memalign(size_t alignment, size_t size)
{
  (void)alignment;
  (void)size;
  ++memalign_calls;
  errno = 110 + (int)memalign_calls;
  return (void *)(uintptr_t)(UINT64_C(0x7000) + memalign_calls * UINT64_C(0x100));
}

void *__real_valloc(size_t size)
{
  (void)size;
  ++valloc_calls;
  errno = 120 + (int)valloc_calls;
  return (void *)(uintptr_t)(UINT64_C(0x8000) + valloc_calls * UINT64_C(0x100));
}

void *__real_pvalloc(size_t size)
{
  (void)size;
  ++pvalloc_calls;
  errno = 130 + (int)pvalloc_calls;
  return (void *)(uintptr_t)(UINT64_C(0x9000) + pvalloc_calls * UINT64_C(0x100));
}

char *__real_strdup(const char *source)
{
  (void)source;
  ++strdup_calls;
  errno = 140 + (int)strdup_calls;
  return (char *)(uintptr_t)(UINT64_C(0xa000) + strdup_calls * UINT64_C(0x100));
}

char *__real_strndup(const char *source, size_t size)
{
  (void)source;
  (void)size;
  ++strndup_calls;
  errno = 150 + (int)strndup_calls;
  return (char *)(uintptr_t)(UINT64_C(0xb000) + strndup_calls * UINT64_C(0x100));
}

typedef struct sink_s {
  oai_memprof_event_v1_t events[16];
  size_t count;
} sink_t;

static bool collect(void *context, const oai_memprof_event_v1_t *event)
{
  sink_t *sink = context;
  CHECK(sink->count < sizeof(sink->events) / sizeof(sink->events[0]));
  sink->events[sink->count++] = *event;
  return true;
}

static uint32_t semantic_flags(uint32_t flags)
{
  const uint32_t evidence = (UINT32_C(1) << 5) | (UINT32_C(1) << 6) | (UINT32_C(1) << 7) | (UINT32_C(1) << 8) | (UINT32_C(1) << 18);
  return flags & ~evidence;
}

typedef struct sampled_free_worker_s {
  void *pointer;
} sampled_free_worker_t;

static void *sampled_free_worker(void *argument)
{
  sampled_free_worker_t *worker = argument;
  __wrap_free(worker->pointer);
  return NULL;
}

static int run_sampled(void)
{
  const oai_memprof_active_runtime_config_t config = {
      .core =
          {
              .process_generation = 7,
              .table_entries = 64,
              .sample_seed = 0,
              .sample_threshold = UINT64_MAX,
              .max_threads = 2,
              .ring_records = 16,
              .table_probes = 8,
              .mode_id = OAI_MEMPROF_CORE_SAMPLED,
          },
      .realloc_zero_policy_id = 1,
  };
  CHECK(oai_memprof_active_runtime_bootstrap_v1(&config) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_active_runtime_activate_v1() == OAI_MEMPROF_CORE_OK);

  void *first = __wrap_malloc(64);
  CHECK(first == (void *)(uintptr_t)UINT64_C(0x1100));
  void *replacement = __wrap_realloc(first, 80);
  CHECK(replacement == (void *)(uintptr_t)UINT64_C(0x3100));
  __wrap_free(replacement);

  void *zero = __wrap_calloc(2, 16);
  CHECK(zero == (void *)(uintptr_t)UINT64_C(0x2100));
  CHECK(__wrap_realloc(zero, 0) == NULL);

  void *cross = __wrap_malloc(96);
  CHECK(cross == (void *)(uintptr_t)UINT64_C(0x1200));
  sampled_free_worker_t worker = {.pointer = cross};
  pthread_t thread;
  CHECK(pthread_create(&thread, NULL, sampled_free_worker, &worker) == 0);
  CHECK(pthread_join(thread, NULL) == 0);

  CHECK(oai_memprof_active_runtime_seal_v1(UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  sink_t sink = {0};
  CHECK(oai_memprof_active_runtime_drain_v1(collect, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 7);

  const uint32_t before = UINT32_C(1) << 0;
  const uint32_t after = UINT32_C(1) << 1;
  const uint32_t arg0 = UINT32_C(1) << 2;
  const uint32_t arg1 = UINT32_C(1) << 3;
  const uint32_t arg2 = UINT32_C(1) << 4;
  const uint32_t zero_size = UINT32_C(1) << 9;
  const uint32_t successor_created = UINT32_C(1) << 11;
  const uint32_t predecessor_ended = UINT32_C(1) << 12;
  const uint32_t predecessor_match = OAI_MEMPROF_CORE_PREDECESSOR_MATCH_VALID;
  const uint32_t predecessor_selected = OAI_MEMPROF_CORE_PREDECESSOR_SELECTED;
  const uint32_t successor_selected = OAI_MEMPROF_CORE_SUCCESSOR_SELECTED;
  const uint32_t cross_thread = OAI_MEMPROF_CORE_CROSS_THREAD_ENDPOINT;
  const uint32_t result_errno = UINT32_C(1) << 24;

  CHECK(sink.events[0].thread_index == 1 && sink.events[0].thread_sequence == 1
        && semantic_flags(sink.events[0].flags) == (after | arg0 | successor_created | successor_selected | result_errno));
  CHECK(sink.events[1].thread_index == 1 && sink.events[1].thread_sequence == 2
        && semantic_flags(sink.events[1].flags)
               == (before | after | arg0 | arg1 | arg2 | successor_created | predecessor_ended | predecessor_match
                   | predecessor_selected | successor_selected | result_errno));
  CHECK(sink.events[1].arg0 == 80 && sink.events[1].arg1 == 1 && sink.events[1].arg2 == 1);
  CHECK(sink.events[2].thread_index == 1 && sink.events[2].thread_sequence == 3
        && semantic_flags(sink.events[2].flags)
               == (before | arg0 | arg1 | arg2 | predecessor_ended | predecessor_match | predecessor_selected | result_errno));
  CHECK(sink.events[2].arg0 == 80 && sink.events[2].arg1 == 1 && sink.events[2].arg2 == 2);
  CHECK(sink.events[3].thread_index == 1 && sink.events[3].thread_sequence == 4
        && (semantic_flags(sink.events[3].flags) & successor_selected) != 0);
  CHECK(sink.events[4].thread_index == 1 && sink.events[4].thread_sequence == 5
        && semantic_flags(sink.events[4].flags)
               == (before | after | arg0 | arg1 | arg2 | zero_size | predecessor_ended | predecessor_match | predecessor_selected
                   | result_errno));
  CHECK(sink.events[4].arg0 == 0 && sink.events[4].arg1 == 1 && sink.events[4].arg2 == 4);
  CHECK(sink.events[5].thread_index == 1 && sink.events[5].thread_sequence == 6
        && (semantic_flags(sink.events[5].flags) & successor_selected) != 0);
  CHECK(sink.events[6].thread_index == 2 && sink.events[6].thread_sequence == 1
        && semantic_flags(sink.events[6].flags)
               == (before | arg0 | arg1 | arg2 | predecessor_ended | predecessor_match | predecessor_selected | cross_thread
                   | result_errno));
  CHECK(sink.events[6].arg0 == 96 && sink.events[6].arg1 == 1 && sink.events[6].arg2 == 6);

  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_active_runtime_snapshot_v1(&snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.mode_id == OAI_MEMPROF_CORE_SAMPLED && snapshot.admitted_transactions == 7 && snapshot.completed_transactions == 7
        && snapshot.emitted_events == 7 && snapshot.table_entries == 64 && snapshot.table_probes == 8 && snapshot.sample_seed == 0
        && snapshot.sample_threshold == UINT64_MAX);
  for (uint32_t slot = 0; slot < 2; ++slot) {
    oai_memprof_core_thread_info_t info = {0};
    CHECK(oai_memprof_active_runtime_thread_info_v1(slot, &info) == OAI_MEMPROF_CORE_OK);
    CHECK(info.sample_insertion_failures == 0 && info.sample_lookup_failures == 0 && info.sample_probe_exhaustions == 0
          && info.sample_pairing_failures == 0);
  }
  CHECK(oai_memprof_active_runtime_complete_v1() == OAI_MEMPROF_CORE_OK);
  puts("sampled active allocator wrapper tests passed");
  return EXIT_SUCCESS;
}

static int run_exact(void)
{
  errno = 1;
  CHECK(__wrap_malloc(16) == (void *)(uintptr_t)UINT64_C(0x1100));
  CHECK(errno == 41);
  CHECK(oai_memprof_active_control_load_v1() == 0);

  const oai_memprof_active_runtime_config_t config = {
      .core = {.process_generation = 7, .max_threads = 2, .ring_records = 8, .mode_id = OAI_MEMPROF_CORE_EXACT_EVENTS},
      .realloc_zero_policy_id = 1,
  };
  CHECK(oai_memprof_active_runtime_bootstrap_v1(&config) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_active_runtime_activate_v1() == OAI_MEMPROF_CORE_OK);

  CHECK(__wrap_malloc(64) == (void *)(uintptr_t)UINT64_C(0x1200));
  CHECK(errno == 42);
  CHECK(__wrap_calloc(2, 16) == (void *)(uintptr_t)UINT64_C(0x2100));
  CHECK(errno == 51);
  CHECK(__wrap_calloc(SIZE_MAX, 2) == NULL);
  CHECK(errno == 52);
  CHECK(__wrap_realloc((void *)(uintptr_t)UINT64_C(0x2100), 32) == (void *)(uintptr_t)UINT64_C(0x3100));
  CHECK(errno == 61);
  CHECK(__wrap_realloc((void *)(uintptr_t)UINT64_C(0x3100), 0) == NULL);
  CHECK(errno == 62);
  __wrap_free((void *)(uintptr_t)UINT64_C(0x3200));
  CHECK(errno == 71);

  CHECK(oai_memprof_active_runtime_seal_v1(UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  CHECK(__wrap_malloc(1) == (void *)(uintptr_t)UINT64_C(0x1300));
  CHECK(errno == 43);

  sink_t sink = {0};
  CHECK(oai_memprof_active_runtime_drain_v1(collect, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 6);
  for (size_t index = 0; index < sink.count; ++index) {
    CHECK(sink.events[index].thread_sequence == index + 1);
    CHECK(sink.events[index].thread_index == 1);
    CHECK(sink.events[index].callsite_id == 0);
    CHECK(sink.events[index].context_id == 0);
  }

  const uint32_t result_errno = UINT32_C(1) << 24;
  CHECK(semantic_flags(sink.events[0].flags) == ((UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 11) | result_errno));
  CHECK(sink.events[0].api_id == 1 && sink.events[0].event_kind == 1 && sink.events[0].arg0 == 64
        && sink.events[0].result_code == 42);
  CHECK(
      semantic_flags(sink.events[1].flags)
      == ((UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 3) | (UINT32_C(1) << 4) | (UINT32_C(1) << 11) | result_errno));
  CHECK(sink.events[1].arg0 == 2 && sink.events[1].arg1 == 16 && sink.events[1].arg2 == 32 && sink.events[1].result_code == 51);
  CHECK(
      semantic_flags(sink.events[2].flags)
      == ((UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 3) | (UINT32_C(1) << 10) | (UINT32_C(1) << 13) | result_errno));
  CHECK(sink.events[2].arg2 == 0 && sink.events[2].result_code == 52);
  CHECK(
      semantic_flags(sink.events[3].flags)
      == ((UINT32_C(1) << 0) | (UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 11) | (UINT32_C(1) << 12) | result_errno));
  CHECK(
      semantic_flags(sink.events[4].flags)
      == ((UINT32_C(1) << 0) | (UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 9) | (UINT32_C(1) << 12) | result_errno));
  CHECK(semantic_flags(sink.events[5].flags) == ((UINT32_C(1) << 0) | (UINT32_C(1) << 12) | result_errno));

  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_active_runtime_snapshot_v1(&snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.admitted_transactions == 6);
  CHECK(snapshot.completed_transactions == 6);
  CHECK(snapshot.emitted_events == 6);
  CHECK(snapshot.requested_bytes == 128);
  CHECK(snapshot.ring_full_losses == 0);
  CHECK(oai_memprof_active_runtime_realloc_zero_policy_v1() == 1);
  CHECK(oai_memprof_active_runtime_complete_v1() == OAI_MEMPROF_CORE_OK);
  puts("exact active allocator wrapper tests passed");
  return EXIT_SUCCESS;
}

static int run_deferred_disabled(void)
{
  CHECK(oai_memprof_active_control_load_v1() == 0);
  CHECK(__wrap_reallocarray(NULL, 2, 8) == (void *)(uintptr_t)UINT64_C(0x4100));
  CHECK(__wrap_aligned_alloc(64, 128) == (void *)(uintptr_t)UINT64_C(0x5100));
  void *aligned = NULL;
  CHECK(__wrap_posix_memalign(&aligned, 64, 96) == 0);
  CHECK(aligned == (void *)(uintptr_t)UINT64_C(0x6100));
  CHECK(__wrap_memalign(64, 80) == (void *)(uintptr_t)UINT64_C(0x7100));
  CHECK(__wrap_valloc(24) == (void *)(uintptr_t)UINT64_C(0x8100));
  CHECK(__wrap_pvalloc(40) == (void *)(uintptr_t)UINT64_C(0x9100));
  CHECK(__wrap_strdup((const char *)(uintptr_t)UINT64_C(0xdeadbeef)) == (char *)(uintptr_t)UINT64_C(0xa100));
  CHECK(__wrap_strndup((const char *)(uintptr_t)UINT64_C(0xcafebabe), 7) == (char *)(uintptr_t)UINT64_C(0xb100));
  CHECK(reallocarray_calls == 1 && aligned_alloc_calls == 1 && posix_memalign_calls == 1 && memalign_calls == 1 && valloc_calls == 1
        && pvalloc_calls == 1 && strdup_calls == 1 && strndup_calls == 1);
  puts("disabled deferred allocator wrapper tests passed");
  return EXIT_SUCCESS;
}

static int run_deferred_exact(void)
{
  const oai_memprof_active_runtime_config_t config = {
      .core = {.process_generation = 7, .max_threads = 1, .ring_records = 16, .mode_id = OAI_MEMPROF_CORE_EXACT_EVENTS},
      .realloc_zero_policy_id = 1,
  };
  CHECK(oai_memprof_active_runtime_bootstrap_v1(&config) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_active_runtime_activate_v1() == OAI_MEMPROF_CORE_OK);

  CHECK(__wrap_reallocarray((void *)(uintptr_t)UINT64_C(0x3000), 3, 16) == (void *)(uintptr_t)UINT64_C(0x4100));
  CHECK(errno == 81);
  CHECK(__wrap_reallocarray((void *)(uintptr_t)UINT64_C(0x4100), SIZE_MAX, 2) == NULL);
  CHECK(errno == 82);
  CHECK(__wrap_aligned_alloc(64, 128) == (void *)(uintptr_t)UINT64_C(0x5100));
  CHECK(errno == 91);
  void *aligned = NULL;
  CHECK(__wrap_posix_memalign(&aligned, 64, 96) == 0);
  CHECK(aligned == (void *)(uintptr_t)UINT64_C(0x6100) && errno == 101);
  CHECK(__wrap_posix_memalign((void **)(uintptr_t)UINT64_C(1), 3, 64) == 22);
  CHECK(errno == 102);
  CHECK(__wrap_memalign(64, 80) == (void *)(uintptr_t)UINT64_C(0x7100));
  CHECK(__wrap_valloc(24) == (void *)(uintptr_t)UINT64_C(0x8100));
  CHECK(__wrap_pvalloc(40) == (void *)(uintptr_t)UINT64_C(0x9100));
  CHECK(__wrap_strdup((const char *)(uintptr_t)UINT64_C(0xdeadbeef)) == (char *)(uintptr_t)UINT64_C(0xa100));
  CHECK(__wrap_strndup((const char *)(uintptr_t)UINT64_C(0xcafebabe), 7) == (char *)(uintptr_t)UINT64_C(0xb100));

  CHECK(oai_memprof_active_runtime_seal_v1(UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  sink_t sink = {0};
  CHECK(oai_memprof_active_runtime_drain_v1(collect, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 10);
  const uint16_t expected_api[] = {5, 5, 6, 7, 7, 8, 9, 10, 11, 12};
  for (size_t index = 0; index < sink.count; ++index) {
    CHECK(sink.events[index].api_id == expected_api[index]);
    CHECK(sink.events[index].thread_sequence == index + 1);
    CHECK(sink.events[index].thread_index == 1);
  }

  const uint32_t before = UINT32_C(1) << 0;
  const uint32_t after = UINT32_C(1) << 1;
  const uint32_t arg0 = UINT32_C(1) << 2;
  const uint32_t arg1 = UINT32_C(1) << 3;
  const uint32_t arg2 = UINT32_C(1) << 4;
  const uint32_t successor_created = UINT32_C(1) << 11;
  const uint32_t predecessor_ended = UINT32_C(1) << 12;
  const uint32_t operation_failed = UINT32_C(1) << 13;
  const uint32_t reallocarray_overflow = UINT32_C(1) << 19;
  const uint32_t result_errno = UINT32_C(1) << 24;
  const uint32_t direct_return = UINT32_C(1) << 25;
  CHECK(semantic_flags(sink.events[0].flags)
        == (before | after | arg0 | arg1 | arg2 | successor_created | predecessor_ended | result_errno));
  CHECK(sink.events[0].arg0 == 48 && sink.events[0].arg1 == 3 && sink.events[0].arg2 == 16 && sink.events[0].result_code == 81);
  CHECK(semantic_flags(sink.events[1].flags)
        == (before | after | arg1 | arg2 | operation_failed | reallocarray_overflow | result_errno));
  CHECK(sink.events[1].arg0 == 0 && sink.events[1].result_code == 82);
  CHECK(semantic_flags(sink.events[2].flags) == (after | arg0 | arg1 | successor_created | result_errno));
  CHECK(sink.events[2].arg0 == 64 && sink.events[2].arg1 == 128);
  CHECK(semantic_flags(sink.events[3].flags) == (after | arg0 | arg1 | successor_created | direct_return));
  CHECK(sink.events[3].address_after == UINT64_C(0x6100) && sink.events[3].result_code == 0);
  CHECK(semantic_flags(sink.events[4].flags) == (arg0 | arg1 | operation_failed | direct_return));
  CHECK(sink.events[4].address_after == 0 && sink.events[4].result_code == 22);
  CHECK(sink.events[8].arg0 == UINT64_C(0xdeadbeef)
        && semantic_flags(sink.events[8].flags) == (after | arg0 | successor_created | result_errno));
  CHECK(sink.events[9].arg0 == UINT64_C(0xcafebabe) && sink.events[9].arg1 == 7
        && semantic_flags(sink.events[9].flags) == (after | arg0 | arg1 | successor_created | result_errno));

  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_active_runtime_snapshot_v1(&snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.admitted_transactions == 10 && snapshot.completed_transactions == 10 && snapshot.emitted_events == 10
        && snapshot.requested_bytes == 480);
  oai_memprof_core_thread_info_t info = {0};
  CHECK(oai_memprof_active_runtime_thread_info_v1(0, &info) == OAI_MEMPROF_CORE_OK);
  CHECK(info.api_attempts[4] == 2 && info.api_attempts[6] == 2 && info.api_attempts[5] == 1 && info.api_attempts[7] == 1
        && info.api_attempts[8] == 1 && info.api_attempts[9] == 1 && info.api_attempts[10] == 1 && info.api_attempts[11] == 1
        && info.size_unknowns == 3);
  CHECK(oai_memprof_active_runtime_complete_v1() == OAI_MEMPROF_CORE_OK);
  puts("exact deferred allocator wrapper tests passed");
  return EXIT_SUCCESS;
}

static int run_deferred_sampled(void)
{
  const oai_memprof_active_runtime_config_t config = {
      .core = {.process_generation = 7,
               .table_entries = 64,
               .sample_seed = 0,
               .sample_threshold = UINT64_MAX,
               .max_threads = 1,
               .ring_records = 8,
               .table_probes = 8,
               .mode_id = OAI_MEMPROF_CORE_SAMPLED},
      .realloc_zero_policy_id = 1,
  };
  CHECK(oai_memprof_active_runtime_bootstrap_v1(&config) == OAI_MEMPROF_CORE_OK);
  CHECK(oai_memprof_active_runtime_activate_v1() == OAI_MEMPROF_CORE_OK);

  void *base = __wrap_aligned_alloc(16, 48);
  CHECK(base == (void *)(uintptr_t)UINT64_C(0x5100));
  void *replacement = __wrap_reallocarray(base, 3, 16);
  CHECK(replacement == (void *)(uintptr_t)UINT64_C(0x4100));
  __wrap_free(replacement);
  char *string = __wrap_strdup((const char *)(uintptr_t)UINT64_C(0xdeadbeef));
  CHECK(string == (char *)(uintptr_t)UINT64_C(0xa100));
  __wrap_free(string);

  CHECK(oai_memprof_active_runtime_seal_v1(UINT64_C(100000000)) == OAI_MEMPROF_CORE_OK);
  sink_t sink = {0};
  CHECK(oai_memprof_active_runtime_drain_v1(collect, &sink) == OAI_MEMPROF_CORE_OK);
  CHECK(sink.count == 5);
  const uint32_t arg0 = UINT32_C(1) << 2;
  const uint32_t predecessor_match = OAI_MEMPROF_CORE_PREDECESSOR_MATCH_VALID;
  const uint32_t predecessor_selected = OAI_MEMPROF_CORE_PREDECESSOR_SELECTED;
  const uint32_t successor_selected = OAI_MEMPROF_CORE_SUCCESSOR_SELECTED;
  CHECK(sink.events[0].api_id == 6 && sink.events[0].arg0 == 16 && sink.events[0].arg1 == 48
        && (sink.events[0].flags & successor_selected) != 0);
  CHECK(sink.events[1].api_id == 5 && sink.events[1].arg0 == 48 && sink.events[1].arg1 == 1 && sink.events[1].arg2 == 1
        && (sink.events[1].flags & (predecessor_match | predecessor_selected | successor_selected))
               == (predecessor_match | predecessor_selected | successor_selected));
  CHECK(sink.events[2].api_id == 4 && sink.events[2].arg0 == 48 && sink.events[2].arg1 == 1 && sink.events[2].arg2 == 2);
  CHECK(sink.events[3].api_id == 11 && sink.events[3].arg0 == UINT64_C(0xdeadbeef)
        && (sink.events[3].flags & successor_selected) != 0);
  CHECK(sink.events[4].api_id == 4 && sink.events[4].arg0 == 0 && (sink.events[4].flags & arg0) == 0 && sink.events[4].arg1 == 1
        && sink.events[4].arg2 == 4 && (sink.events[4].flags & predecessor_selected) != 0);

  oai_memprof_core_snapshot_t snapshot = {0};
  CHECK(oai_memprof_active_runtime_snapshot_v1(&snapshot) == OAI_MEMPROF_CORE_OK);
  CHECK(snapshot.admitted_transactions == 5 && snapshot.completed_transactions == 5 && snapshot.emitted_events == 5
        && snapshot.requested_bytes == 96);
  oai_memprof_core_thread_info_t info = {0};
  CHECK(oai_memprof_active_runtime_thread_info_v1(0, &info) == OAI_MEMPROF_CORE_OK);
  CHECK(info.api_attempts[5] == 1 && info.api_attempts[4] == 1 && info.api_attempts[3] == 2 && info.api_attempts[10] == 1
        && info.size_unknowns == 1);
  CHECK(oai_memprof_active_runtime_complete_v1() == OAI_MEMPROF_CORE_OK);
  puts("sampled deferred allocator wrapper tests passed");
  return EXIT_SUCCESS;
}

typedef int (*test_body_t)(void);

static void run_isolated(test_body_t body)
{
  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0)
    _Exit(body());
  int status = 0;
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS);
}

int main(void)
{
  run_isolated(run_exact);
  run_isolated(run_sampled);
  run_isolated(run_deferred_disabled);
  run_isolated(run_deferred_exact);
  run_isolated(run_deferred_sampled);
  puts("active allocator wrapper tests passed");
  return EXIT_SUCCESS;
}
