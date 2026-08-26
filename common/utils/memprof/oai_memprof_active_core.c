/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _POSIX_C_SOURCE 200809L

#include "oai_memprof_active_core.h"

#include <stdalign.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define OAI_MEMPROF_CACHE_LINE 64U
#define OAI_MEMPROF_NATIVE_SLOT_ALIGNMENT 128U
#define OAI_MEMPROF_CONTROL_STATE_MASK UINT64_C(0xff)
#define OAI_MEMPROF_CONTROL_MODE_SHIFT 8U
#define OAI_MEMPROF_CONTROL_MODE_MASK (UINT64_C(0xff) << OAI_MEMPROF_CONTROL_MODE_SHIFT)
#define OAI_MEMPROF_CONTROL_GENERATION_SHIFT 16U
#define OAI_MEMPROF_MAX_GENERATION ((UINT64_C(1) << 48) - UINT64_C(1))
#define OAI_MEMPROF_MAX_THREADS UINT32_C(65534)
#define OAI_MEMPROF_MAX_RING_RECORDS (UINT32_C(1) << 20)
#define OAI_MEMPROF_MAX_MEMBERSHIP_SHARDS UINT32_C(256)
#define OAI_MEMPROF_INVALID_CPU UINT16_MAX
#define OAI_MEMPROF_SEAL_POLL_NS UINT64_C(50000)
/*
 * A producer-facing weak CAS must not turn transient contention or a spurious
 * failure into unbounded allocator-path work. Eight attempts permit short
 * contention bursts while keeping the retry cost a small compile-time constant.
 */
#define OAI_MEMPROF_PRODUCER_CAS_MAX_ATTEMPTS UINT32_C(8)

#define OAI_MEMPROF_MEMBER_EMPTY UINT64_C(0)
#define OAI_MEMPROF_MEMBER_TOMBSTONE UINT64_C(1)
#define OAI_MEMPROF_MEMBER_BUSY UINT64_C(2)
#define OAI_MEMPROF_MEMBER_LIVE UINT64_C(3)

#define OAI_MEMPROF_DIAGNOSTIC_RING_FULL (UINT32_C(1) << 0)
#define OAI_MEMPROF_DIAGNOSTIC_RECURSION (UINT32_C(1) << 1)
#define OAI_MEMPROF_DIAGNOSTIC_SIZE_UNKNOWN (UINT32_C(1) << 4)
#define OAI_MEMPROF_DIAGNOSTIC_SAMPLE_INSERTION (UINT32_C(1) << 5)
#define OAI_MEMPROF_DIAGNOSTIC_SAMPLE_LOOKUP (UINT32_C(1) << 6)
#define OAI_MEMPROF_DIAGNOSTIC_SAMPLE_PROBE (UINT32_C(1) << 7)
#define OAI_MEMPROF_DIAGNOSTIC_SAMPLE_PAIRING (UINT32_C(1) << 8)
#define OAI_MEMPROF_DIAGNOSTIC_COUNTER_INVALID (UINT32_C(1) << 9)

typedef struct __attribute__((aligned(OAI_MEMPROF_NATIVE_SLOT_ALIGNMENT))) oai_memprof_native_slot_s {
  oai_memprof_event_v1_t event;
  uint8_t padding[OAI_MEMPROF_NATIVE_SLOT_ALIGNMENT - sizeof(oai_memprof_event_v1_t)];
} oai_memprof_native_slot_t;

typedef struct oai_memprof_ring_descriptor_s {
  _Alignas(OAI_MEMPROF_CACHE_LINE) _Atomic(uint64_t) head;
  _Alignas(OAI_MEMPROF_CACHE_LINE) _Atomic(uint64_t) tail;
  _Alignas(OAI_MEMPROF_CACHE_LINE) _Atomic(uint64_t) in_flight;
  _Atomic(uint64_t) ready;
  uint64_t generation;
  uint64_t registration_ordinal;
  uint32_t thread_index;
  uint32_t reserved_zero;
  _Alignas(OAI_MEMPROF_CACHE_LINE) _Atomic(uint64_t) thread_sequence;
  _Atomic(uint64_t) api_attempts[OAI_MEMPROF_CORE_API_SLOT_COUNT];
  _Atomic(uint64_t) requested_bytes;
  _Atomic(uint64_t) completed_transactions;
  _Atomic(uint64_t) recursion_bypasses;
  _Atomic(uint64_t) ring_full_losses;
  _Atomic(uint64_t) size_unknowns;
  _Atomic(uint64_t) sample_insertion_failures;
  _Atomic(uint64_t) sample_lookup_failures;
  _Atomic(uint64_t) sample_probe_exhaustions;
  _Atomic(uint64_t) sample_pairing_failures;
  _Atomic(uint64_t) counter_invalids;
  _Atomic(uint32_t) diagnostic_saturated_mask;
} oai_memprof_ring_descriptor_t;

typedef struct __attribute__((aligned(OAI_MEMPROF_CACHE_LINE))) oai_memprof_membership_entry_s {
  _Atomic(uint64_t) state;
  _Atomic(uint64_t) tag;
  _Atomic(uint64_t) address;
  _Atomic(uint64_t) generation;
  _Atomic(uint64_t) thread_sequence;
  _Atomic(uint64_t) requested_bytes;
  _Atomic(uint64_t) birth_counter;
  _Atomic(uint32_t) thread_index;
  _Atomic(uint32_t) requested_bytes_valid;
} oai_memprof_membership_entry_t;

struct oai_memprof_core_s {
  _Alignas(OAI_MEMPROF_CACHE_LINE) _Atomic(uint64_t) control;
  _Alignas(OAI_MEMPROF_CACHE_LINE) _Atomic(uint64_t) reservations;
  _Alignas(OAI_MEMPROF_CACHE_LINE) _Atomic(uint64_t) consumer_busy;
  _Atomic(uint64_t) registration_capacity_failures;
  _Atomic(uint64_t) unregistered_active_thread_failures;
  _Atomic(uint64_t) diagnostic_saturation_transitions;
  _Atomic(uint32_t) registration_diagnostic_saturated_mask;
  size_t mapped_bytes;
  uint64_t process_generation;
  uint64_t table_entries;
  uint64_t sample_seed;
  uint64_t sample_threshold;
  uint32_t max_threads;
  uint32_t ring_records;
  uint32_t ring_mask;
  uint32_t table_probes;
  uint32_t table_shards;
  uint8_t mode_id;
  oai_memprof_ring_descriptor_t *rings;
  oai_memprof_native_slot_t *slots;
  oai_memprof_membership_entry_t *membership;
};

typedef struct oai_memprof_tls_s {
  oai_memprof_core_t *core;
  uint64_t generation;
  uint32_t slot_index;
  bool registered;
  bool capacity_failed;
  bool guard;
} oai_memprof_tls_t;

static _Thread_local __attribute__((tls_model("initial-exec"))) oai_memprof_tls_t producer_tls;

_Static_assert(sizeof(oai_memprof_event_v1_t) <= OAI_MEMPROF_NATIVE_SLOT_ALIGNMENT,
               "the native event representation must fit one aligned ring slot");
_Static_assert(sizeof(oai_memprof_native_slot_t) == OAI_MEMPROF_NATIVE_SLOT_ALIGNMENT,
               "the native ring slot must be exactly 128 bytes");
_Static_assert(alignof(oai_memprof_native_slot_t) == OAI_MEMPROF_NATIVE_SLOT_ALIGNMENT,
               "the native ring slot must be 128-byte aligned");
_Static_assert(sizeof(oai_memprof_membership_entry_t) == OAI_MEMPROF_CACHE_LINE,
               "one membership entry must occupy exactly one cache line");
_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2 || ATOMIC_LONG_LOCK_FREE == 2,
               "the producer core requires an always-lock-free 64-bit integer atomic");
_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "the producer core requires an always-lock-free 32-bit integer atomic");

static bool is_power_of_two(uint32_t value)
{
  return value != 0 && (value & (value - UINT32_C(1))) == 0;
}

static bool align_up_size(size_t value, size_t alignment, size_t *result)
{
  const size_t mask = alignment - 1U;
  if (value > SIZE_MAX - mask)
    return false;
  *result = (value + mask) & ~mask;
  return true;
}

static bool add_size(size_t left, size_t right, size_t *result)
{
  if (left > SIZE_MAX - right)
    return false;
  *result = left + right;
  return true;
}

static bool multiply_size(size_t left, size_t right, size_t *result)
{
  if (left != 0 && right > SIZE_MAX / left)
    return false;
  *result = left * right;
  return true;
}

static uint64_t pack_control(uint64_t generation, uint8_t mode_id, uint8_t state)
{
  return (generation << OAI_MEMPROF_CONTROL_GENERATION_SHIFT) | ((uint64_t)mode_id << OAI_MEMPROF_CONTROL_MODE_SHIFT)
         | (uint64_t)state;
}

static uint8_t control_state(uint64_t control)
{
  return (uint8_t)(control & OAI_MEMPROF_CONTROL_STATE_MASK);
}

static uint8_t control_mode(uint64_t control)
{
  return (uint8_t)((control & OAI_MEMPROF_CONTROL_MODE_MASK) >> OAI_MEMPROF_CONTROL_MODE_SHIFT);
}

static uint64_t control_generation(uint64_t control)
{
  return control >> OAI_MEMPROF_CONTROL_GENERATION_SHIFT;
}

static uint64_t rotate_left64(uint64_t value, unsigned distance)
{
  return (value << distance) | (value >> (64U - distance));
}

static uint64_t mix64(uint64_t value)
{
  value ^= value >> 30U;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27U;
  value *= UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31U;
  return value;
}

bool oai_memprof_core_selection_value(uint64_t process_generation,
                                      uint32_t thread_index,
                                      uint64_t thread_sequence,
                                      uint64_t sample_seed,
                                      uint64_t *value)
{
  if (process_generation == 0 || thread_index == 0 || thread_index == UINT32_MAX || thread_sequence == 0 || value == NULL)
    return false;
  const uint64_t duplicated_thread = ((uint64_t)thread_index << 32U) | thread_index;
  uint64_t state = mix64(UINT64_C(0x243f6a8885a308d3) ^ process_generation ^ UINT64_C(0x13198a2e03707344));
  state = mix64(state ^ rotate_left64(duplicated_thread ^ UINT64_C(0xa4093822299f31d0), 21U));
  *value = mix64(state ^ rotate_left64(thread_sequence ^ UINT64_C(0x082efa98ec4e6c89), 42U)) ^ sample_seed;
  return true;
}

static uint32_t membership_shards(uint64_t entries)
{
  const uint64_t limit = entries < OAI_MEMPROF_MAX_MEMBERSHIP_SHARDS ? entries : OAI_MEMPROF_MAX_MEMBERSHIP_SHARDS;
  uint32_t shards = 1;
  while ((uint64_t)shards * UINT64_C(2) <= limit)
    shards *= UINT32_C(2);
  return shards;
}

typedef struct membership_range_s {
  size_t offset;
  size_t capacity;
  size_t first;
  size_t probes;
} membership_range_t;

static membership_range_t membership_range(const oai_memprof_core_t *core, uint64_t address)
{
  const uint64_t hash = mix64(address ^ rotate_left64(core->process_generation, 17U) ^ UINT64_C(0xd6e8feb86659fd93));
  const uint32_t shard = (uint32_t)(hash & (core->table_shards - UINT32_C(1)));
  const size_t base = (size_t)(core->table_entries / core->table_shards);
  const size_t remainder = (size_t)(core->table_entries % core->table_shards);
  const size_t capacity = base + (shard < remainder ? 1U : 0U);
  const size_t offset = (size_t)shard * base + (shard < remainder ? shard : remainder);
  const size_t probes = core->table_probes < capacity ? core->table_probes : capacity;
  return (membership_range_t){
      .offset = offset,
      .capacity = capacity,
      .first = capacity == 0 ? 0 : (size_t)(mix64(hash ^ UINT64_C(0x9e3779b97f4a7c15)) % capacity),
      .probes = probes,
  };
}

static void saturating_increment(_Atomic(uint64_t) *value)
{
  uint64_t observed = atomic_load_explicit(value, memory_order_relaxed);
  for (uint32_t attempt = 0; attempt < OAI_MEMPROF_PRODUCER_CAS_MAX_ATTEMPTS; ++attempt) {
    if (observed == UINT64_MAX)
      return;
    if (atomic_compare_exchange_weak_explicit(value, &observed, observed + UINT64_C(1), memory_order_relaxed, memory_order_relaxed))
      return;
  }
  atomic_store_explicit(value, UINT64_MAX, memory_order_relaxed);
}

static void saturating_diagnostic_increment(_Atomic(uint64_t) *value,
                                            _Atomic(uint32_t) *saturated_mask,
                                            uint32_t mask,
                                            _Atomic(uint64_t) *saturation_transitions)
{
  uint64_t observed = atomic_load_explicit(value, memory_order_relaxed);
  for (uint32_t attempt = 0; attempt < OAI_MEMPROF_PRODUCER_CAS_MAX_ATTEMPTS; ++attempt) {
    if (observed == UINT64_MAX)
      break;
    const uint64_t next = observed + UINT64_C(1);
    if (atomic_compare_exchange_weak_explicit(value, &observed, next, memory_order_relaxed, memory_order_relaxed)) {
      if (next != UINT64_MAX)
        return;
      break;
    }
  }
  atomic_store_explicit(value, UINT64_MAX, memory_order_relaxed);
  const uint32_t previous = atomic_fetch_or_explicit(saturated_mask, mask, memory_order_relaxed);
  if ((previous & mask) == 0)
    saturating_increment(saturation_transitions);
}

static void saturating_add(_Atomic(uint64_t) *value, uint64_t increment)
{
  uint64_t observed = atomic_load_explicit(value, memory_order_relaxed);
  for (uint32_t attempt = 0; attempt < OAI_MEMPROF_PRODUCER_CAS_MAX_ATTEMPTS; ++attempt) {
    const uint64_t next = increment > UINT64_MAX - observed ? UINT64_MAX : observed + increment;
    if (atomic_compare_exchange_weak_explicit(value, &observed, next, memory_order_relaxed, memory_order_relaxed))
      return;
  }
  atomic_store_explicit(value, UINT64_MAX, memory_order_relaxed);
}

static bool reserve_slot(oai_memprof_core_t *core, uint64_t *reservation)
{
  uint64_t observed = atomic_load_explicit(&core->reservations, memory_order_seq_cst);
  for (uint32_t attempt = 0; attempt < OAI_MEMPROF_PRODUCER_CAS_MAX_ATTEMPTS; ++attempt) {
    if (observed == UINT64_MAX)
      return false;
    if (atomic_compare_exchange_weak_explicit(&core->reservations,
                                              &observed,
                                              observed + UINT64_C(1),
                                              memory_order_seq_cst,
                                              memory_order_seq_cst)) {
      *reservation = observed;
      return true;
    }
  }
  return false;
}

typedef struct counter_sample_s {
  uint64_t counter;
  uint16_t cpu;
  bool counter_valid;
  bool cpu_valid;
} counter_sample_t;

static counter_sample_t read_counter(void)
{
  counter_sample_t sample = {.counter = 0, .cpu = OAI_MEMPROF_INVALID_CPU, .counter_valid = false, .cpu_valid = false};
#if defined(__x86_64__)
  uint32_t low = 0;
  uint32_t high = 0;
  uint32_t auxiliary = 0;
  __asm__ volatile("rdtscp" : "=a"(low), "=d"(high), "=c"(auxiliary) : : "memory");
  sample.counter = ((uint64_t)high << 32) | low;
  sample.counter_valid = sample.counter != 0;
  if (auxiliary <= UINT16_MAX - UINT32_C(1)) {
    sample.cpu = (uint16_t)auxiliary;
    sample.cpu_valid = true;
  }
#elif defined(__aarch64__)
  __asm__ volatile("mrs %0, cntvct_el0" : "=r"(sample.counter) : : "memory");
  sample.counter_valid = sample.counter != 0;
#else
#error "the active memory-profiler core admits only x86-64 and AArch64"
#endif
  return sample;
}

static uint64_t monotonic_now_ns(bool *valid)
{
  struct timespec now = {0};
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
    *valid = false;
    return 0;
  }
  *valid = true;
  return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static oai_memprof_ring_descriptor_t *ticket_ring(const oai_memprof_core_ticket_t *ticket)
{
  return &ticket->core->rings[ticket->slot_index];
}

static void ticket_diagnostic(oai_memprof_core_ticket_t *ticket, _Atomic(uint64_t) *counter, uint32_t mask)
{
  oai_memprof_ring_descriptor_t *ring = ticket_ring(ticket);
  saturating_diagnostic_increment(counter,
                                  &ring->diagnostic_saturated_mask,
                                  mask,
                                  &ticket->core->diagnostic_saturation_transitions);
}

static void sample_probe_failure(oai_memprof_core_ticket_t *ticket)
{
  ticket_diagnostic(ticket, &ticket_ring(ticket)->sample_probe_exhaustions, OAI_MEMPROF_DIAGNOSTIC_SAMPLE_PROBE);
}

static void sample_pairing_failure(oai_memprof_core_ticket_t *ticket)
{
  ticket_diagnostic(ticket, &ticket_ring(ticket)->sample_pairing_failures, OAI_MEMPROF_DIAGNOSTIC_SAMPLE_PAIRING);
}

static void reset_tls_for_generation(oai_memprof_core_t *core)
{
  producer_tls = (oai_memprof_tls_t){.core = core, .generation = core->process_generation};
}

oai_memprof_core_status_t oai_memprof_core_bootstrap(const oai_memprof_core_config_t *config, oai_memprof_core_t **core_out)
{
  if (config == NULL || core_out == NULL)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  const bool sampled = config->mode_id == OAI_MEMPROF_CORE_SAMPLED;
  const bool table_pair_valid = (config->table_entries == 0 && config->table_probes == 0)
                                || (config->table_entries != 0 && config->table_entries <= SIZE_MAX && config->table_probes != 0
                                    && config->table_probes <= config->table_entries);
  if (config->process_generation == 0 || config->process_generation > OAI_MEMPROF_MAX_GENERATION || config->max_threads == 0
      || config->max_threads > OAI_MEMPROF_MAX_THREADS || config->ring_records < UINT32_C(2)
      || config->ring_records > OAI_MEMPROF_MAX_RING_RECORDS || !is_power_of_two(config->ring_records)
      || (config->mode_id != OAI_MEMPROF_CORE_COUNTERS && !sampled && config->mode_id != OAI_MEMPROF_CORE_EXACT_EVENTS)
      || !table_pair_valid || (sampled && (config->table_entries == 0 || config->sample_threshold == 0))
      || (!sampled && (config->sample_seed != 0 || config->sample_threshold != 0)))
    return OAI_MEMPROF_CORE_INVALID_CONFIGURATION;

  size_t core_bytes = 0;
  size_t descriptor_bytes = 0;
  size_t slot_count = 0;
  size_t slot_bytes = 0;
  size_t descriptors_end = 0;
  size_t slots_end = 0;
  size_t membership_offset = 0;
  size_t membership_bytes = 0;
  size_t total = 0;
  if (!align_up_size(sizeof(oai_memprof_core_t), alignof(oai_memprof_ring_descriptor_t), &core_bytes)
      || !multiply_size(config->max_threads, sizeof(oai_memprof_ring_descriptor_t), &descriptor_bytes)
      || !add_size(core_bytes, descriptor_bytes, &descriptors_end)
      || !align_up_size(descriptors_end, alignof(oai_memprof_native_slot_t), &descriptors_end)
      || !multiply_size(config->max_threads, config->ring_records, &slot_count)
      || !multiply_size(slot_count, sizeof(oai_memprof_native_slot_t), &slot_bytes)
      || !add_size(descriptors_end, slot_bytes, &slots_end))
    return OAI_MEMPROF_CORE_INVALID_CONFIGURATION;
  total = slots_end;
  if (sampled
      && (!align_up_size(slots_end, alignof(oai_memprof_membership_entry_t), &membership_offset)
          || !multiply_size((size_t)config->table_entries, sizeof(oai_memprof_membership_entry_t), &membership_bytes)
          || !add_size(membership_offset, membership_bytes, &total)))
    return OAI_MEMPROF_CORE_INVALID_CONFIGURATION;

  void *mapping = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED)
    return OAI_MEMPROF_CORE_NO_MEMORY;

  oai_memprof_core_t *core = mapping;
  core->mapped_bytes = total;
  core->process_generation = config->process_generation;
  core->table_entries = config->table_entries;
  core->sample_seed = config->sample_seed;
  core->sample_threshold = config->sample_threshold;
  core->max_threads = config->max_threads;
  core->ring_records = config->ring_records;
  core->ring_mask = config->ring_records - UINT32_C(1);
  core->table_probes = config->table_probes;
  core->table_shards = sampled ? membership_shards(config->table_entries) : 0;
  core->mode_id = config->mode_id;
  core->rings = (oai_memprof_ring_descriptor_t *)((uint8_t *)mapping + core_bytes);
  core->slots = (oai_memprof_native_slot_t *)((uint8_t *)mapping + descriptors_end);
  core->membership = sampled ? (oai_memprof_membership_entry_t *)((uint8_t *)mapping + membership_offset) : NULL;
  atomic_init(&core->control, pack_control(config->process_generation, config->mode_id, OAI_MEMPROF_CORE_BOOTSTRAP));
  atomic_init(&core->reservations, 0);
  atomic_init(&core->consumer_busy, 0);
  atomic_init(&core->registration_capacity_failures, 0);
  atomic_init(&core->unregistered_active_thread_failures, 0);
  atomic_init(&core->diagnostic_saturation_transitions, 0);
  atomic_init(&core->registration_diagnostic_saturated_mask, 0);

  for (uint32_t index = 0; index < config->max_threads; ++index) {
    oai_memprof_ring_descriptor_t *ring = &core->rings[index];
    atomic_init(&ring->head, 0);
    atomic_init(&ring->tail, 0);
    atomic_init(&ring->in_flight, 0);
    atomic_init(&ring->ready, 0);
    atomic_init(&ring->thread_sequence, 0);
    atomic_init(&ring->requested_bytes, 0);
    atomic_init(&ring->completed_transactions, 0);
    atomic_init(&ring->recursion_bypasses, 0);
    atomic_init(&ring->ring_full_losses, 0);
    atomic_init(&ring->size_unknowns, 0);
    atomic_init(&ring->sample_insertion_failures, 0);
    atomic_init(&ring->sample_lookup_failures, 0);
    atomic_init(&ring->sample_probe_exhaustions, 0);
    atomic_init(&ring->sample_pairing_failures, 0);
    atomic_init(&ring->counter_invalids, 0);
    atomic_init(&ring->diagnostic_saturated_mask, 0);
    for (size_t api = 0; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api)
      atomic_init(&ring->api_attempts[api], 0);
  }

  for (size_t index = 0; sampled && index < (size_t)config->table_entries; ++index) {
    oai_memprof_membership_entry_t *entry = &core->membership[index];
    atomic_init(&entry->state, OAI_MEMPROF_MEMBER_EMPTY);
    atomic_init(&entry->tag, 0);
    atomic_init(&entry->address, 0);
    atomic_init(&entry->generation, 0);
    atomic_init(&entry->thread_sequence, 0);
    atomic_init(&entry->requested_bytes, 0);
    atomic_init(&entry->birth_counter, 0);
    atomic_init(&entry->thread_index, 0);
    atomic_init(&entry->requested_bytes_valid, 0);
  }

  if (!atomic_is_lock_free(&core->control) || !atomic_is_lock_free(&core->reservations)
      || !atomic_is_lock_free(&core->consumer_busy) || !atomic_is_lock_free(&core->rings[0].head)
      || !atomic_is_lock_free(&core->rings[0].tail) || !atomic_is_lock_free(&core->rings[0].in_flight)
      || !atomic_is_lock_free(&core->rings[0].ready)
      || (sampled
          && (!atomic_is_lock_free(&core->membership[0].state) || !atomic_is_lock_free(&core->membership[0].address)
              || !atomic_is_lock_free(&core->membership[0].thread_index)
              || !atomic_is_lock_free(&core->membership[0].requested_bytes_valid)))) {
    (void)munmap(mapping, total);
    return OAI_MEMPROF_CORE_INVALID_CONFIGURATION;
  }

  *core_out = core;
  return OAI_MEMPROF_CORE_OK;
}

oai_memprof_core_status_t oai_memprof_core_activate(oai_memprof_core_t *core)
{
  if (core == NULL)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  uint64_t expected = pack_control(core->process_generation, core->mode_id, OAI_MEMPROF_CORE_BOOTSTRAP);
  const uint64_t active = pack_control(core->process_generation, core->mode_id, OAI_MEMPROF_CORE_ACTIVE);
  if (!atomic_compare_exchange_strong_explicit(&core->control, &expected, active, memory_order_seq_cst, memory_order_seq_cst))
    return OAI_MEMPROF_CORE_INVALID_STATE;
  return OAI_MEMPROF_CORE_OK;
}

bool oai_memprof_core_begin(oai_memprof_core_t *core,
                            uint16_t api_id,
                            uint64_t requested_bytes,
                            bool requested_bytes_valid,
                            oai_memprof_core_ticket_t *ticket)
{
  if (ticket != NULL)
    *ticket = (oai_memprof_core_ticket_t){0};
  if (core == NULL || ticket == NULL || api_id < 1 || api_id > OAI_MEMPROF_CORE_ADMITTED_API_COUNT)
    return false;

  const uint64_t preliminary = atomic_load_explicit(&core->control, memory_order_seq_cst);
  if (control_state(preliminary) != OAI_MEMPROF_CORE_ACTIVE || control_generation(preliminary) != core->process_generation
      || control_mode(preliminary) != core->mode_id)
    return false;

  if (producer_tls.core != core || producer_tls.generation != core->process_generation)
    reset_tls_for_generation(core);
  if (producer_tls.guard) {
    if (producer_tls.registered) {
      oai_memprof_ring_descriptor_t *ring = &core->rings[producer_tls.slot_index];
      saturating_diagnostic_increment(&ring->recursion_bypasses,
                                      &ring->diagnostic_saturated_mask,
                                      OAI_MEMPROF_DIAGNOSTIC_RECURSION,
                                      &core->diagnostic_saturation_transitions);
    } else {
      saturating_diagnostic_increment(&core->unregistered_active_thread_failures,
                                      &core->registration_diagnostic_saturated_mask,
                                      UINT32_C(1),
                                      &core->diagnostic_saturation_transitions);
    }
    return false;
  }
  producer_tls.guard = true;

  if (producer_tls.capacity_failed) {
    saturating_diagnostic_increment(&core->registration_capacity_failures,
                                    &core->registration_diagnostic_saturated_mask,
                                    UINT32_C(1) << 1,
                                    &core->diagnostic_saturation_transitions);
    producer_tls.guard = false;
    return false;
  }

  if (!producer_tls.registered) {
    uint64_t reservation = 0;
    if (!reserve_slot(core, &reservation) || reservation >= core->max_threads) {
      saturating_diagnostic_increment(&core->registration_capacity_failures,
                                      &core->registration_diagnostic_saturated_mask,
                                      UINT32_C(1) << 1,
                                      &core->diagnostic_saturation_transitions);
      producer_tls.capacity_failed = true;
      producer_tls.guard = false;
      return false;
    }
    oai_memprof_ring_descriptor_t *ring = &core->rings[reservation];
    ring->generation = core->process_generation;
    ring->registration_ordinal = reservation + UINT64_C(1);
    ring->thread_index = (uint32_t)reservation + UINT32_C(1);
    atomic_store_explicit(&ring->ready, UINT64_C(1), memory_order_release);
    producer_tls.slot_index = (uint32_t)reservation;
    producer_tls.registered = true;
  }

  oai_memprof_ring_descriptor_t *ring = &core->rings[producer_tls.slot_index];
  atomic_store_explicit(&ring->in_flight, UINT64_C(1), memory_order_seq_cst);
  const uint64_t admitted = atomic_load_explicit(&core->control, memory_order_seq_cst);
  if (admitted != preliminary || control_state(admitted) != OAI_MEMPROF_CORE_ACTIVE) {
    atomic_store_explicit(&ring->in_flight, UINT64_C(0), memory_order_seq_cst);
    producer_tls.guard = false;
    return false;
  }

  uint64_t sequence = atomic_load_explicit(&ring->thread_sequence, memory_order_relaxed);
  if (sequence == UINT64_MAX) {
    atomic_store_explicit(&ring->in_flight, UINT64_C(0), memory_order_seq_cst);
    producer_tls.guard = false;
    return false;
  }
  sequence += UINT64_C(1);
  atomic_store_explicit(&ring->thread_sequence, sequence, memory_order_relaxed);
  saturating_increment(&ring->api_attempts[api_id - UINT16_C(1)]);
  if (requested_bytes_valid) {
    saturating_add(&ring->requested_bytes, requested_bytes);
  } else if (api_id != UINT16_C(4)) {
    saturating_diagnostic_increment(&ring->size_unknowns,
                                    &ring->diagnostic_saturated_mask,
                                    OAI_MEMPROF_DIAGNOSTIC_SIZE_UNKNOWN,
                                    &core->diagnostic_saturation_transitions);
  }

  const counter_sample_t sample = core->mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS || core->mode_id == OAI_MEMPROF_CORE_SAMPLED
                                      ? read_counter()
                                      : (counter_sample_t){.cpu = OAI_MEMPROF_INVALID_CPU};
  uint64_t selection = 0;
  const bool successor_selected =
      core->mode_id == OAI_MEMPROF_CORE_SAMPLED
      && oai_memprof_core_selection_value(core->process_generation, ring->thread_index, sequence, core->sample_seed, &selection)
      && selection < core->sample_threshold;
  *ticket = (oai_memprof_core_ticket_t){
      .core = core,
      .generation = core->process_generation,
      .thread_sequence = sequence,
      .counter_enter = sample.counter,
      .requested_bytes = requested_bytes,
      .slot_index = producer_tls.slot_index,
      .thread_index = ring->thread_index,
      .cpu_enter = sample.cpu,
      .api_id = api_id,
      .counter_enter_valid = sample.counter_valid,
      .cpu_enter_valid = sample.cpu_valid,
      .requested_bytes_valid = requested_bytes_valid,
      .successor_selected = successor_selected,
      .admitted = true,
  };
  return true;
}

bool oai_memprof_core_sample_predecessor(oai_memprof_core_ticket_t *ticket, uint64_t address)
{
  if (ticket == NULL || !ticket->admitted || ticket->core == NULL || ticket->core->mode_id != OAI_MEMPROF_CORE_SAMPLED
      || address == 0 || (ticket->api_id != UINT16_C(3) && ticket->api_id != UINT16_C(4) && ticket->api_id != UINT16_C(5))
      || ticket->slot_index >= ticket->core->max_threads || ticket->predecessor_match)
    return false;
  oai_memprof_core_t *core = ticket->core;
  if (producer_tls.core != core || producer_tls.generation != ticket->generation || !producer_tls.registered
      || producer_tls.slot_index != ticket->slot_index || !producer_tls.guard)
    return false;

  const membership_range_t range = membership_range(core, address);
  bool found = false;
  size_t found_slot = 0;
  uint64_t found_tag = 0;
  uint64_t found_sequence = 0;
  uint64_t found_requested = 0;
  uint64_t found_birth = 0;
  uint32_t found_thread = 0;
  bool found_requested_valid = false;
  bool conclusive = range.probes == range.capacity;

  for (size_t probe = 0; probe < range.probes; ++probe) {
    const size_t slot = range.offset + (range.first + probe) % range.capacity;
    oai_memprof_membership_entry_t *entry = &core->membership[slot];
    const uint64_t state = atomic_load_explicit(&entry->state, memory_order_acquire);
    if (state == OAI_MEMPROF_MEMBER_EMPTY) {
      conclusive = true;
      break;
    }
    if (state == OAI_MEMPROF_MEMBER_TOMBSTONE)
      continue;
    if (state == OAI_MEMPROF_MEMBER_BUSY) {
      sample_probe_failure(ticket);
      return false;
    }
    if (state != OAI_MEMPROF_MEMBER_LIVE) {
      sample_pairing_failure(ticket);
      return false;
    }

    const uint64_t entry_tag = atomic_load_explicit(&entry->tag, memory_order_relaxed);
    const uint64_t entry_address = atomic_load_explicit(&entry->address, memory_order_relaxed);
    const uint64_t entry_generation = atomic_load_explicit(&entry->generation, memory_order_relaxed);
    const uint64_t entry_sequence = atomic_load_explicit(&entry->thread_sequence, memory_order_relaxed);
    const uint64_t entry_requested = atomic_load_explicit(&entry->requested_bytes, memory_order_relaxed);
    const uint64_t entry_birth = atomic_load_explicit(&entry->birth_counter, memory_order_relaxed);
    const uint32_t entry_thread = atomic_load_explicit(&entry->thread_index, memory_order_relaxed);
    const uint32_t entry_requested_valid = atomic_load_explicit(&entry->requested_bytes_valid, memory_order_relaxed);
    if (entry_requested_valid > UINT32_C(1)) {
      sample_pairing_failure(ticket);
      return false;
    }
    if (atomic_load_explicit(&entry->state, memory_order_acquire) != OAI_MEMPROF_MEMBER_LIVE
        || atomic_load_explicit(&entry->tag, memory_order_relaxed) != entry_tag) {
      sample_probe_failure(ticket);
      return false;
    }
    if (entry_address != address)
      continue;
    if (entry_tag == 0 || entry_generation != core->process_generation || entry_sequence == 0 || entry_thread == 0
        || entry_thread == UINT32_MAX) {
      sample_pairing_failure(ticket);
      return false;
    }
    if (found) {
      sample_pairing_failure(ticket);
      return false;
    }
    found = true;
    found_slot = slot;
    found_tag = entry_tag;
    found_sequence = entry_sequence;
    found_requested = entry_requested;
    found_birth = entry_birth;
    found_thread = entry_thread;
    found_requested_valid = entry_requested_valid != 0;
  }
  if (!conclusive) {
    sample_probe_failure(ticket);
    return false;
  }
  if (!found)
    return false;
  ticket->predecessor_address = address;
  ticket->predecessor_sequence = found_sequence;
  ticket->predecessor_requested_bytes = found_requested;
  ticket->predecessor_requested_bytes_valid = found_requested_valid;
  ticket->predecessor_birth_counter = found_birth;
  ticket->predecessor_tag = found_tag;
  ticket->predecessor_slot = found_slot;
  ticket->predecessor_thread_index = found_thread;
  ticket->predecessor_match = true;
  return true;
}

static bool remove_predecessor(oai_memprof_core_ticket_t *ticket)
{
  oai_memprof_membership_entry_t *entry = &ticket->core->membership[ticket->predecessor_slot];
  uint64_t expected = OAI_MEMPROF_MEMBER_LIVE;
  if (!atomic_compare_exchange_strong_explicit(&entry->state,
                                               &expected,
                                               OAI_MEMPROF_MEMBER_BUSY,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) {
    sample_pairing_failure(ticket);
    return false;
  }
  const bool matches = atomic_load_explicit(&entry->tag, memory_order_relaxed) == ticket->predecessor_tag
                       && atomic_load_explicit(&entry->address, memory_order_relaxed) == ticket->predecessor_address
                       && atomic_load_explicit(&entry->generation, memory_order_relaxed) == ticket->generation
                       && atomic_load_explicit(&entry->thread_index, memory_order_relaxed) == ticket->predecessor_thread_index
                       && atomic_load_explicit(&entry->thread_sequence, memory_order_relaxed) == ticket->predecessor_sequence;
  atomic_store_explicit(&entry->state, matches ? OAI_MEMPROF_MEMBER_TOMBSTONE : OAI_MEMPROF_MEMBER_LIVE, memory_order_release);
  if (!matches)
    sample_pairing_failure(ticket);
  return matches;
}

static bool insert_successor(oai_memprof_core_ticket_t *ticket,
                             uint64_t address,
                             uint64_t requested_bytes,
                             bool requested_bytes_valid,
                             uint64_t birth_counter)
{
  oai_memprof_core_t *core = ticket->core;
  const membership_range_t range = membership_range(core, address);
  size_t candidate = SIZE_MAX;
  uint64_t candidate_state = OAI_MEMPROF_MEMBER_EMPTY;
  bool conclusive = range.probes == range.capacity;

  for (size_t probe = 0; probe < range.probes; ++probe) {
    const size_t slot = range.offset + (range.first + probe) % range.capacity;
    oai_memprof_membership_entry_t *entry = &core->membership[slot];
    const uint64_t state = atomic_load_explicit(&entry->state, memory_order_acquire);
    if (state == OAI_MEMPROF_MEMBER_BUSY) {
      sample_probe_failure(ticket);
      return false;
    }
    if (state == OAI_MEMPROF_MEMBER_TOMBSTONE) {
      if (candidate == SIZE_MAX) {
        candidate = slot;
        candidate_state = state;
      }
      continue;
    }
    if (state == OAI_MEMPROF_MEMBER_EMPTY) {
      conclusive = true;
      if (candidate == SIZE_MAX) {
        candidate = slot;
        candidate_state = state;
      }
      break;
    }
    if (state != OAI_MEMPROF_MEMBER_LIVE) {
      sample_pairing_failure(ticket);
      return false;
    }
    const uint64_t entry_tag = atomic_load_explicit(&entry->tag, memory_order_relaxed);
    const uint64_t entry_address = atomic_load_explicit(&entry->address, memory_order_relaxed);
    if (atomic_load_explicit(&entry->state, memory_order_acquire) != OAI_MEMPROF_MEMBER_LIVE
        || atomic_load_explicit(&entry->tag, memory_order_relaxed) != entry_tag) {
      sample_probe_failure(ticket);
      return false;
    }
    if (entry_address == address) {
      sample_pairing_failure(ticket);
      return false;
    }
  }
  if (!conclusive) {
    sample_probe_failure(ticket);
    return false;
  }
  if (candidate == SIZE_MAX) {
    ticket_diagnostic(ticket, &ticket_ring(ticket)->sample_insertion_failures, OAI_MEMPROF_DIAGNOSTIC_SAMPLE_INSERTION);
    return false;
  }

  oai_memprof_membership_entry_t *entry = &core->membership[candidate];
  uint64_t expected = candidate_state;
  if (!atomic_compare_exchange_strong_explicit(&entry->state,
                                               &expected,
                                               OAI_MEMPROF_MEMBER_BUSY,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) {
    sample_probe_failure(ticket);
    return false;
  }
  const uint64_t previous_tag = atomic_load_explicit(&entry->tag, memory_order_relaxed);
  if (previous_tag == UINT64_MAX) {
    atomic_store_explicit(&entry->state, candidate_state, memory_order_release);
    sample_pairing_failure(ticket);
    return false;
  }
  const uint64_t tag = previous_tag + UINT64_C(1);
  atomic_store_explicit(&entry->address, address, memory_order_relaxed);
  atomic_store_explicit(&entry->generation, ticket->generation, memory_order_relaxed);
  atomic_store_explicit(&entry->thread_sequence, ticket->thread_sequence, memory_order_relaxed);
  atomic_store_explicit(&entry->requested_bytes, requested_bytes, memory_order_relaxed);
  atomic_store_explicit(&entry->birth_counter, birth_counter, memory_order_relaxed);
  atomic_store_explicit(&entry->thread_index, ticket->thread_index, memory_order_relaxed);
  atomic_store_explicit(&entry->requested_bytes_valid, requested_bytes_valid ? UINT32_C(1) : UINT32_C(0), memory_order_relaxed);
  atomic_store_explicit(&entry->tag, tag, memory_order_relaxed);
  atomic_store_explicit(&entry->state, OAI_MEMPROF_MEMBER_LIVE, memory_order_release);
  return true;
}

oai_memprof_core_status_t oai_memprof_core_end(oai_memprof_core_ticket_t *ticket, const oai_memprof_core_payload_t *payload)
{
  if (ticket == NULL || payload == NULL || !ticket->admitted || ticket->core == NULL || payload->api_id != ticket->api_id
      || ticket->slot_index >= ticket->core->max_threads)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  oai_memprof_core_t *core = ticket->core;
  oai_memprof_ring_descriptor_t *ring = ticket_ring(ticket);
  if (producer_tls.core != core || producer_tls.generation != ticket->generation || !producer_tls.registered
      || producer_tls.slot_index != ticket->slot_index || !producer_tls.guard)
    return OAI_MEMPROF_CORE_INVALID_STATE;

  const counter_sample_t exit_sample = core->mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS || core->mode_id == OAI_MEMPROF_CORE_SAMPLED
                                           ? read_counter()
                                           : (counter_sample_t){.cpu = OAI_MEMPROF_INVALID_CPU};
  uint32_t flags = payload->flags
                   & ~(OAI_MEMPROF_CORE_PREDECESSOR_MATCH_VALID | OAI_MEMPROF_CORE_PREDECESSOR_SELECTED
                       | OAI_MEMPROF_CORE_SUCCESSOR_SELECTED | OAI_MEMPROF_CORE_CROSS_THREAD_ENDPOINT);
  uint64_t arg0 = payload->arg0;
  uint64_t arg1 = payload->arg1;
  uint64_t arg2 = payload->arg2;
  const bool predecessor_ended = (payload->flags & (UINT32_C(1) << 12)) != 0;
  const bool successor_created = (payload->flags & (UINT32_C(1) << 11)) != 0;
  const bool selected_predecessor = core->mode_id == OAI_MEMPROF_CORE_SAMPLED && predecessor_ended && ticket->predecessor_match;
  const bool selected_successor = core->mode_id == OAI_MEMPROF_CORE_SAMPLED && successor_created && ticket->successor_selected;

  if (selected_predecessor) {
    (void)remove_predecessor(ticket);
    flags |=
        OAI_MEMPROF_CORE_PREDECESSOR_MATCH_VALID | OAI_MEMPROF_CORE_PREDECESSOR_SELECTED | (UINT32_C(1) << 3) | (UINT32_C(1) << 4);
    arg1 = ticket->predecessor_thread_index;
    arg2 = ticket->predecessor_sequence;
    if (payload->api_id == UINT16_C(4) && ticket->predecessor_requested_bytes_valid) {
      flags |= UINT32_C(1) << 2;
      arg0 = ticket->predecessor_requested_bytes;
    }
    if (ticket->predecessor_thread_index != ticket->thread_index)
      flags |= OAI_MEMPROF_CORE_CROSS_THREAD_ENDPOINT;
  }
  if (selected_successor) {
    flags |= OAI_MEMPROF_CORE_SUCCESSOR_SELECTED;
    (void)insert_successor(ticket,
                           payload->address_after,
                           ticket->requested_bytes,
                           ticket->requested_bytes_valid,
                           exit_sample.counter_valid ? exit_sample.counter : 0);
  }
  if (ticket->counter_enter_valid)
    flags |= OAI_MEMPROF_CORE_COUNTER_ENTER_VALID;
  if (exit_sample.counter_valid)
    flags |= OAI_MEMPROF_CORE_COUNTER_EXIT_VALID;
  if (ticket->cpu_enter_valid)
    flags |= OAI_MEMPROF_CORE_CPU_ENTER_VALID;
  if (exit_sample.cpu_valid)
    flags |= OAI_MEMPROF_CORE_CPU_EXIT_VALID;

  const uint64_t current_control = atomic_load_explicit(&core->control, memory_order_seq_cst);
  if (control_generation(current_control) == ticket->generation && control_state(current_control) == OAI_MEMPROF_CORE_DRAINING)
    flags |= OAI_MEMPROF_CORE_BOUNDARY_STRADDLING;

  const oai_memprof_event_v1_t event = {
      .thread_sequence = ticket->thread_sequence,
      .counter_enter = ticket->counter_enter_valid ? ticket->counter_enter : 0,
      .counter_exit = exit_sample.counter_valid ? exit_sample.counter : 0,
      .address_before = payload->address_before,
      .address_after = payload->address_after,
      .arg0 = arg0,
      .arg1 = arg1,
      .arg2 = arg2,
      .context_id = payload->context_id,
      .callsite_id = payload->callsite_id,
      .thread_index = ring->thread_index,
      .flags = flags,
      .result_code = payload->result_code,
      .api_id = payload->api_id,
      .event_kind = payload->event_kind,
      .cpu_enter = ticket->cpu_enter_valid ? ticket->cpu_enter : OAI_MEMPROF_INVALID_CPU,
      .cpu_exit = exit_sample.cpu_valid ? exit_sample.cpu : OAI_MEMPROF_INVALID_CPU,
  };

  if ((core->mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS || core->mode_id == OAI_MEMPROF_CORE_SAMPLED)
      && (!ticket->counter_enter_valid || !exit_sample.counter_valid || exit_sample.counter < ticket->counter_enter))
    saturating_diagnostic_increment(&ring->counter_invalids,
                                    &ring->diagnostic_saturated_mask,
                                    OAI_MEMPROF_DIAGNOSTIC_COUNTER_INVALID,
                                    &core->diagnostic_saturation_transitions);

  saturating_increment(&ring->completed_transactions);
  const bool emit = core->mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS || selected_predecessor || selected_successor;
  if (emit) {
    const uint64_t head = atomic_load_explicit(&ring->head, memory_order_relaxed);
    const uint64_t tail = atomic_load_explicit(&ring->tail, memory_order_acquire);
    if (head - tail >= core->ring_records) {
      saturating_diagnostic_increment(&ring->ring_full_losses,
                                      &ring->diagnostic_saturated_mask,
                                      OAI_MEMPROF_DIAGNOSTIC_RING_FULL,
                                      &core->diagnostic_saturation_transitions);
    } else {
      const size_t slot = (size_t)ticket->slot_index * core->ring_records + (size_t)(head & core->ring_mask);
      core->slots[slot].event = event;
      atomic_store_explicit(&ring->head, head + UINT64_C(1), memory_order_release);
    }
  }

  atomic_store_explicit(&ring->in_flight, UINT64_C(0), memory_order_seq_cst);
  producer_tls.guard = false;
  ticket->admitted = false;
  return OAI_MEMPROF_CORE_OK;
}

oai_memprof_core_status_t oai_memprof_core_seal(oai_memprof_core_t *core, uint64_t timeout_ns)
{
  if (core == NULL)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  uint64_t expected = pack_control(core->process_generation, core->mode_id, OAI_MEMPROF_CORE_ACTIVE);
  const uint64_t draining = pack_control(core->process_generation, core->mode_id, OAI_MEMPROF_CORE_DRAINING);
  if (!atomic_compare_exchange_strong_explicit(&core->control, &expected, draining, memory_order_seq_cst, memory_order_seq_cst))
    return OAI_MEMPROF_CORE_INVALID_STATE;

  const uint64_t reservations = atomic_load_explicit(&core->reservations, memory_order_seq_cst);
  const uint64_t high_water = reservations < core->max_threads ? reservations : core->max_threads;
  bool valid_time = false;
  const uint64_t start = monotonic_now_ns(&valid_time);
  if (!valid_time)
    return OAI_MEMPROF_CORE_SYSTEM_ERROR;
  const uint64_t deadline = timeout_ns > UINT64_MAX - start ? UINT64_MAX : start + timeout_ns;

  for (;;) {
    bool busy = false;
    for (uint64_t index = 0; index < high_water; ++index) {
      const oai_memprof_ring_descriptor_t *ring = &core->rings[index];
      if (atomic_load_explicit(&ring->ready, memory_order_acquire) != 0
          && atomic_load_explicit(&ring->in_flight, memory_order_seq_cst) != 0) {
        busy = true;
        break;
      }
    }
    if (!busy)
      return OAI_MEMPROF_CORE_OK;
    const uint64_t now = monotonic_now_ns(&valid_time);
    if (!valid_time)
      return OAI_MEMPROF_CORE_SYSTEM_ERROR;
    if (now >= deadline)
      return OAI_MEMPROF_CORE_SEAL_TIMEOUT;
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = (long)OAI_MEMPROF_SEAL_POLL_NS};
    (void)nanosleep(&pause, NULL);
  }
}

oai_memprof_core_status_t oai_memprof_core_drain(oai_memprof_core_t *core, oai_memprof_core_sink_t sink, void *context)
{
  if (core == NULL || sink == NULL)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  const uint8_t state = control_state(atomic_load_explicit(&core->control, memory_order_seq_cst));
  if (state != OAI_MEMPROF_CORE_ACTIVE && state != OAI_MEMPROF_CORE_DRAINING)
    return OAI_MEMPROF_CORE_INVALID_STATE;

  uint64_t expected_idle = 0;
  if (!atomic_compare_exchange_strong_explicit(&core->consumer_busy,
                                               &expected_idle,
                                               UINT64_C(1),
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return OAI_MEMPROF_CORE_INVALID_STATE;

  oai_memprof_core_status_t status = OAI_MEMPROF_CORE_OK;
  const uint64_t reservations = atomic_load_explicit(&core->reservations, memory_order_seq_cst);
  const uint64_t high_water = reservations < core->max_threads ? reservations : core->max_threads;
  for (uint64_t index = 0; index < high_water && status == OAI_MEMPROF_CORE_OK; ++index) {
    oai_memprof_ring_descriptor_t *ring = &core->rings[index];
    if (atomic_load_explicit(&ring->ready, memory_order_acquire) == 0)
      continue;
    uint64_t tail = atomic_load_explicit(&ring->tail, memory_order_relaxed);
    const uint64_t head = atomic_load_explicit(&ring->head, memory_order_acquire);
    while (tail != head) {
      const size_t slot = (size_t)index * core->ring_records + (size_t)(tail & core->ring_mask);
      if (!sink(context, &core->slots[slot].event)) {
        status = OAI_MEMPROF_CORE_SINK_ERROR;
        break;
      }
      ++tail;
      atomic_store_explicit(&ring->tail, tail, memory_order_release);
    }
  }
  atomic_store_explicit(&core->consumer_busy, UINT64_C(0), memory_order_release);
  return status;
}

oai_memprof_core_status_t oai_memprof_core_complete(oai_memprof_core_t *core)
{
  if (core == NULL)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  uint64_t expected = pack_control(core->process_generation, core->mode_id, OAI_MEMPROF_CORE_DRAINING);
  const uint64_t complete = pack_control(core->process_generation, core->mode_id, OAI_MEMPROF_CORE_COMPLETE);
  uint64_t expected_idle = 0;
  if (!atomic_compare_exchange_strong_explicit(&core->consumer_busy,
                                               &expected_idle,
                                               UINT64_C(1),
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return OAI_MEMPROF_CORE_INVALID_STATE;

  oai_memprof_core_status_t status = OAI_MEMPROF_CORE_OK;
  const uint64_t reservations = atomic_load_explicit(&core->reservations, memory_order_seq_cst);
  const uint64_t high_water = reservations < core->max_threads ? reservations : core->max_threads;
  for (uint64_t index = 0; index < high_water; ++index) {
    const oai_memprof_ring_descriptor_t *ring = &core->rings[index];
    if (atomic_load_explicit(&ring->ready, memory_order_acquire) != 0
        && (atomic_load_explicit(&ring->in_flight, memory_order_seq_cst) != 0
            || atomic_load_explicit(&ring->tail, memory_order_acquire)
                   != atomic_load_explicit(&ring->head, memory_order_acquire))) {
      status = OAI_MEMPROF_CORE_INVALID_STATE;
      break;
    }
  }
  if (status == OAI_MEMPROF_CORE_OK
      && !atomic_compare_exchange_strong_explicit(&core->control, &expected, complete, memory_order_seq_cst, memory_order_seq_cst))
    status = OAI_MEMPROF_CORE_INVALID_STATE;
  atomic_store_explicit(&core->consumer_busy, UINT64_C(0), memory_order_release);
  return status;
}

oai_memprof_core_status_t oai_memprof_core_snapshot(const oai_memprof_core_t *core, oai_memprof_core_snapshot_t *snapshot)
{
  if (core == NULL || snapshot == NULL)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  const uint64_t control = atomic_load_explicit(&core->control, memory_order_seq_cst);
  const uint64_t reservations = atomic_load_explicit(&core->reservations, memory_order_seq_cst);
  const uint64_t high_water = reservations < core->max_threads ? reservations : core->max_threads;
  oai_memprof_core_snapshot_t result = {
      .process_generation = core->process_generation,
      .reservations = reservations,
      .registration_capacity_failures = atomic_load_explicit(&core->registration_capacity_failures, memory_order_relaxed),
      .unregistered_active_thread_failures = atomic_load_explicit(&core->unregistered_active_thread_failures, memory_order_relaxed),
      .diagnostic_saturation_transitions = atomic_load_explicit(&core->diagnostic_saturation_transitions, memory_order_relaxed),
      .registration_diagnostic_saturated_mask =
          atomic_load_explicit(&core->registration_diagnostic_saturated_mask, memory_order_relaxed),
      .table_entries = core->table_entries,
      .sample_seed = core->sample_seed,
      .sample_threshold = core->sample_threshold,
      .table_probes = core->table_probes,
      .table_shards = core->table_shards,
      .state = control_state(control),
      .mode_id = control_mode(control),
  };
  for (uint64_t index = 0; index < high_water; ++index) {
    const oai_memprof_ring_descriptor_t *ring = &core->rings[index];
    if (atomic_load_explicit(&ring->ready, memory_order_acquire) == 0)
      continue;
    ++result.ready_threads;
    uint64_t attempts = 0;
    for (size_t api = 0; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api) {
      const uint64_t api_attempts = atomic_load_explicit(&ring->api_attempts[api], memory_order_relaxed);
      attempts = api_attempts > UINT64_MAX - attempts ? UINT64_MAX : attempts + api_attempts;
    }
    result.admitted_transactions =
        attempts > UINT64_MAX - result.admitted_transactions ? UINT64_MAX : result.admitted_transactions + attempts;
    const uint64_t completed = atomic_load_explicit(&ring->completed_transactions, memory_order_relaxed);
    result.completed_transactions =
        completed > UINT64_MAX - result.completed_transactions ? UINT64_MAX : result.completed_transactions + completed;
    const uint64_t requested = atomic_load_explicit(&ring->requested_bytes, memory_order_relaxed);
    result.requested_bytes = requested > UINT64_MAX - result.requested_bytes ? UINT64_MAX : result.requested_bytes + requested;
    const uint64_t recursion = atomic_load_explicit(&ring->recursion_bypasses, memory_order_relaxed);
    result.recursion_bypasses =
        recursion > UINT64_MAX - result.recursion_bypasses ? UINT64_MAX : result.recursion_bypasses + recursion;
    const uint64_t ring_full = atomic_load_explicit(&ring->ring_full_losses, memory_order_relaxed);
    result.ring_full_losses = ring_full > UINT64_MAX - result.ring_full_losses ? UINT64_MAX : result.ring_full_losses + ring_full;
    const uint64_t head = atomic_load_explicit(&ring->head, memory_order_acquire);
    result.emitted_events = head > UINT64_MAX - result.emitted_events ? UINT64_MAX : result.emitted_events + head;
  }
  *snapshot = result;
  return OAI_MEMPROF_CORE_OK;
}

oai_memprof_core_status_t oai_memprof_core_thread_info(const oai_memprof_core_t *core,
                                                       uint32_t slot_index,
                                                       oai_memprof_core_thread_info_t *info)
{
  if (core == NULL || info == NULL || slot_index >= core->max_threads)
    return OAI_MEMPROF_CORE_INVALID_ARGUMENT;
  const oai_memprof_ring_descriptor_t *ring = &core->rings[slot_index];
  if (atomic_load_explicit(&ring->ready, memory_order_acquire) == 0)
    return OAI_MEMPROF_CORE_INVALID_STATE;
  oai_memprof_core_thread_info_t result = {
      .process_generation = ring->generation,
      .registration_ordinal = ring->registration_ordinal,
      .thread_sequence = atomic_load_explicit(&ring->thread_sequence, memory_order_relaxed),
      .requested_bytes = atomic_load_explicit(&ring->requested_bytes, memory_order_relaxed),
      .completed_transactions = atomic_load_explicit(&ring->completed_transactions, memory_order_relaxed),
      .recursion_bypasses = atomic_load_explicit(&ring->recursion_bypasses, memory_order_relaxed),
      .ring_full_losses = atomic_load_explicit(&ring->ring_full_losses, memory_order_relaxed),
      .size_unknowns = atomic_load_explicit(&ring->size_unknowns, memory_order_relaxed),
      .sample_insertion_failures = atomic_load_explicit(&ring->sample_insertion_failures, memory_order_relaxed),
      .sample_lookup_failures = atomic_load_explicit(&ring->sample_lookup_failures, memory_order_relaxed),
      .sample_probe_exhaustions = atomic_load_explicit(&ring->sample_probe_exhaustions, memory_order_relaxed),
      .sample_pairing_failures = atomic_load_explicit(&ring->sample_pairing_failures, memory_order_relaxed),
      .counter_invalids = atomic_load_explicit(&ring->counter_invalids, memory_order_relaxed),
      .diagnostic_saturated_mask = atomic_load_explicit(&ring->diagnostic_saturated_mask, memory_order_relaxed),
      .thread_index = ring->thread_index,
  };
  for (size_t api = 0; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api)
    result.api_attempts[api] = atomic_load_explicit(&ring->api_attempts[api], memory_order_relaxed);
  *info = result;
  return OAI_MEMPROF_CORE_OK;
}

uint64_t oai_memprof_core_control(const oai_memprof_core_t *core)
{
  return core == NULL ? 0 : atomic_load_explicit(&core->control, memory_order_seq_cst);
}
