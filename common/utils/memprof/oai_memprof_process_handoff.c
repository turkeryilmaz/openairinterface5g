/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "oai_memprof_process_handoff.h"

#include <limits.h>
#include <string.h>
#include <sys/mman.h>

#define HANDOFF_MAJOR UINT16_C(1)
#define HANDOFF_MINOR UINT16_C(5)
#define HANDOFF_MAGIC_BYTES 16U
#define HANDOFF_OPENING_OFFSET 128U
#define HANDOFF_WRITER_OFFSET 640U
#define HANDOFF_BOOTSTRAP_SHA256_OFFSET 952U
#define HANDOFF_MAPS_SHA256_OFFSET 984U
#define HANDOFF_OPENING_SHA256_OFFSET 1016U
#define HANDOFF_OPENING_SAMPLE_OFFSET 1048U
#define HANDOFF_PREFIX_SHA256_OFFSET 1080U
#define HANDOFF_SAMPLING_CONTROL_OFFSET 1112U
#define HANDOFF_RESERVED_FINAL_OFFSET 1144U
#define HANDOFF_SELF_SHA256_BYTES 32U
#define HANDOFF_THREAD_API_ATTEMPTS_OFFSET 24U
#define HANDOFF_THREAD_REQUESTED_BYTES_OFFSET 280U
#define HANDOFF_THREAD_COMPLETED_TRANSACTIONS_OFFSET 288U
#define HANDOFF_THREAD_RECURSION_BYPASSES_OFFSET 296U
#define HANDOFF_THREAD_RING_FULL_LOSSES_OFFSET 304U
#define HANDOFF_THREAD_SIZE_UNKNOWNS_OFFSET 312U
#define HANDOFF_THREAD_COUNTER_INVALIDS_OFFSET 320U
#define HANDOFF_THREAD_INDEX_OFFSET 328U
#define HANDOFF_THREAD_DIAGNOSTIC_SATURATED_MASK_OFFSET 332U
#define HANDOFF_THREAD_SAMPLE_INSERTION_FAILURES_OFFSET 336U
#define HANDOFF_THREAD_SAMPLE_LOOKUP_FAILURES_OFFSET 344U
#define HANDOFF_THREAD_SAMPLE_PROBE_EXHAUSTIONS_OFFSET 352U
#define HANDOFF_THREAD_SAMPLE_PAIRING_FAILURES_OFFSET 360U
#define HANDOFF_THREAD_RUNTIME_BYTES 368U

static const uint8_t handoff_magic[HANDOFF_MAGIC_BYTES] = {
    'O',
    'A',
    'I',
    'M',
    'P',
    'H',
    'A',
    'N',
    'D',
    'O',
    'F',
    'F',
    'V',
    '1',
    0,
    0,
};

_Static_assert(HANDOFF_THREAD_REQUESTED_BYTES_OFFSET == HANDOFF_THREAD_API_ATTEMPTS_OFFSET + OAI_MEMPROF_CORE_API_SLOT_COUNT * 8U,
               "handoff API counter extent mismatch");
_Static_assert(HANDOFF_THREAD_RUNTIME_BYTES + OAI_MEMPROF_PROCESS_HANDOFF_V1_DIAGNOSTIC_COUNT * 8U
                   == OAI_MEMPROF_PROCESS_HANDOFF_V1_THREAD_BYTES,
               "handoff thread-row extent mismatch");
_Static_assert(OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_WIRE_BYTES
                   == OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES + OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_BOOTSTRAP_BYTES
                          + OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_MAPS_BYTES
                          + OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_THREADS * OAI_MEMPROF_PROCESS_HANDOFF_V1_THREAD_BYTES
                          + OAI_MEMPROF_PROCESS_HANDOFF_V1_DIGEST_BYTES,
               "handoff maximum wire extent mismatch");

static void put_u16(uint8_t *destination, uint16_t value)
{
  destination[0] = (uint8_t)value;
  destination[1] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *destination, uint32_t value)
{
  for (unsigned index = 0; index < 4U; ++index)
    destination[index] = (uint8_t)(value >> (index * 8U));
}

static void put_u64(uint8_t *destination, uint64_t value)
{
  for (unsigned index = 0; index < 8U; ++index)
    destination[index] = (uint8_t)(value >> (index * 8U));
}

static uint16_t get_u16(const uint8_t *source)
{
  return (uint16_t)source[0] | ((uint16_t)source[1] << 8);
}

static uint32_t get_u32(const uint8_t *source)
{
  uint32_t value = 0;
  for (unsigned index = 0; index < 4U; ++index)
    value |= (uint32_t)source[index] << (index * 8U);
  return value;
}

static uint64_t get_u64(const uint8_t *source)
{
  uint64_t value = 0;
  for (unsigned index = 0; index < 8U; ++index)
    value |= (uint64_t)source[index] << (index * 8U);
  return value;
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

static bool is_power_of_two(uint32_t value)
{
  return value != 0 && (value & (value - UINT32_C(1))) == 0;
}

static uint32_t expected_table_shards(uint64_t entries)
{
  const uint64_t limit = entries < UINT64_C(256) ? entries : UINT64_C(256);
  uint32_t shards = 1;
  while ((uint64_t)shards * UINT64_C(2) <= limit)
    shards *= UINT32_C(2);
  return shards;
}

static bool digest_is_zero(const uint8_t digest[32])
{
  uint8_t aggregate = 0;
  for (size_t index = 0; index < 32U; ++index)
    aggregate |= digest[index];
  return aggregate == 0;
}

static bool supplied_digest_matches(const uint8_t supplied[32], const uint8_t expected[32])
{
  return digest_is_zero(supplied) || memcmp(supplied, expected, 32U) == 0;
}

static uint64_t saturating_add(uint64_t left, uint64_t right)
{
  return right > UINT64_MAX - left ? UINT64_MAX : left + right;
}

oai_memprof_process_handoff_status_t oai_memprof_process_handoff_v1_size(size_t bootstrap_size,
                                                                         size_t maps_size,
                                                                         size_t thread_count,
                                                                         size_t *wire_size)
{
  if (wire_size == NULL)
    return OAI_MEMPROF_PROCESS_HANDOFF_INVALID_ARGUMENT;
  if (bootstrap_size == 0 || bootstrap_size > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_BOOTSTRAP_BYTES || maps_size == 0
      || maps_size > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_MAPS_BYTES || thread_count > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_THREADS)
    return OAI_MEMPROF_PROCESS_HANDOFF_INVALID_CONFIGURATION;
  size_t thread_bytes = 0;
  size_t total = OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES;
  if (!multiply_size(thread_count, OAI_MEMPROF_PROCESS_HANDOFF_V1_THREAD_BYTES, &thread_bytes)
      || !add_size(total, bootstrap_size, &total) || !add_size(total, maps_size, &total) || !add_size(total, thread_bytes, &total)
      || !add_size(total, HANDOFF_SELF_SHA256_BYTES, &total))
    return OAI_MEMPROF_PROCESS_HANDOFF_INTEGER_OVERFLOW;
  *wire_size = total;
  return OAI_MEMPROF_PROCESS_HANDOFF_OK;
}

static bool sample_is_zero(const oai_memprof_clock_sample_v1_t *sample)
{
  return sample->counter == 0 && sample->monotonic_raw_before_ns == 0 && sample->monotonic_raw_after_ns == 0
         && sample->realtime_unix_ns == 0;
}

static bool valid_sample_prefix(const oai_memprof_stream_writer_result_t *writer, bool require_complete)
{
  const oai_memprof_clock_sample_v1_t *samples[] = {
      &writer->seal_before_sample,
      &writer->seal_after_sample,
      &writer->drain_complete_sample,
      &writer->final_sample,
  };
  bool reached_zero = false;
  const oai_memprof_clock_sample_v1_t *previous = NULL;
  for (size_t index = 0; index < 4U; ++index) {
    const oai_memprof_clock_sample_v1_t *sample = samples[index];
    if (sample_is_zero(sample)) {
      reached_zero = true;
      continue;
    }
    if (reached_zero)
      return false;
    if (sample->counter == 0 || sample->monotonic_raw_after_ns < sample->monotonic_raw_before_ns || sample->realtime_unix_ns == 0)
      return false;
    if (previous != NULL
        && (previous->counter >= sample->counter || previous->monotonic_raw_after_ns > sample->monotonic_raw_before_ns))
      return false;
    previous = sample;
  }
  return !require_complete || !reached_zero;
}

static bool diagnostic_mask_matches_values(const uint64_t values[OAI_MEMPROF_PROCESS_HANDOFF_V1_DIAGNOSTIC_COUNT], uint32_t mask)
{
  uint32_t expected = 0;
  for (size_t diagnostic = 0; diagnostic < OAI_MEMPROF_PROCESS_HANDOFF_V1_DIAGNOSTIC_COUNT; ++diagnostic)
    if (values[diagnostic] == UINT64_MAX)
      expected |= UINT32_C(1) << diagnostic;
  return expected == mask;
}

static bool valid_handoff(const oai_memprof_process_handoff_v1_t *handoff)
{
  if (handoff == NULL || handoff->bootstrap_bytes == NULL || handoff->maps_bytes == NULL
      || (handoff->thread_count != 0 && handoff->threads == NULL) || digest_is_zero(handoff->prefix_sha256)
      || handoff->thread_count > UINT32_MAX || handoff->ring_records < UINT32_C(2)
      || handoff->ring_records > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_RING_RECORDS || !is_power_of_two(handoff->ring_records)
      || handoff->flush_records == 0 || handoff->flush_records > OAI_MEMPROF_STREAM_WRITER_MAX_FLUSH_RECORDS
      || (handoff->realloc_zero_policy_id != 1 && handoff->realloc_zero_policy_id != 2)
      || (handoff->registration_diagnostic_saturated_mask & ~UINT32_C(0x3)) != 0)
    return false;

  const oai_memprof_stream_writer_result_t *writer = &handoff->writer;
  const oai_memprof_core_snapshot_t *snapshot = &writer->runtime_snapshot;
  if (handoff->opening_header.process_generation == 0 || snapshot->process_generation != handoff->opening_header.process_generation
      || snapshot->ready_threads != handoff->thread_count || snapshot->reservations < snapshot->ready_threads
      || snapshot->unregistered_active_thread_failures != handoff->unregistered_active_thread_failures
      || snapshot->diagnostic_saturation_transitions != handoff->diagnostic_saturation_transitions
      || snapshot->registration_diagnostic_saturated_mask != handoff->registration_diagnostic_saturated_mask
      || (snapshot->mode_id != OAI_MEMPROF_CORE_COUNTERS && snapshot->mode_id != OAI_MEMPROF_CORE_SAMPLED
          && snapshot->mode_id != OAI_MEMPROF_CORE_EXACT_EVENTS)
      || snapshot->state != OAI_MEMPROF_CORE_DRAINING || writer->system_errno < 0
      || writer->status > OAI_MEMPROF_STREAM_WRITER_CLOCK_ERROR || writer->runtime_status > OAI_MEMPROF_CORE_SINK_ERROR
      || writer->clock_status > OAI_MEMPROF_CLOCK_SEQUENCE_ERROR
      || writer->clock_info.counter_frequency_numerator != handoff->opening_header.counter_frequency_numerator
      || writer->clock_info.counter_frequency_denominator != handoff->opening_header.counter_frequency_denominator
      || writer->clock_info.clock_kind != handoff->opening_header.clock_kind
      || !valid_sample_prefix(writer, writer->clock_status == OAI_MEMPROF_CLOCK_OK)
      || handoff->opening_sample.counter != handoff->opening_header.start_counter
      || handoff->opening_sample.realtime_unix_ns != handoff->opening_header.start_realtime_unix_ns
      || handoff->opening_sample.monotonic_raw_before_ns > handoff->opening_header.start_monotonic_raw_ns
      || handoff->opening_sample.monotonic_raw_after_ns < handoff->opening_header.start_monotonic_raw_ns
      || (writer->clock_status == OAI_MEMPROF_CLOCK_OK && writer->status == OAI_MEMPROF_STREAM_WRITER_CLOCK_ERROR)
      || (writer->clock_status != OAI_MEMPROF_CLOCK_OK && writer->status == OAI_MEMPROF_STREAM_WRITER_OK))
    return false;
  const bool sampled = snapshot->mode_id == OAI_MEMPROF_CORE_SAMPLED;
  if ((sampled
       && (snapshot->table_entries == 0 || snapshot->table_entries > SIZE_MAX || snapshot->sample_threshold == 0
           || snapshot->table_probes == 0 || snapshot->table_probes > snapshot->table_entries
           || snapshot->table_shards != expected_table_shards(snapshot->table_entries)))
      || (!sampled
          && (snapshot->table_entries != 0 || snapshot->sample_seed != 0 || snapshot->sample_threshold != 0
              || snapshot->table_probes != 0 || snapshot->table_shards != 0)))
    return false;
  if (!sample_is_zero(&writer->seal_before_sample)
      && (handoff->opening_sample.counter >= writer->seal_before_sample.counter
          || handoff->opening_sample.monotonic_raw_after_ns > writer->seal_before_sample.monotonic_raw_before_ns))
    return false;

  if (writer->record_count > UINT64_MAX / OAI_MEMPROF_EVENT_V1_WIRE_SIZE
      || writer->chunk_count > UINT64_MAX / OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE
      || writer->payload_bytes != writer->record_count * OAI_MEMPROF_EVENT_V1_WIRE_SIZE
      || writer->stream_bytes < OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE
      || (writer->chunk_count == 0) != (writer->record_count == 0) || writer->chunk_count > writer->record_count)
    return false;
  uint64_t complete_prefix = OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE;
  const uint64_t chunk_bytes = writer->chunk_count * OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE;
  if (chunk_bytes > UINT64_MAX - complete_prefix || writer->payload_bytes > UINT64_MAX - complete_prefix - chunk_bytes)
    return false;
  complete_prefix += chunk_bytes + writer->payload_bytes;
  if (complete_prefix > writer->stream_bytes || writer->record_count > snapshot->emitted_events
      || (snapshot->mode_id == OAI_MEMPROF_CORE_COUNTERS && (writer->chunk_count != 0 || writer->record_count != 0)))
    return false;

  uint64_t admitted = 0;
  uint64_t completed = 0;
  uint64_t requested = 0;
  uint64_t recursion = 0;
  uint64_t ring_full = 0;
  uint64_t saturated_instances = 0;
  for (size_t index = 0; index < handoff->thread_count; ++index) {
    const oai_memprof_process_handoff_thread_v1_t *thread = &handoff->threads[index];
    const uint32_t mask = thread->runtime.diagnostic_saturated_mask;
    if (thread->runtime.process_generation != handoff->opening_header.process_generation
        || thread->runtime.registration_ordinal != index + UINT64_C(1) || thread->runtime.thread_index != index + UINT32_C(1)
        || thread->diagnostic_saturated_mask != mask || (mask & ~UINT32_C(0x3ff)) != 0
        || !diagnostic_mask_matches_values(thread->diagnostic_values, mask))
      return false;

    uint64_t attempts = 0;
    for (size_t api = 0; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api)
      attempts = saturating_add(attempts, thread->runtime.api_attempts[api]);
    for (size_t api = OAI_MEMPROF_CORE_ADMITTED_API_COUNT; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api)
      if (thread->runtime.api_attempts[api] != 0)
        return false;
    if (thread->runtime.completed_transactions > attempts)
      return false;
    admitted = saturating_add(admitted, attempts);
    completed = saturating_add(completed, thread->runtime.completed_transactions);
    requested = saturating_add(requested, thread->runtime.requested_bytes);
    recursion = saturating_add(recursion, thread->runtime.recursion_bypasses);
    ring_full = saturating_add(ring_full, thread->runtime.ring_full_losses);

    if (thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_RING_FULL] != thread->runtime.ring_full_losses
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_RECURSION_BYPASS] != thread->runtime.recursion_bypasses
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_INTERNAL_BYPASS] != 0
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_UNSUPPORTED_DOMAIN] != 0
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SIZE_UNKNOWN] != thread->runtime.size_unknowns
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_INSERTION] != thread->runtime.sample_insertion_failures
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_LOOKUP] != thread->runtime.sample_lookup_failures
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_PROBE] != thread->runtime.sample_probe_exhaustions
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_PAIRING] != thread->runtime.sample_pairing_failures
        || thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_COUNTER_INVALID] != thread->runtime.counter_invalids)
      return false;
    if ((snapshot->mode_id != OAI_MEMPROF_CORE_SAMPLED
         && (thread->runtime.sample_insertion_failures != 0 || thread->runtime.sample_lookup_failures != 0
             || thread->runtime.sample_probe_exhaustions != 0 || thread->runtime.sample_pairing_failures != 0))
        || (snapshot->mode_id == OAI_MEMPROF_CORE_COUNTERS
            && (thread->runtime.ring_full_losses != 0 || thread->runtime.counter_invalids != 0)))
      return false;
    for (size_t diagnostic = 0; diagnostic < OAI_MEMPROF_PROCESS_HANDOFF_V1_DIAGNOSTIC_COUNT; ++diagnostic)
      saturated_instances += thread->diagnostic_values[diagnostic] == UINT64_MAX ? UINT64_C(1) : UINT64_C(0);
  }

  const uint64_t registration_values[] = {
      handoff->unregistered_active_thread_failures,
      snapshot->registration_capacity_failures,
  };
  for (size_t index = 0; index < 2U; ++index) {
    const bool saturated = registration_values[index] == UINT64_MAX;
    if (saturated != ((handoff->registration_diagnostic_saturated_mask & (UINT32_C(1) << index)) != 0))
      return false;
    saturated_instances += saturated ? UINT64_C(1) : UINT64_C(0);
  }
  const bool writer_saturated = handoff->writer_io_or_finalization_failures == UINT64_MAX;
  saturated_instances += writer_saturated ? UINT64_C(1) : UINT64_C(0);

  if (admitted != snapshot->admitted_transactions || completed != snapshot->completed_transactions
      || requested != snapshot->requested_bytes || recursion != snapshot->recursion_bypasses
      || ring_full != snapshot->ring_full_losses || saturated_instances != handoff->diagnostic_saturation_transitions)
    return false;

  if (writer->status == OAI_MEMPROF_STREAM_WRITER_OK) {
    if (writer->runtime_status != OAI_MEMPROF_CORE_OK || !writer->prefooter_closed || admitted != completed
        || writer->record_count != snapshot->emitted_events || writer->stream_bytes != complete_prefix
        || handoff->writer_io_or_finalization_failures != 0)
      return false;
  } else if (handoff->writer_io_or_finalization_failures == 0) {
    return false;
  }
  return true;
}

static void encode_sample(uint8_t *wire, const oai_memprof_clock_sample_v1_t *sample)
{
  put_u64(wire, sample->counter);
  put_u64(wire + 8U, sample->monotonic_raw_before_ns);
  put_u64(wire + 16U, sample->monotonic_raw_after_ns);
  put_u64(wire + 24U, sample->realtime_unix_ns);
}

static void decode_sample(oai_memprof_clock_sample_v1_t *sample, const uint8_t *wire)
{
  *sample = (oai_memprof_clock_sample_v1_t){
      .counter = get_u64(wire),
      .monotonic_raw_before_ns = get_u64(wire + 8U),
      .monotonic_raw_after_ns = get_u64(wire + 16U),
      .realtime_unix_ns = get_u64(wire + 24U),
  };
}

static void encode_writer(uint8_t *wire, const oai_memprof_stream_writer_result_t *writer)
{
  put_u32(wire, (uint32_t)writer->status);
  put_u32(wire + 4U, (uint32_t)writer->runtime_status);
  put_u32(wire + 8U, (uint32_t)writer->clock_status);
  put_u32(wire + 12U, writer->prefooter_closed ? 1U : 0U);
  put_u32(wire + 16U, (uint32_t)writer->system_errno);
  put_u64(wire + 24U, writer->runtime_snapshot.process_generation);
  put_u64(wire + 32U, writer->runtime_snapshot.reservations);
  put_u64(wire + 40U, writer->runtime_snapshot.ready_threads);
  put_u64(wire + 48U, writer->runtime_snapshot.registration_capacity_failures);
  put_u64(wire + 56U, writer->runtime_snapshot.recursion_bypasses);
  put_u64(wire + 64U, writer->runtime_snapshot.ring_full_losses);
  put_u64(wire + 72U, writer->runtime_snapshot.admitted_transactions);
  put_u64(wire + 80U, writer->runtime_snapshot.completed_transactions);
  put_u64(wire + 88U, writer->runtime_snapshot.emitted_events);
  put_u64(wire + 96U, writer->runtime_snapshot.requested_bytes);
  wire[104U] = writer->runtime_snapshot.state;
  wire[105U] = writer->runtime_snapshot.mode_id;
  put_u64(wire + 112U, writer->clock_info.counter_frequency_numerator);
  put_u64(wire + 120U, writer->clock_info.counter_frequency_denominator);
  put_u16(wire + 128U, writer->clock_info.architecture_id);
  put_u16(wire + 130U, writer->clock_info.acquisition_source_id);
  wire[132U] = writer->clock_info.clock_kind;
  encode_sample(wire + 136U, &writer->seal_before_sample);
  encode_sample(wire + 168U, &writer->seal_after_sample);
  encode_sample(wire + 200U, &writer->drain_complete_sample);
  encode_sample(wire + 232U, &writer->final_sample);
  put_u64(wire + 264U, writer->chunk_count);
  put_u64(wire + 272U, writer->record_count);
  put_u64(wire + 280U, writer->payload_bytes);
  put_u64(wire + 288U, writer->stream_bytes);
  put_u64(wire + 296U, writer->file_device);
  put_u64(wire + 304U, writer->file_inode);
}

static bool decode_writer(oai_memprof_stream_writer_result_t *writer, const uint8_t *wire)
{
  const uint32_t status = get_u32(wire);
  const uint32_t runtime_status = get_u32(wire + 4U);
  const uint32_t clock_status = get_u32(wire + 8U);
  const uint32_t closed = get_u32(wire + 12U);
  const uint32_t system_errno = get_u32(wire + 16U);
  if (status > OAI_MEMPROF_STREAM_WRITER_CLOCK_ERROR || runtime_status > OAI_MEMPROF_CORE_SINK_ERROR
      || clock_status > OAI_MEMPROF_CLOCK_SEQUENCE_ERROR || closed > 1U || system_errno > INT32_MAX || get_u32(wire + 20U) != 0
      || wire[106U] != 0 || wire[107U] != 0 || wire[108U] != 0 || wire[109U] != 0 || wire[110U] != 0 || wire[111U] != 0
      || wire[133U] != 0 || wire[134U] != 0 || wire[135U] != 0)
    return false;
  *writer = (oai_memprof_stream_writer_result_t){
      .status = (oai_memprof_stream_writer_status_t)status,
      .runtime_status = (oai_memprof_core_status_t)runtime_status,
      .clock_status = (oai_memprof_clock_status_t)clock_status,
      .prefooter_closed = closed != 0,
      .system_errno = (int32_t)system_errno,
      .runtime_snapshot =
          {
              .process_generation = get_u64(wire + 24U),
              .reservations = get_u64(wire + 32U),
              .ready_threads = get_u64(wire + 40U),
              .registration_capacity_failures = get_u64(wire + 48U),
              .recursion_bypasses = get_u64(wire + 56U),
              .ring_full_losses = get_u64(wire + 64U),
              .admitted_transactions = get_u64(wire + 72U),
              .completed_transactions = get_u64(wire + 80U),
              .emitted_events = get_u64(wire + 88U),
              .requested_bytes = get_u64(wire + 96U),
              .state = wire[104U],
              .mode_id = wire[105U],
          },
      .clock_info =
          {
              .counter_frequency_numerator = get_u64(wire + 112U),
              .counter_frequency_denominator = get_u64(wire + 120U),
              .architecture_id = get_u16(wire + 128U),
              .acquisition_source_id = get_u16(wire + 130U),
              .clock_kind = wire[132U],
          },
      .chunk_count = get_u64(wire + 264U),
      .record_count = get_u64(wire + 272U),
      .payload_bytes = get_u64(wire + 280U),
      .stream_bytes = get_u64(wire + 288U),
      .file_device = get_u64(wire + 296U),
      .file_inode = get_u64(wire + 304U),
  };
  decode_sample(&writer->seal_before_sample, wire + 136U);
  decode_sample(&writer->seal_after_sample, wire + 168U);
  decode_sample(&writer->drain_complete_sample, wire + 200U);
  decode_sample(&writer->final_sample, wire + 232U);
  return true;
}

static void encode_thread(uint8_t *wire, const oai_memprof_process_handoff_thread_v1_t *thread)
{
  put_u64(wire, thread->runtime.process_generation);
  put_u64(wire + 8U, thread->runtime.registration_ordinal);
  put_u64(wire + 16U, thread->runtime.thread_sequence);
  for (size_t api = 0; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api)
    put_u64(wire + HANDOFF_THREAD_API_ATTEMPTS_OFFSET + api * 8U, thread->runtime.api_attempts[api]);
  put_u64(wire + HANDOFF_THREAD_REQUESTED_BYTES_OFFSET, thread->runtime.requested_bytes);
  put_u64(wire + HANDOFF_THREAD_COMPLETED_TRANSACTIONS_OFFSET, thread->runtime.completed_transactions);
  put_u64(wire + HANDOFF_THREAD_RECURSION_BYPASSES_OFFSET, thread->runtime.recursion_bypasses);
  put_u64(wire + HANDOFF_THREAD_RING_FULL_LOSSES_OFFSET, thread->runtime.ring_full_losses);
  put_u64(wire + HANDOFF_THREAD_SIZE_UNKNOWNS_OFFSET, thread->runtime.size_unknowns);
  put_u64(wire + HANDOFF_THREAD_COUNTER_INVALIDS_OFFSET, thread->runtime.counter_invalids);
  put_u32(wire + HANDOFF_THREAD_INDEX_OFFSET, thread->runtime.thread_index);
  put_u32(wire + HANDOFF_THREAD_DIAGNOSTIC_SATURATED_MASK_OFFSET, thread->diagnostic_saturated_mask);
  put_u64(wire + HANDOFF_THREAD_SAMPLE_INSERTION_FAILURES_OFFSET, thread->runtime.sample_insertion_failures);
  put_u64(wire + HANDOFF_THREAD_SAMPLE_LOOKUP_FAILURES_OFFSET, thread->runtime.sample_lookup_failures);
  put_u64(wire + HANDOFF_THREAD_SAMPLE_PROBE_EXHAUSTIONS_OFFSET, thread->runtime.sample_probe_exhaustions);
  put_u64(wire + HANDOFF_THREAD_SAMPLE_PAIRING_FAILURES_OFFSET, thread->runtime.sample_pairing_failures);
  for (size_t diagnostic = 0; diagnostic < OAI_MEMPROF_PROCESS_HANDOFF_V1_DIAGNOSTIC_COUNT; ++diagnostic)
    put_u64(wire + HANDOFF_THREAD_RUNTIME_BYTES + diagnostic * 8U, thread->diagnostic_values[diagnostic]);
}

static bool decode_thread(oai_memprof_process_handoff_thread_v1_t *thread, const uint8_t *wire)
{
  *thread = (oai_memprof_process_handoff_thread_v1_t){
      .runtime =
          {
              .process_generation = get_u64(wire),
              .registration_ordinal = get_u64(wire + 8U),
              .thread_sequence = get_u64(wire + 16U),
              .requested_bytes = get_u64(wire + HANDOFF_THREAD_REQUESTED_BYTES_OFFSET),
              .completed_transactions = get_u64(wire + HANDOFF_THREAD_COMPLETED_TRANSACTIONS_OFFSET),
              .recursion_bypasses = get_u64(wire + HANDOFF_THREAD_RECURSION_BYPASSES_OFFSET),
              .ring_full_losses = get_u64(wire + HANDOFF_THREAD_RING_FULL_LOSSES_OFFSET),
              .size_unknowns = get_u64(wire + HANDOFF_THREAD_SIZE_UNKNOWNS_OFFSET),
              .counter_invalids = get_u64(wire + HANDOFF_THREAD_COUNTER_INVALIDS_OFFSET),
              .thread_index = get_u32(wire + HANDOFF_THREAD_INDEX_OFFSET),
              .sample_insertion_failures = get_u64(wire + HANDOFF_THREAD_SAMPLE_INSERTION_FAILURES_OFFSET),
              .sample_lookup_failures = get_u64(wire + HANDOFF_THREAD_SAMPLE_LOOKUP_FAILURES_OFFSET),
              .sample_probe_exhaustions = get_u64(wire + HANDOFF_THREAD_SAMPLE_PROBE_EXHAUSTIONS_OFFSET),
              .sample_pairing_failures = get_u64(wire + HANDOFF_THREAD_SAMPLE_PAIRING_FAILURES_OFFSET),
          },
      .diagnostic_saturated_mask = get_u32(wire + HANDOFF_THREAD_DIAGNOSTIC_SATURATED_MASK_OFFSET),
  };
  for (size_t api = 0; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api)
    thread->runtime.api_attempts[api] = get_u64(wire + HANDOFF_THREAD_API_ATTEMPTS_OFFSET + api * 8U);
  for (size_t diagnostic = 0; diagnostic < OAI_MEMPROF_PROCESS_HANDOFF_V1_DIAGNOSTIC_COUNT; ++diagnostic)
    thread->diagnostic_values[diagnostic] = get_u64(wire + HANDOFF_THREAD_RUNTIME_BYTES + diagnostic * 8U);
  thread->runtime.diagnostic_saturated_mask = thread->diagnostic_saturated_mask;
  return !(thread->diagnostic_saturated_mask & ~UINT32_C(0x3ff));
}

oai_memprof_process_handoff_status_t oai_memprof_process_handoff_v1_encode(const oai_memprof_process_handoff_v1_t *handoff,
                                                                           uint8_t *wire,
                                                                           size_t wire_size)
{
  if (handoff == NULL || wire == NULL)
    return OAI_MEMPROF_PROCESS_HANDOFF_INVALID_ARGUMENT;
  size_t expected_size = 0;
  oai_memprof_process_handoff_status_t status =
      oai_memprof_process_handoff_v1_size(handoff->bootstrap_size, handoff->maps_size, handoff->thread_count, &expected_size);
  if (status != OAI_MEMPROF_PROCESS_HANDOFF_OK)
    return status;
  if (wire_size != expected_size)
    return OAI_MEMPROF_PROCESS_HANDOFF_WRONG_SIZE;
  if (!valid_handoff(handoff))
    return OAI_MEMPROF_PROCESS_HANDOFF_INVALID_CONFIGURATION;

  uint8_t opening_wire[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE];
  uint8_t bootstrap_sha256[32];
  uint8_t maps_sha256[32];
  uint8_t opening_sha256[32];
  if (oai_memprof_container_v1_opening_header_encode(&handoff->opening_header, opening_wire, sizeof(opening_wire))
          != OAI_MEMPROF_CONTAINER_V1_OK
      || oai_memprof_container_v1_sha256(handoff->bootstrap_bytes, handoff->bootstrap_size, bootstrap_sha256)
             != OAI_MEMPROF_CONTAINER_V1_OK
      || oai_memprof_container_v1_sha256(handoff->maps_bytes, handoff->maps_size, maps_sha256) != OAI_MEMPROF_CONTAINER_V1_OK
      || oai_memprof_container_v1_sha256(opening_wire, sizeof(opening_wire), opening_sha256) != OAI_MEMPROF_CONTAINER_V1_OK)
    return OAI_MEMPROF_PROCESS_HANDOFF_CODEC_ERROR;
  if (memcmp(bootstrap_sha256, handoff->opening_header.configuration_instance_sha256, 32U) != 0
      || !supplied_digest_matches(handoff->bootstrap_sha256, bootstrap_sha256)
      || !supplied_digest_matches(handoff->maps_sha256, maps_sha256)
      || !supplied_digest_matches(handoff->opening_header_sha256, opening_sha256))
    return OAI_MEMPROF_PROCESS_HANDOFF_BAD_CHECKSUM;

  void *temporary = mmap(NULL, wire_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (temporary == MAP_FAILED)
    return OAI_MEMPROF_PROCESS_HANDOFF_NO_MEMORY;
  uint8_t *encoded = temporary;
  memcpy(encoded, handoff_magic, sizeof(handoff_magic));
  put_u16(encoded + 16U, HANDOFF_MAJOR);
  put_u16(encoded + 18U, HANDOFF_MINOR);
  put_u32(encoded + 20U, OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES);
  put_u64(encoded + 24U, wire_size);
  const size_t bootstrap_offset = OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES;
  const size_t maps_offset = bootstrap_offset + handoff->bootstrap_size;
  const size_t threads_offset = maps_offset + handoff->maps_size;
  put_u64(encoded + 32U, bootstrap_offset);
  put_u64(encoded + 40U, handoff->bootstrap_size);
  put_u64(encoded + 48U, maps_offset);
  put_u64(encoded + 56U, handoff->maps_size);
  put_u64(encoded + 64U, threads_offset);
  put_u32(encoded + 72U, (uint32_t)handoff->thread_count);
  put_u32(encoded + 76U, OAI_MEMPROF_PROCESS_HANDOFF_V1_THREAD_BYTES);
  put_u64(encoded + 80U, handoff->unregistered_active_thread_failures);
  put_u64(encoded + 88U, handoff->diagnostic_saturation_transitions);
  put_u32(encoded + 96U, handoff->registration_diagnostic_saturated_mask);
  put_u32(encoded + 100U, handoff->ring_records);
  put_u64(encoded + 104U, handoff->writer_io_or_finalization_failures);
  put_u32(encoded + 112U, handoff->flush_records);
  put_u16(encoded + 116U, handoff->realloc_zero_policy_id);
  put_u64(encoded + 120U, handoff->flush_interval_ns);
  put_u64(encoded + HANDOFF_SAMPLING_CONTROL_OFFSET, handoff->writer.runtime_snapshot.table_entries);
  put_u64(encoded + HANDOFF_SAMPLING_CONTROL_OFFSET + 8U, handoff->writer.runtime_snapshot.sample_seed);
  put_u64(encoded + HANDOFF_SAMPLING_CONTROL_OFFSET + 16U, handoff->writer.runtime_snapshot.sample_threshold);
  put_u32(encoded + HANDOFF_SAMPLING_CONTROL_OFFSET + 24U, handoff->writer.runtime_snapshot.table_probes);
  put_u32(encoded + HANDOFF_SAMPLING_CONTROL_OFFSET + 28U, handoff->writer.runtime_snapshot.table_shards);
  memcpy(encoded + HANDOFF_OPENING_OFFSET, opening_wire, sizeof(opening_wire));
  encode_writer(encoded + HANDOFF_WRITER_OFFSET, &handoff->writer);
  memcpy(encoded + HANDOFF_BOOTSTRAP_SHA256_OFFSET, bootstrap_sha256, 32U);
  memcpy(encoded + HANDOFF_MAPS_SHA256_OFFSET, maps_sha256, 32U);
  memcpy(encoded + HANDOFF_OPENING_SHA256_OFFSET, opening_sha256, 32U);
  encode_sample(encoded + HANDOFF_OPENING_SAMPLE_OFFSET, &handoff->opening_sample);
  memcpy(encoded + HANDOFF_PREFIX_SHA256_OFFSET, handoff->prefix_sha256, 32U);
  memcpy(encoded + bootstrap_offset, handoff->bootstrap_bytes, handoff->bootstrap_size);
  memcpy(encoded + maps_offset, handoff->maps_bytes, handoff->maps_size);
  for (size_t index = 0; index < handoff->thread_count; ++index)
    encode_thread(encoded + threads_offset + index * OAI_MEMPROF_PROCESS_HANDOFF_V1_THREAD_BYTES, &handoff->threads[index]);
  uint8_t self_sha256[32];
  if (oai_memprof_container_v1_sha256(encoded, wire_size - 32U, self_sha256) != OAI_MEMPROF_CONTAINER_V1_OK
      || !supplied_digest_matches(handoff->handoff_sha256, self_sha256)) {
    (void)munmap(temporary, wire_size);
    return OAI_MEMPROF_PROCESS_HANDOFF_BAD_CHECKSUM;
  }
  memcpy(encoded + wire_size - 32U, self_sha256, 32U);
  memcpy(wire, encoded, wire_size);
  (void)munmap(temporary, wire_size);
  return OAI_MEMPROF_PROCESS_HANDOFF_OK;
}

oai_memprof_process_handoff_status_t oai_memprof_process_handoff_v1_decode(oai_memprof_process_handoff_v1_t *handoff,
                                                                           oai_memprof_process_handoff_thread_v1_t *threads,
                                                                           size_t thread_capacity,
                                                                           const uint8_t *wire,
                                                                           size_t wire_size)
{
  if (handoff == NULL || wire == NULL || (thread_capacity != 0 && threads == NULL))
    return OAI_MEMPROF_PROCESS_HANDOFF_INVALID_ARGUMENT;
  if (wire_size < OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES + HANDOFF_SELF_SHA256_BYTES)
    return OAI_MEMPROF_PROCESS_HANDOFF_WRONG_SIZE;
  if (wire_size > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_WIRE_BYTES)
    return OAI_MEMPROF_PROCESS_HANDOFF_WRONG_SIZE;
  if (memcmp(wire, handoff_magic, sizeof(handoff_magic)) != 0)
    return OAI_MEMPROF_PROCESS_HANDOFF_BAD_MAGIC;
  uint8_t self_sha256[32];
  if (oai_memprof_container_v1_sha256(wire, wire_size - 32U, self_sha256) != OAI_MEMPROF_CONTAINER_V1_OK
      || memcmp(self_sha256, wire + wire_size - 32U, 32U) != 0)
    return OAI_MEMPROF_PROCESS_HANDOFF_BAD_CHECKSUM;
  if (get_u16(wire + 16U) != HANDOFF_MAJOR || get_u16(wire + 18U) != HANDOFF_MINOR)
    return OAI_MEMPROF_PROCESS_HANDOFF_UNSUPPORTED_VERSION;
  if (get_u32(wire + 20U) != OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES || get_u64(wire + 24U) != wire_size
      || get_u32(wire + 76U) != OAI_MEMPROF_PROCESS_HANDOFF_V1_THREAD_BYTES)
    return OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION;
  for (size_t index = 118U; index < 120U; ++index)
    if (wire[index] != 0)
      return OAI_MEMPROF_PROCESS_HANDOFF_NONZERO_RESERVED;
  for (size_t index = HANDOFF_RESERVED_FINAL_OFFSET; index < OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES; ++index)
    if (wire[index] != 0)
      return OAI_MEMPROF_PROCESS_HANDOFF_NONZERO_RESERVED;

  const uint64_t bootstrap_offset = get_u64(wire + 32U);
  const uint64_t bootstrap_size = get_u64(wire + 40U);
  const uint64_t maps_offset = get_u64(wire + 48U);
  const uint64_t maps_size = get_u64(wire + 56U);
  const uint64_t threads_offset = get_u64(wire + 64U);
  const uint32_t thread_count = get_u32(wire + 72U);
  size_t expected_size = 0;
  oai_memprof_process_handoff_status_t status =
      oai_memprof_process_handoff_v1_size((size_t)bootstrap_size, (size_t)maps_size, thread_count, &expected_size);
  if (status != OAI_MEMPROF_PROCESS_HANDOFF_OK)
    return status;
  if (expected_size != wire_size || bootstrap_offset != OAI_MEMPROF_PROCESS_HANDOFF_V1_HEADER_BYTES
      || maps_offset != bootstrap_offset + bootstrap_size || threads_offset != maps_offset + maps_size)
    return OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION;
  if (thread_count > thread_capacity)
    return OAI_MEMPROF_PROCESS_HANDOFF_WRONG_SIZE;

  uint8_t bootstrap_sha256[32];
  uint8_t maps_sha256[32];
  uint8_t opening_sha256[32];
  if (oai_memprof_container_v1_sha256(wire + bootstrap_offset, (size_t)bootstrap_size, bootstrap_sha256)
          != OAI_MEMPROF_CONTAINER_V1_OK
      || oai_memprof_container_v1_sha256(wire + maps_offset, (size_t)maps_size, maps_sha256) != OAI_MEMPROF_CONTAINER_V1_OK
      || oai_memprof_container_v1_sha256(wire + HANDOFF_OPENING_OFFSET,
                                         OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE,
                                         opening_sha256)
             != OAI_MEMPROF_CONTAINER_V1_OK
      || memcmp(bootstrap_sha256, wire + HANDOFF_BOOTSTRAP_SHA256_OFFSET, 32U) != 0
      || memcmp(maps_sha256, wire + HANDOFF_MAPS_SHA256_OFFSET, 32U) != 0
      || memcmp(opening_sha256, wire + HANDOFF_OPENING_SHA256_OFFSET, 32U) != 0)
    return OAI_MEMPROF_PROCESS_HANDOFF_BAD_CHECKSUM;

  oai_memprof_process_handoff_v1_t decoded = {
      .bootstrap_bytes = wire + bootstrap_offset,
      .bootstrap_size = (size_t)bootstrap_size,
      .maps_bytes = wire + maps_offset,
      .maps_size = (size_t)maps_size,
      .thread_count = thread_count,
      .ring_records = get_u32(wire + 100U),
      .flush_records = get_u32(wire + 112U),
      .flush_interval_ns = get_u64(wire + 120U),
      .realloc_zero_policy_id = get_u16(wire + 116U),
      .unregistered_active_thread_failures = get_u64(wire + 80U),
      .diagnostic_saturation_transitions = get_u64(wire + 88U),
      .registration_diagnostic_saturated_mask = get_u32(wire + 96U),
      .writer_io_or_finalization_failures = get_u64(wire + 104U),
  };
  if (oai_memprof_container_v1_opening_header_decode(&decoded.opening_header,
                                                     wire + HANDOFF_OPENING_OFFSET,
                                                     OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE)
      != OAI_MEMPROF_CONTAINER_V1_OK)
    return OAI_MEMPROF_PROCESS_HANDOFF_CODEC_ERROR;
  decode_sample(&decoded.opening_sample, wire + HANDOFF_OPENING_SAMPLE_OFFSET);
  if (!decode_writer(&decoded.writer, wire + HANDOFF_WRITER_OFFSET))
    return OAI_MEMPROF_PROCESS_HANDOFF_NONZERO_RESERVED;
  decoded.writer.runtime_snapshot.unregistered_active_thread_failures = decoded.unregistered_active_thread_failures;
  decoded.writer.runtime_snapshot.diagnostic_saturation_transitions = decoded.diagnostic_saturation_transitions;
  decoded.writer.runtime_snapshot.registration_diagnostic_saturated_mask = decoded.registration_diagnostic_saturated_mask;
  decoded.writer.runtime_snapshot.table_entries = get_u64(wire + HANDOFF_SAMPLING_CONTROL_OFFSET);
  decoded.writer.runtime_snapshot.sample_seed = get_u64(wire + HANDOFF_SAMPLING_CONTROL_OFFSET + 8U);
  decoded.writer.runtime_snapshot.sample_threshold = get_u64(wire + HANDOFF_SAMPLING_CONTROL_OFFSET + 16U);
  decoded.writer.runtime_snapshot.table_probes = get_u32(wire + HANDOFF_SAMPLING_CONTROL_OFFSET + 24U);
  decoded.writer.runtime_snapshot.table_shards = get_u32(wire + HANDOFF_SAMPLING_CONTROL_OFFSET + 28U);
  void *temporary =
      thread_count == 0
          ? NULL
          : mmap(NULL, (size_t)thread_count * sizeof(*threads), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (thread_count != 0 && temporary == MAP_FAILED)
    return OAI_MEMPROF_PROCESS_HANDOFF_NO_MEMORY;
  oai_memprof_process_handoff_thread_v1_t *decoded_threads = temporary;
  for (size_t index = 0; index < thread_count; ++index)
    if (!decode_thread(&decoded_threads[index], wire + threads_offset + index * OAI_MEMPROF_PROCESS_HANDOFF_V1_THREAD_BYTES)) {
      (void)munmap(temporary, (size_t)thread_count * sizeof(*threads));
      return OAI_MEMPROF_PROCESS_HANDOFF_NONZERO_RESERVED;
    }
  decoded.threads = decoded_threads;
  memcpy(decoded.bootstrap_sha256, bootstrap_sha256, 32U);
  memcpy(decoded.maps_sha256, maps_sha256, 32U);
  memcpy(decoded.opening_header_sha256, opening_sha256, 32U);
  memcpy(decoded.prefix_sha256, wire + HANDOFF_PREFIX_SHA256_OFFSET, 32U);
  memcpy(decoded.handoff_sha256, self_sha256, 32U);
  if (memcmp(bootstrap_sha256, decoded.opening_header.configuration_instance_sha256, 32U) != 0 || !valid_handoff(&decoded)) {
    if (thread_count != 0)
      (void)munmap(temporary, (size_t)thread_count * sizeof(*threads));
    return OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION;
  }
  if (thread_count != 0) {
    memcpy(threads, decoded_threads, (size_t)thread_count * sizeof(*threads));
    (void)munmap(temporary, (size_t)thread_count * sizeof(*threads));
  }
  decoded.threads = threads;
  *handoff = decoded;
  return OAI_MEMPROF_PROCESS_HANDOFF_OK;
}
