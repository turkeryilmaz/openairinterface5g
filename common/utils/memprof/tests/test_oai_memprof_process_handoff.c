/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "common/utils/memprof/oai_memprof_process_handoff.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define LITERAL_BYTES 1745U
#define THREAD_OFFSET 1265U
#define THREAD_RUNTIME_BYTES 368U
#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      exit(EXIT_FAILURE);                                                             \
    }                                                                                 \
  } while (0)

static uint8_t nibble(int value)
{
  if (value >= '0' && value <= '9')
    return (uint8_t)(value - '0');
  if (value >= 'a' && value <= 'f')
    return (uint8_t)(value - 'a' + 10);
  CHECK(false);
  return 0;
}

static void read_literal(const char *path, uint8_t wire[LITERAL_BYTES])
{
  FILE *file = fopen(path, "rb");
  CHECK(file != NULL);
  for (size_t index = 0; index < LITERAL_BYTES; ++index) {
    const int high = fgetc(file);
    const int low = fgetc(file);
    CHECK(high != EOF && low != EOF);
    wire[index] = (uint8_t)((nibble(high) << 4) | nibble(low));
  }
  CHECK(fgetc(file) == '\n');
  CHECK(fgetc(file) == EOF);
  CHECK(fclose(file) == 0);
}

static void refresh_self_hash(uint8_t wire[LITERAL_BYTES])
{
  uint8_t digest[32];
  CHECK(oai_memprof_container_v1_sha256(wire, LITERAL_BYTES - 32U, digest) == OAI_MEMPROF_CONTAINER_V1_OK);
  memcpy(wire + LITERAL_BYTES - 32U, digest, sizeof(digest));
}

static void put_u32_test(uint8_t *wire, uint32_t value)
{
  for (unsigned index = 0; index < 4U; ++index)
    wire[index] = (uint8_t)(value >> (index * 8U));
}

static void put_u64_test(uint8_t *wire, uint64_t value)
{
  for (unsigned index = 0; index < 8U; ++index)
    wire[index] = (uint8_t)(value >> (index * 8U));
}

static void assert_unchanged(const uint8_t *left, const uint8_t *right, size_t size)
{
  CHECK(memcmp(left, right, size) == 0);
}

int main(int argc, char **argv)
{
  CHECK(argc == 2);
  uint8_t literal[LITERAL_BYTES];
  read_literal(argv[1], literal);
  CHECK(literal[16] == 1 && literal[17] == 0 && literal[18] == 5 && literal[19] == 0);

  oai_memprof_process_handoff_thread_v1_t thread;
  memset(&thread, 0xa5, sizeof(thread));
  const oai_memprof_process_handoff_thread_v1_t thread_sentinel = thread;
  oai_memprof_process_handoff_v1_t handoff;
  memset(&handoff, 0x5a, sizeof(handoff));
  const oai_memprof_process_handoff_v1_t handoff_sentinel = handoff;

  const size_t too_large_wire_size = (size_t)OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_WIRE_BYTES + 1U;
  const uint8_t *too_large_wire = mmap(NULL, too_large_wire_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(too_large_wire != MAP_FAILED);
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, too_large_wire, too_large_wire_size)
        == OAI_MEMPROF_PROCESS_HANDOFF_WRONG_SIZE);
  assert_unchanged((const uint8_t *)&handoff, (const uint8_t *)&handoff_sentinel, sizeof(handoff));
  assert_unchanged((const uint8_t *)&thread, (const uint8_t *)&thread_sentinel, sizeof(thread));
  CHECK(munmap((void *)too_large_wire, too_large_wire_size) == 0);

  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, literal, sizeof(literal)) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
  CHECK(handoff.opening_header.process_generation == 7);
  CHECK(handoff.opening_sample.counter == handoff.opening_header.start_counter);
  CHECK(handoff.opening_sample.monotonic_raw_before_ns == 999999950);
  CHECK(handoff.opening_sample.monotonic_raw_after_ns == 1000000050);
  CHECK(handoff.opening_sample.realtime_unix_ns == handoff.opening_header.start_realtime_unix_ns);
  CHECK(handoff.writer.record_count == 2 && handoff.writer.chunk_count == 1);
  CHECK(handoff.writer.payload_bytes == 192 && handoff.writer.stream_bytes == 736);
  CHECK(handoff.writer.runtime_snapshot.admitted_transactions == 2);
  CHECK(handoff.writer.runtime_snapshot.completed_transactions == 2);
  CHECK(handoff.writer.runtime_snapshot.emitted_events == 2);
  CHECK(handoff.writer.runtime_snapshot.requested_bytes == 96);
  CHECK(handoff.writer.runtime_snapshot.table_entries == 0 && handoff.writer.runtime_snapshot.sample_seed == 0
        && handoff.writer.runtime_snapshot.sample_threshold == 0 && handoff.writer.runtime_snapshot.table_probes == 0
        && handoff.writer.runtime_snapshot.table_shards == 0);
  CHECK(handoff.ring_records == 8 && handoff.flush_records == 2);
  CHECK(handoff.flush_interval_ns == UINT64_C(100000000) && handoff.realloc_zero_policy_id == 1);
  CHECK(handoff.thread_count == 1 && thread.runtime.thread_index == 1);
  CHECK(thread.runtime.api_attempts[0] == 1 && thread.runtime.api_attempts[1] == 1);
  for (size_t api = 2; api < OAI_MEMPROF_CORE_API_SLOT_COUNT; ++api)
    CHECK(thread.runtime.api_attempts[api] == 0);
  CHECK(thread.runtime.completed_transactions == 2 && thread.runtime.requested_bytes == 96);
  CHECK(thread.runtime.size_unknowns == 0 && thread.runtime.counter_invalids == 0 && thread.runtime.sample_insertion_failures == 0
        && thread.runtime.sample_lookup_failures == 0 && thread.runtime.sample_probe_exhaustions == 0
        && thread.runtime.sample_pairing_failures == 0);
  CHECK(handoff.bootstrap_size == 64 && handoff.maps_size == 49);
  CHECK(memcmp(handoff.prefix_sha256, literal + 1080U, 32U) == 0);
  CHECK(memcmp(handoff.handoff_sha256, literal + LITERAL_BYTES - 32U, 32U) == 0);

  uint8_t encoded[LITERAL_BYTES];
  uint8_t encoded_sentinel[sizeof(encoded)];
  memset(encoded, 0x3c, sizeof(encoded));
  handoff.handoff_sha256[0] = 0;
  memset(handoff.handoff_sha256, 0, sizeof(handoff.handoff_sha256));
  CHECK(oai_memprof_process_handoff_v1_encode(&handoff, encoded, sizeof(encoded)) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
  CHECK(memcmp(encoded, literal, sizeof(encoded)) == 0);

  for (size_t api = 4; api < OAI_MEMPROF_CORE_ADMITTED_API_COUNT; ++api) {
    thread.runtime.api_attempts[1] = 0;
    thread.runtime.api_attempts[api] = 1;
    memset(handoff.handoff_sha256, 0, sizeof(handoff.handoff_sha256));
    CHECK(oai_memprof_process_handoff_v1_encode(&handoff, encoded, sizeof(encoded)) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
    thread.runtime.api_attempts[api] = 0;
    thread.runtime.api_attempts[1] = 1;
  }

  thread.runtime.api_attempts[1] = 0;
  thread.runtime.api_attempts[OAI_MEMPROF_CORE_ADMITTED_API_COUNT] = 1;
  memset(encoded, 0x3c, sizeof(encoded));
  memcpy(encoded_sentinel, encoded, sizeof(encoded));
  CHECK(oai_memprof_process_handoff_v1_encode(&handoff, encoded, sizeof(encoded))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_CONFIGURATION);
  assert_unchanged(encoded, encoded_sentinel, sizeof(encoded));
  thread.runtime.api_attempts[OAI_MEMPROF_CORE_ADMITTED_API_COUNT] = 0;
  thread.runtime.api_attempts[1] = 1;
  memset(handoff.handoff_sha256, 0xff, sizeof(handoff.handoff_sha256));
  memset(encoded, 0x3c, sizeof(encoded));
  memcpy(encoded_sentinel, encoded, sizeof(encoded));
  CHECK(oai_memprof_process_handoff_v1_encode(&handoff, encoded, sizeof(encoded)) == OAI_MEMPROF_PROCESS_HANDOFF_BAD_CHECKSUM);
  assert_unchanged(encoded, encoded_sentinel, sizeof(encoded));

  uint8_t mutant[LITERAL_BYTES];
  memcpy(mutant, literal, sizeof(mutant));
  mutant[1300] ^= 1U;
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_BAD_CHECKSUM);
  assert_unchanged((const uint8_t *)&handoff, (const uint8_t *)&handoff_sentinel, sizeof(handoff));
  assert_unchanged((const uint8_t *)&thread, (const uint8_t *)&thread_sentinel, sizeof(thread));

  memcpy(mutant, literal, sizeof(mutant));
  mutant[18] = 4;
  mutant[19] = 0;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_UNSUPPORTED_VERSION);
  assert_unchanged((const uint8_t *)&handoff, (const uint8_t *)&handoff_sentinel, sizeof(handoff));
  assert_unchanged((const uint8_t *)&thread, (const uint8_t *)&thread_sentinel, sizeof(thread));

  for (size_t api = 4; api < OAI_MEMPROF_CORE_ADMITTED_API_COUNT; ++api) {
    memcpy(mutant, literal, sizeof(mutant));
    put_u64_test(mutant + THREAD_OFFSET + 24U + 1U * 8U, UINT64_C(0));
    put_u64_test(mutant + THREAD_OFFSET + 24U + api * 8U, UINT64_C(1));
    refresh_self_hash(mutant);
    CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant)) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
    CHECK(thread.runtime.api_attempts[1] == 0 && thread.runtime.api_attempts[api] == 1);
  }

  memcpy(mutant, literal, sizeof(mutant));
  put_u64_test(mutant + THREAD_OFFSET + 24U + 1U * 8U, UINT64_C(0));
  put_u64_test(mutant + THREAD_OFFSET + 24U + OAI_MEMPROF_CORE_ADMITTED_API_COUNT * 8U, UINT64_C(1));
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);
  assert_unchanged((const uint8_t *)&handoff, (const uint8_t *)&handoff_sentinel, sizeof(handoff));
  assert_unchanged((const uint8_t *)&thread, (const uint8_t *)&thread_sentinel, sizeof(thread));

  memcpy(mutant, literal, sizeof(mutant));
  mutant[118] = 1;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_NONZERO_RESERVED);
  assert_unchanged((const uint8_t *)&handoff, (const uint8_t *)&handoff_sentinel, sizeof(handoff));
  assert_unchanged((const uint8_t *)&thread, (const uint8_t *)&thread_sentinel, sizeof(thread));

  const struct {
    size_t offset;
    uint8_t value;
  } invalid_runtime_config[] = {
      {100U, 3U},
      {112U, 0U},
      {116U, 3U},
  };
  for (size_t index = 0; index < sizeof(invalid_runtime_config) / sizeof(invalid_runtime_config[0]); ++index) {
    memcpy(mutant, literal, sizeof(mutant));
    mutant[invalid_runtime_config[index].offset] = invalid_runtime_config[index].value;
    refresh_self_hash(mutant);
    handoff = handoff_sentinel;
    thread = thread_sentinel;
    CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
          == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);
    assert_unchanged((const uint8_t *)&handoff, (const uint8_t *)&handoff_sentinel, sizeof(handoff));
    assert_unchanged((const uint8_t *)&thread, (const uint8_t *)&thread_sentinel, sizeof(thread));
  }

  memcpy(mutant, literal, sizeof(mutant));
  mutant[640U + 105U] = OAI_MEMPROF_CORE_SAMPLED;
  put_u64_test(mutant + 1112U, UINT64_C(64));
  put_u64_test(mutant + 1120U, UINT64_C(0x0123456789abcdef));
  put_u64_test(mutant + 1128U, UINT64_MAX);
  put_u32_test(mutant + 1136U, UINT32_C(8));
  put_u32_test(mutant + 1140U, UINT32_C(64));
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant)) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
  CHECK(handoff.writer.runtime_snapshot.mode_id == OAI_MEMPROF_CORE_SAMPLED);
  CHECK(handoff.writer.runtime_snapshot.table_entries == 64
        && handoff.writer.runtime_snapshot.sample_seed == UINT64_C(0x0123456789abcdef)
        && handoff.writer.runtime_snapshot.sample_threshold == UINT64_MAX && handoff.writer.runtime_snapshot.table_probes == 8
        && handoff.writer.runtime_snapshot.table_shards == 64);

  uint8_t sampled[LITERAL_BYTES];
  memcpy(sampled, mutant, sizeof(sampled));
  put_u64_test(sampled + THREAD_OFFSET + 336U, UINT64_C(1));
  put_u64_test(sampled + THREAD_OFFSET + THREAD_RUNTIME_BYTES + 5U * 8U, UINT64_C(1));
  refresh_self_hash(sampled);
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, sampled, sizeof(sampled)) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
  CHECK(thread.runtime.sample_insertion_failures == 1
        && thread.diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_INSERTION] == 1);

  put_u64_test(sampled + THREAD_OFFSET + THREAD_RUNTIME_BYTES + 5U * 8U, UINT64_C(2));
  refresh_self_hash(sampled);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, sampled, sizeof(sampled))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  memcpy(sampled, mutant, sizeof(sampled));
  put_u32_test(sampled + 1140U, UINT32_C(32));
  refresh_self_hash(sampled);
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, sampled, sizeof(sampled))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  memcpy(sampled, mutant, sizeof(sampled));
  put_u64_test(sampled + 1128U, UINT64_C(0));
  refresh_self_hash(sampled);
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, sampled, sizeof(sampled))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  memcpy(sampled, literal, sizeof(sampled));
  put_u64_test(sampled + 1112U, UINT64_C(64));
  refresh_self_hash(sampled);
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, sampled, sizeof(sampled))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  memcpy(mutant, literal, sizeof(mutant));
  mutant[1056] = 0x01;
  mutant[1057] = 0xca;
  mutant[1058] = 0x9a;
  mutant[1059] = 0x3b;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  memcpy(mutant, literal, sizeof(mutant));
  mutant[1144] = 1;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_NONZERO_RESERVED);

  memcpy(mutant, literal, sizeof(mutant));
  mutant[640 + 288] = 0xe1;
  mutant[640 + 289] = 0x02;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  memcpy(mutant, literal, sizeof(mutant));
  mutant[THREAD_OFFSET + THREAD_RUNTIME_BYTES] = 1;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  memcpy(mutant, literal, sizeof(mutant));
  mutant[640] = OAI_MEMPROF_STREAM_WRITER_IO_ERROR;
  mutant[640 + 288] = 0x00;
  mutant[640 + 289] = 0x03;
  mutant[104] = 1;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant)) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
  CHECK(handoff.writer.status == OAI_MEMPROF_STREAM_WRITER_IO_ERROR);
  CHECK(handoff.writer_io_or_finalization_failures == 1);
  CHECK(handoff.writer.stream_bytes == 768);

  memcpy(mutant, literal, sizeof(mutant));
  mutant[640] = OAI_MEMPROF_STREAM_WRITER_CLOCK_ERROR;
  mutant[640 + 8] = OAI_MEMPROF_CLOCK_BRACKET_TOO_WIDE;
  memset(mutant + 640 + 232, 0, 32);
  mutant[104] = 1;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant)) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
  CHECK(handoff.writer.clock_status == OAI_MEMPROF_CLOCK_BRACKET_TOO_WIDE);
  CHECK(handoff.writer.final_sample.counter == 0);

  memset(mutant + 640 + 136, 0, 32);
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  memcpy(mutant, literal, sizeof(mutant));
  mutant[640] = OAI_MEMPROF_STREAM_WRITER_IO_ERROR;
  mutant[640 + 288] = 0x00;
  mutant[640 + 289] = 0x03;
  refresh_self_hash(mutant);
  handoff = handoff_sentinel;
  thread = thread_sentinel;
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, mutant, sizeof(mutant))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_RELATION);

  size_t wire_size = 0;
  memset(handoff.prefix_sha256, 0, sizeof(handoff.prefix_sha256));
  memset(encoded, 0x3c, sizeof(encoded));
  memcpy(encoded_sentinel, encoded, sizeof(encoded));
  CHECK(oai_memprof_process_handoff_v1_encode(&handoff, encoded, sizeof(encoded))
        == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_CONFIGURATION);
  assert_unchanged(encoded, encoded_sentinel, sizeof(encoded));

  CHECK(oai_memprof_process_handoff_v1_size(64, 49, 1, &wire_size) == OAI_MEMPROF_PROCESS_HANDOFF_OK);
  CHECK(wire_size == LITERAL_BYTES);
  CHECK(oai_memprof_process_handoff_v1_size(0, 49, 1, &wire_size) == OAI_MEMPROF_PROCESS_HANDOFF_INVALID_CONFIGURATION);
  CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 0, literal, sizeof(literal))
        == OAI_MEMPROF_PROCESS_HANDOFF_WRONG_SIZE);

  puts("process handoff schema-v1 tests passed");
  return EXIT_SUCCESS;
}
