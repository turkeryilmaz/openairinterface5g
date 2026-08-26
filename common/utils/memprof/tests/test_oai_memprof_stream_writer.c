/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "common/utils/memprof/oai_memprof_stream_writer.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      exit(EXIT_FAILURE);                                                             \
    }                                                                                 \
  } while (0)

static size_t maximum_write = SIZE_MAX;
static uint64_t fail_after = UINT64_MAX;
static uint64_t observed_written;
static bool inject_join_failure;
static bool join_failure_injected;
static bool captured_join_thread_valid;
static pthread_t captured_join_thread;
static uint64_t fsync_after_join_failure;

ssize_t __real_write(int fd, const void *buffer, size_t size);
int __real_pthread_join(pthread_t thread, void **value_ptr);
int __real_fsync(int fd);

ssize_t __wrap_write(int fd, const void *buffer, size_t size)
{
  if (fd <= STDERR_FILENO)
    return __real_write(fd, buffer, size);
  if (observed_written >= fail_after) {
    errno = EIO;
    return -1;
  }
  size_t allowed = size < maximum_write ? size : maximum_write;
  if ((uint64_t)allowed > fail_after - observed_written)
    allowed = (size_t)(fail_after - observed_written);
  if (allowed == 0) {
    errno = EIO;
    return -1;
  }
  const ssize_t result = __real_write(fd, buffer, allowed);
  if (result > 0)
    observed_written += (uint64_t)result;
  return result;
}

int __wrap_pthread_join(pthread_t thread, void **value_ptr)
{
  if (inject_join_failure && !join_failure_injected) {
    captured_join_thread = thread;
    captured_join_thread_valid = true;
    join_failure_injected = true;
    return ESRCH;
  }
  return __real_pthread_join(thread, value_ptr);
}

int __wrap_fsync(int fd)
{
  if (join_failure_injected)
    ++fsync_after_join_failure;
  return __real_fsync(fd);
}

static uint8_t hex_digit(char value)
{
  if (value >= '0' && value <= '9')
    return (uint8_t)(value - '0');
  if (value >= 'a' && value <= 'f')
    return (uint8_t)(value - 'a' + 10);
  CHECK(false);
  return 0;
}

static void read_literal(const char *path, uint8_t output[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE])
{
  FILE *file = fopen(path, "rb");
  CHECK(file != NULL);
  for (size_t index = 0; index < OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE; ++index) {
    const int high = fgetc(file);
    const int low = fgetc(file);
    CHECK(high != EOF && low != EOF);
    output[index] = (uint8_t)((hex_digit((char)high) << 4) | hex_digit((char)low));
  }
  CHECK(fgetc(file) == '\n');
  CHECK(fgetc(file) == EOF);
  CHECK(fclose(file) == 0);
}

static oai_memprof_container_v1_opening_header_t opening_header(const oai_memprof_clock_info_v1_t *clock)
{
  oai_memprof_container_v1_opening_header_t opening = {
      .page_size_bytes = 4096,
      .scope_kind = OAI_MEMPROF_CONTAINER_V1_SCOPE_MEASUREMENT_INTERVAL,
      .role_kind = OAI_MEMPROF_CONTAINER_V1_ROLE_GNB,
      .clock_kind = clock->clock_kind,
      .calibration_kind = OAI_MEMPROF_CONTAINER_V1_CALIBRATION_EXACT_RATE,
      .process_generation = 7,
      .counter_frequency_numerator = clock->counter_frequency_numerator,
      .counter_frequency_denominator = clock->counter_frequency_denominator,
      .calibration_error_bound_ns = OAI_MEMPROF_STREAM_WRITER_CLOCK_BRACKET_NS,
      .start_counter = UINT64_C(0x1122334455667788),
      .start_monotonic_raw_ns = UINT64_C(0x0102030405060708),
      .start_realtime_unix_ns = UINT64_C(1700000000000000000),
      .pid = 12345,
      .configured_thread_capacity = 1,
      .run_uuid = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x46, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
      .process_uuid = {0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x49, 0x88, 0x87, 0x76, 0x65, 0x54, 0x43, 0x32, 0x21, 0x10},
      .source_object_kind = OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_COMMIT,
      .source_object_algorithm = OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_SHA1,
      .source_object_length = 20,
  };
  for (uint8_t index = 0; index < 20; ++index)
    opening.source_object_value[index] = index;
  for (uint8_t index = 0; index < 32; ++index) {
    opening.primary_binary_sha256[index] = (uint8_t)(0x20 + index);
    opening.schema_bundle_definition_sha256[index] = (uint8_t)(0x40 + index);
    opening.api_catalog_definition_sha256[index] = (uint8_t)(0x60 + index);
    opening.callsite_catalog_definition_sha256[index] = (uint8_t)(0x80 + index);
    opening.configuration_instance_sha256[index] = (uint8_t)(0xa0 + index);
    opening.primary_build_id_sha256[index] = (uint8_t)(0xc0 + index);
  }
  return opening;
}

static void publish_events(size_t count)
{
  for (size_t index = 0; index < count; ++index) {
    oai_memprof_core_ticket_t ticket = {0};
    CHECK(oai_memprof_active_runtime_begin_v1(1, 64 + index, true, &ticket));
    const oai_memprof_core_payload_t payload = {
        .address_after = UINT64_C(0x1000) + index * UINT64_C(0x100),
        .arg0 = 64 + index,
        .flags = (UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 11) | (UINT32_C(1) << 24),
        .result_code = (int32_t)index,
        .api_id = 1,
        .event_kind = 1,
    };
    CHECK(oai_memprof_active_runtime_end_v1(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
  }
}

static void validate_boundary_samples(const oai_memprof_stream_writer_result_t *result)
{
  CHECK(result->clock_status == OAI_MEMPROF_CLOCK_OK);
  CHECK(result->clock_info.counter_frequency_numerator != 0);
  CHECK(result->clock_info.counter_frequency_denominator != 0);
  const oai_memprof_clock_sample_v1_t samples[] = {
      result->seal_before_sample,
      result->seal_after_sample,
      result->drain_complete_sample,
      result->final_sample,
  };
  for (size_t index = 0; index < sizeof(samples) / sizeof(samples[0]); ++index) {
    CHECK(samples[index].counter != 0);
    CHECK(samples[index].monotonic_raw_before_ns <= samples[index].monotonic_raw_after_ns);
    CHECK(samples[index].monotonic_raw_after_ns - samples[index].monotonic_raw_before_ns
          <= OAI_MEMPROF_STREAM_WRITER_CLOCK_BRACKET_NS);
    CHECK(samples[index].realtime_unix_ns != 0);
    if (index != 0) {
      CHECK(samples[index - 1].counter < samples[index].counter);
      CHECK(samples[index - 1].monotonic_raw_after_ns <= samples[index].monotonic_raw_before_ns);
    }
  }
}

static void validate_zero_clock_sample(const oai_memprof_clock_sample_v1_t *sample)
{
  CHECK(sample->counter == 0);
  CHECK(sample->monotonic_raw_before_ns == 0);
  CHECK(sample->monotonic_raw_after_ns == 0);
  CHECK(sample->realtime_unix_ns == 0);
}

static void validate_zero_runtime_snapshot(const oai_memprof_core_snapshot_t *snapshot)
{
  CHECK(snapshot->process_generation == 0);
  CHECK(snapshot->reservations == 0);
  CHECK(snapshot->ready_threads == 0);
  CHECK(snapshot->registration_capacity_failures == 0);
  CHECK(snapshot->unregistered_active_thread_failures == 0);
  CHECK(snapshot->diagnostic_saturation_transitions == 0);
  CHECK(snapshot->registration_diagnostic_saturated_mask == 0);
  CHECK(snapshot->recursion_bypasses == 0);
  CHECK(snapshot->ring_full_losses == 0);
  CHECK(snapshot->admitted_transactions == 0);
  CHECK(snapshot->completed_transactions == 0);
  CHECK(snapshot->emitted_events == 0);
  CHECK(snapshot->requested_bytes == 0);
  CHECK(snapshot->table_entries == 0);
  CHECK(snapshot->sample_seed == 0);
  CHECK(snapshot->sample_threshold == 0);
  CHECK(snapshot->table_probes == 0);
  CHECK(snapshot->table_shards == 0);
  CHECK(snapshot->state == 0);
  CHECK(snapshot->mode_id == 0);
}

static void validate_join_failure_result(const oai_memprof_stream_writer_result_t *result)
{
  CHECK(result->status == OAI_MEMPROF_STREAM_WRITER_THREAD_ERROR);
  CHECK(result->runtime_status == OAI_MEMPROF_CORE_OK);
  validate_zero_runtime_snapshot(&result->runtime_snapshot);
  CHECK(result->clock_status == OAI_MEMPROF_CLOCK_OK);
  CHECK(result->clock_info.counter_frequency_numerator == 0);
  CHECK(result->clock_info.counter_frequency_denominator == 0);
  CHECK(result->clock_info.architecture_id == 0);
  CHECK(result->clock_info.acquisition_source_id == 0);
  CHECK(result->clock_info.clock_kind == 0);
  for (size_t index = 0; index < sizeof(result->clock_info.reserved_zero); ++index)
    CHECK(result->clock_info.reserved_zero[index] == 0);
  validate_zero_clock_sample(&result->seal_before_sample);
  validate_zero_clock_sample(&result->seal_after_sample);
  validate_zero_clock_sample(&result->drain_complete_sample);
  validate_zero_clock_sample(&result->final_sample);
  CHECK(result->chunk_count == 0);
  CHECK(result->record_count == 0);
  CHECK(result->payload_bytes == 0);
  CHECK(result->stream_bytes == 0);
  CHECK(result->file_device == 0);
  CHECK(result->file_inode == 0);
  CHECK(result->system_errno == ESRCH);
  CHECK(!result->prefooter_closed);
}

static uint8_t *read_file(const char *path, size_t *size)
{
  const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  CHECK(fd >= 0);
  struct stat status;
  CHECK(fstat(fd, &status) == 0 && S_ISREG(status.st_mode) && status.st_size >= 0);
  *size = (size_t)status.st_size;
  uint8_t *bytes = malloc(*size == 0 ? 1 : *size);
  CHECK(bytes != NULL);
  size_t offset = 0;
  while (offset != *size) {
    const ssize_t got = read(fd, bytes + offset, *size - offset);
    CHECK(got > 0);
    offset += (size_t)got;
  }
  CHECK(close(fd) == 0);
  return bytes;
}

static void validate_complete_stream(const uint8_t *bytes,
                                     size_t size,
                                     size_t event_count,
                                     size_t expected_chunks,
                                     const uint8_t opening_literal[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE])
{
  CHECK(memcmp(bytes, opening_literal, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE) == 0);
  size_t offset = OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE;
  size_t decoded_events = 0;
  uint64_t expected_sequence = 0;
  while (offset != size) {
    CHECK(size - offset >= OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE);
    const uint8_t *header_wire = bytes + offset;
    const uint32_t records = (uint32_t)header_wire[16] | ((uint32_t)header_wire[17] << 8) | ((uint32_t)header_wire[18] << 16)
                             | ((uint32_t)header_wire[19] << 24);
    const size_t payload_size = (size_t)records * OAI_MEMPROF_EVENT_V1_WIRE_SIZE;
    CHECK(size - offset - OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE >= payload_size);
    const uint8_t *payload = header_wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE;
    oai_memprof_container_v1_chunk_header_t header = {0};
    CHECK(oai_memprof_container_v1_chunk_header_decode(&header,
                                                       header_wire,
                                                       OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE,
                                                       payload,
                                                       payload_size)
          == OAI_MEMPROF_CONTAINER_V1_OK);
    CHECK(header.writer_chunk_sequence == expected_sequence++);
    for (uint32_t record = 0; record < records; ++record) {
      oai_memprof_event_v1_t event = {0};
      CHECK(oai_memprof_event_v1_decode(&event,
                                        payload + (size_t)record * OAI_MEMPROF_EVENT_V1_WIRE_SIZE,
                                        OAI_MEMPROF_EVENT_V1_WIRE_SIZE)
            == OAI_MEMPROF_WIRE_OK);
      CHECK(event.thread_sequence == decoded_events + 1);
      CHECK(event.thread_index == 1);
      CHECK(event.api_id == 1 && event.event_kind == 1);
      CHECK(event.address_after == UINT64_C(0x1000) + decoded_events * UINT64_C(0x100));
      ++decoded_events;
    }
    offset += OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE + payload_size;
  }
  CHECK(decoded_events == event_count);
  CHECK(expected_sequence == expected_chunks);
}

int main(int argc, char **argv)
{
  CHECK(argc == 3);
  const char *mode = argv[1];
  oai_memprof_clock_info_v1_t clock = {0};
  const oai_memprof_clock_status_t clock_status = oai_memprof_clock_info_v1(&clock);
  if (clock_status == OAI_MEMPROF_CLOCK_UNSUPPORTED) {
    puts("stream writer clock test skipped: no admitted architectural exact-rate source");
    return 77;
  }
  CHECK(clock_status == OAI_MEMPROF_CLOCK_OK);
  uint8_t opening_literal[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE];
  read_literal(argv[2], opening_literal);
  oai_memprof_container_v1_opening_header_t literal_header = {0};
  CHECK(oai_memprof_container_v1_opening_header_decode(&literal_header, opening_literal, sizeof(opening_literal))
        == OAI_MEMPROF_CONTAINER_V1_OK);

  char directory[] = "/tmp/oai-memprof-stream-writer.XXXXXX";
  CHECK(mkdtemp(directory) != NULL);
  char path[sizeof(directory) + 32];
  CHECK(snprintf(path, sizeof(path), "%s/pre-footer.bin", directory) > 0);

  size_t event_count = 3;
  uint32_t flush_records = 2;
  uint8_t core_mode = OAI_MEMPROF_CORE_EXACT_EVENTS;
  uint64_t flush_interval_ns = UINT64_C(100000000);
  if (strcmp(mode, "counters") == 0) {
    event_count = 0;
    core_mode = OAI_MEMPROF_CORE_COUNTERS;
  } else if (strcmp(mode, "timer") == 0) {
    event_count = 1;
    flush_records = 8;
    flush_interval_ns = UINT64_C(1000000);
  } else if (strcmp(mode, "short") == 0) {
    maximum_write = 7;
    event_count = 2;
  } else if (strcmp(mode, "failure") == 0) {
    fail_after = OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE + OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE + 48;
    event_count = 1;
    flush_records = 1;
  } else if (strcmp(mode, "join-failure") == 0) {
    event_count = 0;
  } else {
    CHECK(strcmp(mode, "positive") == 0);
  }

  const int directory_fd = open(directory, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  CHECK(directory_fd >= 0);
  const oai_memprof_stream_writer_config_t config = {
      .directory_fd = directory_fd,
      .file_name = "pre-footer.bin",
      .runtime =
          {
              .core = {.process_generation = 7, .max_threads = 1, .ring_records = 8, .mode_id = core_mode},
              .realloc_zero_policy_id = 1,
          },
      .opening_header = opening_header(&clock),
      .flush_records = flush_records,
      .flush_interval_ns = flush_interval_ns,
  };
  uint8_t expected_opening[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE];
  CHECK(oai_memprof_container_v1_opening_header_encode(&config.opening_header, expected_opening, sizeof(expected_opening))
        == OAI_MEMPROF_CONTAINER_V1_OK);
  oai_memprof_stream_writer_t *writer = (oai_memprof_stream_writer_t *)(uintptr_t)UINT64_C(0x1234);
  oai_memprof_stream_writer_config_t invalid = config;
  ++invalid.opening_header.counter_frequency_numerator;
  CHECK(oai_memprof_stream_writer_start_v1(&invalid, &writer) == OAI_MEMPROF_STREAM_WRITER_INVALID_CONFIGURATION);
  CHECK(writer == (oai_memprof_stream_writer_t *)(uintptr_t)UINT64_C(0x1234));
  invalid = config;
  invalid.file_name = "../escape";
  CHECK(oai_memprof_stream_writer_start_v1(&invalid, &writer) == OAI_MEMPROF_STREAM_WRITER_INVALID_CONFIGURATION);
  CHECK(writer == (oai_memprof_stream_writer_t *)(uintptr_t)UINT64_C(0x1234));
  const int existing = openat(directory_fd, "existing.bin", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
  CHECK(existing >= 0 && close(existing) == 0);
  invalid = config;
  invalid.file_name = "existing.bin";
  CHECK(oai_memprof_stream_writer_start_v1(&invalid, &writer) == OAI_MEMPROF_STREAM_WRITER_IO_ERROR);
  CHECK(writer == (oai_memprof_stream_writer_t *)(uintptr_t)UINT64_C(0x1234));
  CHECK(unlinkat(directory_fd, "existing.bin", 0) == 0);
  writer = NULL;
  CHECK(oai_memprof_stream_writer_start_v1(&config, &writer) == OAI_MEMPROF_STREAM_WRITER_OK);
  CHECK(writer != NULL);
  CHECK(close(directory_fd) == 0);
  publish_events(event_count);
  if (strcmp(mode, "timer") == 0) {
    struct timespec deadline = {.tv_sec = 0, .tv_nsec = 1000000L};
    bool flushed = false;
    for (unsigned attempt = 0; attempt < 1000 && !flushed; ++attempt) {
      struct stat observed;
      CHECK(stat(path, &observed) == 0);
      flushed = observed.st_size > OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE;
      if (!flushed)
        CHECK(nanosleep(&deadline, NULL) == 0);
    }
    CHECK(flushed);
  }

  if (strcmp(mode, "join-failure") == 0)
    inject_join_failure = true;
  oai_memprof_stream_writer_result_t result;
  memset(&result, 0xa5, sizeof(result));
  const oai_memprof_stream_writer_status_t finish = oai_memprof_stream_writer_finish_v1(writer, UINT64_C(100000000), &result);
  if (strcmp(mode, "join-failure") == 0) {
    CHECK(join_failure_injected && captured_join_thread_valid);
    CHECK(fsync_after_join_failure == 0);
    CHECK(finish == OAI_MEMPROF_STREAM_WRITER_THREAD_ERROR);
    validate_join_failure_result(&result);
    CHECK(__real_pthread_join(captured_join_thread, NULL) == 0);
    CHECK(fsync_after_join_failure == 0);
    CHECK(unlink(path) == 0);
    CHECK(rmdir(directory) == 0);
    puts("stream writer join-failure test passed");
    return EXIT_SUCCESS;
  }
  validate_boundary_samples(&result);
  size_t size = 0;
  uint8_t *bytes = read_file(path, &size);
  CHECK(memcmp(bytes, expected_opening, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE) == 0);

  if (strcmp(mode, "failure") == 0) {
    CHECK(finish == OAI_MEMPROF_STREAM_WRITER_IO_ERROR);
    CHECK(result.status == OAI_MEMPROF_STREAM_WRITER_IO_ERROR);
    CHECK(result.system_errno == EIO);
    CHECK(result.prefooter_closed);
    CHECK(result.runtime_snapshot.state == OAI_MEMPROF_CORE_DRAINING);
    CHECK(result.chunk_count == 0 && result.record_count == 0 && result.payload_bytes == 0);
    CHECK(result.stream_bytes == fail_after && size == fail_after);
  } else {
    const size_t chunks = event_count == 0 ? 0 : (event_count == 3 ? 2 : 1);
    CHECK(finish == OAI_MEMPROF_STREAM_WRITER_OK);
    CHECK(result.status == OAI_MEMPROF_STREAM_WRITER_OK && result.prefooter_closed);
    CHECK(result.runtime_snapshot.state == OAI_MEMPROF_CORE_DRAINING);
    CHECK(result.chunk_count == chunks && result.record_count == event_count);
    CHECK(result.payload_bytes == event_count * OAI_MEMPROF_EVENT_V1_WIRE_SIZE);
    CHECK(result.stream_bytes == size);
    struct stat on_path;
    CHECK(stat(path, &on_path) == 0);
    CHECK(result.file_device == (uint64_t)on_path.st_dev && result.file_inode == (uint64_t)on_path.st_ino);
    validate_complete_stream(bytes, size, event_count, chunks, expected_opening);
  }

  free(bytes);
  CHECK(unlink(path) == 0);
  CHECK(rmdir(directory) == 0);
  printf("stream writer %s test passed\n", mode);
  return EXIT_SUCCESS;
}
