/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "common/utils/memprof/oai_memprof_process_session.h"

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      exit(EXIT_FAILURE);                                                             \
    }                                                                                 \
  } while (0)

static const uint8_t canonical_configuration[] =
    "{\"configuration_id\":\"process-session-fixture\",\"version\":{\"major\":1,\"minor\":0}}\n";

static oai_memprof_container_v1_opening_header_t opening_template(void)
{
  oai_memprof_container_v1_opening_header_t opening = {
      .page_size_bytes = 4096,
      .scope_kind = OAI_MEMPROF_CONTAINER_V1_SCOPE_MEASUREMENT_INTERVAL,
      .role_kind = OAI_MEMPROF_CONTAINER_V1_ROLE_GNB,
      .clock_kind = OAI_MEMPROF_CONTAINER_V1_CLOCK_X86_TSC,
      .calibration_kind = OAI_MEMPROF_CONTAINER_V1_CALIBRATION_EXACT_RATE,
      .process_generation = 7,
      .counter_frequency_numerator = 1,
      .counter_frequency_denominator = 1,
      .calibration_error_bound_ns = OAI_MEMPROF_STREAM_WRITER_CLOCK_BRACKET_NS,
      .start_counter = 1,
      .start_monotonic_raw_ns = 1,
      .start_realtime_unix_ns = 1,
      .pid = 1,
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

static uint8_t *read_at(int directory_fd, const char *name, size_t *size, struct stat *file_status)
{
  const int fd = openat(directory_fd, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  CHECK(fd >= 0);
  CHECK(fstat(fd, file_status) == 0 && S_ISREG(file_status->st_mode) && file_status->st_nlink == 1 && file_status->st_size >= 0);
  *size = (size_t)file_status->st_size;
  uint8_t *bytes = malloc(*size == 0 ? 1 : *size);
  CHECK(bytes != NULL);
  size_t offset = 0;
  while (offset != *size) {
    const ssize_t got = read(fd, bytes + offset, *size - offset);
    if (got < 0 && errno == EINTR)
      continue;
    CHECK(got > 0);
    offset += (size_t)got;
  }
  CHECK(close(fd) == 0);
  return bytes;
}

static bool contains(const uint8_t *haystack, size_t haystack_size, const char *needle)
{
  const size_t needle_size = strlen(needle);
  if (needle_size == 0 || needle_size > haystack_size)
    return false;
  for (size_t offset = 0; offset <= haystack_size - needle_size; ++offset)
    if (memcmp(haystack + offset, needle, needle_size) == 0)
      return true;
  return false;
}

static void publish_events(void)
{
  for (size_t index = 0; index < 3; ++index) {
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

static void assert_existing_handoff_retained(int directory_fd)
{
  const uint8_t expected[] = {'s', 'e', 'n', 't', 'i', 'n', 'e', 'l'};
  size_t size = 0;
  struct stat file_status;
  uint8_t *bytes = read_at(directory_fd, "handoff.bin", &size, &file_status);
  CHECK(size == sizeof(expected) && memcmp(bytes, expected, sizeof(expected)) == 0);
  free(bytes);
}

int main(int argc, char **argv)
{
  CHECK(argc == 2);
  const bool existing_handoff = strcmp(argv[1], "existing") == 0;
  CHECK(existing_handoff || strcmp(argv[1], "positive") == 0);

  oai_memprof_clock_info_v1_t clock = {0};
  const oai_memprof_clock_status_t clock_status = oai_memprof_clock_info_v1(&clock);
  if (clock_status == OAI_MEMPROF_CLOCK_UNSUPPORTED) {
    puts("process session clock test skipped: no admitted architectural exact-rate source");
    return 77;
  }
  CHECK(clock_status == OAI_MEMPROF_CLOCK_OK);

  char directory[] = "/tmp/oai-memprof-process-session.XXXXXX";
  CHECK(mkdtemp(directory) != NULL);
  const int directory_fd = open(directory, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  CHECK(directory_fd >= 0);
  uint8_t configuration[sizeof(canonical_configuration) - 1U];
  memcpy(configuration, canonical_configuration, sizeof(configuration));
  oai_memprof_process_session_config_t config = {
      .directory_fd = directory_fd,
      .stream_file_name = "pre-footer.bin",
      .handoff_file_name = "handoff.bin",
      .configuration_bytes = configuration,
      .configuration_size = sizeof(configuration),
      .runtime =
          {
              .core = {.process_generation = 7, .max_threads = 1, .ring_records = 8, .mode_id = OAI_MEMPROF_CORE_EXACT_EVENTS},
              .realloc_zero_policy_id = 1,
          },
      .opening_header = opening_template(),
      .flush_records = 2,
      .flush_interval_ns = UINT64_C(100000000),
  };

  oai_memprof_process_session_t *session = (oai_memprof_process_session_t *)(uintptr_t)UINT64_C(0x1234);
  oai_memprof_process_session_config_t invalid = config;
  invalid.handoff_file_name = "../escape";
  CHECK(oai_memprof_process_session_start_v1(&invalid, &session) == OAI_MEMPROF_PROCESS_SESSION_INVALID_CONFIGURATION);
  CHECK(session == (oai_memprof_process_session_t *)(uintptr_t)UINT64_C(0x1234));

  session = NULL;
  CHECK(oai_memprof_process_session_start_v1(&config, &session) == OAI_MEMPROF_PROCESS_SESSION_OK);
  CHECK(session != NULL);
  configuration[0] = '!';
  publish_events();

  if (existing_handoff) {
    const int existing =
        openat(directory_fd, "handoff.bin", O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR);
    CHECK(existing >= 0);
    const uint8_t sentinel[] = {'s', 'e', 'n', 't', 'i', 'n', 'e', 'l'};
    CHECK(write(existing, sentinel, sizeof(sentinel)) == (ssize_t)sizeof(sentinel));
    CHECK(fsync(existing) == 0 && close(existing) == 0 && fsync(directory_fd) == 0);
  }

  oai_memprof_process_session_result_t result = {0};
  const oai_memprof_process_session_status_t finish = oai_memprof_process_session_finish_v1(session, UINT64_C(100000000), &result);
  if (existing_handoff) {
    CHECK(finish == OAI_MEMPROF_PROCESS_SESSION_IO_ERROR);
    CHECK(result.status == OAI_MEMPROF_PROCESS_SESSION_IO_ERROR && result.system_errno == EEXIST);
    CHECK(!result.handoff_published && result.writer.status == OAI_MEMPROF_STREAM_WRITER_OK);
    assert_existing_handoff_retained(directory_fd);
  } else {
    CHECK(finish == OAI_MEMPROF_PROCESS_SESSION_OK && result.status == OAI_MEMPROF_PROCESS_SESSION_OK);
    CHECK(result.handoff_published && result.system_errno == 0);
    CHECK(result.writer.status == OAI_MEMPROF_STREAM_WRITER_OK && result.writer.record_count == 3 && result.writer.chunk_count == 2
          && result.writer.runtime_snapshot.ready_threads == 1);

    size_t handoff_size = 0;
    struct stat handoff_status;
    uint8_t *handoff_wire = read_at(directory_fd, "handoff.bin", &handoff_size, &handoff_status);
    CHECK(result.handoff_bytes == handoff_size && result.handoff_device == (uint64_t)handoff_status.st_dev
          && result.handoff_inode == (uint64_t)handoff_status.st_ino);
    oai_memprof_process_handoff_thread_v1_t thread = {0};
    oai_memprof_process_handoff_v1_t handoff = {0};
    CHECK(oai_memprof_process_handoff_v1_decode(&handoff, &thread, 1, handoff_wire, handoff_size)
          == OAI_MEMPROF_PROCESS_HANDOFF_OK);
    CHECK(handoff.bootstrap_size == sizeof(canonical_configuration) - 1U
          && memcmp(handoff.bootstrap_bytes, canonical_configuration, handoff.bootstrap_size) == 0);
    CHECK(handoff.maps_size != 0 && contains(handoff.maps_bytes, handoff.maps_size, "test_oai_memprof_process_session"));
    CHECK(handoff.opening_sample.counter == handoff.opening_header.start_counter
          && handoff.opening_sample.realtime_unix_ns == handoff.opening_header.start_realtime_unix_ns
          && handoff.opening_sample.monotonic_raw_before_ns <= handoff.opening_header.start_monotonic_raw_ns
          && handoff.opening_sample.monotonic_raw_after_ns >= handoff.opening_header.start_monotonic_raw_ns
          && handoff.opening_sample.counter < handoff.writer.seal_before_sample.counter
          && handoff.opening_sample.monotonic_raw_after_ns <= handoff.writer.seal_before_sample.monotonic_raw_before_ns);
    CHECK(handoff.opening_header.pid == (uint32_t)getpid() && handoff.opening_header.configured_thread_capacity == 1);
    CHECK(handoff.ring_records == 8 && handoff.flush_records == 2 && handoff.flush_interval_ns == UINT64_C(100000000)
          && handoff.realloc_zero_policy_id == 1);
    CHECK(handoff.thread_count == 1 && thread.runtime.thread_index == 1 && thread.runtime.completed_transactions == 3
          && thread.runtime.requested_bytes == 195);

    size_t stream_size = 0;
    struct stat stream_status;
    uint8_t *stream = read_at(directory_fd, "pre-footer.bin", &stream_size, &stream_status);
    CHECK(stream_size == handoff.writer.stream_bytes && stream_size == result.writer.stream_bytes);
    uint8_t encoded_opening[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE];
    CHECK(oai_memprof_container_v1_opening_header_encode(&handoff.opening_header, encoded_opening, sizeof(encoded_opening))
          == OAI_MEMPROF_CONTAINER_V1_OK);
    CHECK(memcmp(stream, encoded_opening, sizeof(encoded_opening)) == 0);
    uint8_t prefix_sha256[32];
    CHECK(oai_memprof_container_v1_sha256(stream, stream_size, prefix_sha256) == OAI_MEMPROF_CONTAINER_V1_OK);
    CHECK(memcmp(prefix_sha256, handoff.prefix_sha256, sizeof(prefix_sha256)) == 0);
    free(stream);
    free(handoff_wire);
  }

  CHECK(unlinkat(directory_fd, "pre-footer.bin", 0) == 0);
  CHECK(unlinkat(directory_fd, "handoff.bin", 0) == 0);
  CHECK(close(directory_fd) == 0 && rmdir(directory) == 0);
  printf("process session %s test passed\n", argv[1]);
  return EXIT_SUCCESS;
}
