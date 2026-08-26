/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "common/utils/memprof/oai_memprof_stream_finalizer.h"
#include "common/utils/memprof/oai_memprof_wire.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      exit(EXIT_FAILURE);                                                             \
    }                                                                                 \
  } while (0)

static size_t maximum_pwrite = SIZE_MAX;
static uint64_t fail_after = UINT64_MAX;
static uint64_t observed_pwrite;
static unsigned runtime_snapshot_calls;
static unsigned runtime_complete_calls;

ssize_t __real_pwrite(int fd, const void *buffer, size_t size, off_t offset);

oai_memprof_core_status_t __real_oai_memprof_active_runtime_snapshot_v1(oai_memprof_core_snapshot_t *snapshot);
oai_memprof_core_status_t __real_oai_memprof_active_runtime_complete_v1(void);
static void hash_stream_prefix(const char *path, uint64_t stream_bytes, uint8_t prefix_sha256[32], uint8_t opening_sha256[32]);

oai_memprof_core_status_t __wrap_oai_memprof_active_runtime_snapshot_v1(oai_memprof_core_snapshot_t *snapshot)
{
  ++runtime_snapshot_calls;
  return __real_oai_memprof_active_runtime_snapshot_v1(snapshot);
}

oai_memprof_core_status_t __wrap_oai_memprof_active_runtime_complete_v1(void)
{
  ++runtime_complete_calls;
  return __real_oai_memprof_active_runtime_complete_v1();
}

ssize_t __wrap_pwrite(int fd, const void *buffer, size_t size, off_t offset)
{
  if (observed_pwrite >= fail_after) {
    errno = EIO;
    return -1;
  }
  size_t allowed = size < maximum_pwrite ? size : maximum_pwrite;
  if ((uint64_t)allowed > fail_after - observed_pwrite)
    allowed = (size_t)(fail_after - observed_pwrite);
  if (allowed == 0) {
    errno = EIO;
    return -1;
  }
  const ssize_t result = __real_pwrite(fd, buffer, allowed, offset);
  if (result > 0)
    observed_pwrite += (uint64_t)result;
  return result;
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

static void initialize_objects(oai_memprof_container_v1_object_entry_t objects[12],
                               const oai_memprof_container_v1_opening_header_t *opening)
{
  static const uint32_t flags[12] = {0x05, 0x05, 0x0b, 0x1b, 0x1b, 0x13, 0x03, 0x03, 0x13, 0x07, 0x03, 0x03};
  for (size_t index = 0; index < 12; ++index) {
    objects[index].object_kind = (uint16_t)(index + 1U);
    objects[index].format_id = 1;
    objects[index].object_flags = flags[index];
    objects[index].schema_revision = 1;
    objects[index].entry_count = (index == 0U) ? 13U : 0U;
    if (index == 1U)
      objects[index].entry_count = 12U;
    if (index == 5U || index == 6U || index == 7U || index == 8U || index == 10U)
      objects[index].entry_count = 1U;
    if (index == 9U || index == 11U)
      objects[index].entry_count = 1U;
    objects[index].byte_count = 100U + index;
    for (size_t byte = 0; byte < 32U; ++byte)
      objects[index].sha256[byte] = (uint8_t)(index + byte + 1U);
  }
  memcpy(objects[0].sha256, opening->schema_bundle_definition_sha256, 32U);
  memcpy(objects[1].sha256, opening->api_catalog_definition_sha256, 32U);
  memcpy(objects[9].sha256, opening->configuration_instance_sha256, 32U);
}

static oai_memprof_container_v1_trailer_header_t trailer_header(const oai_memprof_stream_writer_result_t *prefooter,
                                                                const oai_memprof_container_v1_opening_header_t *opening)
{
  return (oai_memprof_container_v1_trailer_header_t){
      .trailer_body_bytes = UINT64_C(256) + UINT64_C(32) + UINT64_C(32) + UINT64_C(12) * UINT64_C(64),
      .process_generation = opening->process_generation,
      .scope_kind = opening->scope_kind,
      .lifecycle_state = OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_COMPLETE,
      .payload_writer_state = OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED,
      .finalization_stage = OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRE_SYNC_TERMINAL_MATERIAL_FROZEN,
      .terminal_flags = UINT64_C(0x0fff),
      .chunk_count = prefooter->chunk_count,
      .record_count = prefooter->record_count,
      .payload_bytes = prefooter->payload_bytes,
      .first_chunk_offset = OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE,
      .chunks_end_offset = prefooter->stream_bytes,
      .active_generation = opening->process_generation,
      .active_start_counter = opening->start_counter,
      .cutoff_before_counter = opening->start_counter + 1U,
      .cutoff_after_counter = opening->start_counter + 2U,
      .quiescence_complete_counter = opening->start_counter + 3U,
      .final_counter = opening->start_counter + 4U,
      .active_start_monotonic_raw_ns = opening->start_monotonic_raw_ns,
      .final_monotonic_raw_ns = opening->start_monotonic_raw_ns + 10U,
      .final_realtime_unix_ns = opening->start_realtime_unix_ns + 10U,
      .event_entry_count = 1,
      .diagnostic_entry_count = 1,
      .object_entry_count = 12,
      .event_table_offset = OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE,
      .diagnostic_table_offset = OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE,
      .object_table_offset = OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE
                             + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE,
      .terminal_reason_code = OAI_MEMPROF_CONTAINER_V1_REASON_NONE,
  };
}

static void run_python_oracle(const char *python, const char *script, const char *path)
{
  const pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    execl(python, python, "-B", script, path, "3", "2", (char *)NULL);
    _exit(127);
  }
  int status = 0;
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

static uint64_t file_size(const char *path)
{
  struct stat status;
  CHECK(stat(path, &status) == 0 && S_ISREG(status.st_mode) && status.st_size >= 0);
  return (uint64_t)status.st_size;
}

int main(int argc, char **argv)
{
  CHECK(argc == 4);
  const char *mode = argv[1];
  const char *python = argv[2];
  const char *oracle = argv[3];
  oai_memprof_clock_info_v1_t clock = {0};
  const oai_memprof_clock_status_t clock_status = oai_memprof_clock_info_v1(&clock);
  if (clock_status == OAI_MEMPROF_CLOCK_UNSUPPORTED) {
    puts("stream finalizer clock test skipped: no admitted architectural exact-rate source");
    return 77;
  }
  CHECK(clock_status == OAI_MEMPROF_CLOCK_OK);

  char directory[] = "/tmp/oai-memprof-stream-finalizer.XXXXXX";
  CHECK(mkdtemp(directory) != NULL);
  char path[sizeof(directory) + 32];
  char original_path[sizeof(directory) + 32];
  CHECK(snprintf(path, sizeof(path), "%s/stream.bin", directory) > 0);
  CHECK(snprintf(original_path, sizeof(original_path), "%s/original.bin", directory) > 0);
  const int directory_fd = open(directory, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  CHECK(directory_fd >= 0);

  const oai_memprof_container_v1_opening_header_t opening = opening_header(&clock);
  const oai_memprof_stream_writer_config_t writer_config = {
      .directory_fd = directory_fd,
      .file_name = "stream.bin",
      .runtime =
          {
              .core = {.process_generation = 7, .max_threads = 1, .ring_records = 8, .mode_id = OAI_MEMPROF_CORE_EXACT_EVENTS},
              .realloc_zero_policy_id = 1,
          },
      .opening_header = opening,
      .flush_records = 2,
      .flush_interval_ns = UINT64_C(100000000),
  };
  oai_memprof_stream_writer_t *writer = NULL;
  CHECK(oai_memprof_stream_writer_start_v1(&writer_config, &writer) == OAI_MEMPROF_STREAM_WRITER_OK);
  publish_events();
  oai_memprof_stream_writer_result_t prefooter = {0};
  CHECK(oai_memprof_stream_writer_finish_v1(writer, UINT64_C(100000000), &prefooter) == OAI_MEMPROF_STREAM_WRITER_OK);
  CHECK(prefooter.chunk_count == 2 && prefooter.record_count == 3 && prefooter.prefooter_closed);

  oai_memprof_container_v1_event_total_entry_t event_entries[1] = {{.event_kind = 1, .api_id = 1, .record_count = 3}};
  oai_memprof_container_v1_diagnostic_total_entry_t diagnostic_entries[1] = {{.reason_id = 1}};
  oai_memprof_container_v1_object_entry_t object_entries[12] = {0};
  initialize_objects(object_entries, &opening);
  oai_memprof_stream_finalizer_config_t finalizer_config = {
      .directory_fd = directory_fd,
      .file_name = "stream.bin",
      .prefooter = prefooter,
      .trailer_header = trailer_header(&prefooter, &opening),
      .event_entries = event_entries,
      .event_entry_count = 1,
      .diagnostic_entries = diagnostic_entries,
      .diagnostic_entry_count = 1,
      .object_entries = object_entries,
      .object_entry_count = 12,
  };

  const bool offline = strcmp(mode, "offline") == 0 || strncmp(mode, "offline-", 8U) == 0;
  uint8_t authenticated_prefix_sha256[32] = {0};
  uint8_t authenticated_opening_header_sha256[32] = {0};
  if (offline) {
    hash_stream_prefix(path, prefooter.stream_bytes, authenticated_prefix_sha256, authenticated_opening_header_sha256);
    finalizer_config.authenticated_prefix_sha256 = authenticated_prefix_sha256;
    finalizer_config.authenticated_opening_header_sha256 = authenticated_opening_header_sha256;
  }

  if (strcmp(mode, "positive") == 0) {
    const uint64_t bytes_before_invalid = file_size(path);
    oai_memprof_stream_finalizer_config_t invalid_config = finalizer_config;
    invalid_config.event_entries = NULL;
    oai_memprof_stream_finalizer_result_t invalid_result = {
        .status = OAI_MEMPROF_STREAM_FINALIZER_RUNTIME_ERROR,
        .stream_bytes = UINT64_MAX,
    };
    CHECK(oai_memprof_stream_finalize_v1(&invalid_config, &invalid_result) == OAI_MEMPROF_STREAM_FINALIZER_INVALID_ARGUMENT);
    CHECK(invalid_result.status == OAI_MEMPROF_STREAM_FINALIZER_INVALID_ARGUMENT);
    CHECK(!invalid_result.stream_verified && !invalid_result.runtime_complete);
    CHECK(invalid_result.appended_bytes == 0 && file_size(path) == bytes_before_invalid);
    oai_memprof_core_snapshot_t invalid_runtime = {0};
    CHECK(oai_memprof_active_runtime_snapshot_v1(&invalid_runtime) == OAI_MEMPROF_CORE_OK);
    CHECK(invalid_runtime.state == OAI_MEMPROF_CORE_DRAINING);
  }

  if (strcmp(mode, "short") == 0) {
    maximum_pwrite = 7;
  } else if (strcmp(mode, "failure") == 0) {
    fail_after = 100;
  } else if (strcmp(mode, "corrupt") == 0) {
    const int fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    CHECK(fd >= 0);
    uint8_t value = 0;
    const off_t corruption_offset = OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE + OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE + 92;
    CHECK(pread(fd, &value, 1, corruption_offset) == 1);
    value ^= 1U;
    CHECK(__real_pwrite(fd, &value, 1, corruption_offset) == 1);
    CHECK(fsync(fd) == 0 && close(fd) == 0);
  } else if (strcmp(mode, "mismatch") == 0) {
    event_entries[0].record_count = 2;
  } else if (strcmp(mode, "identity") == 0) {
    CHECK(renameat(directory_fd, "stream.bin", directory_fd, "original.bin") == 0);
    const int replacement =
        openat(directory_fd, "stream.bin", O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR);
    CHECK(replacement >= 0);
    CHECK(ftruncate(replacement, (off_t)prefooter.stream_bytes) == 0);
    CHECK(fsync(replacement) == 0 && close(replacement) == 0);
    CHECK(fsync(directory_fd) == 0);
  } else if (strcmp(mode, "offline-prefix-mismatch") == 0) {
    struct stat before = {0};
    struct stat after = {0};
    CHECK(stat(path, &before) == 0);
    const int fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    CHECK(fd >= 0);
    uint8_t opening_wire[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE];
    CHECK(pread(fd, opening_wire, sizeof(opening_wire), 0) == (ssize_t)sizeof(opening_wire));
    oai_memprof_container_v1_opening_header_t mutated_opening = {0};
    CHECK(oai_memprof_container_v1_opening_header_decode(&mutated_opening, opening_wire, sizeof(opening_wire))
          == OAI_MEMPROF_CONTAINER_V1_OK);
    CHECK(mutated_opening.pid != UINT32_MAX);
    ++mutated_opening.pid;
    CHECK(oai_memprof_container_v1_opening_header_encode(&mutated_opening, opening_wire, sizeof(opening_wire))
          == OAI_MEMPROF_CONTAINER_V1_OK);
    CHECK(__real_pwrite(fd, opening_wire, sizeof(opening_wire), 0) == (ssize_t)sizeof(opening_wire));
    CHECK(fsync(fd) == 0 && close(fd) == 0);
    CHECK(stat(path, &after) == 0);
    CHECK(before.st_dev == after.st_dev && before.st_ino == after.st_ino && before.st_size == after.st_size);
  } else if (strcmp(mode, "offline-opening-mismatch") == 0) {
    struct stat before = {0};
    struct stat after = {0};
    CHECK(stat(path, &before) == 0);
    const int fd = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    CHECK(fd >= 0);
    uint8_t opening_wire[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE];
    CHECK(pread(fd, opening_wire, sizeof(opening_wire), 0) == (ssize_t)sizeof(opening_wire));
    oai_memprof_container_v1_opening_header_t mutated_opening = {0};
    CHECK(oai_memprof_container_v1_opening_header_decode(&mutated_opening, opening_wire, sizeof(opening_wire))
          == OAI_MEMPROF_CONTAINER_V1_OK);
    CHECK(mutated_opening.pid != UINT32_MAX);
    ++mutated_opening.pid;
    CHECK(oai_memprof_container_v1_opening_header_encode(&mutated_opening, opening_wire, sizeof(opening_wire))
          == OAI_MEMPROF_CONTAINER_V1_OK);
    CHECK(__real_pwrite(fd, opening_wire, sizeof(opening_wire), 0) == (ssize_t)sizeof(opening_wire));
    CHECK(fsync(fd) == 0 && close(fd) == 0);
    CHECK(stat(path, &after) == 0);
    CHECK(before.st_dev == after.st_dev && before.st_ino == after.st_ino && before.st_size == after.st_size);
    uint8_t observed_opening_sha256[32] = {0};
    hash_stream_prefix(path, prefooter.stream_bytes, authenticated_prefix_sha256, observed_opening_sha256);
  } else if (strcmp(mode, "offline-writer-io") == 0) {
    finalizer_config.prefooter.status = OAI_MEMPROF_STREAM_WRITER_IO_ERROR;
    finalizer_config.trailer_header.lifecycle_state = OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED;
    finalizer_config.trailer_header.payload_writer_state =
        OAI_MEMPROF_CONTAINER_V1_WRITER_IO_FAILED_AT_SAFE_BOUNDARY_AND_CLOSED_VERIFIED;
    finalizer_config.trailer_header.terminal_reason_code = OAI_MEMPROF_CONTAINER_V1_REASON_PAYLOAD_IO_FAILED_AT_SAFE_BOUNDARY;
    finalizer_config.trailer_header.terminal_flags = UINT64_C(0x017f);
  } else {
    CHECK(strcmp(mode, "positive") == 0 || strcmp(mode, "offline") == 0 || strcmp(mode, "offline-prefix-mismatch") == 0
          || strcmp(mode, "offline-opening-mismatch") == 0);
  }

  observed_pwrite = 0;
  runtime_snapshot_calls = 0;
  runtime_complete_calls = 0;
  oai_memprof_stream_finalizer_result_t result = {0};
  /* offline was determined before the test mutates the authenticated prefix. */
  const oai_memprof_stream_finalizer_status_t status = offline ? oai_memprof_stream_finalize_offline_v1(&finalizer_config, &result)
                                                               : oai_memprof_stream_finalize_v1(&finalizer_config, &result);

  oai_memprof_core_snapshot_t runtime = {0};
  CHECK(__real_oai_memprof_active_runtime_snapshot_v1(&runtime) == OAI_MEMPROF_CORE_OK);
  if (strcmp(mode, "positive") == 0 || strcmp(mode, "short") == 0 || strcmp(mode, "offline") == 0
      || strcmp(mode, "offline-writer-io") == 0) {
    CHECK(status == OAI_MEMPROF_STREAM_FINALIZER_OK);
    CHECK(result.status == OAI_MEMPROF_STREAM_FINALIZER_OK);
    CHECK(result.stream_verified);
    CHECK(result.runtime_complete == !offline);
    CHECK(runtime.state == (offline ? OAI_MEMPROF_CORE_DRAINING : OAI_MEMPROF_CORE_COMPLETE));
    CHECK(runtime_snapshot_calls == (offline ? 0U : 1U));
    CHECK(runtime_complete_calls == (offline ? 0U : 1U));
    CHECK(result.appended_bytes == finalizer_config.trailer_header.trailer_body_bytes + OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE);
    CHECK(result.stream_bytes == file_size(path));
    CHECK(result.file_device == prefooter.file_device && result.file_inode == prefooter.file_inode);
    if (strcmp(mode, "offline-writer-io") == 0) {
      const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
      CHECK(fd >= 0);
      uint8_t trailer_wire[OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE];
      CHECK(pread(fd, trailer_wire, sizeof(trailer_wire), (off_t)prefooter.stream_bytes) == (ssize_t)sizeof(trailer_wire));
      oai_memprof_container_v1_trailer_header_t terminal = {0};
      CHECK(oai_memprof_container_v1_trailer_header_decode(&terminal, trailer_wire, sizeof(trailer_wire))
            == OAI_MEMPROF_CONTAINER_V1_OK);
      CHECK(terminal.lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED
            && terminal.terminal_reason_code == OAI_MEMPROF_CONTAINER_V1_REASON_PAYLOAD_IO_FAILED_AT_SAFE_BOUNDARY);
      CHECK(result.runtime_status == OAI_MEMPROF_CORE_OK && close(fd) == 0);
    } else {
      run_python_oracle(python, oracle, path);
    }
  } else {
    CHECK(runtime_snapshot_calls == (offline ? 0U : 1U));
    CHECK(runtime_complete_calls == 0U);
    CHECK(!result.runtime_complete);
    CHECK(runtime.state == OAI_MEMPROF_CORE_DRAINING);
    if (strcmp(mode, "failure") == 0) {
      CHECK(status == OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR);
      CHECK(result.system_errno == EIO && result.appended_bytes == 100);
      CHECK(file_size(path) == prefooter.stream_bytes + 100);
    } else if (strcmp(mode, "corrupt") == 0) {
      CHECK(status == OAI_MEMPROF_STREAM_FINALIZER_PREFOOTER_INVALID);
      CHECK(result.appended_bytes == 0 && file_size(path) == prefooter.stream_bytes);
    } else if (strcmp(mode, "mismatch") == 0) {
      CHECK(status == OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR);
      CHECK(result.appended_bytes == 0 && file_size(path) == prefooter.stream_bytes);
    } else if (strcmp(mode, "offline-prefix-mismatch") == 0 || strcmp(mode, "offline-opening-mismatch") == 0) {
      CHECK(status == OAI_MEMPROF_STREAM_FINALIZER_AUTHENTICATION_MISMATCH && result.system_errno == EILSEQ);
      CHECK(result.appended_bytes == 0 && file_size(path) == prefooter.stream_bytes);
    } else {
      CHECK(strcmp(mode, "identity") == 0);
      CHECK(status == OAI_MEMPROF_STREAM_FINALIZER_IDENTITY_MISMATCH);
      CHECK(result.appended_bytes == 0 && file_size(path) == prefooter.stream_bytes);
      CHECK(file_size(original_path) == prefooter.stream_bytes);
    }
  }

  CHECK(close(directory_fd) == 0);
  CHECK(unlink(path) == 0);
  if (strcmp(mode, "identity") == 0)
    CHECK(unlink(original_path) == 0);
  CHECK(rmdir(directory) == 0);
  printf("stream finalizer %s test passed\n", mode);
  return EXIT_SUCCESS;
}

static void hash_stream_prefix(const char *path, uint64_t stream_bytes, uint8_t prefix_sha256[32], uint8_t opening_sha256[32])
{
  CHECK(stream_bytes <= SIZE_MAX);
  const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  CHECK(fd >= 0);
  void *mapping = mmap(NULL, (size_t)stream_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  CHECK(mapping != MAP_FAILED);
  CHECK(oai_memprof_container_v1_sha256(mapping, (size_t)stream_bytes, prefix_sha256) == OAI_MEMPROF_CONTAINER_V1_OK
        && oai_memprof_container_v1_sha256(mapping, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE, opening_sha256)
               == OAI_MEMPROF_CONTAINER_V1_OK);
  CHECK(munmap(mapping, (size_t)stream_bytes) == 0);
  CHECK(close(fd) == 0);
}
