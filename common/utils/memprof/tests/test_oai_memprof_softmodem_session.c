/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "common/utils/memprof/oai_memprof_softmodem_session.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition)                                                                                \
  do {                                                                                                  \
    if (!(condition)) {                                                                                 \
      fprintf(stderr, "CHECK failed at %s:%d: %s (errno=%d)\n", __FILE__, __LINE__, #condition, errno); \
      return EXIT_FAILURE;                                                                              \
    }                                                                                                   \
  } while (0)

static const char *const environment_names[] = {
    "OAI_MEMPROF_SESSION_ENABLE",
    "OAI_MEMPROF_SESSION_ARCHIVE_FD",
    "OAI_MEMPROF_SESSION_BOOTSTRAP_FD",
    "OAI_MEMPROF_SESSION_PROCESS_GENERATION",
    "OAI_MEMPROF_SESSION_MAX_THREADS",
    "OAI_MEMPROF_SESSION_RING_RECORDS",
    "OAI_MEMPROF_SESSION_MODE_ID",
    "OAI_MEMPROF_SESSION_TABLE_ENTRIES",
    "OAI_MEMPROF_SESSION_SAMPLE_SEED",
    "OAI_MEMPROF_SESSION_SAMPLE_THRESHOLD",
    "OAI_MEMPROF_SESSION_TABLE_PROBES",
    "OAI_MEMPROF_SESSION_REALLOC_ZERO_POLICY_ID",
    "OAI_MEMPROF_SESSION_FLUSH_RECORDS",
    "OAI_MEMPROF_SESSION_FLUSH_INTERVAL_NS",
    "OAI_MEMPROF_SESSION_SEAL_TIMEOUT_NS",
};

static const char *const legacy_environment_names[] = {
    "OAI_MEMPROF_SESSION_ARCHIVE_DIRECTORY",
    "OAI_MEMPROF_SESSION_CONFIGURATION_PATH",
    "OAI_MEMPROF_SESSION_OPENING_PATH",
};

static bool clear_environment(void)
{
  for (size_t index = 0; index < sizeof(environment_names) / sizeof(environment_names[0]); ++index)
    if (unsetenv(environment_names[index]) != 0)
      return false;
  for (size_t index = 0; index < sizeof(legacy_environment_names) / sizeof(legacy_environment_names[0]); ++index)
    if (unsetenv(legacy_environment_names[index]) != 0)
      return false;
  return true;
}

static bool write_bytes(const char *path, const void *bytes, size_t size)
{
  const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR);
  if (fd < 0)
    return false;
  size_t offset = 0;
  while (offset != size) {
    const ssize_t count = write(fd, (const uint8_t *)bytes + offset, size - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      (void)close(fd);
      return false;
    }
    offset += (size_t)count;
  }
  return fsync(fd) == 0 && close(fd) == 0;
}

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
    opening.primary_build_id_sha256[index] = (uint8_t)(0xc0 + index);
  }
  return opening;
}

static int environment_case(bool role_mismatch,
                            bool sampled,
                            bool replace_roots,
                            bool configuration_mismatch,
                            bool insecure_streams)
{
  oai_memprof_clock_info_v1_t clock = {0};
  const oai_memprof_clock_status_t clock_status = oai_memprof_clock_info_v1(&clock);
  if (clock_status == OAI_MEMPROF_CLOCK_UNSUPPORTED)
    return 77;
  CHECK(clock_status == OAI_MEMPROF_CLOCK_OK);

  char root[] = "/tmp/oai-memprof-softmodem-session.XXXXXX";
  CHECK(mkdtemp(root) != NULL);
  char archive[sizeof(root) + sizeof("/archive")];
  char bootstrap[sizeof(root) + sizeof("/bootstrap")];
  char streams[sizeof(root) + sizeof("/archive-original/streams")];
  char configuration_path[sizeof(root) + sizeof("/bootstrap-original/effective-config.json")];
  char opening_path[sizeof(root) + sizeof("/bootstrap-original/opening.bin")];
  CHECK(snprintf(archive, sizeof(archive), "%s/archive", root) > 0);
  CHECK(snprintf(bootstrap, sizeof(bootstrap), "%s/bootstrap", root) > 0);
  CHECK(snprintf(streams, sizeof(streams), "%s/streams", archive) > 0);
  CHECK(snprintf(configuration_path, sizeof(configuration_path), "%s/effective-config.json", bootstrap) > 0);
  CHECK(snprintf(opening_path, sizeof(opening_path), "%s/opening.bin", bootstrap) > 0);
  CHECK(mkdir(archive, S_IRWXU) == 0);
  CHECK(mkdir(bootstrap, S_IRWXU) == 0);
  CHECK(mkdir(streams, S_IRWXU) == 0);
  if (insecure_streams)
    CHECK(chmod(streams, S_IRWXU | S_IRWXG | S_IRWXO) == 0);
  static const uint8_t configuration[] = "{\"catalog_id\":\"softmodem-session-fixture\"}\n";
  CHECK(write_bytes(configuration_path, configuration, sizeof(configuration) - 1U));
  uint8_t opening_wire[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE];
  oai_memprof_container_v1_opening_header_t opening = opening_template();
  CHECK(oai_memprof_container_v1_sha256(configuration, sizeof(configuration) - 1U, opening.configuration_instance_sha256)
        == OAI_MEMPROF_CONTAINER_V1_OK);
  if (configuration_mismatch)
    opening.configuration_instance_sha256[0] ^= UINT8_C(1);
  CHECK(oai_memprof_container_v1_opening_header_encode(&opening, opening_wire, sizeof(opening_wire))
        == OAI_MEMPROF_CONTAINER_V1_OK);
  CHECK(write_bytes(opening_path, opening_wire, sizeof(opening_wire)));

  const int archive_fd = open(archive, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  const int bootstrap_fd = open(bootstrap, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  CHECK(archive_fd >= 3 && bootstrap_fd >= 3);
  char archive_fd_text[32];
  char bootstrap_fd_text[32];
  CHECK(snprintf(archive_fd_text, sizeof(archive_fd_text), "%d", archive_fd) > 0);
  CHECK(snprintf(bootstrap_fd_text, sizeof(bootstrap_fd_text), "%d", bootstrap_fd) > 0);

  CHECK(setenv(environment_names[0], "1", 1) == 0);
  CHECK(setenv(environment_names[1], archive_fd_text, 1) == 0);
  CHECK(setenv(environment_names[2], bootstrap_fd_text, 1) == 0);
  CHECK(setenv(environment_names[3], "7", 1) == 0);
  CHECK(setenv(environment_names[4], "1", 1) == 0);
  CHECK(setenv(environment_names[5], "8", 1) == 0);
  CHECK(setenv(environment_names[6], sampled ? "3" : "4", 1) == 0);
  const uint64_t sample_seed = sampled ? UINT64_C(4825167233289836708) : UINT64_C(0);
  char sample_seed_text[32] = "0";
  if (sampled)
    CHECK(snprintf(sample_seed_text, sizeof(sample_seed_text), "%" PRIu64, sample_seed) > 0);
  CHECK(setenv(environment_names[7], sampled ? "64" : "0", 1) == 0);
  CHECK(setenv(environment_names[8], sample_seed_text, 1) == 0);
  CHECK(setenv(environment_names[9], sampled ? "1" : "0", 1) == 0);
  CHECK(setenv(environment_names[10], sampled ? "8" : "0", 1) == 0);
  CHECK(setenv(environment_names[11], "1", 1) == 0);
  CHECK(setenv(environment_names[12], "1", 1) == 0);
  CHECK(setenv(environment_names[13], "1000000", 1) == 0);
  CHECK(setenv(environment_names[14], "1000000000", 1) == 0);

  char original_archive[sizeof(root) + sizeof("/archive-original")];
  char original_bootstrap[sizeof(root) + sizeof("/bootstrap-original")];
  if (replace_roots) {
    CHECK(snprintf(original_archive, sizeof(original_archive), "%s/archive-original", root) > 0);
    CHECK(snprintf(original_bootstrap, sizeof(original_bootstrap), "%s/bootstrap-original", root) > 0);
    CHECK(rename(archive, original_archive) == 0);
    CHECK(rename(bootstrap, original_bootstrap) == 0);
    CHECK(mkdir(archive, S_IRWXU) == 0);
    CHECK(mkdir(bootstrap, S_IRWXU) == 0);
    char replacement_streams[sizeof(root) + sizeof("/archive/streams")];
    CHECK(snprintf(replacement_streams, sizeof(replacement_streams), "%s/streams", archive) > 0);
    CHECK(mkdir(replacement_streams, S_IRWXU) == 0);
  }
  const char *const active_archive = replace_roots ? original_archive : archive;
  const char *const active_bootstrap = replace_roots ? original_bootstrap : bootstrap;

  const uint16_t role = role_mismatch ? OAI_MEMPROF_SOFTMODEM_ROLE_NR_UE : OAI_MEMPROF_SOFTMODEM_ROLE_GNB;
  const oai_memprof_softmodem_session_status_t start = oai_memprof_softmodem_session_start_v1(role);
  if (role_mismatch || configuration_mismatch || insecure_streams) {
    CHECK(start
          == (insecure_streams ? OAI_MEMPROF_SOFTMODEM_SESSION_IO_ERROR : OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_CONFIGURATION));
    char stream_path[sizeof(root) + sizeof("/archive-original/streams/memory-lifetime.bin")];
    char handoff_path[sizeof(root) + sizeof("/archive-original/streams/process-handoff.bin")];
    CHECK(snprintf(stream_path, sizeof(stream_path), "%s/streams/memory-lifetime.bin", active_archive) > 0);
    CHECK(snprintf(handoff_path, sizeof(handoff_path), "%s/streams/process-handoff.bin", active_archive) > 0);
    CHECK(access(stream_path, F_OK) != 0 && access(handoff_path, F_OK) != 0);
  } else {
    CHECK(start == OAI_MEMPROF_SOFTMODEM_SESSION_OK);
    oai_memprof_core_ticket_t ticket = {0};
    CHECK(oai_memprof_active_runtime_begin_v1(1, 64, true, &ticket));
    const oai_memprof_core_payload_t payload = {
        .address_after = UINT64_C(0x1000),
        .arg0 = 64,
        .flags = (UINT32_C(1) << 1) | (UINT32_C(1) << 2) | (UINT32_C(1) << 11) | (UINT32_C(1) << 24),
        .api_id = 1,
        .event_kind = 1,
    };
    CHECK(oai_memprof_active_runtime_end_v1(&ticket, &payload) == OAI_MEMPROF_CORE_OK);
    oai_memprof_process_session_result_t result = {0};
    CHECK(oai_memprof_softmodem_session_finish_v1(&result) == OAI_MEMPROF_SOFTMODEM_SESSION_OK);
    CHECK(result.status == OAI_MEMPROF_PROCESS_SESSION_OK && result.writer.status == OAI_MEMPROF_STREAM_WRITER_OK
          && result.writer.record_count == 1 && result.handoff_published);
    CHECK(result.writer.runtime_snapshot.table_entries == (sampled ? UINT64_C(64) : UINT64_C(0)));
    CHECK(result.writer.runtime_snapshot.sample_seed == sample_seed);
    CHECK(result.writer.runtime_snapshot.sample_threshold == (sampled ? UINT64_C(1) : UINT64_C(0)));
    CHECK(result.writer.runtime_snapshot.table_probes == (sampled ? UINT32_C(8) : UINT32_C(0)));
    CHECK(result.writer.runtime_snapshot.table_shards == (sampled ? UINT32_C(64) : UINT32_C(0)));
    oai_memprof_process_session_result_t repeated = {0};
    CHECK(oai_memprof_softmodem_session_finish_v1(&repeated) == OAI_MEMPROF_SOFTMODEM_SESSION_ALREADY_FINISHED);
    CHECK(repeated.handoff_inode == result.handoff_inode && repeated.handoff_bytes == result.handoff_bytes);
    char stream_path[sizeof(root) + sizeof("/archive-original/streams/memory-lifetime.bin")];
    char handoff_path[sizeof(root) + sizeof("/archive-original/streams/process-handoff.bin")];
    CHECK(snprintf(stream_path, sizeof(stream_path), "%s/streams/memory-lifetime.bin", active_archive) > 0);
    CHECK(snprintf(handoff_path, sizeof(handoff_path), "%s/streams/process-handoff.bin", active_archive) > 0);
    CHECK(unlink(stream_path) == 0 && unlink(handoff_path) == 0);
    if (replace_roots) {
      char replacement_stream_path[sizeof(root) + sizeof("/archive/streams/memory-lifetime.bin")];
      char replacement_handoff_path[sizeof(root) + sizeof("/archive/streams/process-handoff.bin")];
      CHECK(snprintf(replacement_stream_path, sizeof(replacement_stream_path), "%s/streams/memory-lifetime.bin", archive) > 0);
      CHECK(snprintf(replacement_handoff_path, sizeof(replacement_handoff_path), "%s/streams/process-handoff.bin", archive) > 0);
      CHECK(access(replacement_stream_path, F_OK) != 0 && access(replacement_handoff_path, F_OK) != 0);
    }
  }
  char active_configuration_path[sizeof(root) + sizeof("/bootstrap-original/effective-config.json")];
  char active_opening_path[sizeof(root) + sizeof("/bootstrap-original/opening.bin")];
  char active_streams[sizeof(root) + sizeof("/archive-original/streams")];
  CHECK(snprintf(active_configuration_path, sizeof(active_configuration_path), "%s/effective-config.json", active_bootstrap) > 0);
  CHECK(snprintf(active_opening_path, sizeof(active_opening_path), "%s/opening.bin", active_bootstrap) > 0);
  CHECK(snprintf(active_streams, sizeof(active_streams), "%s/streams", active_archive) > 0);
  CHECK(unlink(active_configuration_path) == 0 && unlink(active_opening_path) == 0);
  CHECK(rmdir(active_streams) == 0 && rmdir(active_archive) == 0 && rmdir(active_bootstrap) == 0);
  if (replace_roots) {
    char replacement_streams[sizeof(root) + sizeof("/archive/streams")];
    CHECK(snprintf(replacement_streams, sizeof(replacement_streams), "%s/streams", archive) > 0);
    CHECK(rmdir(replacement_streams) == 0 && rmdir(archive) == 0 && rmdir(bootstrap) == 0);
  }
  CHECK(rmdir(root) == 0);
  return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
  CHECK(argc == 2);
  CHECK(clear_environment());
  if (strcmp(argv[1], "disabled") == 0) {
    CHECK(oai_memprof_softmodem_session_finish_v1(NULL) == OAI_MEMPROF_SOFTMODEM_SESSION_DISABLED);
    CHECK(oai_memprof_softmodem_session_start_v1(OAI_MEMPROF_SOFTMODEM_ROLE_GNB) == OAI_MEMPROF_SOFTMODEM_SESSION_DISABLED);
    CHECK(oai_memprof_softmodem_session_finish_v1(NULL) == OAI_MEMPROF_SOFTMODEM_SESSION_DISABLED);
  } else if (strcmp(argv[1], "partial") == 0) {
    CHECK(setenv(environment_names[0], "1", 1) == 0);
    CHECK(oai_memprof_softmodem_session_start_v1(OAI_MEMPROF_SOFTMODEM_ROLE_GNB)
          == OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_ENVIRONMENT);
  } else if (strcmp(argv[1], "legacy-path") == 0) {
    CHECK(setenv(legacy_environment_names[0], "/tmp/legacy-session-path", 1) == 0);
    CHECK(oai_memprof_softmodem_session_start_v1(OAI_MEMPROF_SOFTMODEM_ROLE_GNB)
          == OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_ENVIRONMENT);
  } else if (strcmp(argv[1], "role-mismatch") == 0) {
    return environment_case(true, false, false, false, false);
  } else if (strcmp(argv[1], "positive") == 0) {
    return environment_case(false, false, false, false, false);
  } else if (strcmp(argv[1], "sampled") == 0) {
    return environment_case(false, true, false, false, false);
  } else if (strcmp(argv[1], "fd-roots-replaced") == 0) {
    return environment_case(false, false, true, false, false);
  } else if (strcmp(argv[1], "configuration-mismatch") == 0) {
    return environment_case(false, false, false, true, false);
  } else if (strcmp(argv[1], "insecure-streams") == 0) {
    return environment_case(false, false, false, false, true);
  } else {
    return 64;
  }
  return EXIT_SUCCESS;
}
