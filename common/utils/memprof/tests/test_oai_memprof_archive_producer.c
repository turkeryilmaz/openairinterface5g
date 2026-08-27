/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "common/utils/memprof/oai_memprof_process_session.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition)                                                                                                 \
  do {                                                                                                                   \
    if (!(condition)) {                                                                                                  \
      fprintf(stderr, "archive producer check failed at %s:%d: %s (errno=%d)\n", __FILE__, __LINE__, #condition, errno); \
      return 1;                                                                                                          \
    }                                                                                                                    \
  } while (0)

static bool parse_u64_argument(const char *text, bool nonzero, uint64_t *value)
{
  if (text == NULL || value == NULL || text[0] == '\0')
    return false;
  if (text[0] == '0' && text[1] != '\0')
    return false;
  uint64_t parsed = 0;
  for (const unsigned char *cursor = (const unsigned char *)text; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9')
      return false;
    const uint64_t digit = (uint64_t)(*cursor - '0');
    if (parsed > (UINT64_MAX - digit) / UINT64_C(10))
      return false;
    parsed = parsed * UINT64_C(10) + digit;
  }
  if (nonzero && parsed == 0)
    return false;
  *value = parsed;
  return true;
}

static uint8_t *read_regular(const char *path, size_t maximum, size_t *size)
{
  const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0)
    return NULL;
  struct stat before = {0};
  struct stat after = {0};
  uint8_t *bytes = NULL;
  bool ok = fstat(fd, &before) == 0 && S_ISREG(before.st_mode) && before.st_nlink == 1 && before.st_size > 0
            && (uint64_t)before.st_size <= maximum && (uint64_t)before.st_size <= SIZE_MAX;
  if (ok) {
    *size = (size_t)before.st_size;
    bytes = malloc(*size);
    ok = bytes != NULL;
  }
  size_t offset = 0;
  while (ok && offset != *size) {
    const ssize_t count = read(fd, bytes + offset, *size - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      ok = false;
      break;
    }
    offset += (size_t)count;
  }
  if (ok)
    ok = fstat(fd, &after) == 0 && after.st_dev == before.st_dev && after.st_ino == before.st_ino && after.st_mode == before.st_mode
         && after.st_nlink == before.st_nlink && after.st_size == before.st_size && after.st_mtim.tv_sec == before.st_mtim.tv_sec
         && after.st_mtim.tv_nsec == before.st_mtim.tv_nsec && after.st_ctim.tv_sec == before.st_ctim.tv_sec
         && after.st_ctim.tv_nsec == before.st_ctim.tv_nsec;
  if (close(fd) != 0)
    ok = false;
  if (!ok) {
    free(bytes);
    bytes = NULL;
  }
  return bytes;
}

int main(int argc, char **argv)
{
  if (argc != 4 && argc != 7) {
    fputs("usage: test_oai_memprof_archive_producer ARCHIVE CONFIG OPENING [3 SEED THRESHOLD]\n", stderr);
    return 64;
  }
  uint8_t mode_id = OAI_MEMPROF_CORE_EXACT_EVENTS;
  uint64_t sample_seed = 0;
  uint64_t sample_threshold = 0;
  if (argc == 7) {
    CHECK(strcmp(argv[4], "3") == 0);
    mode_id = OAI_MEMPROF_CORE_SAMPLED;
    CHECK(parse_u64_argument(argv[5], false, &sample_seed));
    CHECK(parse_u64_argument(argv[6], true, &sample_threshold));
  }

  size_t configuration_size = 0;
  uint8_t *configuration = read_regular(argv[2], UINT32_C(1048576), &configuration_size);
  CHECK(configuration != NULL);
  size_t opening_size = 0;
  uint8_t *opening_bytes = read_regular(argv[3], OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE, &opening_size);
  CHECK(opening_bytes != NULL && opening_size == OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE);
  oai_memprof_container_v1_opening_header_t opening = {0};
  CHECK(oai_memprof_container_v1_opening_header_decode(&opening, opening_bytes, opening_size) == OAI_MEMPROF_CONTAINER_V1_OK);

  const int archive_fd = open(argv[1], O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  CHECK(archive_fd >= 0);
  const int streams_fd = openat(archive_fd, "streams", O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  CHECK(streams_fd >= 0);

  const oai_memprof_process_session_config_t config = {
      .directory_fd = streams_fd,
      .stream_file_name = "memory-lifetime.bin",
      .handoff_file_name = "process-handoff.bin",
      .configuration_bytes = configuration,
      .configuration_size = configuration_size,
      .runtime =
          {
              .core =
                  {
                      .process_generation = 1,
                      .table_entries = mode_id == OAI_MEMPROF_CORE_SAMPLED ? UINT64_C(512) : UINT64_C(0),
                      .sample_seed = sample_seed,
                      .sample_threshold = sample_threshold,
                      .max_threads = 1,
                      .ring_records = 64,
                      .table_probes = mode_id == OAI_MEMPROF_CORE_SAMPLED ? UINT32_C(8) : UINT32_C(0),
                      .mode_id = mode_id,
                  },
              .realloc_zero_policy_id = 1,
          },
      .opening_header = opening,
      .flush_records = 4,
      .flush_interval_ns = UINT64_C(1000000),
  };
  oai_memprof_process_session_t *session = NULL;
  CHECK(oai_memprof_process_session_start_v1(&config, &session) == OAI_MEMPROF_PROCESS_SESSION_OK);
  CHECK(session != NULL);

  volatile size_t malloc_size = 64;
  volatile size_t calloc_count = 2;
  volatile size_t calloc_size = 32;
  volatile size_t realloc_size = 96;
  void *malloc_result = malloc(malloc_size);
  CHECK(malloc_result != NULL);
  void *calloc_result = calloc(calloc_count, calloc_size);
  CHECK(calloc_result != NULL);
  void *realloc_result = realloc(malloc_result, realloc_size);
  CHECK(realloc_result != NULL);
  malloc_result = NULL;
  free(calloc_result);
  calloc_result = NULL;

  oai_memprof_process_session_result_t result = {0};
  CHECK(oai_memprof_process_session_finish_v1(session, UINT64_C(1000000000), &result) == OAI_MEMPROF_PROCESS_SESSION_OK);
  CHECK(result.status == OAI_MEMPROF_PROCESS_SESSION_OK && result.handoff_published);
  CHECK(result.writer.status == OAI_MEMPROF_STREAM_WRITER_OK
        && (mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS ? result.writer.record_count == 4
                                                     : (result.writer.record_count > 0 && result.writer.record_count <= 4))
        && result.writer.runtime_snapshot.ready_threads == 1);

  free(realloc_result);
  free(opening_bytes);
  free(configuration);
  CHECK(close(streams_fd) == 0);
  CHECK(close(archive_fd) == 0);
  printf("archive producer emitted %" PRIu64 " wrapped allocation events in mode %u\n",
         result.writer.record_count,
         (unsigned)mode_id);
  return 0;
}
