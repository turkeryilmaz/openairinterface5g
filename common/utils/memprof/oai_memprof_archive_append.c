/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "oai_memprof_process_handoff.h"
#include "oai_memprof_stream_finalizer.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define TRAILER_LIMIT_BYTES UINT64_C(1048576)

typedef struct immutable_file_s {
  uint8_t *bytes;
  size_t size;
} immutable_file_t;

static uint32_t load_u32_le(const uint8_t *source)
{
  return (uint32_t)source[0] | ((uint32_t)source[1] << 8) | ((uint32_t)source[2] << 16) | ((uint32_t)source[3] << 24);
}

static int hex_digit(unsigned char value)
{
  if (value >= '0' && value <= '9')
    return value - '0';
  if (value >= 'a' && value <= 'f')
    return value - 'a' + 10;
  if (value >= 'A' && value <= 'F')
    return value - 'A' + 10;
  return -1;
}

static bool decode_sha256_hex(const char *source, uint8_t digest[32])
{
  if (source == NULL || digest == NULL || strlen(source) != 64U)
    return false;
  for (size_t index = 0; index < 32U; ++index) {
    const int high = hex_digit((unsigned char)source[index * 2U]);
    const int low = hex_digit((unsigned char)source[index * 2U + 1U]);
    if (high < 0 || low < 0)
      return false;
    digest[index] = (uint8_t)(((unsigned int)high << 4U) | (unsigned int)low);
  }
  return true;
}

static bool valid_leaf(const char *name)
{
  if (name == NULL || name[0] == '\0' || strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
    return false;
  size_t length = 0;
  for (; name[length] != '\0'; ++length) {
    const unsigned char value = (unsigned char)name[length];
    const bool allowed = (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9')
                         || value == '.' || value == '_' || value == '-';
    if (!allowed || length == 127U)
      return false;
  }
  return length != 0;
}

static bool read_immutable_leaf(int directory_fd, const char *name, uint64_t maximum, immutable_file_t *result)
{
  if (!valid_leaf(name) || result == NULL)
    return false;
  const int fd = openat(directory_fd, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0)
    return false;
  struct stat status;
  bool ok = fstat(fd, &status) == 0 && S_ISREG(status.st_mode) && status.st_nlink == 1 && status.st_size > 0
            && (uint64_t)status.st_size <= maximum && (uint64_t)status.st_size <= SIZE_MAX;
  uint8_t *bytes = NULL;
  size_t size = 0;
  if (ok) {
    size = (size_t)status.st_size;
    bytes = malloc(size);
    ok = bytes != NULL;
  }
  size_t offset = 0;
  while (ok && offset != size) {
    const ssize_t count = read(fd, bytes + offset, size - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      ok = false;
      break;
    }
    offset += (size_t)count;
  }
  struct stat after;
  if (ok)
    ok = fstat(fd, &after) == 0 && after.st_dev == status.st_dev && after.st_ino == status.st_ino && after.st_size == status.st_size
         && after.st_mode == status.st_mode && after.st_nlink == status.st_nlink && after.st_mtim.tv_sec == status.st_mtim.tv_sec
         && after.st_mtim.tv_nsec == status.st_mtim.tv_nsec && after.st_ctim.tv_sec == status.st_ctim.tv_sec
         && after.st_ctim.tv_nsec == status.st_ctim.tv_nsec;
  struct stat leaf_status;
  if (ok)
    ok = fstatat(directory_fd, name, &leaf_status, AT_SYMLINK_NOFOLLOW) == 0 && S_ISREG(leaf_status.st_mode)
         && leaf_status.st_nlink == 1 && leaf_status.st_dev == status.st_dev && leaf_status.st_ino == status.st_ino
         && leaf_status.st_mode == status.st_mode && leaf_status.st_size == status.st_size;
  if (close(fd) != 0)
    ok = false;
  if (!ok) {
    free(bytes);
    return false;
  }
  result->bytes = bytes;
  result->size = size;
  return true;
}

static bool immutable_file_sha256_matches(const immutable_file_t *file, const uint8_t expected[32])
{
  uint8_t calculated[32] = {0};
  return file != NULL && expected != NULL
         && oai_memprof_container_v1_sha256(file->bytes, file->size, calculated) == OAI_MEMPROF_CONTAINER_V1_OK
         && memcmp(calculated, expected, sizeof(calculated)) == 0;
}

static bool multiply_size(size_t left, size_t right, size_t *result)
{
  if (left != 0 && right > SIZE_MAX / left)
    return false;
  *result = left * right;
  return true;
}

static bool add_size(size_t left, size_t right, size_t *result)
{
  if (right > SIZE_MAX - left)
    return false;
  *result = left + right;
  return true;
}

static bool decode_trailer(const immutable_file_t *file,
                           oai_memprof_container_v1_trailer_header_t *header,
                           oai_memprof_container_v1_event_total_entry_t **events,
                           oai_memprof_container_v1_diagnostic_total_entry_t **diagnostics,
                           oai_memprof_container_v1_object_entry_t **objects)
{
  if (file == NULL || header == NULL || events == NULL || diagnostics == NULL || objects == NULL
      || file->size < OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE
      || oai_memprof_container_v1_trailer_header_decode(header, file->bytes, OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE)
             != OAI_MEMPROF_CONTAINER_V1_OK
      || header->trailer_body_bytes != file->size)
    return false;
  size_t event_bytes = 0;
  size_t diagnostic_bytes = 0;
  size_t object_bytes = 0;
  size_t expected_diagnostic_offset = 0;
  size_t expected_object_offset = 0;
  size_t expected_size = 0;
  if (!multiply_size(header->event_entry_count, OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE, &event_bytes)
      || !multiply_size(header->diagnostic_entry_count, OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE, &diagnostic_bytes)
      || !multiply_size(header->object_entry_count, OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE, &object_bytes)
      || !add_size(OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE, event_bytes, &expected_diagnostic_offset)
      || !add_size(expected_diagnostic_offset, diagnostic_bytes, &expected_object_offset)
      || !add_size(expected_object_offset, object_bytes, &expected_size)
      || header->event_table_offset != OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE
      || header->diagnostic_table_offset != expected_diagnostic_offset || header->object_table_offset != expected_object_offset
      || expected_size != file->size)
    return false;

  if (header->event_entry_count != 0) {
    *events = calloc(header->event_entry_count, sizeof(**events));
    if (*events == NULL)
      return false;
  }
  if (header->diagnostic_entry_count != 0) {
    *diagnostics = calloc(header->diagnostic_entry_count, sizeof(**diagnostics));
    if (*diagnostics == NULL)
      return false;
  }
  if (header->object_entry_count != 0) {
    *objects = calloc(header->object_entry_count, sizeof(**objects));
    if (*objects == NULL)
      return false;
  }
  for (uint32_t index = 0; index < header->event_entry_count; ++index)
    if (oai_memprof_container_v1_event_total_entry_decode(
            &(*events)[index],
            file->bytes + header->event_table_offset + (size_t)index * OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE,
            OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE)
        != OAI_MEMPROF_CONTAINER_V1_OK)
      return false;
  for (uint32_t index = 0; index < header->diagnostic_entry_count; ++index)
    if (oai_memprof_container_v1_diagnostic_total_entry_decode(
            &(*diagnostics)[index],
            file->bytes + header->diagnostic_table_offset + (size_t)index * OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE,
            OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE)
        != OAI_MEMPROF_CONTAINER_V1_OK)
      return false;
  for (uint32_t index = 0; index < header->object_entry_count; ++index)
    if (oai_memprof_container_v1_object_entry_decode(
            &(*objects)[index],
            file->bytes + header->object_table_offset + (size_t)index * OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE,
            OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE)
        != OAI_MEMPROF_CONTAINER_V1_OK)
      return false;
  return true;
}

int main(int argc, char **argv)
{
  uint8_t expected_handoff_sha256[32] = {0};
  if (argc != 6 || !valid_leaf(argv[2]) || !valid_leaf(argv[3]) || !valid_leaf(argv[4])
      || !decode_sha256_hex(argv[5], expected_handoff_sha256)) {
    fputs("usage: oai_memprof_archive_append DIRECTORY STREAM_LEAF HANDOFF_LEAF TRAILER_LEAF HANDOFF_SHA256\n", stderr);
    return 64;
  }
  const int directory_fd = open(argv[1], O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  if (directory_fd < 0) {
    perror("archive directory");
    return 65;
  }
  immutable_file_t handoff_file = {0};
  immutable_file_t trailer_file = {0};
  const uint64_t handoff_limit = OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_WIRE_BYTES;
  bool ok = read_immutable_leaf(directory_fd, argv[3], handoff_limit, &handoff_file);
  if (ok)
    ok = immutable_file_sha256_matches(&handoff_file, expected_handoff_sha256);
  if (ok)
    ok = read_immutable_leaf(directory_fd, argv[4], TRAILER_LIMIT_BYTES, &trailer_file);
  oai_memprof_process_handoff_thread_v1_t *threads = NULL;
  oai_memprof_process_handoff_v1_t handoff = {0};
  if (ok) {
    const uint32_t thread_count = handoff_file.size >= 76U ? load_u32_le(handoff_file.bytes + 72U) : UINT32_MAX;
    if (thread_count > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_THREADS)
      ok = false;
    else if (thread_count != 0) {
      threads = calloc(thread_count, sizeof(*threads));
      ok = threads != NULL;
    }
    if (ok)
      ok = oai_memprof_process_handoff_v1_decode(&handoff, threads, thread_count, handoff_file.bytes, handoff_file.size)
           == OAI_MEMPROF_PROCESS_HANDOFF_OK;
  }

  oai_memprof_container_v1_trailer_header_t header = {0};
  oai_memprof_container_v1_event_total_entry_t *events = NULL;
  oai_memprof_container_v1_diagnostic_total_entry_t *diagnostics = NULL;
  oai_memprof_container_v1_object_entry_t *objects = NULL;
  if (ok)
    ok = decode_trailer(&trailer_file, &header, &events, &diagnostics, &objects);

  oai_memprof_stream_finalizer_result_t result = {0};
  oai_memprof_stream_finalizer_status_t finalizer_status = OAI_MEMPROF_STREAM_FINALIZER_INVALID_CONFIGURATION;
  if (ok) {
    const oai_memprof_stream_finalizer_config_t config = {
        .directory_fd = directory_fd,
        .file_name = argv[2],
        .prefooter = handoff.writer,
        .trailer_header = header,
        .event_entries = events,
        .event_entry_count = header.event_entry_count,
        .diagnostic_entries = diagnostics,
        .diagnostic_entry_count = header.diagnostic_entry_count,
        .object_entries = objects,
        .object_entry_count = header.object_entry_count,
        .authenticated_prefix_sha256 = handoff.prefix_sha256,
        .authenticated_opening_header_sha256 = handoff.opening_header_sha256,
    };
    finalizer_status = oai_memprof_stream_finalize_offline_v1(&config, &result);
    ok = finalizer_status == OAI_MEMPROF_STREAM_FINALIZER_OK && result.status == OAI_MEMPROF_STREAM_FINALIZER_OK
         && result.stream_verified && !result.runtime_complete;
  }
  free(objects);
  free(diagnostics);
  free(events);
  free(threads);
  free(trailer_file.bytes);
  free(handoff_file.bytes);
  if (close(directory_fd) != 0)
    ok = false;
  if (!ok) {
    fprintf(stderr,
            "archive append failed status=%u system_errno=%d appended_bytes=%" PRIu64 "\n",
            (unsigned)finalizer_status,
            result.system_errno,
            result.appended_bytes);
    return 1;
  }
  printf("archive append complete stream_bytes=%" PRIu64 " device=%" PRIu64 " inode=%" PRIu64 "\n",
         result.stream_bytes,
         result.file_device,
         result.file_inode);
  return 0;
}
