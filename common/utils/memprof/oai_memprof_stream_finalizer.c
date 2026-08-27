/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "oai_memprof_stream_finalizer.h"

#include "oai_memprof_wire.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define OAI_MEMPROF_FINALIZER_MAX_FILE_NAME_BYTES 127U

_Static_assert(sizeof(off_t) >= 8, "the stream finalizer requires 64-bit file offsets");

typedef struct oai_memprof_finalizer_context_s {
  const oai_memprof_stream_finalizer_config_t *config;
  oai_memprof_stream_finalizer_result_t result;
  int directory_fd;
  int file_fd;
} oai_memprof_finalizer_context_t;

static uint32_t load_u32_le(const uint8_t *source)
{
  return (uint32_t)source[0] | ((uint32_t)source[1] << 8) | ((uint32_t)source[2] << 16) | ((uint32_t)source[3] << 24);
}

static bool add_u64(uint64_t left, uint64_t right, uint64_t *result)
{
  if (left > UINT64_MAX - right)
    return false;
  *result = left + right;
  return true;
}

static bool multiply_u64(uint64_t left, uint64_t right, uint64_t *result)
{
  if (left != 0 && right > UINT64_MAX / left)
    return false;
  *result = left * right;
  return true;
}

static bool valid_file_name(const char *name)
{
  if (name == NULL || name[0] == '\0' || strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
    return false;
  size_t length = 0;
  while (name[length] != '\0') {
    const unsigned char value = (unsigned char)name[length];
    const bool allowed = (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9')
                         || value == '.' || value == '_' || value == '-';
    if (!allowed || length == OAI_MEMPROF_FINALIZER_MAX_FILE_NAME_BYTES)
      return false;
    ++length;
  }
  return length != 0;
}

static void fail(oai_memprof_finalizer_context_t *context, oai_memprof_stream_finalizer_status_t status, int system_errno)
{
  if (context->result.status == OAI_MEMPROF_STREAM_FINALIZER_OK) {
    context->result.status = status;
    context->result.system_errno = system_errno;
  }
}

static bool same_file(const struct stat *left, const struct stat *right)
{
  return S_ISREG(left->st_mode) && S_ISREG(right->st_mode) && left->st_nlink == 1 && right->st_nlink == 1
         && left->st_dev == right->st_dev && left->st_ino == right->st_ino;
}

static bool validate_identity(oai_memprof_finalizer_context_t *context, uint64_t expected_size)
{
  struct stat descriptor_status;
  struct stat path_status;
  if (fstat(context->file_fd, &descriptor_status) != 0
      || fstatat(context->directory_fd, context->config->file_name, &path_status, AT_SYMLINK_NOFOLLOW) != 0) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    return false;
  }
  if (!same_file(&descriptor_status, &path_status) || descriptor_status.st_size < 0
      || (uint64_t)descriptor_status.st_dev != context->config->prefooter.file_device
      || (uint64_t)descriptor_status.st_ino != context->config->prefooter.file_inode
      || (uint64_t)descriptor_status.st_size != expected_size || descriptor_status.st_size != path_status.st_size) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_IDENTITY_MISMATCH, EIO);
    return false;
  }
  return true;
}

static bool validate_path_after_close(oai_memprof_finalizer_context_t *context, uint64_t expected_size)
{
  struct stat path_status;
  if (fstatat(context->directory_fd, context->config->file_name, &path_status, AT_SYMLINK_NOFOLLOW) != 0) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    return false;
  }
  if (!S_ISREG(path_status.st_mode) || path_status.st_nlink != 1 || path_status.st_size < 0
      || (uint64_t)path_status.st_dev != context->config->prefooter.file_device
      || (uint64_t)path_status.st_ino != context->config->prefooter.file_inode || (uint64_t)path_status.st_size != expected_size) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_IDENTITY_MISMATCH, EIO);
    return false;
  }
  return true;
}

static bool map_file(oai_memprof_finalizer_context_t *context, uint64_t file_size, int protection, const uint8_t **bytes)
{
  if (file_size == 0 || file_size > SIZE_MAX) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_STREAM_LIMIT, EFBIG);
    return false;
  }
  void *mapping = mmap(NULL, (size_t)file_size, protection, MAP_PRIVATE, context->file_fd, 0);
  if (mapping == MAP_FAILED) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    return false;
  }
  *bytes = mapping;
  return true;
}

static bool unmap_file(oai_memprof_finalizer_context_t *context, const uint8_t *bytes, uint64_t file_size)
{
  if (munmap((void *)bytes, (size_t)file_size) != 0) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    return false;
  }
  return true;
}

static int compare_event_key(uint16_t event_kind, uint16_t api_id, const oai_memprof_container_v1_event_total_entry_t *entry)
{
  if (event_kind != entry->event_kind)
    return event_kind < entry->event_kind ? -1 : 1;
  if (api_id != entry->api_id)
    return api_id < entry->api_id ? -1 : 1;
  return 0;
}

static bool find_event_entry(const oai_memprof_stream_finalizer_config_t *config,
                             uint16_t event_kind,
                             uint16_t api_id,
                             size_t *index)
{
  size_t low = 0;
  size_t high = config->event_entry_count;
  while (low < high) {
    const size_t middle = low + (high - low) / 2U;
    const int comparison = compare_event_key(event_kind, api_id, &config->event_entries[middle]);
    if (comparison == 0) {
      *index = middle;
      return true;
    }
    if (comparison < 0)
      high = middle;
    else
      low = middle + 1U;
  }
  return false;
}

static uint64_t saturating_add(uint64_t left, uint64_t right, bool *saturated)
{
  if (right > UINT64_MAX - left) {
    *saturated = true;
    return UINT64_MAX;
  }
  return left + right;
}

static bool validate_tables(oai_memprof_finalizer_context_t *context, const oai_memprof_container_v1_opening_header_t *opening)
{
  const oai_memprof_stream_finalizer_config_t *config = context->config;
  const oai_memprof_container_v1_trailer_header_t *header = &config->trailer_header;
  if (config->event_entry_count != header->event_entry_count || config->diagnostic_entry_count != header->diagnostic_entry_count
      || config->object_entry_count != header->object_entry_count) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_INVALID_CONFIGURATION, 0);
    return false;
  }
  if ((config->event_entry_count != 0 && config->event_entries == NULL)
      || (config->diagnostic_entry_count != 0 && config->diagnostic_entries == NULL)
      || (config->object_entry_count != 0 && config->object_entries == NULL)) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_INVALID_ARGUMENT, 0);
    return false;
  }

  uint64_t event_sum = 0;
  uint8_t encoded_event[OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE];
  for (size_t index = 0; index < config->event_entry_count; ++index) {
    const oai_memprof_container_v1_event_total_entry_t *entry = &config->event_entries[index];
    if (oai_memprof_container_v1_event_total_entry_encode(entry, encoded_event, sizeof(encoded_event))
        != OAI_MEMPROF_CONTAINER_V1_OK) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
      return false;
    }
    if (index != 0 && compare_event_key(entry->event_kind, entry->api_id, &config->event_entries[index - 1U]) <= 0) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
      return false;
    }
    if (!add_u64(event_sum, entry->record_count, &event_sum)) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_STREAM_LIMIT, 0);
      return false;
    }
  }
  if (event_sum != header->record_count) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
    return false;
  }

  uint64_t loss_sum = 0;
  uint64_t bypass_sum = 0;
  uint64_t saturated_instances = 0;
  bool aggregate_saturated = false;
  bool partial = false;
  uint8_t encoded_diagnostic[OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE];
  for (size_t index = 0; index < config->diagnostic_entry_count; ++index) {
    const oai_memprof_container_v1_diagnostic_total_entry_t *entry = &config->diagnostic_entries[index];
    if (oai_memprof_container_v1_diagnostic_total_entry_encode(entry, encoded_diagnostic, sizeof(encoded_diagnostic))
        != OAI_MEMPROF_CONTAINER_V1_OK) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
      return false;
    }
    if (index != 0 && entry->reason_id <= config->diagnostic_entries[index - 1U].reason_id) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
      return false;
    }
    if ((entry->class_flags & UINT16_C(1)) != 0)
      loss_sum = saturating_add(loss_sum, entry->saturating_total, &aggregate_saturated);
    if ((entry->class_flags & UINT16_C(2)) != 0)
      bypass_sum = saturating_add(bypass_sum, entry->saturating_total, &aggregate_saturated);
    if (!add_u64(saturated_instances, entry->saturated_counter_instances, &saturated_instances)) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_STREAM_LIMIT, 0);
      return false;
    }
    aggregate_saturated = aggregate_saturated || (entry->summary_flags & UINT32_C(1)) != 0;
    partial = partial || (entry->summary_flags & UINT32_C(2)) != 0;
  }
  if (loss_sum != header->diagnostic_loss_sum || bypass_sum != header->diagnostic_bypass_sum
      || saturated_instances != header->saturated_counter_instances
      || aggregate_saturated != ((header->terminal_flags & (UINT64_C(1) << 12)) != 0)
      || partial != ((header->terminal_flags & (UINT64_C(1) << 13)) != 0)) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
    return false;
  }

  uint8_t encoded_object[OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE];
  for (size_t index = 0; index < config->object_entry_count; ++index) {
    const oai_memprof_container_v1_object_entry_t *entry = &config->object_entries[index];
    if (oai_memprof_container_v1_object_entry_encode(entry, encoded_object, sizeof(encoded_object))
        != OAI_MEMPROF_CONTAINER_V1_OK) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
      return false;
    }
    if (index != 0 && entry->object_kind <= config->object_entries[index - 1U].object_kind) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
      return false;
    }
  }

  if (header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_COMPLETE) {
    if (config->object_entry_count != 12U) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
      return false;
    }
    for (size_t index = 0; index < 12U; ++index)
      if (config->object_entries[index].object_kind != index + 1U) {
        fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
        return false;
      }
  } else {
    bool diagnostic_object_present = false;
    for (size_t index = 0; index < config->object_entry_count; ++index)
      diagnostic_object_present = diagnostic_object_present || config->object_entries[index].object_kind == 11U;
    if (config->object_entry_count < 2U || config->object_entries[0].object_kind != 1U
        || config->object_entries[config->object_entry_count - 1U].object_kind != 12U
        || (config->diagnostic_entry_count != 0 && !diagnostic_object_present)) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
      return false;
    }
  }

  for (size_t index = 0; index < config->object_entry_count; ++index) {
    const oai_memprof_container_v1_object_entry_t *entry = &config->object_entries[index];
    const uint8_t *expected = NULL;
    if (entry->object_kind == 1U)
      expected = opening->schema_bundle_definition_sha256;
    else if (entry->object_kind == 2U)
      expected = opening->api_catalog_definition_sha256;
    else if (entry->object_kind == 10U)
      expected = opening->configuration_instance_sha256;
    if (expected != NULL && memcmp(entry->sha256, expected, 32U) != 0) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
      return false;
    }
  }
  return true;
}

static bool validate_prefix(oai_memprof_finalizer_context_t *context,
                            const uint8_t *bytes,
                            uint64_t byte_count,
                            oai_memprof_container_v1_opening_header_t *opening)
{
  const oai_memprof_stream_finalizer_config_t *config = context->config;
  if (byte_count < OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE
      || oai_memprof_container_v1_opening_header_decode(opening, bytes, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE)
             != OAI_MEMPROF_CONTAINER_V1_OK) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_PREFOOTER_INVALID, 0);
    return false;
  }
  const oai_memprof_container_v1_trailer_header_t *trailer = &config->trailer_header;
  if (opening->process_generation != trailer->process_generation || opening->scope_kind != trailer->scope_kind
      || opening->process_generation != config->prefooter.runtime_snapshot.process_generation
      || opening->start_counter != trailer->active_start_counter
      || opening->start_monotonic_raw_ns != trailer->active_start_monotonic_raw_ns) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
    return false;
  }

  uint64_t *derived = NULL;
  if (config->event_entry_count != 0) {
    if (config->event_entry_count > SIZE_MAX / sizeof(*derived)) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_STREAM_LIMIT, 0);
      return false;
    }
    derived = calloc(config->event_entry_count, sizeof(*derived));
    if (derived == NULL) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_NO_MEMORY, errno);
      return false;
    }
  }

  uint64_t offset = OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE;
  uint64_t expected_sequence = 0;
  uint64_t record_count = 0;
  uint64_t payload_bytes = 0;
  bool valid = true;
  bool table_valid = true;
  while (offset < byte_count) {
    if (byte_count - offset < OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE) {
      valid = false;
      break;
    }
    const uint8_t *chunk_wire = bytes + (size_t)offset;
    const uint32_t records = load_u32_le(chunk_wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_RECORD_COUNT_OFFSET);
    uint64_t payload = 0;
    uint64_t chunk_bytes = 0;
    if (!multiply_u64(records, OAI_MEMPROF_CONTAINER_V1_EVENT_RECORD_SIZE, &payload)
        || !add_u64(OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE, payload, &chunk_bytes) || chunk_bytes > byte_count - offset
        || payload > SIZE_MAX) {
      valid = false;
      break;
    }
    oai_memprof_container_v1_chunk_header_t chunk = {0};
    const uint8_t *payload_wire = chunk_wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE;
    if (oai_memprof_container_v1_chunk_header_decode(&chunk,
                                                     chunk_wire,
                                                     OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE,
                                                     payload_wire,
                                                     (size_t)payload)
            != OAI_MEMPROF_CONTAINER_V1_OK
        || chunk.writer_chunk_sequence != expected_sequence || chunk.record_count != records) {
      valid = false;
      break;
    }
    for (uint32_t record_index = 0; record_index < records; ++record_index) {
      oai_memprof_event_v1_t event = {0};
      if (oai_memprof_event_v1_decode(&event,
                                      payload_wire + (size_t)record_index * OAI_MEMPROF_EVENT_V1_WIRE_SIZE,
                                      OAI_MEMPROF_EVENT_V1_WIRE_SIZE)
          != OAI_MEMPROF_WIRE_OK) {
        valid = false;
        break;
      }
      size_t entry_index = 0;
      if (!find_event_entry(config, event.event_kind, event.api_id, &entry_index) || derived[entry_index] == UINT64_MAX) {
        table_valid = false;
        valid = false;
        break;
      }
      ++derived[entry_index];
    }
    if (!valid)
      break;
    ++expected_sequence;
    if (!add_u64(record_count, records, &record_count) || !add_u64(payload_bytes, payload, &payload_bytes)
        || !add_u64(offset, chunk_bytes, &offset)) {
      valid = false;
      break;
    }
  }
  if (valid) {
    valid = offset == byte_count && expected_sequence == config->prefooter.chunk_count
            && record_count == config->prefooter.record_count && payload_bytes == config->prefooter.payload_bytes
            && byte_count == config->prefooter.stream_bytes && byte_count == trailer->chunks_end_offset
            && expected_sequence == trailer->chunk_count && record_count == trailer->record_count
            && payload_bytes == trailer->payload_bytes;
  }
  if (valid)
    for (size_t index = 0; index < config->event_entry_count; ++index)
      if (derived[index] != config->event_entries[index].record_count) {
        table_valid = false;
        valid = false;
        break;
      }
  free(derived);
  if (!table_valid)
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_TABLE_ERROR, 0);
  else if (!valid)
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_PREFOOTER_INVALID, 0);
  return valid && table_valid;
}

static bool build_trailer(oai_memprof_finalizer_context_t *context, uint8_t **trailer_bytes, size_t *trailer_size)
{
  const oai_memprof_stream_finalizer_config_t *config = context->config;
  if (config->trailer_header.trailer_body_bytes > SIZE_MAX) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_STREAM_LIMIT, 0);
    return false;
  }
  *trailer_size = (size_t)config->trailer_header.trailer_body_bytes;
  uint8_t *bytes = malloc(*trailer_size);
  if (bytes == NULL) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_NO_MEMORY, errno);
    return false;
  }
  size_t offset = 0;
  if (oai_memprof_container_v1_trailer_header_encode(&config->trailer_header, bytes, OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE)
      != OAI_MEMPROF_CONTAINER_V1_OK) {
    free(bytes);
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
    return false;
  }
  offset += OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE;
  for (size_t index = 0; index < config->event_entry_count; ++index) {
    if (oai_memprof_container_v1_event_total_entry_encode(&config->event_entries[index],
                                                          bytes + offset,
                                                          OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE)
        != OAI_MEMPROF_CONTAINER_V1_OK) {
      free(bytes);
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
      return false;
    }
    offset += OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE;
  }
  for (size_t index = 0; index < config->diagnostic_entry_count; ++index) {
    if (oai_memprof_container_v1_diagnostic_total_entry_encode(&config->diagnostic_entries[index],
                                                               bytes + offset,
                                                               OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE)
        != OAI_MEMPROF_CONTAINER_V1_OK) {
      free(bytes);
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
      return false;
    }
    offset += OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE;
  }
  for (size_t index = 0; index < config->object_entry_count; ++index) {
    if (oai_memprof_container_v1_object_entry_encode(&config->object_entries[index],
                                                     bytes + offset,
                                                     OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE)
        != OAI_MEMPROF_CONTAINER_V1_OK) {
      free(bytes);
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
      return false;
    }
    offset += OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE;
  }
  if (offset != *trailer_size) {
    free(bytes);
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_INVALID_CONFIGURATION, 0);
    return false;
  }
  *trailer_bytes = bytes;
  return true;
}

static bool pwrite_all(oai_memprof_finalizer_context_t *context, const uint8_t *bytes, size_t size, uint64_t start_offset)
{
  size_t written_total = 0;
  while (written_total != size) {
    const uint64_t absolute = start_offset + written_total;
    if (absolute > INT64_MAX) {
      fail(context, OAI_MEMPROF_STREAM_FINALIZER_STREAM_LIMIT, EFBIG);
      return false;
    }
    const ssize_t written = pwrite(context->file_fd, bytes + written_total, size - written_total, (off_t)absolute);
    if (written > 0) {
      written_total += (size_t)written;
      context->result.appended_bytes += (uint64_t)written;
      continue;
    }
    if (written < 0 && errno == EINTR)
      continue;
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, written == 0 ? EIO : errno);
    return false;
  }
  return true;
}

static bool verify_complete_stream(oai_memprof_finalizer_context_t *context,
                                   const uint8_t *bytes,
                                   uint64_t stream_bytes,
                                   const uint8_t *trailer_bytes,
                                   size_t trailer_size,
                                   const uint8_t footer_bytes[OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE])
{
  const oai_memprof_stream_finalizer_config_t *config = context->config;
  if (stream_bytes != context->result.stream_bytes
      || memcmp(bytes + (size_t)config->prefooter.stream_bytes, trailer_bytes, trailer_size) != 0
      || memcmp(bytes + (size_t)stream_bytes - OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE,
                footer_bytes,
                OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE)
             != 0) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_PREFOOTER_INVALID, 0);
    return false;
  }

  oai_memprof_container_v1_footer_t footer = {0};
  if (oai_memprof_container_v1_footer_decode(&footer,
                                             bytes + (size_t)stream_bytes - OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE,
                                             OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE)
      != OAI_MEMPROF_CONTAINER_V1_OK) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
    return false;
  }
  oai_memprof_container_v1_trailer_header_t trailer = {0};
  if (oai_memprof_container_v1_trailer_header_decode(&trailer,
                                                     bytes + (size_t)config->prefooter.stream_bytes,
                                                     OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE)
      != OAI_MEMPROF_CONTAINER_V1_OK) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
    return false;
  }

  uint8_t digest[32];
  if (oai_memprof_container_v1_sha256(bytes, (size_t)config->prefooter.stream_bytes, digest) != OAI_MEMPROF_CONTAINER_V1_OK
      || memcmp(digest, footer.prefix_sha256, sizeof(digest)) != 0
      || oai_memprof_container_v1_sha256(bytes, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE, digest) != OAI_MEMPROF_CONTAINER_V1_OK
      || memcmp(digest, footer.opening_header_sha256, sizeof(digest)) != 0
      || oai_memprof_container_v1_sha256(trailer_bytes, trailer_size, digest) != OAI_MEMPROF_CONTAINER_V1_OK
      || memcmp(digest, footer.trailer_body_sha256, sizeof(digest)) != 0) {
    fail(context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
    return false;
  }

  oai_memprof_container_v1_opening_header_t opening = {0};
  return validate_prefix(context, bytes, config->prefooter.stream_bytes, &opening);
}

static bool offline_terminal_outcome_matches_writer(const oai_memprof_stream_finalizer_config_t *config)
{
  const oai_memprof_stream_writer_result_t *prefooter = &config->prefooter;
  const oai_memprof_container_v1_trailer_header_t *trailer = &config->trailer_header;
  switch (prefooter->status) {
    case OAI_MEMPROF_STREAM_WRITER_OK:
      return prefooter->runtime_status == OAI_MEMPROF_CORE_OK
             && trailer->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_COMPLETE
             && trailer->terminal_reason_code == OAI_MEMPROF_CONTAINER_V1_REASON_NONE
             && trailer->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED;
    case OAI_MEMPROF_STREAM_WRITER_IO_ERROR:
      return trailer->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED
             && trailer->terminal_reason_code == OAI_MEMPROF_CONTAINER_V1_REASON_PAYLOAD_IO_FAILED_AT_SAFE_BOUNDARY
             && trailer->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_IO_FAILED_AT_SAFE_BOUNDARY_AND_CLOSED_VERIFIED;
    case OAI_MEMPROF_STREAM_WRITER_CLOCK_ERROR:
      return trailer->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_INCOMPLETE
             && trailer->terminal_reason_code == OAI_MEMPROF_CONTAINER_V1_REASON_COUNTER_OR_TIME_INVALID
             && trailer->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED;
    default:
      return false;
  }
}

static bool valid_configuration(const oai_memprof_stream_finalizer_config_t *config
#ifndef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
                                ,
                                bool offline
#endif
)
{
  if (config == NULL || config->directory_fd < 0 || !valid_file_name(config->file_name))
    return false;
  if (!config->prefooter.prefooter_closed || config->prefooter.runtime_snapshot.state != OAI_MEMPROF_CORE_DRAINING
      || config->prefooter.stream_bytes < OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE || config->prefooter.file_device == 0
      || config->prefooter.file_inode == 0)
    return false;
  if (config->event_entry_count > UINT32_MAX || config->diagnostic_entry_count > UINT32_MAX
      || config->object_entry_count > UINT32_MAX)
    return false;
#ifdef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
  return config->authenticated_prefix_sha256 != NULL && config->authenticated_opening_header_sha256 != NULL
         && offline_terminal_outcome_matches_writer(config);
#else
  if (offline)
    return config->authenticated_prefix_sha256 != NULL && config->authenticated_opening_header_sha256 != NULL
           && offline_terminal_outcome_matches_writer(config);
  return config->prefooter.status == OAI_MEMPROF_STREAM_WRITER_OK && config->prefooter.runtime_status == OAI_MEMPROF_CORE_OK;
#endif
}

static oai_memprof_stream_finalizer_status_t finalize(const oai_memprof_stream_finalizer_config_t *config,
                                                      oai_memprof_stream_finalizer_result_t *result
#ifndef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
                                                      ,
                                                      bool offline
#endif
)
{
  if (result == NULL)
    return OAI_MEMPROF_STREAM_FINALIZER_INVALID_ARGUMENT;

  oai_memprof_finalizer_context_t context = {
      .config = config,
      .directory_fd = -1,
      .file_fd = -1,
      .result =
          {
              .status = OAI_MEMPROF_STREAM_FINALIZER_OK,
              .runtime_status = OAI_MEMPROF_CORE_INVALID_STATE,
          },
  };
  if (!valid_configuration(config
#ifndef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
                           ,
                           offline
#endif
                           )) {
    context.result.status =
        config == NULL ? OAI_MEMPROF_STREAM_FINALIZER_INVALID_ARGUMENT : OAI_MEMPROF_STREAM_FINALIZER_INVALID_CONFIGURATION;
    *result = context.result;
    return context.result.status;
  }
  context.result.file_device = config->prefooter.file_device;
  context.result.file_inode = config->prefooter.file_inode;

#ifdef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
  context.result.runtime_status = config->prefooter.runtime_status;
  if (config->prefooter.runtime_snapshot.process_generation != config->trailer_header.process_generation
      || (config->prefooter.runtime_snapshot.mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS
          && config->prefooter.runtime_snapshot.emitted_events != config->prefooter.record_count)) {
    context.result.status = OAI_MEMPROF_STREAM_FINALIZER_RUNTIME_ERROR;
    *result = context.result;
    return context.result.status;
  }
#else
  if (offline) {
    context.result.runtime_status = config->prefooter.runtime_status;
    if (config->prefooter.runtime_snapshot.process_generation != config->trailer_header.process_generation
        || (config->prefooter.runtime_snapshot.mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS
            && config->prefooter.runtime_snapshot.emitted_events != config->prefooter.record_count)) {
      context.result.status = OAI_MEMPROF_STREAM_FINALIZER_RUNTIME_ERROR;
      *result = context.result;
      return context.result.status;
    }
  } else {
    oai_memprof_core_snapshot_t runtime_snapshot = {0};
    context.result.runtime_status = oai_memprof_active_runtime_snapshot_v1(&runtime_snapshot);
    if (context.result.runtime_status != OAI_MEMPROF_CORE_OK || runtime_snapshot.state != OAI_MEMPROF_CORE_DRAINING
        || runtime_snapshot.process_generation != config->prefooter.runtime_snapshot.process_generation
        || runtime_snapshot.process_generation != config->trailer_header.process_generation
        || runtime_snapshot.mode_id != config->prefooter.runtime_snapshot.mode_id
        || runtime_snapshot.emitted_events != config->prefooter.runtime_snapshot.emitted_events) {
      context.result.status = OAI_MEMPROF_STREAM_FINALIZER_RUNTIME_ERROR;
      *result = context.result;
      return context.result.status;
    }
    if (runtime_snapshot.mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS
        && runtime_snapshot.emitted_events != config->prefooter.record_count) {
      context.result.status = OAI_MEMPROF_STREAM_FINALIZER_RUNTIME_ERROR;
      *result = context.result;
      return context.result.status;
    }
  }
#endif

  context.directory_fd = fcntl(config->directory_fd, F_DUPFD_CLOEXEC, 3);
  struct stat directory_status;
  if (context.directory_fd < 0 || fstat(context.directory_fd, &directory_status) != 0 || !S_ISDIR(directory_status.st_mode)) {
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno == 0 ? ENOTDIR : errno);
    goto cleanup;
  }
  context.file_fd = openat(context.directory_fd, config->file_name, O_RDWR | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (context.file_fd < 0) {
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    goto cleanup;
  }
  if (!validate_identity(&context, config->prefooter.stream_bytes))
    goto cleanup;

  const uint8_t *prefix = NULL;
  if (!map_file(&context, config->prefooter.stream_bytes, PROT_READ, &prefix))
    goto cleanup;
  oai_memprof_container_v1_opening_header_t opening = {0};
  bool prefix_valid = false;
  if (oai_memprof_container_v1_opening_header_decode(&opening, prefix, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE)
          == OAI_MEMPROF_CONTAINER_V1_OK
      && validate_tables(&context, &opening))
    prefix_valid = validate_prefix(&context, prefix, config->prefooter.stream_bytes, &opening);
  else if (context.result.status == OAI_MEMPROF_STREAM_FINALIZER_OK)
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_PREFOOTER_INVALID, 0);
  uint8_t prefix_sha256[32];
  uint8_t opening_sha256[32];
  if (prefix_valid
      && (oai_memprof_container_v1_sha256(prefix, (size_t)config->prefooter.stream_bytes, prefix_sha256)
              != OAI_MEMPROF_CONTAINER_V1_OK
          || oai_memprof_container_v1_sha256(prefix, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE, opening_sha256)
                 != OAI_MEMPROF_CONTAINER_V1_OK))
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
  if (prefix_valid && context.result.status == OAI_MEMPROF_STREAM_FINALIZER_OK
#ifndef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
      && offline
#endif
      && (memcmp(prefix_sha256, config->authenticated_prefix_sha256, sizeof(prefix_sha256)) != 0
          || memcmp(opening_sha256, config->authenticated_opening_header_sha256, sizeof(opening_sha256)) != 0))
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_AUTHENTICATION_MISMATCH, EILSEQ);
  if (!unmap_file(&context, prefix, config->prefooter.stream_bytes) || !prefix_valid
      || context.result.status != OAI_MEMPROF_STREAM_FINALIZER_OK)
    goto cleanup;

  uint8_t *trailer_bytes = NULL;
  size_t trailer_size = 0;
  if (!build_trailer(&context, &trailer_bytes, &trailer_size))
    goto cleanup;
  uint8_t trailer_sha256[32];
  if (oai_memprof_container_v1_sha256(trailer_bytes, trailer_size, trailer_sha256) != OAI_MEMPROF_CONTAINER_V1_OK) {
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
    free(trailer_bytes);
    goto cleanup;
  }

  uint64_t without_footer = 0;
  uint64_t final_stream_bytes = 0;
  if (!add_u64(config->prefooter.stream_bytes, trailer_size, &without_footer)
      || !add_u64(without_footer, OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE, &final_stream_bytes) || final_stream_bytes > INT64_MAX) {
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_STREAM_LIMIT, EFBIG);
    free(trailer_bytes);
    goto cleanup;
  }
  context.result.stream_bytes = final_stream_bytes;
  oai_memprof_container_v1_footer_t footer = {
      .trailer_offset = config->prefooter.stream_bytes,
      .trailer_body_bytes = trailer_size,
      .stream_bytes = final_stream_bytes,
      .prefix_bytes = config->prefooter.stream_bytes,
      .chunk_count = config->prefooter.chunk_count,
      .record_count = config->prefooter.record_count,
  };
  memcpy(footer.prefix_sha256, prefix_sha256, sizeof(footer.prefix_sha256));
  memcpy(footer.trailer_body_sha256, trailer_sha256, sizeof(footer.trailer_body_sha256));
  memcpy(footer.opening_header_sha256, opening_sha256, sizeof(footer.opening_header_sha256));
  uint8_t footer_bytes[OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE];
  if (oai_memprof_container_v1_footer_encode(&footer, footer_bytes, sizeof(footer_bytes)) != OAI_MEMPROF_CONTAINER_V1_OK) {
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_CODEC_ERROR, 0);
    free(trailer_bytes);
    goto cleanup;
  }

  const bool appended = pwrite_all(&context, trailer_bytes, trailer_size, config->prefooter.stream_bytes)
                        && pwrite_all(&context, footer_bytes, sizeof(footer_bytes), without_footer);
  if (!appended || fsync(context.file_fd) != 0) {
    if (context.result.status == OAI_MEMPROF_STREAM_FINALIZER_OK)
      fail(&context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    free(trailer_bytes);
    goto cleanup;
  }
  if (!validate_identity(&context, final_stream_bytes)) {
    free(trailer_bytes);
    goto cleanup;
  }

  const uint8_t *complete = NULL;
  if (!map_file(&context, final_stream_bytes, PROT_READ, &complete)) {
    free(trailer_bytes);
    goto cleanup;
  }
  const bool verified = verify_complete_stream(&context, complete, final_stream_bytes, trailer_bytes, trailer_size, footer_bytes);
  const bool complete_unmapped = unmap_file(&context, complete, final_stream_bytes);
  free(trailer_bytes);
  if (!verified || !complete_unmapped || context.result.status != OAI_MEMPROF_STREAM_FINALIZER_OK)
    goto cleanup;
  context.result.stream_verified = true;

  if (close(context.file_fd) != 0) {
    context.file_fd = -1;
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    goto cleanup;
  }
  context.file_fd = -1;
  if (fsync(context.directory_fd) != 0 || !validate_path_after_close(&context, final_stream_bytes)) {
    if (context.result.status == OAI_MEMPROF_STREAM_FINALIZER_OK)
      fail(&context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    goto cleanup;
  }
  if (close(context.directory_fd) != 0) {
    context.directory_fd = -1;
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
    goto cleanup;
  }
  context.directory_fd = -1;

#ifndef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
  if (!offline && config->trailer_header.lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_COMPLETE) {
    context.result.runtime_status = oai_memprof_active_runtime_complete_v1();
    if (context.result.runtime_status != OAI_MEMPROF_CORE_OK) {
      fail(&context, OAI_MEMPROF_STREAM_FINALIZER_RUNTIME_ERROR, 0);
      goto cleanup;
    }
    context.result.runtime_complete = true;
  }
#endif
  *result = context.result;
  return context.result.status;

cleanup:
  if (context.file_fd >= 0 && close(context.file_fd) != 0)
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
  if (context.directory_fd >= 0 && close(context.directory_fd) != 0)
    fail(&context, OAI_MEMPROF_STREAM_FINALIZER_IO_ERROR, errno);
  *result = context.result;
  return context.result.status;
}

#ifndef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
oai_memprof_stream_finalizer_status_t oai_memprof_stream_finalize_v1(const oai_memprof_stream_finalizer_config_t *config,
                                                                     oai_memprof_stream_finalizer_result_t *result)
{
  return finalize(config, result, false);
}
#endif

oai_memprof_stream_finalizer_status_t oai_memprof_stream_finalize_offline_v1(const oai_memprof_stream_finalizer_config_t *config,
                                                                             oai_memprof_stream_finalizer_result_t *result)
{
#ifdef OAI_MEMPROF_STREAM_FINALIZER_OFFLINE_ONLY
  return finalize(config, result);
#else
  return finalize(config, result, true);
#endif
}
