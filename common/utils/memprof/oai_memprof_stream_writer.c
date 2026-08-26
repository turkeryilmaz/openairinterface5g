/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "oai_memprof_stream_writer.h"

#include "oai_memprof_wire.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define OAI_MEMPROF_WRITER_START_POLL_NS UINT64_C(50000)
#define OAI_MEMPROF_WRITER_MAX_POLL_NS UINT64_C(1000000)
#define OAI_MEMPROF_WRITER_MAX_FILE_NAME_BYTES 127U

struct oai_memprof_stream_writer_s {
  _Atomic(bool) start_gate;
  _Atomic(bool) cancel;
  _Atomic(bool) stop;
  pthread_t thread;
  size_t mapped_bytes;
  uint8_t *payload;
  uint64_t flush_interval_ns;
  uint64_t chunk_sequence;
  uint64_t chunk_count;
  uint64_t record_count;
  uint64_t payload_bytes;
  uint64_t stream_bytes;
  uint64_t batch_started_ns;
  uint64_t file_device;
  uint64_t file_inode;
  uint32_t flush_records;
  uint32_t batch_count;
  int fd;
  int directory_fd;
  int system_errno;
  oai_memprof_stream_writer_status_t status;
  oai_memprof_clock_status_t clock_status;
  oai_memprof_clock_info_v1_t clock_info;
  oai_memprof_clock_sample_v1_t seal_before_sample;
  oai_memprof_clock_sample_v1_t seal_after_sample;
  oai_memprof_clock_sample_v1_t drain_complete_sample;
  oai_memprof_clock_sample_v1_t final_sample;
  char file_name[OAI_MEMPROF_WRITER_MAX_FILE_NAME_BYTES + 1U];
};

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

static void fail_writer(oai_memprof_stream_writer_t *writer, oai_memprof_stream_writer_status_t status, int system_errno)
{
  if (writer->status == OAI_MEMPROF_STREAM_WRITER_OK) {
    writer->status = status;
    writer->system_errno = system_errno;
  }
}

static void fail_clock(oai_memprof_stream_writer_t *writer, oai_memprof_clock_status_t status)
{
  if (writer->clock_status == OAI_MEMPROF_CLOCK_OK)
    writer->clock_status = status;
  fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_CLOCK_ERROR, 0);
}

static void capture_boundary_sample(oai_memprof_stream_writer_t *writer,
                                    const oai_memprof_clock_sample_v1_t *previous,
                                    oai_memprof_clock_sample_v1_t *sample)
{
  if (writer->clock_status != OAI_MEMPROF_CLOCK_OK)
    return;
  for (unsigned attempt = 0; attempt < 8U; ++attempt) {
    oai_memprof_clock_sample_v1_t value = {0};
    const oai_memprof_clock_status_t status = oai_memprof_clock_sample_v1(OAI_MEMPROF_STREAM_WRITER_CLOCK_BRACKET_NS, &value);
    if (status != OAI_MEMPROF_CLOCK_OK) {
      fail_clock(writer, status);
      return;
    }
    if (previous == NULL
        || (previous->monotonic_raw_after_ns <= value.monotonic_raw_before_ns && previous->counter < value.counter)) {
      *sample = value;
      return;
    }
  }
  fail_clock(writer, OAI_MEMPROF_CLOCK_SEQUENCE_ERROR);
}

static bool monotonic_now_ns(uint64_t *result)
{
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0)
    return false;
  if (value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000L)
    return false;
  const uint64_t seconds = (uint64_t)value.tv_sec;
  if (seconds > UINT64_MAX / UINT64_C(1000000000))
    return false;
  *result = seconds * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
  return true;
}

static bool write_all(oai_memprof_stream_writer_t *writer, const uint8_t *bytes, size_t size)
{
  if ((uint64_t)size > UINT64_MAX - writer->stream_bytes) {
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_STREAM_LIMIT, 0);
    return false;
  }
  size_t offset = 0;
  while (offset != size) {
    const ssize_t written = write(writer->fd, bytes + offset, size - offset);
    if (written > 0) {
      offset += (size_t)written;
      writer->stream_bytes += (uint64_t)written;
      continue;
    }
    if (written < 0 && errno == EINTR)
      continue;
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_IO_ERROR, written == 0 ? EIO : errno);
    return false;
  }
  return true;
}

static bool flush_batch(oai_memprof_stream_writer_t *writer)
{
  if (writer->batch_count == 0)
    return true;

  size_t payload_size = 0;
  if (!multiply_size(writer->batch_count, OAI_MEMPROF_EVENT_V1_WIRE_SIZE, &payload_size)) {
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_INVALID_CONFIGURATION, 0);
    return false;
  }

  if (writer->chunk_sequence == UINT64_MAX || writer->chunk_count == UINT64_MAX
      || writer->record_count > UINT64_MAX - writer->batch_count || writer->payload_bytes > UINT64_MAX - payload_size) {
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_STREAM_LIMIT, 0);
    return false;
  }

  const oai_memprof_container_v1_chunk_header_t header = {
      .writer_chunk_sequence = writer->chunk_sequence,
      .record_count = writer->batch_count,
  };
  uint8_t wire[OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE];
  if (oai_memprof_container_v1_chunk_header_encode(&header, writer->payload, payload_size, wire, sizeof(wire))
      != OAI_MEMPROF_CONTAINER_V1_OK) {
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_CODEC_ERROR, 0);
    return false;
  }
  if (!write_all(writer, wire, sizeof(wire)) || !write_all(writer, writer->payload, payload_size))
    return false;

  ++writer->chunk_sequence;
  ++writer->chunk_count;
  writer->record_count += writer->batch_count;
  writer->payload_bytes += payload_size;
  writer->batch_count = 0;
  writer->batch_started_ns = 0;
  return true;
}

static bool collect_event(void *context, const oai_memprof_event_v1_t *event)
{
  oai_memprof_stream_writer_t *writer = context;
  if (writer->batch_count == 0 && !monotonic_now_ns(&writer->batch_started_ns)) {
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_SYSTEM_ERROR, errno);
    return false;
  }
  uint8_t *destination = writer->payload + (size_t)writer->batch_count * OAI_MEMPROF_EVENT_V1_WIRE_SIZE;
  if (oai_memprof_event_v1_encode(event, destination, OAI_MEMPROF_EVENT_V1_WIRE_SIZE) != OAI_MEMPROF_WIRE_OK) {
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_CODEC_ERROR, 0);
    return false;
  }
  ++writer->batch_count;
  return writer->batch_count != writer->flush_records || flush_batch(writer);
}

static bool drain_once(oai_memprof_stream_writer_t *writer, bool flush_partial)
{
  const oai_memprof_core_status_t status = oai_memprof_active_runtime_drain_v1(collect_event, writer);
  if (status == OAI_MEMPROF_CORE_OK)
    return !flush_partial || flush_batch(writer);
  if (status == OAI_MEMPROF_CORE_SINK_ERROR && writer->status != OAI_MEMPROF_STREAM_WRITER_OK)
    return false;
  fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_RUNTIME_ERROR, 0);
  return false;
}

static void pause_ns(uint64_t nanoseconds)
{
  struct timespec remaining = {
      .tv_sec = (time_t)(nanoseconds / UINT64_C(1000000000)),
      .tv_nsec = (long)(nanoseconds % UINT64_C(1000000000)),
  };
  while (nanosleep(&remaining, &remaining) != 0 && errno == EINTR) {
  }
}

static void *writer_main(void *argument)
{
  oai_memprof_stream_writer_t *writer = argument;
  while (!atomic_load_explicit(&writer->start_gate, memory_order_acquire)
         && !atomic_load_explicit(&writer->cancel, memory_order_acquire))
    pause_ns(OAI_MEMPROF_WRITER_START_POLL_NS);
  if (atomic_load_explicit(&writer->cancel, memory_order_acquire))
    return NULL;

  const uint64_t poll_ns = writer->flush_interval_ns != 0 && writer->flush_interval_ns < OAI_MEMPROF_WRITER_MAX_POLL_NS
                               ? writer->flush_interval_ns
                               : OAI_MEMPROF_WRITER_MAX_POLL_NS;
  while (!atomic_load_explicit(&writer->stop, memory_order_acquire)) {
    pause_ns(poll_ns);
    if (atomic_load_explicit(&writer->stop, memory_order_acquire))
      break;
    if (!drain_once(writer, false))
      return NULL;
    if (writer->batch_count != 0 && writer->flush_interval_ns != 0) {
      uint64_t now = 0;
      if (!monotonic_now_ns(&now)) {
        fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_SYSTEM_ERROR, errno);
        return NULL;
      }
      if (now < writer->batch_started_ns) {
        fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_SYSTEM_ERROR, 0);
        return NULL;
      }
      if (now - writer->batch_started_ns >= writer->flush_interval_ns && !flush_batch(writer))
        return NULL;
    }
  }
  (void)drain_once(writer, true);
  return NULL;
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
    if (!allowed || length == OAI_MEMPROF_WRITER_MAX_FILE_NAME_BYTES)
      return false;
    ++length;
  }
  return length != 0;
}

static bool valid_config(const oai_memprof_stream_writer_config_t *config)
{
  return config != NULL && config->directory_fd >= 0 && valid_file_name(config->file_name) && config->flush_records != 0
         && config->flush_records <= OAI_MEMPROF_STREAM_WRITER_MAX_FLUSH_RECORDS
         && (config->runtime.core.mode_id == OAI_MEMPROF_CORE_COUNTERS || config->runtime.core.mode_id == OAI_MEMPROF_CORE_SAMPLED
             || config->runtime.core.mode_id == OAI_MEMPROF_CORE_EXACT_EVENTS)
         && config->runtime.core.process_generation == config->opening_header.process_generation
         && config->runtime.core.max_threads == config->opening_header.configured_thread_capacity;
}

oai_memprof_stream_writer_status_t oai_memprof_stream_writer_start_v1(const oai_memprof_stream_writer_config_t *config,
                                                                      oai_memprof_stream_writer_t **writer_out)
{
  if (config == NULL || writer_out == NULL)
    return OAI_MEMPROF_STREAM_WRITER_INVALID_ARGUMENT;
  if (!valid_config(config))
    return OAI_MEMPROF_STREAM_WRITER_INVALID_CONFIGURATION;

  uint8_t opening_wire[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE];
  if (oai_memprof_container_v1_opening_header_encode(&config->opening_header, opening_wire, sizeof(opening_wire))
      != OAI_MEMPROF_CONTAINER_V1_OK)
    return OAI_MEMPROF_STREAM_WRITER_INVALID_CONFIGURATION;

  size_t payload_size = 0;
  size_t mapped_bytes = 0;
  if (!multiply_size(config->flush_records, OAI_MEMPROF_EVENT_V1_WIRE_SIZE, &payload_size)
      || !add_size(sizeof(oai_memprof_stream_writer_t), payload_size, &mapped_bytes))
    return OAI_MEMPROF_STREAM_WRITER_INVALID_CONFIGURATION;
  void *mapping = mmap(NULL, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED)
    return OAI_MEMPROF_STREAM_WRITER_NO_MEMORY;

  oai_memprof_stream_writer_t *writer = mapping;
  writer->mapped_bytes = mapped_bytes;
  writer->payload = (uint8_t *)mapping + sizeof(*writer);
  writer->flush_interval_ns = config->flush_interval_ns;
  writer->flush_records = config->flush_records;
  const size_t file_name_bytes = strlen(config->file_name) + 1U;
  memcpy(writer->file_name, config->file_name, file_name_bytes);
  writer->fd = -1;
  writer->directory_fd = -1;
  writer->status = OAI_MEMPROF_STREAM_WRITER_OK;
  writer->clock_status = oai_memprof_clock_info_v1(&writer->clock_info);
  if (writer->clock_status != OAI_MEMPROF_CLOCK_OK) {
    const oai_memprof_stream_writer_status_t status = OAI_MEMPROF_STREAM_WRITER_CLOCK_ERROR;
    (void)munmap(mapping, mapped_bytes);
    return status;
  }
  if (config->opening_header.calibration_kind != OAI_MEMPROF_CONTAINER_V1_CALIBRATION_EXACT_RATE
      || config->opening_header.clock_kind != writer->clock_info.clock_kind
      || config->opening_header.counter_frequency_numerator != writer->clock_info.counter_frequency_numerator
      || config->opening_header.counter_frequency_denominator != writer->clock_info.counter_frequency_denominator) {
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_STREAM_WRITER_INVALID_CONFIGURATION;
  }
  atomic_init(&writer->start_gate, false);
  atomic_init(&writer->cancel, false);
  atomic_init(&writer->stop, false);

  writer->directory_fd = fcntl(config->directory_fd, F_DUPFD_CLOEXEC, 3);
  struct stat directory_status;
  if (writer->directory_fd < 0 || fstat(writer->directory_fd, &directory_status) != 0 || !S_ISDIR(directory_status.st_mode)) {
    const int saved_errno = errno == 0 ? ENOTDIR : errno;
    if (writer->directory_fd >= 0)
      (void)close(writer->directory_fd);
    (void)munmap(mapping, mapped_bytes);
    errno = saved_errno;
    return OAI_MEMPROF_STREAM_WRITER_IO_ERROR;
  }

  writer->fd =
      openat(writer->directory_fd, writer->file_name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR);
  if (writer->fd < 0) {
    const int saved_errno = errno;
    (void)close(writer->directory_fd);
    (void)munmap(mapping, mapped_bytes);
    errno = saved_errno;
    return OAI_MEMPROF_STREAM_WRITER_IO_ERROR;
  }
  struct stat file_status;
  if (fstat(writer->fd, &file_status) != 0 || !S_ISREG(file_status.st_mode) || file_status.st_nlink != 1
      || file_status.st_size != 0) {
    const int saved_errno = errno == 0 ? EINVAL : errno;
    (void)close(writer->fd);
    (void)close(writer->directory_fd);
    (void)munmap(mapping, mapped_bytes);
    errno = saved_errno;
    return OAI_MEMPROF_STREAM_WRITER_IO_ERROR;
  }
  writer->file_device = (uint64_t)file_status.st_dev;
  writer->file_inode = (uint64_t)file_status.st_ino;
  if (!write_all(writer, opening_wire, sizeof(opening_wire))) {
    const oai_memprof_stream_writer_status_t status = writer->status;
    (void)close(writer->fd);
    (void)close(writer->directory_fd);
    (void)munmap(mapping, mapped_bytes);
    return status;
  }

  if (oai_memprof_active_runtime_bootstrap_v1(&config->runtime) != OAI_MEMPROF_CORE_OK) {
    (void)close(writer->fd);
    (void)close(writer->directory_fd);
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_STREAM_WRITER_RUNTIME_ERROR;
  }
  if (pthread_create(&writer->thread, NULL, writer_main, writer) != 0) {
    (void)close(writer->fd);
    (void)close(writer->directory_fd);
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_STREAM_WRITER_THREAD_ERROR;
  }
  if (oai_memprof_active_runtime_activate_v1() != OAI_MEMPROF_CORE_OK) {
    atomic_store_explicit(&writer->cancel, true, memory_order_release);
    (void)pthread_join(writer->thread, NULL);
    (void)close(writer->fd);
    (void)close(writer->directory_fd);
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_STREAM_WRITER_RUNTIME_ERROR;
  }
  atomic_store_explicit(&writer->start_gate, true, memory_order_release);
  *writer_out = writer;
  return OAI_MEMPROF_STREAM_WRITER_OK;
}

oai_memprof_stream_writer_status_t oai_memprof_stream_writer_finish_v1(oai_memprof_stream_writer_t *writer,
                                                                       uint64_t seal_timeout_ns,
                                                                       oai_memprof_stream_writer_result_t *result)
{
  if (writer == NULL || result == NULL)
    return OAI_MEMPROF_STREAM_WRITER_INVALID_ARGUMENT;

  capture_boundary_sample(writer, NULL, &writer->seal_before_sample);
  oai_memprof_core_status_t runtime_status = oai_memprof_active_runtime_seal_v1(seal_timeout_ns);
  capture_boundary_sample(writer, &writer->seal_before_sample, &writer->seal_after_sample);
  atomic_store_explicit(&writer->stop, true, memory_order_release);
  const int join_status = pthread_join(writer->thread, NULL);
  if (join_status != 0) {
    /* The consumer may still access writer, so process teardown retains its mapping and descriptors. */
    const oai_memprof_stream_writer_result_t failed = {
        .status = OAI_MEMPROF_STREAM_WRITER_THREAD_ERROR,
        .runtime_status = runtime_status,
        .system_errno = join_status,
    };
    *result = failed;
    return OAI_MEMPROF_STREAM_WRITER_THREAD_ERROR;
  }
  if (runtime_status != OAI_MEMPROF_CORE_OK)
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_RUNTIME_ERROR, 0);
  capture_boundary_sample(writer, &writer->seal_after_sample, &writer->drain_complete_sample);

  if (fsync(writer->fd) != 0)
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_IO_ERROR, errno);
  struct stat final_status;
  struct stat path_status;
  if (fstat(writer->fd, &final_status) != 0
      || fstatat(writer->directory_fd, writer->file_name, &path_status, AT_SYMLINK_NOFOLLOW) != 0) {
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_IO_ERROR, errno);
  } else if (!S_ISREG(final_status.st_mode) || !S_ISREG(path_status.st_mode) || final_status.st_nlink != 1
             || path_status.st_nlink != 1 || (uint64_t)final_status.st_dev != writer->file_device
             || (uint64_t)final_status.st_ino != writer->file_inode || final_status.st_dev != path_status.st_dev
             || final_status.st_ino != path_status.st_ino || final_status.st_size < 0 || final_status.st_size != path_status.st_size
             || (uint64_t)final_status.st_size != writer->stream_bytes) {
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_IO_ERROR, EIO);
  }
  const bool file_closed = close(writer->fd) == 0;
  if (!file_closed)
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_IO_ERROR, errno);
  writer->fd = -1;
  const bool directory_synced = fsync(writer->directory_fd) == 0;
  if (!directory_synced)
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_IO_ERROR, errno);
  const bool directory_closed = close(writer->directory_fd) == 0;
  if (!directory_closed)
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_IO_ERROR, errno);
  writer->directory_fd = -1;
  const bool prefooter_closed = file_closed && directory_synced && directory_closed;

  oai_memprof_core_snapshot_t snapshot = {0};
  const oai_memprof_core_status_t snapshot_status = oai_memprof_active_runtime_snapshot_v1(&snapshot);
  if (snapshot_status != OAI_MEMPROF_CORE_OK && writer->status == OAI_MEMPROF_STREAM_WRITER_OK) {
    runtime_status = snapshot_status;
    fail_writer(writer, OAI_MEMPROF_STREAM_WRITER_RUNTIME_ERROR, 0);
  }
  capture_boundary_sample(writer, &writer->drain_complete_sample, &writer->final_sample);

  const oai_memprof_stream_writer_result_t completed = {
      .status = writer->status,
      .runtime_status = runtime_status,
      .runtime_snapshot = snapshot,
      .clock_status = writer->clock_status,
      .clock_info = writer->clock_info,
      .seal_before_sample = writer->seal_before_sample,
      .seal_after_sample = writer->seal_after_sample,
      .drain_complete_sample = writer->drain_complete_sample,
      .final_sample = writer->final_sample,
      .chunk_count = writer->chunk_count,
      .record_count = writer->record_count,
      .payload_bytes = writer->payload_bytes,
      .stream_bytes = writer->stream_bytes,
      .file_device = writer->file_device,
      .file_inode = writer->file_inode,
      .system_errno = writer->system_errno,
      .prefooter_closed = prefooter_closed,
  };
  const size_t mapped_bytes = writer->mapped_bytes;
  const oai_memprof_stream_writer_status_t status = writer->status;
  (void)munmap(writer, mapped_bytes);
  *result = completed;
  return status;
}
