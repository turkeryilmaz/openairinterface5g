/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "oai_memprof_process_session.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SESSION_MAPS_INITIAL_BYTES (64U * 1024U)
#define SESSION_MAPS_READ_BYTES 4096U

struct oai_memprof_process_session_s {
  size_t mapped_bytes;
  size_t configuration_size;
  size_t maps_size;
  uint8_t *configuration_bytes;
  uint8_t *maps_bytes;
  oai_memprof_stream_writer_t *writer;
  oai_memprof_container_v1_opening_header_t opening_header;
  oai_memprof_clock_sample_v1_t opening_sample;
  uint32_t max_threads;
  uint32_t ring_records;
  uint32_t flush_records;
  uint64_t flush_interval_ns;
  uint16_t realloc_zero_policy_id;
  int directory_fd;
  char handoff_file_name[OAI_MEMPROF_PROCESS_SESSION_V1_MAX_FILE_NAME_BYTES + 1U];
  char stream_file_name[OAI_MEMPROF_PROCESS_SESSION_V1_MAX_FILE_NAME_BYTES + 1U];
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

static bool valid_file_name(const char *name)
{
  if (name == NULL || name[0] == '\0' || strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
    return false;
  size_t length = 0;
  while (name[length] != '\0') {
    const unsigned char value = (unsigned char)name[length];
    const bool allowed = (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9')
                         || value == '.' || value == '_' || value == '-';
    if (!allowed || length == OAI_MEMPROF_PROCESS_SESSION_V1_MAX_FILE_NAME_BYTES)
      return false;
    ++length;
  }
  return length != 0;
}

static oai_memprof_process_session_status_t read_maps(uint8_t **bytes, size_t *size)
{
  const int fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0)
    return OAI_MEMPROF_PROCESS_SESSION_IO_ERROR;
  size_t capacity = SESSION_MAPS_INITIAL_BYTES;
  void *mapping = mmap(NULL, capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) {
    (void)close(fd);
    return OAI_MEMPROF_PROCESS_SESSION_NO_MEMORY;
  }
  size_t used = 0;
  oai_memprof_process_session_status_t status = OAI_MEMPROF_PROCESS_SESSION_OK;
  for (;;) {
    if (used == capacity) {
      if (capacity == OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_MAPS_BYTES) {
        status = OAI_MEMPROF_PROCESS_SESSION_INVALID_CONFIGURATION;
        break;
      }
      const size_t next = capacity > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_MAPS_BYTES / 2U
                              ? OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_MAPS_BYTES
                              : capacity * 2U;
      void *resized = mremap(mapping, capacity, next, MREMAP_MAYMOVE);
      if (resized == MAP_FAILED) {
        status = OAI_MEMPROF_PROCESS_SESSION_NO_MEMORY;
        break;
      }
      mapping = resized;
      capacity = next;
    }
    const size_t remaining = capacity - used;
    const size_t request = remaining < SESSION_MAPS_READ_BYTES ? remaining : SESSION_MAPS_READ_BYTES;
    const ssize_t got = read(fd, (uint8_t *)mapping + used, request);
    if (got > 0) {
      used += (size_t)got;
      continue;
    }
    if (got == 0)
      break;
    if (errno == EINTR)
      continue;
    status = OAI_MEMPROF_PROCESS_SESSION_IO_ERROR;
    break;
  }
  const int saved_errno = errno;
  if (close(fd) != 0 && status == OAI_MEMPROF_PROCESS_SESSION_OK)
    status = OAI_MEMPROF_PROCESS_SESSION_IO_ERROR;
  if (status != OAI_MEMPROF_PROCESS_SESSION_OK || used == 0) {
    (void)munmap(mapping, capacity);
    errno = saved_errno;
    return status == OAI_MEMPROF_PROCESS_SESSION_OK ? OAI_MEMPROF_PROCESS_SESSION_INVALID_CONFIGURATION : status;
  }
  if (used != capacity) {
    void *resized = mremap(mapping, capacity, used, MREMAP_MAYMOVE);
    if (resized == MAP_FAILED) {
      (void)munmap(mapping, capacity);
      return OAI_MEMPROF_PROCESS_SESSION_NO_MEMORY;
    }
    mapping = resized;
  }
  *bytes = mapping;
  *size = used;
  return OAI_MEMPROF_PROCESS_SESSION_OK;
}

static bool write_all(int fd, const uint8_t *bytes, size_t size)
{
  size_t offset = 0;
  while (offset != size) {
    const ssize_t written = write(fd, bytes + offset, size - offset);
    if (written > 0) {
      offset += (size_t)written;
      continue;
    }
    if (written < 0 && errno == EINTR)
      continue;
    if (written == 0)
      errno = EIO;
    return false;
  }
  return true;
}

static bool hash_stream_prefix(const oai_memprof_process_session_t *session,
                               const oai_memprof_stream_writer_result_t *writer,
                               uint8_t digest[32])
{
  if (writer->stream_bytes == 0 || writer->stream_bytes > SIZE_MAX)
    return false;
  const int fd = openat(session->directory_fd, session->stream_file_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0)
    return false;
  struct stat before = {0};
  struct stat after = {0};
  struct stat path_status = {0};
  bool ok = fstat(fd, &before) == 0 && S_ISREG(before.st_mode) && before.st_nlink == 1
            && (uint64_t)before.st_dev == writer->file_device && (uint64_t)before.st_ino == writer->file_inode
            && (uint64_t)before.st_size == writer->stream_bytes;
  void *mapping = MAP_FAILED;
  if (ok) {
    mapping = mmap(NULL, (size_t)writer->stream_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ok = mapping != MAP_FAILED;
  }
  if (ok)
    ok = oai_memprof_container_v1_sha256(mapping, (size_t)writer->stream_bytes, digest) == OAI_MEMPROF_CONTAINER_V1_OK;
  if (mapping != MAP_FAILED && munmap(mapping, (size_t)writer->stream_bytes) != 0)
    ok = false;
  if (ok)
    ok = fstat(fd, &after) == 0 && after.st_dev == before.st_dev && after.st_ino == before.st_ino && after.st_mode == before.st_mode
         && after.st_nlink == before.st_nlink && after.st_size == before.st_size && after.st_mtim.tv_sec == before.st_mtim.tv_sec
         && after.st_mtim.tv_nsec == before.st_mtim.tv_nsec && after.st_ctim.tv_sec == before.st_ctim.tv_sec
         && after.st_ctim.tv_nsec == before.st_ctim.tv_nsec
         && fstatat(session->directory_fd, session->stream_file_name, &path_status, AT_SYMLINK_NOFOLLOW) == 0
         && S_ISREG(path_status.st_mode) && path_status.st_nlink == 1 && path_status.st_dev == before.st_dev
         && path_status.st_ino == before.st_ino && path_status.st_mode == before.st_mode && path_status.st_size == before.st_size;
  if (close(fd) != 0)
    ok = false;
  if (!ok && errno == 0)
    errno = EIO;
  return ok;
}

static void project_thread(oai_memprof_process_handoff_thread_v1_t *thread)
{
  thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_RING_FULL] = thread->runtime.ring_full_losses;
  thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_RECURSION_BYPASS] = thread->runtime.recursion_bypasses;
  thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SIZE_UNKNOWN] = thread->runtime.size_unknowns;
  thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_INSERTION] = thread->runtime.sample_insertion_failures;
  thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_LOOKUP] = thread->runtime.sample_lookup_failures;
  thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_PROBE] = thread->runtime.sample_probe_exhaustions;
  thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_SAMPLE_PAIRING] = thread->runtime.sample_pairing_failures;
  thread->diagnostic_values[OAI_MEMPROF_HANDOFF_DIAGNOSTIC_COUNTER_INVALID] = thread->runtime.counter_invalids;
  thread->diagnostic_saturated_mask = thread->runtime.diagnostic_saturated_mask;
}

oai_memprof_process_session_status_t oai_memprof_process_session_start_v1(const oai_memprof_process_session_config_t *config,
                                                                          oai_memprof_process_session_t **session_out)
{
  if (config == NULL || session_out == NULL)
    return OAI_MEMPROF_PROCESS_SESSION_INVALID_ARGUMENT;
  if (config->directory_fd < 0 || !valid_file_name(config->stream_file_name) || !valid_file_name(config->handoff_file_name)
      || strcmp(config->stream_file_name, config->handoff_file_name) == 0 || config->configuration_bytes == NULL
      || config->configuration_size == 0 || config->configuration_size > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_BOOTSTRAP_BYTES
      || config->runtime.core.max_threads == 0 || config->runtime.core.max_threads > OAI_MEMPROF_PROCESS_HANDOFF_V1_MAX_THREADS)
    return OAI_MEMPROF_PROCESS_SESSION_INVALID_CONFIGURATION;

  size_t mapped_bytes = 0;
  if (!add_size(sizeof(oai_memprof_process_session_t), config->configuration_size, &mapped_bytes))
    return OAI_MEMPROF_PROCESS_SESSION_INVALID_CONFIGURATION;
  void *mapping = mmap(NULL, mapped_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED)
    return OAI_MEMPROF_PROCESS_SESSION_NO_MEMORY;
  oai_memprof_process_session_t *session = mapping;
  session->mapped_bytes = mapped_bytes;
  session->configuration_size = config->configuration_size;
  session->configuration_bytes = (uint8_t *)mapping + sizeof(*session);
  memcpy(session->configuration_bytes, config->configuration_bytes, config->configuration_size);
  session->max_threads = config->runtime.core.max_threads;
  session->ring_records = config->runtime.core.ring_records;
  session->flush_records = config->flush_records;
  session->flush_interval_ns = config->flush_interval_ns;
  session->realloc_zero_policy_id = config->runtime.realloc_zero_policy_id;
  session->directory_fd = -1;
  memcpy(session->handoff_file_name, config->handoff_file_name, strlen(config->handoff_file_name) + 1U);
  memcpy(session->stream_file_name, config->stream_file_name, strlen(config->stream_file_name) + 1U);

  oai_memprof_process_session_status_t status = read_maps(&session->maps_bytes, &session->maps_size);
  if (status != OAI_MEMPROF_PROCESS_SESSION_OK) {
    (void)munmap(mapping, mapped_bytes);
    return status;
  }
  session->directory_fd = fcntl(config->directory_fd, F_DUPFD_CLOEXEC, 3);
  struct stat directory_status;
  if (session->directory_fd < 0 || fstat(session->directory_fd, &directory_status) != 0 || !S_ISDIR(directory_status.st_mode)) {
    const int saved_errno = errno == 0 ? ENOTDIR : errno;
    if (session->directory_fd >= 0)
      (void)close(session->directory_fd);
    (void)munmap(session->maps_bytes, session->maps_size);
    (void)munmap(mapping, mapped_bytes);
    errno = saved_errno;
    return OAI_MEMPROF_PROCESS_SESSION_IO_ERROR;
  }

  oai_memprof_clock_info_v1_t clock = {0};
  if (oai_memprof_clock_info_v1(&clock) != OAI_MEMPROF_CLOCK_OK) {
    (void)close(session->directory_fd);
    (void)munmap(session->maps_bytes, session->maps_size);
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_PROCESS_SESSION_CLOCK_ERROR;
  }
  oai_memprof_clock_sample_v1_t start = {0};
  if (oai_memprof_clock_sample_v1(OAI_MEMPROF_STREAM_WRITER_CLOCK_BRACKET_NS, &start) != OAI_MEMPROF_CLOCK_OK) {
    (void)close(session->directory_fd);
    (void)munmap(session->maps_bytes, session->maps_size);
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_PROCESS_SESSION_CLOCK_ERROR;
  }

  session->opening_sample = start;
  session->opening_header = config->opening_header;
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size < 4096 || (unsigned long)page_size > UINT32_MAX
      || ((unsigned long)page_size & ((unsigned long)page_size - 1UL)) != 0
      || config->runtime.core.process_generation != session->opening_header.process_generation) {
    (void)close(session->directory_fd);
    (void)munmap(session->maps_bytes, session->maps_size);
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_PROCESS_SESSION_INVALID_CONFIGURATION;
  }
  session->opening_header.page_size_bytes = (uint32_t)page_size;
  session->opening_header.clock_kind = clock.clock_kind;
  session->opening_header.calibration_kind = OAI_MEMPROF_CONTAINER_V1_CALIBRATION_EXACT_RATE;
  session->opening_header.counter_frequency_numerator = clock.counter_frequency_numerator;
  session->opening_header.counter_frequency_denominator = clock.counter_frequency_denominator;
  session->opening_header.calibration_error_bound_ns = OAI_MEMPROF_STREAM_WRITER_CLOCK_BRACKET_NS;
  session->opening_header.calibration_span_ns = 0;
  session->opening_header.start_counter = start.counter;
  session->opening_header.start_monotonic_raw_ns =
      start.monotonic_raw_before_ns + (start.monotonic_raw_after_ns - start.monotonic_raw_before_ns) / 2U;
  session->opening_header.start_realtime_unix_ns = start.realtime_unix_ns;
  session->opening_header.pid = (uint32_t)getpid();
  session->opening_header.configured_thread_capacity = config->runtime.core.max_threads;
  if (oai_memprof_container_v1_sha256(session->configuration_bytes,
                                      session->configuration_size,
                                      session->opening_header.configuration_instance_sha256)
      != OAI_MEMPROF_CONTAINER_V1_OK) {
    (void)close(session->directory_fd);
    (void)munmap(session->maps_bytes, session->maps_size);
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_PROCESS_SESSION_HANDOFF_ERROR;
  }

  const oai_memprof_stream_writer_config_t writer_config = {
      .directory_fd = session->directory_fd,
      .file_name = config->stream_file_name,
      .runtime = config->runtime,
      .opening_header = session->opening_header,
      .flush_records = config->flush_records,
      .flush_interval_ns = config->flush_interval_ns,
  };
  if (oai_memprof_stream_writer_start_v1(&writer_config, &session->writer) != OAI_MEMPROF_STREAM_WRITER_OK) {
    (void)close(session->directory_fd);
    (void)munmap(session->maps_bytes, session->maps_size);
    (void)munmap(mapping, mapped_bytes);
    return OAI_MEMPROF_PROCESS_SESSION_WRITER_ERROR;
  }
  *session_out = session;
  return OAI_MEMPROF_PROCESS_SESSION_OK;
}

oai_memprof_process_session_status_t oai_memprof_process_session_finish_v1(oai_memprof_process_session_t *session,
                                                                           uint64_t seal_timeout_ns,
                                                                           oai_memprof_process_session_result_t *result)
{
  if (session == NULL || result == NULL)
    return OAI_MEMPROF_PROCESS_SESSION_INVALID_ARGUMENT;
  oai_memprof_process_session_result_t completed = {0};
  completed.writer.status = oai_memprof_stream_writer_finish_v1(session->writer, seal_timeout_ns, &completed.writer);
  const bool writer_complete = completed.writer.status == OAI_MEMPROF_STREAM_WRITER_OK && completed.writer.prefooter_closed;
  completed.status = writer_complete ? OAI_MEMPROF_PROCESS_SESSION_OK : OAI_MEMPROF_PROCESS_SESSION_WRITER_ERROR;

  const uint64_t ready_threads = writer_complete ? completed.writer.runtime_snapshot.ready_threads : 0;
  if (ready_threads > session->max_threads || ready_threads > SIZE_MAX) {
    completed.status = OAI_MEMPROF_PROCESS_SESSION_RUNTIME_ERROR;
  }
  size_t thread_bytes = 0;
  oai_memprof_process_handoff_thread_v1_t *threads = NULL;
  if (completed.status == OAI_MEMPROF_PROCESS_SESSION_OK
      && (!multiply_size((size_t)ready_threads, sizeof(*threads), &thread_bytes)
          || (ready_threads != 0
              && (threads = mmap(NULL, thread_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)) == MAP_FAILED))) {
    threads = NULL;
    completed.status = OAI_MEMPROF_PROCESS_SESSION_NO_MEMORY;
  }
  for (uint32_t slot = 0; slot < ready_threads && threads != NULL; ++slot) {
    if (oai_memprof_active_runtime_thread_info_v1(slot, &threads[slot].runtime) != OAI_MEMPROF_CORE_OK) {
      completed.status = OAI_MEMPROF_PROCESS_SESSION_RUNTIME_ERROR;
      break;
    }
    project_thread(&threads[slot]);
  }

  uint8_t prefix_sha256[32] = {0};
  if (completed.status == OAI_MEMPROF_PROCESS_SESSION_OK && !hash_stream_prefix(session, &completed.writer, prefix_sha256)) {
    completed.status = OAI_MEMPROF_PROCESS_SESSION_HANDOFF_ERROR;
    completed.system_errno = errno == 0 ? EIO : errno;
  }

  size_t wire_size = 0;
  uint8_t *wire = NULL;
  if (completed.status == OAI_MEMPROF_PROCESS_SESSION_OK) {
    oai_memprof_process_handoff_v1_t handoff = {
        .opening_header = session->opening_header,
        .opening_sample = session->opening_sample,
        .writer = completed.writer,
        .bootstrap_bytes = session->configuration_bytes,
        .bootstrap_size = session->configuration_size,
        .maps_bytes = session->maps_bytes,
        .maps_size = session->maps_size,
        .threads = threads,
        .thread_count = (size_t)ready_threads,
        .ring_records = session->ring_records,
        .flush_records = session->flush_records,
        .flush_interval_ns = session->flush_interval_ns,
        .realloc_zero_policy_id = session->realloc_zero_policy_id,
        .unregistered_active_thread_failures = completed.writer.runtime_snapshot.unregistered_active_thread_failures,
        .writer_io_or_finalization_failures = completed.writer.status == OAI_MEMPROF_STREAM_WRITER_OK ? 0U : 1U,
        .diagnostic_saturation_transitions = completed.writer.runtime_snapshot.diagnostic_saturation_transitions,
        .registration_diagnostic_saturated_mask = completed.writer.runtime_snapshot.registration_diagnostic_saturated_mask,
    };
    memcpy(handoff.prefix_sha256, prefix_sha256, sizeof(handoff.prefix_sha256));
    const oai_memprof_process_handoff_status_t size_status =
        oai_memprof_process_handoff_v1_size(handoff.bootstrap_size, handoff.maps_size, handoff.thread_count, &wire_size);
    if (size_status == OAI_MEMPROF_PROCESS_HANDOFF_OK) {
      wire = mmap(NULL, wire_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (wire == MAP_FAILED)
        wire = NULL;
    }
    if (size_status != OAI_MEMPROF_PROCESS_HANDOFF_OK || wire == NULL
        || oai_memprof_process_handoff_v1_encode(&handoff, wire, wire_size) != OAI_MEMPROF_PROCESS_HANDOFF_OK)
      completed.status = wire == NULL ? OAI_MEMPROF_PROCESS_SESSION_NO_MEMORY : OAI_MEMPROF_PROCESS_SESSION_HANDOFF_ERROR;
  }

  int handoff_fd = -1;
  if (wire != NULL && completed.status == OAI_MEMPROF_PROCESS_SESSION_OK) {
    handoff_fd = openat(session->directory_fd,
                        session->handoff_file_name,
                        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                        S_IRUSR | S_IWUSR);
    struct stat file_status = {0};
    struct stat path_status = {0};
    bool published = handoff_fd >= 0;
    int saved_errno = handoff_fd < 0 ? errno : 0;
    if (published
        && (fstat(handoff_fd, &file_status) != 0 || !S_ISREG(file_status.st_mode) || file_status.st_nlink != 1
            || file_status.st_size != 0)) {
      published = false;
      saved_errno = errno == 0 ? EIO : errno;
    }
    if (published && (!write_all(handoff_fd, wire, wire_size) || fsync(handoff_fd) != 0)) {
      published = false;
      saved_errno = errno == 0 ? EIO : errno;
    }
    if (published
        && (fstat(handoff_fd, &file_status) != 0 || (uint64_t)file_status.st_size != wire_size
            || fstatat(session->directory_fd, session->handoff_file_name, &path_status, AT_SYMLINK_NOFOLLOW) != 0
            || !S_ISREG(path_status.st_mode) || path_status.st_nlink != 1 || file_status.st_dev != path_status.st_dev
            || file_status.st_ino != path_status.st_ino || file_status.st_mode != path_status.st_mode
            || file_status.st_size != path_status.st_size)) {
      published = false;
      saved_errno = errno == 0 ? EIO : errno;
    }
    if (handoff_fd >= 0) {
      if (close(handoff_fd) != 0 && published) {
        published = false;
        saved_errno = errno == 0 ? EIO : errno;
      }
      handoff_fd = -1;
    }
    if (published && fsync(session->directory_fd) != 0) {
      published = false;
      saved_errno = errno == 0 ? EIO : errno;
    }
    if (published
        && (fstatat(session->directory_fd, session->handoff_file_name, &path_status, AT_SYMLINK_NOFOLLOW) != 0
            || !S_ISREG(path_status.st_mode) || path_status.st_nlink != 1 || file_status.st_dev != path_status.st_dev
            || file_status.st_ino != path_status.st_ino || file_status.st_mode != path_status.st_mode
            || file_status.st_size != path_status.st_size)) {
      published = false;
      saved_errno = errno == 0 ? EIO : errno;
    }
    if (published) {
      completed.handoff_bytes = wire_size;
      completed.handoff_device = (uint64_t)file_status.st_dev;
      completed.handoff_inode = (uint64_t)file_status.st_ino;
      completed.handoff_published = true;
    } else {
      completed.system_errno = saved_errno == 0 ? EIO : saved_errno;
      completed.status = OAI_MEMPROF_PROCESS_SESSION_IO_ERROR;
    }
  }

  if (wire != NULL)
    (void)munmap(wire, wire_size);
  if (threads != NULL)
    (void)munmap(threads, thread_bytes);
  (void)close(session->directory_fd);
  (void)munmap(session->maps_bytes, session->maps_size);
  const size_t mapped_bytes = session->mapped_bytes;
  (void)munmap(session, mapped_bytes);
  *result = completed;
  return completed.status;
}
