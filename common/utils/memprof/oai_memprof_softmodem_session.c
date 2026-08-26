/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include "oai_memprof_softmodem_session.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define OAI_MEMPROF_SOFTMODEM_MAX_CONFIGURATION_BYTES (UINT32_C(1) << 20)
#define OAI_MEMPROF_SOFTMODEM_MAX_GENERATION ((UINT64_C(1) << 48) - UINT64_C(1))

#define ENV_ENABLE "OAI_MEMPROF_SESSION_ENABLE"
#define ENV_ARCHIVE_FD "OAI_MEMPROF_SESSION_ARCHIVE_FD"
#define ENV_BOOTSTRAP_FD "OAI_MEMPROF_SESSION_BOOTSTRAP_FD"
#define ENV_GENERATION "OAI_MEMPROF_SESSION_PROCESS_GENERATION"
#define ENV_MAX_THREADS "OAI_MEMPROF_SESSION_MAX_THREADS"
#define ENV_RING_RECORDS "OAI_MEMPROF_SESSION_RING_RECORDS"
#define ENV_MODE "OAI_MEMPROF_SESSION_MODE_ID"
#define ENV_TABLE_ENTRIES "OAI_MEMPROF_SESSION_TABLE_ENTRIES"
#define ENV_SAMPLE_SEED "OAI_MEMPROF_SESSION_SAMPLE_SEED"
#define ENV_SAMPLE_THRESHOLD "OAI_MEMPROF_SESSION_SAMPLE_THRESHOLD"
#define ENV_TABLE_PROBES "OAI_MEMPROF_SESSION_TABLE_PROBES"
#define ENV_REALLOC_POLICY "OAI_MEMPROF_SESSION_REALLOC_ZERO_POLICY_ID"
#define ENV_FLUSH_RECORDS "OAI_MEMPROF_SESSION_FLUSH_RECORDS"
#define ENV_FLUSH_INTERVAL "OAI_MEMPROF_SESSION_FLUSH_INTERVAL_NS"
#define ENV_SEAL_TIMEOUT "OAI_MEMPROF_SESSION_SEAL_TIMEOUT_NS"

#define LEGACY_ENV_ARCHIVE "OAI_MEMPROF_SESSION_ARCHIVE_DIRECTORY"
#define LEGACY_ENV_CONFIGURATION "OAI_MEMPROF_SESSION_CONFIGURATION_PATH"
#define LEGACY_ENV_OPENING "OAI_MEMPROF_SESSION_OPENING_PATH"

#define CONFIGURATION_LEAF "effective-config.json"
#define OPENING_LEAF "opening.bin"
#define STREAMS_LEAF "streams"

static const char *const environment_names[] = {
    ENV_ENABLE,
    ENV_ARCHIVE_FD,
    ENV_BOOTSTRAP_FD,
    ENV_GENERATION,
    ENV_MAX_THREADS,
    ENV_RING_RECORDS,
    ENV_MODE,
    ENV_TABLE_ENTRIES,
    ENV_SAMPLE_SEED,
    ENV_SAMPLE_THRESHOLD,
    ENV_TABLE_PROBES,
    ENV_REALLOC_POLICY,
    ENV_FLUSH_RECORDS,
    ENV_FLUSH_INTERVAL,
    ENV_SEAL_TIMEOUT,
};

static const char *const legacy_environment_names[] = {
    LEGACY_ENV_ARCHIVE,
    LEGACY_ENV_CONFIGURATION,
    LEGACY_ENV_OPENING,
};

enum lifecycle_state {
  LIFECYCLE_UNINITIALIZED = 0,
  LIFECYCLE_DISABLED,
  LIFECYCLE_ACTIVE,
  LIFECYCLE_FINISHING,
  LIFECYCLE_COMPLETE,
  LIFECYCLE_FAILED,
};

typedef struct frozen_file_s {
  int fd;
  void *mapping;
  size_t size;
  struct stat identity;
} frozen_file_t;

static _Atomic(int) lifecycle = LIFECYCLE_UNINITIALIZED;
static pthread_mutex_t finish_mutex = PTHREAD_MUTEX_INITIALIZER;
static oai_memprof_process_session_t *process_session;
static uint64_t seal_timeout_ns;
static oai_memprof_process_session_result_t completed_result;
static oai_memprof_softmodem_session_status_t completion_status = OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_STATE;

static bool parse_u64(const char *text, uint64_t maximum, bool nonzero, uint64_t *value)
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
    if (parsed > (maximum - digit) / UINT64_C(10))
      return false;
    parsed = parsed * UINT64_C(10) + digit;
  }
  if ((nonzero && parsed == 0) || parsed > maximum)
    return false;
  *value = parsed;
  return true;
}

static bool parse_fd(const char *text, int *fd)
{
  uint64_t parsed = 0;
  if (fd == NULL || !parse_u64(text, INT_MAX, true, &parsed) || parsed < 3)
    return false;
  *fd = (int)parsed;
  return fcntl(*fd, F_GETFD) >= 0;
}

static bool private_directory_fd(int fd)
{
  struct stat identity = {0};
  return fd >= 3 && fstat(fd, &identity) == 0 && S_ISDIR(identity.st_mode) && identity.st_uid == geteuid()
         && (identity.st_mode & S_IWGRP) == 0 && (identity.st_mode & S_IWOTH) == 0;
}

static bool frozen_unchanged(const frozen_file_t *file)
{
  struct stat after = {0};
  return file != NULL && fstat(file->fd, &after) == 0 && after.st_dev == file->identity.st_dev
         && after.st_ino == file->identity.st_ino && after.st_mode == file->identity.st_mode
         && after.st_nlink == file->identity.st_nlink && after.st_size == file->identity.st_size
         && after.st_mtim.tv_sec == file->identity.st_mtim.tv_sec && after.st_mtim.tv_nsec == file->identity.st_mtim.tv_nsec
         && after.st_ctim.tv_sec == file->identity.st_ctim.tv_sec && after.st_ctim.tv_nsec == file->identity.st_ctim.tv_nsec;
}

static bool frozen_openat(int directory_fd, const char *leaf, size_t maximum, frozen_file_t *file)
{
  if (directory_fd < 3 || leaf == NULL || leaf[0] == 0 || strchr(leaf, '/') != NULL || file == NULL || maximum == 0)
    return false;
  *file = (frozen_file_t){.fd = -1};
  file->fd = openat(directory_fd, leaf, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (file->fd < 0 || fstat(file->fd, &file->identity) != 0 || !S_ISREG(file->identity.st_mode) || file->identity.st_nlink != 1
      || file->identity.st_size <= 0 || (uint64_t)file->identity.st_size > maximum || (uint64_t)file->identity.st_size > SIZE_MAX) {
    if (file->fd >= 0)
      (void)close(file->fd);
    file->fd = -1;
    return false;
  }
  file->size = (size_t)file->identity.st_size;
  file->mapping = mmap(NULL, file->size, PROT_READ, MAP_PRIVATE, file->fd, 0);
  if (file->mapping == MAP_FAILED || !frozen_unchanged(file)) {
    if (file->mapping != MAP_FAILED)
      (void)munmap(file->mapping, file->size);
    (void)close(file->fd);
    *file = (frozen_file_t){.fd = -1};
    return false;
  }
  return true;
}

static void frozen_close(frozen_file_t *file)
{
  if (file == NULL)
    return;
  if (file->mapping != NULL && file->mapping != MAP_FAILED)
    (void)munmap(file->mapping, file->size);
  if (file->fd >= 0)
    (void)close(file->fd);
  *file = (frozen_file_t){.fd = -1};
}

static bool all_environment_absent(void)
{
  for (size_t index = 0; index < sizeof(environment_names) / sizeof(environment_names[0]); ++index) {
    if (getenv(environment_names[index]) != NULL)
      return false;
  }
  for (size_t index = 0; index < sizeof(legacy_environment_names) / sizeof(legacy_environment_names[0]); ++index) {
    if (getenv(legacy_environment_names[index]) != NULL)
      return false;
  }
  return true;
}

static bool environment_complete(void)
{
  for (size_t index = 0; index < sizeof(legacy_environment_names) / sizeof(legacy_environment_names[0]); ++index)
    if (getenv(legacy_environment_names[index]) != NULL)
      return false;
  if (getenv(ENV_ENABLE) == NULL || strcmp(getenv(ENV_ENABLE), "1") != 0)
    return false;
  for (size_t index = 1; index < sizeof(environment_names) / sizeof(environment_names[0]); ++index) {
    const char *value = getenv(environment_names[index]);
    if (value == NULL || value[0] == '\0')
      return false;
  }
  return true;
}

oai_memprof_softmodem_session_status_t oai_memprof_softmodem_session_start_v1(uint16_t expected_role_kind)
{
  int expected = LIFECYCLE_UNINITIALIZED;
  if (!atomic_compare_exchange_strong_explicit(&lifecycle, &expected, LIFECYCLE_FAILED, memory_order_acq_rel, memory_order_acquire))
    return OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_STATE;

  if (all_environment_absent()) {
    atomic_store_explicit(&lifecycle, LIFECYCLE_DISABLED, memory_order_release);
    return OAI_MEMPROF_SOFTMODEM_SESSION_DISABLED;
  }
  if (!environment_complete()
      || (expected_role_kind != OAI_MEMPROF_SOFTMODEM_ROLE_GNB && expected_role_kind != OAI_MEMPROF_SOFTMODEM_ROLE_NR_UE))
    return OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_ENVIRONMENT;

  frozen_file_t configuration = {.fd = -1};
  frozen_file_t opening_file = {.fd = -1};
  int archive_fd = -1;
  int bootstrap_fd = -1;
  int streams_fd = -1;
  oai_memprof_softmodem_session_status_t status = OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_ENVIRONMENT;
  if (!parse_fd(getenv(ENV_ARCHIVE_FD), &archive_fd) || !parse_fd(getenv(ENV_BOOTSTRAP_FD), &bootstrap_fd))
    goto cleanup;

  uint64_t generation = 0;
  uint64_t max_threads = 0;
  uint64_t ring_records = 0;
  uint64_t mode_id = 0;
  uint64_t table_entries = 0;
  uint64_t sample_seed = 0;
  uint64_t sample_threshold = 0;
  uint64_t table_probes = 0;
  uint64_t realloc_policy = 0;
  uint64_t flush_records = 0;
  uint64_t flush_interval = 0;
  uint64_t timeout = 0;
  if (!parse_u64(getenv(ENV_GENERATION), OAI_MEMPROF_SOFTMODEM_MAX_GENERATION, true, &generation)
      || !parse_u64(getenv(ENV_MAX_THREADS), UINT32_MAX - UINT64_C(1), true, &max_threads)
      || !parse_u64(getenv(ENV_RING_RECORDS), UINT32_MAX, true, &ring_records)
      || !parse_u64(getenv(ENV_MODE), UINT8_MAX, true, &mode_id)
      || !parse_u64(getenv(ENV_TABLE_ENTRIES), SIZE_MAX, false, &table_entries)
      || !parse_u64(getenv(ENV_SAMPLE_SEED), UINT64_MAX, false, &sample_seed)
      || !parse_u64(getenv(ENV_SAMPLE_THRESHOLD), UINT64_MAX, false, &sample_threshold)
      || !parse_u64(getenv(ENV_TABLE_PROBES), UINT32_MAX, false, &table_probes)
      || !parse_u64(getenv(ENV_REALLOC_POLICY), UINT16_MAX, true, &realloc_policy)
      || !parse_u64(getenv(ENV_FLUSH_RECORDS), UINT32_MAX, true, &flush_records)
      || !parse_u64(getenv(ENV_FLUSH_INTERVAL), UINT64_MAX, false, &flush_interval)
      || !parse_u64(getenv(ENV_SEAL_TIMEOUT), UINT64_MAX, true, &timeout)
      || (mode_id != OAI_MEMPROF_CORE_COUNTERS && mode_id != OAI_MEMPROF_CORE_SAMPLED && mode_id != OAI_MEMPROF_CORE_EXACT_EVENTS)
      || (mode_id == OAI_MEMPROF_CORE_SAMPLED
          && (table_entries == 0 || sample_threshold == 0 || table_probes == 0 || table_probes > table_entries))
      || (mode_id != OAI_MEMPROF_CORE_SAMPLED
          && (table_entries != 0 || sample_seed != 0 || sample_threshold != 0 || table_probes != 0))
      || (realloc_policy != 1 && realloc_policy != 2))
    goto cleanup;

  if (!private_directory_fd(archive_fd) || !private_directory_fd(bootstrap_fd))
    goto cleanup;
  status = OAI_MEMPROF_SOFTMODEM_SESSION_IO_ERROR;
  if (!frozen_openat(bootstrap_fd, CONFIGURATION_LEAF, OAI_MEMPROF_SOFTMODEM_MAX_CONFIGURATION_BYTES, &configuration)
      || !frozen_openat(bootstrap_fd, OPENING_LEAF, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE, &opening_file)
      || opening_file.size != OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE)
    goto cleanup;
  if (close(bootstrap_fd) != 0) {
    bootstrap_fd = -1;
    goto cleanup;
  }
  bootstrap_fd = -1;

  oai_memprof_container_v1_opening_header_t opening = {0};
  if (oai_memprof_container_v1_opening_header_decode(&opening, opening_file.mapping, opening_file.size)
          != OAI_MEMPROF_CONTAINER_V1_OK
      || opening.role_kind != expected_role_kind || opening.scope_kind != OAI_MEMPROF_CONTAINER_V1_SCOPE_MEASUREMENT_INTERVAL
      || opening.process_generation != generation || opening.configured_thread_capacity != max_threads) {
    status = OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_CONFIGURATION;
    goto cleanup;
  }
  uint8_t configuration_sha256[sizeof(opening.configuration_instance_sha256)] = {0};
  if (oai_memprof_container_v1_sha256(configuration.mapping, configuration.size, configuration_sha256)
          != OAI_MEMPROF_CONTAINER_V1_OK
      || memcmp(configuration_sha256, opening.configuration_instance_sha256, sizeof(configuration_sha256)) != 0) {
    status = OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_CONFIGURATION;
    goto cleanup;
  }

  /*
   * These roots are capabilities pinned by the launcher, not path claims.
   * They bind descendant lookup across pathname renames/replacements, but do
   * not claim to make an inode immutable against a same-UID writer that can
   * mutate an already-open object after the launcher has authenticated it.
   */
  streams_fd = openat(archive_fd, STREAMS_LEAF, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
  if (streams_fd < 0)
    goto cleanup;
  if (!private_directory_fd(streams_fd))
    goto cleanup;
  if (close(archive_fd) != 0) {
    archive_fd = -1;
    goto cleanup;
  }
  archive_fd = -1;

  const oai_memprof_process_session_config_t session_config = {
      .directory_fd = streams_fd,
      .stream_file_name = "memory-lifetime.bin",
      .handoff_file_name = "process-handoff.bin",
      .configuration_bytes = configuration.mapping,
      .configuration_size = configuration.size,
      .runtime =
          {
              .core =
                  {
                      .process_generation = generation,
                      .table_entries = table_entries,
                      .sample_seed = sample_seed,
                      .sample_threshold = sample_threshold,
                      .max_threads = (uint32_t)max_threads,
                      .ring_records = (uint32_t)ring_records,
                      .table_probes = (uint32_t)table_probes,
                      .mode_id = (uint8_t)mode_id,
                  },
              .realloc_zero_policy_id = (uint16_t)realloc_policy,
          },
      .opening_header = opening,
      .flush_records = (uint32_t)flush_records,
      .flush_interval_ns = flush_interval,
  };
  if (oai_memprof_process_session_start_v1(&session_config, &process_session) != OAI_MEMPROF_PROCESS_SESSION_OK) {
    status = OAI_MEMPROF_SOFTMODEM_SESSION_PROCESS_ERROR;
    goto cleanup;
  }
  if (!frozen_unchanged(&configuration) || !frozen_unchanged(&opening_file)) {
    oai_memprof_process_session_result_t discarded = {0};
    (void)oai_memprof_process_session_finish_v1(process_session, timeout, &discarded);
    process_session = NULL;
    status = OAI_MEMPROF_SOFTMODEM_SESSION_IO_ERROR;
    goto cleanup;
  }
  seal_timeout_ns = timeout;
  status = OAI_MEMPROF_SOFTMODEM_SESSION_OK;

cleanup:
  if (streams_fd >= 0)
    (void)close(streams_fd);
  if (bootstrap_fd >= 0)
    (void)close(bootstrap_fd);
  if (archive_fd >= 0)
    (void)close(archive_fd);
  frozen_close(&opening_file);
  frozen_close(&configuration);
  atomic_store_explicit(&lifecycle,
                        status == OAI_MEMPROF_SOFTMODEM_SESSION_OK ? LIFECYCLE_ACTIVE : LIFECYCLE_FAILED,
                        memory_order_release);
  return status;
}

oai_memprof_softmodem_session_status_t oai_memprof_softmodem_session_finish_v1(oai_memprof_process_session_result_t *result)
{
  if (pthread_mutex_lock(&finish_mutex) != 0)
    return OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_STATE;

  const int observed = atomic_load_explicit(&lifecycle, memory_order_acquire);
  if (observed == LIFECYCLE_DISABLED) {
    (void)pthread_mutex_unlock(&finish_mutex);
    return OAI_MEMPROF_SOFTMODEM_SESSION_DISABLED;
  }
  if (observed == LIFECYCLE_COMPLETE) {
    if (result != NULL)
      *result = completed_result;
    (void)pthread_mutex_unlock(&finish_mutex);
    return OAI_MEMPROF_SOFTMODEM_SESSION_ALREADY_FINISHED;
  }
  if (observed != LIFECYCLE_ACTIVE) {
    (void)pthread_mutex_unlock(&finish_mutex);
    return OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_STATE;
  }
  atomic_store_explicit(&lifecycle, LIFECYCLE_FINISHING, memory_order_release);

  completed_result = (oai_memprof_process_session_result_t){0};
  const oai_memprof_process_session_status_t session_status =
      oai_memprof_process_session_finish_v1(process_session, seal_timeout_ns, &completed_result);
  process_session = NULL;
  completion_status = session_status == OAI_MEMPROF_PROCESS_SESSION_OK ? OAI_MEMPROF_SOFTMODEM_SESSION_OK
                                                                       : OAI_MEMPROF_SOFTMODEM_SESSION_PROCESS_ERROR;
  atomic_store_explicit(&lifecycle,
                        completion_status == OAI_MEMPROF_SOFTMODEM_SESSION_OK ? LIFECYCLE_COMPLETE : LIFECYCLE_FAILED,
                        memory_order_release);
  if (result != NULL)
    *result = completed_result;
  (void)pthread_mutex_unlock(&finish_mutex);
  return completion_status;
}

const char *oai_memprof_softmodem_session_status_name_v1(oai_memprof_softmodem_session_status_t status)
{
  switch (status) {
    case OAI_MEMPROF_SOFTMODEM_SESSION_DISABLED:
      return "disabled";
    case OAI_MEMPROF_SOFTMODEM_SESSION_OK:
      return "ok";
    case OAI_MEMPROF_SOFTMODEM_SESSION_ALREADY_FINISHED:
      return "already_finished";
    case OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_ENVIRONMENT:
      return "invalid_environment";
    case OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_STATE:
      return "invalid_state";
    case OAI_MEMPROF_SOFTMODEM_SESSION_IO_ERROR:
      return "io_error";
    case OAI_MEMPROF_SOFTMODEM_SESSION_INVALID_CONFIGURATION:
      return "invalid_configuration";
    case OAI_MEMPROF_SOFTMODEM_SESSION_PROCESS_ERROR:
      return "process_error";
  }
  return "unknown";
}
