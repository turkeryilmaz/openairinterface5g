/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_R0_ACTUAL_FIXTURE_COMMON_H
#define OAI_MEMPROF_R0_ACTUAL_FIXTURE_COMMON_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include "r0_actual_fixture.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#if !defined(__GNUC__) || defined(__clang__)
#error "The frozen R0 GNU-wrap fixture requires GNU GCC"
#endif

#if !defined(__GLIBC__)
#error "The frozen R0 actual-loader fixture requires glibc"
#endif

_Static_assert(sizeof(uint64_t) == 8, "the R0 control ABI requires a 64-bit uint64_t");
_Static_assert(sizeof(uintptr_t) == 8, "the admitted R0 ELF systems are 64-bit");
_Static_assert(ATOMIC_LONG_LOCK_FREE == 2 && ATOMIC_LLONG_LOCK_FREE == 2, "the R0 control load must not require libatomic");

typedef void *(*oai_memprof_r0_malloc_fn_t)(size_t size);
typedef void *(*oai_memprof_r0_calloc_fn_t)(size_t count, size_t size);
typedef void *(*oai_memprof_r0_realloc_fn_t)(void *pointer, size_t size);
typedef void (*oai_memprof_r0_free_fn_t)(void *pointer);

enum oai_memprof_r0_allocator_status {
  OAI_MEMPROF_R0_ALLOCATOR_OK = 0,
  OAI_MEMPROF_R0_MALLOC_FAILED = 1U << 0,
  OAI_MEMPROF_R0_CALLOC_FAILED = 1U << 1,
  OAI_MEMPROF_R0_CALLOC_NOT_ZERO = 1U << 2,
  OAI_MEMPROF_R0_REALLOC_SOURCE_FAILED = 1U << 3,
  OAI_MEMPROF_R0_REALLOC_FAILED = 1U << 4,
  OAI_MEMPROF_R0_REALLOC_CONTENT_CHANGED = 1U << 5,
  OAI_MEMPROF_R0_REALLOC_NULL_FAILED = 1U << 6,
  OAI_MEMPROF_R0_FREE_CHANGED_ERRNO = 1U << 7,
};

/*
 * Exercise only normalized relations in the admitted GNU/Linux/glibc domain.
 * Pointer values, whether a zero-size operation returns NULL, and whether
 * realloc moves are deliberately not part of the result. The scripted oracle,
 * rather than this actual-libc fixture, owns the exact wrapper-errno proof.
 */
static uint32_t oai_memprof_r0_exercise_allocators(oai_memprof_r0_malloc_fn_t malloc_fn,
                                                   oai_memprof_r0_calloc_fn_t calloc_fn,
                                                   oai_memprof_r0_realloc_fn_t realloc_fn,
                                                   oai_memprof_r0_free_fn_t free_fn)
{
  uint32_t status = OAI_MEMPROF_R0_ALLOCATOR_OK;

  volatile size_t malloc_size = 73;
  unsigned char *malloc_result = malloc_fn(malloc_size);
  if (malloc_result == NULL) {
    status |= OAI_MEMPROF_R0_MALLOC_FAILED;
  } else {
    for (size_t i = 0; i < malloc_size; ++i)
      malloc_result[i] = (unsigned char)(i ^ 0x5aU);
    errno = EDOM;
    free_fn(malloc_result);
    if (errno != EDOM)
      status |= OAI_MEMPROF_R0_FREE_CHANGED_ERRNO;
  }

  volatile size_t calloc_count = 7;
  volatile size_t calloc_size = 19;
  unsigned char *calloc_result = calloc_fn(calloc_count, calloc_size);
  if (calloc_result == NULL) {
    status |= OAI_MEMPROF_R0_CALLOC_FAILED;
  } else {
    const size_t extent = calloc_count * calloc_size;
    for (size_t i = 0; i < extent; ++i) {
      if (calloc_result[i] != 0) {
        status |= OAI_MEMPROF_R0_CALLOC_NOT_ZERO;
        break;
      }
    }
    errno = EDOM;
    free_fn(calloc_result);
    if (errno != EDOM)
      status |= OAI_MEMPROF_R0_FREE_CHANGED_ERRNO;
  }

  volatile size_t realloc_source_size = 41;
  volatile size_t realloc_target_size = 149;
  unsigned char *realloc_source = malloc_fn(realloc_source_size);
  if (realloc_source == NULL) {
    status |= OAI_MEMPROF_R0_REALLOC_SOURCE_FAILED;
  } else {
    for (size_t i = 0; i < realloc_source_size; ++i)
      realloc_source[i] = (unsigned char)(i + 17U);
    unsigned char *realloc_result = realloc_fn(realloc_source, realloc_target_size);
    if (realloc_result == NULL) {
      status |= OAI_MEMPROF_R0_REALLOC_FAILED;
      errno = EDOM;
      free_fn(realloc_source);
    } else {
      for (size_t i = 0; i < realloc_source_size; ++i) {
        if (realloc_result[i] != (unsigned char)(i + 17U)) {
          status |= OAI_MEMPROF_R0_REALLOC_CONTENT_CHANGED;
          break;
        }
      }
      errno = EDOM;
      free_fn(realloc_result);
    }
    if (errno != EDOM)
      status |= OAI_MEMPROF_R0_FREE_CHANGED_ERRNO;
  }

  volatile size_t realloc_null_size = 37;
  void *realloc_null_result = realloc_fn(NULL, realloc_null_size);
  if (realloc_null_result == NULL) {
    status |= OAI_MEMPROF_R0_REALLOC_NULL_FAILED;
  } else {
    errno = EDOM;
    free_fn(realloc_null_result);
    if (errno != EDOM)
      status |= OAI_MEMPROF_R0_FREE_CHANGED_ERRNO;
  }

  void *malloc_zero_result = malloc_fn(0);
  errno = EDOM;
  free_fn(malloc_zero_result);
  if (errno != EDOM)
    status |= OAI_MEMPROF_R0_FREE_CHANGED_ERRNO;

  void *calloc_zero_result = calloc_fn(0, calloc_size);
  errno = EDOM;
  free_fn(calloc_zero_result);
  if (errno != EDOM)
    status |= OAI_MEMPROF_R0_FREE_CHANGED_ERRNO;

  errno = EDOM;
  free_fn(NULL);
  if (errno != EDOM)
    status |= OAI_MEMPROF_R0_FREE_CHANGED_ERRNO;

  return status;
}

typedef struct oai_memprof_r0_control_observation_s {
  uintptr_t address;
  uint64_t value;
  uint64_t device;
  uint64_t inode;
  uint32_t found;
  uint32_t exact_version_found;
  uint32_t base_namespace_matches;
  uint32_t file_identity_matches;
} oai_memprof_r0_control_observation_t;

enum {
  OAI_MEMPROF_R0_MAPS_TOTAL_BYTES = 1U << 20,
  OAI_MEMPROF_R0_MAPS_LINE_BYTES = 4096,
  OAI_MEMPROF_R0_MAPS_READ_ATTEMPTS = 1024,
};

static int oai_memprof_r0_same_file_version(const struct stat *left, const struct stat *right)
{
  return left->st_dev == right->st_dev && left->st_ino == right->st_ino && left->st_mode == right->st_mode
         && left->st_size == right->st_size && left->st_mtim.tv_sec == right->st_mtim.tv_sec
         && left->st_mtim.tv_nsec == right->st_mtim.tv_nsec && left->st_ctim.tv_sec == right->st_ctim.tv_sec
         && left->st_ctim.tv_nsec == right->st_ctim.tv_nsec;
}

static int oai_memprof_r0_parse_u64(const char **cursor, unsigned int base, char delimiter, uint64_t *value)
{
  if (cursor == NULL || *cursor == NULL || value == NULL)
    return -1;

  errno = 0;
  char *end = NULL;
  const unsigned long long parsed = strtoull(*cursor, &end, (int)base);
  if (errno == ERANGE || end == *cursor || (delimiter != '\0' && *end != delimiter))
    return -1;
  *value = (uint64_t)parsed;
  *cursor = delimiter == '\0' ? end : end + 1;
  return 0;
}

static int oai_memprof_r0_skip_field(const char **cursor)
{
  if (cursor == NULL || *cursor == NULL)
    return -1;
  while (**cursor != '\0' && **cursor != ' ')
    ++*cursor;
  if (**cursor != ' ')
    return -1;
  while (**cursor == ' ')
    ++*cursor;
  return 0;
}

static int oai_memprof_r0_parse_mapping(const char *line, uintptr_t address, const struct stat *expected, int *identity_matches)
{
  const char *cursor = line;
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t offset = 0;
  uint64_t device_major = 0;
  uint64_t device_minor = 0;
  uint64_t inode = 0;
  if (oai_memprof_r0_parse_u64(&cursor, 16, '-', &start) != 0 || oai_memprof_r0_parse_u64(&cursor, 16, ' ', &end) != 0
      || start >= end || oai_memprof_r0_skip_field(&cursor) != 0 || oai_memprof_r0_parse_u64(&cursor, 16, ' ', &offset) != 0
      || oai_memprof_r0_parse_u64(&cursor, 16, ':', &device_major) != 0
      || oai_memprof_r0_parse_u64(&cursor, 16, ' ', &device_minor) != 0)
    return -1;

  errno = 0;
  char *inode_end = NULL;
  const unsigned long long parsed_inode = strtoull(cursor, &inode_end, 10);
  if (errno == ERANGE || inode_end == cursor || (*inode_end != '\0' && *inode_end != ' '))
    return -1;
  inode = (uint64_t)parsed_inode;

  if ((uint64_t)address < start || (uint64_t)address >= end)
    return 0;
  if (identity_matches == NULL || offset != 0 || inode == 0)
    return -1;
  *identity_matches = device_major == (uint64_t)major(expected->st_dev) && device_minor == (uint64_t)minor(expected->st_dev)
                      && inode == (uint64_t)expected->st_ino;
  return 1;
}

static int oai_memprof_r0_mapped_base_matches(uintptr_t base, const struct stat *expected)
{
  const int descriptor = open("/proc/self/maps", O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW);
  if (descriptor < 0)
    return 0;

  char input[4096];
  char line[OAI_MEMPROF_R0_MAPS_LINE_BYTES + 1];
  size_t line_length = 0;
  size_t total = 0;
  unsigned int attempts = 0;
  int found = 0;
  int identity_matches = 0;
  int valid = 1;
  while (valid && attempts++ < OAI_MEMPROF_R0_MAPS_READ_ATTEMPTS) {
    const ssize_t count = read(descriptor, input, sizeof(input));
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0 || (size_t)count > OAI_MEMPROF_R0_MAPS_TOTAL_BYTES - total) {
      valid = 0;
      break;
    }
    if (count == 0)
      break;
    total += (size_t)count;
    for (ssize_t index = 0; index < count; ++index) {
      if (input[index] == '\n') {
        line[line_length] = '\0';
        int line_matches = 0;
        const int parsed = oai_memprof_r0_parse_mapping(line, base, expected, &line_matches);
        if (parsed < 0 || (parsed == 1 && found)) {
          valid = 0;
          break;
        }
        if (parsed == 1) {
          found = 1;
          identity_matches = line_matches;
        }
        line_length = 0;
      } else if (line_length == OAI_MEMPROF_R0_MAPS_LINE_BYTES) {
        valid = 0;
        break;
      } else {
        line[line_length++] = input[index];
      }
    }
  }
  if (attempts >= OAI_MEMPROF_R0_MAPS_READ_ATTEMPTS || line_length != 0)
    valid = 0;
  if (close(descriptor) != 0)
    valid = 0;
  return valid && found && identity_matches;
}

static int oai_memprof_r0_mapped_symbol_matches(const void *symbol,
                                                const char *symbol_name,
                                                const char *expected_path,
                                                uint64_t *device,
                                                uint64_t *inode)
{
  if (symbol == NULL || symbol_name == NULL || expected_path == NULL || expected_path[0] != '/' || device == NULL || inode == NULL)
    return 0;

  const int descriptor = open(expected_path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW);
  if (descriptor < 0)
    return 0;
  struct stat before = {0};
  struct stat after = {0};
  struct stat path_after = {0};
  int matches = fstat(descriptor, &before) == 0 && S_ISREG(before.st_mode) && before.st_size > 0;

  Dl_info information = {0};
  char loaded_real_path[PATH_MAX];
  char expected_real_path[PATH_MAX];
  if (!matches || dladdr(symbol, &information) == 0 || information.dli_fname == NULL || information.dli_sname == NULL
      || information.dli_saddr != symbol || information.dli_fbase == NULL || strcmp(information.dli_sname, symbol_name) != 0
      || realpath(information.dli_fname, loaded_real_path) == NULL || realpath(expected_path, expected_real_path) == NULL
      || strcmp(loaded_real_path, expected_real_path) != 0
      || !oai_memprof_r0_mapped_base_matches((uintptr_t)information.dli_fbase, &before))
    matches = 0;

  if (fstat(descriptor, &after) != 0 || lstat(expected_path, &path_after) != 0 || !S_ISREG(path_after.st_mode)
      || !oai_memprof_r0_same_file_version(&before, &after) || !oai_memprof_r0_same_file_version(&before, &path_after))
    matches = 0;
  if (close(descriptor) != 0)
    matches = 0;
  if (!matches)
    return 0;
  *device = (uint64_t)before.st_dev;
  *inode = (uint64_t)before.st_ino;
  return 1;
}

/* Dynamic lookup keeps the A00 and A01 fixture objects byte-identical. */
static oai_memprof_r0_control_observation_t oai_memprof_r0_observe_control(const char *expected_runtime_path)
{
  (void)dlerror();
  void *versioned_address = dlvsym(RTLD_DEFAULT, OAI_MEMPROF_R0_CONTROL_SYMBOL, OAI_MEMPROF_R0_CONTROL_VERSION);
  const char *versioned_error = dlerror();

  (void)dlerror();
  void *base_address = dlsym(RTLD_DEFAULT, OAI_MEMPROF_R0_CONTROL_SYMBOL);
  const char *base_error = dlerror();

  if (versioned_error != NULL || versioned_address == NULL) {
    return (oai_memprof_r0_control_observation_t){
        .base_namespace_matches = base_error != NULL || base_address == NULL,
    };
  }

  oai_memprof_r0_control_observation_t result = {
      .address = (uintptr_t)versioned_address,
      .found = 1,
      .exact_version_found = 1,
      .base_namespace_matches = base_error == NULL && base_address == versioned_address,
  };

  result.file_identity_matches = oai_memprof_r0_mapped_symbol_matches(versioned_address,
                                                                      OAI_MEMPROF_R0_CONTROL_SYMBOL,
                                                                      expected_runtime_path,
                                                                      &result.device,
                                                                      &result.inode);

  _Atomic(uint64_t) *control = (_Atomic(uint64_t) *)versioned_address;
  result.value = atomic_load_explicit(control, memory_order_seq_cst);
  return result;
}

#endif /* OAI_MEMPROF_R0_ACTUAL_FIXTURE_COMMON_H */
