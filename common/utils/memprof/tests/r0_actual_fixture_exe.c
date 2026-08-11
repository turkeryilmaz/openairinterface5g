/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "r0_actual_fixture_common.h"

#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define R0_HIDDEN_NOINLINE __attribute__((visibility("hidden"), noinline, used))
#define R0_MAX_WRITE_ATTEMPTS UINT32_C(16)

enum {
  R0_MAX_DYNAMIC_ENTRIES = 4096,
  R0_MAX_NEEDED_ENTRIES = 64,
  R0_MAX_DYNAMIC_STRING_BYTES = 1U << 20,
};

R0_HIDDEN_NOINLINE void *oai_memprof_r0_exe_call_malloc(size_t size)
{
  return malloc(size);
}

R0_HIDDEN_NOINLINE void *oai_memprof_r0_exe_call_calloc(size_t count, size_t size)
{
  return calloc(count, size);
}

R0_HIDDEN_NOINLINE void *oai_memprof_r0_exe_call_realloc(void *pointer, size_t size)
{
  return realloc(pointer, size);
}

R0_HIDDEN_NOINLINE void oai_memprof_r0_exe_call_free(void *pointer)
{
  free(pointer);
}

static uint32_t executable_constructor_status = UINT32_MAX;
static uint32_t executable_destructor_status = UINT32_MAX;
static uint32_t final_report_armed;

__attribute__((constructor)) static void oai_memprof_r0_executable_constructor(void)
{
  executable_constructor_status = oai_memprof_r0_exercise_allocators(oai_memprof_r0_exe_call_malloc,
                                                                     oai_memprof_r0_exe_call_calloc,
                                                                     oai_memprof_r0_exe_call_realloc,
                                                                     oai_memprof_r0_exe_call_free);
}

static int write_literal(int descriptor, const char *text, size_t length)
{
  size_t offset = 0;
  uint32_t attempts = 0;
  while (offset < length && attempts < R0_MAX_WRITE_ATTEMPTS) {
    ++attempts;
    const ssize_t written = write(descriptor, text + offset, length - offset);
    if (written > 0) {
      offset += (size_t)written;
      continue;
    }
    if (written < 0 && errno == EINTR)
      continue;
    return -1;
  }
  return offset == length ? 0 : -1;
}

#define WRITE_LITERAL(descriptor, literal) write_literal((descriptor), (literal), sizeof(literal) - 1)

/*
 * GCC runs destructors in reverse priority order. The allocator scenario at
 * priority 300 therefore completes before the verifier/emitter at priority
 * 200. The terminal success line cannot appear unless process teardown passed.
 */
__attribute__((destructor(300))) static void oai_memprof_r0_executable_destructor_scenario(void)
{
  executable_destructor_status = oai_memprof_r0_exercise_allocators(oai_memprof_r0_exe_call_malloc,
                                                                    oai_memprof_r0_exe_call_calloc,
                                                                    oai_memprof_r0_exe_call_realloc,
                                                                    oai_memprof_r0_exe_call_free);
}

__attribute__((destructor(200))) static void oai_memprof_r0_executable_destructor_verify(void)
{
  if (!final_report_armed)
    return;
  if (executable_destructor_status != OAI_MEMPROF_R0_ALLOCATOR_OK) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=executable_destructor\n");
    _exit(79);
  }
  if (WRITE_LITERAL(STDOUT_FILENO,
                    "R0_ACTUAL_V1 semantic=pass pre_main=pass main=pass dso_constructor=pass "
                    "dso_destructor=pass process_destructor=pass\n")
      != 0)
    _exit(78);
}

static int parse_arguments(int argc, char **argv, const char **dso_path, uint32_t *expect_runtime, const char **runtime_path)
{
  if (argc != 5 && argc != 7)
    return -1;
  if (strcmp(argv[1], "--dso") != 0 || argv[2][0] != '/' || strcmp(argv[3], "--runtime") != 0)
    return -1;

  *dso_path = argv[2];
  if (strcmp(argv[4], "absent") == 0) {
    if (argc != 5)
      return -1;
    *expect_runtime = 0;
    *runtime_path = NULL;
    return 0;
  }
  if (strcmp(argv[4], "present-off") == 0) {
    if (argc != 7 || strcmp(argv[5], "--runtime-path") != 0 || argv[6][0] != '/')
      return -1;
    *expect_runtime = 1;
    *runtime_path = argv[6];
    return 0;
  }
  return -1;
}

static int load_function(void *handle, const char *name, void *destination, size_t destination_size, void **raw_symbol)
{
  if (destination_size != sizeof(void *) || raw_symbol == NULL)
    return -1;

  (void)dlerror();
  void *symbol = dlsym(handle, name);
  if (dlerror() != NULL || symbol == NULL)
    return -1;

  memcpy(destination, &symbol, sizeof(symbol));
  *raw_symbol = symbol;
  return 0;
}

static int dso_link_treatment_matches(void *handle,
                                      const char *dso_path,
                                      uint32_t expect_runtime,
                                      const void *observe_symbol,
                                      const void *set_probe_symbol)
{
  struct link_map *map = NULL;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0 || map == NULL || map->l_ld == NULL)
    return 0;

  Dl_info observe_information = {0};
  Dl_info probe_information = {0};
  uint64_t observe_device = 0;
  uint64_t observe_inode = 0;
  uint64_t probe_device = 0;
  uint64_t probe_inode = 0;
  if (dladdr(observe_symbol, &observe_information) == 0 || dladdr(set_probe_symbol, &probe_information) == 0
      || observe_information.dli_fbase != (void *)(uintptr_t)map->l_addr
      || probe_information.dli_fbase != observe_information.dli_fbase
      || !oai_memprof_r0_mapped_symbol_matches(observe_symbol,
                                               OAI_MEMPROF_R0_DSO_OBSERVE_SYMBOL,
                                               dso_path,
                                               &observe_device,
                                               &observe_inode)
      || !oai_memprof_r0_mapped_symbol_matches(set_probe_symbol,
                                               OAI_MEMPROF_R0_DSO_SET_DESTRUCTOR_PROBE_SYMBOL,
                                               dso_path,
                                               &probe_device,
                                               &probe_inode)
      || observe_device != probe_device || observe_inode != probe_inode)
    return 0;

  const char *string_table = NULL;
  size_t string_size = 0;
  size_t string_table_count = 0;
  size_t string_size_count = 0;
  size_t needed_count = 0;
  size_t needed_offsets[R0_MAX_NEEDED_ENTRIES];
  int terminated = 0;
  for (size_t index = 0; index < R0_MAX_DYNAMIC_ENTRIES; ++index) {
    const ElfW(Dyn) *entry = &map->l_ld[index];
    if (entry->d_tag == DT_NULL) {
      terminated = 1;
      break;
    }
    if (entry->d_tag == DT_STRTAB) {
      ++string_table_count;
      string_table = (const char *)(uintptr_t)entry->d_un.d_ptr;
    } else if (entry->d_tag == DT_STRSZ) {
      ++string_size_count;
      string_size = (size_t)entry->d_un.d_val;
    } else if (entry->d_tag == DT_NEEDED) {
      if (needed_count == R0_MAX_NEEDED_ENTRIES)
        return 0;
      needed_offsets[needed_count++] = (size_t)entry->d_un.d_val;
    }
  }

  if (!terminated || string_table_count != 1 || string_size_count != 1 || string_table == NULL || string_size == 0
      || string_size > R0_MAX_DYNAMIC_STRING_BYTES)
    return 0;

  size_t runtime_needed_count = 0;
  for (size_t index = 0; index < needed_count; ++index) {
    if (needed_offsets[index] >= string_size)
      return 0;
    const char *name = &string_table[needed_offsets[index]];
    const size_t remaining = string_size - needed_offsets[index];
    if (strnlen(name, remaining) == remaining)
      return 0;
    if (strcmp(name, OAI_MEMPROF_R0_RUNTIME_SONAME) == 0)
      ++runtime_needed_count;
  }

  return runtime_needed_count == expect_runtime;
}

int main(int argc, char **argv)
{
  const char *dso_path = NULL;
  const char *runtime_path = NULL;
  uint32_t expect_runtime = 0;
  if (parse_arguments(argc, argv, &dso_path, &expect_runtime, &runtime_path) != 0) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=arguments\n");
    return 64;
  }

  if (executable_constructor_status != OAI_MEMPROF_R0_ALLOCATOR_OK) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=executable_constructor\n");
    return 65;
  }

  if (oai_memprof_r0_exercise_allocators(oai_memprof_r0_exe_call_malloc,
                                         oai_memprof_r0_exe_call_calloc,
                                         oai_memprof_r0_exe_call_realloc,
                                         oai_memprof_r0_exe_call_free)
      != OAI_MEMPROF_R0_ALLOCATOR_OK) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=executable_main\n");
    return 66;
  }

  const oai_memprof_r0_control_observation_t executable_control = oai_memprof_r0_observe_control(runtime_path);
  if (expect_runtime) {
    if (!executable_control.found || !executable_control.exact_version_found || !executable_control.base_namespace_matches
        || !executable_control.file_identity_matches || executable_control.value != OAI_MEMPROF_R0_CONTROL_PRESENT_OFF) {
      (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=executable_control\n");
      return 67;
    }
  } else if (executable_control.found || !executable_control.base_namespace_matches) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=unexpected_runtime\n");
    return 68;
  }

  void *dso = dlopen(dso_path, RTLD_NOW | RTLD_GLOBAL);
  if (dso == NULL) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=dlopen\n");
    return 69;
  }

  oai_memprof_r0_dso_observe_fn_t observe = NULL;
  oai_memprof_r0_dso_set_destructor_probe_fn_t set_destructor_probe = NULL;
  void *observe_symbol = NULL;
  void *set_destructor_probe_symbol = NULL;
  if (load_function(dso, OAI_MEMPROF_R0_DSO_OBSERVE_SYMBOL, &observe, sizeof(observe), &observe_symbol) != 0
      || load_function(dso,
                       OAI_MEMPROF_R0_DSO_SET_DESTRUCTOR_PROBE_SYMBOL,
                       &set_destructor_probe,
                       sizeof(set_destructor_probe),
                       &set_destructor_probe_symbol)
             != 0) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=dso_exports\n");
    (void)dlclose(dso);
    return 70;
  }

  if (!dso_link_treatment_matches(dso, dso_path, expect_runtime, observe_symbol, set_destructor_probe_symbol)) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=dso_link_identity\n");
    (void)dlclose(dso);
    return 80;
  }

  oai_memprof_r0_dso_observation_t dso_observation = {0};
  if (observe(&dso_observation, runtime_path) != 0 || dso_observation.abi_version != OAI_MEMPROF_R0_ACTUAL_FIXTURE_ABI
      || dso_observation.reserved_zero != 0 || dso_observation.constructor_status != OAI_MEMPROF_R0_ALLOCATOR_OK) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=dso_constructor\n");
    (void)dlclose(dso);
    return 71;
  }

  if (expect_runtime) {
    if (!dso_observation.control_found || !dso_observation.exact_version_found || !dso_observation.base_namespace_matches
        || !dso_observation.file_identity_matches || dso_observation.control_address != executable_control.address
        || dso_observation.control_value != OAI_MEMPROF_R0_CONTROL_PRESENT_OFF
        || dso_observation.control_device != executable_control.device
        || dso_observation.control_inode != executable_control.inode) {
      (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=shared_control\n");
      (void)dlclose(dso);
      return 72;
    }
  } else if (dso_observation.control_found || !dso_observation.base_namespace_matches) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=dso_unexpected_runtime\n");
    (void)dlclose(dso);
    return 73;
  }

  static volatile uint32_t destructor_probe = 0;
  set_destructor_probe(&destructor_probe);
  if (dlclose(dso) != 0) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=dlclose\n");
    return 74;
  }
  if (destructor_probe != OAI_MEMPROF_R0_DSO_DESTRUCTOR_PASS) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=dso_destructor\n");
    return 75;
  }

  const oai_memprof_r0_control_observation_t final_control = oai_memprof_r0_observe_control(runtime_path);
  if (expect_runtime) {
    if (!final_control.found || !final_control.exact_version_found || !final_control.base_namespace_matches
        || !final_control.file_identity_matches || final_control.address != executable_control.address
        || final_control.value != OAI_MEMPROF_R0_CONTROL_PRESENT_OFF || final_control.device != executable_control.device
        || final_control.inode != executable_control.inode) {
      (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=final_control\n");
      return 76;
    }
  } else if (final_control.found || !final_control.base_namespace_matches) {
    (void)WRITE_LITERAL(STDERR_FILENO, "R0_ACTUAL_V1 error=final_unexpected_runtime\n");
    return 77;
  }

  final_report_armed = 1;
  return 0;
}
