/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "r0_actual_fixture_common.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define R0_HIDDEN_NOINLINE __attribute__((visibility("hidden"), noinline, used))
#define R0_EXPORTED __attribute__((visibility("default")))

R0_HIDDEN_NOINLINE void *oai_memprof_r0_dso_call_malloc(size_t size)
{
  return malloc(size);
}

R0_HIDDEN_NOINLINE void *oai_memprof_r0_dso_call_calloc(size_t count, size_t size)
{
  return calloc(count, size);
}

R0_HIDDEN_NOINLINE void *oai_memprof_r0_dso_call_realloc(void *pointer, size_t size)
{
  return realloc(pointer, size);
}

R0_HIDDEN_NOINLINE void oai_memprof_r0_dso_call_free(void *pointer)
{
  free(pointer);
}

static uint32_t dso_constructor_status = UINT32_MAX;
static volatile uint32_t *dso_destructor_probe;

__attribute__((constructor)) static void oai_memprof_r0_dso_constructor(void)
{
  dso_constructor_status = oai_memprof_r0_exercise_allocators(oai_memprof_r0_dso_call_malloc,
                                                              oai_memprof_r0_dso_call_calloc,
                                                              oai_memprof_r0_dso_call_realloc,
                                                              oai_memprof_r0_dso_call_free);
}

__attribute__((destructor)) static void oai_memprof_r0_dso_destructor(void)
{
  const uint32_t status = oai_memprof_r0_exercise_allocators(oai_memprof_r0_dso_call_malloc,
                                                             oai_memprof_r0_dso_call_calloc,
                                                             oai_memprof_r0_dso_call_realloc,
                                                             oai_memprof_r0_dso_call_free);
  if (dso_destructor_probe != NULL)
    *dso_destructor_probe =
        status == OAI_MEMPROF_R0_ALLOCATOR_OK ? OAI_MEMPROF_R0_DSO_DESTRUCTOR_PASS : OAI_MEMPROF_R0_DSO_DESTRUCTOR_FAIL;
}

R0_EXPORTED int oai_memprof_r0_dso_observe(oai_memprof_r0_dso_observation_t *observation, const char *expected_runtime_path)
{
  if (observation == NULL)
    return -1;

  const oai_memprof_r0_control_observation_t control = oai_memprof_r0_observe_control(expected_runtime_path);
  *observation = (oai_memprof_r0_dso_observation_t){
      .abi_version = OAI_MEMPROF_R0_ACTUAL_FIXTURE_ABI,
      .constructor_status = dso_constructor_status,
      .control_address = control.address,
      .control_value = control.value,
      .control_device = control.device,
      .control_inode = control.inode,
      .control_found = control.found,
      .exact_version_found = control.exact_version_found,
      .base_namespace_matches = control.base_namespace_matches,
      .file_identity_matches = control.file_identity_matches,
      .reserved_zero = 0,
  };
  return 0;
}

R0_EXPORTED void oai_memprof_r0_dso_set_destructor_probe(volatile uint32_t *probe)
{
  dso_destructor_probe = probe;
}
