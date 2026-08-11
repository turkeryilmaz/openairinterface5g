/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_R0_ACTUAL_FIXTURE_H
#define OAI_MEMPROF_R0_ACTUAL_FIXTURE_H

#include <stdint.h>

#define OAI_MEMPROF_R0_ACTUAL_FIXTURE_ABI UINT32_C(2)
#define OAI_MEMPROF_R0_CONTROL_SYMBOL "oai_memprof_control_v1"
#define OAI_MEMPROF_R0_CONTROL_VERSION "OAI_MEMPROF_RUNTIME_1.0"
#define OAI_MEMPROF_R0_CONTROL_PRESENT_OFF UINT64_C(0)
#define OAI_MEMPROF_R0_RUNTIME_SONAME "liboai_memprof_runtime.so.1"

#define OAI_MEMPROF_R0_DSO_DESTRUCTOR_PASS UINT32_C(0x4d504f4b)
#define OAI_MEMPROF_R0_DSO_DESTRUCTOR_FAIL UINT32_C(0x4d504552)

#define OAI_MEMPROF_R0_DSO_OBSERVE_SYMBOL "oai_memprof_r0_dso_observe"
#define OAI_MEMPROF_R0_DSO_SET_DESTRUCTOR_PROBE_SYMBOL "oai_memprof_r0_dso_set_destructor_probe"

typedef struct oai_memprof_r0_dso_observation_s {
  uint32_t abi_version;
  uint32_t constructor_status;
  uintptr_t control_address;
  uint64_t control_value;
  uint64_t control_device;
  uint64_t control_inode;
  uint32_t control_found;
  uint32_t exact_version_found;
  uint32_t base_namespace_matches;
  uint32_t file_identity_matches;
  uint32_t reserved_zero;
} oai_memprof_r0_dso_observation_t;

typedef int (*oai_memprof_r0_dso_observe_fn_t)(oai_memprof_r0_dso_observation_t *observation, const char *expected_runtime_path);
typedef void (*oai_memprof_r0_dso_set_destructor_probe_fn_t)(volatile uint32_t *probe);

#endif /* OAI_MEMPROF_R0_ACTUAL_FIXTURE_H */
