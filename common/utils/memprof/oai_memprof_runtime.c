/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define OAI_MEMPROF_RUNTIME_BUILD 1
#include "oai_memprof_runtime_abi.h"

_Alignas(OAI_MEMPROF_CONTROL_CACHE_LINE_BYTES)
    __attribute__((visibility("protected"))) _Atomic(uint64_t) oai_memprof_control_v1 = OAI_MEMPROF_CONTROL_PRESENT_OFF;
