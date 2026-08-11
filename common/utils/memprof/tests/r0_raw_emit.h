/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_R0_RAW_EMIT_H
#define OAI_MEMPROF_R0_RAW_EMIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Allocation-free, stdio-free raw records used only by the R0 semantic
 * oracle. All functions preserve the caller's errno value.
 */
void oai_memprof_r0_emit_meta(const char *mode);
void oai_memprof_r0_emit_token(const char *token, const void *address);
void oai_memprof_r0_emit_eval(uint32_t transaction,
                              const char *phase,
                              const char *operand,
                              uint32_t evaluator,
                              const char *kind,
                              uintptr_t value);
void oai_memprof_r0_emit_real(uint32_t sequence,
                              uint32_t transaction,
                              const char *api,
                              uintptr_t arg0,
                              uintptr_t arg1,
                              const void *result,
                              int errno_in,
                              int errno_out);
void oai_memprof_r0_emit_caller(uint32_t transaction,
                                const char *phase,
                                const char *api,
                                uintptr_t arg0,
                                uintptr_t arg1,
                                const void *result,
                                int errno_in,
                                int errno_out);
void oai_memprof_r0_emit_summary(uint32_t real_calls,
                                 uint32_t evaluator_calls,
                                 uint32_t live_allocations,
                                 uint32_t context_probes,
                                 uint32_t evaluator_faults);

uint32_t oai_memprof_r0_emit_failures(void);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_R0_RAW_EMIT_H */
