/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_R0_SCRIPTED_BACKEND_H
#define OAI_MEMPROF_R0_SCRIPTED_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void oai_memprof_r0_backend_begin(const char *mode);
void oai_memprof_r0_set_transaction(uint32_t transaction);
uint32_t oai_memprof_r0_current_transaction(void);
uint32_t oai_memprof_r0_real_calls(void);
uint32_t oai_memprof_r0_live_allocations(void);

void oai_memprof_r0_context_probe(void);
uint32_t oai_memprof_r0_context_probes(void);

void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void *__real_realloc(void *pointer, size_t size);
void __real_free(void *pointer);

void *__wrap_malloc(size_t size);
void *__wrap_calloc(size_t count, size_t size);
void *__wrap_realloc(void *pointer, size_t size);
void __wrap_free(void *pointer);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_R0_SCRIPTED_BACKEND_H */
