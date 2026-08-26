/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_CLOCK_H
#define OAI_MEMPROF_CLOCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum oai_memprof_clock_architecture_v1 {
  OAI_MEMPROF_CLOCK_ARCHITECTURE_X86_64 = 1,
  OAI_MEMPROF_CLOCK_ARCHITECTURE_AARCH64 = 2,
};

enum oai_memprof_clock_kind_v1 {
  OAI_MEMPROF_CLOCK_KIND_X86_TSC = 1,
  OAI_MEMPROF_CLOCK_KIND_AARCH64_CNTVCT_EL0 = 2,
};

enum oai_memprof_clock_acquisition_source_v1 {
  OAI_MEMPROF_CLOCK_SOURCE_X86_CPUID_15_EXACT = 1,
  OAI_MEMPROF_CLOCK_SOURCE_AARCH64_CNTFRQ_EL0_EXACT = 2,
};

typedef enum oai_memprof_clock_status_e {
  OAI_MEMPROF_CLOCK_OK = 0,
  OAI_MEMPROF_CLOCK_INVALID_ARGUMENT,
  OAI_MEMPROF_CLOCK_UNSUPPORTED,
  OAI_MEMPROF_CLOCK_SYSTEM_ERROR,
  OAI_MEMPROF_CLOCK_BRACKET_TOO_WIDE,
  OAI_MEMPROF_CLOCK_SEQUENCE_ERROR,
} oai_memprof_clock_status_t;

typedef struct oai_memprof_clock_info_v1_s {
  uint64_t counter_frequency_numerator;
  uint64_t counter_frequency_denominator;
  uint16_t architecture_id;
  uint16_t acquisition_source_id;
  uint8_t clock_kind;
  uint8_t reserved_zero[3];
} oai_memprof_clock_info_v1_t;

typedef struct oai_memprof_clock_sample_v1_s {
  uint64_t counter;
  uint64_t monotonic_raw_before_ns;
  uint64_t monotonic_raw_after_ns;
  uint64_t realtime_unix_ns;
} oai_memprof_clock_sample_v1_t;

/* Output is unchanged on every error. */
oai_memprof_clock_status_t oai_memprof_clock_info_v1(oai_memprof_clock_info_v1_t *info);

/*
 * Acquire counter and realtime inside an ordered CLOCK_MONOTONIC_RAW bracket.
 * Up to eight attempts are made to meet max_bracket_ns. Output is unchanged on
 * every error. A zero maximum is invalid.
 */
oai_memprof_clock_status_t oai_memprof_clock_sample_v1(uint64_t max_bracket_ns, oai_memprof_clock_sample_v1_t *sample);

#ifdef __cplusplus
}
#endif

#endif /* OAI_MEMPROF_CLOCK_H */
