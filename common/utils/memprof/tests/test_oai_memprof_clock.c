/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "common/utils/memprof/oai_memprof_clock.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      exit(EXIT_FAILURE);                                                             \
    }                                                                                 \
  } while (0)

static uint64_t gcd_u64(uint64_t left, uint64_t right)
{
  while (right != 0) {
    const uint64_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

int main(void)
{
  oai_memprof_clock_info_v1_t info;
  memset(&info, 0xa5, sizeof(info));
  const oai_memprof_clock_info_v1_t info_sentinel = info;
  CHECK(oai_memprof_clock_info_v1(NULL) == OAI_MEMPROF_CLOCK_INVALID_ARGUMENT);
  CHECK(memcmp(&info, &info_sentinel, sizeof(info)) == 0);
  const oai_memprof_clock_status_t info_status = oai_memprof_clock_info_v1(&info);
  if (info_status == OAI_MEMPROF_CLOCK_UNSUPPORTED) {
    puts("clock exact-rate test skipped: no admitted architectural exact-rate source");
    return 77;
  }
  CHECK(info_status == OAI_MEMPROF_CLOCK_OK);
  CHECK(info.counter_frequency_numerator != 0 && info.counter_frequency_denominator != 0);
  CHECK(gcd_u64(info.counter_frequency_numerator, info.counter_frequency_denominator) == 1);
  CHECK(info.reserved_zero[0] == 0 && info.reserved_zero[1] == 0 && info.reserved_zero[2] == 0);
#if defined(__x86_64__)
  CHECK(info.architecture_id == OAI_MEMPROF_CLOCK_ARCHITECTURE_X86_64);
  CHECK(info.clock_kind == OAI_MEMPROF_CLOCK_KIND_X86_TSC);
  CHECK(info.acquisition_source_id == OAI_MEMPROF_CLOCK_SOURCE_X86_CPUID_15_EXACT);
#elif defined(__aarch64__)
  CHECK(info.architecture_id == OAI_MEMPROF_CLOCK_ARCHITECTURE_AARCH64);
  CHECK(info.clock_kind == OAI_MEMPROF_CLOCK_KIND_AARCH64_CNTVCT_EL0);
  CHECK(info.acquisition_source_id == OAI_MEMPROF_CLOCK_SOURCE_AARCH64_CNTFRQ_EL0_EXACT);
#endif

  oai_memprof_clock_sample_v1_t sample;
  memset(&sample, 0xa5, sizeof(sample));
  const oai_memprof_clock_sample_v1_t sample_sentinel = sample;
  CHECK(oai_memprof_clock_sample_v1(0, &sample) == OAI_MEMPROF_CLOCK_INVALID_ARGUMENT);
  CHECK(memcmp(&sample, &sample_sentinel, sizeof(sample)) == 0);
  CHECK(oai_memprof_clock_sample_v1(UINT64_C(10000000), NULL) == OAI_MEMPROF_CLOCK_INVALID_ARGUMENT);
  CHECK(memcmp(&sample, &sample_sentinel, sizeof(sample)) == 0);

  oai_memprof_clock_sample_v1_t first = {0};
  oai_memprof_clock_sample_v1_t second = {0};
  CHECK(oai_memprof_clock_sample_v1(UINT64_C(10000000), &first) == OAI_MEMPROF_CLOCK_OK);
  CHECK(oai_memprof_clock_sample_v1(UINT64_C(10000000), &second) == OAI_MEMPROF_CLOCK_OK);
  CHECK(first.monotonic_raw_before_ns <= first.monotonic_raw_after_ns);
  CHECK(first.monotonic_raw_after_ns <= second.monotonic_raw_before_ns);
  CHECK(first.counter < second.counter);
  CHECK(first.realtime_unix_ns != 0 && second.realtime_unix_ns != 0);

  printf("clock exact-rate test passed: architecture=%u source=%u frequency=%" PRIu64 "/%" PRIu64 "\n",
         (unsigned)info.architecture_id,
         (unsigned)info.acquisition_source_id,
         info.counter_frequency_numerator,
         info.counter_frequency_denominator);
  return EXIT_SUCCESS;
}
