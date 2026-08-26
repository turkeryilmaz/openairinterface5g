/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _POSIX_C_SOURCE 200809L

#include "oai_memprof_clock.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#if defined(__x86_64__)
#include <cpuid.h>
#endif

#if defined(__x86_64__)
static uint32_t intel_cpuid_15_crystal_hz(unsigned signature)
{
  const unsigned base_family = (signature >> 8) & UINT32_C(0x0f);
  const unsigned extended_family = (signature >> 20) & UINT32_C(0xff);
  const unsigned family = base_family == UINT32_C(0x0f) ? base_family + extended_family : base_family;
  const unsigned base_model = (signature >> 4) & UINT32_C(0x0f);
  const unsigned extended_model = (signature >> 16) & UINT32_C(0x0f);
  const unsigned model = (family == UINT32_C(0x06) || family == UINT32_C(0x0f)) ? base_model | (extended_model << 4) : base_model;
  /* Intel SDM CPUID.15H model-specific nominal crystal-clock table. */
  if (family == UINT32_C(0x06) && model == UINT32_C(0x9e))
    return UINT32_C(24000000);
  return 0;
}
#endif

#define OAI_MEMPROF_CLOCK_MAX_ATTEMPTS 8U

#if defined(__x86_64__)
static uint64_t gcd_u64(uint64_t left, uint64_t right)
{
  while (right != 0) {
    const uint64_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}
#endif

static bool timespec_ns(clockid_t clock_id, uint64_t *result)
{
  struct timespec value = {0};
  if (clock_gettime(clock_id, &value) != 0 || value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000L)
    return false;
  const uint64_t seconds = (uint64_t)value.tv_sec;
  if (seconds > UINT64_MAX / UINT64_C(1000000000))
    return false;
  *result = seconds * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
  return true;
}

static uint64_t architectural_counter(void)
{
#if defined(__x86_64__)
  uint32_t low = 0;
  uint32_t high = 0;
  uint32_t auxiliary = 0;
  __asm__ volatile("rdtscp" : "=a"(low), "=d"(high), "=c"(auxiliary) : : "memory");
  __asm__ volatile("lfence" : : : "memory");
  (void)auxiliary;
  return ((uint64_t)high << 32) | low;
#elif defined(__aarch64__)
  uint64_t counter = 0;
  __asm__ volatile("isb\n\tmrs %0, cntvct_el0\n\tisb" : "=r"(counter) : : "memory");
  return counter;
#else
#error "the memory-profiler clock admits only x86-64 and AArch64"
#endif
}

oai_memprof_clock_status_t oai_memprof_clock_info_v1(oai_memprof_clock_info_v1_t *info)
{
  if (info == NULL)
    return OAI_MEMPROF_CLOCK_INVALID_ARGUMENT;

  oai_memprof_clock_info_v1_t value = {0};
#if defined(__x86_64__)
  const unsigned maximum_basic = __get_cpuid_max(0, NULL);
  const unsigned maximum_extended = __get_cpuid_max(UINT32_C(0x80000000), NULL);
  unsigned eax = 0;
  unsigned ebx = 0;
  unsigned ecx = 0;
  unsigned edx = 0;
  if (maximum_basic < UINT32_C(0x15) || maximum_extended < UINT32_C(0x80000007)
      || !__get_cpuid(UINT32_C(0x80000001), &eax, &ebx, &ecx, &edx) || (edx & (UINT32_C(1) << 27)) == 0
      || !__get_cpuid(UINT32_C(0x80000007), &eax, &ebx, &ecx, &edx) || (edx & (UINT32_C(1) << 8)) == 0)
    return OAI_MEMPROF_CLOCK_UNSUPPORTED;
  if (!__get_cpuid_count(UINT32_C(0x15), 0, &eax, &ebx, &ecx, &edx) || eax == 0 || ebx == 0)
    return OAI_MEMPROF_CLOCK_UNSUPPORTED;
  const uint64_t denominator = eax;
  const uint64_t ratio_numerator = ebx;
  uint64_t crystal_hz = ecx;
  if (crystal_hz == 0) {
    unsigned signature = 0;
    unsigned ignored_b = 0;
    unsigned ignored_c = 0;
    unsigned ignored_d = 0;
    unsigned vendor_a = 0;
    unsigned vendor_b = 0;
    unsigned vendor_c = 0;
    unsigned vendor_d = 0;
    __cpuid(0, vendor_a, vendor_b, vendor_c, vendor_d);
    (void)vendor_a;
    const bool genuine_intel =
        vendor_b == UINT32_C(0x756e6547) && vendor_d == UINT32_C(0x49656e69) && vendor_c == UINT32_C(0x6c65746e);
    if (!genuine_intel || !__get_cpuid(1, &signature, &ignored_b, &ignored_c, &ignored_d))
      return OAI_MEMPROF_CLOCK_UNSUPPORTED;
    crystal_hz = intel_cpuid_15_crystal_hz(signature);
    if (crystal_hz == 0)
      return OAI_MEMPROF_CLOCK_UNSUPPORTED;
  }
  if (crystal_hz > UINT64_MAX / ratio_numerator)
    return OAI_MEMPROF_CLOCK_UNSUPPORTED;
  uint64_t numerator = crystal_hz * ratio_numerator;
  uint64_t reduced_denominator = denominator;
  const uint64_t divisor = gcd_u64(numerator, reduced_denominator);
  numerator /= divisor;
  reduced_denominator /= divisor;
  value.counter_frequency_numerator = numerator;
  value.counter_frequency_denominator = reduced_denominator;
  value.architecture_id = OAI_MEMPROF_CLOCK_ARCHITECTURE_X86_64;
  value.acquisition_source_id = OAI_MEMPROF_CLOCK_SOURCE_X86_CPUID_15_EXACT;
  value.clock_kind = OAI_MEMPROF_CLOCK_KIND_X86_TSC;
#elif defined(__aarch64__)
  uint64_t frequency = 0;
  __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
  if (frequency == 0)
    return OAI_MEMPROF_CLOCK_UNSUPPORTED;
  value.counter_frequency_numerator = frequency;
  value.counter_frequency_denominator = 1;
  value.architecture_id = OAI_MEMPROF_CLOCK_ARCHITECTURE_AARCH64;
  value.acquisition_source_id = OAI_MEMPROF_CLOCK_SOURCE_AARCH64_CNTFRQ_EL0_EXACT;
  value.clock_kind = OAI_MEMPROF_CLOCK_KIND_AARCH64_CNTVCT_EL0;
#endif
  *info = value;
  return OAI_MEMPROF_CLOCK_OK;
}

oai_memprof_clock_status_t oai_memprof_clock_sample_v1(uint64_t max_bracket_ns, oai_memprof_clock_sample_v1_t *sample)
{
  if (sample == NULL || max_bracket_ns == 0)
    return OAI_MEMPROF_CLOCK_INVALID_ARGUMENT;

  for (unsigned attempt = 0; attempt < OAI_MEMPROF_CLOCK_MAX_ATTEMPTS; ++attempt) {
    oai_memprof_clock_sample_v1_t value = {0};
    if (!timespec_ns(CLOCK_MONOTONIC_RAW, &value.monotonic_raw_before_ns))
      return OAI_MEMPROF_CLOCK_SYSTEM_ERROR;
    value.counter = architectural_counter();
    if (!timespec_ns(CLOCK_REALTIME, &value.realtime_unix_ns) || !timespec_ns(CLOCK_MONOTONIC_RAW, &value.monotonic_raw_after_ns))
      return OAI_MEMPROF_CLOCK_SYSTEM_ERROR;
    if (value.counter == 0 || value.monotonic_raw_after_ns < value.monotonic_raw_before_ns)
      return OAI_MEMPROF_CLOCK_SYSTEM_ERROR;
    if (value.monotonic_raw_after_ns - value.monotonic_raw_before_ns <= max_bracket_ns) {
      *sample = value;
      return OAI_MEMPROF_CLOCK_OK;
    }
  }
  return OAI_MEMPROF_CLOCK_BRACKET_TOO_WIDE;
}
