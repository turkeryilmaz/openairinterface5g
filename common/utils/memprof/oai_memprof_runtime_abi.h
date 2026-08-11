/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_RUNTIME_ABI_H
#define OAI_MEMPROF_RUNTIME_ABI_H

#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>

#if defined(__cplusplus)
#error "the frozen memory-lifetime profiler runtime ABI is C-only"
#endif

#if !defined(__GNUC__) || defined(__clang__)
#error "the frozen memory-lifetime profiler runtime requires GNU GCC"
#endif

#define OAI_MEMPROF_RUNTIME_ABI_VERSION UINT32_C(1)
#define OAI_MEMPROF_CONTROL_CACHE_LINE_BYTES 64

/* R0 deliberately admits no active state or subfield interpretation. */
#define OAI_MEMPROF_CONTROL_PRESENT_OFF UINT64_C(0)

#if defined(__has_attribute)
#define OAI_MEMPROF_HAS_ATTRIBUTE(attribute) __has_attribute(attribute)
#else
#define OAI_MEMPROF_HAS_ATTRIBUTE(attribute) 0
#endif

#if OAI_MEMPROF_HAS_ATTRIBUTE(visibility) || defined(__GNUC__)
#if defined(OAI_MEMPROF_RUNTIME_BUILD)
#define OAI_MEMPROF_RUNTIME_CONTROL_VISIBILITY __attribute__((visibility("protected")))
#else
#define OAI_MEMPROF_RUNTIME_CONTROL_VISIBILITY __attribute__((visibility("default")))
#endif
#else
#error "the memory-lifetime profiler runtime requires ELF symbol visibility support"
#endif

#if OAI_MEMPROF_HAS_ATTRIBUTE(always_inline) || defined(__GNUC__)
#define OAI_MEMPROF_ALWAYS_INLINE __attribute__((always_inline))
#else
#error "the memory-lifetime profiler runtime requires always_inline support"
#endif

#if OAI_MEMPROF_HAS_ATTRIBUTE(no_instrument_function) || defined(__GNUC__)
#define OAI_MEMPROF_NO_INSTRUMENT __attribute__((no_instrument_function))
#else
#error "the memory-lifetime profiler runtime requires no_instrument_function support"
#endif

#if OAI_MEMPROF_HAS_ATTRIBUTE(no_profile_instrument_function)
#define OAI_MEMPROF_NO_PROFILE_INSTRUMENT __attribute__((no_profile_instrument_function))
#else
#define OAI_MEMPROF_NO_PROFILE_INSTRUMENT
#endif

#if OAI_MEMPROF_HAS_ATTRIBUTE(no_sanitize)
#define OAI_MEMPROF_NO_SANITIZE __attribute__((no_sanitize("address", "thread", "undefined")))
#else
#define OAI_MEMPROF_NO_SANITIZE
#endif

#if OAI_MEMPROF_HAS_ATTRIBUTE(no_stack_protector) || defined(__GNUC__)
#define OAI_MEMPROF_NO_STACK_PROTECTOR __attribute__((no_stack_protector))
#else
#error "the memory-lifetime profiler runtime requires no_stack_protector support"
#endif

_Static_assert(CHAR_BIT == 8, "the memory-lifetime profiler control ABI requires 8-bit bytes");
_Static_assert(sizeof(uint64_t) == 8, "the memory-lifetime profiler control ABI requires a 64-bit uint64_t");
_Static_assert(sizeof(_Atomic(uint64_t)) == sizeof(uint64_t), "the memory-lifetime profiler control ABI requires an 8-byte atomic");
#define OAI_MEMPROF_UINT64_LOCK_FREE              \
  _Generic((uint64_t)0,                           \
      unsigned char: ATOMIC_CHAR_LOCK_FREE,       \
      unsigned short: ATOMIC_SHORT_LOCK_FREE,     \
      unsigned int: ATOMIC_INT_LOCK_FREE,         \
      unsigned long: ATOMIC_LONG_LOCK_FREE,       \
      unsigned long long: ATOMIC_LLONG_LOCK_FREE, \
      default: 0)
_Static_assert(OAI_MEMPROF_UINT64_LOCK_FREE == 2, "the memory-lifetime profiler control word must always be lock-free");
_Static_assert(OAI_MEMPROF_CONTROL_PRESENT_OFF == 0, "the PRESENT_OFF control state must have the all-zero representation");
#undef OAI_MEMPROF_UINT64_LOCK_FREE

/*
 * The runtime definition is cache-line aligned and protected against
 * preemption from within its defining DSO. Consumers deliberately use a
 * DEFAULT-visibility import; the terminal link/loader gate proves that every
 * admitted wrapper resolves that import to the same exact runtime object.
 */
extern OAI_MEMPROF_RUNTIME_CONTROL_VISIBILITY _Atomic(uint64_t) oai_memprof_control_v1;

static inline OAI_MEMPROF_ALWAYS_INLINE OAI_MEMPROF_NO_INSTRUMENT OAI_MEMPROF_NO_PROFILE_INSTRUMENT OAI_MEMPROF_NO_SANITIZE
    OAI_MEMPROF_NO_STACK_PROTECTOR uint64_t
    oai_memprof_control_load_v1(void)
{
  return atomic_load_explicit(&oai_memprof_control_v1, memory_order_seq_cst);
}

#undef OAI_MEMPROF_NO_STACK_PROTECTOR
#undef OAI_MEMPROF_NO_SANITIZE
#undef OAI_MEMPROF_NO_PROFILE_INSTRUMENT
#undef OAI_MEMPROF_NO_INSTRUMENT
#undef OAI_MEMPROF_ALWAYS_INLINE
#undef OAI_MEMPROF_RUNTIME_CONTROL_VISIBILITY
#undef OAI_MEMPROF_HAS_ATTRIBUTE

#endif /* OAI_MEMPROF_RUNTIME_ABI_H */
