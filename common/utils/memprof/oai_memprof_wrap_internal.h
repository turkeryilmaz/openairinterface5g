/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_MEMPROF_WRAP_INTERNAL_H
#define OAI_MEMPROF_WRAP_INTERNAL_H

#include "oai_memprof_runtime_abi.h"

#include <stddef.h>

#if defined(__has_attribute)
#define OAI_MEMPROF_HAS_ATTRIBUTE(attribute) __has_attribute(attribute)
#else
#define OAI_MEMPROF_HAS_ATTRIBUTE(attribute) 0
#endif

#if OAI_MEMPROF_HAS_ATTRIBUTE(noclone) || (defined(__GNUC__) && !defined(__clang__))
#define OAI_MEMPROF_NOCLONE __attribute__((noclone))
#else
/* Clang exposes no source-level noclone attribute; the wrapper target must disable IPO. */
#define OAI_MEMPROF_NOCLONE
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

#define OAI_MEMPROF_WRAPPER_ATTRIBUTES                                                                                  \
  __attribute__((visibility("hidden"), used, noinline, no_instrument_function, no_stack_protector)) OAI_MEMPROF_NOCLONE \
      OAI_MEMPROF_NO_PROFILE_INSTRUMENT OAI_MEMPROF_NO_SANITIZE

#endif /* OAI_MEMPROF_WRAP_INTERNAL_H */
