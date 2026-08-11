/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "r0_scripted_backend.h"

#include "r0_raw_emit.h"

#include <errno.h>
#include <stddef.h>

enum {
  OAI_MEMPROF_R0_POINTER_COUNT = 8,
  OAI_MEMPROF_R0_SLOT_SIZE = 128,
};

typedef union oai_memprof_r0_slot_u {
  max_align_t alignment;
  unsigned char bytes[OAI_MEMPROF_R0_SLOT_SIZE];
} oai_memprof_r0_slot_t;

static oai_memprof_r0_slot_t pointer_slots[OAI_MEMPROF_R0_POINTER_COUNT];
static uint32_t current_transaction;
static uint32_t real_calls;
static uint32_t live_mask;
static uint32_t context_probes;

static void *token_pointer(unsigned int token_index)
{
  if (token_index == 0 || token_index > OAI_MEMPROF_R0_POINTER_COUNT)
    return NULL;
  return pointer_slots[token_index - 1].bytes;
}

static unsigned int pointer_token(const void *pointer)
{
  for (unsigned int i = 1; i <= OAI_MEMPROF_R0_POINTER_COUNT; ++i) {
    if (pointer == token_pointer(i))
      return i;
  }
  return 0;
}

static void mark_live(const void *pointer)
{
  const unsigned int token = pointer_token(pointer);
  if (token != 0)
    live_mask |= UINT32_C(1) << (token - 1);
}

static void mark_released(const void *pointer)
{
  const unsigned int token = pointer_token(pointer);
  if (token != 0)
    live_mask &= ~(UINT32_C(1) << (token - 1));
}

static uint32_t count_live(void)
{
  uint32_t bits = live_mask;
  uint32_t count = 0;
  while (bits != 0) {
    count += bits & 1U;
    bits >>= 1;
  }
  return count;
}

static void emit_token(unsigned int token_index)
{
  char name[] = "P0";
  name[1] = (char)('0' + token_index);
  oai_memprof_r0_emit_token(name, token_pointer(token_index));
}

void oai_memprof_r0_backend_begin(const char *mode)
{
  current_transaction = 0;
  real_calls = 0;
  live_mask = 0;
  context_probes = 0;

  oai_memprof_r0_emit_meta(mode);
  oai_memprof_r0_emit_token("NULL", NULL);
  for (unsigned int i = 1; i <= OAI_MEMPROF_R0_POINTER_COUNT; ++i)
    emit_token(i);
}

void oai_memprof_r0_set_transaction(uint32_t transaction)
{
  current_transaction = transaction;
}

uint32_t oai_memprof_r0_current_transaction(void)
{
  return current_transaction;
}

uint32_t oai_memprof_r0_real_calls(void)
{
  return real_calls;
}

uint32_t oai_memprof_r0_live_allocations(void)
{
  return count_live();
}

void oai_memprof_r0_context_probe(void)
{
  ++context_probes;
}

uint32_t oai_memprof_r0_context_probes(void)
{
  return context_probes;
}

void *__real_malloc(size_t size)
{
  const int errno_in = errno;
  void *result = NULL;

  switch (current_transaction) {
    case 1:
      result = token_pointer(1);
      break;
    case 3:
      result = token_pointer(3);
      break;
    case 4:
      errno = ENOMEM;
      break;
    default:
      break;
  }
  mark_live(result);

  const int errno_out = errno;
  const uint32_t sequence = ++real_calls;
  oai_memprof_r0_emit_real(sequence, current_transaction, "malloc", (uintptr_t)size, 0, result, errno_in, errno_out);
  return result;
}

void *__real_calloc(size_t count, size_t size)
{
  const int errno_in = errno;
  void *result = NULL;

  switch (current_transaction) {
    case 2:
      result = token_pointer(2);
      break;
    case 5:
      result = token_pointer(4);
      break;
    case 6:
      result = token_pointer(5);
      break;
    case 7:
    case 8:
      errno = ENOMEM;
      break;
    default:
      break;
  }
  mark_live(result);

  const int errno_out = errno;
  const uint32_t sequence = ++real_calls;
  oai_memprof_r0_emit_real(sequence, current_transaction, "calloc", (uintptr_t)count, (uintptr_t)size, result, errno_in, errno_out);
  return result;
}

void *__real_realloc(void *pointer, size_t size)
{
  const int errno_in = errno;
  void *result = NULL;

  switch (current_transaction) {
    case 9:
      result = token_pointer(6);
      mark_live(result);
      break;
    case 10:
      result = token_pointer(1);
      break;
    case 11:
      mark_released(pointer);
      result = token_pointer(7);
      mark_live(result);
      break;
    case 12:
      errno = ENOMEM;
      break;
    case 13:
      mark_released(pointer);
      break;
    case 14:
      result = token_pointer(8);
      mark_live(result);
      break;
    case 20:
      result = token_pointer(8);
      break;
    default:
      break;
  }

  const int errno_out = errno;
  const uint32_t sequence = ++real_calls;
  oai_memprof_r0_emit_real(sequence,
                           current_transaction,
                           "realloc",
                           (uintptr_t)pointer,
                           (uintptr_t)size,
                           result,
                           errno_in,
                           errno_out);
  return result;
}

void __real_free(void *pointer)
{
  const int errno_in = errno;
  mark_released(pointer);
  const int errno_out = errno;
  const uint32_t sequence = ++real_calls;
  oai_memprof_r0_emit_real(sequence, current_transaction, "free", (uintptr_t)pointer, 0, NULL, errno_in, errno_out);
}
