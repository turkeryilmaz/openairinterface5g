/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "r0_raw_emit.h"

#include <errno.h>
#include <limits.h>
#include <unistd.h>

enum {
  OAI_MEMPROF_R0_RECORD_CAPACITY = 512,
  OAI_MEMPROF_R0_MAX_WRITE_ATTEMPTS = 16,
};

typedef struct oai_memprof_r0_record_s {
  char bytes[OAI_MEMPROF_R0_RECORD_CAPACITY];
  size_t length;
  int valid;
} oai_memprof_r0_record_t;

static uint32_t emit_failures;

static void append_char(oai_memprof_r0_record_t *record, char value)
{
  if (!record->valid)
    return;
  if (record->length == sizeof(record->bytes)) {
    record->valid = 0;
    return;
  }
  record->bytes[record->length++] = value;
}

static void append_text(oai_memprof_r0_record_t *record, const char *text)
{
  if (text == NULL) {
    record->valid = 0;
    return;
  }
  while (*text != '\0')
    append_char(record, *text++);
}

static void append_u64_decimal(oai_memprof_r0_record_t *record, uint64_t value)
{
  char reversed[20];
  size_t digits = 0;

  do {
    reversed[digits++] = (char)('0' + value % 10);
    value /= 10;
  } while (value != 0);

  while (digits != 0)
    append_char(record, reversed[--digits]);
}

static void append_i32_decimal(oai_memprof_r0_record_t *record, int value)
{
  int64_t wide = value;
  if (wide < 0) {
    append_char(record, '-');
    wide = -wide;
  }
  append_u64_decimal(record, (uint64_t)wide);
}

static void append_uintptr_hex(oai_memprof_r0_record_t *record, uintptr_t value)
{
  static const char digits[] = "0123456789abcdef";
  char reversed[sizeof(uintptr_t) * CHAR_BIT / 4];
  size_t count = 0;

  append_text(record, "0x");
  do {
    reversed[count++] = digits[value & 0xfU];
    value >>= 4;
  } while (value != 0);

  while (count != 0)
    append_char(record, reversed[--count]);
}

static void append_separator(oai_memprof_r0_record_t *record)
{
  append_char(record, '|');
}

static void finish_record(oai_memprof_r0_record_t *record)
{
  const int saved_errno = errno;
  size_t written = 0;
  unsigned int attempts = 0;

  append_char(record, '\n');
  if (!record->valid) {
    ++emit_failures;
    errno = saved_errno;
    return;
  }

  while (written != record->length && attempts++ != OAI_MEMPROF_R0_MAX_WRITE_ATTEMPTS) {
    const ssize_t result = write(STDOUT_FILENO, record->bytes + written, record->length - written);
    if (result > 0) {
      written += (size_t)result;
      continue;
    }
    if (result < 0 && errno == EINTR)
      continue;
    break;
  }
  if (written != record->length)
    ++emit_failures;
  errno = saved_errno;
}

void oai_memprof_r0_emit_meta(const char *mode)
{
  oai_memprof_r0_record_t record = {.valid = 1};
  append_text(&record, "META|schema|oai-memprof-r0-raw-v1");
  finish_record(&record);

  record = (oai_memprof_r0_record_t){.valid = 1};
  append_text(&record, "META|mode|");
  append_text(&record, mode);
  finish_record(&record);
}

void oai_memprof_r0_emit_token(const char *token, const void *address)
{
  oai_memprof_r0_record_t record = {.valid = 1};
  append_text(&record, "TOKEN|");
  append_text(&record, token);
  append_separator(&record);
  append_uintptr_hex(&record, (uintptr_t)address);
  finish_record(&record);
}

void oai_memprof_r0_emit_eval(uint32_t transaction,
                              const char *phase,
                              const char *operand,
                              uint32_t evaluator,
                              const char *kind,
                              uintptr_t value)
{
  oai_memprof_r0_record_t record = {.valid = 1};
  append_text(&record, "EVAL|");
  append_u64_decimal(&record, transaction);
  append_separator(&record);
  append_text(&record, phase);
  append_separator(&record);
  append_text(&record, operand);
  append_separator(&record);
  append_u64_decimal(&record, evaluator);
  append_separator(&record);
  append_text(&record, kind);
  append_separator(&record);
  append_uintptr_hex(&record, value);
  finish_record(&record);
}

void oai_memprof_r0_emit_real(uint32_t sequence,
                              uint32_t transaction,
                              const char *api,
                              uintptr_t arg0,
                              uintptr_t arg1,
                              const void *result,
                              int errno_in,
                              int errno_out)
{
  oai_memprof_r0_record_t record = {.valid = 1};
  append_text(&record, "REAL|");
  append_u64_decimal(&record, sequence);
  append_separator(&record);
  append_u64_decimal(&record, transaction);
  append_separator(&record);
  append_text(&record, api);
  append_separator(&record);
  append_uintptr_hex(&record, arg0);
  append_separator(&record);
  append_uintptr_hex(&record, arg1);
  append_separator(&record);
  append_uintptr_hex(&record, (uintptr_t)result);
  append_separator(&record);
  append_i32_decimal(&record, errno_in);
  append_separator(&record);
  append_i32_decimal(&record, errno_out);
  finish_record(&record);
}

void oai_memprof_r0_emit_caller(uint32_t transaction,
                                const char *phase,
                                const char *api,
                                uintptr_t arg0,
                                uintptr_t arg1,
                                const void *result,
                                int errno_in,
                                int errno_out)
{
  oai_memprof_r0_record_t record = {.valid = 1};
  append_text(&record, "CALLER|");
  append_u64_decimal(&record, transaction);
  append_separator(&record);
  append_text(&record, phase);
  append_separator(&record);
  append_text(&record, api);
  append_separator(&record);
  append_uintptr_hex(&record, arg0);
  append_separator(&record);
  append_uintptr_hex(&record, arg1);
  append_separator(&record);
  append_uintptr_hex(&record, (uintptr_t)result);
  append_separator(&record);
  append_i32_decimal(&record, errno_in);
  append_separator(&record);
  append_i32_decimal(&record, errno_out);
  finish_record(&record);
}

void oai_memprof_r0_emit_summary(uint32_t real_calls,
                                 uint32_t evaluator_calls,
                                 uint32_t live_allocations,
                                 uint32_t context_probes,
                                 uint32_t evaluator_faults)
{
  oai_memprof_r0_record_t record = {.valid = 1};
  append_text(&record, "SUMMARY|");
  append_u64_decimal(&record, real_calls);
  append_separator(&record);
  append_u64_decimal(&record, evaluator_calls);
  append_separator(&record);
  append_u64_decimal(&record, live_allocations);
  append_separator(&record);
  append_u64_decimal(&record, context_probes);
  append_separator(&record);
  append_u64_decimal(&record, evaluator_faults);
  append_separator(&record);
  append_u64_decimal(&record, emit_failures);
  finish_record(&record);
}

uint32_t oai_memprof_r0_emit_failures(void)
{
  return emit_failures;
}
