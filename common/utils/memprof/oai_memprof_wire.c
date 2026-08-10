/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "oai_memprof_wire.h"

#include <limits.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8, "schema-v1 requires 8-bit bytes");
_Static_assert(sizeof(uint16_t) == 2, "schema-v1 requires a 16-bit uint16_t");
_Static_assert(sizeof(uint32_t) == 4, "schema-v1 requires a 32-bit uint32_t");
_Static_assert(sizeof(uint64_t) == 8, "schema-v1 requires a 64-bit uint64_t");
_Static_assert(sizeof(int32_t) == 4, "schema-v1 requires a 32-bit int32_t");

_Static_assert(OAI_MEMPROF_EVENT_V1_COUNTER_ENTER_OFFSET == OAI_MEMPROF_EVENT_V1_THREAD_SEQUENCE_OFFSET + 8,
               "invalid schema-v1 thread_sequence extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_COUNTER_EXIT_OFFSET == OAI_MEMPROF_EVENT_V1_COUNTER_ENTER_OFFSET + 8,
               "invalid schema-v1 counter_enter extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_ADDRESS_BEFORE_OFFSET == OAI_MEMPROF_EVENT_V1_COUNTER_EXIT_OFFSET + 8,
               "invalid schema-v1 counter_exit extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_ADDRESS_AFTER_OFFSET == OAI_MEMPROF_EVENT_V1_ADDRESS_BEFORE_OFFSET + 8,
               "invalid schema-v1 address_before extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_ARG0_OFFSET == OAI_MEMPROF_EVENT_V1_ADDRESS_AFTER_OFFSET + 8,
               "invalid schema-v1 address_after extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_ARG1_OFFSET == OAI_MEMPROF_EVENT_V1_ARG0_OFFSET + 8, "invalid schema-v1 arg0 extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_ARG2_OFFSET == OAI_MEMPROF_EVENT_V1_ARG1_OFFSET + 8, "invalid schema-v1 arg1 extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_CONTEXT_ID_OFFSET == OAI_MEMPROF_EVENT_V1_ARG2_OFFSET + 8, "invalid schema-v1 arg2 extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_CALLSITE_ID_OFFSET == OAI_MEMPROF_EVENT_V1_CONTEXT_ID_OFFSET + 4,
               "invalid schema-v1 context_id extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_THREAD_INDEX_OFFSET == OAI_MEMPROF_EVENT_V1_CALLSITE_ID_OFFSET + 4,
               "invalid schema-v1 callsite_id extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_FLAGS_OFFSET == OAI_MEMPROF_EVENT_V1_THREAD_INDEX_OFFSET + 4,
               "invalid schema-v1 thread_index extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_RESULT_CODE_OFFSET == OAI_MEMPROF_EVENT_V1_FLAGS_OFFSET + 4, "invalid schema-v1 flags extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_API_ID_OFFSET == OAI_MEMPROF_EVENT_V1_RESULT_CODE_OFFSET + 4,
               "invalid schema-v1 result_code extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_EVENT_KIND_OFFSET == OAI_MEMPROF_EVENT_V1_API_ID_OFFSET + 2, "invalid schema-v1 api_id extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_CPU_ENTER_OFFSET == OAI_MEMPROF_EVENT_V1_EVENT_KIND_OFFSET + 2,
               "invalid schema-v1 event_kind extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_CPU_EXIT_OFFSET == OAI_MEMPROF_EVENT_V1_CPU_ENTER_OFFSET + 2,
               "invalid schema-v1 cpu_enter extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_RESERVED_ZERO_OFFSET == OAI_MEMPROF_EVENT_V1_CPU_EXIT_OFFSET + 2,
               "invalid schema-v1 cpu_exit extent");
_Static_assert(OAI_MEMPROF_EVENT_V1_WIRE_SIZE == OAI_MEMPROF_EVENT_V1_RESERVED_ZERO_OFFSET + 4,
               "invalid schema-v1 reserved extent");

static void store_u16_le(uint8_t *destination, uint16_t value)
{
  destination[0] = (uint8_t)(value & UINT16_C(0x00ff));
  destination[1] = (uint8_t)(value >> 8);
}

static void store_u32_le(uint8_t *destination, uint32_t value)
{
  destination[0] = (uint8_t)(value & UINT32_C(0x000000ff));
  destination[1] = (uint8_t)((value >> 8) & UINT32_C(0x000000ff));
  destination[2] = (uint8_t)((value >> 16) & UINT32_C(0x000000ff));
  destination[3] = (uint8_t)(value >> 24);
}

static void store_u64_le(uint8_t *destination, uint64_t value)
{
  destination[0] = (uint8_t)(value & UINT64_C(0x00000000000000ff));
  destination[1] = (uint8_t)((value >> 8) & UINT64_C(0x00000000000000ff));
  destination[2] = (uint8_t)((value >> 16) & UINT64_C(0x00000000000000ff));
  destination[3] = (uint8_t)((value >> 24) & UINT64_C(0x00000000000000ff));
  destination[4] = (uint8_t)((value >> 32) & UINT64_C(0x00000000000000ff));
  destination[5] = (uint8_t)((value >> 40) & UINT64_C(0x00000000000000ff));
  destination[6] = (uint8_t)((value >> 48) & UINT64_C(0x00000000000000ff));
  destination[7] = (uint8_t)(value >> 56);
}

static uint16_t load_u16_le(const uint8_t *source)
{
  return (uint16_t)((uint16_t)source[0] | (uint16_t)((uint16_t)source[1] << 8));
}

static uint32_t load_u32_le(const uint8_t *source)
{
  return (uint32_t)source[0] | ((uint32_t)source[1] << 8) | ((uint32_t)source[2] << 16) | ((uint32_t)source[3] << 24);
}

static uint64_t load_u64_le(const uint8_t *source)
{
  return (uint64_t)source[0] | ((uint64_t)source[1] << 8) | ((uint64_t)source[2] << 16) | ((uint64_t)source[3] << 24)
         | ((uint64_t)source[4] << 32) | ((uint64_t)source[5] << 40) | ((uint64_t)source[6] << 48) | ((uint64_t)source[7] << 56);
}

/* Schema-v1 i32 values use a two's-complement little-endian wire encoding. */
static uint32_t encode_i32(int32_t value)
{
  if (value >= 0)
    return (uint32_t)value;

  return UINT32_MAX - (uint32_t)(-(value + INT32_C(1)));
}

static int32_t decode_i32(uint32_t value)
{
  if (value <= (uint32_t)INT32_MAX)
    return (int32_t)value;

  return (int32_t)(-INT64_C(1) - (int64_t)(UINT32_MAX - value));
}

oai_memprof_wire_status_t oai_memprof_event_v1_encode(const oai_memprof_event_v1_t *event, uint8_t *wire, size_t wire_size)
{
  if (event == NULL || wire == NULL)
    return OAI_MEMPROF_WIRE_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_EVENT_V1_WIRE_SIZE)
    return OAI_MEMPROF_WIRE_WRONG_SIZE;

  uint8_t encoded[OAI_MEMPROF_EVENT_V1_WIRE_SIZE] = {0};
  store_u64_le(encoded + OAI_MEMPROF_EVENT_V1_THREAD_SEQUENCE_OFFSET, event->thread_sequence);
  store_u64_le(encoded + OAI_MEMPROF_EVENT_V1_COUNTER_ENTER_OFFSET, event->counter_enter);
  store_u64_le(encoded + OAI_MEMPROF_EVENT_V1_COUNTER_EXIT_OFFSET, event->counter_exit);
  store_u64_le(encoded + OAI_MEMPROF_EVENT_V1_ADDRESS_BEFORE_OFFSET, event->address_before);
  store_u64_le(encoded + OAI_MEMPROF_EVENT_V1_ADDRESS_AFTER_OFFSET, event->address_after);
  store_u64_le(encoded + OAI_MEMPROF_EVENT_V1_ARG0_OFFSET, event->arg0);
  store_u64_le(encoded + OAI_MEMPROF_EVENT_V1_ARG1_OFFSET, event->arg1);
  store_u64_le(encoded + OAI_MEMPROF_EVENT_V1_ARG2_OFFSET, event->arg2);
  store_u32_le(encoded + OAI_MEMPROF_EVENT_V1_CONTEXT_ID_OFFSET, event->context_id);
  store_u32_le(encoded + OAI_MEMPROF_EVENT_V1_CALLSITE_ID_OFFSET, event->callsite_id);
  store_u32_le(encoded + OAI_MEMPROF_EVENT_V1_THREAD_INDEX_OFFSET, event->thread_index);
  store_u32_le(encoded + OAI_MEMPROF_EVENT_V1_FLAGS_OFFSET, event->flags);
  store_u32_le(encoded + OAI_MEMPROF_EVENT_V1_RESULT_CODE_OFFSET, encode_i32(event->result_code));
  store_u16_le(encoded + OAI_MEMPROF_EVENT_V1_API_ID_OFFSET, event->api_id);
  store_u16_le(encoded + OAI_MEMPROF_EVENT_V1_EVENT_KIND_OFFSET, event->event_kind);
  store_u16_le(encoded + OAI_MEMPROF_EVENT_V1_CPU_ENTER_OFFSET, event->cpu_enter);
  store_u16_le(encoded + OAI_MEMPROF_EVENT_V1_CPU_EXIT_OFFSET, event->cpu_exit);

  memcpy(wire, encoded, sizeof(encoded));
  return OAI_MEMPROF_WIRE_OK;
}

oai_memprof_wire_status_t oai_memprof_event_v1_decode(oai_memprof_event_v1_t *event, const uint8_t *wire, size_t wire_size)
{
  if (event == NULL || wire == NULL)
    return OAI_MEMPROF_WIRE_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_EVENT_V1_WIRE_SIZE)
    return OAI_MEMPROF_WIRE_WRONG_SIZE;
  if (load_u32_le(wire + OAI_MEMPROF_EVENT_V1_RESERVED_ZERO_OFFSET) != 0)
    return OAI_MEMPROF_WIRE_NONZERO_RESERVED;

  oai_memprof_event_v1_t decoded = {
      .thread_sequence = load_u64_le(wire + OAI_MEMPROF_EVENT_V1_THREAD_SEQUENCE_OFFSET),
      .counter_enter = load_u64_le(wire + OAI_MEMPROF_EVENT_V1_COUNTER_ENTER_OFFSET),
      .counter_exit = load_u64_le(wire + OAI_MEMPROF_EVENT_V1_COUNTER_EXIT_OFFSET),
      .address_before = load_u64_le(wire + OAI_MEMPROF_EVENT_V1_ADDRESS_BEFORE_OFFSET),
      .address_after = load_u64_le(wire + OAI_MEMPROF_EVENT_V1_ADDRESS_AFTER_OFFSET),
      .arg0 = load_u64_le(wire + OAI_MEMPROF_EVENT_V1_ARG0_OFFSET),
      .arg1 = load_u64_le(wire + OAI_MEMPROF_EVENT_V1_ARG1_OFFSET),
      .arg2 = load_u64_le(wire + OAI_MEMPROF_EVENT_V1_ARG2_OFFSET),
      .context_id = load_u32_le(wire + OAI_MEMPROF_EVENT_V1_CONTEXT_ID_OFFSET),
      .callsite_id = load_u32_le(wire + OAI_MEMPROF_EVENT_V1_CALLSITE_ID_OFFSET),
      .thread_index = load_u32_le(wire + OAI_MEMPROF_EVENT_V1_THREAD_INDEX_OFFSET),
      .flags = load_u32_le(wire + OAI_MEMPROF_EVENT_V1_FLAGS_OFFSET),
      .result_code = decode_i32(load_u32_le(wire + OAI_MEMPROF_EVENT_V1_RESULT_CODE_OFFSET)),
      .api_id = load_u16_le(wire + OAI_MEMPROF_EVENT_V1_API_ID_OFFSET),
      .event_kind = load_u16_le(wire + OAI_MEMPROF_EVENT_V1_EVENT_KIND_OFFSET),
      .cpu_enter = load_u16_le(wire + OAI_MEMPROF_EVENT_V1_CPU_ENTER_OFFSET),
      .cpu_exit = load_u16_le(wire + OAI_MEMPROF_EVENT_V1_CPU_EXIT_OFFSET),
  };

  *event = decoded;
  return OAI_MEMPROF_WIRE_OK;
}
