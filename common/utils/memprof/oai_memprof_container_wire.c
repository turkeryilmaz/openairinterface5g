/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "oai_memprof_container_wire.h"

#include <limits.h>
#include <stdbool.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8, "schema-v1 requires 8-bit bytes");
_Static_assert(sizeof(uint16_t) == 2, "schema-v1 requires a 16-bit uint16_t");
_Static_assert(sizeof(uint32_t) == 4, "schema-v1 requires a 32-bit uint32_t");
_Static_assert(sizeof(uint64_t) == 8, "schema-v1 requires a 64-bit uint64_t");

_Static_assert(OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_CRC32C_OFFSET + 4 == OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE,
               "invalid opening-header extent");
_Static_assert(OAI_MEMPROF_CONTAINER_V1_CHUNK_FLAGS_OFFSET + 4 == OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE,
               "invalid chunk-header extent");
_Static_assert(OAI_MEMPROF_CONTAINER_V1_TRAILER_RESERVED_ZERO_1_OFFSET + 8 == OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE,
               "invalid trailer-header extent");
_Static_assert(OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_RESERVED_ZERO_1_OFFSET + 16 == OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE,
               "invalid event-total extent");
_Static_assert(OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_RESERVED_ZERO_OFFSET + 8
                   == OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE,
               "invalid diagnostic-total extent");
_Static_assert(OAI_MEMPROF_CONTAINER_V1_OBJECT_SHA256_OFFSET + 32 == OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE,
               "invalid object-entry extent");
_Static_assert(OAI_MEMPROF_CONTAINER_V1_FOOTER_FOOTER_SHA256_OFFSET + 32 == OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE,
               "invalid footer extent");
_Static_assert(OAI_MEMPROF_CONTAINER_V1_FOOTER_FOOTER_SHA256_OFFSET == 224, "invalid footer self-hash domain");

static const uint8_t opening_magic[8] = {0x4f, 0x41, 0x49, 0x4d, 0x45, 0x4d, 0x30, 0x31};
static const uint8_t chunk_magic[4] = {0x4f, 0x4d, 0x43, 0x31};
static const uint8_t trailer_magic[16] =
    {0x4f, 0x41, 0x49, 0x5f, 0x4d, 0x45, 0x4d, 0x50, 0x52, 0x4f, 0x46, 0x5f, 0x54, 0x52, 0x31, 0x00};
static const uint8_t footer_magic[16] =
    {0x4f, 0x41, 0x49, 0x5f, 0x4d, 0x45, 0x4d, 0x50, 0x52, 0x4f, 0x46, 0x5f, 0x45, 0x4e, 0x44, 0x00};

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

static uint32_t load_u32_be(const uint8_t *source)
{
  return ((uint32_t)source[0] << 24) | ((uint32_t)source[1] << 16) | ((uint32_t)source[2] << 8) | (uint32_t)source[3];
}

static void store_u32_be(uint8_t *destination, uint32_t value)
{
  destination[0] = (uint8_t)(value >> 24);
  destination[1] = (uint8_t)((value >> 16) & UINT32_C(0x000000ff));
  destination[2] = (uint8_t)((value >> 8) & UINT32_C(0x000000ff));
  destination[3] = (uint8_t)(value & UINT32_C(0x000000ff));
}

static uint32_t rotate_right_u32(uint32_t value, unsigned int shift)
{
  return (value >> shift) | (value << (32U - shift));
}

/* Dependency-free SHA-256 specialized to the fixed 224-byte footer prefix. */
static void sha256_transform(uint32_t state[8], const uint8_t block[64])
{
  static const uint32_t round_constants[64] = {
      UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5), UINT32_C(0x3956c25b),
      UINT32_C(0x59f111f1), UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5), UINT32_C(0xd807aa98), UINT32_C(0x12835b01),
      UINT32_C(0x243185be), UINT32_C(0x550c7dc3), UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7),
      UINT32_C(0xc19bf174), UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786), UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
      UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da), UINT32_C(0x983e5152),
      UINT32_C(0xa831c66d), UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7), UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
      UINT32_C(0x06ca6351), UINT32_C(0x14292967), UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc),
      UINT32_C(0x53380d13), UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
      UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b), UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3), UINT32_C(0xd192e819),
      UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070), UINT32_C(0x19a4c116), UINT32_C(0x1e376c08),
      UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5), UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a), UINT32_C(0x5b9cca4f),
      UINT32_C(0x682e6ff3), UINT32_C(0x748f82ee), UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
      UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2),
  };
  uint32_t schedule[64];
  for (size_t index = 0; index < 16; ++index)
    schedule[index] = load_u32_be(block + index * 4);
  for (size_t index = 16; index < 64; ++index) {
    const uint32_t s0 =
        rotate_right_u32(schedule[index - 15], 7) ^ rotate_right_u32(schedule[index - 15], 18) ^ (schedule[index - 15] >> 3);
    const uint32_t s1 =
        rotate_right_u32(schedule[index - 2], 17) ^ rotate_right_u32(schedule[index - 2], 19) ^ (schedule[index - 2] >> 10);
    schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
  }

  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  uint32_t e = state[4];
  uint32_t f = state[5];
  uint32_t g = state[6];
  uint32_t h = state[7];
  for (size_t index = 0; index < 64; ++index) {
    const uint32_t sum1 = rotate_right_u32(e, 6) ^ rotate_right_u32(e, 11) ^ rotate_right_u32(e, 25);
    const uint32_t choice = (e & f) ^ ((~e) & g);
    const uint32_t temporary1 = h + sum1 + choice + round_constants[index] + schedule[index];
    const uint32_t sum0 = rotate_right_u32(a, 2) ^ rotate_right_u32(a, 13) ^ rotate_right_u32(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

static void sha256_footer_prefix(const uint8_t prefix[OAI_MEMPROF_CONTAINER_V1_FOOTER_FOOTER_SHA256_OFFSET], uint8_t digest[32])
{
  uint32_t state[8] = {
      UINT32_C(0x6a09e667),
      UINT32_C(0xbb67ae85),
      UINT32_C(0x3c6ef372),
      UINT32_C(0xa54ff53a),
      UINT32_C(0x510e527f),
      UINT32_C(0x9b05688c),
      UINT32_C(0x1f83d9ab),
      UINT32_C(0x5be0cd19),
  };
  for (size_t offset = 0; offset < 192; offset += 64)
    sha256_transform(state, prefix + offset);

  uint8_t final_block[64] = {0};
  memcpy(final_block, prefix + 192, 32);
  final_block[32] = UINT8_C(0x80);
  final_block[62] = UINT8_C(0x07);
  final_block[63] = UINT8_C(0x00);
  sha256_transform(state, final_block);

  for (size_t index = 0; index < 8; ++index)
    store_u32_be(digest + index * 4, state[index]);
}

static void sha256_bytes_unchecked(const uint8_t *data, size_t data_size, uint8_t digest[32])
{
  uint32_t state[8] = {
      UINT32_C(0x6a09e667),
      UINT32_C(0xbb67ae85),
      UINT32_C(0x3c6ef372),
      UINT32_C(0xa54ff53a),
      UINT32_C(0x510e527f),
      UINT32_C(0x9b05688c),
      UINT32_C(0x1f83d9ab),
      UINT32_C(0x5be0cd19),
  };
  const size_t full_bytes = data_size - data_size % 64U;
  for (size_t offset = 0; offset < full_bytes; offset += 64U)
    sha256_transform(state, data + offset);

  const size_t remaining = data_size - full_bytes;
  const size_t tail_bytes = remaining <= 55U ? 64U : 128U;
  uint8_t tail[128] = {0};
  if (remaining != 0)
    memcpy(tail, data + full_bytes, remaining);
  tail[remaining] = UINT8_C(0x80);
  const uint64_t bit_length = (uint64_t)data_size * UINT64_C(8);
  for (size_t index = 0; index < 8U; ++index)
    tail[tail_bytes - 1U - index] = (uint8_t)(bit_length >> (index * 8U));
  sha256_transform(state, tail);
  if (tail_bytes == 128U)
    sha256_transform(state, tail + 64U);
  for (size_t index = 0; index < 8U; ++index)
    store_u32_be(digest + index * 4U, state[index]);
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_sha256(const uint8_t *data, size_t data_size, uint8_t digest[32])
{
  if (digest == NULL || (data == NULL && data_size != 0))
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (data_size > UINT64_MAX / UINT64_C(8))
    return OAI_MEMPROF_CONTAINER_V1_INTEGER_OVERFLOW;
  uint8_t calculated[32];
  sha256_bytes_unchecked(data, data_size, calculated);
  memcpy(digest, calculated, sizeof(calculated));
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static bool bytes_are_zero(const uint8_t *bytes, size_t size)
{
  uint8_t combined = 0;
  for (size_t index = 0; index < size; ++index)
    combined = (uint8_t)(combined | bytes[index]);
  return combined == 0;
}

static bool bytes_are_nonzero(const uint8_t *bytes, size_t size)
{
  return !bytes_are_zero(bytes, size);
}

static bool add_u64(uint64_t left, uint64_t right, uint64_t *result)
{
  if (left > UINT64_MAX - right)
    return false;
  *result = left + right;
  return true;
}

static bool multiply_u64(uint64_t left, uint64_t right, uint64_t *result)
{
  if (left != 0 && right > UINT64_MAX / left)
    return false;
  *result = left * right;
  return true;
}

static uint64_t gcd_u64(uint64_t left, uint64_t right)
{
  while (right != 0) {
    const uint64_t remainder = left % right;
    left = right;
    right = remainder;
  }
  return left;
}

static uint32_t crc32c_unchecked(const uint8_t *data, size_t data_size)
{
  uint32_t crc = UINT32_MAX;
  for (size_t index = 0; index < data_size; ++index) {
    crc ^= data[index];
    for (unsigned int bit = 0; bit < 8; ++bit) {
      const uint32_t reflected_polynomial = UINT32_C(0x82f63b78);
      const uint32_t low_bit_mask = UINT32_C(0) - (crc & UINT32_C(1));
      crc = (crc >> 1) ^ (reflected_polynomial & low_bit_mask);
    }
  }
  return crc ^ UINT32_MAX;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_crc32c(const uint8_t *data, size_t data_size, uint32_t *crc32c)
{
  if (crc32c == NULL || (data == NULL && data_size != 0))
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;

  const uint32_t calculated = crc32c_unchecked(data, data_size);
  *crc32c = calculated;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static bool uuid_is_non_nil_rfc_variant(const uint8_t uuid[16])
{
  return bytes_are_nonzero(uuid, 16) && (uuid[8] & UINT8_C(0xc0)) == UINT8_C(0x80);
}

static oai_memprof_container_v1_status_t validate_opening_header(const oai_memprof_container_v1_opening_header_t *header)
{
  if (header->scope_kind != OAI_MEMPROF_CONTAINER_V1_SCOPE_MEASUREMENT_INTERVAL
      && header->scope_kind != OAI_MEMPROF_CONTAINER_V1_SCOPE_PROCESS_LIFETIME)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;
  if (header->role_kind != OAI_MEMPROF_CONTAINER_V1_ROLE_GNB && header->role_kind != OAI_MEMPROF_CONTAINER_V1_ROLE_NR_UE)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;
  if (header->clock_kind != OAI_MEMPROF_CONTAINER_V1_CLOCK_X86_TSC
      && header->clock_kind != OAI_MEMPROF_CONTAINER_V1_CLOCK_AARCH64_CNTVCT_EL0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;
  if (header->calibration_kind != OAI_MEMPROF_CONTAINER_V1_CALIBRATION_EXACT_RATE
      && header->calibration_kind != OAI_MEMPROF_CONTAINER_V1_CALIBRATION_MEASURED_AFFINE)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;
  if (header->source_object_kind != OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_COMMIT)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;
  if (header->source_object_algorithm != OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_SHA1
      && header->source_object_algorithm != OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_SHA256)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;

  if (header->page_size_bytes < UINT32_C(4096) || (header->page_size_bytes & (header->page_size_bytes - UINT32_C(1))) != 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  if (header->process_generation == 0 || header->counter_frequency_numerator == 0 || header->counter_frequency_denominator == 0
      || header->pid == 0 || header->configured_thread_capacity == 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  if (gcd_u64(header->counter_frequency_numerator, header->counter_frequency_denominator) != UINT64_C(1))
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  if (header->calibration_kind == OAI_MEMPROF_CONTAINER_V1_CALIBRATION_EXACT_RATE) {
    if (header->calibration_span_ns != 0)
      return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  } else if (header->calibration_span_ns == 0 || header->calibration_error_bound_ns == 0) {
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  }

  if (!uuid_is_non_nil_rfc_variant(header->run_uuid) || !uuid_is_non_nil_rfc_variant(header->process_uuid))
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  if (memcmp(header->run_uuid, header->process_uuid, 16) == 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  if (header->source_object_algorithm == OAI_MEMPROF_CONTAINER_V1_SOURCE_GIT_SHA1) {
    if (header->source_object_length != 20)
      return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
    if (!bytes_are_zero(header->source_object_value + 20, 12))
      return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  } else if (header->source_object_length != 32) {
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  }
  if (!bytes_are_nonzero(header->source_object_value, header->source_object_length)
      || !bytes_are_nonzero(header->primary_binary_sha256, sizeof(header->primary_binary_sha256))
      || !bytes_are_nonzero(header->schema_bundle_definition_sha256, sizeof(header->schema_bundle_definition_sha256))
      || !bytes_are_nonzero(header->api_catalog_definition_sha256, sizeof(header->api_catalog_definition_sha256))
      || !bytes_are_nonzero(header->callsite_catalog_definition_sha256, sizeof(header->callsite_catalog_definition_sha256))
      || !bytes_are_nonzero(header->configuration_instance_sha256, sizeof(header->configuration_instance_sha256))
      || !bytes_are_nonzero(header->primary_build_id_sha256, sizeof(header->primary_build_id_sha256)))
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;

  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_opening_header_encode(
    const oai_memprof_container_v1_opening_header_t *header,
    uint8_t *wire,
    size_t wire_size)
{
  if (header == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;

  const oai_memprof_container_v1_status_t validation = validate_opening_header(header);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;

  uint8_t encoded[OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE] = {0};
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_MAGIC_OFFSET, opening_magic, sizeof(opening_magic));
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CONTAINER_MAJOR_OFFSET, UINT16_C(1));
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CONTAINER_MINOR_OFFSET, UINT16_C(0));
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_BYTES_OFFSET, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_EVENT_RECORD_BYTES_OFFSET, OAI_MEMPROF_CONTAINER_V1_EVENT_RECORD_SIZE);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CHUNK_HEADER_BYTES_OFFSET, OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_MINIMUM_READER_MINOR_OFFSET, UINT16_C(0));
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_REQUIRED_FEATURES_OFFSET, OAI_MEMPROF_CONTAINER_V1_REQUIRED_FEATURES);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_ENDIAN_MARKER_OFFSET, OAI_MEMPROF_CONTAINER_V1_ENDIAN_MARKER);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_PAGE_SIZE_BYTES_OFFSET, header->page_size_bytes);
  encoded[OAI_MEMPROF_CONTAINER_V1_OPENING_POINTER_WIDTH_BYTES_OFFSET] = UINT8_C(8);
  encoded[OAI_MEMPROF_CONTAINER_V1_OPENING_SCOPE_KIND_OFFSET] = header->scope_kind;
  encoded[OAI_MEMPROF_CONTAINER_V1_OPENING_ROLE_KIND_OFFSET] = header->role_kind;
  encoded[OAI_MEMPROF_CONTAINER_V1_OPENING_CLOCK_KIND_OFFSET] = header->clock_kind;
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_KIND_OFFSET, header->calibration_kind);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_PROCESS_GENERATION_OFFSET, header->process_generation);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_COUNTER_FREQUENCY_NUMERATOR_OFFSET, header->counter_frequency_numerator);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_COUNTER_FREQUENCY_DENOMINATOR_OFFSET,
               header->counter_frequency_denominator);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_ERROR_BOUND_NS_OFFSET, header->calibration_error_bound_ns);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_SPAN_NS_OFFSET, header->calibration_span_ns);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_START_COUNTER_OFFSET, header->start_counter);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_START_MONOTONIC_RAW_NS_OFFSET, header->start_monotonic_raw_ns);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_START_REALTIME_UNIX_NS_OFFSET, header->start_realtime_unix_ns);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_PID_OFFSET, header->pid);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CONFIGURED_THREAD_CAPACITY_OFFSET, header->configured_thread_capacity);
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_RUN_UUID_OFFSET, header->run_uuid, sizeof(header->run_uuid));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_PROCESS_UUID_OFFSET, header->process_uuid, sizeof(header->process_uuid));
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_KIND_OFFSET, header->source_object_kind);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_ALGORITHM_OFFSET, header->source_object_algorithm);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_LENGTH_OFFSET, header->source_object_length);
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_VALUE_OFFSET,
         header->source_object_value,
         sizeof(header->source_object_value));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_PRIMARY_BINARY_SHA256_OFFSET,
         header->primary_binary_sha256,
         sizeof(header->primary_binary_sha256));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_SCHEMA_BUNDLE_DEFINITION_SHA256_OFFSET,
         header->schema_bundle_definition_sha256,
         sizeof(header->schema_bundle_definition_sha256));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_API_CATALOG_DEFINITION_SHA256_OFFSET,
         header->api_catalog_definition_sha256,
         sizeof(header->api_catalog_definition_sha256));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CALLSITE_CATALOG_DEFINITION_SHA256_OFFSET,
         header->callsite_catalog_definition_sha256,
         sizeof(header->callsite_catalog_definition_sha256));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_CONFIGURATION_INSTANCE_SHA256_OFFSET,
         header->configuration_instance_sha256,
         sizeof(header->configuration_instance_sha256));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_PRIMARY_BUILD_ID_SHA256_OFFSET,
         header->primary_build_id_sha256,
         sizeof(header->primary_build_id_sha256));
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_CRC32C_OFFSET,
               crc32c_unchecked(encoded, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_CRC32C_OFFSET));

  memcpy(wire, encoded, sizeof(encoded));
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_opening_header_decode(oai_memprof_container_v1_opening_header_t *header,
                                                                                 const uint8_t *wire,
                                                                                 size_t wire_size)
{
  if (header == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  if (memcmp(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_MAGIC_OFFSET, opening_magic, sizeof(opening_magic)) != 0)
    return OAI_MEMPROF_CONTAINER_V1_BAD_MAGIC;
  if (load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_CRC32C_OFFSET)
      != crc32c_unchecked(wire, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_CRC32C_OFFSET))
    return OAI_MEMPROF_CONTAINER_V1_BAD_CHECKSUM;
  if (load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CONTAINER_MAJOR_OFFSET) != UINT16_C(1)
      || load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CONTAINER_MINOR_OFFSET) != UINT16_C(0)
      || load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_MINIMUM_READER_MINOR_OFFSET) != UINT16_C(0))
    return OAI_MEMPROF_CONTAINER_V1_UNSUPPORTED_VERSION;
  if (load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_BYTES_OFFSET) != OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE
      || load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_EVENT_RECORD_BYTES_OFFSET)
             != OAI_MEMPROF_CONTAINER_V1_EVENT_RECORD_SIZE
      || load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CHUNK_HEADER_BYTES_OFFSET)
             != OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE
      || load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_ENDIAN_MARKER_OFFSET) != OAI_MEMPROF_CONTAINER_V1_ENDIAN_MARKER
      || wire[OAI_MEMPROF_CONTAINER_V1_OPENING_POINTER_WIDTH_BYTES_OFFSET] != UINT8_C(8))
    return OAI_MEMPROF_CONTAINER_V1_INVALID_FIXED_FIELD;
  if (load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_REQUIRED_FEATURES_OFFSET) != OAI_MEMPROF_CONTAINER_V1_REQUIRED_FEATURES)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_FIXED_FIELD;
  if (!bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_RESERVED_ZERO_0_OFFSET, 2)
      || !bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_RESERVED_ZERO_1_OFFSET, 2)
      || !bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_RESERVED_ZERO_2_OFFSET, 132))
    return OAI_MEMPROF_CONTAINER_V1_NONZERO_RESERVED;

  oai_memprof_container_v1_opening_header_t decoded = {0};
  decoded.page_size_bytes = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_PAGE_SIZE_BYTES_OFFSET);
  decoded.scope_kind = wire[OAI_MEMPROF_CONTAINER_V1_OPENING_SCOPE_KIND_OFFSET];
  decoded.role_kind = wire[OAI_MEMPROF_CONTAINER_V1_OPENING_ROLE_KIND_OFFSET];
  decoded.clock_kind = wire[OAI_MEMPROF_CONTAINER_V1_OPENING_CLOCK_KIND_OFFSET];
  decoded.calibration_kind = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_KIND_OFFSET);
  decoded.process_generation = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_PROCESS_GENERATION_OFFSET);
  decoded.counter_frequency_numerator = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_COUNTER_FREQUENCY_NUMERATOR_OFFSET);
  decoded.counter_frequency_denominator = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_COUNTER_FREQUENCY_DENOMINATOR_OFFSET);
  decoded.calibration_error_bound_ns = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_ERROR_BOUND_NS_OFFSET);
  decoded.calibration_span_ns = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CALIBRATION_SPAN_NS_OFFSET);
  decoded.start_counter = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_START_COUNTER_OFFSET);
  decoded.start_monotonic_raw_ns = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_START_MONOTONIC_RAW_NS_OFFSET);
  decoded.start_realtime_unix_ns = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_START_REALTIME_UNIX_NS_OFFSET);
  decoded.pid = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_PID_OFFSET);
  decoded.configured_thread_capacity = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CONFIGURED_THREAD_CAPACITY_OFFSET);
  memcpy(decoded.run_uuid, wire + OAI_MEMPROF_CONTAINER_V1_OPENING_RUN_UUID_OFFSET, sizeof(decoded.run_uuid));
  memcpy(decoded.process_uuid, wire + OAI_MEMPROF_CONTAINER_V1_OPENING_PROCESS_UUID_OFFSET, sizeof(decoded.process_uuid));
  decoded.source_object_kind = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_KIND_OFFSET);
  decoded.source_object_algorithm = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_ALGORITHM_OFFSET);
  decoded.source_object_length = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_LENGTH_OFFSET);
  memcpy(decoded.source_object_value,
         wire + OAI_MEMPROF_CONTAINER_V1_OPENING_SOURCE_OBJECT_VALUE_OFFSET,
         sizeof(decoded.source_object_value));
  memcpy(decoded.primary_binary_sha256,
         wire + OAI_MEMPROF_CONTAINER_V1_OPENING_PRIMARY_BINARY_SHA256_OFFSET,
         sizeof(decoded.primary_binary_sha256));
  memcpy(decoded.schema_bundle_definition_sha256,
         wire + OAI_MEMPROF_CONTAINER_V1_OPENING_SCHEMA_BUNDLE_DEFINITION_SHA256_OFFSET,
         sizeof(decoded.schema_bundle_definition_sha256));
  memcpy(decoded.api_catalog_definition_sha256,
         wire + OAI_MEMPROF_CONTAINER_V1_OPENING_API_CATALOG_DEFINITION_SHA256_OFFSET,
         sizeof(decoded.api_catalog_definition_sha256));
  memcpy(decoded.callsite_catalog_definition_sha256,
         wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CALLSITE_CATALOG_DEFINITION_SHA256_OFFSET,
         sizeof(decoded.callsite_catalog_definition_sha256));
  memcpy(decoded.configuration_instance_sha256,
         wire + OAI_MEMPROF_CONTAINER_V1_OPENING_CONFIGURATION_INSTANCE_SHA256_OFFSET,
         sizeof(decoded.configuration_instance_sha256));
  memcpy(decoded.primary_build_id_sha256,
         wire + OAI_MEMPROF_CONTAINER_V1_OPENING_PRIMARY_BUILD_ID_SHA256_OFFSET,
         sizeof(decoded.primary_build_id_sha256));

  const oai_memprof_container_v1_status_t validation = validate_opening_header(&decoded);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;

  *header = decoded;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static oai_memprof_container_v1_status_t chunk_payload_bytes(uint32_t record_count, size_t *payload_bytes)
{
  if (record_count == 0 || record_count > OAI_MEMPROF_CONTAINER_V1_MAX_CHUNK_RECORD_COUNT)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;

  const uint64_t bytes = (uint64_t)record_count * OAI_MEMPROF_CONTAINER_V1_EVENT_RECORD_SIZE;
  if (bytes > UINT32_MAX || bytes > SIZE_MAX)
    return OAI_MEMPROF_CONTAINER_V1_INTEGER_OVERFLOW;
  *payload_bytes = (size_t)bytes;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_chunk_header_encode(
    const oai_memprof_container_v1_chunk_header_t *header,
    const uint8_t *payload,
    size_t payload_size,
    uint8_t *wire,
    size_t wire_size)
{
  if (header == NULL || payload == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;

  size_t expected_payload_size = 0;
  const oai_memprof_container_v1_status_t relation = chunk_payload_bytes(header->record_count, &expected_payload_size);
  if (relation != OAI_MEMPROF_CONTAINER_V1_OK)
    return relation;
  if (payload_size != expected_payload_size)
    return OAI_MEMPROF_CONTAINER_V1_PAYLOAD_SIZE_MISMATCH;

  uint8_t encoded[OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE] = {0};
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_CHUNK_MAGIC_OFFSET, chunk_magic, sizeof(chunk_magic));
  encoded[OAI_MEMPROF_CONTAINER_V1_CHUNK_MAJOR_OFFSET] = UINT8_C(1);
  encoded[OAI_MEMPROF_CONTAINER_V1_CHUNK_MINOR_OFFSET] = UINT8_C(0);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_BYTES_OFFSET, OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_CHUNK_SEQUENCE_OFFSET, header->writer_chunk_sequence);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_CHUNK_RECORD_COUNT_OFFSET, header->record_count);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_CHUNK_PAYLOAD_BYTES_OFFSET, (uint32_t)expected_payload_size);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_CHUNK_PAYLOAD_CRC32C_OFFSET, crc32c_unchecked(payload, expected_payload_size));
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_CHUNK_FLAGS_OFFSET, UINT32_C(0));

  memcpy(wire, encoded, sizeof(encoded));
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_chunk_header_decode(oai_memprof_container_v1_chunk_header_t *header,
                                                                               const uint8_t *wire,
                                                                               size_t wire_size,
                                                                               const uint8_t *payload,
                                                                               size_t payload_size)
{
  if (header == NULL || wire == NULL || payload == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  if (memcmp(wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_MAGIC_OFFSET, chunk_magic, sizeof(chunk_magic)) != 0)
    return OAI_MEMPROF_CONTAINER_V1_BAD_MAGIC;
  if (wire[OAI_MEMPROF_CONTAINER_V1_CHUNK_MAJOR_OFFSET] != UINT8_C(1)
      || wire[OAI_MEMPROF_CONTAINER_V1_CHUNK_MINOR_OFFSET] != UINT8_C(0))
    return OAI_MEMPROF_CONTAINER_V1_UNSUPPORTED_VERSION;
  if (load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_BYTES_OFFSET) != OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_FIXED_FIELD;
  if (load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_FLAGS_OFFSET) != 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_FIXED_FIELD;

  const uint32_t record_count = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_RECORD_COUNT_OFFSET);
  size_t expected_payload_size = 0;
  const oai_memprof_container_v1_status_t relation = chunk_payload_bytes(record_count, &expected_payload_size);
  if (relation != OAI_MEMPROF_CONTAINER_V1_OK)
    return relation;
  if (load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_PAYLOAD_BYTES_OFFSET) != expected_payload_size
      || payload_size != expected_payload_size)
    return OAI_MEMPROF_CONTAINER_V1_PAYLOAD_SIZE_MISMATCH;
  if (load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_PAYLOAD_CRC32C_OFFSET) != crc32c_unchecked(payload, expected_payload_size))
    return OAI_MEMPROF_CONTAINER_V1_BAD_CHECKSUM;

  const oai_memprof_container_v1_chunk_header_t decoded = {
      .writer_chunk_sequence = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_CHUNK_SEQUENCE_OFFSET),
      .record_count = record_count,
  };
  *header = decoded;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static oai_memprof_container_v1_status_t compute_trailer_layout(uint32_t event_entry_count,
                                                                uint32_t diagnostic_entry_count,
                                                                uint32_t object_entry_count,
                                                                uint64_t *diagnostic_table_offset,
                                                                uint64_t *object_table_offset,
                                                                uint64_t *trailer_body_bytes)
{
  if (event_entry_count > OAI_MEMPROF_CONTAINER_V1_MAX_EVENT_ENTRIES
      || diagnostic_entry_count > OAI_MEMPROF_CONTAINER_V1_MAX_DIAGNOSTIC_ENTRIES
      || object_entry_count > OAI_MEMPROF_CONTAINER_V1_MAX_OBJECT_ENTRIES)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;

  uint64_t event_bytes = 0;
  uint64_t diagnostic_bytes = 0;
  uint64_t object_bytes = 0;
  if (!multiply_u64(event_entry_count, OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE, &event_bytes)
      || !multiply_u64(diagnostic_entry_count, OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE, &diagnostic_bytes)
      || !multiply_u64(object_entry_count, OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE, &object_bytes))
    return OAI_MEMPROF_CONTAINER_V1_INTEGER_OVERFLOW;
  if (!add_u64(OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE, event_bytes, diagnostic_table_offset)
      || !add_u64(*diagnostic_table_offset, diagnostic_bytes, object_table_offset)
      || !add_u64(*object_table_offset, object_bytes, trailer_body_bytes))
    return OAI_MEMPROF_CONTAINER_V1_INTEGER_OVERFLOW;
  if (*trailer_body_bytes > OAI_MEMPROF_CONTAINER_V1_MAX_TRAILER_BODY_BYTES)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static oai_memprof_container_v1_status_t compute_prefix_bytes(uint64_t chunk_count,
                                                              uint64_t record_count,
                                                              uint64_t *prefix_bytes,
                                                              uint64_t *payload_bytes)
{
  if ((chunk_count == 0) != (record_count == 0) || chunk_count > record_count)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  uint64_t chunk_header_bytes = 0;
  if (!multiply_u64(record_count, OAI_MEMPROF_CONTAINER_V1_EVENT_RECORD_SIZE, payload_bytes)
      || !multiply_u64(chunk_count, OAI_MEMPROF_CONTAINER_V1_CHUNK_HEADER_SIZE, &chunk_header_bytes))
    return OAI_MEMPROF_CONTAINER_V1_INTEGER_OVERFLOW;
  uint64_t after_headers = 0;
  if (!add_u64(OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE, chunk_header_bytes, &after_headers)
      || !add_u64(after_headers, *payload_bytes, prefix_bytes))
    return OAI_MEMPROF_CONTAINER_V1_INTEGER_OVERFLOW;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static bool valid_terminal_lifecycle(uint16_t lifecycle_state)
{
  return lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_COMPLETE
         || lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED
         || lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_INCOMPLETE;
}

static bool valid_payload_writer_state(uint16_t payload_writer_state)
{
  return payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED
         || payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_IO_FAILED_AT_SAFE_BOUNDARY_AND_CLOSED_VERIFIED;
}

static bool valid_primary_outcome(const oai_memprof_container_v1_trailer_header_t *header)
{
  switch (header->terminal_reason_code) {
    case OAI_MEMPROF_CONTAINER_V1_REASON_NONE:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_COMPLETE
             && header->finalization_stage == OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRE_SYNC_TERMINAL_MATERIAL_FROZEN
             && header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED;
    case OAI_MEMPROF_CONTAINER_V1_REASON_QUIESCENCE_TIMEOUT:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_INCOMPLETE
             && header->finalization_stage == OAI_MEMPROF_CONTAINER_V1_FINALIZATION_ADMISSION_SEALED
             && header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED;
    case OAI_MEMPROF_CONTAINER_V1_REASON_RING_DRAIN_FAILED:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED
             && header->finalization_stage == OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRODUCERS_QUIESCED
             && header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED;
    case OAI_MEMPROF_CONTAINER_V1_REASON_CATALOG_FREEZE_FAILED:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED
             && header->finalization_stage == OAI_MEMPROF_CONTAINER_V1_FINALIZATION_RINGS_DRAINED_AND_CALLSITES_INTERNED
             && header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED;
    case OAI_MEMPROF_CONTAINER_V1_REASON_DIAGNOSTICS_FREEZE_FAILED:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED
             && header->finalization_stage == OAI_MEMPROF_CONTAINER_V1_FINALIZATION_CATALOGS_FROZEN
             && header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED;
    case OAI_MEMPROF_CONTAINER_V1_REASON_PAYLOAD_SYNC_FAILED_AT_SAFE_BOUNDARY:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED
             && header->finalization_stage == OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRE_SYNC_TERMINAL_MATERIAL_FROZEN
             && header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_IO_FAILED_AT_SAFE_BOUNDARY_AND_CLOSED_VERIFIED;
    case OAI_MEMPROF_CONTAINER_V1_REASON_PAYLOAD_IO_FAILED_AT_SAFE_BOUNDARY:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_FAILED
             && header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_IO_FAILED_AT_SAFE_BOUNDARY_AND_CLOSED_VERIFIED;
    case OAI_MEMPROF_CONTAINER_V1_REASON_COUNTER_OR_TIME_INVALID:
    case OAI_MEMPROF_CONTAINER_V1_REASON_OPERATOR_CANCELLED:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_INCOMPLETE
             && (header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED
                 || header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_IO_FAILED_AT_SAFE_BOUNDARY_AND_CLOSED_VERIFIED);
    case OAI_MEMPROF_CONTAINER_V1_REASON_UNSUPPORTED_SCOPE:
      return header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_INCOMPLETE
             && header->finalization_stage == OAI_MEMPROF_CONTAINER_V1_FINALIZATION_ACTIVE_ONLY
             && header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED;
    default:
      return false;
  }
}

static oai_memprof_container_v1_status_t validate_trailer_header(const oai_memprof_container_v1_trailer_header_t *header)
{
  if (header->scope_kind != OAI_MEMPROF_CONTAINER_V1_SCOPE_MEASUREMENT_INTERVAL
      && header->scope_kind != OAI_MEMPROF_CONTAINER_V1_SCOPE_PROCESS_LIFETIME)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;
  if (!valid_terminal_lifecycle(header->lifecycle_state) || !valid_payload_writer_state(header->payload_writer_state)
      || header->finalization_stage > OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRE_SYNC_TERMINAL_MATERIAL_FROZEN
      || header->terminal_reason_code > OAI_MEMPROF_CONTAINER_V1_REASON_UNSUPPORTED_SCOPE)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;
  if (header->process_generation == 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  if ((header->terminal_flags & ~OAI_MEMPROF_CONTAINER_V1_TERMINAL_FLAGS_MASK) != 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;

  uint64_t diagnostic_table_offset = 0;
  uint64_t object_table_offset = 0;
  uint64_t trailer_body_bytes = 0;
  const oai_memprof_container_v1_status_t table_layout = compute_trailer_layout(header->event_entry_count,
                                                                                header->diagnostic_entry_count,
                                                                                header->object_entry_count,
                                                                                &diagnostic_table_offset,
                                                                                &object_table_offset,
                                                                                &trailer_body_bytes);
  if (table_layout != OAI_MEMPROF_CONTAINER_V1_OK)
    return table_layout;
  if (header->event_table_offset != OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE
      || header->diagnostic_table_offset != diagnostic_table_offset || header->object_table_offset != object_table_offset
      || header->trailer_body_bytes != trailer_body_bytes)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  uint64_t prefix_bytes = 0;
  uint64_t payload_bytes = 0;
  const oai_memprof_container_v1_status_t prefix_layout =
      compute_prefix_bytes(header->chunk_count, header->record_count, &prefix_bytes, &payload_bytes);
  if (prefix_layout != OAI_MEMPROF_CONTAINER_V1_OK)
    return prefix_layout;
  if (header->payload_bytes != payload_bytes || header->first_chunk_offset != OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE
      || header->chunks_end_offset != prefix_bytes)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  if ((header->event_entry_count == 0) != (header->record_count == 0)
      || (header->record_count != 0 && header->event_entry_count > header->record_count))
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  if ((header->terminal_flags & UINT64_C(1)) == 0 || header->active_generation == 0
      || header->active_generation != header->process_generation)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  const uint64_t primary_stage_mask = (UINT64_C(1) << (header->finalization_stage + UINT16_C(1))) - UINT64_C(1);
  if ((header->terminal_flags & UINT64_C(0x7f)) != primary_stage_mask)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  if (header->payload_writer_state == OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED) {
    if ((header->terminal_flags & (UINT64_C(1) << 7)) == 0 || (header->terminal_flags & (UINT64_C(1) << 8)) == 0)
      return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  } else if ((header->terminal_flags & (UINT64_C(1) << 8)) == 0 || (header->terminal_flags & (UINT64_C(1) << 7)) != 0) {
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  }

  if (!valid_primary_outcome(header))
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  if (header->active_start_counter > header->final_counter
      || header->active_start_monotonic_raw_ns > header->final_monotonic_raw_ns)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  if (header->finalization_stage < OAI_MEMPROF_CONTAINER_V1_FINALIZATION_ADMISSION_SEALED) {
    if (header->cutoff_before_counter != 0 || header->cutoff_after_counter != 0)
      return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  } else if (header->active_start_counter > header->cutoff_before_counter
             || header->cutoff_before_counter > header->cutoff_after_counter
             || header->cutoff_after_counter > header->final_counter) {
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  }
  if (header->finalization_stage < OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRODUCERS_QUIESCED) {
    if (header->quiescence_complete_counter != 0)
      return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  } else if (header->cutoff_after_counter > header->quiescence_complete_counter
             || header->quiescence_complete_counter > header->final_counter) {
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  }

  if (header->lifecycle_state == OAI_MEMPROF_CONTAINER_V1_LIFECYCLE_COMPLETE) {
    if (header->scope_kind != OAI_MEMPROF_CONTAINER_V1_SCOPE_MEASUREMENT_INTERVAL || header->terminal_reason_code != 0
        || header->payload_writer_state != OAI_MEMPROF_CONTAINER_V1_WRITER_PAYLOAD_CLOSED_VERIFIED
        || header->finalization_stage != OAI_MEMPROF_CONTAINER_V1_FINALIZATION_PRE_SYNC_TERMINAL_MATERIAL_FROZEN
        || (header->terminal_flags & OAI_MEMPROF_CONTAINER_V1_COMPLETE_REQUIRED_FLAGS)
               != OAI_MEMPROF_CONTAINER_V1_COMPLETE_REQUIRED_FLAGS
        || (header->terminal_flags & (UINT64_C(1) << 13)) != 0 || (header->terminal_flags & (UINT64_C(1) << 16)) != 0)
      return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  }

  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_trailer_header_encode(
    const oai_memprof_container_v1_trailer_header_t *header,
    uint8_t *wire,
    size_t wire_size)
{
  if (header == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  const oai_memprof_container_v1_status_t validation = validate_trailer_header(header);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;

  uint8_t encoded[OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE] = {0};
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_MAGIC_OFFSET, trailer_magic, sizeof(trailer_magic));
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_SCHEMA_MAJOR_OFFSET, UINT16_C(1));
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_SCHEMA_MINOR_OFFSET, UINT16_C(0));
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_FIXED_HEADER_BYTES_OFFSET, OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_BODY_BYTES_OFFSET, header->trailer_body_bytes);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_PROCESS_GENERATION_OFFSET, header->process_generation);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_SCOPE_KIND_OFFSET, header->scope_kind);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_LIFECYCLE_STATE_OFFSET, header->lifecycle_state);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_PAYLOAD_WRITER_STATE_OFFSET, header->payload_writer_state);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_FINALIZATION_STAGE_OFFSET, header->finalization_stage);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_TERMINAL_FLAGS_OFFSET, header->terminal_flags);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_CHUNK_COUNT_OFFSET, header->chunk_count);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_RECORD_COUNT_OFFSET, header->record_count);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_PAYLOAD_BYTES_OFFSET, header->payload_bytes);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_FIRST_CHUNK_OFFSET, header->first_chunk_offset);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_CHUNKS_END_OFFSET, header->chunks_end_offset);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_GENERATION_OFFSET, header->active_generation);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_START_COUNTER_OFFSET, header->active_start_counter);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_CUTOFF_BEFORE_COUNTER_OFFSET, header->cutoff_before_counter);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_CUTOFF_AFTER_COUNTER_OFFSET, header->cutoff_after_counter);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_QUIESCENCE_COMPLETE_COUNTER_OFFSET, header->quiescence_complete_counter);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_COUNTER_OFFSET, header->final_counter);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_START_MONOTONIC_RAW_NS_OFFSET,
               header->active_start_monotonic_raw_ns);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_MONOTONIC_RAW_NS_OFFSET, header->final_monotonic_raw_ns);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_REALTIME_UNIX_NS_OFFSET, header->final_realtime_unix_ns);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_ENTRY_COUNT_OFFSET, header->event_entry_count);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_ENTRY_BYTES_OFFSET,
               OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_ENTRY_COUNT_OFFSET, header->diagnostic_entry_count);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_ENTRY_BYTES_OFFSET,
               OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_ENTRY_COUNT_OFFSET, header->object_entry_count);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_ENTRY_BYTES_OFFSET, OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_TABLE_OFFSET_OFFSET, header->event_table_offset);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_TABLE_OFFSET_OFFSET, header->diagnostic_table_offset);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_TABLE_OFFSET_OFFSET, header->object_table_offset);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_TERMINAL_REASON_CODE_OFFSET, header->terminal_reason_code);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_LOSS_SUM_OFFSET, header->diagnostic_loss_sum);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_BYPASS_SUM_OFFSET, header->diagnostic_bypass_sum);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_TRAILER_SATURATED_COUNTER_INSTANCES_OFFSET, header->saturated_counter_instances);

  memcpy(wire, encoded, sizeof(encoded));
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_trailer_header_decode(oai_memprof_container_v1_trailer_header_t *header,
                                                                                 const uint8_t *wire,
                                                                                 size_t wire_size)
{
  if (header == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  if (memcmp(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_MAGIC_OFFSET, trailer_magic, sizeof(trailer_magic)) != 0)
    return OAI_MEMPROF_CONTAINER_V1_BAD_MAGIC;
  if (load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_SCHEMA_MAJOR_OFFSET) != UINT16_C(1)
      || load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_SCHEMA_MINOR_OFFSET) != UINT16_C(0))
    return OAI_MEMPROF_CONTAINER_V1_UNSUPPORTED_VERSION;
  if (load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_FIXED_HEADER_BYTES_OFFSET) != OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE
      || load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_ENTRY_BYTES_OFFSET)
             != OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE
      || load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_ENTRY_BYTES_OFFSET)
             != OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE
      || load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_ENTRY_BYTES_OFFSET)
             != OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_FIXED_FIELD;
  if (!bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_RESERVED_ZERO_0_OFFSET, 4)
      || !bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_RESERVED_ZERO_1_OFFSET, 8))
    return OAI_MEMPROF_CONTAINER_V1_NONZERO_RESERVED;

  const oai_memprof_container_v1_trailer_header_t decoded = {
      .trailer_body_bytes = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_BODY_BYTES_OFFSET),
      .process_generation = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_PROCESS_GENERATION_OFFSET),
      .scope_kind = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_SCOPE_KIND_OFFSET),
      .lifecycle_state = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_LIFECYCLE_STATE_OFFSET),
      .payload_writer_state = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_PAYLOAD_WRITER_STATE_OFFSET),
      .finalization_stage = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_FINALIZATION_STAGE_OFFSET),
      .terminal_flags = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_TERMINAL_FLAGS_OFFSET),
      .chunk_count = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_CHUNK_COUNT_OFFSET),
      .record_count = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_RECORD_COUNT_OFFSET),
      .payload_bytes = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_PAYLOAD_BYTES_OFFSET),
      .first_chunk_offset = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_FIRST_CHUNK_OFFSET),
      .chunks_end_offset = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_CHUNKS_END_OFFSET),
      .active_generation = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_GENERATION_OFFSET),
      .active_start_counter = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_START_COUNTER_OFFSET),
      .cutoff_before_counter = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_CUTOFF_BEFORE_COUNTER_OFFSET),
      .cutoff_after_counter = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_CUTOFF_AFTER_COUNTER_OFFSET),
      .quiescence_complete_counter = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_QUIESCENCE_COMPLETE_COUNTER_OFFSET),
      .final_counter = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_COUNTER_OFFSET),
      .active_start_monotonic_raw_ns = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_ACTIVE_START_MONOTONIC_RAW_NS_OFFSET),
      .final_monotonic_raw_ns = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_MONOTONIC_RAW_NS_OFFSET),
      .final_realtime_unix_ns = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_FINAL_REALTIME_UNIX_NS_OFFSET),
      .event_entry_count = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_ENTRY_COUNT_OFFSET),
      .diagnostic_entry_count = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_ENTRY_COUNT_OFFSET),
      .object_entry_count = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_ENTRY_COUNT_OFFSET),
      .event_table_offset = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_EVENT_TABLE_OFFSET_OFFSET),
      .diagnostic_table_offset = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_TABLE_OFFSET_OFFSET),
      .object_table_offset = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_OBJECT_TABLE_OFFSET_OFFSET),
      .terminal_reason_code = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_TERMINAL_REASON_CODE_OFFSET),
      .diagnostic_loss_sum = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_LOSS_SUM_OFFSET),
      .diagnostic_bypass_sum = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_DIAGNOSTIC_BYPASS_SUM_OFFSET),
      .saturated_counter_instances = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_TRAILER_SATURATED_COUNTER_INSTANCES_OFFSET),
  };

  const oai_memprof_container_v1_status_t validation = validate_trailer_header(&decoded);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;
  *header = decoded;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static oai_memprof_container_v1_status_t validate_event_total_entry(const oai_memprof_container_v1_event_total_entry_t *entry)
{
  if (entry->record_count == 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_event_total_entry_encode(
    const oai_memprof_container_v1_event_total_entry_t *entry,
    uint8_t *wire,
    size_t wire_size)
{
  if (entry == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  const oai_memprof_container_v1_status_t validation = validate_event_total_entry(entry);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;

  uint8_t encoded[OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE] = {0};
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_EVENT_KIND_OFFSET, entry->event_kind);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_API_ID_OFFSET, entry->api_id);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_RECORD_COUNT_OFFSET, entry->record_count);
  memcpy(wire, encoded, sizeof(encoded));
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_event_total_entry_decode(
    oai_memprof_container_v1_event_total_entry_t *entry,
    const uint8_t *wire,
    size_t wire_size)
{
  if (entry == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_ENTRY_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  if (!bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_RESERVED_ZERO_0_OFFSET, 4)
      || !bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_RESERVED_ZERO_1_OFFSET, 16))
    return OAI_MEMPROF_CONTAINER_V1_NONZERO_RESERVED;

  const oai_memprof_container_v1_event_total_entry_t decoded = {
      .event_kind = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_EVENT_KIND_OFFSET),
      .api_id = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_API_ID_OFFSET),
      .record_count = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_EVENT_TOTAL_RECORD_COUNT_OFFSET),
  };
  const oai_memprof_container_v1_status_t validation = validate_event_total_entry(&decoded);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;
  *entry = decoded;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static oai_memprof_container_v1_status_t validate_diagnostic_total_entry(
    const oai_memprof_container_v1_diagnostic_total_entry_t *entry)
{
  if ((entry->class_flags & (uint16_t)~OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_CLASS_FLAGS_MASK) != 0
      || (entry->summary_flags & ~OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_SUMMARY_FLAGS_MASK) != 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  if (entry->saturated_counter_instances > entry->nonzero_counter_instances)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  if (entry->saturated_counter_instances != 0 && (entry->summary_flags & UINT32_C(1)) == 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_diagnostic_total_entry_encode(
    const oai_memprof_container_v1_diagnostic_total_entry_t *entry,
    uint8_t *wire,
    size_t wire_size)
{
  if (entry == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  const oai_memprof_container_v1_status_t validation = validate_diagnostic_total_entry(entry);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;

  uint8_t encoded[OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE] = {0};
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_REASON_ID_OFFSET, entry->reason_id);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_CLASS_FLAGS_OFFSET, entry->class_flags);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SUMMARY_FLAGS_OFFSET, entry->summary_flags);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SATURATING_TOTAL_OFFSET, entry->saturating_total);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_NONZERO_COUNTER_INSTANCES_OFFSET,
               entry->nonzero_counter_instances);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SATURATED_COUNTER_INSTANCES_OFFSET,
               entry->saturated_counter_instances);
  memcpy(wire, encoded, sizeof(encoded));
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_diagnostic_total_entry_decode(
    oai_memprof_container_v1_diagnostic_total_entry_t *entry,
    const uint8_t *wire,
    size_t wire_size)
{
  if (entry == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_ENTRY_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  if (!bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_RESERVED_ZERO_OFFSET, 8))
    return OAI_MEMPROF_CONTAINER_V1_NONZERO_RESERVED;

  const oai_memprof_container_v1_diagnostic_total_entry_t decoded = {
      .reason_id = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_REASON_ID_OFFSET),
      .class_flags = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_CLASS_FLAGS_OFFSET),
      .summary_flags = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SUMMARY_FLAGS_OFFSET),
      .saturating_total = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SATURATING_TOTAL_OFFSET),
      .nonzero_counter_instances = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_NONZERO_COUNTER_INSTANCES_OFFSET),
      .saturated_counter_instances =
          load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_DIAGNOSTIC_TOTAL_SATURATED_COUNTER_INSTANCES_OFFSET),
  };
  const oai_memprof_container_v1_status_t validation = validate_diagnostic_total_entry(&decoded);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;
  *entry = decoded;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static oai_memprof_container_v1_status_t validate_object_entry(const oai_memprof_container_v1_object_entry_t *entry)
{
  static const uint32_t required_flags[12] = {
      UINT32_C(0x05),
      UINT32_C(0x05),
      UINT32_C(0x0b),
      UINT32_C(0x1b),
      UINT32_C(0x1b),
      UINT32_C(0x13),
      UINT32_C(0x03),
      UINT32_C(0x03),
      UINT32_C(0x13),
      UINT32_C(0x07),
      UINT32_C(0x03),
      UINT32_C(0x03),
  };

  if (entry->object_kind < UINT16_C(1) || entry->object_kind > UINT16_C(12) || entry->format_id != UINT16_C(1)
      || entry->schema_revision != UINT32_C(1))
    return OAI_MEMPROF_CONTAINER_V1_INVALID_ENUM;
  if ((entry->object_flags & ~OAI_MEMPROF_CONTAINER_V1_OBJECT_FLAGS_MASK) != 0
      || entry->object_flags != required_flags[entry->object_kind - UINT16_C(1)]
      || entry->entry_count > OAI_MEMPROF_CONTAINER_V1_MAX_OBJECT_ENTRY_COUNT
      || entry->byte_count > OAI_MEMPROF_CONTAINER_V1_MAX_OBJECT_BYTE_COUNT)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  switch (entry->object_kind) {
    case UINT16_C(1):
      if (entry->entry_count < UINT64_C(7))
        return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
      break;
    case UINT16_C(2):
      if (entry->entry_count != UINT64_C(12))
        return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
      break;
    case UINT16_C(6):
    case UINT16_C(7):
    case UINT16_C(8):
    case UINT16_C(9):
    case UINT16_C(11):
      if (entry->entry_count == 0)
        return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
      break;
    case UINT16_C(10):
    case UINT16_C(12):
      if (entry->entry_count != UINT64_C(1))
        return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
      break;
    default:
      break;
  }
  if (!bytes_are_nonzero(entry->sha256, sizeof(entry->sha256)))
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_object_entry_encode(const oai_memprof_container_v1_object_entry_t *entry,
                                                                               uint8_t *wire,
                                                                               size_t wire_size)
{
  if (entry == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  const oai_memprof_container_v1_status_t validation = validate_object_entry(entry);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;

  uint8_t encoded[OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE] = {0};
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OBJECT_KIND_OFFSET, entry->object_kind);
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_OBJECT_FORMAT_ID_OFFSET, entry->format_id);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_OBJECT_FLAGS_OFFSET, entry->object_flags);
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_OBJECT_SCHEMA_REVISION_OFFSET, entry->schema_revision);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_COUNT_OFFSET, entry->entry_count);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_OBJECT_BYTE_COUNT_OFFSET, entry->byte_count);
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_OBJECT_SHA256_OFFSET, entry->sha256, sizeof(entry->sha256));
  memcpy(wire, encoded, sizeof(encoded));
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_object_entry_decode(oai_memprof_container_v1_object_entry_t *entry,
                                                                               const uint8_t *wire,
                                                                               size_t wire_size)
{
  if (entry == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  if (!bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_OBJECT_RESERVED_ZERO_OFFSET, 4))
    return OAI_MEMPROF_CONTAINER_V1_NONZERO_RESERVED;

  oai_memprof_container_v1_object_entry_t decoded = {
      .object_kind = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OBJECT_KIND_OFFSET),
      .format_id = load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_OBJECT_FORMAT_ID_OFFSET),
      .object_flags = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_OBJECT_FLAGS_OFFSET),
      .schema_revision = load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_OBJECT_SCHEMA_REVISION_OFFSET),
      .entry_count = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OBJECT_ENTRY_COUNT_OFFSET),
      .byte_count = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_OBJECT_BYTE_COUNT_OFFSET),
  };
  memcpy(decoded.sha256, wire + OAI_MEMPROF_CONTAINER_V1_OBJECT_SHA256_OFFSET, sizeof(decoded.sha256));
  const oai_memprof_container_v1_status_t validation = validate_object_entry(&decoded);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;
  *entry = decoded;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

static oai_memprof_container_v1_status_t validate_footer(const oai_memprof_container_v1_footer_t *footer)
{
  if (footer->trailer_body_bytes < OAI_MEMPROF_CONTAINER_V1_TRAILER_HEADER_SIZE
      || footer->trailer_body_bytes > OAI_MEMPROF_CONTAINER_V1_MAX_TRAILER_BODY_BYTES
      || footer->trailer_body_bytes % UINT64_C(32) != 0)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_VALUE;

  uint64_t expected_prefix_bytes = 0;
  uint64_t payload_bytes = 0;
  const oai_memprof_container_v1_status_t prefix_layout =
      compute_prefix_bytes(footer->chunk_count, footer->record_count, &expected_prefix_bytes, &payload_bytes);
  (void)payload_bytes;
  if (prefix_layout != OAI_MEMPROF_CONTAINER_V1_OK)
    return prefix_layout;
  if (footer->trailer_offset != expected_prefix_bytes || footer->prefix_bytes != footer->trailer_offset)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;

  uint64_t stream_without_footer = 0;
  uint64_t expected_stream_bytes = 0;
  if (!add_u64(footer->trailer_offset, footer->trailer_body_bytes, &stream_without_footer)
      || !add_u64(stream_without_footer, OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE, &expected_stream_bytes))
    return OAI_MEMPROF_CONTAINER_V1_INTEGER_OVERFLOW;
  if (footer->stream_bytes != expected_stream_bytes)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_RELATION;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_footer_encode(const oai_memprof_container_v1_footer_t *footer,
                                                                         uint8_t *wire,
                                                                         size_t wire_size)
{
  if (footer == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  const oai_memprof_container_v1_status_t validation = validate_footer(footer);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;

  uint8_t encoded[OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE] = {0};
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_MAGIC_OFFSET, footer_magic, sizeof(footer_magic));
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_SCHEMA_MAJOR_OFFSET, UINT16_C(1));
  store_u16_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_SCHEMA_MINOR_OFFSET, UINT16_C(0));
  store_u32_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_BYTES_OFFSET, OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_FLAGS_OFFSET, OAI_MEMPROF_CONTAINER_V1_FOOTER_FLAGS);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_OFFSET_OFFSET, footer->trailer_offset);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_BODY_BYTES_OFFSET, footer->trailer_body_bytes);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_STREAM_BYTES_OFFSET, footer->stream_bytes);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_PREFIX_BYTES_OFFSET, footer->prefix_bytes);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_HEADER_BYTES_OFFSET, OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_CHUNK_COUNT_OFFSET, footer->chunk_count);
  store_u64_le(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_RECORD_COUNT_OFFSET, footer->record_count);
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_PREFIX_SHA256_OFFSET, footer->prefix_sha256, sizeof(footer->prefix_sha256));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_BODY_SHA256_OFFSET,
         footer->trailer_body_sha256,
         sizeof(footer->trailer_body_sha256));
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_OPENING_HEADER_SHA256_OFFSET,
         footer->opening_header_sha256,
         sizeof(footer->opening_header_sha256));
  uint8_t calculated_footer_sha256[32];
  sha256_footer_prefix(encoded, calculated_footer_sha256);
  if (bytes_are_nonzero(footer->footer_sha256, sizeof(footer->footer_sha256))
      && memcmp(footer->footer_sha256, calculated_footer_sha256, sizeof(calculated_footer_sha256)) != 0)
    return OAI_MEMPROF_CONTAINER_V1_BAD_CHECKSUM;
  memcpy(encoded + OAI_MEMPROF_CONTAINER_V1_FOOTER_FOOTER_SHA256_OFFSET,
         calculated_footer_sha256,
         sizeof(calculated_footer_sha256));
  memcpy(wire, encoded, sizeof(encoded));
  return OAI_MEMPROF_CONTAINER_V1_OK;
}

oai_memprof_container_v1_status_t oai_memprof_container_v1_footer_decode(oai_memprof_container_v1_footer_t *footer,
                                                                         const uint8_t *wire,
                                                                         size_t wire_size)
{
  if (footer == NULL || wire == NULL)
    return OAI_MEMPROF_CONTAINER_V1_NULL_ARGUMENT;
  if (wire_size != OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_WRONG_SIZE;
  if (memcmp(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_MAGIC_OFFSET, footer_magic, sizeof(footer_magic)) != 0)
    return OAI_MEMPROF_CONTAINER_V1_BAD_MAGIC;
  uint8_t calculated_footer_sha256[32];
  sha256_footer_prefix(wire, calculated_footer_sha256);
  if (memcmp(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_FOOTER_SHA256_OFFSET,
             calculated_footer_sha256,
             sizeof(calculated_footer_sha256))
      != 0)
    return OAI_MEMPROF_CONTAINER_V1_BAD_CHECKSUM;
  if (load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_SCHEMA_MAJOR_OFFSET) != UINT16_C(1)
      || load_u16_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_SCHEMA_MINOR_OFFSET) != UINT16_C(0))
    return OAI_MEMPROF_CONTAINER_V1_UNSUPPORTED_VERSION;
  if (load_u32_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_BYTES_OFFSET) != OAI_MEMPROF_CONTAINER_V1_FOOTER_SIZE
      || load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_FLAGS_OFFSET) != OAI_MEMPROF_CONTAINER_V1_FOOTER_FLAGS
      || load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_HEADER_BYTES_OFFSET) != OAI_MEMPROF_CONTAINER_V1_OPENING_HEADER_SIZE)
    return OAI_MEMPROF_CONTAINER_V1_INVALID_FIXED_FIELD;
  if (!bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_RESERVED_ZERO_0_OFFSET, 8)
      || !bytes_are_zero(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_RESERVED_ZERO_1_OFFSET, 32))
    return OAI_MEMPROF_CONTAINER_V1_NONZERO_RESERVED;

  oai_memprof_container_v1_footer_t decoded = {
      .trailer_offset = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_OFFSET_OFFSET),
      .trailer_body_bytes = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_BODY_BYTES_OFFSET),
      .stream_bytes = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_STREAM_BYTES_OFFSET),
      .prefix_bytes = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_PREFIX_BYTES_OFFSET),
      .chunk_count = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_CHUNK_COUNT_OFFSET),
      .record_count = load_u64_le(wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_RECORD_COUNT_OFFSET),
  };
  memcpy(decoded.prefix_sha256, wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_PREFIX_SHA256_OFFSET, sizeof(decoded.prefix_sha256));
  memcpy(decoded.trailer_body_sha256,
         wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_TRAILER_BODY_SHA256_OFFSET,
         sizeof(decoded.trailer_body_sha256));
  memcpy(decoded.opening_header_sha256,
         wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_OPENING_HEADER_SHA256_OFFSET,
         sizeof(decoded.opening_header_sha256));
  memcpy(decoded.footer_sha256, wire + OAI_MEMPROF_CONTAINER_V1_FOOTER_FOOTER_SHA256_OFFSET, sizeof(decoded.footer_sha256));

  const oai_memprof_container_v1_status_t validation = validate_footer(&decoded);
  if (validation != OAI_MEMPROF_CONTAINER_V1_OK)
    return validation;
  *footer = decoded;
  return OAI_MEMPROF_CONTAINER_V1_OK;
}
