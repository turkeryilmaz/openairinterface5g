/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "ntn_assistance.h"

#include <jansson.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

static const unsigned int validity_seconds[] = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120, 180, 240, 900};

unsigned int ntn_assistance_validity_seconds(unsigned int index)
{
  return index < sizeof(validity_seconds) / sizeof(*validity_seconds) ? validity_seconds[index] : 0;
}

static bool fields(const json_t *object, const char *const *names, size_t count)
{
  if (!json_is_object(object) || json_object_size(object) != count)
    return false;
  for (size_t i = 0; i < count; ++i)
    if (!json_object_get(object, names[i]))
      return false;
  return true;
}

static bool string_is(const json_t *value, const char *text)
{
  return json_is_string(value) && json_string_length(value) == strlen(text) && !strcmp(json_string_value(value), text);
}

static bool decimal_u64(const json_t *value, uint64_t *out)
{
  if (!json_is_string(value))
    return false;
  const char *text = json_string_value(value);
  size_t length = json_string_length(value);
  if (!length || length > 20 || (length > 1 && text[0] == '0'))
    return false;
  uint64_t n = 0;
  for (size_t i = 0; i < length; ++i) {
    if (text[i] < '0' || text[i] > '9')
      return false;
    unsigned int digit = text[i] - '0';
    if (n > (UINT64_MAX - digit) / 10)
      return false;
    n = n * 10 + digit;
  }
  *out = n;
  return true;
}

static bool parse_cell(const json_t *object, ntn_assistance_cell_t *cell)
{
  const char *const names[] = {"plmn", "nci"};
  if (!fields(object, names, 2))
    return false;
  const json_t *plmn = json_object_get(object, "plmn");
  const char *text = json_string_value(plmn);
  size_t length = json_string_length(plmn);
  if (!text || (length != 5 && length != 6))
    return false;
  for (size_t i = 0; i < length; ++i)
    if (text[i] < '0' || text[i] > '9')
      return false;
  if (!decimal_u64(json_object_get(object, "nci"), &cell->nci) || cell->nci >= (UINT64_C(1) << 36))
    return false;
  memcpy(cell->plmn, text, length);
  cell->plmn[length] = '\0';
  return true;
}

static bool quantize(const json_t *value, double step, int32_t minimum, int32_t maximum, int32_t *out)
{
  if (!json_is_number(value))
    return false;
  double physical = json_number_value(value);
  if (!isfinite(physical) || physical < minimum * step || physical > maximum * step)
    return false;
  /* Nearest grid point, half-way values away from zero. Bounds are checked
   * before conversion: an out-of-range physical value is never clipped. */
  double rounded = round(physical / step);
  if (rounded < minimum || rounded > maximum)
    return false;
  *out = (int32_t)rounded;
  return true;
}

static bool vector(const json_t *array, double step, int32_t minimum, int32_t maximum, int32_t out[3])
{
  if (!json_is_array(array) || json_array_size(array) != 3)
    return false;
  for (size_t i = 0; i < 3; ++i)
    if (!quantize(json_array_get(array, i), step, minimum, maximum, out + i))
      return false;
  return true;
}

static const char *parse_update(const json_t *root, ntn_assistance_request_t *request)
{
  const char *const names[] =
      {"version", "type", "request", "cell", "session", "sequence", "subject", "epoch", "ephemeris", "ta", "ul_sync_validity_s"};
  if (!fields(root, names, sizeof(names) / sizeof(*names)))
    return "update_fields";
  if (!decimal_u64(json_object_get(root, "sequence"), &request->sequence) || !request->sequence)
    return "sequence";
  const char *const subject_names[] = {"kind"};
  const json_t *subject = json_object_get(root, "subject");
  if (!fields(subject, subject_names, 1) || !string_is(json_object_get(subject, "kind"), "serving"))
    return "unsupported_subject";

  const char *const epoch_names[] = {"kind", "generation", "subframe"};
  const json_t *epoch = json_object_get(root, "epoch");
  if (!fields(epoch, epoch_names, 3) || !string_is(json_object_get(epoch, "kind"), "cell"))
    return "unsupported_epoch";
  if (!decimal_u64(json_object_get(epoch, "generation"), &request->state.epoch.generation) || !request->state.epoch.generation
      || !decimal_u64(json_object_get(epoch, "subframe"), &request->state.epoch.subframe))
    return "epoch";

  const char *const ephemeris_names[] = {"kind", "position_m", "velocity_mps"};
  const json_t *ephemeris = json_object_get(root, "ephemeris");
  if (!fields(ephemeris, ephemeris_names, 3) || !string_is(json_object_get(ephemeris, "kind"), "ecef"))
    return "unsupported_ephemeris";
  if (!vector(json_object_get(ephemeris, "position_m"), 1.3, -33554432, 33554431, request->state.position)
      || !vector(json_object_get(ephemeris, "velocity_mps"), 0.06, -131072, 131071, request->state.velocity))
    return "ephemeris_range";

  const char *const ta_names[] = {"common_us", "drift_us_per_s", "drift_variant_us_per_s2"};
  const json_t *ta = json_object_get(root, "ta");
  if (!fields(ta, ta_names, 3) || !quantize(json_object_get(ta, "common_us"), 0.004072, 0, 66485757, &request->state.ta_common)
      || !quantize(json_object_get(ta, "drift_us_per_s"), 0.0002, -257303, 257303, &request->state.ta_drift)
      || !quantize(json_object_get(ta, "drift_variant_us_per_s2"), 0.00002, 0, 28949, &request->state.ta_drift_variant))
    return "ta_range";

  const json_t *validity = json_object_get(root, "ul_sync_validity_s");
  if (json_is_integer(validity)) {
    for (size_t i = 0; i < sizeof(validity_seconds) / sizeof(*validity_seconds); ++i) {
      if (json_integer_value(validity) == validity_seconds[i]) {
        request->state.validity_index = i;
        return NULL;
      }
    }
  }
  return "validity";
}

const char *ntn_assistance_parse(const void *data, size_t length, ntn_assistance_request_t *out)
{
  if (!data || !out || !length || length > NTN_ASSISTANCE_MAX_DATAGRAM)
    return "datagram_size";
  json_error_t error;
  json_t *root = json_loadb(data, length, JSON_REJECT_DUPLICATES, &error);
  if (!root)
    return "json";
  ntn_assistance_request_t request = {0};
  const char *failure = "header";
  const json_t *version = json_object_get(root, "version");
  if (!json_is_integer(version) || json_integer_value(version) != NTN_ASSISTANCE_VERSION) {
    failure = "version";
    goto done;
  }
  if (!decimal_u64(json_object_get(root, "request"), &request.request) || !parse_cell(json_object_get(root, "cell"), &request.cell))
    goto done;

  const json_t *type = json_object_get(root, "type");
  if (string_is(type, "hello")) {
    const char *const names[] = {"version", "type", "request", "cell"};
    request.kind = NTN_ASSISTANCE_HELLO;
    failure = fields(root, names, 4) ? NULL : "hello_fields";
    goto done;
  }
  if (!decimal_u64(json_object_get(root, "session"), &request.session) || !request.session) {
    failure = "session";
    goto done;
  }
  if (string_is(type, "update")) {
    request.kind = NTN_ASSISTANCE_UPDATE;
    failure = parse_update(root, &request);
  } else if (string_is(type, "time") || string_is(type, "status")) {
    const char *const names[] = {"version", "type", "request", "cell", "session"};
    request.kind = string_is(type, "time") ? NTN_ASSISTANCE_TIME : NTN_ASSISTANCE_STATUS;
    failure = fields(root, names, 5) ? NULL : "request_fields";
  } else {
    failure = "type";
  }
done:
  json_decref(root);
  if (!failure)
    *out = request;
  return failure;
}
