/*
 * Minimal JSON-based injection for EnergyHO prototype
 * ---------------------------------------------------
 * This file implements a deliberately limited JSON key/value extractor to read
 * overrides from a simple schema without adding dependencies. It supports:
 *   - String keys and values (no escapes), numbers, booleans true/false
 *   - Arrays of objects under keys: "cells" and "overrides"
 *
 * Expected schema example:
 * {
 *   "cells": [
 *     {"nr_cellid": 12345678, "is_active": true, "switch_off_prob": 1.0},
 *     {"nr_cellid": 87654321, "is_active": true}
 *   ],
 *   "overrides": [
 *     {"ue_rnti":"0x46A", "serv_cellid":12345678, "targ_cellid":87654321,
 *      "serv": {"rb_usage_ue": 12.0},
 *      "targ": {"sinr_db": 7.0, "delay_ms": 15.0, "rb_usage_ue": 10.0},
 *      "ue":   {"delay_req_ms": 25.0, "bitrate_mbps": 5.0}
 *     }
 *   ]
 * }
 *
 * Limitations:
 *  - Not a general JSON parser; malformed JSON is ignored silently.
 *  - No handling of escape sequences in strings.
 *  - Matching is exact for serv/targ cell IDs and allows ue_rnti as hex (0x....),
 *    decimal, or the string "any" to match any UE.
 */

#include "rrc_gNB_energy_ho_inject.h"

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>  // strtod

// Very small helpers to find key in a JSON-like text. This is not a full JSON parser
// but adequate for our controlled testing schema.

/*
 * find_key: locate the position just after ':' for a JSON key "key"
 * This is a naive search adequate for the constrained schema in testing.
 */
static const char *find_key(const char *buf, const char *key)
{
  // search for "key" (with quotes) followed by ':'
  static char pat[128];
  size_t klen = strlen(key);
  if (klen + 4 >= sizeof(pat)) return NULL;
  pat[0] = '"'; memcpy(pat+1, key, klen); pat[1+klen] = '"'; pat[1+klen+1] = '\0';
  const char *p = buf;
  while ((p = strstr(p, pat))) {
    p += (1 + klen + 1);
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == ':') return p + 1;
  }
  return NULL;
}

static const char *skip_ws(const char *p){ while (p && *p && isspace((unsigned char)*p)) p++; return p; }

/* parse_string_value: read a double-quoted string into out (no escapes) */
static bool parse_string_value(const char *p, char *out, size_t outlen)
{
  p = skip_ws(p);
  if (!p || *p != '"') return false;
  p++;
  size_t i = 0;
  while (*p && *p != '"' && i + 1 < outlen) { out[i++] = *p++; }
  if (*p != '"') return false;
  out[i] = '\0';
  return true;
}

/* parse_number_value: parse a double from the current position */
static bool parse_number_value(const char *p, double *out)
{
  p = skip_ws(p);
  char *end = NULL;
  double v = strtod(p, &end);
  if (end == p) return false;
  *out = v;
  return true;
}

/* parse_bool_value: parse true/false */
static bool parse_bool_value(const char *p, int *out)
{
  p = skip_ws(p);
  if (!strncmp(p, "true", 4)) { *out = 1; return true; }
  if (!strncmp(p, "false", 5)) { *out = 0; return true; }
  return false;
}

/* match_hex_or_any: compare string s with rnti; supports 0xHEX, decimal, or "any" */
static bool match_hex_or_any(const char *s, uint16_t rnti)
{
  if (!s || !*s) return false;
  if (!strcmp(s, "any")) return true;
  unsigned int v = 0;
  if (sscanf(s, "0x%x", &v) == 1) return ((uint16_t)v) == rnti;
  if (sscanf(s, "%u", &v) == 1) return ((uint16_t)v) == rnti;
  return false;
}

bool energy_ho_try_override(const char *json_path,
                            uint16_t ue_rnti,
                            uint32_t serv_cellid,
                            uint32_t targ_cellid,
                            energy_cell_t *serv,
                            energy_cell_t *targ,
                            energy_ue_req_t *ue)
{
  FILE *f = fopen(json_path, "rb");
  if (!f) return false;
  char buf[32768];
  size_t n = fread(buf, 1, sizeof(buf)-1, f);
  fclose(f);
  buf[n] = '\0';

  bool applied = false;

  // 1) Apply cell-level overrides from "cells": array
  const char *cells_p = find_key(buf, "cells");
  if (cells_p) {
    const char *p = strchr(cells_p, '[');
    const char *e = p ? strchr(p, ']') : NULL;
    if (p && e && e > p) {
      const char *cur = p;
      while ((cur = strchr(cur, '{')) && cur < e) {
        const char *obj_end = strchr(cur, '}'); if (!obj_end || obj_end > e) break;
        const char *c_id_p = find_key(cur, "nr_cellid");
        double cid = 0;
        if (c_id_p && parse_number_value(c_id_p, &cid)) {
          uint32_t this_cid = (uint32_t)(cid + 0.5);
          energy_cell_t *dst = NULL;
          if (serv && this_cid == serv_cellid) dst = serv;
          else if (targ && this_cid == targ_cellid) dst = targ;
          if (dst) {
            const char *ia_p = find_key(cur, "is_active");
            if (ia_p) { int b=0; if (parse_bool_value(ia_p, &b)) { dst->is_active = b?true:false; applied = true; } }
            const char *sop_p = find_key(cur, "switch_off_prob");
            if (sop_p) { double v=0; if (parse_number_value(sop_p, &v)) { dst->switch_off_prob = (float)v; applied = true; } }
          }
        }
        cur = obj_end + 1;
      }
    }
  }

  // 2) Apply per-UE overrides from "overrides": array
  const char *ov_p = find_key(buf, "overrides");
  if (ov_p) {
    const char *p = strchr(ov_p, '[');
    const char *e = p ? strchr(p, ']') : NULL;
    if (p && e && e > p) {
      const char *cur = p;
      while ((cur = strchr(cur, '{')) && cur < e) {
        const char *obj_end = strchr(cur, '}'); if (!obj_end || obj_end > e) break;
        char rnti_s[32]={0};
        const char *rnti_p = find_key(cur, "ue_rnti");
        if (rnti_p) parse_string_value(rnti_p, rnti_s, sizeof(rnti_s));

        const char *sc_p = find_key(cur, "serv_cellid");
        const char *tc_p = find_key(cur, "targ_cellid");
        double sc=0, tc=0; parse_number_value(sc_p, &sc); parse_number_value(tc_p, &tc);

        if ((match_hex_or_any(rnti_s, ue_rnti)) && ((uint32_t)(sc+0.5) == serv_cellid) && ((uint32_t)(tc+0.5) == targ_cellid)) {
          // nested objects serv / targ
          const char *serv_obj = find_key(cur, "serv");
          if (serv_obj && serv) {
            const char *rb_p = find_key(serv_obj, "rb_usage_ue");
            double v=0; if (rb_p && parse_number_value(rb_p, &v)) { serv->rb_usage_ue = (float)v; applied = true; }
          }
          const char *targ_obj = find_key(cur, "targ");
          if (targ_obj && targ) {
            const char *rb_p = find_key(targ_obj, "rb_usage_ue");
            double v=0; if (rb_p && parse_number_value(rb_p, &v)) { targ->rb_usage_ue = (float)v; applied = true; }
            const char *sn_p = find_key(targ_obj, "sinr_db");
            if (sn_p && parse_number_value(sn_p, &v)) { targ->sinr_db = (float)v; applied = true; }
            const char *dl_p = find_key(targ_obj, "delay_ms");
            if (dl_p && parse_number_value(dl_p, &v)) { targ->delay_ms = (float)v; applied = true; }
          }
          // per-UE delay requirement optional
          if (ue) {
            const char *ue_obj = find_key(cur, "ue");
            if (ue_obj) {
              const char *dr_p = find_key(ue_obj, "delay_req_ms");
              double v=0; if (dr_p && parse_number_value(dr_p, &v)) { ue->delay_req_ms = (float)v; applied = true; }
              const char *br_p = find_key(ue_obj, "bitrate_mbps");
              if (br_p && parse_number_value(br_p, &v)) { ue->bitrate_mbps = (float)v; applied = true; }
            }
          }
        }
        cur = obj_end + 1;
      }
    }
  }

  return applied;
}
