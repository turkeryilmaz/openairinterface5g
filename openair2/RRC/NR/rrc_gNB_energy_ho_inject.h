/*
 * Minimal JSON-based injection for EnergyHO prototype (CU side only)
 * ------------------------------------------------------------------
 * This header exposes a single helper to apply runtime overrides for
 * target SINR, expected delay, UE-attributed RBs, and cell activity/switch-off
 * flags, using a JSON file polled by the CU RRC. The parser is intentionally
 * tiny and schema-specific to avoid pulling in external JSON dependencies.
 *
 * Usage:
 *   energy_ho_try_override("/tmp/oai_energy_ho.json", rnti, serv_cellid, targ_cellid,
 *                           &serv, &targ, &ue);
 * If a matching override is found, supplied fields overwrite the given structs.
 *
 * Notes:
 *  - The parser is not a full JSON implementation; it tolerates whitespace and
 *    simple string/number/boolean values but does not implement escape sequences.
 *  - This facility is intended for RFsim prototyping only.
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "rrc_gNB_energy_ho.h"

// Attempts to read the injection JSON file and apply overrides for the given UE/serv/target.
// Returns true if any override was applied.
bool energy_ho_try_override(const char *json_path,
                            uint16_t ue_rnti,
                            uint32_t serv_cellid,
                            uint32_t targ_cellid,
                            energy_cell_t *serv,
                            energy_cell_t *targ,
                            energy_ue_req_t *ue);
