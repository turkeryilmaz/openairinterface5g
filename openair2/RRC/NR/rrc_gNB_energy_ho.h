/*
 * Energy-Efficiency-Driven Handover (EnergyHO)
 * --------------------------------------------
 * This header defines a compact, dependency-free C interface for an energy-aware
 * handover (HO) decision module used by the NR gNB RRC (CU side). The evaluator
 * computes a net access-network power differential ΔP between serving and target
 * cells and applies hysteresis and quality constraints (SINR, delay) to decide
 * whether an A3-triggered HO should proceed.
 *
 * Design notes (high level):
 *  - The module is pure C and adds no external dependencies.
 *  - All power quantities are expressed in milliwatts (mW) to avoid floating
 *    point underflow; delays are in milliseconds (ms); SINR in decibels (dB).
 *  - The API is intentionally tiny to reduce friction when wiring into RRC.
 *  - A global configuration struct (energy_ho_cfg_t) is used; getters/setters
 *    return/modify the in-process singleton. This is sufficient because RRC
 *    configuration is typically performed at startup.
 *  - Live metrics (SINR, delay, RB usage) can be measured or injected; the
 *    injection path is for RFsim prototyping and can be disabled in production.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>

/*
 * Per-cell descriptor (serving or target).
 *
 * Units and semantics:
 *  - cell_id:    NR cell identity (gNB RRC uses 64-bit nr_cellid; we keep lower 32 bits here).
 *  - n_trx:      Active TX chains used by the cell (e.g., 4 for 4T4R). Affects dynamic power.
 *  - p0_mw:      Optional static baseline per TRX chain (mW). Not required by the evaluator,
 *                but kept for completeness.
 *  - p_idle_mw:  Per-cell idle baseline when the cell is in an active/idle but not sleeping state.
 *                Used when modeling activation/switch-off effects (boolean flags below).
 *  - delta_p:    Dimensionless slope multiplying load-dependent power term.
 *  - p_rb_mw:    Effective power per RB (mW) used to model dynamic power ∝ N_RB.
 *  - is_active:  true if the target cell is already active (so no activation cost applies).
 *  - is_ntn:     Mark for logging/policy differentiation; not used by evaluator.
 *  - rb_usage_avg: optional average RBs across UEs (for future extensions).
 *  - rb_usage_ue: average RBs attributed to the specific UE under consideration.
 *  - sinr_db:    target-cell SINR (dB) for the UE; required for thresholding. For serving cell,
 *                this field is ignored by the evaluator.
 *  - delay_ms:   expected one-way access delay on target cell for the UE; can be injected.
 *  - switch_off_prob: likelihood that serving can switch off after HO (0..1). Evaluator treats
 *                it as a boolean by comparing to switch_off_prob_threshold.
 *  - connected_ues: instantaneous count of UEs in the cell, used to decide switch_off possibility.
 */
typedef struct {
  uint32_t cell_id;
  uint8_t  n_trx;
  float    p0_mw;
  float    p_idle_mw;
  float    delta_p;
  float    p_rb_mw;
  bool     is_active;
  bool     is_ntn;

  float    rb_usage_avg;
  float    rb_usage_ue;
  float    sinr_db;
  float    delay_ms;
  float    switch_off_prob;

  uint32_t connected_ues;
} energy_cell_t;

/*
 * Per-UE requirements/context used by the evaluator.
 *  - ue_id:        RRC UE index (for logging only).
 *  - rnti:         C-RNTI, used for per-UE stickiness counters.
 *  - delay_req_ms: maximum tolerated access delay for the UE.
 *  - bitrate_mbps: optional expected throughput; used to estimate RB demand on target when
 *                  rb_usage_ue is not explicitly provided.
 *  - last_ho_ms:   timestamp of previous HO for guard-time enforcement (0 if unknown/not used).
 */
typedef struct {
  uint32_t ue_id;
  uint16_t rnti;
  float    delay_req_ms;
  float    bitrate_mbps;
  uint64_t last_ho_ms;
} energy_ue_req_t;

/*
 * Optional debug structure returning the decision breakdown for logging/testing.
 */
typedef struct {
  float deltaP_mw;        // ΔP = (serv_idle+serv_dyn) - (targ_idle+targ_dyn)
  float deltaP_serv_mw;   // serv_idle+serv_dyn
  float deltaP_targ_mw;   // targ_idle+targ_dyn
  bool  p_switch_off_serv;
  bool  p_activation_targ;
  bool  sinr_ok;
  bool  delay_ok;
  bool  hysteresis_ok;
  bool  energy_gain_ok;
} energy_ho_dbg_t;

/*
 * Global configuration (singleton), typically populated from the CU config file.
 * Thread-safety: set once during initialization, then read-only.
 */
typedef struct {
  int    enable;                 // 0 = disabled (bypass gating), 1 = enabled
  int    simple_mode;            // 1 = use evaluate_simple_energy_ho
  float  hysteresis_margin_mw;   // ΔP margin
  float  sinr_threshold_db;      // minimum target SINR
  float  switch_off_prob_threshold;
  float  default_ue_delay_req_ms;
  uint32_t t_win_ms;             // RB EWMA window (future use)
  uint32_t guard_ms;             // guard time between HOs per UE
  uint8_t  k_consecutive;        // consecutive passes required
  int    debug_logs;             // verbose decision logs

  // Per-cell power defaults (can be overridden per cell in future)
  uint8_t  default_n_trx;
  float    default_p_idle_mw;
  float    default_delta_p;
  float    default_p_rb_mw;

  // Injection controls
  int      injection_enable;     // 1 enables JSON injection
  char     injection_json_path[256];
  uint32_t injection_poll_period_ms; // not used in blocking read; kept for future timer
} energy_ho_cfg_t;

/* API: configuration accessors */
const energy_ho_cfg_t *energy_ho_get_cfg(void);
void energy_ho_set_cfg(const energy_ho_cfg_t *cfg);

/* Evaluators: return true if HO should be triggered */
bool energy_ho_evaluate(const energy_cell_t *serv, const energy_cell_t *targ,
                        const energy_ue_req_t *ue, energy_ho_dbg_t *dbg);

bool evaluate_simple_energy_ho(const energy_cell_t *serv, const energy_cell_t *targ,
                               const energy_ue_req_t *ue, energy_ho_dbg_t *dbg);

/* Helpers */
uint64_t energy_ho_time_ms(void);
bool energy_ho_metrics_ready(const energy_cell_t *serv, const energy_cell_t *targ,
                             const energy_ue_req_t *ue);
