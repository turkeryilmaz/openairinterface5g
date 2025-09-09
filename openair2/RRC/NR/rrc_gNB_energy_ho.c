/*
 * Energy-Efficiency-Driven Handover (EnergyHO) evaluator implementation (C only)
 * ----------------------------------------------------------------------------
 * This file implements the core decision logic used to gate A3-triggered handovers
 * based on an energy-efficiency criterion and quality constraints. The evaluator
 * follows the model specified by the user:
 *   p = N_TRX*p0 + N_TRX*δp*p_out, with p_out = P_RB*N_RB
 *   ΔP = (P_idle_serv * p_switch_off_serv + N_TRXserv*δp_serv*P_RBserv*N_RBserv)
 *        - (P_idle_targ * p_activation_targ + N_TRXtarg*δp_targ*P_RBtarg*N_RBtarg)
 *
 * The evaluator returns true only if:
 *   - ΔP > 0 and ΔP ≥ hysteresis_margin_mw (energy gain + hysteresis), and
 *   - SINR_target ≥ sinr_threshold_db, and
 *   - delay_target ≤ UE.delay_req_ms (or default if unset), and
 *   - guard time since last HO has elapsed, and
 *   - the decision persists for k_consecutive evaluations (stickiness).
 *
 * Notes:
 *  - All inputs are assumed to be instantaneous or short-horizon averages. For RBs
 *    we recommend a ~200 ms EWMA but keep the code agnostic; callers can inject.
 *  - We intentionally avoid dependencies (no JSON libs, no STL, etc.).
 *  - The global configuration is a singleton; write at init, then read.
 */

#include "rrc_gNB_energy_ho.h"

#include <math.h>
#include <string.h>
#include <time.h>

#ifndef NAN
#define NAN (0.0/0.0)
#endif

static energy_ho_cfg_t g_cfg;

static void init_defaults_if_needed(void)
{
  static int inited = 0;
  if (inited)
    return;
  memset(&g_cfg, 0, sizeof(g_cfg));
  g_cfg.enable = 1;                 // enabled by default, but only gates when metrics_ready()
  g_cfg.simple_mode = 0;            // use full ΔP evaluator by default
  g_cfg.hysteresis_margin_mw = 1500.0f;
  g_cfg.sinr_threshold_db = -3.0f;
  g_cfg.switch_off_prob_threshold = 0.8f;
  g_cfg.default_ue_delay_req_ms = 25.0f;
  g_cfg.t_win_ms = 200;
  g_cfg.guard_ms = 2000;
  g_cfg.k_consecutive = 1;          // keep simple for now; can be raised to 3 later
  g_cfg.debug_logs = 1;

  g_cfg.default_n_trx = 4;
  g_cfg.default_p_idle_mw = 8000.0f;
  g_cfg.default_delta_p = 4.0f;
  g_cfg.default_p_rb_mw = 50.0f;

  g_cfg.injection_enable = 1;
  strncpy(g_cfg.injection_json_path, "/tmp/oai_energy_ho.json", sizeof(g_cfg.injection_json_path) - 1);
  g_cfg.injection_poll_period_ms = 100;
  inited = 1;
}

const energy_ho_cfg_t *energy_ho_get_cfg(void)
{
  init_defaults_if_needed();
  return &g_cfg;
}

void energy_ho_set_cfg(const energy_ho_cfg_t *cfg)
{
  init_defaults_if_needed();
  if (cfg)
    g_cfg = *cfg;
}

uint64_t energy_ho_time_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)(ts.tv_nsec / 1000000ull);
}

static int isfinitef_safe(float x) { return !(isnan(x) || isinf(x)); }

bool energy_ho_metrics_ready(const energy_cell_t *serv, const energy_cell_t *targ,
                             const energy_ue_req_t *ue)
{
  (void)serv; (void)ue;
  // Require at least SINR and delay for the target cell to be valid
  if (!targ)
    return false;
  if (!isfinitef_safe(targ->sinr_db))
    return false;
  if (!isfinitef_safe(targ->delay_ms))
    return false;
  return true;
}

static float predict_rb_need_from_sinr_and_bitrate(float sinr_db, float bitrate_mbps)
{
  // Very coarse mapping: approximate spectral efficiency vs SINR
  // η [b/s/Hz] piecewise: clamp between 0.2 and 5.0
  float eta;
  if (sinr_db < -5.0f) eta = 0.2f;
  else if (sinr_db < 0.0f) eta = 0.4f;
  else if (sinr_db < 5.0f) eta = 0.9f;
  else if (sinr_db < 10.0f) eta = 2.0f;
  else if (sinr_db < 15.0f) eta = 3.2f;
  else eta = 4.2f;

  // RB throughput rough estimate for 30 kHz SCS, normal CP, 12 subcarriers per RB
  // 1 RB ~ 180 kHz. Assume symbol efficiency ~ 0.85 and 14 symbols/ms => ~ (eta * 180e3 * 0.85) bits/s per layer.
  // For simplicity ignore layers and use conservative factor.
  const float rb_rate_bps = eta * 180000.0f * 0.75f; // conservative
  const float rb_rate_mbps = rb_rate_bps / 1e6f;
  if (rb_rate_mbps <= 1e-3f)
    return 100.0f; // fallback large
  float n_rb = bitrate_mbps / rb_rate_mbps;
  if (n_rb < 1.0f) n_rb = 1.0f;
  return n_rb;
}

// sticky consecutive pass counters (simple fixed-size cache)
typedef struct { uint16_t rnti; uint32_t cell_id; uint8_t pass; uint64_t ts_ms; } stick_entry_t;
static stick_entry_t g_stick[256];

static uint8_t inc_and_get_stick(uint16_t rnti, uint32_t cell_id, uint32_t window_ms)
{
  uint64_t now = energy_ho_time_ms();
  int idx = -1;
  for (int i = 0; i < (int)(sizeof(g_stick)/sizeof(g_stick[0])); ++i) {
    if (g_stick[i].rnti == rnti && g_stick[i].cell_id == cell_id) { idx = i; break; }
    if (idx < 0 && g_stick[i].rnti == 0) idx = i; // free slot
  }
  if (idx < 0) idx = 0;
  if (g_stick[idx].rnti != rnti || g_stick[idx].cell_id != cell_id || (now - g_stick[idx].ts_ms) > window_ms) {
    g_stick[idx].rnti = rnti; g_stick[idx].cell_id = cell_id; g_stick[idx].pass = 0; g_stick[idx].ts_ms = now;
  }
  if (g_stick[idx].pass < 255) g_stick[idx].pass++;
  g_stick[idx].ts_ms = now;
  return g_stick[idx].pass;
}

bool energy_ho_evaluate(const energy_cell_t *serv, const energy_cell_t *targ,
                        const energy_ue_req_t *ue, energy_ho_dbg_t *dbg)
{
  const energy_ho_cfg_t *C = energy_ho_get_cfg();
  energy_ho_dbg_t D = {0};

  // Booleans: switch-off and activation
  bool p_switch_off_serv = false;
  if (serv) {
    if (serv->connected_ues == 1) p_switch_off_serv = true;
    if (serv->switch_off_prob >= C->switch_off_prob_threshold) p_switch_off_serv = true;
  }
  bool p_activation_targ = targ ? !targ->is_active : false;

  // Demand deltas in RBs
  float dRB_serv = (serv && isfinitef_safe(serv->rb_usage_ue) && serv->rb_usage_ue > 0.0f) ? serv->rb_usage_ue : 0.0f;
  float dRB_targ;
  if (targ && isfinitef_safe(targ->rb_usage_ue) && targ->rb_usage_ue > 0.0f) {
    dRB_targ = targ->rb_usage_ue;
  } else {
    float br = (ue && isfinitef_safe(ue->bitrate_mbps) && ue->bitrate_mbps > 0.0f) ? ue->bitrate_mbps : 2.0f; // default 2 Mbps
    float sinr = targ ? targ->sinr_db : 0.0f;
    dRB_targ = predict_rb_need_from_sinr_and_bitrate(sinr, br);
  }

  // ΔP components
  float serv_dyn = (serv ? (serv->n_trx * serv->delta_p * serv->p_rb_mw * dRB_serv) : 0.0f);
  float targ_dyn = (targ ? (targ->n_trx * targ->delta_p * targ->p_rb_mw * dRB_targ) : 0.0f);
  float serv_idle = (serv ? (serv->p_idle_mw * (p_switch_off_serv ? 1.0f : 0.0f)) : 0.0f);
  float targ_idle = (targ ? (targ->p_idle_mw * (p_activation_targ ? 1.0f : 0.0f)) : 0.0f);

  float deltaP_serv = serv_idle + serv_dyn;
  float deltaP_targ = targ_idle + targ_dyn;
  float deltaP = deltaP_serv - deltaP_targ;

  bool energy_ok = (deltaP > 0.0f);
  bool hyst_ok   = (deltaP >= C->hysteresis_margin_mw);
  float sinr_db  = targ ? targ->sinr_db : -1000.0f;
  float dly_ms   = targ ? targ->delay_ms : 1e9f;
  bool sinr_ok   = (sinr_db >= C->sinr_threshold_db);
  float ue_req   = (ue && isfinitef_safe(ue->delay_req_ms) && ue->delay_req_ms > 0.0f) ? ue->delay_req_ms : C->default_ue_delay_req_ms;
  bool delay_ok  = (dly_ms <= ue_req);

  uint64_t now_ms = energy_ho_time_ms();
  uint64_t last   = ue ? ue->last_ho_ms : 0;
  bool guard_ok = (!ue || (now_ms - last >= C->guard_ms));
  uint8_t passes = inc_and_get_stick(ue ? ue->rnti : 0, targ ? targ->cell_id : 0, C->guard_ms);
  bool sticky_ok = (passes >= C->k_consecutive);

  if (dbg) {
    D.deltaP_mw = deltaP; D.deltaP_serv_mw = deltaP_serv; D.deltaP_targ_mw = deltaP_targ;
    D.p_switch_off_serv = p_switch_off_serv; D.p_activation_targ = p_activation_targ;
    D.sinr_ok = sinr_ok; D.delay_ok = delay_ok; D.hysteresis_ok = hyst_ok; D.energy_gain_ok = energy_ok;
    *dbg = D;
  }

  return (energy_ok && hyst_ok && sinr_ok && delay_ok && guard_ok && sticky_ok);
}

bool evaluate_simple_energy_ho(const energy_cell_t *serv, const energy_cell_t *targ,
                               const energy_ue_req_t *ue, energy_ho_dbg_t *dbg)
{
  const energy_ho_cfg_t *C = energy_ho_get_cfg();
  (void)dbg;
  float sop = serv ? serv->switch_off_prob : 0.0f;
  float sinr = targ ? targ->sinr_db : -1000.0f;
  float dly = targ ? targ->delay_ms : 1e9f;
  float req = (ue && isfinitef_safe(ue->delay_req_ms) && ue->delay_req_ms > 0.0f) ? ue->delay_req_ms : C->default_ue_delay_req_ms;
  bool sw_ok = (sop >= C->switch_off_prob_threshold) || (serv && serv->connected_ues == 1);
  bool s_ok  = (sinr >= C->sinr_threshold_db);
  bool d_ok  = (dly <= req);
  return (sw_ok && s_ok && d_ok);
}
