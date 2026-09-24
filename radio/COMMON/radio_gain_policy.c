/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "radio_gain_policy.h"
#include <math.h>

bool radio_rx_policy_config_valid(const radio_rx_policy_config_t *c)
{
  return c && isfinite(c->target_dbfs) && c->target_dbfs < 0 && isfinite(c->deadband_db) && c->deadband_db > 0
         && isfinite(c->maximum_step_db) && c->maximum_step_db > 0 && isfinite(c->search_step_db) && c->search_step_db > 0
         && c->search_step_db <= c->maximum_step_db && isfinite(c->peak_ceiling_dbfs) && c->peak_ceiling_dbfs <= 0
         && c->peak_ceiling_dbfs > c->target_dbfs && isfinite(c->peak_release_db_per_second) && c->peak_release_db_per_second > 0
         && isfinite(c->near_rail_fraction) && c->near_rail_fraction > 0 && c->near_rail_fraction <= 1 && isfinite(c->filter_weight)
         && c->filter_weight > 0 && c->filter_weight <= 1 && c->minimum_interval_ns > 0 && c->maximum_age_ns > 0;
}

radio_rx_decision_t radio_rx_decide(const radio_rx_policy_config_t *c,
                                    radio_rx_policy_state_t *state,
                                    radio_rx_peak_envelope_t *peak_envelope,
                                    const radio_gain_channel_t *channel,
                                    const radio_rx_observation_t *o)
{
  radio_rx_decision_t decision = {.reason = RADIO_RX_HOLD_INVALID};
  if (!radio_rx_policy_config_valid(c) || !state || !peak_envelope || !channel || !o)
    return decision;
  decision.gain_db = o->reported_gain_db;
  if (!o->gain_valid || !o->power_valid || !isfinite(o->reported_gain_db) || !isfinite(o->mean_power_dbfs)
      || !isfinite(o->peak_component_dbfs) || !isfinite(channel->minimum_db) || !isfinite(channel->maximum_db)
      || !isfinite(channel->step_db) || channel->minimum_db > channel->maximum_db || channel->step_db < 0
      || o->reported_gain_db < channel->minimum_db || o->reported_gain_db > channel->maximum_db || o->sampled_components == 0
      || o->near_rail_components > o->sampled_components)
    return decision;
  if (channel->step_db > c->maximum_step_db) {
    decision.reason = RADIO_RX_HOLD_UNSUPPORTED;
    return decision;
  }
  if (o->observation_ns > o->now_ns || o->now_ns - o->observation_ns > c->maximum_age_ns
      || (state->action_recorded && o->now_ns < state->last_action_ns)) {
    decision.reason = RADIO_RX_HOLD_STALE;
    return decision;
  }
  if (o->generation < state->generation) {
    decision.reason = RADIO_RX_HOLD_STALE;
    return decision;
  }
  if (o->generation > state->generation) {
    state->generation = o->generation;
    state->filter_valid = false;
    state->last_observation_valid = false;
  }
  if (state->last_observation_valid && o->observation_ns <= state->last_observation_ns) {
    decision.reason = RADIO_RX_HOLD_STALE;
    return decision;
  }
  state->last_observation_ns = o->observation_ns;
  state->last_observation_valid = true;
  if (!o->settled) {
    state->filter_valid = false;
    decision.reason = RADIO_RX_HOLD_TRANSITION;
    return decision;
  }
  if (peak_envelope->valid && o->now_ns < peak_envelope->last_update_ns) {
    decision.reason = RADIO_RX_HOLD_STALE;
    return decision;
  }

  const double observed_input_peak_dbfs = o->peak_component_dbfs - o->reported_gain_db;
  if (!isfinite(observed_input_peak_dbfs))
    return decision;
  if (!peak_envelope->valid) {
    peak_envelope->input_peak_dbfs = observed_input_peak_dbfs;
    peak_envelope->last_update_ns = o->now_ns;
    peak_envelope->valid = true;
  } else {
    const double elapsed_seconds = (double)(o->now_ns - peak_envelope->last_update_ns) / 1000000000.0;
    const double released_input_peak_dbfs = peak_envelope->input_peak_dbfs - c->peak_release_db_per_second * elapsed_seconds;
    if (!isfinite(released_input_peak_dbfs))
      return decision;
    peak_envelope->input_peak_dbfs = fmax(released_input_peak_dbfs, observed_input_peak_dbfs);
    peak_envelope->last_update_ns = o->now_ns;
  }
  decision.peak_bound_dbfs = peak_envelope->input_peak_dbfs + o->reported_gain_db;
  decision.peak_bound_valid = isfinite(decision.peak_bound_dbfs);
  if (!decision.peak_bound_valid)
    return decision;

  /* Reduce on observed headroom loss even when the serving signal cannot be
   * decoded. Absence of activity, however, never causes a gain increase. */
  const bool overloaded = o->peak_component_dbfs >= c->peak_ceiling_dbfs
                          || (double)o->near_rail_components / o->sampled_components >= c->near_rail_fraction;
  /* A positive step must leave a deadband below the overload threshold.
   * Otherwise variable/sparsely sampled peaks can make level tracking raise
   * gain immediately after the headroom policy lowered it. */
  const double increase_headroom_db = fmax(0, c->peak_ceiling_dbfs - c->deadband_db - decision.peak_bound_dbfs);
  double change = 0;
  if (overloaded) {
    decision.reason = RADIO_RX_REDUCE_OVERLOAD;
    change = -c->maximum_step_db;
  } else if (o->search_failed) {
    decision.reason = RADIO_RX_SEARCH_STEP;
    change = fmin(c->search_step_db, increase_headroom_db);
  } else if (o->activity_valid) {
    const double linear = pow(10.0, o->mean_power_dbfs / 10.0);
    if (!isfinite(linear) || linear <= 0)
      return decision;
    state->filtered_linear_power =
        state->filter_valid ? (1 - c->filter_weight) * state->filtered_linear_power + c->filter_weight * linear : linear;
    state->filter_valid = true;
    decision.error_db = c->target_dbfs - 10.0 * log10(state->filtered_linear_power);
    if (fabs(decision.error_db) <= c->deadband_db) {
      decision.reason = RADIO_RX_HOLD_DEADBAND;
      return decision;
    }
    decision.reason = RADIO_RX_TRACK_LEVEL;
    change = fmax(-c->maximum_step_db, fmin(c->maximum_step_db, decision.error_db));
    if (change > 0)
      change = fmin(change, increase_headroom_db);
  } else {
    state->filter_valid = false;
    decision.reason = RADIO_RX_HOLD_INACTIVE;
    return decision;
  }
  if (state->action_recorded && o->now_ns - state->last_action_ns < c->minimum_interval_ns) {
    decision.reason = RADIO_RX_HOLD_COOLDOWN;
    return decision;
  }
  double selected = fmax(channel->minimum_db, fmin(channel->maximum_db, o->reported_gain_db + change));
  if (channel->step_db > 0) {
    /* Quantize toward the current gain to preserve the per-step/headroom bound.
     * Driver readback still decides the actual applied value. */
    const double steps = (selected - channel->minimum_db) / channel->step_db;
    selected = channel->minimum_db + (change > 0 ? floor(steps) : ceil(steps)) * channel->step_db;
    selected = fmax(channel->minimum_db, fmin(channel->maximum_db, selected));
  }
  if ((change > 0 && selected <= o->reported_gain_db) || (change < 0 && selected >= o->reported_gain_db) || change == 0) {
    decision.reason = RADIO_RX_HOLD_LIMIT;
    return decision;
  }
  decision.gain_db = selected;
  decision.change = true;
  return decision;
}

void radio_rx_action_completed(radio_rx_policy_state_t *state, uint64_t generation, uint64_t now_ns)
{
  if (!state || generation < state->generation || (state->last_observation_valid && now_ns < state->last_observation_ns))
    return;
  if (state->action_recorded && (generation < state->last_action_generation || now_ns <= state->last_action_ns))
    return;
  state->last_action_generation = generation;
  state->last_action_ns = now_ns;
  state->action_recorded = true;
}

radio_tx_power_mapping_t radio_tx_map_power(const radio_tx_power_profile_t *p,
                                            double requested_dbm,
                                            double allowed_maximum_dbm,
                                            double unscaled_mean_power_fs,
                                            double unscaled_peak_component_fs,
                                            double peak_limit_fs)
{
  radio_tx_power_mapping_t mapping = {.requested_dbm = requested_dbm};
  if (!p || !p->qualified || !isfinite(p->reference_dbm) || !isfinite(p->uncertainty_db) || p->uncertainty_db < 0
      || !isfinite(p->minimum_dbm) || !isfinite(p->maximum_dbm) || p->minimum_dbm > p->maximum_dbm || !isfinite(requested_dbm)
      || !isfinite(allowed_maximum_dbm) || !isfinite(unscaled_mean_power_fs) || unscaled_mean_power_fs <= 0
      || !isfinite(unscaled_peak_component_fs) || unscaled_peak_component_fs <= 0 || unscaled_peak_component_fs > 32768
      || unscaled_mean_power_fs > 2 * unscaled_peak_component_fs * unscaled_peak_component_fs || !isfinite(peak_limit_fs)
      || peak_limit_fs <= 0 || peak_limit_fs > 1)
    return mapping;
  const double maximum = fmin(allowed_maximum_dbm, p->maximum_dbm);
  /* Below the calibrated range is unsupported; raising the request to a minimum
   * would transmit more power than MAC asked for. Hardware limits and PHR must
   * already agree at the caller, rather than hiding a new physical ceiling. */
  if (maximum < p->minimum_dbm || requested_dbm < p->minimum_dbm || allowed_maximum_dbm > p->maximum_dbm)
    return mapping;
  const double selected = fmin(requested_dbm, maximum);
  const double target_power_fs = pow(10.0, (selected - p->reference_dbm) / 10.0);
  const double scale = sqrt(target_power_fs / unscaled_mean_power_fs);
  const double scaled_peak = scale * unscaled_peak_component_fs;
  if (!isfinite(target_power_fs) || target_power_fs <= 0 || !isfinite(scale) || scale <= 0 || !isfinite(scaled_peak))
    return mapping;
  if (scaled_peak > peak_limit_fs) {
    mapping.peak_limit_exceeded = true;
    return mapping;
  }
  mapping.valid = true;
  mapping.limited = selected != requested_dbm;
  mapping.selected_dbm = selected;
  mapping.amplitude_scale = scale;
  mapping.estimated_dbm = selected;
  mapping.uncertainty_db = p->uncertainty_db;
  return mapping;
}
