/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "radio_gain_policy.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x)                                              \
  do {                                                        \
    if (!(x)) {                                               \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      return EXIT_FAILURE;                                    \
    }                                                         \
  } while (0)

static radio_rx_observation_t valid_observation(uint64_t generation, uint64_t now_ns)
{
  return (radio_rx_observation_t){.generation = generation,
                                  .observation_ns = now_ns,
                                  .now_ns = now_ns,
                                  .reported_gain_db = 50,
                                  .mean_power_dbfs = -30,
                                  .peak_component_dbfs = -12,
                                  .sampled_components = 1024,
                                  .power_valid = true,
                                  .gain_valid = true,
                                  .settled = true,
                                  .activity_valid = true};
}

int main(void)
{
  const radio_rx_policy_config_t config = {.target_dbfs = -18,
                                           .deadband_db = 3,
                                           .maximum_step_db = 3,
                                           .search_step_db = 3,
                                           .peak_ceiling_dbfs = -3,
                                           .peak_release_db_per_second = 3,
                                           .near_rail_fraction = 0.01,
                                           .filter_weight = 0.25,
                                           .minimum_interval_ns = 100,
                                           .maximum_age_ns = 200};
  radio_gain_channel_t channel = {.minimum_db = 0, .maximum_db = 76, .step_db = 1};
  radio_rx_policy_state_t state = {0};
  radio_rx_peak_envelope_t envelope = {0};
  radio_rx_policy_config_t invalid_config = config;
  invalid_config.peak_release_db_per_second = 0;
  CHECK(!radio_rx_policy_config_valid(&invalid_config));
  invalid_config.peak_release_db_per_second = NAN;
  CHECK(!radio_rx_policy_config_valid(&invalid_config));
  radio_rx_observation_t o = valid_observation(1, 1000);
  o.activity_valid = false;
  CHECK(radio_rx_policy_config_valid(&config));
  radio_rx_decision_t d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_INACTIVE);

  o = valid_observation(1, 1001);
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(d.change && d.gain_db == 53 && d.reason == RADIO_RX_TRACK_LEVEL);
  /* Observe proposals have not consumed the actuation cooldown. */
  o = valid_observation(1, 1002);
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(d.change && d.gain_db == 53);
  radio_rx_action_completed(&state, 1, o.now_ns);

  o = valid_observation(1, 1003);
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_COOLDOWN);

  o = valid_observation(1, 1203);
  o.activity_valid = false;
  o.near_rail_components = 12;
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(d.change && d.reason == RADIO_RX_REDUCE_OVERLOAD && d.gain_db == 47);

  o = valid_observation(1, 1204);
  o.search_failed = true;
  o.peak_component_dbfs = -4.4;
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_LIMIT); // reserve below the overload threshold wins

  o = valid_observation(1, 1205);
  o.search_failed = true;
  o.peak_component_dbfs = -4.4;
  o.reported_gain_db = 76;
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_LIMIT);

  o = valid_observation(1, 1206);
  o.settled = false;
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_TRANSITION && !state.filter_valid);

  o = valid_observation(1, 1407);
  o.observation_ns = 1206;
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_STALE);

  o = valid_observation(2, 1408);
  o.mean_power_dbfs = -18;
  o.peak_component_dbfs = -6;
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_DEADBAND);
  CHECK(fabs(state.filtered_linear_power - pow(10, -1.8)) < 1e-12);

  const uint64_t retained_action_ns = state.last_action_ns;
  radio_rx_action_completed(&state, 1, 1500);
  CHECK(state.last_action_ns == retained_action_ns && state.last_action_generation == 1);
  radio_rx_action_completed(&state, 2, 1407);
  CHECK(state.last_action_ns == retained_action_ns && state.last_action_generation == 1);
  radio_rx_action_completed(&state, 2, 2000);
  CHECK(state.last_action_ns == 2000 && state.last_action_generation == 2);
  radio_rx_action_completed(&state, 3, 1999);
  CHECK(state.last_action_ns == 2000 && state.last_action_generation == 2);

  o = valid_observation(2, 1999);
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_STALE);

  o = valid_observation(1, 2001);
  d = radio_rx_decide(&config, &state, &envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_STALE);

  radio_rx_policy_state_t zero_state = {0};
  radio_rx_peak_envelope_t zero_envelope = {0};
  o = valid_observation(1, 0);
  d = radio_rx_decide(&config, &zero_state, &zero_envelope, &channel, &o);
  CHECK(d.change && zero_state.last_observation_valid);
  d = radio_rx_decide(&config, &zero_state, &zero_envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_STALE);

  radio_rx_policy_state_t invalid_state = {0};
  radio_rx_peak_envelope_t invalid_envelope = {0};
  o = valid_observation(1, 1);
  o.reported_gain_db = 77;
  d = radio_rx_decide(&config, &invalid_state, &invalid_envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_INVALID);

  radio_rx_policy_state_t coarse_state = {0};
  radio_rx_peak_envelope_t coarse_envelope = {0};
  radio_gain_channel_t coarse_channel = {.minimum_db = 0, .maximum_db = 76, .step_db = 5};
  o = valid_observation(1, 1);
  d = radio_rx_decide(&config, &coarse_state, &coarse_envelope, &coarse_channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_UNSUPPORTED);

  /* A high raw peak at gain57 retains input headroom across the physical gain
   * generation and a different source state. A weak PUSCH 200 ms later would
   * otherwise request the full +3 dB tracking step. */
  radio_rx_policy_state_t burst_headroom_state = {0};
  radio_rx_policy_state_t burst_pusch_state = {0};
  radio_rx_peak_envelope_t burst_envelope = {0};
  o = valid_observation(1, UINT64_C(10000000000));
  o.reported_gain_db = 57;
  o.mean_power_dbfs = -15.920;
  o.peak_component_dbfs = -1.7;
  o.activity_valid = false;
  d = radio_rx_decide(&config, &burst_headroom_state, &burst_envelope, &channel, &o);
  CHECK(d.change && d.reason == RADIO_RX_REDUCE_OVERLOAD && d.gain_db == 54 && d.peak_bound_valid);
  CHECK(fabs(d.peak_bound_dbfs + 1.7) < 1e-12);
  radio_rx_action_completed(&burst_headroom_state, 1, o.now_ns);

  o = valid_observation(2, UINT64_C(10200000000));
  o.reported_gain_db = 54;
  o.mean_power_dbfs = -22;
  o.peak_component_dbfs = -26;
  d = radio_rx_decide(&config, &burst_pusch_state, &burst_envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_LIMIT && d.peak_bound_valid);
  CHECK(d.reason != RADIO_RX_REDUCE_OVERLOAD && fabs(d.peak_bound_dbfs + 5.3) < 1e-9);

  o = valid_observation(3, UINT64_C(10400000000));
  o.reported_gain_db = 54;
  o.mean_power_dbfs = -22;
  o.peak_component_dbfs = -26;
  d = radio_rx_decide(&config, &burst_pusch_state, &burst_envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_LIMIT && d.peak_bound_valid);

  /* A sustained fade releases at 3 dB/s and eventually permits only the
   * normal maximum step. The retained peak never manufactures a reduction. */
  o = valid_observation(4, UINT64_C(11500000000));
  o.reported_gain_db = 54;
  o.mean_power_dbfs = -22;
  o.peak_component_dbfs = -26;
  d = radio_rx_decide(&config, &burst_pusch_state, &burst_envelope, &channel, &o);
  CHECK(d.change && d.reason == RADIO_RX_TRACK_LEVEL && d.gain_db == 57 && d.peak_bound_valid);

  o = valid_observation(4, UINT64_C(11500000001));
  o.reported_gain_db = 54;
  o.mean_power_dbfs = -22;
  o.peak_component_dbfs = -1.7;
  o.activity_valid = false;
  d = radio_rx_decide(&config, &burst_headroom_state, &burst_envelope, &channel, &o);
  CHECK(d.change && d.reason == RADIO_RX_REDUCE_OVERLOAD && d.gain_db == 51 && d.peak_bound_valid);

  /* Rejected, stale, transition, and backward-clock observations cannot alter
   * the shared envelope. Use a fresh source state to exercise its clock guard. */
  radio_rx_policy_state_t poison_state = {0};
  radio_rx_policy_state_t backward_state = {0};
  radio_rx_peak_envelope_t poison_envelope = {0};
  o = valid_observation(1, UINT64_C(20000000000));
  o.reported_gain_db = 57;
  o.peak_component_dbfs = -1.7;
  o.activity_valid = false;
  d = radio_rx_decide(&config, &poison_state, &poison_envelope, &channel, &o);
  CHECK(d.change && poison_envelope.valid);
  const double retained_input_peak_dbfs = poison_envelope.input_peak_dbfs;
  const uint64_t retained_update_ns = poison_envelope.last_update_ns;

  o = valid_observation(1, retained_update_ns + 1);
  o.gain_valid = false;
  d = radio_rx_decide(&config, &poison_state, &poison_envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_INVALID && poison_envelope.input_peak_dbfs == retained_input_peak_dbfs
        && poison_envelope.last_update_ns == retained_update_ns);

  o = valid_observation(1, retained_update_ns + 2);
  o.observation_ns = retained_update_ns;
  d = radio_rx_decide(&config, &poison_state, &poison_envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_STALE && poison_envelope.input_peak_dbfs == retained_input_peak_dbfs
        && poison_envelope.last_update_ns == retained_update_ns);

  o = valid_observation(1, retained_update_ns + 3);
  o.settled = false;
  d = radio_rx_decide(&config, &poison_state, &poison_envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_TRANSITION && poison_envelope.input_peak_dbfs == retained_input_peak_dbfs
        && poison_envelope.last_update_ns == retained_update_ns);

  o = valid_observation(1, retained_update_ns - 1);
  d = radio_rx_decide(&config, &backward_state, &poison_envelope, &channel, &o);
  CHECK(!d.change && d.reason == RADIO_RX_HOLD_STALE && poison_envelope.input_peak_dbfs == retained_input_peak_dbfs
        && poison_envelope.last_update_ns == retained_update_ns);

  radio_tx_power_profile_t profile = {.qualified = true,
                                      .reference_dbm = 20,
                                      .uncertainty_db = 0.7,
                                      .minimum_dbm = -40,
                                      .maximum_dbm = 15};
  radio_tx_power_mapping_t a = radio_tx_map_power(&profile, 0, 15, 0.01, 0.1, 0.8);
  radio_tx_power_mapping_t b = radio_tx_map_power(&profile, 6, 15, 0.01, 0.1, 0.8);
  CHECK(a.valid && b.valid && fabs(a.amplitude_scale - 1) < 1e-12);
  CHECK(fabs(b.amplitude_scale / a.amplitude_scale - pow(10, 6.0 / 20)) < 1e-12);
  /* Four times the unscaled active-channel energy requires half the amplitude. */
  b = radio_tx_map_power(&profile, 0, 15, 0.04, 0.2, 0.8);
  CHECK(b.valid && fabs(b.amplitude_scale - 0.5) < 1e-12);
  b = radio_tx_map_power(&profile, 18, 15, 0.01, 0.1, 0.8);
  CHECK(b.valid && b.limited && b.selected_dbm == 15 && b.uncertainty_db == 0.7);
  CHECK(!radio_tx_map_power(&profile, 15, 15, 0.01, 0.5, 0.8).valid); // high PAPR exceeds the headroom limit
  /* Constant I = Q = 0.5 has complex mean power 0.5 and component peak 0.5. */
  CHECK(radio_tx_map_power(&profile, 0, 15, 0.5, 0.5, 0.8).valid);
  CHECK(radio_tx_map_power(&profile, 0, 15, 2, 1, 0.8).valid); // both components continuously at full scale
  CHECK(!radio_tx_map_power(&profile, 0, 15, 0.51, 0.5, 0.8).valid); // mean exceeds 2 * component peak squared
  CHECK(!radio_tx_map_power(&profile, 0, 15, 2.01, 1, 0.8).valid); // complex mean cannot exceed two
  CHECK(!radio_tx_map_power(&profile, 0, 23, 0.01, 0.1, 0.8).valid); // no fictitious P_CMAX
  CHECK(!radio_tx_map_power(&profile, -41, 15, 0.01, 0.1, 0.8).valid); // never raise the requested power
  CHECK(!radio_tx_map_power(&profile, 0, 15, 0, 0.1, 0.8).valid);
  CHECK(!radio_tx_map_power(&profile, 0, 15, 0.01, 0.05, 0.8).valid); // mean exceeds 2 * component peak squared
  CHECK(!radio_tx_map_power(&profile, 0, 15, 0.01, 0.1, 1.1).valid);
  CHECK(!radio_tx_map_power(&profile, NAN, 15, 0.01, 0.1, 0.8).valid);
  profile.qualified = false;
  CHECK(!radio_tx_map_power(&profile, 0, 15, 0.01, 0.1, 0.8).valid);
  puts("radio gain policy tests passed");
  return 0;
}
