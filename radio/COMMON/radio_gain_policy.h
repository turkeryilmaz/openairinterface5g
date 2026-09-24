/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef OAI_RADIO_GAIN_POLICY_H
#define OAI_RADIO_GAIN_POLICY_H
#include "radio_gain.h"

typedef enum {
  RADIO_RX_HOLD_INVALID,
  RADIO_RX_HOLD_UNSUPPORTED,
  RADIO_RX_HOLD_STALE,
  RADIO_RX_HOLD_TRANSITION,
  RADIO_RX_HOLD_INACTIVE,
  RADIO_RX_HOLD_DEADBAND,
  RADIO_RX_HOLD_COOLDOWN,
  RADIO_RX_HOLD_LIMIT,
  RADIO_RX_REDUCE_OVERLOAD,
  RADIO_RX_TRACK_LEVEL,
  RADIO_RX_SEARCH_STEP,
} radio_rx_reason_t;

typedef struct {
  double target_dbfs;
  double deadband_db;
  double maximum_step_db;
  double search_step_db;
  double peak_ceiling_dbfs;
  /* Input-referred peak retention decay. This engineering guard bounds only
   * positive gain changes; it is neither a sample-rate limit nor a 3GPP timer. */
  double peak_release_db_per_second;
  double near_rail_fraction;
  double filter_weight;
  uint64_t minimum_interval_ns;
  uint64_t maximum_age_ns;
} radio_rx_policy_config_t;

typedef struct {
  uint64_t generation;
  /* Both timestamps use the local host monotonic clock, never radio sample ticks. */
  uint64_t observation_ns;
  uint64_t now_ns;
  double reported_gain_db;
  double mean_power_dbfs;
  double peak_component_dbfs;
  uint32_t sampled_components;
  uint32_t near_rail_components;
  bool power_valid;
  bool gain_valid;
  bool settled;
  bool activity_valid;
  bool search_failed;
} radio_rx_observation_t;

typedef struct {
  bool filter_valid;
  double filtered_linear_power;
  uint64_t generation;
  uint64_t last_observation_ns;
  bool last_observation_valid;
  uint64_t last_action_generation;
  uint64_t last_action_ns;
  bool action_recorded;
} radio_rx_policy_state_t;

typedef struct {
  bool valid;
  /* Strongest observed component peak, referred to the receiver input. */
  double input_peak_dbfs;
  /* Local host monotonic time of the most recent accepted envelope update. */
  uint64_t last_update_ns;
} radio_rx_peak_envelope_t;

typedef struct {
  radio_rx_reason_t reason;
  bool change;
  double gain_db;
  double error_db;
  /* At the current reported gain, the retained input-referred peak expressed
   * at the converter input. Raw peak telemetry remains in the observation. */
  bool peak_bound_valid;
  double peak_bound_dbfs;
} radio_rx_decision_t;

bool radio_rx_policy_config_valid(const radio_rx_policy_config_t *config);
/* Single decision-owner context. No allocation, I/O, driver calls, or global state.
 * A higher generation invalidates filter and observation ordering only. An actual
 * action's monotonic cooldown remains valid across its resulting gain generation. */
radio_rx_decision_t radio_rx_decide(const radio_rx_policy_config_t *config,
                                    radio_rx_policy_state_t *state,
                                    radio_rx_peak_envelope_t *peak_envelope,
                                    const radio_gain_channel_t *channel,
                                    const radio_rx_observation_t *observation);
/* Record only an action that actually completed. Older generations, completions
 * preceding the accepted host-monotonic observation, and backward completions are ignored. */
void radio_rx_action_completed(radio_rx_policy_state_t *state, uint64_t generation, uint64_t now_ns);

typedef struct {
  bool qualified;
  double reference_dbm; /* connector power at unit complex RMS/full scale */
  double uncertainty_db;
  double minimum_dbm;
  double maximum_dbm;
} radio_tx_power_profile_t;

typedef struct {
  bool valid;
  bool limited;
  bool peak_limit_exceeded; /* finite mapping rejected before sample mutation */
  double requested_dbm;
  double selected_dbm;
  double amplitude_scale;
  double estimated_dbm;
  double uncertainty_db;
} radio_tx_power_mapping_t;

/* The waveform-specific caller supplies verified pre-mix active-channel complex
 * mean power E[I^2 + Q^2] / component_full_scale^2 and peak component magnitude
 * max(|I|, |Q|) / component_full_scale. peak_limit_fs is the single-channel
 * pre-mix component headroom limit. The complex mean is at most twice the square
 * of the peak component. Unscaled samples may exceed converter full scale:
 * attenuation can bring a valid int16_t waveform into range before conversion.
 * A valid mapping is only feasible for that channel; the
 * caller must still validate post-mix composite peaks and the NR channel budget
 * before emission. */
radio_tx_power_mapping_t radio_tx_map_power(const radio_tx_power_profile_t *profile,
                                            double requested_dbm,
                                            double allowed_maximum_dbm,
                                            double unscaled_mean_power_fs,
                                            double unscaled_peak_component_fs,
                                            double peak_limit_fs);
#endif
