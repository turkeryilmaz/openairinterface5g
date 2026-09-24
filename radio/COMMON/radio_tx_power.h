/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef OAI_RADIO_TX_POWER_H
#define OAI_RADIO_TX_POWER_H
#include "common/platform_types.h"
#include "radio_gain_policy.h"
#define RADIO_TX_POWER_MAX_SAMPLES 65536U

/* A profile is local to one physical connector and a verified operating range.
 * Its qualification is an operator/evidence decision, not a calibration implied
 * by the radio model or by a nonzero gain readback. */
typedef struct {
  radio_tx_power_profile_t power;
  char id[96];
  char identity[96];
  char antenna[32];
  char provenance[256];
  double minimum_frequency_hz;
  double maximum_frequency_hz;
  double sample_rate_hz;
  double bandwidth_hz;
  double reported_gain_db;
  uint32_t component_full_scale;
  double peak_limit_fs;
  double maximum_quantization_error_db;
  double maximum_quantization_evm;
} radio_tx_profile_t;

bool radio_tx_profile_valid(const radio_tx_profile_t *profile);
bool radio_tx_profile_matches(const radio_tx_profile_t *profile, const radio_gain_channel_t *channel);
/* OAI's unitary IFFT and SSS generator (23170/32768 BPSK components).
 * Configure one common amplitude, preserving channel-relative amplitudes. */
bool radio_tx_sss_amplitude(const radio_tx_profile_t *profile,
                            uint32_t fft_size,
                            double requested_re_dbm,
                            int16_t *amplitude,
                            double *estimated_re_dbm);

typedef enum {
  RADIO_TX_POWER_OK,
  RADIO_TX_POWER_INVALID,
  RADIO_TX_POWER_UNQUALIFIED,
  RADIO_TX_POWER_MAPPING_REJECTED,
  RADIO_TX_POWER_HEADROOM,
  RADIO_TX_POWER_QUANTIZATION,
} radio_tx_power_status_t;

typedef struct {
  radio_tx_power_status_t status;
  radio_tx_power_mapping_t mapping;
  bool applied;
  uint32_t sample_count;
  uint64_t input_energy;
  uint64_t output_energy;
  uint32_t input_peak_component;
  uint32_t output_peak_component;
  double quantization_error_db;
  double quantization_evm;
  double estimated_output_dbm;
} radio_tx_power_result_t;

/* One complete active time-domain channel span, including cyclic prefixes and
 * excluding idle slot samples. A single coefficient preserves relative channel
 * amplitudes and phase; this is not per-symbol or per-slot normalization.
 *
 * The caller owns the buffer exclusively. Validation and quantization checks
 * finish before the first write; every failure leaves it unchanged. Observe
 * uses the identical calculation with apply=false. A successful result is a
 * digital waveform/profile estimate, not evidence of RF emission. Composite
 * budgets and the profile-to-emission epoch remain caller obligations. */
radio_tx_power_result_t radio_tx_apply_power(c16_t *samples,
                                             uint32_t count,
                                             uint32_t component_full_scale,
                                             const radio_tx_power_profile_t *profile,
                                             double requested_dbm,
                                             double allowed_maximum_dbm,
                                             double peak_limit_fs,
                                             double maximum_quantization_error_db,
                                             double maximum_quantization_evm,
                                             bool apply);
#endif
