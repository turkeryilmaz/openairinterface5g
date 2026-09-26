/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "radio_tx_power.h"
#include <limits.h>
#include <math.h>
#include <string.h>

#define TX_SCALE_FRACTION_BITS 30
#define TX_SCALE_ONE (INT64_C(1) << TX_SCALE_FRACTION_BITS)
#define TX_RELATIVE_NOMINAL_REFERENCE 23.0
#define TX_RELATIVE_PEAK_LIMIT_FS 0.7
#define TX_RELATIVE_MAXIMUM_QUANTIZATION_ERROR_DB 0.5
#define TX_RELATIVE_MAXIMUM_QUANTIZATION_EVM 0.03

static uint32_t component_magnitude(int32_t component)
{
  return component < 0 ? -component : component;
}

static int64_t scaled_component(int16_t component, int64_t coefficient)
{
  const int64_t product = (int64_t)component * coefficient;
  /* Symmetric round-to-nearest, half-way away from zero. The admitted
   * coefficient bounds product magnitude below 2^62. */
  const int64_t magnitude = product < 0 ? -product : product;
  const int64_t rounded = (magnitude + TX_SCALE_ONE / 2) / TX_SCALE_ONE;
  return product < 0 ? -rounded : rounded;
}

static radio_tx_power_result_t relative_result(radio_tx_power_status_t status)
{
  return (radio_tx_power_result_t){
      .status = status,
      .mapping = {.requested_dbm = NAN, .selected_dbm = NAN, .amplitude_scale = NAN, .estimated_dbm = NAN, .uncertainty_db = NAN},
      .quantization_error_db = NAN,
      .quantization_evm = NAN,
      .estimated_output_dbm = NAN,
      .requested_power_dbfs = NAN,
      .realized_power_dbfs = NAN};
}

bool radio_tx_relative_configure(uint32_t full_scale,
                                 double reference_amplitude,
                                 double backoff_db,
                                 radio_tx_relative_config_t *config)
{
  if (config != NULL)
    *config = (radio_tx_relative_config_t){0};
  if (config == NULL || !full_scale || full_scale > 32768U || !isfinite(reference_amplitude) || reference_amplitude <= 0
      || reference_amplitude > full_scale || !isfinite(backoff_db) || backoff_db < 0)
    return false;

  const double reference_dbfs = 20.0 * log10(reference_amplitude / full_scale);
  const double normalization = (double)full_scale * full_scale;
  const double power_error_evm = 1.0 - pow(10.0, -TX_RELATIVE_MAXIMUM_QUANTIZATION_ERROR_DB / 20.0);
  const double evm_floor =
      TX_RELATIVE_NOMINAL_REFERENCE
      + 10.0 * log10((0.5 / (TX_RELATIVE_MAXIMUM_QUANTIZATION_EVM * TX_RELATIVE_MAXIMUM_QUANTIZATION_EVM)) / normalization)
      - reference_dbfs;
  const double power_error_floor =
      TX_RELATIVE_NOMINAL_REFERENCE + 10.0 * log10((0.5 / (power_error_evm * power_error_evm)) / normalization) - reference_dbfs;
  const double minimum = ceil(fmax(evm_floor, power_error_floor));
  const double maximum = floor(TX_RELATIVE_NOMINAL_REFERENCE - backoff_db);
  if (!isfinite(reference_dbfs) || !isfinite(power_error_evm) || power_error_evm <= 0 || !isfinite(minimum) || !isfinite(maximum)
      || minimum < INT_MIN || minimum > INT_MAX || maximum < INT_MIN || maximum > INT_MAX || minimum > maximum)
    return false;

  *config = (radio_tx_relative_config_t){.component_full_scale = full_scale,
                                         .reference_amplitude = reference_amplitude,
                                         .backoff_db = backoff_db,
                                         .reference_dbfs = reference_dbfs,
                                         .nominal_reference = TX_RELATIVE_NOMINAL_REFERENCE,
                                         .nominal_min = (int)minimum,
                                         .nominal_max = (int)maximum,
                                         .peak_limit_fs = TX_RELATIVE_PEAK_LIMIT_FS,
                                         .maximum_quantization_error_db = TX_RELATIVE_MAXIMUM_QUANTIZATION_ERROR_DB,
                                         .maximum_quantization_evm = TX_RELATIVE_MAXIMUM_QUANTIZATION_EVM};
  return true;
}

static bool relative_config_valid(const radio_tx_relative_config_t *config)
{
  if (config == NULL)
    return false;
  radio_tx_relative_config_t expected;
  return radio_tx_relative_configure(config->component_full_scale, config->reference_amplitude, config->backoff_db, &expected)
         && config->reference_dbfs == expected.reference_dbfs && config->nominal_reference == expected.nominal_reference
         && config->nominal_min == expected.nominal_min && config->nominal_max == expected.nominal_max
         && config->peak_limit_fs == expected.peak_limit_fs
         && config->maximum_quantization_error_db == expected.maximum_quantization_error_db
         && config->maximum_quantization_evm == expected.maximum_quantization_evm;
}

radio_tx_power_result_t radio_tx_apply_power(c16_t *samples,
                                             uint32_t count,
                                             uint32_t full_scale,
                                             const radio_tx_power_profile_t *profile,
                                             double requested_dbm,
                                             double allowed_maximum_dbm,
                                             double peak_limit_fs,
                                             double maximum_quantization_error_db,
                                             double maximum_quantization_evm,
                                             bool apply)
{
  radio_tx_power_result_t result = {.status = RADIO_TX_POWER_INVALID, .requested_power_dbfs = NAN, .realized_power_dbfs = NAN};
  if (!samples || !count || count > RADIO_TX_POWER_MAX_SAMPLES || !full_scale || full_scale > 32768U || !isfinite(peak_limit_fs)
      || peak_limit_fs <= 0 || peak_limit_fs > 1 || !isfinite(maximum_quantization_error_db) || maximum_quantization_error_db < 0
      || !isfinite(maximum_quantization_evm) || maximum_quantization_evm <= 0 || maximum_quantization_evm > 0.03)
    return result;
  if (!profile || !profile->qualified) {
    result.status = RADIO_TX_POWER_UNQUALIFIED;
    return result;
  }
  result.sample_count = count;
  for (uint32_t i = 0; i < count; ++i) {
    const int32_t r = samples[i].r, q = samples[i].i;
    result.input_energy += (int64_t)r * r + (int64_t)q * q;
    const uint32_t peak = component_magnitude(r) > component_magnitude(q) ? component_magnitude(r) : component_magnitude(q);
    if (peak > result.input_peak_component)
      result.input_peak_component = peak;
  }
  const double normalization = (double)count * full_scale * full_scale;
  result.mapping = radio_tx_map_power(profile,
                                      requested_dbm,
                                      allowed_maximum_dbm,
                                      result.input_energy / normalization,
                                      (double)result.input_peak_component / full_scale,
                                      peak_limit_fs);
  /* MAC already selected its request under the network/device limit. Do not
   * lower it here while retaining incompatible PHR state. */
  if (!result.mapping.valid || result.mapping.limited) {
    result.status = result.mapping.peak_limit_exceeded ? RADIO_TX_POWER_HEADROOM : RADIO_TX_POWER_MAPPING_REJECTED;
    return result;
  }
  const double coefficient_value = result.mapping.amplitude_scale * TX_SCALE_ONE;
  if (!isfinite(coefficient_value) || coefficient_value < 0.5 || coefficient_value > (double)(INT64_C(1) << 46)) {
    result.status = RADIO_TX_POWER_QUANTIZATION;
    return result;
  }
  const int64_t coefficient = llround(coefficient_value);
  const int64_t positive_limit = (int64_t)fmin(full_scale - 1U, floor(peak_limit_fs * full_scale));
  const int64_t negative_limit = -(int64_t)floor(peak_limit_fs * full_scale);
  int64_t input_output_dot = 0;
  /* Preflight exact fixed-point outputs. Never saturate/wrap or leave a partly
   * scaled waveform when a later sample violates the limit. */
  for (uint32_t i = 0; i < count; ++i) {
    const int64_t r = scaled_component(samples[i].r, coefficient);
    const int64_t q = scaled_component(samples[i].i, coefficient);
    if (r < negative_limit || r > positive_limit || q < negative_limit || q > positive_limit) {
      result.status = RADIO_TX_POWER_HEADROOM;
      return result;
    }
    result.output_energy += r * r + q * q;
    input_output_dot += r * samples[i].r + q * samples[i].i;
    const uint32_t peak = component_magnitude(r) > component_magnitude(q) ? component_magnitude(r) : component_magnitude(q);
    if (peak > result.output_peak_component)
      result.output_peak_component = peak;
  }
  if (!result.output_energy) {
    result.status = RADIO_TX_POWER_QUANTIZATION;
    return result;
  }
  const double actual_scale = (double)coefficient / TX_SCALE_ONE;
  const double ideal_energy = actual_scale * actual_scale * result.input_energy;
  const double error_energy = fmax(0.0, result.output_energy - 2.0 * actual_scale * input_output_dot + ideal_energy);
  result.quantization_evm = sqrt(error_energy / ideal_energy);
  if (!isfinite(result.quantization_evm) || result.quantization_evm > maximum_quantization_evm) {
    result.status = RADIO_TX_POWER_QUANTIZATION;
    return result;
  }
  result.estimated_output_dbm = profile->reference_dbm + 10.0 * log10(result.output_energy / normalization);
  result.quantization_error_db = result.estimated_output_dbm - result.mapping.selected_dbm;
  if (!isfinite(result.estimated_output_dbm) || !isfinite(result.quantization_error_db)
      || fabs(result.quantization_error_db) > maximum_quantization_error_db) {
    result.status = RADIO_TX_POWER_QUANTIZATION;
    return result;
  }
  if (apply) {
    for (uint32_t i = 0; i < count; ++i) {
      samples[i].r = scaled_component(samples[i].r, coefficient);
      samples[i].i = scaled_component(samples[i].i, coefficient);
    }
    result.applied = true;
  }
  result.status = RADIO_TX_POWER_OK;
  return result;
}

radio_tx_power_result_t radio_tx_apply_relative_power(c16_t *samples,
                                                      uint32_t count,
                                                      const radio_tx_relative_config_t *config,
                                                      double selected_nominal,
                                                      bool apply)
{
  radio_tx_power_result_t result = relative_result(RADIO_TX_POWER_INVALID);
  if (!samples || !count || count > RADIO_TX_POWER_MAX_SAMPLES || !relative_config_valid(config))
    return result;
  if (!isfinite(selected_nominal) || selected_nominal < config->nominal_min || selected_nominal > config->nominal_max) {
    result.status = RADIO_TX_POWER_MAPPING_REJECTED;
    return result;
  }

  result.sample_count = count;
  for (uint32_t i = 0; i < count; ++i) {
    const int32_t r = samples[i].r, q = samples[i].i;
    result.input_energy += (int64_t)r * r + (int64_t)q * q;
    const uint32_t peak = component_magnitude(r) > component_magnitude(q) ? component_magnitude(r) : component_magnitude(q);
    if (peak > result.input_peak_component)
      result.input_peak_component = peak;
  }
  if (!result.input_energy) {
    result.status = RADIO_TX_POWER_MAPPING_REJECTED;
    return result;
  }

  const double normalization = (double)count * config->component_full_scale * config->component_full_scale;
  const double requested_power_dbfs = config->reference_dbfs + selected_nominal - config->nominal_reference;
  result.requested_power_dbfs = requested_power_dbfs;
  const double target_energy = normalization * pow(10.0, requested_power_dbfs / 10.0);
  const double coefficient_value = sqrt(target_energy / result.input_energy) * TX_SCALE_ONE;
  if (!isfinite(normalization) || !isfinite(requested_power_dbfs) || !isfinite(target_energy) || target_energy <= 0
      || !isfinite(coefficient_value) || coefficient_value < 0.5 || coefficient_value > (double)(INT64_C(1) << 46)) {
    result.status = RADIO_TX_POWER_QUANTIZATION;
    return result;
  }

  const int64_t coefficient = llround(coefficient_value);
  const int64_t positive_limit =
      (int64_t)fmin(config->component_full_scale - 1U, floor(config->peak_limit_fs * config->component_full_scale));
  const int64_t negative_limit = -(int64_t)floor(config->peak_limit_fs * config->component_full_scale);
  uint64_t output_energy = 0;
  uint32_t output_peak_component = 0;
  int64_t input_output_dot = 0;
  /* Preflight every exact output before mutating the active span. An exceptional
   * crest is an erasure decision for the caller, never saturation or restart. */
  for (uint32_t i = 0; i < count; ++i) {
    const int64_t r = scaled_component(samples[i].r, coefficient);
    const int64_t q = scaled_component(samples[i].i, coefficient);
    if (r < negative_limit || r > positive_limit || q < negative_limit || q > positive_limit) {
      result.status = RADIO_TX_POWER_HEADROOM;
      return result;
    }
    output_energy += r * r + q * q;
    input_output_dot += r * samples[i].r + q * samples[i].i;
    const uint32_t peak = component_magnitude(r) > component_magnitude(q) ? component_magnitude(r) : component_magnitude(q);
    if (peak > output_peak_component)
      output_peak_component = peak;
  }
  if (!output_energy) {
    result.status = RADIO_TX_POWER_QUANTIZATION;
    return result;
  }

  const double actual_scale = (double)coefficient / TX_SCALE_ONE;
  const double ideal_energy = actual_scale * actual_scale * result.input_energy;
  const double error_energy = fmax(0.0, output_energy - 2.0 * actual_scale * input_output_dot + ideal_energy);
  const double quantization_evm = sqrt(error_energy / ideal_energy);
  const double realized_power_dbfs = 10.0 * log10(output_energy / normalization);
  const double quantization_error_db = realized_power_dbfs - requested_power_dbfs;
  if (!isfinite(quantization_evm) || quantization_evm > config->maximum_quantization_evm || !isfinite(realized_power_dbfs)
      || !isfinite(quantization_error_db) || fabs(quantization_error_db) > config->maximum_quantization_error_db) {
    result.status = RADIO_TX_POWER_QUANTIZATION;
    return result;
  }

  if (apply) {
    for (uint32_t i = 0; i < count; ++i) {
      samples[i].r = scaled_component(samples[i].r, coefficient);
      samples[i].i = scaled_component(samples[i].i, coefficient);
    }
    result.applied = true;
  }
  result.mapping.valid = true;
  result.mapping.amplitude_scale = actual_scale;
  result.output_energy = output_energy;
  result.output_peak_component = output_peak_component;
  result.quantization_error_db = quantization_error_db;
  result.quantization_evm = quantization_evm;
  result.requested_power_dbfs = requested_power_dbfs;
  result.realized_power_dbfs = realized_power_dbfs;
  result.status = RADIO_TX_POWER_OK;
  return result;
}

bool radio_tx_profile_valid(const radio_tx_profile_t *p)
{
  return p && p->power.qualified && p->id[0] && memchr(p->id, 0, sizeof(p->id)) && p->identity[0]
         && memchr(p->identity, 0, sizeof(p->identity)) && p->antenna[0] && memchr(p->antenna, 0, sizeof(p->antenna))
         && p->provenance[0] && memchr(p->provenance, 0, sizeof(p->provenance)) && isfinite(p->minimum_frequency_hz)
         && p->minimum_frequency_hz > 0 && isfinite(p->maximum_frequency_hz) && p->maximum_frequency_hz >= p->minimum_frequency_hz
         && isfinite(p->sample_rate_hz) && p->sample_rate_hz > 0 && isfinite(p->bandwidth_hz) && p->bandwidth_hz > 0
         && isfinite(p->reported_gain_db) && p->component_full_scale > 0 && p->component_full_scale <= 32768
         && isfinite(p->power.reference_dbm) && isfinite(p->power.uncertainty_db) && p->power.uncertainty_db > 0
         && isfinite(p->power.minimum_dbm) && isfinite(p->power.maximum_dbm) && p->power.minimum_dbm <= p->power.maximum_dbm
         && isfinite(p->peak_limit_fs) && p->peak_limit_fs > 0 && p->peak_limit_fs <= 1
         && isfinite(p->maximum_quantization_error_db) && p->maximum_quantization_error_db >= 0
         && p->maximum_quantization_error_db <= 1 && isfinite(p->maximum_quantization_evm) && p->maximum_quantization_evm > 0
         && p->maximum_quantization_evm <= 0.03;
}

bool radio_tx_profile_matches(const radio_tx_profile_t *p, const radio_gain_channel_t *c)
{
  if (!radio_tx_profile_valid(p) || !c || !memchr(c->identity, 0, sizeof(c->identity))
      || !memchr(c->antenna, 0, sizeof(c->antenna)))
    return false;
  /* Tolerances admit numerical device readback, not another gain/rate setting. */
  return strcmp(p->identity, c->identity) == 0 && strcmp(p->antenna, c->antenna) == 0 && isfinite(c->frequency_hz)
         && c->frequency_hz >= p->minimum_frequency_hz && c->frequency_hz <= p->maximum_frequency_hz && isfinite(c->sample_rate_hz)
         && fabs(c->sample_rate_hz - p->sample_rate_hz) <= fmax(0.01, p->sample_rate_hz * 1e-8) && isfinite(c->reported_db)
         && fabs(c->reported_db - p->reported_gain_db) <= 0.01 && isfinite(c->bandwidth_hz)
         && fabs(c->bandwidth_hz - p->bandwidth_hz) <= fmax(0.01, p->bandwidth_hz * 1e-8)
         && c->component_full_scale == p->component_full_scale;
}

bool radio_tx_sss_amplitude(const radio_tx_profile_t *p,
                            uint32_t fft_size,
                            double requested_re_dbm,
                            int16_t *amplitude,
                            double *estimated_re_dbm)
{
  if (!radio_tx_profile_valid(p) || !fft_size || fft_size > 65536 || !isfinite(requested_re_dbm) || !amplitude || !estimated_re_dbm)
    return false;
  const double wanted_component =
      p->component_full_scale * sqrt((double)fft_size) * pow(10.0, (requested_re_dbm - p->power.reference_dbm) / 20.0);
  const double candidate = wanted_component * (32768.0 / 23170.0);
  /* Leave room for the existing largest channel multiplier (CSI-RS +6 dB).
   * Composite time-domain headroom remains a separate emission check. */
  if (!isfinite(candidate) || candidate < 1 || candidate > INT16_MAX / 2)
    return false;
  const int16_t selected = lround(candidate);
  const int positive = ((int32_t)selected * 23170) / 32768;
  const int negative_magnitude = (((int32_t)selected * 23170) + 32767) / 32768;
  const double mean_re_energy = ((double)positive * positive + (double)negative_magnitude * negative_magnitude) / 2;
  const double realized = p->power.reference_dbm
                          + 10.0 * log10(mean_re_energy / ((double)fft_size * p->component_full_scale * p->component_full_scale));
  if (!isfinite(realized) || fabs(realized - requested_re_dbm) > p->maximum_quantization_error_db)
    return false;
  *amplitude = selected;
  *estimated_re_dbm = realized;
  return true;
}
