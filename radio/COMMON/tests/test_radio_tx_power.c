/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "radio_tx_power.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x)                                                 \
  do {                                                           \
    if (!(x)) {                                                  \
      fprintf(stderr, "check failed at %d: %s\n", __LINE__, #x); \
      return 1;                                                  \
    }                                                            \
  } while (0)

int main(void)
{
  const radio_tx_power_profile_t profile = {.qualified = true,
                                            .reference_dbm = 0,
                                            .uncertainty_db = 0.5,
                                            .minimum_dbm = -100,
                                            .maximum_dbm = 3.1};
  c16_t samples[] = {{1024, 0}, {-1024, 0}};
  c16_t original[2];
  memcpy(original, samples, sizeof(samples));
  const double quarter_power_db = -12.041199826559248;
  radio_tx_power_result_t r = radio_tx_apply_power(samples, 2, 2048, &profile, quarter_power_db, 0, 0.9, 0.01, 0.03, false);
  CHECK(r.status == RADIO_TX_POWER_OK && !r.applied && !memcmp(samples, original, sizeof(samples)));
  CHECK(r.input_energy == 2097152 && r.output_energy == 524288 && r.output_peak_component == 512);
  CHECK(fabs(r.estimated_output_dbm - quarter_power_db) < 1e-10);
  CHECK(isnan(r.requested_power_dbfs) && isnan(r.realized_power_dbfs));
  r = radio_tx_apply_power(samples, 2, 2048, &profile, quarter_power_db, 0, 0.9, 0.01, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_OK && r.applied && samples[0].r == 512 && samples[1].r == -512);

  /* A pre-conversion waveform can be larger than converter FS if attenuated. */
  samples[0] = (c16_t){4096, -4096};
  r = radio_tx_apply_power(samples, 1, 2048, &profile, -3.010299956639812, 0, 0.9, 0.01, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_OK && samples[0].r == 1024 && samples[0].i == -1024);
  CHECK(r.input_energy == 33554432 && r.output_energy == 2097152);

  /* Negative full-scale is representable; positive full-scale is not. */
  samples[0] = (c16_t){-1024, 0};
  r = radio_tx_apply_power(samples, 1, 2048, &profile, 0, 0, 1, 0.01, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_OK && samples[0].r == -2048);
  samples[0] = (c16_t){0, 0};
  samples[1] = (c16_t){1024, 0};
  memcpy(original, samples, sizeof(samples));
  r = radio_tx_apply_power(samples, 2, 2048, &profile, -3.010299956639812, 0, 1, 0.01, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_HEADROOM && !r.applied && !memcmp(samples, original, sizeof(samples)));
  samples[0] = (c16_t){INT16_MIN, 0};
  r = radio_tx_apply_power(samples, 1, 32768, &profile, 0, 0, 1, 0.01, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_OK && samples[0].r == INT16_MIN);

  /* A finite power request can exceed the component peak cap before integer
   * preflight. Report headroom, preserve the waveform, and do not call it an
   * invalid profile/range. */
  samples[0] = (c16_t){1000, 0};
  memcpy(original, samples, sizeof(samples));
  r = radio_tx_apply_power(samples, 1, 2048, &profile, -3, 0, 0.7, 0.1, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_HEADROOM && r.mapping.peak_limit_exceeded && !r.applied);
  CHECK(!memcmp(samples, original, sizeof(samples)));
  r = radio_tx_apply_power(samples, 1, 2048, &profile, -101, 0, 0.7, 0.1, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_MAPPING_REJECTED && !r.mapping.peak_limit_exceeded && !r.applied);
  CHECK(!memcmp(samples, original, sizeof(samples)));

  samples[0] = (c16_t){1000, -1000};
  memcpy(original, samples, sizeof(samples));
  r = radio_tx_apply_power(samples, 1, 2048, &profile, -80, 0, 0.9, 0.1, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_QUANTIZATION && !memcmp(samples, original, sizeof(samples)));
  r = radio_tx_apply_power(samples, 1, 2048, &profile, -5, -10, 0.9, 0.1, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_MAPPING_REJECTED && !memcmp(samples, original, sizeof(samples)));
  samples[0] = (c16_t){1000, 500};
  r = radio_tx_apply_power(samples, 1, 2048, &profile, 10 * log10(12.8 / (2048.0 * 2048)), 0, 0.9, 0.5, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_QUANTIZATION && samples[0].r == 1000 && samples[0].i == 500);
  memcpy(samples, original, sizeof(samples));
  radio_tx_power_profile_t unqualified = profile;
  unqualified.qualified = false;
  r = radio_tx_apply_power(samples, 1, 2048, &unqualified, -10, 0, 0.9, 0.1, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_UNQUALIFIED && !memcmp(samples, original, sizeof(samples)));
  r = radio_tx_apply_power(samples, 1, 2048, &profile, NAN, 0, 0.9, 0.1, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_MAPPING_REJECTED && !memcmp(samples, original, sizeof(samples)));
  CHECK(radio_tx_apply_power(NULL, 1, 2048, &profile, -10, 0, 0.9, 0.1, 0.03, true).status == RADIO_TX_POWER_INVALID);
  CHECK(radio_tx_apply_power(samples, 0, 2048, &profile, -10, 0, 0.9, 0.1, 0.03, true).status == RADIO_TX_POWER_INVALID);
  CHECK(radio_tx_apply_power(samples, RADIO_TX_POWER_MAX_SAMPLES + 1, 2048, &profile, -10, 0, 0.9, 0.1, 0.03, true).status
        == RADIO_TX_POWER_INVALID);
  CHECK(radio_tx_apply_power(samples, 1, 32769, &profile, -10, 0, 0.9, 0.1, 0.03, true).status == RADIO_TX_POWER_INVALID);
  CHECK(radio_tx_apply_power(samples, 1, 2048, &profile, -10, 0, NAN, 0.1, 0.03, true).status == RADIO_TX_POWER_INVALID);
  CHECK(radio_tx_apply_power(samples, 1, 2048, &profile, -10, 0, 0.9, -0.1, 0.03, true).status == RADIO_TX_POWER_INVALID);

  /* The relative path is a declared digital RMS envelope only. AMP512 at
   * nominal23 is -12.041 dBFS on a 2048 component scale; default 6 dB
   * backoff therefore caps requests at nominal17. */
  radio_tx_relative_config_t relative = {0};
  CHECK(radio_tx_relative_configure(2048, 512, 6, &relative));
  CHECK(relative.nominal_reference == 23 && relative.nominal_min == -3 && relative.nominal_max == 17);
  CHECK(fabs(relative.reference_dbfs - 20 * log10(0.25)) < 1e-12 && relative.peak_limit_fs == 0.7
        && relative.maximum_quantization_error_db == 0.5 && relative.maximum_quantization_evm == 0.03);

  c16_t relative_samples[] = {{512, 0}, {-512, 0}};
  c16_t relative_original[2];
  memcpy(relative_original, relative_samples, sizeof(relative_samples));
  r = radio_tx_apply_relative_power(relative_samples, 2, &relative, 17, false);
  CHECK(r.status == RADIO_TX_POWER_OK && !r.applied && r.mapping.valid
        && !memcmp(relative_samples, relative_original, sizeof(relative_samples)));
  CHECK(isnan(r.mapping.requested_dbm) && isnan(r.mapping.selected_dbm) && isnan(r.mapping.estimated_dbm)
        && isnan(r.mapping.uncertainty_db) && isnan(r.estimated_output_dbm));
  CHECK(r.input_energy == 524288 && r.output_energy == 132098 && r.input_peak_component == 512 && r.output_peak_component == 257);
  /* mapping.amplitude_scale is the admitted Q30 coefficient, not the
   * unquantized ideal scale. One Q30 unit is below 1e-9. */
  CHECK(fabs(r.mapping.amplitude_scale - pow(10.0, -6.0 / 20.0)) < 1e-9 && fabs(r.requested_power_dbfs + 18.041199826559248) < 1e-12
        && fabs(r.realized_power_dbfs - r.requested_power_dbfs) < 0.02 && fabs(r.quantization_error_db) < 0.02
        && r.quantization_evm < 0.03);

  r = radio_tx_apply_relative_power(relative_samples, 2, &relative, 17, true);
  CHECK(r.status == RADIO_TX_POWER_OK && r.applied && relative_samples[0].r == 257 && relative_samples[1].r == -257);
  memcpy(relative_samples, relative_original, sizeof(relative_samples));
  r = radio_tx_apply_relative_power(relative_samples, 2, &relative, 16, false);
  CHECK(r.status == RADIO_TX_POWER_OK && fabs(r.requested_power_dbfs + 19.041199826559248) < 1e-12
        && fabs(r.realized_power_dbfs - r.requested_power_dbfs) < 0.02);

  /* Target RMS covers the entire active span, including zeros. The sparse
   * representation has the same source energy but twice the span length. */
  c16_t dense[] = {{512, 0}, {-512, 0}};
  c16_t sparse[] = {{512, 0}, {0, 0}, {-512, 0}, {0, 0}};
  c16_t sparse_original[4];
  memcpy(sparse_original, sparse, sizeof(sparse));
  radio_tx_power_result_t dense_result = radio_tx_apply_relative_power(dense, 2, &relative, 17, false);
  radio_tx_power_result_t sparse_result = radio_tx_apply_relative_power(sparse, 4, &relative, 17, false);
  CHECK(dense_result.status == RADIO_TX_POWER_OK && sparse_result.status == RADIO_TX_POWER_OK
        && dense_result.input_energy == sparse_result.input_energy && sparse_result.output_energy > dense_result.output_energy
        && fabs(dense_result.realized_power_dbfs - sparse_result.realized_power_dbfs) < 0.02
        && !memcmp(sparse, sparse_original, sizeof(sparse)));

  memcpy(relative_samples, relative_original, sizeof(relative_samples));
  r = radio_tx_apply_relative_power(relative_samples, 2, &relative, relative.nominal_min, false);
  CHECK(r.status == RADIO_TX_POWER_OK && r.quantization_evm <= 0.03 && fabs(r.quantization_error_db) <= 0.5
        && !memcmp(relative_samples, relative_original, sizeof(relative_samples)));
  r = radio_tx_apply_relative_power(relative_samples, 2, &relative, relative.nominal_min - 1, true);
  CHECK(r.status == RADIO_TX_POWER_MAPPING_REJECTED && !r.applied && isnan(r.requested_power_dbfs) && isnan(r.realized_power_dbfs)
        && isnan(r.mapping.amplitude_scale) && !memcmp(relative_samples, relative_original, sizeof(relative_samples)));
  r = radio_tx_apply_relative_power(relative_samples, 2, &relative, relative.nominal_max + 1, true);
  CHECK(r.status == RADIO_TX_POWER_MAPPING_REJECTED && !r.applied
        && !memcmp(relative_samples, relative_original, sizeof(relative_samples)));

  c16_t silent[] = {{0, 0}, {0, 0}};
  c16_t silent_original[2];
  memcpy(silent_original, silent, sizeof(silent));
  r = radio_tx_apply_relative_power(silent, 2, &relative, 17, true);
  CHECK(r.status == RADIO_TX_POWER_MAPPING_REJECTED && !r.applied && isnan(r.requested_power_dbfs)
        && !memcmp(silent, silent_original, sizeof(silent)));
  CHECK(radio_tx_apply_relative_power(NULL, 1, &relative, 17, false).status == RADIO_TX_POWER_INVALID);
  CHECK(radio_tx_apply_relative_power(relative_samples, 0, &relative, 17, false).status == RADIO_TX_POWER_INVALID);

  c16_t extreme[] = {{INT16_MIN, INT16_MIN}};
  c16_t extreme_original[1];
  memcpy(extreme_original, extreme, sizeof(extreme));
  r = radio_tx_apply_relative_power(extreme, 1, &relative, 17, false);
  CHECK(r.status == RADIO_TX_POWER_OK && !r.applied && !memcmp(extreme, extreme_original, sizeof(extreme)));

  c16_t crest[32] = {{512, 0}};
  c16_t crest_original[32];
  memcpy(crest_original, crest, sizeof(crest));
  r = radio_tx_apply_relative_power(crest, 32, &relative, 17, true);
  CHECK(r.status == RADIO_TX_POWER_HEADROOM && !r.applied && isnan(r.mapping.amplitude_scale)
        && !memcmp(crest, crest_original, sizeof(crest)));

  radio_tx_relative_config_t malformed = relative;
  malformed.peak_limit_fs = 0.9;
  r = radio_tx_apply_relative_power(relative_samples, 2, &malformed, 17, false);
  CHECK(r.status == RADIO_TX_POWER_INVALID && isnan(r.requested_power_dbfs));
  CHECK(!radio_tx_relative_configure(0, 512, 6, &malformed));
  CHECK(!radio_tx_relative_configure(2048, 0, 6, &malformed));
  CHECK(!radio_tx_relative_configure(2048, 2049, 6, &malformed));
  CHECK(!radio_tx_relative_configure(2048, NAN, 6, &malformed));
  CHECK(!radio_tx_relative_configure(2048, 512, -1, &malformed));
  CHECK(!radio_tx_relative_configure(2048, 512, 27, &malformed));

  static c16_t maximum[RADIO_TX_POWER_MAX_SAMPLES];
  for (unsigned i = 0; i < RADIO_TX_POWER_MAX_SAMPLES; ++i)
    maximum[i] = (c16_t){INT16_MIN, INT16_MIN};
  r = radio_tx_apply_power(maximum, RADIO_TX_POWER_MAX_SAMPLES, 32768, &profile, -6.020599913279624, 0, 0.9, 0.01, 0.03, true);
  CHECK(r.status == RADIO_TX_POWER_OK && r.applied && r.input_energy == (UINT64_C(1) << 47));
  CHECK(maximum[0].r == -11585 && maximum[RADIO_TX_POWER_MAX_SAMPLES - 1].i == -11585);
  radio_tx_profile_t qualified = {.power = profile,
                                  .id = "synthetic-unit-test",
                                  .identity = "test:123:0",
                                  .antenna = "TX/RX",
                                  .provenance = "synthetic-test-only",
                                  .minimum_frequency_hz = 710749000,
                                  .maximum_frequency_hz = 710751000,
                                  .sample_rate_hz = 7680000,
                                  .bandwidth_hz = 20000000,
                                  .reported_gain_db = 60.75,
                                  .component_full_scale = 2048,
                                  .peak_limit_fs = 0.7,
                                  .maximum_quantization_error_db = 0.5,
                                  .maximum_quantization_evm = 0.03};
  radio_gain_channel_t channel = {.identity = "test:123:0",
                                  .antenna = "TX/RX",
                                  .frequency_hz = 710750000,
                                  .sample_rate_hz = 7680000.0006,
                                  .bandwidth_hz = 20000000,
                                  .reported_db = 60.75,
                                  .component_full_scale = 2048};
  CHECK(radio_tx_profile_valid(&qualified) && radio_tx_profile_matches(&qualified, &channel));
  int16_t reference_amp = 0;
  double reference_estimate = 0;
  CHECK(radio_tx_sss_amplitude(&qualified, 512, -40, &reference_amp, &reference_estimate));
  CHECK(reference_amp >= 654 && reference_amp <= 657 && fabs(reference_estimate + 40) < 0.03);
  CHECK(!radio_tx_sss_amplitude(&qualified, 512, 30, &reference_amp, &reference_estimate));
  CHECK(!radio_tx_sss_amplitude(&qualified, 0, -40, &reference_amp, &reference_estimate));
  CHECK(!radio_tx_sss_amplitude(&qualified, 512, NAN, &reference_amp, &reference_estimate));
  channel.bandwidth_hz = 10000000;
  CHECK(!radio_tx_profile_matches(&qualified, &channel));
  channel.bandwidth_hz = 20000000;
  channel.reported_db += 0.1;
  CHECK(!radio_tx_profile_matches(&qualified, &channel));
  channel.reported_db = 60.75;
  channel.frequency_hz = 710751001;
  CHECK(!radio_tx_profile_matches(&qualified, &channel));
  channel.frequency_hz = 710750000;
  channel.component_full_scale = 32768;
  CHECK(!radio_tx_profile_matches(&qualified, &channel));
  channel.component_full_scale = 2048;
  channel.sample_rate_hz = 15360000;
  CHECK(!radio_tx_profile_matches(&qualified, &channel));
  channel.sample_rate_hz = 7680000;
  channel.identity[5] = '9';
  CHECK(!radio_tx_profile_matches(&qualified, &channel));
  qualified.power.uncertainty_db = 0;
  CHECK(!radio_tx_profile_valid(&qualified));
  qualified.power.uncertainty_db = 1;
  memset(qualified.identity, 'x', sizeof(qualified.identity));
  CHECK(!radio_tx_profile_valid(&qualified));
  puts("radio TX waveform power tests passed");
  return 0;
}
