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
