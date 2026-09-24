/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef OAI_RADIO_GAIN_DEVICE_H
#define OAI_RADIO_GAIN_DEVICE_H
#include "radio_gain.h"
#include "radio_gain_samples.h"
#include "radio_tx_power.h"
struct openair0_device;
struct openair0_config;
/* NR startup only, before streaming. Legacy/no-option devices remain untouched. */
int radio_gain_device_attach(struct openair0_device *device, struct openair0_config *config, const radio_gain_api_t *api);
/* Acquisition context only: this retains the legacy synchronous completion
 * contract while executing the driver call in the sole settings worker. */
int radio_gain_device_adjust_rx(struct openair0_device *device,
                                double delta_db,
                                double *applied_delta_db,
                                double *reported_gain_db);
/* UE synchronization calls this at acquisition/tracking boundaries. The call is
 * bounded and does not wait for a settings transaction; a legacy adjustment
 * serializes and records a preceding policy result under the sole consumer. */
typedef enum {
  RADIO_GAIN_RX_PHASE_ACQUISITION,
  RADIO_GAIN_RX_PHASE_TRACKING,
} radio_gain_device_rx_phase_t;
void radio_gain_device_set_rx_phase(radio_gain_device_rx_phase_t phase);
radio_gain_sample_context_t radio_gain_device_samples(struct openair0_device *device, int64_t first, int64_t end);
/* One physical RX chain in this integration. The reference input is mean raw
 * unitary-FFT bin power, before gain compensation. A full grid at that level
 * has the same mean complex sample power; no occupancy or FFT-size multiplier.
 * The function tries admission once and never waits for a driver operation. */
typedef enum {
  RADIO_RX_SOURCE_UE_SSB = 1,
  RADIO_RX_SOURCE_GNB_PUSCH = 2,
  RADIO_RX_SOURCE_UE_SEARCH = 3,
  RADIO_RX_SOURCE_HEADROOM = 4,
} radio_rx_source_t;
void radio_gain_device_observe_rx(const radio_gain_sample_context_t *context,
                                  double reference_bin_power,
                                  bool activity_valid,
                                  bool search_failed,
                                  radio_rx_source_t source);
/* Bounded PHY producer interface. Disabled/observe never mutates samples.
 * Any managed failure latches TX admission closed until process teardown;
 * failure is never interpreted as a request to transmit an unscaled waveform. */
/* Initialization before gNB workers/streaming, followed by read-only guards. */
bool radio_gain_device_configure_gnb_tx(double requested_sss_dbm, uint32_t fft_size, int16_t *amplitude);
bool radio_gain_device_validate_gnb_tx(const c16_t *samples, uint32_t count, int frame, int slot);
bool radio_gain_device_validate_gnb_reference(double requested_sss_dbm, int16_t amplitude, int frame, int slot);
enum {
  RADIO_TX_REJECT_LAYOUT = 1,
  RADIO_TX_REJECT_SPAN = 2,
  RADIO_TX_REJECT_OVERLAP = 3,
  RADIO_TX_REJECT_POWER_LIMIT = 4,
  RADIO_TX_REJECT_PROFILE = 5,
  RADIO_TX_REJECT_POWER_CONTROL = 6,
};
bool radio_gain_device_tx_selected(void);
bool radio_gain_device_tx_actuating(void);
/* True only for an admission error after a managed TX fault has already queued
 * coordinated shutdown. Other backend errors retain their ordinary handling. */
bool radio_gain_device_tx_cancelled(int result);
/* Validate the network ceiling before RA or a scheduled transmission. PHR and
 * the PHY must use a ceiling supported by the same physical operating point. */
bool radio_gain_device_validate_ue_power_limit(int p_max, int p_max_alt, int frame, int slot);
bool radio_gain_device_apply_tx(c16_t *samples, uint32_t count, double requested_dbm, int frame, int slot, unsigned channel);
void radio_gain_device_reject_tx(int frame, int slot, unsigned channel, int reason);
#endif
