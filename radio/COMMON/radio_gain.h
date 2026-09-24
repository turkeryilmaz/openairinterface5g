/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef OAI_RADIO_GAIN_H
#define OAI_RADIO_GAIN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Optional radio extension: independent of the legacy openair0_device layout. */
#define OAI_RADIO_GAIN_ABI 1U
#define OAI_RADIO_GAIN_SYMBOL "oai_radio_gain_get_api_v1"

typedef enum { RADIO_GAIN_RX, RADIO_GAIN_TX } radio_gain_direction_t;
typedef enum { RADIO_GAIN_SET_RX, RADIO_GAIN_SET_TX, RADIO_GAIN_RETUNE } radio_gain_operation_t;
typedef enum {
  RADIO_GAIN_OK,
  RADIO_GAIN_BUSY,
  RADIO_GAIN_UNSUPPORTED,
  RADIO_GAIN_INVALID,
  RADIO_GAIN_STALE,
  RADIO_GAIN_CLOSED,
  RADIO_GAIN_BACKEND_ERROR,
  RADIO_GAIN_TX_PENDING,
} radio_gain_status_t;

typedef struct {
  double minimum_db;
  double maximum_db;
  double step_db;
  double reported_db;
  double frequency_hz;
  double sample_rate_hz;
  double bandwidth_hz; /* actual analog/filter bandwidth, zero if unavailable */
  double power_reference_dbm;
  bool power_reference_valid;
  /* OAI converter component range, not the int16_t storage container range. */
  uint32_t component_full_scale;
  char identity[96];
  char antenna[32];
} radio_gain_channel_t;

typedef struct {
  uint32_t abi_version;
  uint32_t struct_size;
  int (*query)(void *device, radio_gain_direction_t direction, unsigned channel, radio_gain_channel_t *result);
  /* Set exactly one direction/channel; return hardware-reported/coerced gain. */
  int (*set_gain)(void *device, radio_gain_direction_t direction, unsigned channel, double gain_db, double *reported_db);
  int (*set_rx_agc)(void *device, unsigned channel, bool enable);
  /* Resolve routing on the caller; this operation only changes the given radio. */
  int (*retune)(void *device, unsigned rx_channel, unsigned tx_channel, double rx_hz, double tx_hz, double offset_hz);
  int (*device_ticks)(void *device, double sample_rate_hz, int64_t *ticks);
} radio_gain_api_t;

typedef const radio_gain_api_t *(*radio_gain_get_api_t)(uint32_t abi_version, size_t minimum_size);

typedef struct {
  radio_gain_operation_t operation;
  uint64_t generation;
  uint64_t request_id;
  double gain_db;
  double rx_frequency_hz;
  double tx_frequency_hz;
  double tune_offset_hz;
} radio_gain_request_t;

typedef struct {
  radio_gain_request_t request;
  radio_gain_status_t status;
  uint64_t generation;
  double reported_rx_db;
  double reported_tx_db;
  int64_t begin_device_ticks;
  int64_t end_device_ticks;
  bool device_time_valid;
  bool rx_gain_valid;
  bool tx_gain_valid;
} radio_gain_result_t;

typedef struct radio_gain_owner radio_gain_owner_t;

/* Startup/teardown only. The owner copies the API and adopts initialized settings.
 * Its worker is the only subsequent caller of query/set/retune/device_ticks.
 * No native RX AGC change is made unless host_rx_control is explicitly enabled. */
radio_gain_owner_t *radio_gain_owner_create(const radio_gain_api_t *api,
                                            void *device,
                                            unsigned rx_channel,
                                            unsigned tx_channel,
                                            bool host_rx_control);
/* Idempotently stop new admissions without joining the settings worker. It is
 * safe for a fatal-path coordinator to call before all producers quiesce; the
 * owner remains valid until a non-worker caller subsequently destroys it. */
void radio_gain_owner_close(radio_gain_owner_t *owner);
/* Install once at startup, before publishing the owner to producers. The first
 * positive failure code is delivered once from the existing settings worker.
 * Producer submission is a lock-free atomic operation, with no signal/I/O. */
void radio_gain_owner_set_failure_handler(radio_gain_owner_t *owner, void (*handler)(int));
void radio_gain_owner_report_failure(radio_gain_owner_t *owner, int failure);
/* Never called by a real-time producer. Caller first quiesces all producers.
 * Stop admissions, finish/cancel work and join before destroying the radio.
 * A blocked vendor call is left to the process-level shutdown boundary. */
void radio_gain_owner_destroy(radio_gain_owner_t *owner);

/* One bounded mailbox, try-admission only: safe for distinct acquisition and RX
 * producers without pretending that an SPSC queue supports multiple writers.
 * At most one request remains outstanding until its result is consumed. */
radio_gain_status_t radio_gain_submit(radio_gain_owner_t *owner, const radio_gain_request_t *request);
bool radio_gain_take_result(radio_gain_owner_t *owner, radio_gain_result_t *result);
/* Bounded race-free snapshot; false means unavailable/transition, never zero gain. */
bool radio_gain_snapshot(const radio_gain_owner_t *owner, radio_gain_result_t *result);
/* Non-real-time only: takes a mutex to copy the current worker-published profile. */
bool radio_gain_channels(const radio_gain_owner_t *owner, radio_gain_channel_t *rx, radio_gain_channel_t *tx);
/* Transition TX producer admission. Opening is refused while a TX setting or
 * retune is in flight; callers must honor false and defer submitted samples.
 * Before closing, every admitted producer must publish its last end tick;
 * closing then precedes the TX quiescence/tick gate. */
bool radio_gain_set_tx_admission(radio_gain_owner_t *owner, bool open);
/* Mark every submitted future TX interval while admission is open. A late end
 * violates the producer handoff and latches conservative TX uncertainty. */
void radio_gain_note_tx_end(radio_gain_owner_t *owner, int64_t end_ticks);

#ifdef __cplusplus
}
#endif
#endif
