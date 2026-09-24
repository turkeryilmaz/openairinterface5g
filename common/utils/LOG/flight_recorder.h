/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/**
 * @file flight_recorder.h
 * @brief Bounded numeric flight-recorder API for OAI diagnostic capture.
 *
 * The recorder is disabled unless OAI_FLIGHT_RECORDER_DIR names an existing
 * output directory before flight_recorder_init() runs. When disabled, emit
 * returns after one atomic state load and creates no files. Call sites must
 * guard any expensive argument preparation with flight_recorder_enabled().
 *
 * A successful initialization is one capture lifetime per process. Call init
 * before real-time producers and shutdown after they stop. Static rings are
 * intentionally retained after shutdown so a late producer cannot access
 * freed storage.
 */

#ifndef FLIGHT_RECORDER_H_
#define FLIGHT_RECORDER_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLIGHT_RECORDER_SCHEMA_VERSION 1U
#define FLIGHT_RECORDER_MAX_THREAD_RINGS 64U
#define FLIGHT_RECORDER_RING_RECORDS 1024U
#define FLIGHT_RECORDER_MAX_FILES 8U
#define FLIGHT_RECORDER_MAX_FILE_BYTES (16U * 1024U * 1024U)
#define FLIGHT_RECORDER_DEFAULT_MIN_FREE_BYTES (512ULL * 1024ULL * 1024ULL)

/**
 * Numeric event descriptor catalog. The a..f slots are signed 64-bit numeric
 * values. Their event-specific meanings are owned by the emitting call site;
 * no configuration, payload, UE identity string, or credential is captured.
 *
 *  1  LIFECYCLE       module or application lifecycle milestone
 * 10  UE_SYNC         UE synchronization result
 * 11  UE_AGC          UE automatic-gain-control result
 * 12  UE_MEASUREMENTS UE measurement summary
 * 13  UE_RA           UE random-access result
 * 14  UE_RRC          UE RRC transition/result
 * 15  UE_PDU          UE PDU result
 * 16  UE_TA           UE timing-advance observation
 * 17  UE_NAS         NAS dispatch/message/rejection metadata
 * 18  UE_RRC_TIMER   RRC timer lifecycle
 * 19  UE_CONTROL     RRC transitions and NAS recovery classification
 * 20  GNB_SLOT        gNB slot processing observation
 * 21  GNB_UE_BYTES    gNB per-UE byte counters
 * 22  GNB_UE_RADIO    gNB per-UE radio metrics
 * 23  GNB_RA          gNB random-access result
 * 24  GNB_UE_LINK     gNB per-UE link state/result
 * 25  GNB_DL_HARQ     gNB downlink HARQ result
 * 26  GNB_UL_HARQ     gNB uplink HARQ result
 * 27  UE_NAS_COUNT    NAS security counters/context presence, never keys
 * 30  RADIO_RX              radio receive result
 * 31  RADIO_TX              radio transmit result
 * 40  RADIO_GAIN            radio gain result
 * 41  RADIO_GAIN_TIME       radio gain device-time metadata
 * 42  RADIO_RX_LEVEL        sampled radio input level
 * 43  UE_TX_POWER_REQUEST   UE generated-channel request and digital reference
 * 44  GNB_TX_REFERENCE      gNB generated SSB configured/digital reference
 * 45  RADIO_RX_DECISION     RX policy decision and admission
 * 46  RADIO_RX_DECISION_INPUT RX decision gain/peak/error
 * 47  RADIO_TX_LEVEL        complete selected TX buffer before conversion
 * 48  RADIO_TX_LEVEL_STATE  TX buffer backend return and gain snapshot
 * 49  RADIO_TX_POWER        channel power mapping and application result
 * 50  RADIO_TX_POWER_SAMPLES mapped channel sample energy and peak
 * 51  RADIO_TX_REJECT       managed TX rejection reason
 * 52  RADIO_TX_POWER_QUALITY digital quantization error and profile uncertainty
 * 53  UE_TX_CONTROL         MAC channel power-control calculation context
 * 54  UE_PATHLOSS_STATE     checked serving-SSB pathloss availability transition
 * 55  UE_SSB_MEASUREMENT    SSB measurement acceptance and raw digital power
 * 56  UE_SSB_MEASUREMENT_CONTEXT sample/gain/overload evidence for SSB qualification
 * 57  RADIO_RX_PEAK_ENVELOPE retained peak constraint for gain increases
 */
typedef enum {
  FLIGHT_EVENT_LIFECYCLE = 1,
  FLIGHT_EVENT_UE_SYNC = 10,
  FLIGHT_EVENT_UE_AGC = 11,
  FLIGHT_EVENT_UE_MEASUREMENTS = 12,
  FLIGHT_EVENT_UE_RA = 13,
  FLIGHT_EVENT_UE_RRC = 14,
  FLIGHT_EVENT_UE_PDU = 15,
  FLIGHT_EVENT_UE_TA = 16,
  FLIGHT_EVENT_UE_NAS = 17,
  FLIGHT_EVENT_UE_RRC_TIMER = 18,
  FLIGHT_EVENT_UE_CONTROL = 19,
  FLIGHT_EVENT_GNB_SLOT = 20,
  FLIGHT_EVENT_GNB_UE_BYTES = 21,
  FLIGHT_EVENT_GNB_UE_RADIO = 22,
  FLIGHT_EVENT_GNB_RA = 23,
  FLIGHT_EVENT_GNB_UE_LINK = 24,
  FLIGHT_EVENT_GNB_DL_HARQ = 25,
  FLIGHT_EVENT_GNB_UL_HARQ = 26,
  FLIGHT_EVENT_UE_NAS_COUNT = 27,
  FLIGHT_EVENT_RADIO_RX = 30,
  FLIGHT_EVENT_RADIO_TX = 31,
  FLIGHT_EVENT_RADIO_GAIN = 40,
  FLIGHT_EVENT_RADIO_GAIN_TIME = 41,
  FLIGHT_EVENT_RADIO_RX_LEVEL = 42,
  FLIGHT_EVENT_UE_TX_POWER_REQUEST = 43,
  FLIGHT_EVENT_GNB_TX_REFERENCE = 44,
  FLIGHT_EVENT_RADIO_RX_DECISION = 45,
  FLIGHT_EVENT_RADIO_RX_DECISION_INPUT = 46,
  FLIGHT_EVENT_RADIO_TX_LEVEL = 47,
  FLIGHT_EVENT_RADIO_TX_LEVEL_STATE = 48,
  FLIGHT_EVENT_RADIO_TX_POWER = 49,
  FLIGHT_EVENT_RADIO_TX_POWER_SAMPLES = 50,
  FLIGHT_EVENT_RADIO_TX_REJECT = 51,
  FLIGHT_EVENT_RADIO_TX_POWER_QUALITY = 52,
  FLIGHT_EVENT_UE_TX_CONTROL = 53,
  FLIGHT_EVENT_UE_PATHLOSS_STATE = 54,
  FLIGHT_EVENT_UE_SSB_MEASUREMENT = 55,
  FLIGHT_EVENT_UE_SSB_MEASUREMENT_CONTEXT = 56,
  FLIGHT_EVENT_RADIO_RX_PEAK_ENVELOPE = 57,
} flight_recorder_event_t;

/** Channel discriminator carried by FLIGHT_EVENT_UE_TX_POWER_REQUEST. */
typedef enum {
  FLIGHT_UE_TX_CHANNEL_PRACH = 1,
  FLIGHT_UE_TX_CHANNEL_PUSCH = 2,
  FLIGHT_UE_TX_CHANNEL_PUCCH = 3,
  FLIGHT_UE_TX_CHANNEL_SRS = 4,
} flight_ue_tx_channel_t;

/**
 * Initialize one process-lifetime capture session.
 *
 * OAI_FLIGHT_RECORDER_DIR must name an existing directory. Files are created
 * directly in that directory with unique O_EXCL names and mode 0600. Optional
 * OAI_FLIGHT_RECORDER_MAX_BYTES accepts 0 (default: no total byte cap), or
 * 8192..INT64_MAX bytes. Sequential files are never overwritten. The writer
 * checks OAI_FLIGHT_RECORDER_MIN_FREE_BYTES (default 512 MiB) outside producer
 * paths. Exhaustion disables recording and reports a diagnostic; OAI continues.
 * Invalid paths or limits report a fixed diagnostic to stderr.
 *
 * This function prefaults the static recorder pages and starts its writer with
 * explicit SCHED_OTHER attributes. It is not for a real-time producer path.
 */
void flight_recorder_init(void);

/** Stop capture, drain enrolled producers, join the writer, and flush a footer. */
void flight_recorder_shutdown(void);

/** True only while producers may submit numeric records. */
bool flight_recorder_enabled(void);

/** Sample one in 1024 calls using a caller-owned counter; independent of chunk size. */
static inline bool flight_recorder_sample_due(uint32_t *calls)
{
  return (++*calls & 1023U) == 0;
}

/**
 * Submit a fixed-size numeric record. This is bounded producer work: two
 * normally-vDSO clock_gettime reads, atomic loads/fetch-adds, a fixed record
 * copy, and one SPSC release store. It performs no allocation, I/O, lock,
 * sleep, or retry loop. The first emit from a thread assigns one of the fixed
 * thread rings; excess threads increment no-slot metadata in the footer.
 *
 * Every written event has a globally unique sequence plus mono_ns and
 * realtime_ns (UTC epoch) fields. A failed timestamp read uses INT64_MIN and
 * increments invalid_timestamp metadata; unavailable values are never zero.
 * While capture runs, the writer also emits a health line at least once per
 * second with the current overflow, no-slot, and timestamp-invalid counters.
 */
void flight_recorder_emit(uint32_t event, int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f);

#ifdef __cplusplus
}
#endif

#endif /* FLIGHT_RECORDER_H_ */
