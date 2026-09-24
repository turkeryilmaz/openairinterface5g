/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef OAI_AGC_OPTIONS_H
#define OAI_AGC_OPTIONS_H

#include "common/config/config_userapi.h"
#include "radio/COMMON/radio_tx_power.h"
#include <stdbool.h>
#include <stddef.h>

#define AGC_OPTION_ERROR_MAX 192
#define AGC_RX_SETTLE_DEFAULT_US 20000

typedef enum {
  AGC_ROLE_UE,
  AGC_ROLE_GNB,
} agc_role_t;

typedef enum {
  AGC_MODE_OFF,
  AGC_MODE_ACQUISITION,
  AGC_MODE_OBSERVE,
  AGC_MODE_CONTINUOUS,
} agc_mode_t;

typedef enum {
  AGC_DIRECTIONS_BOTH,
  AGC_DIRECTIONS_RX,
  AGC_DIRECTIONS_TX,
} agc_directions_t;

typedef enum {
  AGC_OPTION_SOURCE_DEFAULT,
  AGC_OPTION_SOURCE_CONFIG,
  AGC_OPTION_SOURCE_CLI,
} agc_option_source_t;

typedef enum {
  AGC_RX_ACQUISITION_CONFIGURED,
  AGC_RX_ACQUISITION_LEGACY,
  AGC_RX_ACQUISITION_NEW,
} agc_rx_acquisition_t;

typedef enum {
  AGC_RX_TRACKING_HOLD,
  AGC_RX_TRACKING_NEW,
} agc_rx_tracking_t;

typedef enum {
  AGC_TX_POLICY_BASELINE,
  AGC_TX_POLICY_MANAGED,
} agc_tx_policy_t;

/* Input to the pure resolver. Sources are also retained for startup reporting. */
typedef struct {
  agc_role_t role;
  const char *mode;
  const char *directions;
  agc_option_source_t mode_source;
  agc_option_source_t directions_source;
  bool legacy_set;
  bool legacy_requested;
  agc_option_source_t legacy_source;
} agc_option_request_t;

/*
 * A resolved policy is immutable after agc_start_options() succeeds. Observe
 * retains selected policies while both actuation flags remain false.
 */
typedef struct {
  agc_role_t role;
  agc_mode_t mode;
  agc_directions_t directions;
  agc_option_source_t mode_source;
  agc_option_source_t directions_source;
  bool legacy_requested;
  agc_option_source_t legacy_source;
  agc_rx_acquisition_t rx_acquisition;
  agc_rx_tracking_t rx_tracking;
  agc_tx_policy_t tx_policy;
  bool rx_actuation;
  bool tx_actuation;
  unsigned int rx_settle_us; /* sample exclusion after a completed device transaction */
  radio_tx_profile_t tx_profile;
} agc_options_t;

int agc_resolve_options(const agc_option_request_t *request, agc_options_t *resolved, char *error, size_t error_size);

/* Managed ownership defaults to software CFO correction rather than an unsafe
 * hardware retune across queued TX. Explicit supported choices are preserved. */
int agc_resolve_ue_cfo(const agc_options_t *options, bool supplied, int requested, int *resolved);

/* Startup only: parse normal OAI configuration descriptors, resolve, and log. */
int agc_start_options(configmodule_interface_t *cfg, int argc, char **argv, agc_role_t role);

const agc_options_t *get_agc_options(void);

const char *agc_role_name(agc_role_t role);
const char *agc_mode_name(agc_mode_t mode);
const char *agc_directions_name(agc_directions_t directions);
const char *agc_option_source_name(agc_option_source_t source);
const char *agc_rx_acquisition_name(agc_rx_acquisition_t acquisition);
const char *agc_rx_tracking_name(agc_rx_tracking_t tracking);
const char *agc_tx_policy_name(agc_tx_policy_t policy);

#endif
