/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "agc_options.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static agc_options_t active_options;

static int set_error(char *error, size_t error_size, const char *format, ...)
{
  if (error != NULL && error_size != 0) {
    va_list args;
    va_start(args, format);
    vsnprintf(error, error_size, format, args);
    va_end(args);
  }
  return -1;
}

static bool is_cli_option_present(int argc, char **argv, const char *option)
{
  char spelling[MAX_OPTNAME_SIZE + 3];
  const int length = snprintf(spelling, sizeof(spelling), "--%s", option);
  if (length < 0 || (size_t)length >= sizeof(spelling))
    return false;
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], spelling) == 0)
      return true;
  return false;
}

static agc_option_source_t string_option_source(const char *value, bool cli_present)
{
  if (value == NULL)
    return AGC_OPTION_SOURCE_DEFAULT;
  return cli_present ? AGC_OPTION_SOURCE_CLI : AGC_OPTION_SOURCE_CONFIG;
}

static agc_option_source_t parameter_option_source(const paramdef_t *options, int index, bool cli_present)
{
  if (!config_isparamset((paramdef_t *)options, index))
    return AGC_OPTION_SOURCE_DEFAULT;
  return cli_present ? AGC_OPTION_SOURCE_CLI : AGC_OPTION_SOURCE_CONFIG;
}

static int parse_mode(const char *mode, agc_mode_t *parsed, char *error, size_t error_size)
{
  if (strcmp(mode, "off") == 0)
    *parsed = AGC_MODE_OFF;
  else if (strcmp(mode, "acquisition") == 0)
    *parsed = AGC_MODE_ACQUISITION;
  else if (strcmp(mode, "observe") == 0)
    *parsed = AGC_MODE_OBSERVE;
  else if (strcmp(mode, "continuous") == 0)
    *parsed = AGC_MODE_CONTINUOUS;
  else
    return set_error(error, error_size, "invalid agc-mode '%s'; use off, acquisition, observe, or continuous", mode);
  return 0;
}

static int parse_directions(const char *directions, agc_directions_t *parsed, char *error, size_t error_size)
{
  if (strcmp(directions, "both") == 0)
    *parsed = AGC_DIRECTIONS_BOTH;
  else if (strcmp(directions, "rx") == 0)
    *parsed = AGC_DIRECTIONS_RX;
  else if (strcmp(directions, "tx") == 0)
    *parsed = AGC_DIRECTIONS_TX;
  else
    return set_error(error, error_size, "invalid agc-directions '%s'; use both, rx, or tx", directions);
  return 0;
}

static int parse_tx_power_mode(const char *mode, agc_tx_power_mode_t *parsed, char *error, size_t error_size)
{
  if (strcmp(mode, "absolute") == 0)
    *parsed = AGC_TX_POWER_ABSOLUTE;
  else if (strcmp(mode, "relative") == 0)
    *parsed = AGC_TX_POWER_RELATIVE;
  else
    return set_error(error, error_size, "invalid tx-power-mode '%s'; use absolute or relative", mode);
  return 0;
}

int agc_resolve_options(const agc_option_request_t *request, agc_options_t *resolved, char *error, size_t error_size)
{
  if (request == NULL || resolved == NULL)
    return set_error(error, error_size, "AGC resolver requires request and result storage");
  if (request->role != AGC_ROLE_UE && request->role != AGC_ROLE_GNB)
    return set_error(error, error_size, "invalid AGC role");

  agc_options_t options = {
      .role = request->role,
      .mode = AGC_MODE_OFF,
      .directions = AGC_DIRECTIONS_BOTH,
      .mode_source = AGC_OPTION_SOURCE_DEFAULT,
      .directions_source = AGC_OPTION_SOURCE_DEFAULT,
      .legacy_requested = request->legacy_requested,
      .legacy_source = request->legacy_source,
      .rx_acquisition = AGC_RX_ACQUISITION_CONFIGURED,
      .rx_tracking = AGC_RX_TRACKING_HOLD,
      .tx_policy = AGC_TX_POLICY_BASELINE,
      .tx_power_mode = AGC_TX_POWER_ABSOLUTE,
      .tx_power_mode_source = AGC_OPTION_SOURCE_DEFAULT,
      .rx_settle_us = AGC_RX_SETTLE_DEFAULT_US,
  };

  if (request->mode_source != AGC_OPTION_SOURCE_DEFAULT) {
    if (request->mode == NULL)
      return set_error(error, error_size, "agc-mode has no value");
    if (parse_mode(request->mode, &options.mode, error, error_size) != 0)
      return -1;
    options.mode_source = request->mode_source;
  } else if (request->legacy_requested) {
    options.mode = AGC_MODE_ACQUISITION;
    options.mode_source = request->legacy_source;
  }

  if (request->directions_source != AGC_OPTION_SOURCE_DEFAULT) {
    if (request->directions == NULL)
      return set_error(error, error_size, "agc-directions has no value");
    if (parse_directions(request->directions, &options.directions, error, error_size) != 0)
      return -1;
    options.directions_source = request->directions_source;
  }

  if (request->tx_power_mode_source != AGC_OPTION_SOURCE_DEFAULT) {
    if (request->tx_power_mode == NULL)
      return set_error(error, error_size, "tx-power-mode has no value");
    if (parse_tx_power_mode(request->tx_power_mode, &options.tx_power_mode, error, error_size) != 0)
      return -1;
    options.tx_power_mode_source = request->tx_power_mode_source;
  }

  if (request->role == AGC_ROLE_GNB && request->legacy_set)
    return set_error(error, error_size, "gNB does not support the UE-only legacy agc option");
  if (request->role == AGC_ROLE_GNB && options.mode == AGC_MODE_ACQUISITION)
    return set_error(error, error_size, "gNB does not support agc-mode acquisition");
  if (request->legacy_requested && (options.mode == AGC_MODE_OFF || options.mode == AGC_MODE_OBSERVE))
    return set_error(error, error_size, "legacy agc conflicts with agc-mode %s", agc_mode_name(options.mode));
  if (options.mode == AGC_MODE_ACQUISITION && options.directions == AGC_DIRECTIONS_TX
      && options.directions_source != AGC_OPTION_SOURCE_DEFAULT)
    return set_error(error, error_size, "agc-mode acquisition cannot use explicitly selected agc-directions tx");

  switch (options.mode) {
    case AGC_MODE_OFF:
      break;

    case AGC_MODE_ACQUISITION:
      options.rx_acquisition = AGC_RX_ACQUISITION_LEGACY;
      options.rx_actuation = true;
      break;

    case AGC_MODE_OBSERVE:
      if (options.directions != AGC_DIRECTIONS_TX) {
        options.rx_acquisition = AGC_RX_ACQUISITION_NEW;
        options.rx_tracking = AGC_RX_TRACKING_NEW;
      }
      if (options.directions != AGC_DIRECTIONS_RX)
        options.tx_policy = AGC_TX_POLICY_MANAGED;
      break;

    case AGC_MODE_CONTINUOUS:
      if (request->legacy_requested) {
        options.rx_acquisition = AGC_RX_ACQUISITION_LEGACY;
        options.rx_actuation = true;
      } else if (options.directions != AGC_DIRECTIONS_TX) {
        options.rx_acquisition = AGC_RX_ACQUISITION_NEW;
        options.rx_actuation = true;
      }
      if (options.directions != AGC_DIRECTIONS_TX)
        options.rx_tracking = AGC_RX_TRACKING_NEW;
      if (options.directions != AGC_DIRECTIONS_RX) {
        options.tx_policy = AGC_TX_POLICY_MANAGED;
        options.tx_actuation = true;
      }
      break;
  }

  if (options.tx_power_mode_source != AGC_OPTION_SOURCE_DEFAULT && options.tx_policy != AGC_TX_POLICY_MANAGED)
    return set_error(error,
                     error_size,
                     "tx-power-mode requires managed TX; use agc-mode observe or continuous with agc-directions both or tx");

  *resolved = options;
  return 0;
}

int agc_resolve_ue_cfo(const agc_options_t *options, bool supplied, int requested, int *resolved)
{
  if (options == NULL || resolved == NULL || options->role != AGC_ROLE_UE)
    return -1;
  if (options->mode == AGC_MODE_OFF) {
    *resolved = requested;
    return 0;
  }
  const int selected = supplied ? requested : 1;
  if (selected < 1 || selected > 3)
    return -1;
  *resolved = selected;
  return 0;
}

/* If profile is NULL, inspect only whether an agc-tx-profile field was supplied. */
static int inspect_tx_profile(configmodule_interface_t *cfg, radio_tx_profile_t *profile, bool *supplied)
{
  char *id = NULL, *identity = NULL, *antenna = NULL, *provenance = NULL;
  int qualified = 0, full_scale = 0;
  radio_tx_profile_t p = {0};
  paramdef_t params[] = {
      STRINGPARAM("id", "Local TX profile identifier.\n", 0, &id, ""),
      STRINGPARAM("device", "Exact backend TX connector identity.\n", 0, &identity, ""),
      STRINGPARAM("antenna", "Exact TX antenna/port name.\n", 0, &antenna, ""),
      STRINGPARAM("provenance", "Local qualification evidence identifier.\n", 0, &provenance, ""),
      INTPARAM("qualified", "Explicit qualification for this device and operating range.\n", PARAMFLAG_BOOL, &qualified, 0),
      INTPARAM("component-full-scale", "OAI converter component full scale.\n", 0, &full_scale, 0),
      DOUBLEPARAM("minimum-frequency-hz", "Qualified lower frequency.\n", 0, &p.minimum_frequency_hz, NAN),
      DOUBLEPARAM("maximum-frequency-hz", "Qualified upper frequency.\n", 0, &p.maximum_frequency_hz, NAN),
      DOUBLEPARAM("sample-rate-hz", "Qualified sample rate.\n", 0, &p.sample_rate_hz, NAN),
      DOUBLEPARAM("bandwidth-hz", "Qualified analog/filter bandwidth.\n", 0, &p.bandwidth_hz, NAN),
      DOUBLEPARAM("reported-gain-db", "Fixed analog TX gain readback.\n", 0, &p.reported_gain_db, NAN),
      DOUBLEPARAM("reference-dbm", "Connector power at unit complex RMS/full scale.\n", 0, &p.power.reference_dbm, NAN),
      DOUBLEPARAM("minimum-dbm", "Lowest qualified active channel output.\n", 0, &p.power.minimum_dbm, NAN),
      DOUBLEPARAM("maximum-dbm", "Highest qualified active channel output.\n", 0, &p.power.maximum_dbm, NAN),
      DOUBLEPARAM("uncertainty-db", "Qualification uncertainty in dB.\n", 0, &p.power.uncertainty_db, NAN),
      DOUBLEPARAM("peak-limit-fs", "Maximum digital component magnitude/full scale.\n", 0, &p.peak_limit_fs, 0.7),
      DOUBLEPARAM("quantization-tolerance-db", "Maximum realized digital power error.\n", 0, &p.maximum_quantization_error_db, 0.5),
      DOUBLEPARAM("quantization-evm-limit",
                  "Maximum relative digital quantization EVM (at most0.03).\n",
                  0,
                  &p.maximum_quantization_evm,
                  0.03),
  };
  if (config_get(cfg, params, sizeofArray(params), "agc-tx-profile") < 0)
    return -1;
  if (supplied != NULL) {
    *supplied = false;
    for (unsigned int i = 0; i < sizeofArray(params); ++i)
      if (config_isparamset(params, i)) {
        *supplied = true;
        break;
      }
  }
  if (profile == NULL)
    return 0;
  if (!qualified) {
    *profile = p;
    return 0;
  }
  if (!id || !identity || !antenna || !provenance || strlen(id) >= sizeof(p.id) || strlen(identity) >= sizeof(p.identity)
      || strlen(antenna) >= sizeof(p.antenna) || strlen(provenance) >= sizeof(p.provenance) || full_scale < 1 || full_scale > 32768)
    return -1;
  strcpy(p.id, id);
  strcpy(p.identity, identity);
  strcpy(p.antenna, antenna);
  strcpy(p.provenance, provenance);
  p.component_full_scale = full_scale;
  p.power.qualified = true;
  *profile = p;
  return radio_tx_profile_valid(profile) ? 0 : -1;
}

int agc_start_options(configmodule_interface_t *cfg, int argc, char **argv, agc_role_t role)
{
  char *mode = NULL;
  char *directions = NULL;
  char *tx_power_mode = NULL;
  int legacy = 0;
  int rx_settle_us = AGC_RX_SETTLE_DEFAULT_US;
  paramdef_t options[] = {
      STRINGPARAM("agc-mode", "AGC mode: off, acquisition (UE only), observe, or continuous.\n", 0, &mode, NULL),
      STRINGPARAM("agc-directions", "AGC directions: both (default), rx, or tx.\n", 0, &directions, NULL),
      INTPARAM("agc", "UE-only legacy RX acquisition selector.\n", PARAMFLAG_BOOL, &legacy, 0),
      INTPARAM("agc-rx-settle-us",
               "RX sample exclusion after a radio transaction, microseconds (default 20000).\n",
               0,
               &rx_settle_us,
               AGC_RX_SETTLE_DEFAULT_US),
      STRINGPARAM("tx-power-mode", "TX power mode: absolute (default) or relative.\n", 0, &tx_power_mode, NULL),
  };

  const uint32_t flags = cfg->rtflags;
  cfg->rtflags |= CONFIG_NOEXITONHELP;
  const int result = config_get(cfg, options, sizeofArray(options), NULL);
  cfg->rtflags = flags;
  if (result < 0)
    return -1;
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], "-h") == 0 || strncmp(argv[i], "--help", 6) == 0)
      return 0;

  const agc_option_request_t request = {
      .role = role,
      .mode = mode,
      .directions = directions,
      .tx_power_mode = tx_power_mode,
      .mode_source = string_option_source(mode, is_cli_option_present(argc, argv, "agc-mode")),
      .directions_source = string_option_source(directions, is_cli_option_present(argc, argv, "agc-directions")),
      .tx_power_mode_source = string_option_source(tx_power_mode, is_cli_option_present(argc, argv, "tx-power-mode")),
      .legacy_set = config_isparamset(options, 2),
      .legacy_requested = legacy != 0,
      .legacy_source = parameter_option_source(options, 2, is_cli_option_present(argc, argv, "agc")),
  };
  agc_options_t resolved;
  char error[AGC_OPTION_ERROR_MAX];
  if (agc_resolve_options(&request, &resolved, error, sizeof(error)) != 0) {
    fprintf(stderr, "[AGC] %s\n", error);
    return -1;
  }

  if (rx_settle_us < 1 || rx_settle_us > 1000000) {
    fprintf(stderr, "[AGC] agc-rx-settle-us must be in 1..1000000 microseconds\n");
    return -1;
  }
  resolved.rx_settle_us = rx_settle_us;
  if (resolved.tx_policy == AGC_TX_POLICY_MANAGED) {
    if (resolved.tx_power_mode == AGC_TX_POWER_RELATIVE) {
      bool tx_profile_supplied = false;
      if (inspect_tx_profile(cfg, NULL, &tx_profile_supplied) != 0) {
        fprintf(stderr, "[AGC] cannot inspect agc-tx-profile\n");
        return -1;
      }
      if (tx_profile_supplied) {
        fprintf(stderr, "[AGC] tx-power-mode relative conflicts with an explicitly supplied agc-tx-profile\n");
        return -1;
      }
    } else if (inspect_tx_profile(cfg, &resolved.tx_profile, NULL) != 0) {
      fprintf(stderr, "[AGC] invalid agc-tx-profile; check identity, evidence and numerical bounds\n");
      return -1;
    }
  }
  if (resolved.tx_actuation && resolved.tx_power_mode == AGC_TX_POWER_ABSOLUTE && !radio_tx_profile_valid(&resolved.tx_profile)) {
    fprintf(stderr,
            "[AGC] managed TX requires a qualified agc-tx-profile for the actual device; use observe or directions rx otherwise\n");
    return -1;
  }
  active_options = resolved;
  fprintf(stderr,
          "[AGC] role=%s mode=%s (%s) directions=%s (%s) legacy=%s (%s) rx-acquisition=%s "
          "rx-tracking=%s tx-policy=%s tx-power-mode=%s (%s) rx-actuation=%s tx-actuation=%s rx-settle-us=%u (%s)\n",
          agc_role_name(resolved.role),
          agc_mode_name(resolved.mode),
          agc_option_source_name(resolved.mode_source),
          agc_directions_name(resolved.directions),
          agc_option_source_name(resolved.directions_source),
          resolved.legacy_requested ? "enabled" : "disabled",
          agc_option_source_name(resolved.legacy_source),
          agc_rx_acquisition_name(resolved.rx_acquisition),
          agc_rx_tracking_name(resolved.rx_tracking),
          agc_tx_policy_name(resolved.tx_policy),
          agc_tx_power_mode_name(resolved.tx_power_mode),
          agc_option_source_name(resolved.tx_power_mode_source),
          resolved.rx_actuation ? "yes" : "no",
          resolved.tx_actuation ? "yes" : "no",
          resolved.rx_settle_us,
          agc_option_source_name(parameter_option_source(options, 3, is_cli_option_present(argc, argv, "agc-rx-settle-us"))));
  return 0;
}

const agc_options_t *get_agc_options(void)
{
  return &active_options;
}

const char *agc_role_name(agc_role_t role)
{
  return role == AGC_ROLE_UE ? "ue" : role == AGC_ROLE_GNB ? "gnb" : "invalid";
}

const char *agc_mode_name(agc_mode_t mode)
{
  switch (mode) {
    case AGC_MODE_OFF:
      return "off";
    case AGC_MODE_ACQUISITION:
      return "acquisition";
    case AGC_MODE_OBSERVE:
      return "observe";
    case AGC_MODE_CONTINUOUS:
      return "continuous";
  }
  return "invalid";
}

const char *agc_directions_name(agc_directions_t directions)
{
  switch (directions) {
    case AGC_DIRECTIONS_BOTH:
      return "both";
    case AGC_DIRECTIONS_RX:
      return "rx";
    case AGC_DIRECTIONS_TX:
      return "tx";
  }
  return "invalid";
}

const char *agc_option_source_name(agc_option_source_t source)
{
  switch (source) {
    case AGC_OPTION_SOURCE_DEFAULT:
      return "default";
    case AGC_OPTION_SOURCE_CONFIG:
      return "config";
    case AGC_OPTION_SOURCE_CLI:
      return "cli";
  }
  return "invalid";
}

const char *agc_rx_acquisition_name(agc_rx_acquisition_t acquisition)
{
  switch (acquisition) {
    case AGC_RX_ACQUISITION_CONFIGURED:
      return "configured";
    case AGC_RX_ACQUISITION_LEGACY:
      return "legacy";
    case AGC_RX_ACQUISITION_NEW:
      return "new";
  }
  return "invalid";
}

const char *agc_rx_tracking_name(agc_rx_tracking_t tracking)
{
  switch (tracking) {
    case AGC_RX_TRACKING_HOLD:
      return "hold";
    case AGC_RX_TRACKING_NEW:
      return "new";
  }
  return "invalid";
}

const char *agc_tx_policy_name(agc_tx_policy_t policy)
{
  switch (policy) {
    case AGC_TX_POLICY_BASELINE:
      return "baseline";
    case AGC_TX_POLICY_MANAGED:
      return "managed";
  }
  return "invalid";
}

const char *agc_tx_power_mode_name(agc_tx_power_mode_t mode)
{
  switch (mode) {
    case AGC_TX_POWER_ABSOLUTE:
      return "absolute";
    case AGC_TX_POWER_RELATIVE:
      return "relative";
  }
  return "invalid";
}
