/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "executables/agc_options.h"
#include "executables/flight_options.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

configmodule_interface_t *uniqCfg;

static int failures;

#define CHECK(condition, format, ...)                       \
  do {                                                      \
    if (!(condition)) {                                     \
      fprintf(stderr, "FAIL: " format "\n", ##__VA_ARGS__); \
      ++failures;                                           \
    }                                                       \
  } while (0)

typedef struct {
  const char *name;
  agc_role_t role;
  const char *mode;
  const char *directions;
  bool legacy;
  bool valid;
  agc_rx_acquisition_t rx_acquisition;
  agc_rx_tracking_t rx_tracking;
  agc_tx_policy_t tx_policy;
  bool rx_actuation;
  bool tx_actuation;
} resolver_case_t;

static void check_case(const resolver_case_t *test)
{
  const agc_option_request_t request = {
      .role = test->role,
      .mode = test->mode,
      .directions = test->directions,
      .mode_source = AGC_OPTION_SOURCE_CONFIG,
      .directions_source = AGC_OPTION_SOURCE_CLI,
      .legacy_set = test->legacy,
      .legacy_requested = test->legacy,
      .legacy_source = AGC_OPTION_SOURCE_CONFIG,
  };
  agc_options_t resolved;
  char error[AGC_OPTION_ERROR_MAX];
  const int result = agc_resolve_options(&request, &resolved, error, sizeof(error));
  if (!test->valid) {
    CHECK(result != 0, "%s unexpectedly resolved", test->name);
    return;
  }
  CHECK(result == 0, "%s rejected: %s", test->name, error);
  if (result != 0)
    return;
  CHECK(resolved.rx_acquisition == test->rx_acquisition,
        "%s RX acquisition is %s",
        test->name,
        agc_rx_acquisition_name(resolved.rx_acquisition));
  CHECK(resolved.rx_tracking == test->rx_tracking, "%s RX tracking is %s", test->name, agc_rx_tracking_name(resolved.rx_tracking));
  CHECK(resolved.tx_policy == test->tx_policy, "%s TX policy is %s", test->name, agc_tx_policy_name(resolved.tx_policy));
  CHECK(resolved.rx_actuation == test->rx_actuation, "%s RX actuation mismatch", test->name);
  CHECK(resolved.tx_actuation == test->tx_actuation, "%s TX actuation mismatch", test->name);
}

static void test_default(void)
{
  const agc_option_request_t request = {.role = AGC_ROLE_UE};
  agc_options_t resolved;
  char error[AGC_OPTION_ERROR_MAX];
  CHECK(agc_resolve_options(&request, &resolved, error, sizeof(error)) == 0, "default rejected: %s", error);
  CHECK(resolved.mode == AGC_MODE_OFF, "default mode is %s", agc_mode_name(resolved.mode));
  CHECK(resolved.directions == AGC_DIRECTIONS_BOTH, "default directions are %s", agc_directions_name(resolved.directions));
  CHECK(resolved.mode_source == AGC_OPTION_SOURCE_DEFAULT, "default mode provenance changed");
  CHECK(resolved.directions_source == AGC_OPTION_SOURCE_DEFAULT, "default directions provenance changed");
  CHECK(resolved.tx_power_mode == AGC_TX_POWER_ABSOLUTE,
        "default TX power mode is %s",
        agc_tx_power_mode_name(resolved.tx_power_mode));
  CHECK(resolved.tx_power_mode_source == AGC_OPTION_SOURCE_DEFAULT, "default TX power mode provenance changed");
  CHECK(!resolved.rx_actuation && !resolved.tx_actuation, "default enables actuation");
}

static void test_cfo_resolution(void)
{
  agc_options_t options = {.role = AGC_ROLE_UE, .mode = AGC_MODE_OFF};
  int selected = -1;
  CHECK(agc_resolve_ue_cfo(&options, false, 0, &selected) == 0 && selected == 0, "off CFO default changed");
  for (agc_mode_t mode = AGC_MODE_ACQUISITION; mode <= AGC_MODE_CONTINUOUS; ++mode) {
    options.mode = mode;
    CHECK(agc_resolve_ue_cfo(&options, false, 0, &selected) == 0 && selected == 1, "managed CFO default missing");
    for (int value = 0; value <= 4; ++value) {
      const int result = agc_resolve_ue_cfo(&options, true, value, &selected);
      CHECK(value >= 1 && value <= 3 ? result == 0 && selected == value : result != 0,
            "explicit CFO %d incorrectly resolved for mode %d",
            value,
            mode);
    }
  }
}

static void test_resolver_matrix(void)
{
  static const resolver_case_t cases[] = {
      {"ue-off-both",
       AGC_ROLE_UE,
       "off",
       "both",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       false,
       false},
      {"ue-off-rx",
       AGC_ROLE_UE,
       "off",
       "rx",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       false,
       false},
      {"ue-off-tx",
       AGC_ROLE_UE,
       "off",
       "tx",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       false,
       false},
      {"ue-off-both-legacy", AGC_ROLE_UE, "off", "both", true, false},
      {"ue-off-rx-legacy", AGC_ROLE_UE, "off", "rx", true, false},
      {"ue-off-tx-legacy", AGC_ROLE_UE, "off", "tx", true, false},
      {"ue-acquisition-both",
       AGC_ROLE_UE,
       "acquisition",
       "both",
       false,
       true,
       AGC_RX_ACQUISITION_LEGACY,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       true,
       false},
      {"ue-acquisition-rx",
       AGC_ROLE_UE,
       "acquisition",
       "rx",
       false,
       true,
       AGC_RX_ACQUISITION_LEGACY,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       true,
       false},
      {"ue-acquisition-tx", AGC_ROLE_UE, "acquisition", "tx", false, false},
      {"ue-acquisition-both-legacy",
       AGC_ROLE_UE,
       "acquisition",
       "both",
       true,
       true,
       AGC_RX_ACQUISITION_LEGACY,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       true,
       false},
      {"ue-acquisition-rx-legacy",
       AGC_ROLE_UE,
       "acquisition",
       "rx",
       true,
       true,
       AGC_RX_ACQUISITION_LEGACY,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       true,
       false},
      {"ue-acquisition-tx-legacy", AGC_ROLE_UE, "acquisition", "tx", true, false},
      {"ue-observe-both",
       AGC_ROLE_UE,
       "observe",
       "both",
       false,
       true,
       AGC_RX_ACQUISITION_NEW,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_MANAGED,
       false,
       false},
      {"ue-observe-rx",
       AGC_ROLE_UE,
       "observe",
       "rx",
       false,
       true,
       AGC_RX_ACQUISITION_NEW,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_BASELINE,
       false,
       false},
      {"ue-observe-tx",
       AGC_ROLE_UE,
       "observe",
       "tx",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_MANAGED,
       false,
       false},
      {"ue-observe-both-legacy", AGC_ROLE_UE, "observe", "both", true, false},
      {"ue-observe-rx-legacy", AGC_ROLE_UE, "observe", "rx", true, false},
      {"ue-observe-tx-legacy", AGC_ROLE_UE, "observe", "tx", true, false},
      {"ue-continuous-both",
       AGC_ROLE_UE,
       "continuous",
       "both",
       false,
       true,
       AGC_RX_ACQUISITION_NEW,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_MANAGED,
       true,
       true},
      {"ue-continuous-rx",
       AGC_ROLE_UE,
       "continuous",
       "rx",
       false,
       true,
       AGC_RX_ACQUISITION_NEW,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_BASELINE,
       true,
       false},
      {"ue-continuous-tx",
       AGC_ROLE_UE,
       "continuous",
       "tx",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_MANAGED,
       false,
       true},
      {"ue-continuous-both-legacy",
       AGC_ROLE_UE,
       "continuous",
       "both",
       true,
       true,
       AGC_RX_ACQUISITION_LEGACY,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_MANAGED,
       true,
       true},
      {"ue-continuous-rx-legacy",
       AGC_ROLE_UE,
       "continuous",
       "rx",
       true,
       true,
       AGC_RX_ACQUISITION_LEGACY,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_BASELINE,
       true,
       false},
      {"ue-continuous-tx-legacy",
       AGC_ROLE_UE,
       "continuous",
       "tx",
       true,
       true,
       AGC_RX_ACQUISITION_LEGACY,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_MANAGED,
       true,
       true},
      {"gnb-off-both",
       AGC_ROLE_GNB,
       "off",
       "both",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       false,
       false},
      {"gnb-off-rx",
       AGC_ROLE_GNB,
       "off",
       "rx",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       false,
       false},
      {"gnb-off-tx",
       AGC_ROLE_GNB,
       "off",
       "tx",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_BASELINE,
       false,
       false},
      {"gnb-off-both-legacy", AGC_ROLE_GNB, "off", "both", true, false},
      {"gnb-off-rx-legacy", AGC_ROLE_GNB, "off", "rx", true, false},
      {"gnb-off-tx-legacy", AGC_ROLE_GNB, "off", "tx", true, false},
      {"gnb-acquisition-both", AGC_ROLE_GNB, "acquisition", "both", false, false},
      {"gnb-acquisition-rx", AGC_ROLE_GNB, "acquisition", "rx", false, false},
      {"gnb-acquisition-tx", AGC_ROLE_GNB, "acquisition", "tx", false, false},
      {"gnb-acquisition-both-legacy", AGC_ROLE_GNB, "acquisition", "both", true, false},
      {"gnb-acquisition-rx-legacy", AGC_ROLE_GNB, "acquisition", "rx", true, false},
      {"gnb-acquisition-tx-legacy", AGC_ROLE_GNB, "acquisition", "tx", true, false},
      {"gnb-observe-both",
       AGC_ROLE_GNB,
       "observe",
       "both",
       false,
       true,
       AGC_RX_ACQUISITION_NEW,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_MANAGED,
       false,
       false},
      {"gnb-observe-rx",
       AGC_ROLE_GNB,
       "observe",
       "rx",
       false,
       true,
       AGC_RX_ACQUISITION_NEW,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_BASELINE,
       false,
       false},
      {"gnb-observe-tx",
       AGC_ROLE_GNB,
       "observe",
       "tx",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_MANAGED,
       false,
       false},
      {"gnb-observe-both-legacy", AGC_ROLE_GNB, "observe", "both", true, false},
      {"gnb-observe-rx-legacy", AGC_ROLE_GNB, "observe", "rx", true, false},
      {"gnb-observe-tx-legacy", AGC_ROLE_GNB, "observe", "tx", true, false},
      {"gnb-continuous-both",
       AGC_ROLE_GNB,
       "continuous",
       "both",
       false,
       true,
       AGC_RX_ACQUISITION_NEW,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_MANAGED,
       true,
       true},
      {"gnb-continuous-rx",
       AGC_ROLE_GNB,
       "continuous",
       "rx",
       false,
       true,
       AGC_RX_ACQUISITION_NEW,
       AGC_RX_TRACKING_NEW,
       AGC_TX_POLICY_BASELINE,
       true,
       false},
      {"gnb-continuous-tx",
       AGC_ROLE_GNB,
       "continuous",
       "tx",
       false,
       true,
       AGC_RX_ACQUISITION_CONFIGURED,
       AGC_RX_TRACKING_HOLD,
       AGC_TX_POLICY_MANAGED,
       false,
       true},
      {"gnb-continuous-both-legacy", AGC_ROLE_GNB, "continuous", "both", true, false},
      {"gnb-continuous-rx-legacy", AGC_ROLE_GNB, "continuous", "rx", true, false},
      {"gnb-continuous-tx-legacy", AGC_ROLE_GNB, "continuous", "tx", true, false},
  };
  for (unsigned int i = 0; i < sizeofArray(cases); ++i)
    check_case(&cases[i]);
}

static void check_tx_power_selection_error(const char *name, const char *mode, const char *directions)
{
  const agc_option_request_t request = {
      .role = AGC_ROLE_UE,
      .mode = mode,
      .directions = directions,
      .tx_power_mode = "relative",
      .mode_source = AGC_OPTION_SOURCE_CONFIG,
      .directions_source = AGC_OPTION_SOURCE_CONFIG,
      .tx_power_mode_source = AGC_OPTION_SOURCE_CONFIG,
  };
  agc_options_t resolved;
  char error[AGC_OPTION_ERROR_MAX];
  CHECK(agc_resolve_options(&request, &resolved, error, sizeof(error)) != 0,
        "%s accepted a TX power mode without managed TX",
        name);
}

static void test_invalid_values(void)
{
  const agc_option_request_t bad_mode = {.role = AGC_ROLE_UE, .mode = "adaptive", .mode_source = AGC_OPTION_SOURCE_CONFIG};
  const agc_option_request_t bad_directions = {.role = AGC_ROLE_UE,
                                               .directions = "uplink",
                                               .directions_source = AGC_OPTION_SOURCE_CLI};
  const agc_option_request_t bad_role = {.role = (agc_role_t)99};
  const agc_option_request_t gnb_legacy_disabled = {.role = AGC_ROLE_GNB,
                                                    .legacy_set = true,
                                                    .legacy_source = AGC_OPTION_SOURCE_CONFIG};
  const agc_option_request_t bad_tx_power = {.role = AGC_ROLE_UE,
                                             .tx_power_mode = "nominal",
                                             .tx_power_mode_source = AGC_OPTION_SOURCE_CONFIG};
  agc_options_t resolved;
  char error[AGC_OPTION_ERROR_MAX];
  CHECK(agc_resolve_options(&bad_mode, &resolved, error, sizeof(error)) != 0, "invalid mode accepted");
  CHECK(agc_resolve_options(&bad_directions, &resolved, error, sizeof(error)) != 0, "invalid directions accepted");
  CHECK(agc_resolve_options(&bad_role, &resolved, error, sizeof(error)) != 0, "invalid role accepted");
  CHECK(agc_resolve_options(&gnb_legacy_disabled, &resolved, error, sizeof(error)) != 0, "gNB accepted an explicit legacy option");
  CHECK(agc_resolve_options(&bad_tx_power, &resolved, error, sizeof(error)) != 0, "invalid TX power mode accepted");
  check_tx_power_selection_error("off", "off", "both");
  check_tx_power_selection_error("acquisition", "acquisition", "both");
  check_tx_power_selection_error("directions-rx", "continuous", "rx");
}

static void test_argument_retention(void)
{
  int argc = 7;
  char *argv[] = {"test_agc_options", "--flight", "log", "--agc-mode", "continuous", "--agc-directions", "tx", NULL};
  CHECK(flight_normalize_arguments(&argc, argv) == 0, "flight normalization rejected retained AGC arguments");
  CHECK(argc == 7, "flight normalization changed AGC argument count to %d", argc);
  CHECK(strcmp(argv[3], "--agc-mode") == 0 && strcmp(argv[4], "continuous") == 0,
        "flight normalization did not retain agc-mode arguments");
  CHECK(strcmp(argv[5], "--agc-directions") == 0 && strcmp(argv[6], "tx") == 0,
        "flight normalization did not retain agc-directions arguments");
  free(argv[2]);
}

static int check_configuration_case(const char *name, int argc, char **argv)
{
  uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY | CONFIG_NO_CMDLINE_ECHO);
  if (uniqCfg == NULL)
    return 1;
  const agc_role_t role = strcmp(name, "gnb-legacy") == 0 ? AGC_ROLE_GNB : AGC_ROLE_UE;
  const int result = agc_start_options(uniqCfg, argc, argv, role);
  if (strcmp(name, "help") == 0)
    return result == 0 ? 0 : 1;
  if (strcmp(name, "gnb-legacy") == 0 || strcmp(name, "absolute-invalid-profile") == 0
      || strcmp(name, "relative-profile-conflict") == 0 || strcmp(name, "tx-power-mode-typo") == 0
      || strcmp(name, "tx-power-mode-off") == 0)
    return result != 0 ? 0 : 1;
  if (result != 0)
    return 1;
  const agc_options_t *options = get_agc_options();
  int cfo = 0, selected = -1;
  paramdef_t params[] = {INTPARAM("cont-fo-comp", "test CFO mode", 0, &cfo, 0)};
  if (config_get(uniqCfg, params, sizeofArray(params), NULL) < 0)
    return 1;
  const bool supplied = config_isparamset(params, 0);
  const int cfo_result = agc_resolve_ue_cfo(options, supplied, cfo, &selected);
  if (strcmp(name, "cfo-config-zero") == 0 || strcmp(name, "cfo-cli-zero") == 0)
    return supplied && cfo == 0 && cfo_result != 0 ? 0 : 1;
  if (strncmp(name, "cfo-cli-", 8) == 0) {
    const int expected = name[8] - '0';
    return supplied && cfo_result == 0 && selected == expected ? 0 : 1;
  }
  if (supplied || cfo_result != 0 || selected != (options->mode == AGC_MODE_OFF ? 0 : 1))
    return 1;
  if (strcmp(name, "config-only") == 0)
    return options->mode == AGC_MODE_ACQUISITION && options->mode_source == AGC_OPTION_SOURCE_CONFIG && options->legacy_requested
                   && options->legacy_source == AGC_OPTION_SOURCE_CONFIG
               ? 0
               : 1;
  if (strcmp(name, "config-cli") == 0)
    return options->mode == AGC_MODE_CONTINUOUS && options->mode_source == AGC_OPTION_SOURCE_CLI
                   && options->directions == AGC_DIRECTIONS_TX && options->directions_source == AGC_OPTION_SOURCE_CLI
                   && options->legacy_requested && options->legacy_source == AGC_OPTION_SOURCE_CONFIG
                   && options->rx_acquisition == AGC_RX_ACQUISITION_LEGACY && options->rx_tracking == AGC_RX_TRACKING_HOLD
                   && options->tx_policy == AGC_TX_POLICY_MANAGED
               ? 0
               : 1;
  if (strcmp(name, "legacy-cli") == 0)
    return options->mode == AGC_MODE_ACQUISITION && options->mode_source == AGC_OPTION_SOURCE_CLI && options->legacy_requested
                   && options->legacy_source == AGC_OPTION_SOURCE_CLI && options->rx_acquisition == AGC_RX_ACQUISITION_LEGACY
               ? 0
               : 1;
  if (strcmp(name, "default") == 0)
    return options->mode == AGC_MODE_OFF && options->mode_source == AGC_OPTION_SOURCE_DEFAULT
                   && options->directions == AGC_DIRECTIONS_BOTH && options->tx_power_mode == AGC_TX_POWER_ABSOLUTE
                   && options->tx_power_mode_source == AGC_OPTION_SOURCE_DEFAULT
               ? 0
               : 1;
  if (strcmp(name, "relative-config") == 0)
    return options->mode == AGC_MODE_OBSERVE && options->directions == AGC_DIRECTIONS_TX
                   && options->tx_power_mode == AGC_TX_POWER_RELATIVE && options->tx_power_mode_source == AGC_OPTION_SOURCE_CONFIG
                   && !options->tx_actuation
               ? 0
               : 1;
  if (strcmp(name, "relative-cli") == 0)
    return options->mode == AGC_MODE_OBSERVE && options->directions == AGC_DIRECTIONS_TX
                   && options->tx_power_mode == AGC_TX_POWER_RELATIVE && options->tx_power_mode_source == AGC_OPTION_SOURCE_CLI
                   && !options->tx_actuation
               ? 0
               : 1;
  if (strcmp(name, "relative-continuous") == 0)
    return options->mode == AGC_MODE_CONTINUOUS && options->directions == AGC_DIRECTIONS_TX
                   && options->tx_power_mode == AGC_TX_POWER_RELATIVE && options->tx_power_mode_source == AGC_OPTION_SOURCE_CLI
                   && options->tx_actuation
               ? 0
               : 1;
  if (strcmp(name, "observe-absolute") == 0)
    return options->mode == AGC_MODE_OBSERVE && options->directions == AGC_DIRECTIONS_TX
                   && options->tx_power_mode == AGC_TX_POWER_ABSOLUTE && options->tx_power_mode_source == AGC_OPTION_SOURCE_DEFAULT
                   && !options->tx_actuation
               ? 0
               : 1;
  return 1;
}

static int run_child(const char *self, const char *name, const char *config, bool cli_override)
{
  const pid_t child = fork();
  if (child < 0)
    return -1;
  if (child == 0) {
    if (setenv("OAI_AGC_OPTIONS_TEST_CASE", name, 1) != 0)
      _exit(127);
    if (strcmp(name, "help") == 0)
      execl(self, self, "--help", (char *)NULL);
    else if (strcmp(name, "legacy-cli") == 0)
      execl(self, self, "--agc", (char *)NULL);
    else if (strcmp(name, "cfo-cli-zero") == 0)
      execl(self, self, "-O", config, "--cont-fo-comp", "0", (char *)NULL);
    else if (strncmp(name, "cfo-cli-", 8) == 0)
      execl(self, self, "-O", config, "--cont-fo-comp", name + 8, (char *)NULL);
    else if (strcmp(name, "relative-cli") == 0)
      execl(self, self, "-O", config, "--tx-power-mode", "relative", (char *)NULL);
    else if (strcmp(name, "relative-profile-conflict") == 0)
      execl(self,
            self,
            "-O",
            config,
            "--agc-mode",
            "continuous",
            "--agc-directions",
            "tx",
            "--tx-power-mode",
            "relative",
            (char *)NULL);
    else if (strcmp(name, "relative-continuous") == 0)
      execl(self, self, "--agc-mode", "continuous", "--agc-directions", "tx", "--tx-power-mode", "relative", (char *)NULL);
    else if (strcmp(name, "observe-absolute") == 0)
      execl(self, self, "--agc-mode", "observe", "--agc-directions", "tx", (char *)NULL);
    else if (strcmp(name, "tx-power-mode-typo") == 0)
      execl(self, self, "--agc-mode", "observe", "--agc-directions", "tx", "--tx-power-mode", "nominal", (char *)NULL);
    else if (strcmp(name, "tx-power-mode-off") == 0)
      execl(self, self, "--tx-power-mode", "relative", (char *)NULL);
    else if (config != NULL && cli_override)
      execl(self, self, "-O", config, "--agc-mode", "continuous", "--agc-directions", "tx", (char *)NULL);
    else if (config != NULL)
      execl(self, self, "-O", config, (char *)NULL);
    else
      execl(self, self, (char *)NULL);
    _exit(127);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void test_configuration_integration(const char *self)
{
  char config_only[] = "/tmp/oai-agc-options-config-only-XXXXXX";
  char config_cli[] = "/tmp/oai-agc-options-config-cli-XXXXXX";
  char config_relative[] = "/tmp/oai-agc-options-config-relative-XXXXXX";
  const int only_fd = mkstemp(config_only);
  const int cli_fd = mkstemp(config_cli);
  const int relative_fd = mkstemp(config_relative);
  CHECK(only_fd >= 0 && cli_fd >= 0 && relative_fd >= 0, "could not create temporary configuration files");
  if (only_fd < 0 || cli_fd < 0 || relative_fd < 0) {
    if (only_fd >= 0) {
      close(only_fd);
      unlink(config_only);
    }
    if (cli_fd >= 0) {
      close(cli_fd);
      unlink(config_cli);
    }
    if (relative_fd >= 0) {
      close(relative_fd);
      unlink(config_relative);
    }
    return;
  }
  static const char only_contents[] = "agc = 1;\n";
  static const char cli_contents[] =
      "agc-mode = \"observe\";\nagc-directions = \"rx\";\nagc = 1;\n"
      "agc-tx-profile = { id = \"test-profile\"; device = \"fake:0\"; antenna = \"TX/RX\"; "
      "provenance = \"synthetic-test-only\"; qualified = 1; component-full-scale = 2048; "
      "minimum-frequency-hz = 710749000.0; maximum-frequency-hz = 710751000.0; "
      "sample-rate-hz = 7680000.0; bandwidth-hz = 20000000.0; reported-gain-db = 60.75; reference-dbm = 0.0; "
      "minimum-dbm = -30.0; maximum-dbm = 0.0; uncertainty-db = 1.0; };\n";
  static const char relative_contents[] =
      "agc-mode = \042observe\042;\nagc-directions = \042tx\042;\ntx-power-mode = \042relative\042;\n";
  CHECK(write(only_fd, only_contents, sizeof(only_contents) - 1) == (ssize_t)(sizeof(only_contents) - 1),
        "could not write config-only fixture");
  CHECK(write(cli_fd, cli_contents, sizeof(cli_contents) - 1) == (ssize_t)(sizeof(cli_contents) - 1),
        "could not write config-cli fixture");
  CHECK(write(relative_fd, relative_contents, sizeof(relative_contents) - 1) == (ssize_t)(sizeof(relative_contents) - 1),
        "could not write relative configuration fixture");
  close(only_fd);
  close(cli_fd);
  close(relative_fd);
  CHECK(run_child(self, "default", NULL, false) == 0, "default config-library integration failed");
  CHECK(run_child(self, "config-only", config_only, false) == 0, "config-only legacy integration failed");
  CHECK(run_child(self, "config-cli", config_cli, true) == 0, "config/CLI precedence integration failed");
  CHECK(run_child(self, "absolute-invalid-profile", config_only, true) == 0, "missing managed absolute TX profile was accepted");
  CHECK(run_child(self, "relative-config", config_relative, false) == 0, "relative configuration integration failed");
  CHECK(run_child(self, "relative-cli", config_relative, false) == 0, "relative CLI integration failed");
  CHECK(run_child(self, "relative-profile-conflict", config_cli, false) == 0, "relative/profile conflict was accepted");
  CHECK(run_child(self, "relative-continuous", NULL, false) == 0, "relative continuous mode without a profile was rejected");
  CHECK(run_child(self, "observe-absolute", NULL, false) == 0, "absolute observe mode without a profile was rejected");
  CHECK(run_child(self, "tx-power-mode-typo", NULL, false) == 0, "invalid TX power mode was accepted");
  CHECK(run_child(self, "tx-power-mode-off", NULL, false) == 0, "TX power mode was accepted while AGC was off");
  CHECK(run_child(self, "legacy-cli", NULL, false) == 0, "legacy CLI integration failed");
  CHECK(run_child(self, "gnb-legacy", config_only, false) == 0, "gNB legacy configuration was accepted");
  CHECK(run_child(self, "help", NULL, false) == 0, "help integration failed");
  CHECK(run_child(self, "cfo-cli-zero", config_only, false) == 0, "explicit CLI CFO zero accepted");
  FILE *cfo_fixture = fopen(config_only, "a");
  CHECK(cfo_fixture != NULL, "could not extend CFO fixture");
  if (cfo_fixture != NULL) {
    CHECK(fputs("cont-fo-comp = 0;\n", cfo_fixture) >= 0 && fclose(cfo_fixture) == 0, "could not write CFO fixture");
    CHECK(run_child(self, "cfo-config-zero", config_only, false) == 0, "explicit config CFO zero accepted");
    CHECK(run_child(self, "cfo-cli-1", config_only, false) == 0, "CLI CFO one did not override config zero");
    CHECK(run_child(self, "cfo-cli-2", config_only, false) == 0, "CLI CFO two did not override config zero");
    CHECK(run_child(self, "cfo-cli-3", config_only, false) == 0, "CLI CFO three did not override config zero");
  }
  unlink(config_only);
  unlink(config_cli);
  unlink(config_relative);
}

void exit_function(const char *file, const char *function, const int line, const char *message, const int assertion)
{
  (void)file;
  (void)function;
  (void)line;
  (void)message;
  exit(assertion ? EXIT_FAILURE : EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
  const char *configuration_case = getenv("OAI_AGC_OPTIONS_TEST_CASE");
  if (configuration_case != NULL)
    return check_configuration_case(configuration_case, argc, argv);

  test_default();
  test_cfo_resolution();
  test_resolver_matrix();
  test_invalid_values();
  test_argument_retention();
  test_configuration_integration(argv[0]);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
