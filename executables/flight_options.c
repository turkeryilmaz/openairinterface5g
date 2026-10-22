/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#define _GNU_SOURCE
#include "flight_options.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef OAI_FLIGHT_SOURCE_DIR
#define OAI_FLIGHT_SOURCE_DIR "."
#endif

int flight_normalize_arguments(int *argc, char **argv)
{
  int output = 1;
  bool seen = false;
  for (int i = 1; i < *argc; ++i) {
    if (strcmp(argv[i], "--flight") != 0) {
      argv[output++] = argv[i];
      continue;
    }
    if (seen || i + 1 == *argc || argv[i + 1][0] == '-') {
      fprintf(stderr, "[FLIGHT] Use one --flight followed by log [recovery] [traffic] or off\n");
      return -1;
    }
    seen = true;
    argv[output++] = argv[i];
    int end = i + 1;
    size_t length = 0;
    while (end < *argc && argv[end][0] != '-') {
      length += strlen(argv[end]) + 1;
      ++end;
    }
    char *features = calloc(length, 1);
    if (features == NULL)
      return -1;
    for (int j = i + 1; j < end; ++j) {
      if (j != i + 1)
        strcat(features, " ");
      strcat(features, argv[j]);
    }
    /* argv/config retain this one startup allocation for the process lifetime. */
    argv[output++] = features;
    i = end - 1;
  }
  argv[output] = NULL;
  *argc = output;
  return 0;
}

static bool find_repo_above(const char *start, char *repo)
{
  char directory[PATH_MAX];
  if (realpath(start, directory) == NULL)
    return false;
  for (;;) {
    char candidate[PATH_MAX];
    int size = snprintf(candidate, sizeof(candidate), "%s/tools/flight_test/capture.py", directory);
    if (size > 0 && (size_t)size < sizeof(candidate) && access(candidate, R_OK) == 0) {
      strcpy(repo, directory);
      return true;
    }
    char *slash = strrchr(directory, '/');
    if (slash == NULL || slash == directory)
      return false;
    *slash = '\0';
  }
}

static bool resolve_executable(const char *name, char *executable)
{
  if (strchr(name, '/') != NULL)
    return realpath(name, executable) != NULL;
  const char *path = getenv("PATH");
  char *copy = strdup(path ? path : "");
  if (copy == NULL)
    return false;
  bool found = false;
  char *context = NULL;
  for (char *dir = strtok_r(copy, ":", &context); dir; dir = strtok_r(NULL, ":", &context)) {
    char candidate[PATH_MAX];
    int size = snprintf(candidate, sizeof(candidate), "%s/%s", dir, name);
    if (size > 0 && (size_t)size < sizeof(candidate) && access(candidate, X_OK) == 0 && realpath(candidate, executable)) {
      found = true;
      break;
    }
  }
  free(copy);
  return found;
}

int flight_start_capture(configmodule_interface_t *cfg, int argc, char **argv, const char *role)
{
  char *features = NULL;
  char *output = NULL, *repo_option = NULL, *core = NULL, *interface = NULL, *gpsd = NULL;
  char *recorder_budget = NULL, *recorder_detail = NULL, *stdout_budget = NULL, *host_budget = NULL, *min_free = NULL;
  char *stall = NULL, *attempt = NULL;
  char *traffic_peer = NULL, *traffic_bind = NULL, *traffic_port = NULL, *traffic_direction = NULL, *traffic_rate = NULL;
  char *capability_xml = NULL;
  paramdef_t options[] = {
      STRINGPARAM("flight", "Flight features: log [recovery] [traffic], or off. CLI replaces config.\n", 0, &features, "off"),
      STRINGPARAM("flight-traffic-peer", "Required UE traffic destination IP.\n", 0, &traffic_peer, NULL),
      STRINGPARAM("flight-traffic-bind", "Optional gNB traffic server bind IP.\n", 0, &traffic_bind, NULL),
      STRINGPARAM("flight-traffic-port", "iperf3 server port (default 5201).\n", 0, &traffic_port, NULL),
      STRINGPARAM("flight-traffic-direction", "UDP traffic ul, dl, or both (default ul).\n", 0, &traffic_direction, NULL),
      STRINGPARAM("flight-traffic-rate",
                  "UE UDP rate per direction: 500K, 1.1M, 1.1Mbps or integer bit/s (default 1.1M).\n",
                  0,
                  &traffic_rate,
                  NULL),
      STRINGPARAM("uecap_file", "UE capabilities XER file; flight checks XML syntax.\n", 0, &capability_xml, NULL),
      STRINGPARAM("flight-recovery-stall", "Native RX progress timeout in seconds (default 10).\n", 0, &stall, NULL),
      STRINGPARAM("flight-recovery-attempt", "Cell-present acquisition bound in seconds (default 120).\n", 0, &attempt, NULL),
      STRINGPARAM("flight-output", "Capture directory; default: cmake_targets/log/FlightTests/YYYY-MM-DD.\n", 0, &output, NULL),
      STRINGPARAM("flight-repo", "Optional relocated source checkout containing tools/flight_test.\n", 0, &repo_option, NULL),
      STRINGPARAM("flight-core-ip", "Optional remote endpoint observation; not required for logging.\n", 0, &core, NULL),
      STRINGPARAM("flight-interface", "Observed UE interface (default oaitun_ue1).\n", 0, &interface, NULL),
      STRINGPARAM("flight-gpsd", "Optional GPSD endpoint HOST:PORT.\n", 0, &gpsd, NULL),
      STRINGPARAM("flight-min-free-bytes", "Free-space reserve in bytes (default 512 MiB).\n", 0, &min_free, NULL),
      STRINGPARAM("flight-recorder-detail",
                  "Numeric detail: balanced (default) or full; fault detail is not sampled.\n",
                  0,
                  &recorder_detail,
                  NULL),
      STRINGPARAM("flight-recorder-budget",
                  "Numeric recorder byte limit; 0 (default) retains until the space reserve.\n",
                  0,
                  &recorder_budget,
                  NULL),
      STRINGPARAM("flight-stdout-budget", "Combined stdout/stderr byte limit.\n", 0, &stdout_budget, NULL),
      STRINGPARAM("flight-host-budget", "Host observation byte limit.\n", 0, &host_budget, NULL),
  };
  const uint32_t flags = cfg->rtflags;
  cfg->rtflags |= CONFIG_NOEXITONHELP;
  int result = config_get(cfg, options, sizeofArray(options), NULL);
  cfg->rtflags = flags;
  if (result < 0)
    return -1;
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], "-h") == 0 || strncmp(argv[i], "--help", 6) == 0)
      return 0;

  bool log = false, off = false, recovery = false, traffic = false;
  char *copy = strdup(features ? features : "off");
  if (copy == NULL)
    return -1;
  char *context = NULL;
  for (char *feature = strtok_r(copy, " ,\t\r\n", &context); feature; feature = strtok_r(NULL, " ,\t\r\n", &context)) {
    if (strcmp(feature, "log") == 0)
      log = true;
    else if (strcmp(feature, "off") == 0)
      off = true;
    else if (strcmp(feature, "recovery") == 0)
      recovery = true;
    else if (strcmp(feature, "traffic") == 0)
      traffic = true;
    else {
      fprintf(stderr, "[FLIGHT] Unsupported feature '%s'; supported: log, recovery, traffic, off.\n", feature);
      free(copy);
      return -1;
    }
  }
  free(copy);
  if (log == off || ((recovery || traffic) && !log)) {
    fprintf(stderr, "[FLIGHT] Select log [recovery] [traffic] for UE or gNB, or off\n");
    return -1;
  }
  if (off)
    return 0;

  /* Only the collector that directly launched this worker may suppress bootstrap. */
  const char *parent = getenv("_OAI_FLIGHT_CAPTURE_PARENT");
  char expected_parent[32];
  snprintf(expected_parent, sizeof(expected_parent), "%ld", (long)getppid());
  if (parent && strcmp(parent, expected_parent) == 0)
    return 0;

  char executable[PATH_MAX], repo[PATH_MAX], script[PATH_MAX], cwd[PATH_MAX], config[PATH_MAX];
  if (!resolve_executable(argv[0], executable) || getcwd(cwd, sizeof(cwd)) == NULL) {
    fprintf(stderr, "[FLIGHT] Cannot resolve executable or launch directory\n");
    return -1;
  }
  bool found = repo_option ? find_repo_above(repo_option, repo)
                           : (find_repo_above(executable, repo) || find_repo_above(cwd, repo)
                              || find_repo_above(OAI_FLIGHT_SOURCE_DIR, repo));
  int size = found ? snprintf(script, sizeof(script), "%s/tools/flight_test/capture.py", repo) : -1;
  if (size < 0 || (size_t)size >= sizeof(script)) {
    fprintf(stderr, "[FLIGHT] Cannot locate tools/flight_test/capture.py; keep the source checkout or set flight-repo\n");
    return -1;
  }
  const char *pairs[][2] = {{"--output", output},
                            {"--core-ip", core},
                            {"--interface", interface},
                            {"--gpsd", gpsd},
                            {"--recorder-budget", recorder_budget},
                            {"--recorder-detail", recorder_detail},
                            {"--stdout-budget", stdout_budget},
                            {"--host-budget", host_budget},
                            {"--min-free-bytes", min_free},
                            {"--recovery-stall", stall},
                            {"--recovery-attempt", attempt},
                            {"--traffic-peer", traffic_peer},
                            {"--traffic-bind", traffic_bind},
                            {"--traffic-port", traffic_port},
                            {"--traffic-direction", traffic_direction},
                            {"--traffic-rate", traffic_rate},
                            {"--capability-xml", strcmp(role, "ue") == 0 ? capability_xml : NULL}};
  /* Fixed bootstrap arguments, optional pairs, original argv and the NULL end. */
  const size_t command_capacity = (size_t)argc + 24 + 2 * sizeofArray(pairs);
  char **command = calloc(command_capacity, sizeof(*command));
  if (command == NULL)
    return -1;
  size_t n = 0;
  command[n++] = "python3";
  command[n++] = "-B";
  command[n++] = script;
  command[n++] = "--role";
  command[n++] = (char *)role;
  command[n++] = "--repo";
  command[n++] = repo;
  command[n++] = "--console";
  if (recovery)
    command[n++] = "--recovery";
  if (traffic)
    command[n++] = "--traffic";
  command[n++] = "--working-directory";
  command[n++] = cwd;
  if (cfg->num_cfgP > 0 && cfg->cfgP[0] && realpath(cfg->cfgP[0], config)) {
    command[n++] = "--config";
    command[n++] = config;
    command[n++] = "--config-backend";
    command[n++] = cfg->cfgmode ? cfg->cfgmode : "unknown";
  }
  for (unsigned int i = 0; i < sizeofArray(pairs); ++i) {
    if (pairs[i][1]) {
      command[n++] = (char *)pairs[i][0];
      command[n++] = (char *)pairs[i][1];
    }
  }
  command[n++] = "--";
  command[n++] = executable;
  for (int i = 1; i < argc; ++i)
    command[n++] = argv[i];
  fflush(NULL);
  execvp(command[0], command);
  fprintf(stderr, "[FLIGHT] Cannot start bundled capture collector (python3 required): %s\n", strerror(errno));
  free(command);
  return -1;
}
