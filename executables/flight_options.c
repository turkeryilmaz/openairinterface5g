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
      fprintf(stderr, "[FLIGHT] Use one --flight followed by log or off\n");
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
  char *recorder_budget = NULL, *stdout_budget = NULL, *host_budget = NULL, *min_free = NULL;
  paramdef_t options[] = {
      STRINGPARAM("flight", "Flight features: log, or off. CLI replaces the config value.\n", 0, &features, "off"),
      STRINGPARAM("flight-output", "Capture directory; default: cmake_targets/log/FlightTests/YYYY-MM-DD.\n", 0, &output, NULL),
      STRINGPARAM("flight-repo", "Optional relocated source checkout containing tools/flight_test.\n", 0, &repo_option, NULL),
      STRINGPARAM("flight-core-ip", "Optional remote endpoint observation; not required for logging.\n", 0, &core, NULL),
      STRINGPARAM("flight-interface", "Observed UE interface (default oaitun_ue1).\n", 0, &interface, NULL),
      STRINGPARAM("flight-gpsd", "Optional GPSD endpoint HOST:PORT.\n", 0, &gpsd, NULL),
      STRINGPARAM("flight-min-free-bytes", "Free-space reserve in bytes (default 512 MiB).\n", 0, &min_free, NULL),
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

  bool log = false, off = false;
  char *copy = strdup(features ? features : "off");
  if (copy == NULL)
    return -1;
  char *context = NULL;
  for (char *feature = strtok_r(copy, " ,\t\r\n", &context); feature; feature = strtok_r(NULL, " ,\t\r\n", &context)) {
    if (strcmp(feature, "log") == 0)
      log = true;
    else if (strcmp(feature, "off") == 0)
      off = true;
    else {
      fprintf(stderr,
              "[FLIGHT] Unsupported feature '%s'; this build supports log and off. Recovery/AGC are not implemented.\n",
              feature);
      free(copy);
      return -1;
    }
  }
  free(copy);
  if (log == off) {
    fprintf(stderr, "[FLIGHT] Select log or off, not an empty or conflicting feature list\n");
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
  char **command = calloc((size_t)argc + 40, sizeof(*command));
  if (command == NULL)
    return -1;
  int n = 0;
  command[n++] = "python3";
  command[n++] = "-B";
  command[n++] = script;
  command[n++] = "--role";
  command[n++] = (char *)role;
  command[n++] = "--repo";
  command[n++] = repo;
  command[n++] = "--console";
  command[n++] = "--working-directory";
  command[n++] = cwd;
  if (cfg->num_cfgP > 0 && cfg->cfgP[0] && realpath(cfg->cfgP[0], config)) {
    command[n++] = "--config";
    command[n++] = config;
  }
  const char *pairs[][2] = {{"--output", output},
                            {"--core-ip", core},
                            {"--interface", interface},
                            {"--gpsd", gpsd},
                            {"--recorder-budget", recorder_budget},
                            {"--stdout-budget", stdout_budget},
                            {"--host-budget", host_budget},
                            {"--min-free-bytes", min_free}};
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
