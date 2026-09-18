/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "executables/flight_options.h"
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/flight_recorder.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

configmodule_interface_t *uniqCfg;
static volatile sig_atomic_t stop;
static void stopped(int signal_number)
{
  stop = signal_number;
}
void exit_function(const char *file, const char *function, const int line, const char *message, const int assertion)
{
  exit(assertion ? 1 : 0);
}
int main(int argc, char **argv)
{
  if (flight_normalize_arguments(&argc, argv))
    return 2;
  uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY | CONFIG_NO_CMDLINE_ECHO);
  if (!uniqCfg || flight_start_capture(uniqCfg, argc, argv, "ue"))
    return 2;
  char *input = NULL, *synthetic_key = NULL;
  int hold = 0;
  paramdef_t options[] = {
      STRINGPARAM("fixture-input", "relative input fixture\n", 0, &input, NULL),
      STRINGPARAM("uicc0.key", "synthetic privacy fixture\n", 0, &synthetic_key, NULL),
      INTPARAM("fixture-hold", "wait for signal\n", 0, &hold, 0),
  };
  if (config_get(uniqCfg, options, sizeofArray(options), NULL) < 0)
    return 3;
  if (input) {
    FILE *f = fopen(input, "r");
    if (!f)
      return 4;
    fclose(f);
  }
  if (config_check_unknown_cmdlineopt(uniqCfg, CONFIG_CHECKALLSECTIONS))
    return 5;
  signal(SIGINT, stopped);
  signal(SIGTERM, stopped);
  logInit();
  flight_recorder_emit(FLIGHT_EVENT_LIFECYCLE, 1, 2, 3, 4, 5, 6);
  printf("fixture-recording=%d\n", flight_recorder_enabled());
  fflush(stdout);
  while (hold && !stop)
    usleep(10000);
  flight_recorder_shutdown();
  return 0;
}
