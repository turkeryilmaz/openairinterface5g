/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#ifndef OAI_FLIGHT_OPTIONS_H
#define OAI_FLIGHT_OPTIONS_H
#include "common/config/config_userapi.h"

/* Startup only, before background processes, logging threads, or radio access. */
int flight_normalize_arguments(int *argc, char **argv);
int flight_start_capture(configmodule_interface_t *cfg, int argc, char **argv, const char *role);
#endif
