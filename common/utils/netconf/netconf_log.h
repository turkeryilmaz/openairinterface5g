#pragma once

#include <stdio.h>

// #define netconf_log_error(format, args...)      fprintf(stderr, format, ## args);
#define netconf_log_error(format, ...)      { fprintf(stderr, "\033[1;31m"); fprintf(stderr, format, ##__VA_ARGS__); fprintf(stderr, "\033[0m\n"); }
