#pragma once

#include <stdio.h>

// #define netconf_log_error(format, args...)      fprintf(stderr, format, ## args);
#define netconf_log_error(format, ...)      { fprintf(stderr, "\033[1;31m[err/:%5d]\033[0m ", __LINE__); fprintf(stderr, format, ##__VA_ARGS__); fprintf(stderr, "\n"); }

// #define netconf_log(format, args...)      fprintf(stderr, format, ## args);
#define netconf_log(format, ...)      { fprintf(stderr, "\033[1m[log/:%5d]\033[0m ", __LINE__); fprintf(stderr, format, ##__VA_ARGS__); fprintf(stderr, "\n"); }
