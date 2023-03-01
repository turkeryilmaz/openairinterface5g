#pragma once

long int get_microseconds_since_epoch(void);
char *str_replace(const char *orig, const char *rep, const char *with);
char *str_replace_inplace(char *s, const char *rep, const char *with);
