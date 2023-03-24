#pragma once

long int get_microseconds_since_epoch(void);
char *get_netconf_timestamp(void);
char *str_replace(const char *orig, const char *rep, const char *with);
char *str_replace_inplace(char *s, const char *rep, const char *with);
char *get_hostname(void);
unsigned long int get_file_size(const char *filename);
