#define _GNU_SOURCE

#include "utils.h"
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

long int get_microseconds_since_epoch(void) {
    time_t t = time(0);
    struct timeval tv;
    long int useconds;

    gettimeofday(&tv, 0);
    useconds = t*1000000 + tv.tv_usec; //add the microseconds to the seconds

    return useconds;
}

char *str_replace(const char *orig, const char *rep, const char *with) {
    char *result; // the return string
    const char *ins;    // the next insert point
    char *tmp;    // varies
    int len_rep;  // length of rep (the string to remove)
    int len_with; // length of with (the string to replace rep with)
    int len_front; // distance between rep and end of last rep
    int count;    // number of replacements

    // sanity checks and initialization
    if(!orig || !rep) {
        return 0;
    }

    len_rep = strlen(rep);
    if(len_rep == 0) {
        return 0; // empty rep causes infinite loop during count
    }

    if (!with) {
        with = "";
    }
    len_with = strlen(with);

    // count the number of replacements needed
    ins = orig;
    for(count = 0; (tmp = strstr(ins, rep)); ++count) {
        ins = tmp + len_rep;
    }

    tmp = result = malloc(strlen(orig) + (len_with - len_rep) * count + 1);

    if(!result) {
        return 0;
    }

    // first time through the loop, all the variable are set correctly
    // from here on,
    //    tmp points to the end of the result string
    //    ins points to the next occurrence of rep in orig
    //    orig points to the remainder of orig after "end of rep"
    while(count--) {
        ins = strstr(orig, rep);
        len_front = ins - orig;
        tmp = strncpy(tmp, orig, len_front) + len_front;
        tmp = strcpy(tmp, with) + len_with;
        orig += len_front + len_rep; // move to next "end of rep"
    }

    strcpy(tmp, orig);
    return result;
}

char *str_replace_inplace(char *s, const char *rep, const char *with) {
    char *ret = str_replace(s, rep, with);
    free(s);
    if(ret == 0) {
        return 0;
    }
    return ret;
}

char *get_hostname(void) {
    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname, 1023);
    return strdup(hostname);
}
