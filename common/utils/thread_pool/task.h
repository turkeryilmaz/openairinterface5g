#ifndef TASK_WORK_STEALING_THREAD_POOL_H
#define TASK_WORK_STEALING_THREAD_POOL_H 

#ifndef __cplusplus
#include <stdalign.h>
#else
#define _Alignas(X) alignas(X) 
#endif

#if defined (__i386__) || defined(__x86_64__)
  #define LEVEL1_DCACHE_LINESIZE 64
#elif __aarch64__
  // This is not true for ARM in the general case
  // in linux, you can obtain the size at runtime using sysconf (_SC_LEVEL1_DCACHE_LINESIZE)
  // in c++ using std::hardware_destructive_interference_size
  #define LEVEL1_DCACHE_LINESIZE 64
#else
    static_assert(0!=0, "Unknown CPU architecture");
#endif


typedef struct{
  // Avoid false sharing. Doing it in the first member
 _Alignas(LEVEL1_DCACHE_LINESIZE) void* args;
  void (*func)(void* args);
} task_t;

#endif

