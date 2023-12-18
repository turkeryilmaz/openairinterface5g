#ifndef TASK_MANAGER_WORKING_STEALING_H
#define TASK_MANAGER_WORKING_STEALING_H 


// Comment for deactivating ws tpool
#define TASK_MANAGER
#define TASK_MANAGER_DEMODULATION
#define TASK_MANAGER_CODING
#define TASK_MANAGER_RU

#define TASK_MANAGER_UE
#define TASK_MANAGER_UE_DECODING

#define TASK_MANAGER_SIM

// LTE
#define TASK_MANAGER_LTE


#include "task.h"


#ifndef __cplusplus
#include <stdalign.h>
#include <stdatomic.h>
#else
#include <atomic>
#define _Atomic(X) std::atomic< X >
#define  _Alignas(X) alignas(X) 
#endif

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

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
  // Avoid false sharing
 _Alignas(LEVEL1_DCACHE_LINESIZE) _Atomic(int) completed;
} task_status_t;


typedef struct{
  uint8_t* buf;
  size_t len;
  task_status_t* task_status;
} thread_info_tm_t;

typedef struct{

  pthread_t* t_arr;
  size_t len_thr;
  
  _Atomic(uint64_t) index;

  void* q_arr;

  _Atomic(uint64_t) num_task;

//  pthread_cond_t  wait_cv; 
//  pthread_mutex_t wait_mtx;

//  _Atomic(int32_t) futex;

  //_Atomic(bool) waiting;
} task_manager_t;

void init_task_manager(task_manager_t* man, uint32_t num_threads);

void free_task_manager(task_manager_t* man, void (*clean)(task_t* args) );

void async_task_manager(task_manager_t* man, task_t t);

// This function triggers the futex if the thread is waiting.
// Note that if the thread was working, this call is superflous
// This method proved a bit faster as it is only one syscall
// instead of a syscall when async_task_manager is called 
// void trigger_all_task_manager(task_manager_t* man);

//void trigger_and_spin_task_manager(task_manager_t* man);

//void stop_spining_task_manager(task_manager_t* man);

//void trigger_and_wait_all_task_manager(task_manager_t* man);

//void wait_all_task_manager(task_manager_t* man);

// This function does not belong here.
// It should be in an algorithm file
void wait_task_status_completed(size_t len, task_status_t* arr); 

#endif

