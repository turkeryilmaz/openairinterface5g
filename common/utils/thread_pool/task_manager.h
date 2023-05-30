#ifndef TASK_MANAGER_WORKING_STEALING_H
#define TASK_MANAGER_WORKING_STEALING_H 

// Comment for deactivating ws tpool
//#define TASK_MANAGER


#include "task.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct{

  pthread_t* t_arr;
  size_t len_thr;
  
  atomic_uint_fast64_t index;

  void* q_arr;

  atomic_uint_fast64_t num_task;

  pthread_cond_t  wait_cv; 
  pthread_mutex_t wait_mtx;

  _Atomic int32_t futex;

  _Atomic bool waiting;

} task_manager_t;

void init_task_manager(task_manager_t* man, uint32_t num_threads);

void free_task_manager(task_manager_t* man, void (*clean)(task_t* args) );

void async_task_manager(task_manager_t* man, task_t t);

void trigger_and_spin(task_manager_t* man);

void trigger_and_wait_all_task_manager(task_manager_t* man);

void wait_all_task_manager(task_manager_t* man);

#endif

