/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef _BARRIER_H_
#define _BARRIER_H_
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

typedef void (*callback_t)(void*);

/// @brief This thread barrier allows modifying the maximum joins after it is initialized
typedef struct {
  pthread_mutex_t barrier_lock;
  int completed_jobs;
  int max_joins;
  callback_t callback;
  void* callback_arg;
} dynamic_barrier_t;

/// @brief Initialize the barrier
/// @param barrier
void dynamic_barrier_init(dynamic_barrier_t* barrier);

/// @brief Reset the barrier after all joins and updates have stopped
/// @param barrier
void dynamic_barrier_reset(dynamic_barrier_t* barrier);

/// @brief Perform join on the barrier. May run callback if it is already set
/// @param barrier
void dynamic_barrier_join(dynamic_barrier_t* barrier);

/// @brief Try to set callback and max_joins, or run callback if max_joins joins have already completed
/// @return false if a callback is pending; the barrier is unchanged and the caller retains arg
/// @note An accepted callback owns arg until it runs or a quiescent reset frees arg
bool dynamic_barrier_try_update(dynamic_barrier_t* barrier, int max_joins, callback_t callback, void* arg);

/// @brief Set callback and max_joins. Requires that no callback is pending
/// @param barrier
/// @param max_joins maximum number of joins before barrier triggers
/// @param callback callback function
/// @param arg callback function argument
void dynamic_barrier_update(dynamic_barrier_t* barrier, int max_joins, callback_t callback, void* arg);

#endif
