/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file radio/dummy/dummy_rf.c
 * \brief RF library that does nothing to be used in benchmarks with phy-test without RF
 */

#include <errno.h>
#include <string.h>

#include <common/utils/assertions.h>
#include <common/utils/LOG/log.h>
#include "common/utils/threadPool/pthread_utils.h"
#include "common_lib.h"

// structures and timing thread job for timing
typedef struct {
  uint64_t timestamp;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
} dummy_timestamp_t;

typedef struct {
  uint64_t last_received_sample;
  double timescale;
  double sample_rate;
  dummy_timestamp_t dummy_timestamp;
  pthread_t timing_thread;
  bool run_timing_thread;
} dummy_state_t;

static void *dummy_timing_job(void *arg)
{
  dummy_state_t *dummy_state = (dummy_state_t *)arg;
  struct timespec timestamp;
  if (clock_gettime(CLOCK_REALTIME, &timestamp)) {
    LOG_E(UTIL, "clock_gettime failed\n");
    exit(1);
  }
  double leftover_samples = 0;
  while (dummy_state->run_timing_thread) {
    struct timespec current_time;
    if (clock_gettime(CLOCK_REALTIME, &current_time)) {
      LOG_E(UTIL, "clock_gettime failed\n");
      exit(1);
    }
    uint64_t diff = (current_time.tv_sec - timestamp.tv_sec) * 1000000000 + (current_time.tv_nsec - timestamp.tv_nsec);
    timestamp = current_time;
    double samples_to_produce = dummy_state->sample_rate * dummy_state->timescale * diff / 1e9;

    // Attempt to correct compounding rounding error
    leftover_samples += samples_to_produce - (uint64_t)samples_to_produce;
    if (leftover_samples > 1.0f) {
      samples_to_produce += 1;
      leftover_samples -= 1;
    }

    dummy_timestamp_t *dummy_timestamp = &dummy_state->dummy_timestamp;
    mutexlock(dummy_timestamp->mutex);
    dummy_timestamp->timestamp += samples_to_produce;
    condbroadcast(dummy_timestamp->cond);
    mutexunlock(dummy_timestamp->mutex);

    usleep(1);
  }
  return 0;
}

/*! \brief Called to start the RF transceiver. Return 0 if OK, < 0 if error
    @param device pointer to the device structure specific to the RF hardware target
*/
static int dummy_start(openair0_device *device)
{
  // Start the timing thread
  dummy_state_t *dummy_state = (dummy_state_t *)device->priv;
  if (!dummy_state->run_timing_thread) {
    dummy_timestamp_t *dummy_timestamp = &dummy_state->dummy_timestamp;
    memset(dummy_timestamp, 0, sizeof(dummy_timestamp_t));

    pthread_mutexattr_t mutex_attr;
    pthread_condattr_t cond_attr;
    int ret = pthread_mutexattr_init(&mutex_attr);
    AssertFatal(ret == 0, "pthread_mutexattr_init() failed: errno %d, %s\n", errno, strerror(errno));

    ret = pthread_condattr_init(&cond_attr);
    AssertFatal(ret == 0, "pthread_condattr_init() failed: errno %d, %s\n", errno, strerror(errno));

    ret = pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    AssertFatal(ret == 0, "pthread_mutexattr_setpshared() failed: errno %d, %s\n", errno, strerror(errno));

    ret = pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);
    AssertFatal(ret == 0, "pthread_condattr_setpshared() failed: errno %d, %s\n", errno, strerror(errno));

    ret = pthread_mutex_init(&dummy_timestamp->mutex, &mutex_attr);
    AssertFatal(ret == 0, "pthread_mutex_init() failed: errno %d, %s\n", errno, strerror(errno));

    ret = pthread_cond_init(&dummy_timestamp->cond, &cond_attr);
    AssertFatal(ret == 0, "pthread_cond_init() failed: errno %d, %s\n", errno, strerror(errno));

    dummy_state->run_timing_thread = true;
    ret = pthread_create(&dummy_state->timing_thread, NULL, dummy_timing_job, dummy_state);
    AssertFatal(ret == 0, "pthread_create() failed: errno: %d, %s\n", errno, strerror(errno));
  }

  return 0;
}

/*! \brief print the RF statistics
 * \param device the hardware to use
 * \returns  0 on success
 */
int dummy_get_stats(openair0_device *device)
{
  return (0);
}

/*! \brief Reset the RF statistics
 * \param device the hardware to use
 * \returns  0 on success
 */
int dummy_reset_stats(openair0_device *device)
{
  return (0);
}

/*! \brief Terminate operation of the RF transceiver -- free all associated resources (if any)
 * \param device the hardware to use
 */
static void dummy_end(openair0_device *device)
{
  // Stop the timing thread
  dummy_state_t *dummy_state = (dummy_state_t *)device->priv;
  if (dummy_state->run_timing_thread) {
    dummy_state->run_timing_thread = false;
    int ret = pthread_join(dummy_state->timing_thread, NULL);
    AssertFatal(ret == 0, "pthread_join() failed: errno: %d, %s\n", errno, strerror(errno));
  }
}

/*! \brief Stop RF
 * \param card refers to the hardware index to use
 */
int dummy_stop(openair0_device *device)
{
  return (0);
}

/*! \brief Set Gains (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg RF frontend parameters set by application
 * \returns 0 in success
 */
int dummy_set_gains(openair0_device *device, openair0_config_t *openair0_cfg)
{
  return (0);
}

/*! \brief Set frequencies (TX/RX). Spawns a thread to handle the frequency change to not block the calling thread
 * \param device the hardware to use
 * \param openair0_cfg RF frontend parameters set by application
 * \param dummy dummy variable not used
 * \returns 0 in success
 */
int dummy_set_freq(openair0_device *device, openair0_config_t *openair0_cfg)
{
  return (0);
}

int dummy_write_init(openair0_device *device)
{
  return (0);
}

/*! \brief Called to send samples to the RF target
      @param device pointer to the device structure specific to the RF hardware target
      @param timestamp The timestamp at which the first sample MUST be sent
      @param buff Buffer which holds the samples
      @param nsamps number of samples to be sent
      @param antenna_id index of the antenna if the device has multiple antennas
      @param flags flags must be set to true if timestamp parameter needs to be applied
*/
static int dummy_write(openair0_device *device, openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags)
{
  return 0;
}

/*! \brief Receive samples from hardware.
 * Read \ref nsamps samples from each channel to buffers. buff[0] is the array for
 * the first channel. *ptimestamp is the time at which the first sample
 * was received.
 * \param device the hardware to use
 * \param[out] ptimestamp the time at which the first sample was received.
 * \param[out] buff An array of pointers to buffers for received samples. The buffers must be large enough to hold the number of
 * samples \ref nsamps. \param nsamps Number of samples. One sample is 2 byte I + 2 byte Q => 4 byte. \param antenna_id Index of
 * antenna for which to receive samples \returns the number of sample read
 */
static int dummy_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc)
{
  dummy_state_t *dummy_state = (dummy_state_t *)device->priv;

  dummy_timestamp_t *dummy_timestamp = &dummy_state->dummy_timestamp;
  uint64_t timestamp = dummy_state->last_received_sample + nsamps;
  uint64_t current_timestamp = dummy_timestamp->timestamp;
  if (current_timestamp < timestamp) {
    mutexlock(dummy_timestamp->mutex);
    while (current_timestamp < timestamp) {
      condwait(dummy_timestamp->cond, dummy_timestamp->mutex);
      current_timestamp = dummy_timestamp->timestamp;
    }
    mutexunlock(dummy_timestamp->mutex);
  } else {
    LOG_W(HW, "RF read is late!\n");
  }
  *ptimestamp = dummy_state->last_received_sample;
  dummy_state->last_received_sample += nsamps;

  for (int i = 0; i < cc; i++) {
    memset(buff[i], 0, nsamps * sizeof(uint32_t)); // TODO Play random data
  }

  return nsamps;
}

int device_init(openair0_device *device, openair0_config_t *openair0_cfg)
{
  LOG_I(HW, "This is dummy RF that does nothing\n");

  dummy_state_t *dummy_state = calloc_or_fail(1, sizeof(dummy_state_t));
  dummy_state->last_received_sample = 0;
  dummy_state->timescale = 1.0;
  dummy_state->sample_rate = openair0_cfg->sample_rate;
  dummy_state->run_timing_thread = false;

  device->priv = dummy_state;
  device->openair0_cfg = openair0_cfg;
  device->trx_start_func = dummy_start;
  device->trx_get_stats_func = dummy_get_stats;
  device->trx_reset_stats_func = dummy_reset_stats;
  device->trx_end_func = dummy_end;
  device->trx_stop_func = dummy_stop;
  device->trx_set_freq_func = dummy_set_freq;
  device->trx_set_gains_func = dummy_set_gains;
  device->trx_write_init = dummy_write_init;
  device->type = USRP_X400_DEV;
  device->trx_write_func = dummy_write;
  device->trx_read_func = dummy_read;

  return 0;
}
