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

#ifndef _ORAN_ISOLATE_H_
#define _ORAN_ISOLATE_H_

#include <stdio.h>

#include <pthread.h>
#include <stdint.h>

/*
 * Structure added to bear the information needed from OAI RU
 */
typedef struct ru_info_s {
  // Needed for UL
  int nb_rx;
  int32_t **rxdataF;

  // Needed for DL
  int nb_tx;
  int32_t **txdataF_BF;

  // Needed for Prach
  int **prach_buf;
} ru_info_t;

#ifdef __cplusplus
extern "C" {
#endif

void *define_oran_pointer();

void dump_oran_config(void *xranlib_);

int setup_oran(void *xranlib_);

int open_oran_callback(void *xranlib_);

int open_oran(void *xranlib_);

int initialize_oran(void *xranlib_);

int start_oran(void *xranlib_);

int stop_oran(void *xranlib_);

int close_oran(void *xranlib_);

int register_physide_callbacks(void *xranlib_);

int load_iq_from_file(void *xranlib_);

int xran_fh_rx_read_slot(void *xranlib_, ru_info_t *ru, int *frame, int *slot, int oframe, int oslot, uint8_t sync);

int xran_fh_tx_send_buffer(void *xranlib_);

int xran_fh_tx_send_slot(void *xranlib_, ru_info_t *ru, int frame, int slot, uint64_t timestamp);

int compute_xran_statistics(void *xranlib_);

void check_xran_ptp_sync();

#ifdef __cplusplus
}
#endif

#endif /* _ORAN_ISOLATE_H_ */
