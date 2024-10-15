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
 * Author and copyright: Laurent Thomas, open-cells.com
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

#ifndef RFSIMULATOR_H
#define RFSIMULATOR_H

#include <common/utils/telnetsrv/telnetsrv.h>
#include <openair1/SIMULATION/TOOLS/sim.h>
#include "hashtable.h"

// This needs to be re-architected in the future
//
// File Descriptors management in rfsimulator is not optimized
// Relying on FD_SETSIZE (actually 1024) is not appropriated
// Also the use of fd value as returned by Linux as an index for buf[] structure is not appropriated
// especially for client (UE) side since only 1 fd per connection to a gNB is needed. On the server
// side the value should be tuned to the maximum number of connections with UE's which corresponds
// to the maximum number of UEs hosted by a gNB which is unlikely to be in the order of thousands
// since all I/Q's would flow through the same TCP transport.
// Until a convenient management is implemented, the MAX_FD_RFSIMU is used everywhere (instead of
// FD_SETSIE) and reduced to 125. This should allow for around 20 simultaeous UEs.
//
// #define MAX_FD_RFSIMU FD_SETSIZE
#define MAX_FD_RFSIMU 250

typedef c16_t sample_t; // 2*16 bits complex number

// Simulator role
typedef enum { SIMU_ROLE_SERVER = 1, SIMU_ROLE_CLIENT } simuRole;

typedef struct buffer_s {
  int conn_sock;
  openair0_timestamp lastReceivedTS;
  bool headerMode;
  bool trashingPacket;
  samplesBlockHeader_t th;
  char *transferPtr;
  uint64_t remainToTransfer;
  char *circularBufEnd;
  sample_t *circularBuf;
  channel_desc_t *channel_model;
} buffer_t;

typedef struct {
  int listen_sock, epollfd;
  openair0_timestamp nextRxTstamp;
  openair0_timestamp lastWroteTS;
  simuRole role;
  char *ip;
  uint16_t port;
  int saveIQfile;
  buffer_t buf[MAX_FD_RFSIMU];
  int next_buf;
  // Hashtable used as an indirection level between file descriptor and the buf array
  hash_table_t *fd_to_buf_map;
  int rx_num_channels;
  int tx_num_channels;
  double sample_rate;
  double rx_freq;
  double tx_bw;
  int channelmod;
  double chan_pathloss;
  double chan_forgetfact;
  uint64_t chan_offset;
  float  noise_power_dB;
  void *telnetcmd_qid;
  poll_telnetcmdq_func_t poll_telnetcmdq;
  int wait_timeout;
  double prop_delay_ms;
} rfsimulator_state_t;

//
// CirSize defines the number of samples inquired for a read cycle
// It is bounded by a slot read capability (which depends on bandwidth and numerology)
// up to multiple slots read to allow I/Q buffering of the I/Q TCP stream
//
// As a rule of thumb:
// -it can't be less than the number of samples for a slot
// -it can range up to multiple slots
//
// The default value is chosen for 10ms buffering which makes 23040*20 = 460800 samples
// The previous value is kept below in comment it was computed for 100ms 1x 20MHz
// #define CirSize 6144000 // 100ms SiSo 20MHz LTE
#define minCirSize 460800 // 10ms  SiSo 40Mhz 3/4 sampling NR78 FR1

void rxAddInput(const c16_t *input_sig,
                c16_t *after_channel_sig,
                int rxAnt,
                channel_desc_t *channelDesc,
                int nbSamples,
                uint64_t TS,
                uint32_t CirSize);

#endif
