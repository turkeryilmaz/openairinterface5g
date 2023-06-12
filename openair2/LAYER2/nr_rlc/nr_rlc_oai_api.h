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

/*! \file       nr_rlc_oai_api.h
 * \brief       Header file for nr_rlc_oai_api
 * \author      Guido Casati
 * \date        2020
 * \email:      guido.casati@iis.fraunhofe.de
 * \version     1.0
 * @ingroup     _rlc

 */

#ifndef NR_RLC_OAI_API_H_MIR
#define NR_RLC_OAI_API_H_MIR 


//#include "NR_RLC-BearerConfig.h"
//#include "NR_RLC-Config.h"
//#include "NR_LogicalChannelIdentity.h"
//#include "NR_RadioBearerConfig.h"
//#include "NR_CellGroupConfig.h"
//#include "openair2/RRC/NR/nr_rrc_proto.h"

#include <stdbool.h>
#include <stdint.h>

typedef int32_t               sdu_size_t;
typedef unsigned int       logical_chan_id_t;
typedef uint16_t           rnti_t;

typedef enum {
  NR_RLC_AM,
  NR_RLC_UM,
  NR_RLC_TM,
} nr_rlc_mode_t;

typedef struct {
  nr_rlc_mode_t mode;          /* AM, UM, or TM */

  /* PDU stats */
  /* TX */
  uint32_t txpdu_pkts;         /* aggregated number of transmitted RLC PDUs */
  uint32_t txpdu_bytes;        /* aggregated amount of transmitted bytes in RLC PDUs */
  /* TODO? */
  uint32_t txpdu_wt_ms;      /* aggregated head-of-line tx packet waiting time to be transmitted (i.e. send to the MAC layer) */
  uint32_t txpdu_dd_pkts;      /* aggregated number of dropped or discarded tx packets by RLC */
  uint32_t txpdu_dd_bytes;     /* aggregated amount of bytes dropped or discarded tx packets by RLC */
  uint32_t txpdu_retx_pkts;    /* aggregated number of tx pdus/pkts to be re-transmitted (only applicable to RLC AM) */
  uint32_t txpdu_retx_bytes;   /* aggregated amount of bytes to be re-transmitted (only applicable to RLC AM) */
  uint32_t txpdu_segmented;    /* aggregated number of segmentations */
  uint32_t txpdu_status_pkts;  /* aggregated number of tx status pdus/pkts (only applicable to RLC AM) */
  uint32_t txpdu_status_bytes; /* aggregated amount of tx status bytes  (only applicable to RLC AM) */
  /* TODO? */
  uint32_t txbuf_occ_bytes;    /* current tx buffer occupancy in terms of amount of bytes (average: NOT IMPLEMENTED) */
  /* TODO? */
  uint32_t txbuf_occ_pkts;     /* current tx buffer occupancy in terms of number of packets (average: NOT IMPLEMENTED) */
  /* txbuf_wd_ms: the time window for which the txbuf  occupancy value is obtained - NOT IMPLEMENTED */

  /* RX */
  uint32_t rxpdu_pkts;         /* aggregated number of received RLC PDUs */
  uint32_t rxpdu_bytes;        /* amount of bytes received by the RLC */
  uint32_t rxpdu_dup_pkts;     /* aggregated number of duplicate packets */
  uint32_t rxpdu_dup_bytes;    /* aggregated amount of duplicated bytes */
  uint32_t rxpdu_dd_pkts;      /* aggregated number of rx packets dropped or discarded by RLC */
  uint32_t rxpdu_dd_bytes;     /* aggregated amount of rx bytes dropped or discarded by RLC */
  uint32_t rxpdu_ow_pkts;      /* aggregated number of out of window received RLC pdu */
  uint32_t rxpdu_ow_bytes;     /* aggregated number of out of window bytes received RLC pdu */
  uint32_t rxpdu_status_pkts;  /* aggregated number of rx status pdus/pkts (only applicable to RLC AM) */
  uint32_t rxpdu_status_bytes; /* aggregated amount of rx status bytes  (only applicable to RLC AM) */
  /* rxpdu_rotout_ms: flag indicating rx reordering  timeout in ms - NOT IMPLEMENTED */
  /* rxpdu_potout_ms: flag indicating the poll retransmit time out in ms - NOT IMPLEMENTED */
  /* rxpdu_sptout_ms: flag indicating status prohibit timeout in ms - NOT IMPLEMENTED */
  /* TODO? */
  uint32_t rxbuf_occ_bytes;    /* current rx buffer occupancy in terms of amount of bytes (average: NOT IMPLEMENTED) */
  /* TODO? */
  uint32_t rxbuf_occ_pkts;     /* current rx buffer occupancy in terms of number of packets (average: NOT IMPLEMENTED) */

  /* SDU stats */
  /* TX */
  uint32_t txsdu_pkts;         /* number of SDUs delivered */
  uint32_t txsdu_bytes;        /* number of bytes of SDUs delivered */

  /* RX */
  uint32_t rxsdu_pkts;         /* number of SDUs received */
  uint32_t rxsdu_bytes;        /* number of bytes of SDUs received */
  uint32_t rxsdu_dd_pkts;      /* number of dropped or discarded SDUs */
  uint32_t rxsdu_dd_bytes;     /* number of bytes of SDUs dropped or discarded */

  /* Average time for an SDU to be passed to MAC.
   * Actually measures the time it takes for any part of an SDU to be
   * passed to MAC for the first time, that is: the first TX of (part of) the
   * SDU.
   * Since the MAC schedules in advance, it does not measure the time of
   * transmission over the air, just the time to reach the MAC layer.
   */
  double txsdu_avg_time_to_tx;

} nr_rlc_statistics_t;

//struct NR_RLC_Config;
//struct NR_LogicalChannelConfig;

typedef struct NR_RLC_BearerConfig  NR_RLC_BearerConfig_t;


void nr_rlc_add_srb(int rnti, int srb_id, const NR_RLC_BearerConfig_t *rlc_BearerConfig);
void nr_rlc_add_drb(int rnti, int drb_id, const NR_RLC_BearerConfig_t *rlc_BearerConfig);
void nr_rlc_remove_drb(int rnti, int drb_id);

void nr_rlc_remove_ue(int rnti);

int nr_rlc_get_available_tx_space(
  const rnti_t            rntiP,
  const logical_chan_id_t channel_idP);

void nr_rlc_activate_avg_time_to_tx(
  const rnti_t            rnti,
  const logical_chan_id_t channel_id,
  const bool              is_on);

void nr_rlc_srb_recv_sdu(const int rnti, const logical_chan_id_t channel_id, unsigned char *buf, int size);

struct gNB_MAC_INST_s;
void nr_rlc_activate_srb0(int rnti, struct gNB_MAC_INST_s *mac, void *rawUE,
                          void (*send_initial_ul_rrc_message)(
                                     struct gNB_MAC_INST_s *mac,
                                     int                    rnti,
                                     const uint8_t         *sdu,
                                     sdu_size_t             sdu_len,
                                     void                  *rawUE));


bool nr_rlc_get_statistics(
  int rnti,
  int srb_flag,
  int rb_id,
  nr_rlc_statistics_t *out);

int nr_rlc_get_rnti(void);

#endif
