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

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include "debug.h"
#include "nfapi/oai_integration/vendor_ext.h"
#include "nfapi_pnf_interface.h"
#include "nfapi_nr_interface.h"
#include "nfapi_nr_interface_scf.h"

#include "nfapi.h"
#include "nfapi_pnf.h"
#include "common/ran_context.h"
#include "openair2/PHY_INTERFACE/phy_stub_UE.h"

#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <assert.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>

#include <vendor_ext.h>
#include "fapi_stub.h"

#include "common/utils/LOG/log.h"

#include "PHY/INIT/phy_init.h"
#include "PHY/INIT/nr_phy_init.h"
#include "PHY/LTE_TRANSPORT/transport_proto.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair1/SCHED_NR/fapi_nr_l1.h"
#include "openair1/PHY/NR_TRANSPORT/nr_dlsch.h"
#include "openair1/PHY/defs_gNB.h"
#include <openair1/SCHED/fapi_l1.h>
#include <openair1/PHY/NR_TRANSPORT/nr_transport_proto.h>
#include "executables/lte-softmodem.h"
#include "nfapi/open-nFAPI/pnf/inc/pnf_p7.h"

#define NUM_P5_PHY 2

#define _GNU_SOURCE

extern void phy_init_RU(RU_t *);
extern int config_sync_var;
extern RAN_CONTEXT_t RC;

extern pthread_cond_t nfapi_sync_cond;
extern pthread_mutex_t nfapi_sync_mutex;
extern int nfapi_sync_var;

extern int sync_var;

nfapi_tx_request_pdu_t *tx_request_pdu[1023][10][10]; // [frame][subframe][max_num_pdus]
uint8_t nr_tx_pdus[32][16][4096];
nfapi_nr_pdu_t *tx_data_request[1023][20][10]; //[frame][slot][max_num_pdus]
uint8_t tx_pdus[32][8][4096];

nfapi_ue_release_request_body_t release_rntis;

nfapi_nr_pnf_param_response_t g_pnf_param_resp;


nfapi_pnf_p7_config_t *p7_config_g = NULL;

void *pnf_allocate(size_t size) {
  return malloc(size);
}

void pnf_deallocate(void *ptr) {
  free(ptr);
}


typedef struct {
  //public:
  uint8_t enabled;
  uint32_t rx_port;
  uint32_t tx_port;
  //std::string tx_addr;
  char tx_addr[80];
} udp_data;

typedef struct {
  uint16_t index;
  uint16_t id;
  uint8_t rfs[2];
  uint8_t excluded_rfs[2];

  udp_data udp;

  char local_addr[80];
  int local_port;

  char *remote_addr;
  int remote_port;

  uint8_t duplex_mode;
  uint16_t dl_channel_bw_support;
  uint16_t ul_channel_bw_support;
  uint8_t num_dl_layers_supported;
  uint8_t num_ul_layers_supported;
  uint16_t release_supported;
  uint8_t nmm_modes_supported;

  uint8_t dl_ues_per_subframe;
  uint8_t ul_ues_per_subframe;

  uint8_t first_subframe_ind;

  // timing information recevied from the vnf
  uint8_t timing_window;
  uint8_t timing_info_mode;
  uint8_t timing_info_period;

} phy_info;

typedef struct {
  //public:
  uint16_t index;
  uint16_t band;
  int16_t max_transmit_power;
  int16_t min_transmit_power;
  uint8_t num_antennas_supported;
  uint32_t min_downlink_frequency;
  uint32_t max_downlink_frequency;
  uint32_t max_uplink_frequency;
  uint32_t min_uplink_frequency;
} rf_info;


typedef struct {

  int release;
  phy_info phys[2];
  rf_info rfs[2];

  uint8_t sync_mode;
  uint8_t location_mode;
  uint8_t location_coordinates[6];
  uint32_t dl_config_timing;
  uint32_t ul_config_timing;
  uint32_t tx_timing;
  uint32_t hi_dci0_timing;

  uint16_t max_phys;
  uint16_t max_total_bw;
  uint16_t max_total_dl_layers;
  uint16_t max_total_ul_layers;
  uint8_t shared_bands;
  uint8_t shared_pa;
  int16_t max_total_power;
  uint8_t oui;

  uint8_t wireshark_test_mode;

} pnf_info;

typedef struct {
  uint16_t phy_id;
  nfapi_pnf_config_t *config;
  phy_info *phy;
  nfapi_pnf_p7_config_t *p7_config;
} pnf_phy_user_data_t;

static pnf_info pnf;
static pthread_t pnf_start_pthread;

int nfapitooai_level(int nfapi_level) {
  switch(nfapi_level) {
    case NFAPI_TRACE_ERROR:
      return OAILOG_ERR;

    case NFAPI_TRACE_WARN:
      return OAILOG_WARNING;

    case NFAPI_TRACE_NOTE:
      return OAILOG_INFO;

    case NFAPI_TRACE_INFO:
      return OAILOG_DEBUG;
  }

  return OAILOG_ERR;
}

void pnf_nfapi_trace(nfapi_trace_level_t nfapi_level, const char *message, ...) {
  va_list args;
  va_start(args, message);
  VLOG( NFAPI_PNF, nfapitooai_level(nfapi_level), message, args);
  va_end(args);
}

void pnf_set_thread_priority(int priority) {
  set_priority(priority);

  pthread_attr_t ptAttr;
  if(pthread_attr_setschedpolicy(&ptAttr, SCHED_RR) != 0) {
    printf("failed to set pthread SCHED_RR %d\n", errno);
  }

  pthread_attr_setinheritsched(&ptAttr, PTHREAD_EXPLICIT_SCHED);
  struct sched_param thread_params;
  thread_params.sched_priority = 20;

  if(pthread_attr_setschedparam(&ptAttr, &thread_params) != 0) {
    printf("failed to set sched param\n");
  }
}

void *pnf_p7_thread_start(void *ptr) {
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] P7 THREAD %s\n", __FUNCTION__);
  pnf_set_thread_priority(79);
  nfapi_pnf_p7_config_t *config = (nfapi_pnf_p7_config_t *)ptr;
  nfapi_pnf_p7_start(config);
  return 0;
}

void *pnf_nr_p7_thread_start(void *ptr) {
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[NR_PNF] NR P7 THREAD %s\n", __FUNCTION__);
  pnf_set_thread_priority(79);
  nfapi_pnf_p7_config_t *config = (nfapi_pnf_p7_config_t *)ptr;
  nfapi_nr_pnf_p7_start(config);
  return 0;
}

nfapi_p7_message_header_t *pnf_phy_allocate_p7_vendor_ext(uint16_t message_id, uint16_t *msg_size) {
  if(message_id == P7_VENDOR_EXT_REQ) {
    (*msg_size) = sizeof(vendor_ext_p7_req);
    return (nfapi_p7_message_header_t *)malloc(sizeof(vendor_ext_p7_req));
  }

  return 0;
}

void pnf_phy_deallocate_p7_vendor_ext(nfapi_p7_message_header_t *header) {
  free(header);
}

notifiedFIFO_elt_t *l1tx_message_extract(PHY_VARS_gNB *gNB, int frame, int slot) {
  notifiedFIFO_elt_t *res;

  //TODO: This needs to be reworked for nfapi to work
  res = pullTpool(&gNB->L1_tx_free, &gNB->threadPool);
  return res;
}

int pnf_phy_ul_dci_req(gNB_L1_rxtx_proc_t *proc, nfapi_pnf_p7_config_t *pnf_p7, nfapi_nr_ul_dci_request_t *req) {
  
  //   LOG_D(PHY,"[PNF] HI_DCI0_REQUEST SFN/SF:%05d dci:%d hi:%d\n", NFAPI_SFNSF2DEC(req->sfn_sf), req->hi_dci0_request_body.number_of_dci, req->hi_dci0_request_body.number_of_hi);

  struct PHY_VARS_gNB_s *gNB = RC.gNB[0];

  // extract the next available thread message (priority to message with current slot, then free message)
  notifiedFIFO_elt_t *res;
  res = l1tx_message_extract(gNB, req->SFN, req->Slot);
  processingData_L1tx_t *msgTx = (processingData_L1tx_t *)NotifiedFifoData(res);

  if (proc ==NULL) 
    proc = &gNB->proc.L1_proc;

  if (req->numPdus > 0) {
    for (int i=0; i<req->numPdus; i++) {
      if (req->ul_dci_pdu_list[i].PDUType == NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE) // only possible value 0: PDCCH PDU
        msgTx->ul_pdcch_pdu[i] = req->ul_dci_pdu_list[i];
      else
        LOG_E(PHY,"[PNF] UL_DCI_REQ sfn_slot:%d PDU[%d] - unknown pdu type:%d\n", NFAPI_SFNSLOT2DEC(req->SFN, req->Slot), req->numPdus-1, req->ul_dci_pdu_list[req->numPdus-1].PDUType);
    }
  }

  pushNotifiedFIFO(&gNB->L1_tx_filled, res);

  return 0;
}


int pnf_phy_hi_dci0_req(L1_rxtx_proc_t *proc, nfapi_pnf_p7_config_t *pnf_p7, nfapi_hi_dci0_request_t *req) {
  if (req->hi_dci0_request_body.number_of_dci == 0 && req->hi_dci0_request_body.number_of_hi == 0)
    LOG_D(PHY,"[PNF] HI_DCI0_REQUEST SFN/SF:%05d dci:%d hi:%d\n", NFAPI_SFNSF2DEC(req->sfn_sf), req->hi_dci0_request_body.number_of_dci, req->hi_dci0_request_body.number_of_hi);

  //phy_info* phy = (phy_info*)(pnf_p7->user_data);
  struct PHY_VARS_eNB_s *eNB = RC.eNB[0][0];
  if (proc ==NULL) 
    proc = &eNB->proc.L1_proc;

  for (int i=0; i<req->hi_dci0_request_body.number_of_dci + req->hi_dci0_request_body.number_of_hi; i++) {
    //LOG_D(PHY,"[PNF] HI_DCI0_REQ sfn_sf:%d PDU[%d]\n", NFAPI_SFNSF2DEC(req->sfn_sf), i);
    if (req->hi_dci0_request_body.hi_dci0_pdu_list[i].pdu_type == NFAPI_HI_DCI0_DCI_PDU_TYPE) {
      //LOG_D(PHY,"[PNF] HI_DCI0_REQ sfn_sf:%d PDU[%d] - NFAPI_HI_DCI0_DCI_PDU_TYPE\n", NFAPI_SFNSF2DEC(req->sfn_sf), i);
      nfapi_hi_dci0_request_pdu_t *hi_dci0_req_pdu = &req->hi_dci0_request_body.hi_dci0_pdu_list[i];
      handle_nfapi_hi_dci0_dci_pdu(eNB,NFAPI_SFNSF2SFN(req->sfn_sf),NFAPI_SFNSF2SF(req->sfn_sf),proc,hi_dci0_req_pdu);
      eNB->pdcch_vars[NFAPI_SFNSF2SF(req->sfn_sf)&1].num_dci++;
    } else if (req->hi_dci0_request_body.hi_dci0_pdu_list[i].pdu_type == NFAPI_HI_DCI0_HI_PDU_TYPE) {
      LOG_D(PHY,"[PNF] HI_DCI0_REQ sfn_sf:%d PDU[%d] - NFAPI_HI_DCI0_HI_PDU_TYPE\n", NFAPI_SFNSF2DEC(req->sfn_sf), i);
      nfapi_hi_dci0_request_pdu_t *hi_dci0_req_pdu = &req->hi_dci0_request_body.hi_dci0_pdu_list[i];
      handle_nfapi_hi_dci0_hi_pdu(eNB, NFAPI_SFNSF2SFN(req->sfn_sf),NFAPI_SFNSF2SF(req->sfn_sf), proc, hi_dci0_req_pdu);
    } else {
      LOG_E(PHY,"[PNF] HI_DCI0_REQ sfn_sf:%d PDU[%d] - unknown pdu type:%d\n", NFAPI_SFNSF2DEC(req->sfn_sf), i, req->hi_dci0_request_body.hi_dci0_pdu_list[i].pdu_type);
    }
  }

  return 0;
}


int pnf_phy_dl_tti_req(gNB_L1_rxtx_proc_t *proc, nfapi_pnf_p7_config_t *pnf_p7, nfapi_nr_dl_tti_request_t *req) {
  if (RC.ru == 0) {
    return -1;
  }

  if (RC.gNB == 0) {
    return -2;
  }

  if (RC.gNB[0] == 0) {
    return -3;
  }

  if (sync_var != 0) {
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() Main system not up - is this a dummy subframe?\n", __FUNCTION__);
    return -4;
  }

  int sfn = req->SFN;
  int slot =  req->Slot;
  struct PHY_VARS_gNB_s *gNB = RC.gNB[0];
  if (proc==NULL)
     proc = &gNB->proc.L1_proc;
  nfapi_nr_dl_tti_request_pdu_t *dl_tti_pdu_list = req->dl_tti_request_body.dl_tti_pdu_list;

  if (req->dl_tti_request_body.nPDUs)
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() TX:%d/%d RX:%d/%d; sfn:%d, slot:%d, nGroup:%u, nPDUs: %u, nUE: %p, PduIdx: %p,\n",
                __FUNCTION__, proc->frame_tx, proc->slot_tx, proc->frame_rx, proc->slot_rx, // TODO: change subframes to slot
                req->SFN,
                req->Slot,
                req->dl_tti_request_body.nGroup,
                req->dl_tti_request_body.nPDUs,
                req->dl_tti_request_body.nUe,
                req->dl_tti_request_body.PduIdx);

  for (int i=0; i<req->dl_tti_request_body.nPDUs; i++) {
    // NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() sfn/sf:%d PDU[%d] size:%d pdcch_vars->num_dci:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), i, dl_config_pdu_list[i].pdu_size,pdcch_vars->num_dci);
    notifiedFIFO_elt_t *res;
    res = l1tx_message_extract(gNB, sfn, slot);
    processingData_L1tx_t *msgTx = (processingData_L1tx_t *)NotifiedFifoData(res);

    if (dl_tti_pdu_list[i].PDUType == NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE) {
      msgTx->pdcch_pdu[i] = dl_tti_pdu_list[i].pdcch_pdu; // copies all the received PDCCH PDUs
    } 
    else if (dl_tti_pdu_list[i].PDUType == NFAPI_NR_DL_TTI_SSB_PDU_TYPE) {
      //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() PDU:%d BCH: pdu_index:%u pdu_length:%d sdu_length:%d BCH_SDU:%x,%x,%x\n", __FUNCTION__, i, pdu_index, bch_pdu->bch_pdu_rel8.length, tx_request_pdu[sfn][sf][pdu_index]->segments[0].segment_length, sdu[0], sdu[1], sdu[2]);
      handle_nr_nfapi_ssb_pdu(msgTx, sfn, slot, &dl_tti_pdu_list[i]);
    }
    else if (dl_tti_pdu_list[i].PDUType == NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE) {
      nfapi_nr_dl_tti_pdsch_pdu *pdsch_pdu = &dl_tti_pdu_list[i].pdsch_pdu;
      nfapi_nr_dl_tti_pdsch_pdu_rel15_t *rel15_pdu = &pdsch_pdu->pdsch_pdu_rel15;
      nfapi_nr_pdu_t *tx_data = tx_data_request[sfn][slot][rel15_pdu->pduIndex];

      if (tx_data != NULL) {
        uint8_t *dlsch_sdu = (uint8_t *)tx_data->TLVs[0].value.direct;
        //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() DLSCH:pdu_index:%d handle_nfapi_dlsch_pdu(eNB, proc_rxtx, dlsch_pdu, transport_blocks:%d sdu:%p) eNB->pdcch_vars[proc->subframe_tx & 1].num_pdcch_symbols:%d\n", __FUNCTION__, rel8_pdu->pdu_index, rel8_pdu->transport_blocks, dlsch_sdu, eNB->pdcch_vars[proc->subframe_tx & 1].num_pdcch_symbols);
        AssertFatal(msgTx->num_pdsch_slot < gNB->max_nb_pdsch,
                    "Number of PDSCH PDUs %d exceeded the limit %d\n",
                    msgTx->num_pdsch_slot,
                    gNB->max_nb_pdsch);
        handle_nr_nfapi_pdsch_pdu(msgTx, pdsch_pdu, dlsch_sdu);
      } 
      else {
        NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() DLSCH NULL TX PDU SFN/SF:%d PDU_INDEX:%d\n", __FUNCTION__, NFAPI_SFNSLOT2DEC(sfn,slot), rel15_pdu->pduIndex);     
      }
    }
    else if (dl_tti_pdu_list[i].PDUType == NFAPI_NR_DL_TTI_CSI_RS_PDU_TYPE) {
      nfapi_nr_dl_tti_csi_rs_pdu *csi_rs_pdu = &dl_tti_pdu_list[i].csi_rs_pdu;
      handle_nfapi_nr_csirs_pdu(msgTx, sfn, slot, csi_rs_pdu);
    }
    else {
      NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() UNKNOWN:%d\n", __FUNCTION__, dl_tti_pdu_list[i].PDUType);
    }
    pushNotifiedFIFO(&gNB->L1_tx_filled, res);
  }

  if(req->vendor_extension)
    free(req->vendor_extension);

  return 0;
}


int pnf_phy_dl_config_req(L1_rxtx_proc_t *proc, nfapi_pnf_p7_config_t *pnf_p7, nfapi_dl_config_request_t *req) {

  if (RC.eNB == 0) {
    return -2;
  }

  if (RC.eNB[0][0] == 0) {
    return -3;
  }

  if (sync_var != 0) {
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() Main system not up - is this a dummy subframe?\n", __FUNCTION__);
    return -4;
  }

  int sfn = NFAPI_SFNSF2SFN(req->sfn_sf);
  int sf = NFAPI_SFNSF2SF(req->sfn_sf);
  struct PHY_VARS_eNB_s *eNB = RC.eNB[0][0];
  if (proc==NULL)
     proc = &eNB->proc.L1_proc;
  nfapi_dl_config_request_pdu_t *dl_config_pdu_list = req->dl_config_request_body.dl_config_pdu_list;
  LTE_eNB_PDCCH *pdcch_vars = &eNB->pdcch_vars[sf&1];
  pdcch_vars->num_pdcch_symbols = req->dl_config_request_body.number_pdcch_ofdm_symbols;
  pdcch_vars->num_dci = 0;

  if (req->dl_config_request_body.number_dci ||
      req->dl_config_request_body.number_pdu ||
      req->dl_config_request_body.number_pdsch_rnti)
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() TX:%d/%d RX:%d/%d sfn_sf:%d pdcch:%u dl_cfg[dci:%u pdus:%d pdsch_rnti:%d] pcfich:%u\n",
                __FUNCTION__, proc->frame_tx, proc->subframe_tx, proc->frame_rx, proc->subframe_rx,
                NFAPI_SFNSF2DEC(req->sfn_sf),
                req->dl_config_request_body.number_pdcch_ofdm_symbols,
                req->dl_config_request_body.number_dci,
                req->dl_config_request_body.number_pdu,
                req->dl_config_request_body.number_pdsch_rnti,
                req->dl_config_request_body.transmission_power_pcfich);

  for (int i=0; i<req->dl_config_request_body.number_pdu; i++) {
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() sfn/sf:%d PDU[%d] size:%d pdcch_vars->num_dci:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), i, dl_config_pdu_list[i].pdu_size,pdcch_vars->num_dci);

    if (dl_config_pdu_list[i].pdu_type == NFAPI_DL_CONFIG_DCI_DL_PDU_TYPE) {
      handle_nfapi_dci_dl_pdu(eNB,NFAPI_SFNSF2SFN(req->sfn_sf),NFAPI_SFNSF2SF(req->sfn_sf),proc,&dl_config_pdu_list[i]);
      pdcch_vars->num_dci++; // Is actually number of DCI PDUs
      NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() pdcch_vars->num_dci:%d\n", __FUNCTION__, pdcch_vars->num_dci);
    } else if (dl_config_pdu_list[i].pdu_type == NFAPI_DL_CONFIG_BCH_PDU_TYPE) {
      nfapi_dl_config_bch_pdu *bch_pdu = &dl_config_pdu_list[i].bch_pdu;
      uint16_t pdu_index = bch_pdu->bch_pdu_rel8.pdu_index;

      if (tx_request_pdu[sfn][sf][pdu_index] != NULL) {
        //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() PDU:%d BCH: pdu_index:%u pdu_length:%d sdu_length:%d BCH_SDU:%x,%x,%x\n", __FUNCTION__, i, pdu_index, bch_pdu->bch_pdu_rel8.length, tx_request_pdu[sfn][sf][pdu_index]->segments[0].segment_length, sdu[0], sdu[1], sdu[2]);
        handle_nfapi_bch_pdu(eNB, proc, &dl_config_pdu_list[i], tx_request_pdu[sfn][sf][pdu_index]->segments[0].segment_data);
        eNB->pbch_configured=1;
      } else {
        NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() BCH NULL TX PDU SFN/SF:%d PDU_INDEX:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), pdu_index);
      }
    } else if (dl_config_pdu_list[i].pdu_type == NFAPI_DL_CONFIG_DLSCH_PDU_TYPE) {
      nfapi_dl_config_dlsch_pdu *dlsch_pdu = &dl_config_pdu_list[i].dlsch_pdu;
      nfapi_dl_config_dlsch_pdu_rel8_t *rel8_pdu = &dlsch_pdu->dlsch_pdu_rel8;
      nfapi_tx_request_pdu_t *tx_pdu = tx_request_pdu[sfn][sf][rel8_pdu->pdu_index];

      if (tx_pdu != NULL) {
        int UE_id = find_dlsch(rel8_pdu->rnti,eNB,SEARCH_EXIST_OR_FREE);
        AssertFatal(UE_id!=-1,"no free or exiting dlsch_context\n");
        AssertFatal(UE_id<NUMBER_OF_UE_MAX,"returned UE_id %d >= %d(NUMBER_OF_UE_MAX)\n",UE_id,NUMBER_OF_UE_MAX);
        LTE_eNB_DLSCH_t *dlsch0 = eNB->dlsch[UE_id][0];
        //LTE_eNB_DLSCH_t *dlsch1 = eNB->dlsch[UE_id][1];
        int harq_pid = dlsch0->harq_ids[sfn%2][sf];

        if(harq_pid >= dlsch0->Mdlharq) {
          LOG_E(PHY,"pnf_phy_dl_config_req illegal harq_pid %d\n", harq_pid);
          return(-1);
        }

        uint8_t *dlsch_sdu = tx_pdus[UE_id][harq_pid];
        memcpy(dlsch_sdu, tx_pdu->segments[0].segment_data, tx_pdu->segments[0].segment_length);
        //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() DLSCH:pdu_index:%d handle_nfapi_dlsch_pdu(eNB, proc_rxtx, dlsch_pdu, transport_blocks:%d sdu:%p) eNB->pdcch_vars[proc->subframe_tx & 1].num_pdcch_symbols:%d\n", __FUNCTION__, rel8_pdu->pdu_index, rel8_pdu->transport_blocks, dlsch_sdu, eNB->pdcch_vars[proc->subframe_tx & 1].num_pdcch_symbols);
        handle_nfapi_dlsch_pdu( eNB, sfn,sf, proc, &dl_config_pdu_list[i], rel8_pdu->transport_blocks-1, dlsch_sdu);
      } else {
        NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() DLSCH NULL TX PDU SFN/SF:%d PDU_INDEX:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), rel8_pdu->pdu_index);
      }
    } else {
      NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() UNKNOWN:%d\n", __FUNCTION__, dl_config_pdu_list[i].pdu_type);
    }
  }

  if(req->vendor_extension)
    free(req->vendor_extension);

  return 0;
}



int pnf_phy_tx_data_req(nfapi_pnf_p7_config_t *pnf_p7, nfapi_nr_tx_data_request_t *req) {
  uint16_t sfn = req->SFN;
  uint16_t slot = req->Slot;

  if (req->Number_of_PDUs == 0)
    LOG_D(PHY,"%s() SFN/SLOT:%d%d PDUs:%d\n", __FUNCTION__, sfn, slot, req->Number_of_PDUs);

  //if (req->pdu_list[0].TLVs->tag ==  NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST) {
    for (int i=0; i<req->Number_of_PDUs; i++) {
      // LOG_D(PHY,"%s() SFN/SF:%d%d number_of_pdus:%d [PDU:%d] pdu_length:%d pdu_index:%d num_segments:%d\n",
      //       __FUNCTION__,
      //       sfn, sf,
      //       req->tx_request_body.number_of_pdus,
      //       i,
      //       req->tx_request_body.tx_pdu_list[i].pdu_length,
      //       req->tx_request_body.tx_pdu_list[i].pdu_index,
      //       req->tx_request_body.tx_pdu_list[i].num_segments
      //      );
      // tx_request_pdu[sfn][sf][i] = &req->tx_request_body.tx_pdu_list[i];
      tx_data_request[sfn][slot][i] = &req->pdu_list[i];
    }
  //}

  return 0;
}


int pnf_phy_tx_req(nfapi_pnf_p7_config_t *pnf_p7, nfapi_tx_request_t *req) {
  uint16_t sfn = NFAPI_SFNSF2SFN(req->sfn_sf);
  uint16_t sf = NFAPI_SFNSF2SF(req->sfn_sf);

  if (req->tx_request_body.number_of_pdus==0)
    LOG_D(PHY,"%s() SFN/SF:%d%d PDUs:%d\n", __FUNCTION__, sfn, sf, req->tx_request_body.number_of_pdus);

  if (req->tx_request_body.tl.tag==NFAPI_TX_REQUEST_BODY_TAG) {
    for (int i=0; i<req->tx_request_body.number_of_pdus; i++) {
      LOG_D(PHY,"%s() SFN/SF:%d%d number_of_pdus:%d [PDU:%d] pdu_length:%d pdu_index:%d num_segments:%d\n",
            __FUNCTION__,
            sfn, sf,
            req->tx_request_body.number_of_pdus,
            i,
            req->tx_request_body.tx_pdu_list[i].pdu_length,
            req->tx_request_body.tx_pdu_list[i].pdu_index,
            req->tx_request_body.tx_pdu_list[i].num_segments
           );
      tx_request_pdu[sfn][sf][i] = &req->tx_request_body.tx_pdu_list[i];
    }
  }

  return 0;
}



int pnf_phy_ul_tti_req(gNB_L1_rxtx_proc_t *proc, nfapi_pnf_p7_config_t *pnf_p7, nfapi_nr_ul_tti_request_t *req) {
  LOG_D(PHY,"[PNF] UL_TTI_REQ recvd, writing into structs, SFN/slot:%d.%d pdu:%d \n",
                req->SFN,req->Slot,
                req->n_pdus
               );

  if (RC.ru == 0) {
    return -1;
  }

  if (RC.gNB == 0) {
    return -2;
  }

  if (RC.gNB[0] == 0) {
    return -3;
  }

  if (sync_var != 0) {
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() Main system not up - is this a dummy slot?\n", __FUNCTION__);
    return -4;
  }
  uint16_t curr_sfn = req->SFN;
  uint16_t curr_slot = req->Slot;
  struct PHY_VARS_gNB_s *gNB = RC.gNB[0];

  if (proc==NULL)
     proc = &gNB->proc.L1_proc;

  nfapi_nr_ul_tti_request_number_of_pdus_t *ul_tti_pdu_list = req->pdus_list;

  for (int i=0; i< req->n_pdus; i++) {
    switch (ul_tti_pdu_list[i].pdu_type) {
      case NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE:
        //LOG_D(PHY,"frame %d, slot %d, Got NFAPI_NR_UL_TTI_PUSCH_PDU_TYPE for %d.%d\n", frame, slot, UL_tti_req->SFN, UL_tti_req->Slot);
        //curr_sfn = curr_sfn + 3; //Gokul
        nr_fill_ulsch(gNB,curr_sfn, curr_slot, &ul_tti_pdu_list[i].pusch_pdu);
        break;
      case NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE:
        //LOG_D(PHY,"frame %d, slot %d, Got NFAPI_NR_UL_TTI_PUCCH_PDU_TYPE for %d.%d\n", frame, slot, UL_tti_req->SFN, UL_tti_req->Slot);
        nr_fill_pucch(gNB,curr_sfn, curr_slot, &ul_tti_pdu_list[i].pucch_pdu);
        break;
      case NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE:
        //LOG_D(PHY,"frame %d, slot %d, Got NFAPI_NR_UL_TTI_PRACH_PDU_TYPE for %d.%d\n", frame, slot, UL_tti_req->SFN, UL_tti_req->Slot);
        nr_fill_prach(gNB, curr_sfn, curr_slot, &ul_tti_pdu_list[i].prach_pdu);
        if (gNB->RU_list[0]->if_south == LOCAL_RF) nr_fill_prach_ru(gNB->RU_list[0], curr_sfn, curr_slot, &ul_tti_pdu_list[i].prach_pdu);
        break;
      default:
      NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() PDU:%i UNKNOWN type :%d\n", __FUNCTION__, i, ul_tti_pdu_list[i].pdu_type);
      break;

    }
    // //LOG_D(PHY, "%s() sfn/sf:%d PDU[%d] size:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), i, ul_config_pdu_list[i].pdu_size);
    // if (
    //   ul_tti_pdu_list[i].pdu_type == NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE ||
    //   ul_tti_pdu_list[i].pdu_type == NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE ||
    //   ul_tti_pdu_list[i].pdu_type == NFAPI_NR_UL_CONFIG_PUCCH_PDU_TYPE ||
    //   ul_tti_pdu_list[i].pdu_type == NFAPI_NR_UL_CONFIG_SRS_PDU_TYPE
    // ) {
    //   //LOG_D(PHY, "%s() handle_nfapi_ul_pdu() for PDU:%d\n", __FUNCTION__, i);
    //   // handle_nfapi_ul_pdu(eNB,proc,&ul_config_pdu_list[i],curr_sfn,curr_sf,req->ul_config_request_body.srs_present);
      
    //   // TODO: dont have an NR function for this, also srs_present flag not there
      
    // } else {
    //   NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() PDU:%i UNKNOWN type :%d\n", __FUNCTION__, i, ul_tti_pdu_list[i].pdu_type);
    // }
  }

  return 0;
}

int pnf_phy_ul_config_req(L1_rxtx_proc_t *proc, nfapi_pnf_p7_config_t *pnf_p7, nfapi_ul_config_request_t *req) {
  if (0)LOG_D(PHY,"[PNF] UL_CONFIG_REQ %s() sfn_sf:%d pdu:%d rach_prach_frequency_resources:%d srs_present:%u\n",
                __FUNCTION__,
                NFAPI_SFNSF2DEC(req->sfn_sf),
                req->ul_config_request_body.number_of_pdus,
                req->ul_config_request_body.rach_prach_frequency_resources,
                req->ul_config_request_body.srs_present
               );


  if (RC.eNB == 0) {
    return -2;
  }

  if (RC.eNB[0][0] == 0) {
    return -3;
  }

  if (sync_var != 0) {
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() Main system not up - is this a dummy subframe?\n", __FUNCTION__);
    return -4;
  }

  uint16_t curr_sfn = NFAPI_SFNSF2SFN(req->sfn_sf);
  uint16_t curr_sf = NFAPI_SFNSF2SF(req->sfn_sf);
  struct PHY_VARS_eNB_s *eNB = RC.eNB[0][0];
  if (proc==NULL)
     proc = &eNB->proc.L1_proc;
  nfapi_ul_config_request_pdu_t *ul_config_pdu_list = req->ul_config_request_body.ul_config_pdu_list;

  for (int i=0; i<req->ul_config_request_body.number_of_pdus; i++) {
    //LOG_D(PHY, "%s() sfn/sf:%d PDU[%d] size:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), i, ul_config_pdu_list[i].pdu_size);
    if (
      ul_config_pdu_list[i].pdu_type == NFAPI_UL_CONFIG_ULSCH_PDU_TYPE ||
      ul_config_pdu_list[i].pdu_type == NFAPI_UL_CONFIG_ULSCH_HARQ_PDU_TYPE ||
      ul_config_pdu_list[i].pdu_type == NFAPI_UL_CONFIG_ULSCH_CQI_RI_PDU_TYPE ||
      ul_config_pdu_list[i].pdu_type == NFAPI_UL_CONFIG_ULSCH_CQI_HARQ_RI_PDU_TYPE ||
      ul_config_pdu_list[i].pdu_type == NFAPI_UL_CONFIG_UCI_HARQ_PDU_TYPE ||
      ul_config_pdu_list[i].pdu_type == NFAPI_UL_CONFIG_UCI_SR_PDU_TYPE ||
      ul_config_pdu_list[i].pdu_type == NFAPI_UL_CONFIG_UCI_SR_HARQ_PDU_TYPE
    ) {
      //LOG_D(PHY, "%s() handle_nfapi_ul_pdu() for PDU:%d\n", __FUNCTION__, i);
      handle_nfapi_ul_pdu(eNB,proc,&ul_config_pdu_list[i],curr_sfn,curr_sf,req->ul_config_request_body.srs_present);
    } else {
      NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() PDU:%i UNKNOWN type :%d\n", __FUNCTION__, i, ul_config_pdu_list[i].pdu_type);
    }
  }

  return 0;
}

int pnf_phy_lbt_dl_config_req(nfapi_pnf_p7_config_t *config, nfapi_lbt_dl_config_request_t *req) {
  //printf("[PNF] lbt dl config request\n");
  return 0;
}

int pnf_phy_ue_release_req(nfapi_pnf_p7_config_t* config, nfapi_ue_release_request_t* req) {
  if (req->ue_release_request_body.number_of_TLVs==0)
    return -1;

  release_rntis.number_of_TLVs = req->ue_release_request_body.number_of_TLVs;
  memcpy(&release_rntis.ue_release_request_TLVs_list, req->ue_release_request_body.ue_release_request_TLVs_list, sizeof(nfapi_ue_release_request_TLVs_t)*req->ue_release_request_body.number_of_TLVs);
  return 0;
}

int pnf_phy_vendor_ext(nfapi_pnf_p7_config_t *config, nfapi_p7_message_header_t *msg) {
  if(msg->message_id == P7_VENDOR_EXT_REQ) {
    //vendor_ext_p7_req* req = (vendor_ext_p7_req*)msg;
    //printf("[PNF] vendor request (1:%d 2:%d)\n", req->dummy1, req->dummy2);
  } else {
    printf("[PNF] unknown vendor ext\n");
  }

  return 0;
}

int pnf_phy_pack_p7_vendor_extension(nfapi_p7_message_header_t *header, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p7_codec_config_t *codex) {
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
  if(header->message_id == P7_VENDOR_EXT_IND) {
    vendor_ext_p7_ind *ind = (vendor_ext_p7_ind *)(header);

    if(!push16(ind->error_code, ppWritePackedMsg, end))
      return 0;

    return 1;
  }

  return -1;
}

int pnf_phy_unpack_p7_vendor_extension(nfapi_p7_message_header_t *header, uint8_t **ppReadPackedMessage, uint8_t *end, nfapi_p7_codec_config_t *codec) {
  if(header->message_id == P7_VENDOR_EXT_REQ) {
    //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
    vendor_ext_p7_req *req = (vendor_ext_p7_req *)(header);

    if(!(pull16(ppReadPackedMessage, &req->dummy1, end) &&
         pull16(ppReadPackedMessage, &req->dummy2, end)))
      return 0;

    return 1;
  }

  return -1;
}

int pnf_phy_unpack_vendor_extension_tlv(nfapi_tl_t *tl, uint8_t **ppReadPackedMessage, uint8_t *end, void **ve, nfapi_p7_codec_config_t *config) {
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "pnf_phy_unpack_vendor_extension_tlv\n");
  switch(tl->tag) {
    case VENDOR_EXT_TLV_1_TAG:
      *ve = malloc(sizeof(vendor_ext_tlv_1));

      if(!pull32(ppReadPackedMessage, &((vendor_ext_tlv_1 *)(*ve))->dummy, end))
        return 0;

      return 1;
      break;
  }

  return -1;
}

int pnf_phy_pack_vendor_extention_tlv(void *ve, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p7_codec_config_t *config) {
  //printf("%s\n", __FUNCTION__);
  (void)ve;
  (void)ppWritePackedMsg;
  return -1;
}

int pnf_sim_unpack_vendor_extension_tlv(nfapi_tl_t *tl, uint8_t **ppReadPackedMessage, uint8_t *end, void **ve, nfapi_p4_p5_codec_config_t *config) {
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "pnf_sim_unpack_vendor_extension_tlv\n");
  switch(tl->tag) {
    case VENDOR_EXT_TLV_2_TAG:
      *ve = malloc(sizeof(vendor_ext_tlv_2));

      if(!pull32(ppReadPackedMessage, &((vendor_ext_tlv_2 *)(*ve))->dummy, end))
        return 0;

      return 1;
      break;
  }

  return -1;
}

int pnf_sim_pack_vendor_extention_tlv(void *ve, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  //printf("%s\n", __FUNCTION__);
  (void)ve;
  (void)ppWritePackedMsg;

  return -1;
}

nfapi_dl_config_request_t dummy_dl_config_req;
nfapi_tx_request_t dummy_tx_req;

nfapi_pnf_p7_subframe_buffer_t dummy_subframe;

int start_request(nfapi_pnf_config_t *config, nfapi_pnf_phy_config_t *phy, nfapi_start_request_t *req) {
  printf("[PNF] Received NFAPI_START_REQ phy_id:%d\n", req->header.phy_id);
  pnf_info *pnf = (pnf_info *)(config->user_data);
  phy_info *phy_info = pnf->phys;
  nfapi_pnf_p7_config_t *p7_config = nfapi_pnf_p7_config_create();
  p7_config->phy_id = phy->phy_id;
  p7_config->remote_p7_port = phy_info->remote_port;
  p7_config->remote_p7_addr = phy_info->remote_addr;
  p7_config->local_p7_port = phy_info->udp.rx_port;
  p7_config->local_p7_addr = phy_info->local_addr;
  printf("[PNF] P7 remote:%s:%d local:%s:%d\n", p7_config->remote_p7_addr, p7_config->remote_p7_port, p7_config->local_p7_addr, p7_config->local_p7_port);
  p7_config->user_data = phy_info;
  p7_config->malloc = &pnf_allocate;
  p7_config->free = &pnf_deallocate;
  p7_config->codec_config.allocate = &pnf_allocate;
  p7_config->codec_config.deallocate = &pnf_deallocate;
  p7_config->trace = &pnf_nfapi_trace;
  phy->user_data = p7_config;
  p7_config->subframe_buffer_size = phy_info->timing_window;
  printf("subframe_buffer_size configured using phy_info->timing_window:%d\n", phy_info->timing_window);

  if(phy_info->timing_info_mode & 0x1) {
    p7_config->timing_info_mode_periodic = 1;
    p7_config->timing_info_period = phy_info->timing_info_period;
  }

  if(phy_info->timing_info_mode & 0x2) {
    p7_config->timing_info_mode_aperiodic = 1;
  }

  p7_config->dl_config_req = &pnf_phy_dl_config_req;
  p7_config->ul_config_req = &pnf_phy_ul_config_req;
  p7_config->hi_dci0_req = &pnf_phy_hi_dci0_req;
  p7_config->tx_req = &pnf_phy_tx_req;
  p7_config->lbt_dl_config_req = &pnf_phy_lbt_dl_config_req;
  p7_config->ue_release_req = &pnf_phy_ue_release_req;
  if (NFAPI_MODE==NFAPI_UE_STUB_PNF || NFAPI_MODE==NFAPI_MODE_STANDALONE_PNF) {
    p7_config->dl_config_req = NULL;
    p7_config->ul_config_req = NULL;
    p7_config->hi_dci0_req = NULL;
    p7_config->tx_req = NULL;
  } else {
    p7_config->dl_config_req = &pnf_phy_dl_config_req;
    p7_config->ul_config_req = &pnf_phy_ul_config_req;
    p7_config->hi_dci0_req = &pnf_phy_hi_dci0_req;
    p7_config->tx_req = &pnf_phy_tx_req;
  }

  p7_config->lbt_dl_config_req = &pnf_phy_lbt_dl_config_req;
  memset(&dummy_dl_config_req, 0, sizeof(dummy_dl_config_req));
  dummy_dl_config_req.dl_config_request_body.tl.tag=NFAPI_DL_CONFIG_REQUEST_BODY_TAG;
  dummy_dl_config_req.dl_config_request_body.number_pdcch_ofdm_symbols=1;
  dummy_dl_config_req.dl_config_request_body.number_dci=0;
  dummy_dl_config_req.dl_config_request_body.number_pdu=0;
  dummy_dl_config_req.dl_config_request_body.number_pdsch_rnti=0;
  dummy_dl_config_req.dl_config_request_body.transmission_power_pcfich=6000;
  dummy_dl_config_req.dl_config_request_body.dl_config_pdu_list=0;
  memset(&dummy_tx_req, 0, sizeof(dummy_tx_req));
  dummy_tx_req.tx_request_body.number_of_pdus=0;
  dummy_tx_req.tx_request_body.tl.tag=NFAPI_TX_REQUEST_BODY_TAG;
  dummy_subframe.dl_config_req = &dummy_dl_config_req;
  dummy_subframe.tx_req = 0;//&dummy_tx_req;
  dummy_subframe.ul_config_req=0;
  dummy_subframe.hi_dci0_req=0;
  dummy_subframe.lbt_dl_config_req=0;
  p7_config->dummy_subframe = dummy_subframe;
  p7_config->vendor_ext = &pnf_phy_vendor_ext;
  p7_config->allocate_p7_vendor_ext = &pnf_phy_allocate_p7_vendor_ext;
  p7_config->deallocate_p7_vendor_ext = &pnf_phy_deallocate_p7_vendor_ext;
  p7_config->codec_config.unpack_p7_vendor_extension = &pnf_phy_unpack_p7_vendor_extension;
  p7_config->codec_config.pack_p7_vendor_extension = &pnf_phy_pack_p7_vendor_extension;
  p7_config->codec_config.unpack_vendor_extension_tlv = &pnf_phy_unpack_vendor_extension_tlv;
  p7_config->codec_config.pack_vendor_extension_tlv = &pnf_phy_pack_vendor_extention_tlv;
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Creating P7 thread %s\n", __FUNCTION__);
  pthread_t p7_thread;
  pthread_create(&p7_thread, NULL, &pnf_p7_thread_start, p7_config);
  //((pnf_phy_user_data_t*)(phy_info->fapi->user_data))->p7_config = p7_config;
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Calling l1_north_init_eNB() %s\n", __FUNCTION__);
  l1_north_init_eNB();
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] HACK - Set p7_config global ready for subframe ind%s\n", __FUNCTION__);
  p7_config_g = p7_config;

  // Need to wait for main thread to create RU structures
  while(config_sync_var<0) {
    usleep(5000000);
    printf("[PNF] waiting for OAI to be configured (eNB/RU)\n");
  }

  printf("[PNF] OAI eNB/RU configured\n");
  //printf("[PNF] About to call phy_init_RU() for RC.ru[0]:%p\n", RC.ru[0]);
  //phy_init_RU(RC.ru[0]);
  printf("[PNF] About to call init_eNB_afterRU()\n");

  if (NFAPI_MODE!=NFAPI_UE_STUB_PNF && NFAPI_MODE!=NFAPI_MODE_STANDALONE_PNF) {
    init_eNB_afterRU();
  }

  // Signal to main thread that it can carry on - otherwise RU will startup too quickly and it is not initialised
  {
    pthread_mutex_lock(&nfapi_sync_mutex);
    nfapi_sync_var=0;
    pthread_cond_broadcast(&nfapi_sync_cond);
    pthread_mutex_unlock(&nfapi_sync_mutex);
  }

  while(sync_var<0) {
    usleep(5000000);
    printf("[PNF] waiting for OAI to be started\n");
  }

  printf("[PNF] Sending PNF_START_RESP\n");
  printf("[PNF] Sending first P7 subframe ind\n");
  nfapi_pnf_p7_subframe_ind(p7_config, p7_config->phy_id, 0); // SFN_SF set to zero - correct???
  printf("[PNF] Sent first P7 subframe ind\n");
  return 0;
}

int nr_start_request(nfapi_pnf_config_t *config, nfapi_pnf_phy_config_t *phy,  nfapi_nr_start_request_scf_t *req) {
  printf("[PNF] Received NFAPI_START_REQ phy_id:%d\n", req->header.phy_id);
  pnf_info *pnf = (pnf_info *)(config->user_data);
  phy_info *phy_info = pnf->phys;
  nfapi_pnf_p7_config_t *p7_config = nfapi_pnf_p7_config_create();
  p7_config->phy_id = phy->phy_id;
  p7_config->remote_p7_port = phy_info->remote_port;
  p7_config->remote_p7_addr = phy_info->remote_addr;

  p7_config->local_p7_port = phy_info->udp.rx_port;
  p7_config->local_p7_addr = phy_info->local_addr;
  printf("[PNF] P7 remote:%s:%d local:%s:%d\n", p7_config->remote_p7_addr, p7_config->remote_p7_port, p7_config->local_p7_addr, p7_config->local_p7_port);
  p7_config->user_data = phy_info;
  p7_config->malloc = &pnf_allocate;
  p7_config->free = &pnf_deallocate;
  p7_config->codec_config.allocate = &pnf_allocate;
  p7_config->codec_config.deallocate = &pnf_deallocate;
  p7_config->trace = &pnf_nfapi_trace;
  phy->user_data = p7_config;
  p7_config->subframe_buffer_size = phy_info->timing_window;
  p7_config->slot_buffer_size = phy_info->timing_window; // TODO: check if correct for NR
  printf("subframe_buffer_size configured using phy_info->timing_window:%d\n", phy_info->timing_window);

  if(phy_info->timing_info_mode & 0x1) {
    p7_config->timing_info_mode_periodic = 1;
    p7_config->timing_info_period = phy_info->timing_info_period;
  }

  if(phy_info->timing_info_mode & 0x2) {
    p7_config->timing_info_mode_aperiodic = 1;
  }

  // NR
  p7_config->dl_tti_req_fn  = &pnf_phy_dl_tti_req;
  p7_config->ul_tti_req_fn  = &pnf_phy_ul_tti_req;
  p7_config->ul_dci_req_fn  = &pnf_phy_ul_dci_req;
  p7_config->tx_data_req_fn = &pnf_phy_tx_data_req;

  // LTE
  p7_config->dl_config_req = &pnf_phy_dl_config_req;
  p7_config->ul_config_req = &pnf_phy_ul_config_req;
  p7_config->hi_dci0_req = &pnf_phy_hi_dci0_req;
  p7_config->tx_req = &pnf_phy_tx_req;
  p7_config->lbt_dl_config_req = &pnf_phy_lbt_dl_config_req;
  p7_config->ue_release_req = &pnf_phy_ue_release_req;
  if (NFAPI_MODE==NFAPI_UE_STUB_PNF) {
    p7_config->dl_config_req = NULL;
    p7_config->ul_config_req = NULL;
    p7_config->hi_dci0_req = NULL;
    p7_config->tx_req = NULL;
  } else {
    p7_config->dl_config_req = &pnf_phy_dl_config_req;
    p7_config->ul_config_req = &pnf_phy_ul_config_req;
    p7_config->hi_dci0_req = &pnf_phy_hi_dci0_req;
    p7_config->tx_req = &pnf_phy_tx_req;
  }

  p7_config->lbt_dl_config_req = &pnf_phy_lbt_dl_config_req;
  memset(&dummy_dl_config_req, 0, sizeof(dummy_dl_config_req));
  dummy_dl_config_req.dl_config_request_body.tl.tag=NFAPI_DL_CONFIG_REQUEST_BODY_TAG;
  dummy_dl_config_req.dl_config_request_body.number_pdcch_ofdm_symbols=1;
  dummy_dl_config_req.dl_config_request_body.number_dci=0;
  dummy_dl_config_req.dl_config_request_body.number_pdu=0;
  dummy_dl_config_req.dl_config_request_body.number_pdsch_rnti=0;
  dummy_dl_config_req.dl_config_request_body.transmission_power_pcfich=6000;
  dummy_dl_config_req.dl_config_request_body.dl_config_pdu_list=0;
  memset(&dummy_tx_req, 0, sizeof(dummy_tx_req));
  dummy_tx_req.tx_request_body.number_of_pdus=0;
  dummy_tx_req.tx_request_body.tl.tag=NFAPI_TX_REQUEST_BODY_TAG;
  dummy_subframe.dl_config_req = &dummy_dl_config_req;
  dummy_subframe.tx_req = 0;//&dummy_tx_req;
  dummy_subframe.ul_config_req=0;
  dummy_subframe.hi_dci0_req=0;
  dummy_subframe.lbt_dl_config_req=0;
  p7_config->dummy_subframe = dummy_subframe;
  p7_config->vendor_ext = &pnf_phy_vendor_ext;
  p7_config->allocate_p7_vendor_ext = &pnf_phy_allocate_p7_vendor_ext;
  p7_config->deallocate_p7_vendor_ext = &pnf_phy_deallocate_p7_vendor_ext;
  p7_config->codec_config.unpack_p7_vendor_extension = &pnf_phy_unpack_p7_vendor_extension;
  p7_config->codec_config.pack_p7_vendor_extension = &pnf_phy_pack_p7_vendor_extension;
  p7_config->codec_config.unpack_vendor_extension_tlv = &pnf_phy_unpack_vendor_extension_tlv;
  p7_config->codec_config.pack_vendor_extension_tlv = &pnf_phy_pack_vendor_extention_tlv;
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Creating P7 thread %s\n", __FUNCTION__);
  pthread_t p7_thread;
  pthread_create(&p7_thread, NULL, &pnf_nr_p7_thread_start, p7_config);
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Calling l1_north_init_eNB() %s\n", __FUNCTION__);
  l1_north_init_gNB();
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] HACK - Set p7_config global ready for subframe ind%s\n", __FUNCTION__);
  p7_config_g = p7_config;

  // Need to wait for main thread to create RU structures
  while(config_sync_var<0) {
    usleep(5000000);
    printf("[PNF] waiting for OAI to be configured (eNB/RU)\n");
  }

  printf("[PNF] OAI eNB/RU configured\n");
  //printf("[PNF] About to call phy_init_RU() for RC.ru[0]:%p\n", RC.ru[0]);
  //phy_init_RU(RC.ru[0]);
  printf("[PNF] About to call init_eNB_afterRU()\n");

  if (NFAPI_MODE!=NFAPI_UE_STUB_PNF) {
    init_eNB_afterRU();
  }

  // Signal to main thread that it can carry on - otherwise RU will startup too quickly and it is not initialised
  {
    pthread_mutex_lock(&nfapi_sync_mutex);
    nfapi_sync_var=0;
    pthread_cond_broadcast(&nfapi_sync_cond);
    pthread_mutex_unlock(&nfapi_sync_mutex);
  }

  while(sync_var<0) {
    usleep(5000000);
    printf("[PNF] waiting for OAI to be started\n");
  }

  printf("[PNF] Sending PNF_START_RESP\n");
  printf("[PNF] Sending first P7 slot indication\n");
#if 1
  nfapi_pnf_p7_slot_ind(p7_config, p7_config->phy_id, 0, 0);
  printf("[PNF] Sent first P7 slot ind\n");
#else
  nfapi_pnf_p7_subframe_ind(p7_config, p7_config->phy_id, 0); // SFN_SF set to zero - correct???
  printf("[PNF] Sent first P7 subframe ind\n");
#endif
  
  
  return 0;
}

nfapi_p4_p5_message_header_t *pnf_sim_allocate_p4_p5_vendor_ext(uint16_t message_id, uint16_t *msg_size) {
  if(message_id == P5_VENDOR_EXT_REQ) {
    (*msg_size) = sizeof(vendor_ext_p5_req);
    return (nfapi_p4_p5_message_header_t *)malloc(sizeof(vendor_ext_p5_req));
  }

  return 0;
}

void pnf_sim_deallocate_p4_p5_vendor_ext(nfapi_p4_p5_message_header_t *header) {
  free(header);
}

int pnf_sim_pack_p4_p5_vendor_extension(nfapi_p4_p5_message_header_t *header, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
  if(header->message_id == P5_VENDOR_EXT_RSP) {
    vendor_ext_p5_rsp *rsp = (vendor_ext_p5_rsp *)(header);
    return (!push16(rsp->error_code, ppWritePackedMsg, end));
  }

  return 0;
}

int pnf_sim_unpack_p4_p5_vendor_extension(nfapi_p4_p5_message_header_t *header, uint8_t **ppReadPackedMessage, uint8_t *end, nfapi_p4_p5_codec_config_t *codec) {
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
  if(header->message_id == P5_VENDOR_EXT_REQ) {
    vendor_ext_p5_req *req = (vendor_ext_p5_req *)(header);
    return (!(pull16(ppReadPackedMessage, &req->dummy1, end) &&
              pull16(ppReadPackedMessage, &req->dummy2, end)));
    //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s (%d %d)\n", __FUNCTION__, req->dummy1, req->dummy2);
  }

  return 0;
}

/*------------------------------------------------------------------------------*/
void *pnf_start_thread(void *ptr) {
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] IN PNF NFAPI start thread %s\n", __FUNCTION__);
  struct sched_param sp;
  sp.sched_priority = 20;
  pthread_setschedparam(pthread_self(),SCHED_FIFO,&sp);
  return (void *)0;
}

void *pnf_nr_start_thread(void *ptr) {
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] IN PNF NFAPI start thread %s\n", __FUNCTION__);
  struct sched_param sp;
  sp.sched_priority = 20;
  pthread_setschedparam(pthread_self(),SCHED_FIFO,&sp);
  return (void *)0;
}

void configure_nr_nfapi_pnf(char *vnf_ip_addr, int vnf_p5_port, char *pnf_ip_addr, int pnf_p7_port, int vnf_p7_port) {
  printf("%s() PNF\n\n\n\n\n\n", __FUNCTION__);
    nfapi_setmode(NFAPI_MODE_PNF);  // PNF!

  nfapi_pnf_config_t *config = nfapi_pnf_config_create();
  config->vnf_ip_addr = vnf_ip_addr;
  config->vnf_p5_port = vnf_p5_port;
  pnf.phys[0].udp.enabled = 1;
  pnf.phys[0].udp.rx_port = pnf_p7_port;
  pnf.phys[0].udp.tx_port = vnf_p7_port;
  strcpy(pnf.phys[0].udp.tx_addr, vnf_ip_addr);
  strcpy(pnf.phys[0].local_addr, pnf_ip_addr);
  printf("%s() VNF:%s:%d PNF_PHY[addr:%s UDP:tx_addr:%s:%u rx:%u]\n",
         __FUNCTION__,
         config->vnf_ip_addr,
         config->vnf_p5_port,
         pnf.phys[0].local_addr,
         pnf.phys[0].udp.tx_addr,
         pnf.phys[0].udp.tx_port,
         pnf.phys[0].udp.rx_port);
  config->nr_start_req = &nr_start_request;
  config->trace = &pnf_nfapi_trace;
  config->user_data = &pnf;
  // To allow custom vendor extentions to be added to nfapi
  config->codec_config.unpack_vendor_extension_tlv = &pnf_sim_unpack_vendor_extension_tlv;
  config->codec_config.pack_vendor_extension_tlv = &pnf_sim_pack_vendor_extention_tlv;
  config->allocate_p4_p5_vendor_ext = &pnf_sim_allocate_p4_p5_vendor_ext;
  config->deallocate_p4_p5_vendor_ext = &pnf_sim_deallocate_p4_p5_vendor_ext;
  config->codec_config.unpack_p4_p5_vendor_extension = &pnf_sim_unpack_p4_p5_vendor_extension;
  config->codec_config.pack_p4_p5_vendor_extension = &pnf_sim_pack_p4_p5_vendor_extension;
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Creating PNF NFAPI start thread %s\n", __FUNCTION__);
  pthread_create(&pnf_start_pthread, NULL, &pnf_nr_start_thread, config);
  pthread_setname_np(pnf_start_pthread, "NFAPI_PNF");
}

void configure_nfapi_pnf(char *vnf_ip_addr, int vnf_p5_port, char *pnf_ip_addr, int pnf_p7_port, int vnf_p7_port) {
  printf("%s() PNF\n\n\n\n\n\n", __FUNCTION__);

  if(NFAPI_MODE!=NFAPI_UE_STUB_PNF && NFAPI_MODE!=NFAPI_MODE_STANDALONE_PNF) {
    nfapi_setmode(NFAPI_MODE_PNF);  // PNF!
  }
  nfapi_pnf_config_t *config = nfapi_pnf_config_create();
  config->vnf_ip_addr = vnf_ip_addr;
  config->vnf_p5_port = vnf_p5_port;
  pnf.phys[0].udp.enabled = 1;
  pnf.phys[0].udp.rx_port = pnf_p7_port;
  pnf.phys[0].udp.tx_port = vnf_p7_port;
  strcpy(pnf.phys[0].udp.tx_addr, vnf_ip_addr);
  strcpy(pnf.phys[0].local_addr, pnf_ip_addr);
  printf("%s() VNF:%s:%d PNF_PHY[addr:%s UDP:tx_addr:%s:%u rx:%u]\n",
         __FUNCTION__,
         config->vnf_ip_addr,
         config->vnf_p5_port,
         pnf.phys[0].local_addr,
         pnf.phys[0].udp.tx_addr,
         pnf.phys[0].udp.tx_port,
         pnf.phys[0].udp.rx_port);
  config->start_req = &start_request;
  config->trace = &pnf_nfapi_trace;
  config->user_data = &pnf;
  // To allow custom vendor extentions to be added to nfapi
  config->codec_config.unpack_vendor_extension_tlv = &pnf_sim_unpack_vendor_extension_tlv;
  config->codec_config.pack_vendor_extension_tlv = &pnf_sim_pack_vendor_extention_tlv;
  config->allocate_p4_p5_vendor_ext = &pnf_sim_allocate_p4_p5_vendor_ext;
  config->deallocate_p4_p5_vendor_ext = &pnf_sim_deallocate_p4_p5_vendor_ext;
  config->codec_config.unpack_p4_p5_vendor_extension = &pnf_sim_unpack_p4_p5_vendor_extension;
  config->codec_config.pack_p4_p5_vendor_extension = &pnf_sim_pack_p4_p5_vendor_extension;
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Creating PNF NFAPI start thread %s\n", __FUNCTION__);
  pthread_create(&pnf_start_pthread, NULL, &pnf_start_thread, config);
  pthread_setname_np(pnf_start_pthread, "NFAPI_PNF");
}

void oai_subframe_ind(uint16_t sfn, uint16_t sf) {
  //LOG_D(PHY,"%s(sfn:%d, sf:%d)\n", __FUNCTION__, sfn, sf);

  //TODO FIXME - HACK - using a global to bodge it in
  if (p7_config_g != NULL && sync_var==0) {
    uint16_t sfn_sf_tx = sfn<<4 | sf; 

    if ((sfn % 100 == 0) && sf==0) {
      NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] (sfn:%u sf:%u) SFN/SF(TX):%u\n", sfn, sf, NFAPI_SFNSF2DEC(sfn_sf_tx));
    }

    int subframe_ret = nfapi_pnf_p7_subframe_ind(p7_config_g, p7_config_g->phy_id, sfn_sf_tx);

    if (subframe_ret) {
      NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] (frame:%u subframe:%u) SFN/SF(TX):%u - PROBLEM with pnf_p7_subframe_ind()\n", sfn, sf, sfn_sf_tx);
    } else {
      //NFAPI_TRACE(NFAPI_TRACE_INFO, "***NFAPI subframe handler finished *** \n");
    }
  } else {
  }
}

void handle_nr_slot_ind(uint16_t sfn, uint16_t slot) {

    //send VNF slot indication, which is aligned with TX thread, so that it can call the scheduler
    nfapi_nr_slot_indication_scf_t *ind;
    ind = (nfapi_nr_slot_indication_scf_t *) malloc(sizeof(nfapi_nr_slot_indication_scf_t));
    uint8_t slot_ahead = 6;
    uint32_t sfn_slot_tx = sfnslot_add_slot(sfn, slot, slot_ahead);
    uint16_t sfn_tx = NFAPI_SFNSLOT2SFN(sfn_slot_tx);
    uint8_t slot_tx = NFAPI_SFNSLOT2SLOT(sfn_slot_tx);

    ind->sfn = sfn_tx;
    ind->slot = slot_tx;
    oai_nfapi_nr_slot_indication(ind); 

    //copy data from appropriate p7 slot buffers into channel structures for PHY processing
    nfapi_pnf_p7_slot_ind(p7_config_g, p7_config_g->phy_id, sfn, slot); 

    return;
}

int oai_nfapi_rach_ind(nfapi_rach_indication_t *rach_ind) {
  rach_ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  LOG_D(PHY, "%s() sfn_sf:%d preambles:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(rach_ind->sfn_sf), rach_ind->rach_indication_body.number_of_preambles);
  return nfapi_pnf_p7_rach_ind(p7_config_g, rach_ind);
}

int oai_nfapi_harq_indication(nfapi_harq_indication_t *harq_ind) {
  harq_ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  harq_ind->header.message_id = NFAPI_HARQ_INDICATION;
  LOG_D(PHY, "%s() sfn_sf:%d number_of_harqs:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(harq_ind->sfn_sf), harq_ind->harq_indication_body.number_of_harqs);
  int retval = nfapi_pnf_p7_harq_ind(p7_config_g, harq_ind);

  if (retval != 0)
    LOG_E(PHY, "%s() sfn_sf:%d number_of_harqs:%d nfapi_pnf_p7_harq_ind()=%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(harq_ind->sfn_sf), harq_ind->harq_indication_body.number_of_harqs, retval);

  return retval;
}

int oai_nfapi_crc_indication(nfapi_crc_indication_t *crc_ind) {
  crc_ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  crc_ind->header.message_id = NFAPI_CRC_INDICATION;
  //LOG_D(PHY, "%s() sfn_sf:%d number_of_crcs:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(crc_ind->sfn_sf), crc_ind->crc_indication_body.number_of_crcs);
  return nfapi_pnf_p7_crc_ind(p7_config_g, crc_ind);
}

int oai_nfapi_cqi_indication(nfapi_cqi_indication_t *ind) {
  ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  ind->header.message_id = NFAPI_RX_CQI_INDICATION;
  //LOG_D(PHY, "%s() sfn_sf:%d number_of_cqis:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(ind->sfn_sf), ind->cqi_indication_body.number_of_cqis);
  return nfapi_pnf_p7_cqi_ind(p7_config_g, ind);
}

int oai_nfapi_rx_ind(nfapi_rx_indication_t *ind) {
  ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  ind->header.message_id = NFAPI_RX_ULSCH_INDICATION;
  int retval = nfapi_pnf_p7_rx_ind(p7_config_g, ind);
  //LOG_D(PHY,"%s() SFN/SF:%d pdus:%d retval:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(ind->sfn_sf), ind->rx_indication_body.number_of_pdus, retval);
  //free(ind.rx_indication_body.rx_pdu_list);
  return retval;
}

int oai_nfapi_sr_indication(nfapi_sr_indication_t *ind) {
  ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  int retval = nfapi_pnf_p7_sr_ind(p7_config_g, ind);
  //LOG_D(PHY,"%s() SFN/SF:%d srs:%d retval:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(ind->sfn_sf), ind->sr_indication_body.number_of_srs, retval);
  //free(ind.rx_indication_body.rx_pdu_list);
  return retval;
}

//NR UPLINK INDICATION

int oai_nfapi_nr_slot_indication(nfapi_nr_slot_indication_scf_t *ind) {
  ind->header.phy_id = 1;
  ind->header.message_id = NFAPI_NR_PHY_MSG_TYPE_SLOT_INDICATION;
  return nfapi_pnf_p7_nr_slot_ind(p7_config_g, ind);
}

int oai_nfapi_nr_rx_data_indication(nfapi_nr_rx_data_indication_t *ind) {
  ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  ind->header.message_id = NFAPI_NR_PHY_MSG_TYPE_RX_DATA_INDICATION;
  return nfapi_pnf_p7_nr_rx_data_ind(p7_config_g, ind);
}

int oai_nfapi_nr_crc_indication(nfapi_nr_crc_indication_t *ind) {
  ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  ind->header.message_id = NFAPI_NR_PHY_MSG_TYPE_CRC_INDICATION;
  return nfapi_pnf_p7_nr_crc_ind(p7_config_g, ind);
}

int oai_nfapi_nr_srs_indication(nfapi_nr_srs_indication_t *ind) {
  ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  ind->header.message_id = NFAPI_NR_PHY_MSG_TYPE_SRS_INDICATION;
  return nfapi_pnf_p7_nr_srs_ind(p7_config_g, ind);
}

int oai_nfapi_nr_uci_indication(nfapi_nr_uci_indication_t *ind) {
  ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  ind->header.message_id = NFAPI_NR_PHY_MSG_TYPE_UCI_INDICATION;
  return nfapi_pnf_p7_nr_uci_ind(p7_config_g, ind);
}

int oai_nfapi_nr_rach_indication(nfapi_nr_rach_indication_t *ind) {
  ind->header.phy_id = 1; // HACK TODO FIXME - need to pass this around!!!!
  ind->header.message_id = NFAPI_NR_PHY_MSG_TYPE_RACH_INDICATION;
  return nfapi_pnf_p7_nr_rach_ind(p7_config_g, ind);
}

