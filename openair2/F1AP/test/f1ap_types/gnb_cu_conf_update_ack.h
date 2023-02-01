#ifndef GNB_CU_CONF_UPDATE_ACK_MIR_H
#define GNB_CU_CONF_UPDATE_ACK_MIR_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "cells_failed_to_activate.h"
#include "criticality_diagnostic_f1ap.h"
#include "ded_si_del_need.h"
#include "gnb_cu_tnl_assoc_stp.h"
#include "gnb_cu_tnl_assoc_failed_stp.h"
#include "trans_layer_add_info.h"

typedef struct {

  // Mandatory
  // Transaction ID 9.3.1.23
  uint8_t trans_id; 

  // [0-512]
  // Cells Failed to be Activated List
  size_t sz_cells_failed_to_activate;
  cells_failed_to_activate_t* cells_failed_to_activate;

  // Optional
  // Criticality Diagnostics 9.3.1.3 
  criticallity_diagnostic_f1ap_t* crit_diagn;

  // [0-32]
  // gNB-CU TNL Association Setup List
  size_t sz_gnb_cu_tnl_assoc_stp;
  gnb_cu_tnl_assoc_stp_t* gnb_cu_tnl_assoc_stp;

  // [0-32] 
  // gNB-CU TNL Association Failed toSetup List
  size_t sz_gnb_cu_tnl_assoc_failed_stp;
  gnb_cu_tnl_assoc_failed_stp_t* gnb_cu_tnl_assoc_failed_stp; 

  // [0 - 65536]
  // Dedicated SI Delivery Needed UE List
  size_t sz_ded_si_del_need;
  ded_si_del_need_t* ded_si_del_need;

  // Optional
  // 9.3.2.5
  // Transport Layer Address Info
  trans_layer_add_info_t* trans_layer_addr; 

} gnb_cu_conf_update_ack_t;

void free_gnb_cu_conf_update_ack(gnb_cu_conf_update_ack_t * src);

bool eq_gnb_cu_conf_update_ack(gnb_cu_conf_update_ack_t const* m0, gnb_cu_conf_update_ack_t const* m1);

#endif

