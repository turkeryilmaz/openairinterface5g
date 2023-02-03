#ifndef DL_RRC_MSG_F1AP_H
#define DL_RRC_MSG_F1AP_H

#include <stdint.h>

#include "../byte_array.h"
#include "rat_freq_prio_info.h"

typedef enum{
  
  TRUE_EXECUTE_DUP_DL_RRC_MSG,

  END_EXECUTE_DUP_DL_RRC_MSG,

} execute_dup_e;

typedef enum{

  TRUE_RRC_DELIVERY_STATUS_REQ,

  END_RRC_DELIVERY_STATUS_REQ,

} rrc_delivery_status_req_e;

typedef enum{

  TRUE_UE_CTX_NOT_RETRIABLE,

  END_UE_CTX_NOT_RETRIABLE,

} ue_ctx_not_retriable_e ;




typedef struct{

  // Message Type
  // Mandatory 
  // 9.3.1.1

  // gNB-CU UE F1AP ID
  // Mandatory
  // 9.3.1.4 [0-2^32-1]
  uint32_t gnb_cu_ue; 

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5 [0-2^32-1]
  uint32_t gnb_du_ue; 

  // old gNB-DU UE F1AP ID
  // Optional
  // 9.3.1.5 [0-2^32-1]
  uint32_t* old_gnb_du_ue; 

  // SRB ID
  // Mandatory
  // 9.3.1.7
  uint8_t srb_id; // [0-3]

  // Execute Duplication
  // Optional
  execute_dup_e* exe_dup;

  // RRC-Container
  // Mandatory
  // 9.3.1.6
  // Includes the DL-DCCH-
  // Message IE as defined
  // in subclause 6.2 of TS
  // 38.331 [8]
  byte_array_t rrc_cntnr;

  // RAT-Frequency Priority Information
  // Optional
  // 9.3.1.34
  rat_freq_prio_info_t* rat_freq;

  // RRC Delivery Status Request
  // Optional
  rrc_delivery_status_req_e* rrc_delivery_status_req;

  // UE Context not retrievable
  // Optional
  ue_ctx_not_retriable_e* ue_ctx_not_retriable;

  // Redirected RRC message
  // Optional
  //  9.3.1.6
  byte_array_t* redirected_rrc_msg;

  //PLMN Assistance Info for Network Sharing
  // Optional
  // 9.3.1.14
  byte_array_t* plmn_assis_info_netwrk_shr; // [size(3)]

  // New gNB-CU UE F1AP ID
  // Optional
  // 9.3.1.4
  uint32_t* new_gnb_cu_ue; 

  // Additional RRM Policy Index
  // Optional
  // 9.3.1.90
  uint32_t* add_rrm_pol_idx; // bit string of 4

} dl_rrc_msg_t;

void free_dl_rrc_msg(dl_rrc_msg_t* src);

#endif

