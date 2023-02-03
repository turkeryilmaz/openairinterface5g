#include "dl_rrc_msg.h"

#include <assert.h>

void free_dl_rrc_msg(dl_rrc_msg_t* src)
{
  assert(src != NULL);

  // Message Type
  // Mandatory 
  // 9.3.1.1

  // gNB-CU UE F1AP ID
  // Mandatory
  // 9.3.1.4 [0-2^32-1]
//  uint32_t gnb_cu_ue; 

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5 [0-2^32-1]
//  uint32_t gnb_du_ue; 

  // old gNB-DU UE F1AP ID
  // Optional
  // 9.3.1.5 [0-2^32-1]
  assert(src->old_gnb_du_ue == NULL && "Not implemented");

  // SRB ID
  // Mandatory
  // 9.3.1.7
  //uint8_t srb_id; // [0-3]

  // Execute Duplication
  // Optional
  assert(src->exe_dup == NULL && "Not implemented");

  // RRC-Container
  // Mandatory
  // 9.3.1.6
  // Includes the DL-DCCH-
  // Message IE as defined
  // in subclause 6.2 of TS
  // 38.331 [8]
  free_byte_array(src->rrc_cntnr);

  // RAT-Frequency Priority Information
  // Optional
  // 9.3.1.34
  assert(src->rat_freq == NULL && "Not implemented");

  // RRC Delivery Status Request
  // Optional
  assert(src->rrc_delivery_status_req== NULL && "Not implemented");

  // UE Context not retrievable
  // Optional
  assert(src->ue_ctx_not_retriable== NULL && "Not implemented");

  // Redirected RRC message
  // Optional
  //  9.3.1.6
  assert(src->redirected_rrc_msg== NULL && "Not implemented");

  //PLMN Assistance Info for Network Sharing
  // Optional
  // 9.3.1.14
  assert(src->plmn_assis_info_netwrk_shr== NULL && "Not implemented"); // [size(3)]

  // New gNB-CU UE F1AP ID
  // Optional
  // 9.3.1.4
  assert(src->new_gnb_cu_ue== NULL && "Not implemented"); 

  // Additional RRM Policy Index
  // Optional
  // 9.3.1.90
  assert(src->add_rrm_pol_idx== NULL && "Not implemented"); // bit string of 4

}

