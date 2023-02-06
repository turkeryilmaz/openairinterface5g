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

bool eq_dl_rrc_msg(dl_rrc_msg_t const* m0, dl_rrc_msg_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;

  // Message Type
  // Mandatory 
  // 9.3.1.1

  // gNB-CU UE F1AP ID
  // Mandatory
  // 9.3.1.4 [0-2^32-1]
  if(m0->gnb_cu_ue != m1->gnb_cu_ue)
    return false;

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5 [0-2^32-1]
  if(m0->gnb_du_ue != m1->gnb_du_ue)
    return false;

  // old gNB-DU UE F1AP ID
  // Optional
  // 9.3.1.5 [0-2^32-1]
  assert(m0->old_gnb_du_ue == NULL && "Not implemented");
  assert(m1->old_gnb_du_ue == NULL && "Not implemented");

  // SRB ID
  // Mandatory
  // 9.3.1.7
  assert(m0->srb_id < 4);
  assert(m1->srb_id < 4);
  if(m0->srb_id != m1->srb_id)
    return false;

  // Execute Duplication
  // Optional
  assert(m0->exe_dup == NULL && "Not implemented" );
  assert(m1->exe_dup == NULL && "Not implemented" );

  // RRC-Container
  // Mandatory
  // 9.3.1.6
  // Includes the DL-DCCH-
  // Message IE as defined
  // in subclause 6.2 of TS
  // 38.331 [8]
  if(eq_byte_array(&m0->rrc_cntnr, &m1->rrc_cntnr) == false)
    return false;

  // RAT-Frequency Priority Information
  // Optional
  // 9.3.1.34
  assert(m0-> rat_freq== NULL && "Not implemented" );
  assert(m1-> rat_freq== NULL && "Not implemented" );

  // RRC Delivery Status Request
  // Optional
  assert(m0-> rrc_delivery_status_req== NULL && "Not implemented" );
  assert(m1-> rrc_delivery_status_req== NULL && "Not implemented" );

  // UE Context not retrievable
  // Optional
  assert(m0-> ue_ctx_not_retriable== NULL && "Not implemented" );
  assert(m1-> ue_ctx_not_retriable== NULL && "Not implemented" );

  // Redirected RRC message
  // Optional
  //  9.3.1.6
  assert(m0-> redirected_rrc_msg== NULL && "Not implemented" );
  assert(m1-> redirected_rrc_msg== NULL && "Not implemented" );

  //PLMN Assistance Info for Network Sharing
  // Optional
  // 9.3.1.14
  assert(m0-> plmn_assis_info_netwrk_shr== NULL && "Not implemented" );
  assert(m1-> plmn_assis_info_netwrk_shr== NULL && "Not implemented" );

  // New gNB-CU UE F1AP ID
  // Optional
  // 9.3.1.4
  assert(m0-> new_gnb_cu_ue== NULL && "Not implemented" );
  assert(m1-> new_gnb_cu_ue== NULL && "Not implemented" );

  // Additional RRM Policy Index
  // Optional
  // 9.3.1.90
  assert(m0-> add_rrm_pol_idx== NULL && "Not implemented" );
  assert(m1-> add_rrm_pol_idx== NULL && "Not implemented" );

  return true;
}

