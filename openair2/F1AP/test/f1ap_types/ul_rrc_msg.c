#include "ul_rrc_msg.h"

#include <assert.h>
#include <stdlib.h>

void free_ul_rrc_msg(ul_rrc_msg_t* src)
{
  assert(src != NULL);
  //  Message Type
  // 9.3.1.1
  // Mandatory

  // gNB-CU UE F1AP ID
  // 9.3.1.4
  // Mandatory
  // uint32_t gnb_cu_ue; // [0-2^32-1] 

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  // uint32_t gnb_du_ue; // [0-2^32-1] 

  //SRB ID
  //Mandatory
  //9.3.1.7
  // uint8_t srb_id; // [0-3]

  //RRC-Container
  //Mandatory
  //9.3.1.6
  free_byte_array(src->rrc_cntnr);

  //Selected PLMN ID
  //Optional
  // PLMN Identity 9.3.1.14
  if(src->plmn_id != NULL)
    free(src->plmn_id ); // (size(3) i.e., plmn_id[3])

  // New gNB-DU UE F1AP ID
  // Optional
  // gNB-DU UE F1AP ID 9.3.1.5
  if(src-> new_gnb_du_ue != NULL)
    free(src->new_gnb_du_ue);

}

bool eq_ul_rrc_msg(ul_rrc_msg_t const* m0, ul_rrc_msg_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;

  //  Message Type
  // 9.3.1.1
  // Mandatory

  // gNB-CU UE F1AP ID
  // 9.3.1.4
  // Mandatory
  if(m0->gnb_cu_ue != m1-> gnb_cu_ue )
    return false;

  // gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  if(m0->gnb_du_ue != m1->gnb_du_ue  )
    return false;

  //SRB ID
  //Mandatory
  //9.3.1.7
  assert(m0->srb_id < 4); // [0-3]
  assert(m1->srb_id < 4); 
  if(m0->srb_id != m1->srb_id)
    return false;

  //RRC-Container
  //Mandatory
  //9.3.1.6
  if(eq_byte_array(&m0->rrc_cntnr, &m1->rrc_cntnr) == false)
    return false;

  //Selected PLMN ID
  //Optional
  // PLMN Identity 9.3.1.14
  assert(m0->plmn_id == NULL && "Not implemented" ); // (size(3) i.e., plmn_id[3])
  assert(m1->plmn_id == NULL && "Not implemented" );

  // New gNB-DU UE F1AP ID
  // Optional
  // gNB-DU UE F1AP ID 9.3.1.5
  assert(m0->new_gnb_du_ue == NULL && "Not implemented" );
  assert(m1->new_gnb_du_ue == NULL && "Not implemented" );

  return true;
}

