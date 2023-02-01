#include "gnb_cu_conf_update_ack.h"

#include <assert.h>

void free_gnb_cu_conf_update_ack(gnb_cu_conf_update_ack_t * src)
{
  assert(src != NULL);

  // Mandatory
  // Transaction ID 9.3.1.23
  //  uint8_t trans_id; 

  // [0-512]
  // Cells Failed to be Activated List
  for(size_t i = 0; i < src->sz_cells_failed_to_activate; ++i){
    free_cells_failed_to_activate(&src->cells_failed_to_activate[i]); 
  }
  if(src->sz_cells_failed_to_activate > 0){
    free(src->cells_failed_to_activate);
  }

  // Optional
  // Criticality Diagnostics 9.3.1.3 
  assert(src->crit_diagn == NULL && "Not implemented");

  // [0-32]
  // gNB-CU TNL Association Setup List
  for(size_t i = 0; i < src->sz_gnb_cu_tnl_assoc_stp; ++i){
    free_gnb_cu_tnl_assoc_stp(&src->gnb_cu_tnl_assoc_stp[i]);
  }
  if(src->sz_gnb_cu_tnl_assoc_stp > 0){
    free(src->gnb_cu_tnl_assoc_stp);
  }

  // [0-32] 
  // gNB-CU TNL Association Failed to Setup List
  for(size_t i = 0; i < src->sz_gnb_cu_tnl_assoc_failed_stp; ++i){
     free_gnb_cu_tnl_assoc_failed_stp(&src->gnb_cu_tnl_assoc_failed_stp[i]);
  }
  if(src->sz_gnb_cu_tnl_assoc_failed_stp > 0){
    free(src->gnb_cu_tnl_assoc_failed_stp); 
  }
  
  // [0 - 65536]
  // Dedicated SI Delivery Needed UE List
  for(size_t i = 0; i < src-> sz_ded_si_del_need; ++i){
    free_ded_si_del_need(&src->ded_si_del_need[i]);
  }
  if(src-> sz_ded_si_del_need > 0){
     free(src->ded_si_del_need); 
  }

  // Optional
  // 9.3.2.5
  // Transport Layer Address Info
  assert(src->trans_layer_addr == NULL && "Not implemented"); 
}

bool eq_gnb_cu_conf_update_ack(gnb_cu_conf_update_ack_t const* m0, gnb_cu_conf_update_ack_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;

  // Mandatory
  // Transaction ID 9.3.1.23
  if(m0->trans_id != m1->trans_id)
    return false;

  // [0-512]
  // Cells Failed to be Activated List
  assert(m0->sz_cells_failed_to_activate < 513);
  assert(m1->sz_cells_failed_to_activate < 513);

  if(m0->sz_cells_failed_to_activate != m1->sz_cells_failed_to_activate)
    return false;

  for(size_t i = 0; i < m0->sz_cells_failed_to_activate; ++i){
    if(eq_cells_failed_to_activate(&m0->cells_failed_to_activate[i], &m1->cells_failed_to_activate[i]) == false)
      return false;
  }

  // Optional
  // Criticality Diagnostics 9.3.1.3 
  assert(m0->crit_diagn == NULL && "Not implemented");
  assert(m1->crit_diagn == NULL && "Not implemented");

  // [0-32]
  // gNB-CU TNL Association Setup List
  assert(m0->sz_gnb_cu_tnl_assoc_stp < 33);
  assert(m1->sz_gnb_cu_tnl_assoc_stp < 33);
  
  if(m0->sz_gnb_cu_tnl_assoc_stp != m1->sz_gnb_cu_tnl_assoc_stp )
    return false;

  for(size_t i = 0; i < m0->sz_gnb_cu_tnl_assoc_stp; ++i){
    if(eq_gnb_cu_tnl_assoc_stp(&m0->gnb_cu_tnl_assoc_stp[i], &m1->gnb_cu_tnl_assoc_stp[i]) == false)
      return false;
  }

  // [0-32] 
  // gNB-CU TNL Association Failed to Setup List
  assert(m0->sz_gnb_cu_tnl_assoc_failed_stp < 33);
  assert(m1->sz_gnb_cu_tnl_assoc_failed_stp < 33);
  
  if(m0->sz_gnb_cu_tnl_assoc_failed_stp != m1->sz_gnb_cu_tnl_assoc_failed_stp )
    return false;

  for(size_t i = 0; i < m0->sz_gnb_cu_tnl_assoc_failed_stp; ++i){
    if(eq_gnb_cu_tnl_assoc_failed_stp(&m0->gnb_cu_tnl_assoc_failed_stp[i], &m1->gnb_cu_tnl_assoc_failed_stp[i]) == false )
        return false;
  }  
  // [0 - 65536]
  // Dedicated SI Delivery Needed UE List
  assert(m0->sz_ded_si_del_need < 65537 && "Not implemented");
  assert(m1->sz_ded_si_del_need < 65537 && "Not implemented");

  // Optional
  // 9.3.2.5
  // Transport Layer Address Info
  assert(m0->trans_layer_addr == NULL && "Not implemented"); 
  assert(m1->trans_layer_addr == NULL && "Not implemented"); 

  return true;
}

