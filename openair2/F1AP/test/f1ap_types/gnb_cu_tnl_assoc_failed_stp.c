#include "gnb_cu_tnl_assoc_failed_stp.h"

#include <assert.h>


void free_gnb_cu_tnl_assoc_failed_stp( gnb_cu_tnl_assoc_failed_stp_t* src)
{
  assert(src != NULL);

  // Mandatory
  // 9.3.2.4
  // TNL Association Transport Layer Address
  free_cp_trans_layer_info(&src->tnl_assoc_trans_layer_addr);

  // Mandatory
  // 9.3.1.2
  // Cause   
  // cause_f1ap_t cause;

}

bool eq_gnb_cu_tnl_assoc_failed_stp(gnb_cu_tnl_assoc_failed_stp_t const* m0, gnb_cu_tnl_assoc_failed_stp_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;

  // Mandatory
  // 9.3.2.4
  // TNL Association Transport Layer Address
  if( eq_cp_trans_layer_info(&m0->tnl_assoc_trans_layer_addr, &m1->tnl_assoc_trans_layer_addr) == false)
    return false;

  // Mandatory
  // 9.3.1.2
  // Cause   
  if(eq_cause_f1ap(&m0->cause, &m1->cause) == false)
    return false;

  return true;
}

