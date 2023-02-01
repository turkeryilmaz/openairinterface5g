#include "gnb_cu_tnl_assoc_stp.h"

#include <assert.h>

void free_gnb_cu_tnl_assoc_stp(gnb_cu_tnl_assoc_stp_t* src)
{
  assert(src != NULL);

  free_cp_trans_layer_info(&src->tnl_assoc_trans_layer_addr);
}

bool eq_gnb_cu_tnl_assoc_stp(gnb_cu_tnl_assoc_stp_t const* m0, gnb_cu_tnl_assoc_stp_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;

  if(eq_cp_trans_layer_info(&m0->tnl_assoc_trans_layer_addr, &m1->tnl_assoc_trans_layer_addr) == false)
    return false;

  return true;
}

