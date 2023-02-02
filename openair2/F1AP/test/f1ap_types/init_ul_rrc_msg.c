#include "init_ul_rrc_msg.h"

#include <assert.h>
#include <stdlib.h>

void free_init_ul_rrc_msg(init_ul_rrc_msg_t* src)
{
  assert(src != NULL);
  // Message Type
  // Mandatory
  // 9.3.1.1

  //  gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  // src->gnb_du_ue_id = rand(); // (0 .. 2^32 -1)

  // NR CGI
  // Mandatory
  // 9.3.1.12
  free_nr_cgi(&src->nr_cgi);

  // C-RNTI
  // Mandatory
  // 9.3.1.32
  //src->c_rnti = rand()%65536; // (0..65535)

  //RRC-Container
  //Mandatory
  //9.3.1.6
  free_byte_array(src->rrc_contnr);

  //DU to CU RRC Container
  // Optional
  // CellGroupConfig IE as defined in subclause 6.3.2 in TS 38.331 [8].
  assert(src->du_to_cu_rrc == NULL && "Not implemented");

  //SUL Access Indication
  // Optional
  assert(src->sul_access_ind == NULL && "Not implemented");

  //Transaction ID
  // Mandatory
  // 9.3.1.23
  // src->trans_id = rand()%256;

  // RAN UE ID
  // Optional
  assert(src->ran_ue_id == NULL && "Not implemented");

  //  RRC-Container-RRCSetupComplete
  // Optional
  // 9.3.1.6
  assert(src->rrc_cntnr_rrc_setup == NULL && "Not implemented");
}

bool eq_init_ul_rrc_msg(init_ul_rrc_msg_t const* m0, init_ul_rrc_msg_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;


  // Message Type
  // Mandatory
  // 9.3.1.1

  //  gNB-DU UE F1AP ID
  // Mandatory
  // 9.3.1.5
  if(m0->gnb_du_ue_id != m1->gnb_du_ue_id)
    return false;

  // NR CGI
  // Mandatory
  // 9.3.1.12
  if(eq_nr_cgi(&m0->nr_cgi, &m1->nr_cgi) == false)
    return false;

  // C-RNTI
  // Mandatory
  // 9.3.1.32
  if(m0->c_rnti != m1->c_rnti)  // (0..65535)
    return false;

  //RRC-Container
  //Mandatory
  //9.3.1.6
  if(eq_byte_array(&m0->rrc_contnr, &m1->rrc_contnr) == false)
    return false;

  //DU to CU RRC Container
  // Optional
  // CellGroupConfig IE as defined in subclause 6.3.2 in TS 38.331 [8].
  if(eq_byte_array(m0->du_to_cu_rrc, m1->du_to_cu_rrc) == false)
    return false;

  //SUL Access Indication
  // Optional
  assert(m0->sul_access_ind == NULL && "Not implemented");
  assert(m1->sul_access_ind == NULL && "Not implemented");

  //Transaction ID
  // Mandatory
  // 9.3.1.23
  if(m0->trans_id != m1->trans_id )
    return false;

  // RAN UE ID
  // Optional
  if(eq_byte_array(m0->ran_ue_id, m1->ran_ue_id) == false)
    return false;

  //  RRC-Container-RRCSetupComplete
  // Optional
  // 9.3.1.6
  if(eq_byte_array(m0->rrc_cntnr_rrc_setup, m1->rrc_cntnr_rrc_setup) == false)
    return false;

  return true;
}

