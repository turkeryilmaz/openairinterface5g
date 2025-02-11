#ifndef E1AP_CONTEXT_H_
#define E1AP_CONTEXT_H_

#include <stdint.h>

#define __unused __attribute__((unused))
#include "common/utils/collection/tree.h"

#include "nr_pdcp/nr_pdcp_entity.h"
#include "SDAP/nr_sdap/nr_sdap_entity.h"
#include "common/platform_constants.h"

RB_HEAD(cuup_pdu_sessions_rbtree_t, e1ap_cuup_context_t);
typedef struct cuup_pdu_sessions_rbtree_t cuup_pdu_sessions_rbtree_t;

typedef struct {
  nr_pdcp_entity_t *pdcp;
  teid_t gtpu_tunnel_id;
} e1ap_drb_context_t;

typedef struct e1ap_cuup_context_t {
  /* ID of a context is made of cu_ue_id and sessionId */
  uint64_t cu_up_ue_id;
  long sessionId;
  uint32_t n3_tunnel_id;
  nr_sdap_entity_t *sdap;
  e1ap_drb_context_t drbs[MAX_DRBS_PER_PDU_SESSION];  /* index this array with drb_id - 1 */
  RB_ENTRY(e1ap_cuup_context_t) rb_opaque;
} e1ap_cuup_context_t;

e1ap_cuup_context_t *new_e1ap_cuup_context(uint64_t cu_up_ue_id, long sessionId);
e1ap_cuup_context_t *get_e1ap_cuup_context(uint64_t cu_up_ue_id, long sessionId);
void remove_e1ap_cuup_ue(uint64_t cu_up_ue_id);

#endif /* E1AP_CONTEXT_H_ */
