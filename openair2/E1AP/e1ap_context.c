#include "openair2/E1AP/e1ap_context.h"

#include "common/utils/utils.h"

static int e1ap_context_cmp(e1ap_cuup_context_t *a, e1ap_cuup_context_t *b)
{
  if (a->cu_up_ue_id < b->cu_up_ue_id)
    return -1;
  if (a->cu_up_ue_id > b->cu_up_ue_id)
    return 1;
  if (a->sessionId < b->sessionId)
    return -1;
  if (a->sessionId > b->sessionId)
    return 1;
  return 0;
}

RB_GENERATE_STATIC(cuup_pdu_sessions_rbtree_t, e1ap_cuup_context_t, rb_opaque, e1ap_context_cmp);

static cuup_pdu_sessions_rbtree_t e1_contexts;

e1ap_cuup_context_t *new_e1ap_cuup_context(uint64_t cu_up_ue_id, long sessionId)
{
  e1ap_cuup_context_t *ctx = calloc_or_fail(1, sizeof(*ctx));

  ctx->cu_up_ue_id = cu_up_ue_id;
  ctx->sessionId = sessionId;

  e1ap_cuup_context_t *ret = RB_INSERT(cuup_pdu_sessions_rbtree_t, &e1_contexts, ctx);
  DevAssert(ret == NULL);

  return ctx;
}

e1ap_cuup_context_t *get_e1ap_cuup_context(uint64_t cu_up_ue_id, long sessionId)
{
  e1ap_cuup_context_t ctx;

  ctx.cu_up_ue_id = cu_up_ue_id;
  ctx.sessionId = sessionId;

  return RB_FIND(cuup_pdu_sessions_rbtree_t, &e1_contexts, &ctx);
}

void remove_e1ap_cuup_ue(uint64_t cu_up_ue_id)
{
  e1ap_cuup_context_t *e1_context;

  /* remove all PDU sessions for the UE */
  /* It's not clear if calling RB_REMOVE inside RB_FOREACH is safe
   * (what is the state of the RB tree after a RB_REMOVE? can the RB_FOREACH
   * continue?) so we first look for a session for this UE, then remove it
   * then restart the search from the beginning until all sessions for the
   * UE have been removed.
   * Maybe we can call RB_REMOVE inside RB_FOREACH safely in which case
   * this code may be greatly simplified.
   */
  while (true) {
    /* look for one session */
    RB_FOREACH(e1_context, cuup_pdu_sessions_rbtree_t, &e1_contexts) {
      /* once found, break from the search */
      if (e1_context->cu_up_ue_id == cu_up_ue_id) break;
    }
    /* stop if no more session for this UE */
    if (e1_context == NULL)
      break;

    RB_REMOVE(cuup_pdu_sessions_rbtree_t, &e1_contexts, e1_context);

    /* release the session */
    /* SDAP entity */
    if (e1_context->sdap != NULL) {
      e1_context->sdap->remove(e1_context->sdap);
      e1_context->sdap = NULL;
    }
    /* PDCP bearers */
    for (int i = 0; i < MAX_DRBS_PER_PDU_SESSION; i++)
      if (e1_context->drbs[i].pdcp != NULL) {
        e1_context->drbs[i].pdcp->delete_entity(e1_context->drbs[i].pdcp);
        e1_context->drbs[i].pdcp = NULL;
      }
    free(e1_context);
    /* the search will now restart from the beginning of the RB tree */
  }
}
