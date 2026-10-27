/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/***********************************************************************
*
* FILENAME    :  harq_nr.h
*
* MODULE      :  HARQ
*
* DESCRIPTION :  header related to Hybrid Automatic Repeat Request Acknowledgment
*                This feature allows to acknowledge downlink and uplink transport blocks
*
************************************************************************/

#ifndef HARQ_NR_H
#define HARQ_NR_H

/************** DEFINE ********************************************/

#define NR_DEFAULT_DLSCH_HARQ_PROCESSES (8) /* TS 38.214 5.1 */

/************** INCLUDE *******************************************/

#include "PHY/defs_nr_UE.h"
#include <limits.h>

/************* TYPE ***********************************************/


/************** VARIABLES *****************************************/


/*************** FUNCTIONS ****************************************/

/** \brief This function initialises downlink HARQ status
    @param pointer to downlink harq status
    @returns none */

void init_downlink_harq_status(NR_DL_UE_HARQ_t *dl_harq);

static inline bool nr_ue_is_common_dlsch(const NR_UE_DLSCH_t *dlsch)
{
  return dlsch->rnti_type == TYPE_SI_RNTI_ || dlsch->rnti_type == TYPE_RA_RNTI_ || dlsch->rnti_type == TYPE_P_RNTI_;
}

// Two codewords per ordinary PID, followed by two common contexts per DL actor (one actor for inline processing).
static inline int nr_ue_dl_harq_context_count(int num_dl_actors)
{
  const int actors = num_dl_actors > 0 ? num_dl_actors : 1;
  if (actors > INT_MAX / 2 - NR_MAX_HARQ_PROCESSES)
    return -1;
  return 2 * (NR_MAX_HARQ_PROCESSES + actors);
}

static inline uint32_t nr_ue_dl_harq_context_id(const PHY_VARS_NR_UE *ue,
                                                const UE_nr_rxtx_proc_t *proc,
                                                const NR_UE_DLSCH_t *dlsch,
                                                int cw_idx,
                                                int harq_pid)
{
  DevAssert(cw_idx >= 0 && cw_idx < 2 && harq_pid >= 0 && harq_pid < NR_MAX_HARQ_PROCESSES);
  if (!nr_ue_is_common_dlsch(dlsch))
    return 2 * harq_pid + cw_idx;
  DevAssert(ue->pdsch_scratch && ue->pdsch_num_actors > 0 && proc->nr_slot_rx >= 0);
  const uint32_t actor_idx = proc->nr_slot_rx % ue->pdsch_num_actors;
  return 2U * (NR_MAX_HARQ_PROCESSES + actor_idx) + cw_idx;
}

static inline NR_DL_UE_HARQ_t *nr_ue_dl_harq_context(PHY_VARS_NR_UE *ue,
                                                     const UE_nr_rxtx_proc_t *proc,
                                                     const NR_UE_DLSCH_t *dlsch,
                                                     int cw_idx,
                                                     int harq_pid)
{
  uint32_t id = nr_ue_dl_harq_context_id(ue, proc, dlsch, cw_idx, harq_pid);
  if (id < 2 * NR_MAX_HARQ_PROCESSES)
    return &ue->dl_harq_processes[id % 2][id / 2];
  id -= 2 * NR_MAX_HARQ_PROCESSES;
  return &ue->pdsch_scratch[id / 2].common_harq[id % 2];
}

static inline const NR_DL_UE_HARQ_t *nr_ue_dl_harq_context_const(const PHY_VARS_NR_UE *ue,
                                                                 const UE_nr_rxtx_proc_t *proc,
                                                                 const NR_UE_DLSCH_t *dlsch,
                                                                 int cw_idx,
                                                                 int harq_pid)
{
  uint32_t id = nr_ue_dl_harq_context_id(ue, proc, dlsch, cw_idx, harq_pid);
  if (id < 2 * NR_MAX_HARQ_PROCESSES)
    return &ue->dl_harq_processes[id % 2][id / 2];
  id -= 2 * NR_MAX_HARQ_PROCESSES;
  return &ue->pdsch_scratch[id / 2].common_harq[id % 2];
}

// Called by the owning DL actor after its previous job has finished, never by scheduled-response configuration.
void nr_ue_prepare_common_dl_harq(PHY_VARS_NR_UE *ue,
                                  const UE_nr_rxtx_proc_t *proc,
                                  const NR_UE_DLSCH_t *dlsch,
                                  int cw_idx,
                                  int harq_pid);

/** \brief This function update downlink harq context and return reception type (new transmission or retransmission)
    @param dlsch downlink harq context
    @param harq process identifier harq_pid
    @param rnti_type type of rnti
    @returns retransmission or new transmission */

void downlink_harq_process(NR_DL_UE_HARQ_t *dlsch, int harq_pid, int ndi, int rv, uint8_t rnti_type);

#undef EXTERN
#undef INIT_VARIABLES_HARQ_NR_H

#endif /* HARQ_NR_H */
