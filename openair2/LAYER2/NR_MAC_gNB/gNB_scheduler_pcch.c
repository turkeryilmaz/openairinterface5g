/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief gNB PCCH (paging) scheduling procedures
 */

#include <stdbool.h>
#include <stdint.h>
#include "assertions.h"
#include "NR_MAC_gNB/nr_mac_gNB.h"
#include "NR_MAC_gNB/mac_proto.h"
#include "common/utils/LOG/log.h"
#include "common/utils/ds/byte_array.h"
#include "openair2/RRC/NR/MESSAGES/asn1_msg.h"

void nr_mac_pcch_queue_free(NR_COMMON_channels_t *cc)
{
  DevAssert(cc);
  spsc_q_free(&cc->pcch_queue);
}

void nr_mac_pcch_queue_init(NR_COMMON_channels_t *cc)
{
  DevAssert(cc);
  const bool ok = spsc_q_alloc(&cc->pcch_queue, NR_PCCH_MAX_PAGING_RECORDS, sizeof(nr_mac_pcch_record_t));
  AssertFatal(ok, "failed to allocate PCCH queue\n");
}

static void nr_mac_pcch_queue_push(spsc_q_t *q, module_id_t module_id, const nr_mac_pcch_record_t *item)
{
  DevAssert(q);
  DevAssert(item);

  if (!spsc_q_put(q, item, sizeof(*item)))
    LOG_W(NR_MAC, "[gNB %d] PCCH queue full, dropping paging record\n", module_id);
}

/** @brief Enqueue a pending CN paging record for transmission at the UE's PO.
 *
 * Stores identity only: PCCH-Message (Paging, TS 38.331 §6.2.2) encoding is deferred
 * until the UE's paging occasion (TS 38.304 §7.1).
 *
 * @param fiveg_s_tmsi ng-5G-S-TMSI from NGAP/F1AP Paging (TS 38.413 / TS 38.473).
 * @param ue_id        UE identity index value mod 1024 (= UE_ID for PF/PO, TS 38.304 §7.1). */
void nr_mac_pcch_enqueue(module_id_t module_id, uint64_t fiveg_s_tmsi, uint16_t ue_id)
{
  gNB_MAC_INST *mac = RC.nrmac[module_id];
  DevAssert(mac);
  const int CC_id = 0;
  NR_COMMON_channels_t *cc = &mac->common_channels[CC_id];

  const nr_mac_pcch_record_t item = {
      .ue_id = ue_id % 1024,
      .fiveg_s_tmsi = fiveg_s_tmsi & ((1ULL << 48) - 1),
  };

  NR_SCHED_LOCK(&mac->sched_lock);
  nr_mac_pcch_queue_push(&cc->pcch_queue, module_id, &item);
  NR_SCHED_UNLOCK(&mac->sched_lock);

  LOG_I(NR_MAC, "[gNB %d] PCCH record enqueued UE_ID=%u (5G-S-TMSI=0x%012lx)\n", module_id, item.ue_id, fiveg_s_tmsi);
}
