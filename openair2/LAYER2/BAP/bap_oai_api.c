/* from openair */
#include "rlc.h"
#include "LAYER2/nr_pdcp/nr_pdcp_oai_api.h"

/* from nr rlc module */
#include "openair2/LAYER2/nr_rlc/nr_rlc_asn1_utils.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_ue_manager.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_entity.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_oai_api.h"
#include "NR_RLC-BearerConfig.h"
#include "NR_DRB-ToAddMod.h"
#include "NR_DRB-ToAddModList.h"
#include "NR_SRB-ToAddModList.h"
#include "NR_DRB-ToReleaseList.h"
#include "NR_CellGroupConfig.h"
#include "NR_RLC-Config.h"
#include "common/ran_context.h"
#include "NR_UL-CCCH-Message.h"

extern RAN_CONTEXT_t RC;

#include <stdint.h>

#include <executables/softmodem-common.h>

// Same as RLC?
static void deliver_sdu(void *_ue, nr_rlc_entity_t *entity, char *buf, int size)
{
  // TODO
}

static void successful_delivery(void *_ue, nr_rlc_entity_t *entity, int sdu_id)
{
  // TODO
}

static void max_retx_reached(void *_ue, nr_rlc_entity_t *entity){
  // TODO
}

static void add_bhch_um(int rnti, int bhch_id, const NR_RLC_BearerConfig_t *rlc_BearerConfig)
{
  struct NR_RLC_Config *r = rlc_BearerConfig->rlc_Config;

  int sn_field_length;
  int t_reassembly;

  AssertFatal(drb_id > 0 && drb_id <= MAX_DRBS_PER_UE,
              "Invalid DRB ID %d\n", drb_id);

  switch (r->present) {
  case NR_RLC_Config_PR_um_Bi_Directional: {
    struct NR_RLC_Config__um_Bi_Directional *um;
    um = r->choice.um_Bi_Directional;
    t_reassembly = decode_t_reassembly(um->dl_UM_RLC.t_Reassembly);
    if (*um->dl_UM_RLC.sn_FieldLength != *um->ul_UM_RLC.sn_FieldLength) {
      LOG_E(RLC, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
      exit(1);
    }
    sn_field_length = decode_sn_field_length_um(*um->dl_UM_RLC.sn_FieldLength);
    break;
  }
  default:
    LOG_E(RLC, "%s:%d:%s: fatal error\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }
  
  // Remember that an IAB-MT is an UE
  nr_rlc_ue_manager_t* nr_rlc_ue_manager = get_rlc_ue_manager();
    
  nr_rlc_manager_lock(nr_rlc_ue_manager);
  nr_rlc_ue_t *ue = nr_rlc_manager_get_ue(nr_rlc_ue_manager, rnti);
  AssertFatal(rlc_BearerConfig->servedRadioBearer &&
              (rlc_BearerConfig->servedRadioBearer->present ==
              NR_RLC_BearerConfig__servedRadioBearer_PR_drb_Identity),
              "servedRadioBearer for DRB mandatory present when setting up an SRB RLC entity\n");
  int local_id = rlc_BearerConfig->logicalChannelIdentity - 1; // LCID 0 for SRB 0 not mapped
  ue->lcid2rb[local_id].type = NR_RLC_DRB;
  ue->lcid2rb[local_id].choice.drb_id = rlc_BearerConfig->servedRadioBearer->choice.drb_Identity;
  
  ue->bhch[bhch_id].mapped_rb.drb_id = rlc_BearerConfig->servedRadioBearer->choice.drb_Identity;

  if (ue->drb[drb_id-1] != NULL) {
    LOG_E(RLC, "DEBUG add_drb_um %s:%d:%s: warning DRB %d already exist for IAB-MT %d, do nothing\n", __FILE__, __LINE__, __FUNCTION__, drb_id, rnti);
  } else {
    nr_rlc_entity_t *nr_rlc_um = new_nr_rlc_entity_um(RLC_RX_MAXSIZE,
                                                      RLC_TX_MAXSIZE,
                                                      deliver_sdu, ue,
                                                      t_reassembly,
                                                      sn_field_length);
    nr_rlc_ue_add_drb_rlc_entity(ue, drb_id, nr_rlc_um);

    LOG_D(RLC, "%s:%d:%s: added BH RLC Channel %d with drb %d to IAB-MT with RNTI 0x%x\n", __FILE__, __LINE__, __FUNCTION__, bhch_id, drb_id, rnti);
  }
  nr_rlc_manager_unlock(nr_rlc_ue_manager);
}

static void add_bhch_am(int rnti, int bhch_id, const NR_RLC_BearerConfig_t *rlc_BearerConfig)
{
  struct NR_RLC_Config *r = rlc_BearerConfig->rlc_Config;

  int t_status_prohibit;
  int t_poll_retransmit;
  int poll_pdu;
  int poll_byte;
  int max_retx_threshold;
  int t_reassembly;
  int sn_field_length;

  AssertFatal(bhch_id > 0 && bhch_id <= 32,
              "Invalid BH RLC Channel ID %d\n", bhch_id);

  switch (r->present) {
  case NR_RLC_Config_PR_am: {
    struct NR_RLC_Config__am *am;
    am = r->choice.am;
    t_reassembly       = decode_t_reassembly(am->dl_AM_RLC.t_Reassembly);
    t_status_prohibit  = decode_t_status_prohibit(am->dl_AM_RLC.t_StatusProhibit);
    t_poll_retransmit  = decode_t_poll_retransmit(am->ul_AM_RLC.t_PollRetransmit);
    poll_pdu           = decode_poll_pdu(am->ul_AM_RLC.pollPDU);
    poll_byte          = decode_poll_byte(am->ul_AM_RLC.pollByte);
    max_retx_threshold = decode_max_retx_threshold(am->ul_AM_RLC.maxRetxThreshold);
    if (*am->dl_AM_RLC.sn_FieldLength != *am->ul_AM_RLC.sn_FieldLength) {
      LOG_E(RLC, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
      exit(1);
    }
    sn_field_length    = decode_sn_field_length_am(*am->dl_AM_RLC.sn_FieldLength);
    break;
  }
  default:
    LOG_E(RLC, "%s:%d:%s: fatal error\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  // Remember that an IAB-MT is an UE
  nr_rlc_ue_manager_t* nr_rlc_ue_manager = get_rlc_ue_manager();
    
  nr_rlc_manager_lock(nr_rlc_ue_manager);
  nr_rlc_ue_t *ue = nr_rlc_manager_get_ue(nr_rlc_ue_manager, rnti);

  AssertFatal(rlc_BearerConfig->servedRadioBearer &&
              (rlc_BearerConfig->servedRadioBearer->present ==
              NR_RLC_BearerConfig__servedRadioBearer_PR_drb_Identity),
              "servedRadioBearer for DRB mandatory present when setting up an SRB RLC entity\n");
  int local_id = rlc_BearerConfig->logicalChannelIdentity - 1; // LCID 0 for SRB 0 not mapped
  ue->lcid2rb[local_id].type = NR_RLC_DRB;
  ue->lcid2rb[local_id].choice.drb_id = rlc_BearerConfig->servedRadioBearer->choice.drb_Identity;
  
  ue->bhch[bhch_id].mapped_rb.drb_id = rlc_BearerConfig->servedRadioBearer->choice.drb_Identity;

  if (ue->drb[drb_id-1] != NULL) {
    LOG_E(RLC, "%s:%d:%s: DRB %d already exists for IAB-MT with RNTI %04x, do nothing\n", __FILE__, __LINE__, __FUNCTION__, drb_id, rnti);
  } else {
    nr_rlc_entity_t *nr_rlc_am = new_nr_rlc_entity_am(RLC_RX_MAXSIZE,
                                                      RLC_TX_MAXSIZE,
                                                      deliver_sdu, ue,
                                                      successful_delivery, ue,
                                                      max_retx_reached, ue,
                                                      t_poll_retransmit,
                                                      t_reassembly, t_status_prohibit,
                                                      poll_pdu, poll_byte, max_retx_threshold,
                                                      sn_field_length);
    nr_rlc_ue_add_drb_rlc_entity(ue, drb_id, nr_rlc_am);

    LOG_I(RLC, "%s:%d:%s: added BH RLC Channel %d with drb %d to IAB-MT with RNTI 0x%x\n", __FILE__, __LINE__, __FUNCTION__, bhch_id, drb_id, rnti);
  }
  nr_rlc_manager_unlock(nr_rlc_ue_manager);
}

void bap_add_bhch_drb(int rnti, int bhch_id, const NR_RLC_BearerConfig_t *rlc_BearerConfig){
  switch (rlc_BearerConfig->rlc_Config->present) {
  case NR_RLC_Config_PR_am:
    add_bhch_am(rnti, bhch_id, rlc_BearerConfig);
    break;
  case NR_RLC_Config_PR_um_Bi_Directional:
    add_bhch_um(rnti, bhch_id, rlc_BearerConfig);
    break;
  default:
    LOG_E(RLC, "%s:%d:%s: fatal: unhandled DRB type\n",
          __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }
  LOG_I(RLC, "%s:%s:%d: added BH RLC Channel to IAB-MT with RNTI 0x%x\n", __FILE__, __FUNCTION__, __LINE__, rnti);
}
