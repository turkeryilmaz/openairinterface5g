/* from openair */
#include "rlc.h"

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
#include "conversions.h"
#include "bap_oai_api.h"

// Having trouble changing this to .h
#include "bap_entity.c"

extern RAN_CONTEXT_t RC;

#include <executables/softmodem-common.h>

static bap_entity_t *bap_entity;

// Same as RLC?
static void deliver_bap_sdu(void *_ue, nr_rlc_entity_t *entity, char *buf, int size)
{
  // TODO
}

static void successful_bap_delivery(void *_ue, nr_rlc_entity_t *entity, int sdu_id)
{
  // TODO
}

static void max_retx_reached(void *_ue, nr_rlc_entity_t *entity){
  // TODO
}

static void add_bhch_um(int rnti, int bhch_id, const NR_BH_RLC_ChannelConfig_r16_t *rlc_bhchConfig)
{
  struct NR_RLC_Config *r = rlc_bhchConfig->rlc_Config_r16;

  int sn_field_length;
  int t_reassembly;

  AssertFatal(bhch_id > 0 && bhch_id <= 32,
              "Invalid bhch ID %d\n", bhch_id);

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

  /*AssertFatal(rlc_bhchConfig->bh_LogicalChannelIdentity_r16->choice.bh_LogicalChannelIdentity_r16 != NULL, 
              "Didn't pass Logical Channel ID during BH-RLC-Channel_config");
  AssertFatal(rlc_bhchConfig->bh_LogicalChannelIdentity_r16->present == NR_BH_LogicalChannelIdentity_r16_PR_bh_LogicalChannelIdentity_r16,
              "Given LogicalChannelIdentity isn't supported");
  */
  int local_id = rlc_bhchConfig->bh_LogicalChannelIdentity_r16->choice.bh_LogicalChannelIdentity_r16 - 1;
  BIT_STRING_TO_NR_BHRLCCHANNELID(&rlc_bhchConfig->bh_RLC_ChannelID_r16, ue->lcid2bhch[local_id]);

  if (ue->bhch[bhch_id-1] != NULL) {
    LOG_E(RLC, "%s:%d:%s: BH Channel %d already exists for IAB-MT with RNTI %04x, do nothing\n", __FILE__, __LINE__, __FUNCTION__, bhch_id, rnti);
  } else {
    nr_rlc_entity_t *nr_rlc_um = new_nr_rlc_entity_um(RLC_RX_MAXSIZE,
                                                      RLC_TX_MAXSIZE,
                                                      deliver_bap_sdu, ue,
                                                      t_reassembly,
                                                      sn_field_length);
    nr_rlc_ue_add_bhch_rlc_entity(ue, bhch_id, nr_rlc_um);

    LOG_D(RLC, "%s:%d:%s: added BH RLC Channel %d to IAB-MT with RNTI 0x%x\n", __FILE__, __LINE__, __FUNCTION__, bhch_id, rnti);
  }
  nr_rlc_manager_unlock(nr_rlc_ue_manager);
}

static void add_bhch_am(int rnti, int bhch_id, const NR_BH_RLC_ChannelConfig_r16_t *rlc_bhchConfig)
{
  struct NR_RLC_Config *r = rlc_bhchConfig->rlc_Config_r16;

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

  /*AssertFatal(rlc_bhchConfig->bh_LogicalChannelIdentity_r16->choice.bh_LogicalChannelIdentity_r16 != NULL, 
              "Didn't pass Logical Channel ID during BH-RLC-Channel_config");
  AssertFatal(rlc_bhchConfig->bh_LogicalChannelIdentity_r16->present == NR_rlc_bhchConfig__bh_LogicalChannelIdentity_r16_PR_bh_LogicalChannelIdentity_r16,
              "Given LogicalChannelIdentity isn't supported");
  */
  int local_id = rlc_bhchConfig->bh_LogicalChannelIdentity_r16->choice.bh_LogicalChannelIdentity_r16 - 1;
  BIT_STRING_TO_NR_BHRLCCHANNELID(&rlc_bhchConfig->bh_RLC_ChannelID_r16, ue->lcid2bhch[local_id]);

  if (ue->bhch[bhch_id-1] != NULL) {
    LOG_E(RLC, "%s:%d:%s: BH Channel %d already exists for IAB-MT with RNTI %04x, do nothing\n", __FILE__, __LINE__, __FUNCTION__, bhch_id, rnti);
  } else {
    nr_rlc_entity_t *nr_rlc_am = new_nr_rlc_entity_am(RLC_RX_MAXSIZE,
                                                      RLC_TX_MAXSIZE,
                                                      deliver_bap_sdu, ue,
                                                      successful_bap_delivery, ue,
                                                      max_retx_reached, ue,
                                                      t_poll_retransmit,
                                                      t_reassembly, t_status_prohibit,
                                                      poll_pdu, poll_byte, max_retx_threshold,
                                                      sn_field_length);
    nr_rlc_ue_add_bhch_rlc_entity(ue, bhch_id, nr_rlc_am);

    LOG_I(RLC, "%s:%d:%s: added BH RLC Channel %d to IAB-MT with RNTI 0x%x\n", __FILE__, __LINE__, __FUNCTION__, bhch_id, rnti);
  }
  nr_rlc_manager_unlock(nr_rlc_ue_manager);
}

void bap_add_bhch(int rnti, int bhch_id, const NR_BH_RLC_ChannelConfig_r16_t *rlc_bhchConfig){
  switch (rlc_bhchConfig->rlc_Config_r16->present) {
  case NR_RLC_Config_PR_am:
    add_bhch_am(rnti, bhch_id, rlc_bhchConfig);
    break;
  case NR_RLC_Config_PR_um_Bi_Directional:
    add_bhch_um(rnti, bhch_id, rlc_bhchConfig);
    break;
  default:
    LOG_E(RLC, "%s:%d:%s: fatal: unhandled BH Channel type\n",
          __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }
  LOG_I(RLC, "%s:%s:%d: added BH RLC Channel to IAB-MT with RNTI 0x%x\n", __FILE__, __FUNCTION__, __LINE__, rnti);
}

void nr_bap_layer_init(bool is_du)
{
  /* hack: be sure to initialize only once */
  static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
  static int inited_DU = 0;
  static int inited_MT = 0;

  if (pthread_mutex_lock(&m) != 0) abort();
  
  if (is_du && inited_DU) {
    // Change LOG to BAP
    LOG_E(RLC, "%s:%d:%s: fatal, inited_du already 1\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  if (!is_du && inited_MT) {
    // Change LOG to BAP
    LOG_E(RLC, "%s:%d:%s: fatal, inited_mt already 1\n", __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }

  if (is_du) inited_DU = 1;
  if (!is_du) inited_MT = 1;

  // nr_rlc_ue_manager = new_nr_rlc_ue_manager(enb_flag);

  if (pthread_mutex_unlock(&m)) abort();
}

void init_bap_entity(uint16_t bap_addr, bool is_du){
  bap_entity = new_bap_entity(bap_addr, is_du);

  if (is_du)
    printf("[BAP]   BAP entity initialized as IAB-DU with BAPAddress = %d\n", bap_entity->bap_address);
  else
    printf("[BAP]   BAP entity initialized as IAB-MT with BAPAddress = %d\n", bap_entity->bap_address);
}
