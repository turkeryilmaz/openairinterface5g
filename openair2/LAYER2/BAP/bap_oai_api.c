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


void bap_add_bhch_drb(int rnti, int bhch_id, int drb_id, const NR_RLC_BearerConfig_t *rlc_BearerConfig){
    nr_rlc_ue_manager_t* nr_rlc_ue_manager = get_rlc_ue_manager();
}