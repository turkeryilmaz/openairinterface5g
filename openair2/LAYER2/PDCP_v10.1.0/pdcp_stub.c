/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file pdcp.c
 * \brief pdcp interface with RLC
 * \author Navid Nikaein and Lionel GAUTHIER
 * \date 2009-2012
 * \email navid.nikaein@eurecom.fr
 * \version 1.0
 */

#define PDCP_C

#define MBMS_MULTICAST_OUT

#include "assertions.h"
#include "hashtable.h"
#include "pdcp.h"
#include "pdcp_util.h"
#include "pdcp_sequence_manager.h"
#include "LAYER2/RLC/rlc.h"
#include "LAYER2/MAC/mac_extern.h"
#include "RRC/LTE/rrc_proto.h"
#include "pdcp_primitives.h"
#include "OCG.h"
#include "OCG_extern.h"
#include "otg_rx.h"
#include "common/utils/LOG/log.h"
#include <inttypes.h>
#include "platform_constants.h"
#include "nfapi/oai_integration/vendor_ext.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/ngran_types.h"
#include "targets/COMMON/openairinterface5g_limits.h"
#include "targets/RT/USER/lte-softmodem.h"
#include "SIMULATION/ETH_TRANSPORT/proto.h"
#include "UTIL/OSA/osa_defs.h"
#include "openair2/RRC/NAS/nas_config.h"
#include "intertask_interface.h"
#include "openair3/S1AP/s1ap_eNB.h"
#include <pthread.h>

#  include "gtpv1u_eNB_task.h"
#include <openair3/ocp-gtpu/gtp_itf.h>

#include "ENB_APP/enb_config.h"



int pdcp_fill_ss_pdcp_cnt (
  pdcp_t *pdcp_p,
  uint32_t rb_id,
  ss_get_pdcp_cnt_t *pc
)
{
	return 0;
}


void
pdcp_config_set_security_cipher(
  pdcp_t          *pdcp_pP,
  uint8_t         security_modeP)
//-----------------------------------------------------------------------------
{
  DevAssert(pdcp_pP != NULL);
  pdcp_pP->cipheringAlgorithm     = security_modeP;
  pdcp_pP->security_activated = 1;
}


