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

#include "PHY/defs_UE.h"
#include "PHY/phy_extern_ue.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
#include "executables/softmodem-common.h"
#include "openair1/PHY/defs_nr_UE.h"

#define SCI2_LEN_SIZE 35

void nr_attach_crc_to_payload(unsigned char *in, uint8_t *out, int max_payload_bytes, uint32_t in_size, uint32_t *out_size) {

    unsigned int crc = 1;
    if (in_size > NR_MAX_PSSCH_TBS) {
      // Add 24-bit crc (polynomial A) to payload
      crc = crc24a(in, in_size) >> 8;
      in[in_size >> 3] = ((uint8_t*)&crc)[2];
      in[1 + (in_size >> 3)] = ((uint8_t*)&crc)[1];
      in[2 + (in_size >> 3)] = ((uint8_t*)&crc)[0];
      *out_size = in_size + 24;

      AssertFatal((in_size / 8) + 4 <= max_payload_bytes,
                  "A %d is too big (A / 8 + 4 = %d > %d)\n", in_size, (in_size / 8) + 4, max_payload_bytes);

      memcpy(out, in, (in_size / 8) + 4);
    } else {
      // Add 16-bit crc (polynomial A) to payload
      crc = crc16(in, in_size) >> 16;
      in[in_size >> 3] = ((uint8_t*)&crc)[1];
      in[1 + (in_size >> 3)] = ((uint8_t*)&crc)[0];
      *out_size = in_size + 16;

      AssertFatal((in_size / 8) + 3 <= max_payload_bytes,
                  "A %d is too big (A / 8 + 3 = %d > %d)\n", in_size, (in_size / 8) + 3, max_payload_bytes);

      memcpy(out, in, (in_size / 8) + 3);  // using 3 bytes to mimic the case of 24 bit crc
    }
}

void nr_ue_set_slsch_rx(PHY_VARS_NR_UE *ue, unsigned char harq_pid)
{
  int nb_rb = ue->frame_parms.N_RB_SL;
  uint16_t nb_symb_sch = 12;
  uint8_t dmrsConfigType = 0;
  uint8_t nb_re_dmrs = 6;
  uint8_t Nl = 1; // number of layers
  uint8_t Imcs = 9;
  uint16_t dmrsSymbPos = 16 + 1024; // symbol 4 and 10
  uint8_t length_dmrs = get_num_dmrs(dmrsSymbPos);
  uint16_t start_symbol = 1; // start from 0

  uint8_t mod_order = nr_get_Qm_ul(Imcs, 0);
  uint16_t code_rate = nr_get_code_rate_ul(Imcs, 0);
  unsigned int TBS = get_softmodem_params()->sl_mode != 0 ? 2048 : nr_compute_tbs(mod_order, code_rate, nb_rb, nb_symb_sch, nb_re_dmrs * length_dmrs, 0, 0, Nl);
  LOG_I(NR_PHY, "\nTBS %u mod_order %d\n", TBS, mod_order);

  NR_UE_DLSCH_t *slsch_ue_rx = ue->slsch_rx[0][0];
  NR_DL_UE_HARQ_t *harq = slsch_ue_rx->harq_processes[harq_pid];
  harq->Nl = Nl;
  harq->Qm = mod_order;
  harq->nb_rb = nb_rb;
  harq->TBS = TBS >> 3;
  harq->n_dmrs_cdm_groups = 1;
  harq->dlDmrsSymbPos = dmrsSymbPos;
  harq->mcs = Imcs;
  harq->dmrsConfigType = dmrsConfigType;
  harq->R = code_rate;
  harq->nb_symbols = nb_symb_sch;
  harq->codeword = 0;
  harq->start_symbol = start_symbol;
  harq->B_sci2 = 1024; // This should be updated from SCI1 parameter.
  harq->status = ACTIVE;

  nfapi_nr_pssch_pdu_t *rel16_sl_rx = &harq->pssch_pdu;
  rel16_sl_rx->mcs_index            = Imcs;
  rel16_sl_rx->pssch_data.rv_index  = 0;
  rel16_sl_rx->target_code_rate     = code_rate;
  rel16_sl_rx->pssch_data.tb_size   = TBS >> 3; // bytes
  rel16_sl_rx->pssch_data.sci2_size = SCI2_LEN_SIZE >> 3;
  rel16_sl_rx->maintenance_parms_v3.ldpcBaseGraph = get_BG(TBS, code_rate);
  rel16_sl_rx->nr_of_symbols  = nb_symb_sch; // number of symbols per slot
  rel16_sl_rx->start_symbol_index = start_symbol;
  rel16_sl_rx->ul_dmrs_symb_pos = harq->dlDmrsSymbPos;
  rel16_sl_rx->nrOfLayers = harq->Nl;
  rel16_sl_rx->num_dmrs_cdm_grps_no_data = 1;
  rel16_sl_rx->rb_size = nb_rb;
  rel16_sl_rx->bwp_start = 0;
  rel16_sl_rx->rb_start = 0;
  rel16_sl_rx->dmrs_config_type = dmrsConfigType;
}

void nr_ue_set_slsch(NR_DL_FRAME_PARMS *fp,
                     unsigned char harq_pid,
                     NR_UE_ULSCH_t *slsch,
                     uint32_t frame,
                     uint8_t slot) {
  NR_UL_UE_HARQ_t *harq = slsch->harq_processes[harq_pid];
  uint8_t nb_codewords = 1;
  uint8_t N_PRB_oh = 0;
  uint16_t nb_symb_sch = 12;
  uint8_t nb_re_dmrs = 6;
  int nb_rb = fp->N_RB_SL;
  uint8_t Imcs = 9;
  uint8_t Nl = 1; // number of layers
  uint16_t start_symbol = 1; // start from 0
  SCI_1_A *sci1 = &harq->pssch_pdu.sci1;
  sci1->period = 0;
  sci1->dmrs_pattern = (1 << 4) + (1 << 10);
  sci1->beta_offset = 0;
  sci1->dmrs_port = 0;
  sci1->priority = 0;
  sci1->freq_res = 1;
  sci1->time_res = 1;
  sci1->mcs = Imcs;
  uint16_t dmrsSymbPos = sci1->dmrs_pattern; // symbol 4 and 10
  uint8_t dmrsConfigType = 0;
  uint8_t length_dmrs = get_num_dmrs(dmrsSymbPos);
  uint16_t code_rate = nr_get_code_rate_ul(Imcs, 0);
  uint8_t mod_order = nr_get_Qm_ul(Imcs, 0);
  uint16_t N_RE_prime = NR_NB_SC_PER_RB * nb_symb_sch - nb_re_dmrs - N_PRB_oh;
  unsigned int TBS = get_softmodem_params()->sl_mode != 0 ? 2048 : nr_compute_tbs(mod_order, code_rate, nb_rb, nb_symb_sch, nb_re_dmrs * length_dmrs, 0, 0, Nl);

  harq->pssch_pdu.mcs_index = Imcs;
  harq->pssch_pdu.nrOfLayers = Nl;
  harq->pssch_pdu.rb_size = nb_rb;
  harq->pssch_pdu.nr_of_symbols = nb_symb_sch;
  harq->pssch_pdu.dmrs_config_type = dmrsConfigType;
  harq->num_of_mod_symbols = N_RE_prime * nb_rb * nb_codewords;
  harq->pssch_pdu.pssch_data.rv_index = 0;
  harq->pssch_pdu.pssch_data.tb_size  = TBS >> 3;
  harq->pssch_pdu.pssch_data.sci2_size = SCI2_LEN_SIZE >> 3;
  harq->pssch_pdu.target_code_rate = code_rate;
  harq->pssch_pdu.qam_mod_order = mod_order;
  harq->pssch_pdu.sl_dmrs_symb_pos = dmrsSymbPos;
  harq->pssch_pdu.num_dmrs_cdm_grps_no_data = 1;
  harq->pssch_pdu.start_symbol_index = start_symbol;
  harq->pssch_pdu.transform_precoding = transformPrecoder_disabled;
  harq->first_tx = 1;

  harq->status = ACTIVE;
  unsigned char *test_input = harq->a;
  uint64_t *sci_input = harq->a_sci2;

  bool payload_type_string = true;
  if (payload_type_string) {
    for (int i = 0; i < 32; i++) {
      test_input[i] = get_softmodem_params()->sl_user_msg[i];
    }
  } else {
    srand(time(NULL));
    for (int i = 0; i < TBS / 8; i++)
      test_input[i] = (unsigned char) (i+3);//rand();
    test_input[0] = (unsigned char) (slot);
    test_input[1] = (unsigned char) (frame & 0xFF); // 8 bits LSB
    test_input[2] = (unsigned char) ((frame >> 8) & 0x3); //
    test_input[3] = (unsigned char) ((frame & 0x111) << 5) + (unsigned char) (slot) + rand() % 256;
    LOG_D(NR_PHY, "SLSCH_TX will send %u\n", test_input[3]);
  }
  uint64_t u = pow(2,SCI2_LEN_SIZE) - 1;
  *sci_input = u;//rand() % (u - 0 + 1);
}
