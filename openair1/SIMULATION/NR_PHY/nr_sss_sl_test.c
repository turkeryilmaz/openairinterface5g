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

/**********************************************************************
*
* FILENAME    :  nr_sss_sl_test.c
*
* MODULE      :  UE test bench for sss tests
*
* DESCRIPTION :  it allows unitary tests for SSS on host machine
*
************************************************************************/
#include "nr_sss_sl_test.h"
#include "nr_pss_sl_test.h"

void insert_sss_sl_nr(int16_t *sss_time, NR_DL_FRAME_PARMS *frame_parms)
{
  unsigned int ofdm_symbol_size = frame_parms->ofdm_symbol_size;
  assert((ofdm_symbol_size * IQ_SIZE) <= SYNCF_TMP_SIZE);
  assert((ofdm_symbol_size * IQ_SIZE) <= frame_parms->nb_antennas_rx * SYNC_TMP_SIZE);
  bzero(synchroF_tmp, (ofdm_symbol_size * IQ_SIZE));
  bzero(synchro_tmp, (ofdm_symbol_size * IQ_SIZE));

  int Nid1 = GET_NID1(frame_parms->Nid_cell);
  int Nid2 = GET_NID2(frame_parms->Nid_cell);
  unsigned int k = ofdm_symbol_size - ((LENGTH_SSS_NR / 2) + 1);
  /* SSS is directly mapped to subcarrier */
  for (int i = 0; i < LENGTH_SSS_NR; i++) {
    synchroF_tmp[2 * k] = d_sss[Nid2][Nid1][i];
    synchroF_tmp[2 * k + 1] = 0;
    k++;
    if (k >= ofdm_symbol_size) {
      k++;
      k-=ofdm_symbol_size;
    }
  }

  idft(IDFT_2048, synchroF_tmp, synchro_tmp, 1);
  for (unsigned int i = 0; i < ofdm_symbol_size; i++) {
    ((int32_t *)sss_time)[i] = ((int32_t *)synchro_tmp)[i];
  }
}

int test_synchro_pss_sss_sl_nr(PHY_VARS_NR_UE *UE, int position_symbol)
{
  printf("Test nr pss with Nid2 %i at position %i \n", GET_NID2(UE->frame_parms.Nid_cell), position_symbol);
  set_sequence_pss_sl(UE, position_symbol, GET_NID2(UE->frame_parms.Nid_cell));
  int synchro_position = pss_synchro_nr(UE, 0, SYNCHRO_RATE_CHANGE_FACTOR);
  printf("Test nr sss with Nid1 %i \n", GET_NID1(UE->frame_parms.Nid_cell));
  synchro_position = synchro_position * SYNCHRO_RATE_CHANGE_FACTOR;
  if (abs(synchro_position - position_symbol) > PSS_DETECTION_MARGIN_MAX) {
    printf("NR PSS has been detected at position %d instead of %d \n", synchro_position, position_symbol);
  }

  int offset = (position_symbol + (SSS_SYMBOL_NB - PSS_SYMBOL_NB)*(UE->frame_parms.ofdm_symbol_size + UE->frame_parms.nb_prefix_samples));
  int16_t *tmp = (int16_t *)&UE->common_vars.rxdata[0][offset];
  insert_sss_sl_nr(tmp, &UE->frame_parms);
  UE->rx_offset = position_symbol;

  UE_nr_rxtx_proc_t proc = {0};
  int32_t metric_fdd_ncp = 0;
  uint8_t phase_fdd_ncp;
  rx_sss_nr(UE, &proc, &metric_fdd_ncp, &phase_fdd_ncp, &offset);
  return phase_fdd_ncp;
}


int test_sss_sl(PHY_VARS_NR_UE *UE)
{
  test_t test = {"SSS NR", 0, 0, 0, 0};
  printf("***********************************\n");
  printf("    %s Test synchronisation \n", test.test_current);
  printf("***********************************\n");

  int phase, Nid1, Nid2;
  int Nid_cell[] = {(3*0+0), (3*71+0), (3*21+2), (3*21+2), (3*55+1), (3*111+2)};
  int test_position[] = {0, 492, 493, 56788, 111111, 222222};
  for (unsigned int index = 0; index < (sizeof(Nid_cell) / sizeof(int)); index++) {
    UE->frame_parms.Nid_cell = Nid_cell[index];
    Nid2 = GET_NID2(Nid_cell[index]);
    Nid1 = GET_NID1(Nid_cell[index]);
    for (int position = 0; position < sizeof(test_position) / sizeof(test_position[0]); position++) {
      UE->frame_parms.Nid_cell = (3 * N_ID_1_NUMBER) + N_ID_2_NUMBER;
      phase = test_synchro_pss_sss_sl_nr(UE, test_position[position]);
      test.number_of_tests++;
      printf("%s ", test.test_current);
      if (UE->frame_parms.Nid_cell == (3 * Nid1) + Nid2) {
        if (phase != INDEX_NO_PHASE_DIFFERENCE) {
          printf("Test is pass with warning due to phase difference %d (instead of %d) offset %d Nid1 %d Nid2 %d \n",
                 phase, INDEX_NO_PHASE_DIFFERENCE, test_position[position], Nid1, Nid2);
          test.number_of_pass_warning++;
        } else {
          printf("Test is pass with offset %d Nid1 %d Nid2 %d \n", test_position[position], Nid1, Nid2);
          test.number_of_pass++;
        }
      } else {
        printf("Test is fail with offset %d Nid1 %d Nid2 %d \n", test_position[position], Nid1, Nid2);
        test.number_of_fail++;
      }
    }
  }

  printf("\n%s Number of tests : %d  Pass : %d Pass with warning : %d Fail : %d \n",
         test.test_current, test.number_of_tests, test.number_of_pass, test.number_of_pass_warning, test.number_of_fail);
  printf("%s Synchronisaton test is terminated.\n\n", test.test_current);
  test.number_of_tests = test.number_of_pass = test.number_of_pass_warning = test.number_of_fail = 0;
  return(0);
}
