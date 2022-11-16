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

/*********************************************************************
*
* FILENAME    :  nr_pss_sl_test.c
*
* MODULE      :  UE test bench for unit tests
*
* DESCRIPTION :  it allows unitary tests of NR UE SL
*
************************************************************************/
#include "nr_pss_sl_test.h"

void display_sl_data(int pss_sequence_number, int16_t *rxdata, int position) {
#ifdef DEBUG_TEST_PSS
  int16_t *pss_sequence[NUMBER_PSS_SEQUENCE] = {primary_synch0_time, primary_synch1_time, primary_synch2_time};
  int16_t *pss_sequence_time = pss_sequence[pss_sequence_number];
  printf("pss %6d data \n", pss_sequence_number);
  for (int i = 0; i < 4; i++) {
    if (pss_sequence_number < NUMBER_PSS_SEQUENCE) {
      printf("[i %6d] : %4d [i %6d] : %8i at address : %p \n", i, pss_sequence_time[2*i], (i + position), rxdata[2*i + (position*2)],  &(rxdata[2*i + (position*2)]));
      printf("[q %6d] : %4d [q %6d] : %8i at address : %p \n", i, pss_sequence_time[2*i+1], (i + position), rxdata[2*i + 1 + (position*2)],  &(rxdata[2*i + 1 + (position*2)]));
    } else {
      printf("[i %6d] : Undef [i %6d] : %8i at address : %p \n", i, (i + position), rxdata[2*i + (position*2)], &(rxdata[2*i + (position*2)]));
      printf("[q %6d] : Undef [q %6d] : %8i  at address : %p \n", i, (i + position), rxdata[2*i + 1 + (position*2)], &(rxdata[2*i + 1 + (position*2)]));
    }
  }
#else
  (void) pss_sequence_number;
  (void) rxdata;
  (void) position;
#endif
}

typedef enum {
  ZERO_DATA,
  SINUSOIDAL_DATA,
  RANDOM_DATA,
  RANDOM_MAX_DATA
} random_data_format_t;

/* data is a pure cosinus wave */
#define  SAMPLING_RATE           (30720000L)
#define  SCALING_SINUSOIDAL_DATA (4)                    /* 16 is max value without decimation */
#define  FREQUENCY_15_MHZ        (15360000L)
#define  FREQUENCY               (FREQUENCY_15_MHZ)     /* to generate a frequency with a sampling of 30,72 MHz  5 gives 770 KHz, 20 gives 1,5 MHz, 40 gives 3 MHz */
# define PI		         3.14159265358979323846	/* pi */
void set_random_sl_rx_buffer(PHY_VARS_NR_UE *UE, int amp) {
  int samples_for_frame = UE->frame_parms.samples_per_frame;
  srand(0);
  int16_t random;
  random_data_format_t data_format = SINUSOIDAL_DATA;
  for (int aa = 0; aa < UE->frame_parms.nb_antennas_rx; aa++) {
     int16_t *data_p = (int16_t *) &(UE->common_vars.rxdata[aa][0]);
    int frequency_switch = samples_for_frame / LTE_NUMBER_OF_SUBFRAMES_PER_FRAME;
    int frequency_step = 0;
    double beat = (2 * PI  *FREQUENCY_15_MHZ) / (SAMPLING_RATE);

    for (int i = 0; i < samples_for_frame; i++) {
      switch(data_format) {
        case ZERO_DATA: {
          random = 0;
          break;
        }
        case SINUSOIDAL_DATA: {
          /* sinusoidal signal */
          double n = cos(beat * i);
          random =  n * (amp * SCALING_SINUSOIDAL_DATA);
          frequency_step++;
          if (frequency_step == frequency_switch) {
            beat = beat / 2;  /* frequency is divided by 2 */
            frequency_step = 0;
          }
          break;
        }
        case RANDOM_DATA: {
#define SCALING_RANDOM_DATA       (24)      /* 48 is max value without decimation */
#define RANDOM_MAX_AMP            (amp * SCALING_RANDOM_DATA)
          random = ((rand() % RANDOM_MAX_AMP) - RANDOM_MAX_AMP / 2);
          break;
        }
        case RANDOM_MAX_DATA: {
#define SCALING_RANDOM_MAX_DATA   (8)
#define RANDOM_VALUE              (amp * SCALING_RANDOM_DATA)
          const int random_number[2] = {-1,+1};
          random = random_number[rand()%2] * RANDOM_VALUE;
          break;
        }
        default: {
          printf("Format of data is undefined \n");
          assert(0);
          break;
        }
      }
      data_p[2 * i] = random;
      data_p[2 * i + 1] = random;
    }
  }
}

int set_pss_sl_in_rx_buffer(PHY_VARS_NR_UE *UE, int position_symbol, int pss_sequence_number) {
  NR_DL_FRAME_PARMS *frame_parms = &( UE->frame_parms);
  if ((position_symbol > frame_parms->samples_per_frame)
      || ((position_symbol + frame_parms->ofdm_symbol_size) > frame_parms->samples_per_frame)) {
    printf("This pss sequence can not be fully written in the received window \n");
    return (-1);
  }
  if ((pss_sequence_number >= NUMBER_PSS_SEQUENCE) && (pss_sequence_number < 0)) {
    printf("Unknow pss sequence %d \n", pss_sequence_number);
    return (-1);
  }
  int16_t *pss_sequence_time = primary_synchro_time_nr[pss_sequence_number];
  for (int aa = 0; aa< UE->frame_parms.nb_antennas_rx; aa++) {
    for (int i = 0; i < frame_parms->ofdm_symbol_size; i++) {
      ((int16_t *) UE->common_vars.rxdata[aa])[(position_symbol * 2) + (2 * i)] = pss_sequence_time[2 * i];     /* real part */
      ((int16_t *) UE->common_vars.rxdata[aa])[(position_symbol * 2) + (2 * i + 1)] = pss_sequence_time[2 * i + 1]; /* imaginary part */
    }
  }
  for (int aa = 0; aa < UE->frame_parms.nb_antennas_rx; aa++) {
    for (int i = 0; i < frame_parms->ofdm_symbol_size; i++) {
      if ((pss_sequence_time[2 * i] != ((int16_t *)UE->common_vars.rxdata[aa])[(position_symbol * 2) + (2 * i)])
          || (pss_sequence_time[2 * i + 1] != ((int16_t *)UE->common_vars.rxdata[aa])[(position_symbol * 2) + (2 * i + 1)])) {
        printf("Sequence pss was not properly copied into received buffer at index %d \n", i);
        exit(-1);
      }
    }
  }
  return (0);
}

void set_sequence_pss_sl(PHY_VARS_NR_UE *UE, int position_symbol, int pss_sequence_number) {
  NR_DL_FRAME_PARMS *frame_parms = &(UE->frame_parms);
  /* initialise received ue data with random */
  set_random_sl_rx_buffer(UE, AMP);
  /* write pss sequence in received ue buffer */
  if (pss_sequence_number < NUMBER_PSS_SEQUENCE) {
    if (position_symbol > (frame_parms->samples_per_frame - frame_parms->ofdm_symbol_size)) {
      printf("This position for pss sequence %d is not supported because it exceeds the frame length %d!\n", position_symbol, frame_parms->samples_per_frame);
      exit(0);
    }
    if (set_pss_sl_in_rx_buffer(UE, position_symbol, pss_sequence_number) != 0)
      printf("Warning: pss sequence can not be properly written into received buffer !\n");
  }
  display_sl_data(pss_sequence_number, (int16_t *)&(UE->common_vars.rxdata[0][0]), position_symbol);
}

void test_synchro_pss_sl_nr(PHY_VARS_NR_UE *UE, int position_symbol, int pss_sequence_number, test_t *test)
{
  printf("Test nr pss with Nid2 %i \n", pss_sequence_number);
  set_sequence_pss_sl(UE, position_symbol, pss_sequence_number);

  int NID2_value;
  if (pss_sequence_number < NUMBER_PSS_SEQUENCE) {
    NID2_value = pss_sequence_number;
  } else {
    NID2_value = NUMBER_PSS_SEQUENCE;
  }
  if (NID2_value < NUMBER_PSS_SEQUENCE) {
    test->number_of_tests++;
    int synchro_position = pss_synchro_nr(UE, 0, SYNCHRO_RATE_CHANGE_FACTOR);
    int delta_position = abs(position_symbol - (synchro_position* SYNCHRO_RATE_CHANGE_FACTOR));
    printf("%s ", test->test_current);
    if (UE->common_vars.eNb_id == pss_sequence_number) {
      if (delta_position !=  0) {
        if (delta_position > PSS_DETECTION_MARGIN_MAX * SYNCHRO_RATE_CHANGE_FACTOR) {
        printf("Test is fail due to wrong position %d instead of %d \n", (synchro_position * SYNCHRO_RATE_CHANGE_FACTOR), position_symbol);
        printf("%s ", test->test_current);
        printf("Test is fail : pss detected with a shift of %d \n", delta_position);
        test->number_of_fail++;
        } else {
        printf("Test is pass with warning: pss detected with a shift of %d \n", delta_position);
        test->number_of_pass_warning++;
        }
      } else {
        printf("Test is pass: pss detected at right position \n");
        test->number_of_pass++;
      }
    } else {
      printf("Test is fail due to wrong NID2 detection %d instead of %d \n", UE->common_vars.eNb_id, NID2_value);
      test->number_of_fail++;
    }
  }
}

int test_pss_sl(PHY_VARS_NR_UE *UE)
{
  test_t test = {"PSS NR", 0, 0, 0, 0};
  printf("***********************************\n");
  printf("    %s Test synchronisation \n", test.test_current);
  printf("***********************************\n");
  int test_position[] = {0, 492, 493, 56788, 56888, 111111, 151234, 151500, 200000, 250004, (307200-2048)};
  for (int index_position = 0; index_position < sizeof(test_position) / sizeof(test_position[0]); index_position++) {
    for (int number_pss_sequence = 0; number_pss_sequence < NUMBER_PSS_SEQUENCE; number_pss_sequence++) {
      test_synchro_pss_sl_nr(UE, test_position[index_position], number_pss_sequence, &test);
    }
  }
  printf("\n%s Number of tests : %d  Pass : %d Pass with warning : %d Fail : %d \n\n",
         test.test_current, test.number_of_tests, test.number_of_pass, test.number_of_pass_warning, test.number_of_fail);
  printf("%s Synchronisaton test is terminated.\n\n", test.test_current);
  test.number_of_tests = test.number_of_pass = test.number_of_pass_warning = test.number_of_fail = 0;
  free_context_synchro_nr();
  return(0);
}
