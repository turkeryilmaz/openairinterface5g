/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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


#include <string.h>
#include "SCHED_NR_UE/defs.h"
#include "nr_estimation.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "PHY/NR_REFSIG/nr_refsig.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "common/utils/nr/nr_common.h"
#include "filt16a_32.h"
#include <openair1/PHY/TOOLS/phy_scope_interface.h>

//#define DEBUG_PDSCH
//#define DEBUG_PDCCH

#define CH_INTERP 0
#define NO_INTERP 1

__attribute__((always_inline)) inline c16_t c32x16cumulVectVectWithSteps(c16_t *in1,
                                                                         int *offset1,
                                                                         const int step1,
                                                                         c16_t *in2,
                                                                         int *offset2,
                                                                         const int step2,
                                                                         const int modulo2,
                                                                         const int N) {

  int localOffset1=*offset1;
  int localOffset2=*offset2;
  c32_t cumul={0}; 
  for (int i=0; i<N; i++) {
    cumul=c32x16maddShift(in1[localOffset1], in2[localOffset2], cumul, 15);
    localOffset1+=step1;
    localOffset2= (localOffset2 + step2) % modulo2;
  }
  *offset1=localOffset1;
  *offset2=localOffset2;
  return c16x32div(cumul, N);
}

int nr_pbch_dmrs_correlation(PHY_VARS_NR_UE *ue,
                             UE_nr_rxtx_proc_t *proc,
                             uint8_t gNB_id,
                             unsigned char Ns,
                             unsigned char symbol,
                             int dmrss,
                             NR_UE_SSB *current_ssb)
{
  int pilot[200] __attribute__((aligned(16)));
  unsigned short k;
  unsigned int pilot_cnt;
  int16_t ch[2],*pil,*rxF;
  int symbol_offset;


  uint8_t nushift;
  uint8_t ssb_index=current_ssb->i_ssb;
  uint8_t n_hf=current_ssb->n_hf;
  int **rxdataF=ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;

  nushift =  ue->frame_parms.Nid_cell%4;
  ue->frame_parms.nushift = nushift;
  unsigned int  ssb_offset = ue->frame_parms.first_carrier_offset + ue->frame_parms.ssb_start_subcarrier;
  if (ssb_offset>= ue->frame_parms.ofdm_symbol_size) ssb_offset-=ue->frame_parms.ofdm_symbol_size;

  AssertFatal(dmrss >= 0 && dmrss < 3,
	      "symbol %d is illegal for PBCH DM-RS \n",
	      dmrss);

  symbol_offset = ue->frame_parms.ofdm_symbol_size*symbol;


  k = nushift;

#ifdef DEBUG_CH
  printf("PBCH DMRS Correlation : ThreadId %d, gNB_id %d , OFDM size %d, Ncp=%d, Ns=%d, k=%d symbol %d\n",proc->thread_id, gNB_id,ue->frame_parms.ofdm_symbol_size,
         ue->frame_parms.Ncp,Ns,k, symbol);
#endif

  // generate pilot
  nr_pbch_dmrs_rx(dmrss,ue->nr_gold_pbch[n_hf][ssb_index], &pilot[0]);

  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++) {

    int re_offset = ssb_offset;
    pil   = (int16_t *)&pilot[0];
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];

#ifdef DEBUG_CH
    printf("pbch ch est pilot addr %p RB_DL %d\n",&pilot[0], ue->frame_parms.N_RB_DL);
    printf("k %d, first_carrier %d\n",k,ue->frame_parms.first_carrier_offset);
    printf("rxF addr %p\n", rxF);
#endif
    //if ((ue->frame_parms.N_RB_DL&1)==0) {

    // Treat first 2 pilots specially (left edge)
    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

    current_ssb->c_re += ch[0];
    current_ssb->c_im += ch[1];

#ifdef DEBUG_CH
    printf("ch 0 %d\n",((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1]));
    printf("pilot 0 : rxF - > (%d,%d) addr %p  ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],&rxF[0],ch[0],ch[1],pil[0],pil[1]);
#endif

    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];


    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

    current_ssb->c_re += ch[0];
    current_ssb->c_im += ch[1];

#ifdef DEBUG_CH
    printf("pilot 1 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];

    current_ssb->c_re += ch[0];
    current_ssb->c_im += ch[1];

#ifdef DEBUG_CH
    printf("pilot 2 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];

    for (pilot_cnt=3; pilot_cnt<(3*20); pilot_cnt += 3) {

      //	if (pilot_cnt == 30)
      //	  rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k)];

      // in 2nd symbol, skip middle  REs (48 with DMRS,  144 for SSS, and another 48 with DMRS) 
      if (dmrss == 1 && pilot_cnt == 12) {
	pilot_cnt=48;
	re_offset = (re_offset+144) % ue->frame_parms.ofdm_symbol_size;
	rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
      }
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
      
      current_ssb->c_re += ch[0];
      current_ssb->c_im += ch[1];

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
        
  
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

      current_ssb->c_re += ch[0];
      current_ssb->c_im += ch[1];

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+1,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
        

      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

      current_ssb->c_re += ch[0];
      current_ssb->c_im += ch[1];

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+2,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];

    }


    //}

  }
  return(0);
}

int nr_psbch_dmrs_correlation(PHY_VARS_NR_UE *ue,
                             UE_nr_rxtx_proc_t *proc,
                             uint8_t gNB_id,
                             unsigned char Ns,
                             unsigned char symbol,
                             int dmrss,
                             NR_UE_SSB *current_ssb)
{
  int pilot[300] __attribute__((aligned(16)));
  unsigned short k;
  unsigned int pilot_cnt;
  int16_t ch[2],*pil,*rxF;
  int symbol_offset;


  uint8_t nushift;
  uint8_t ssb_index=current_ssb->i_ssb;
  int **rxdataF=ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;

  nushift =  0; //ue->frame_parms.Nid_cell%4;
  ue->frame_parms.nushift = nushift;
  unsigned int  ssb_offset = ue->frame_parms.first_carrier_offset + ue->frame_parms.ssb_start_subcarrier;
  if (ssb_offset>= ue->frame_parms.ofdm_symbol_size) ssb_offset-=ue->frame_parms.ofdm_symbol_size;

  AssertFatal(dmrss >= 0 && dmrss <= 12,
	      "symbol %d is illegal for PSBCH DM-RS \n",
	      dmrss);

  symbol_offset = ue->frame_parms.ofdm_symbol_size*symbol;

  k = nushift;

#ifdef DEBUG_CH
  printf("PSBCH DMRS Correlation : ThreadId %d, gNB_id %d , OFDM size %d, Ncp=%d, Ns=%d, k=%d symbol %d\n",proc->thread_id, gNB_id,ue->frame_parms.ofdm_symbol_size,
         ue->frame_parms.Ncp,Ns,k, symbol);
#endif

  if (0 < dmrss && dmrss < 5) return 0;
  // generate pilot
  nr_psbch_dmrs_rx(dmrss, &ue->nr_gold_psbch[ssb_index], &pilot[0]);

  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++) {
    int re_offset = ssb_offset;
    pil   = (int16_t *)&pilot[0];
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
    current_ssb->c_re += ch[0];
    current_ssb->c_im += ch[1];

    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
    current_ssb->c_re += ch[0];
    current_ssb->c_im += ch[1];

    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
    current_ssb->c_re += ch[0];
    current_ssb->c_im += ch[1];

    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];

    for (pilot_cnt = 3; pilot_cnt < (3 * 11); pilot_cnt += 3) {
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
      current_ssb->c_re += ch[0];
      current_ssb->c_im += ch[1];

      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
      current_ssb->c_re += ch[0];
      current_ssb->c_im += ch[1];

      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
      current_ssb->c_re += ch[0];
      current_ssb->c_im += ch[1];

      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    }
  }
  return(0);
}


int nr_pbch_channel_estimation(PHY_VARS_NR_UE *ue,
                               int estimateSz,
                               struct complex16 dl_ch_estimates [][estimateSz],
                               struct complex16 dl_ch_estimates_time [][ue->frame_parms.ofdm_symbol_size],
                               UE_nr_rxtx_proc_t *proc,
                               uint8_t gNB_id,
                               unsigned char Ns,
                               unsigned char symbol,
                               int dmrss,
                               uint8_t ssb_index,
                               uint8_t n_hf)
{
  int pilot[200] __attribute__((aligned(16)));
  unsigned short k;
  unsigned int pilot_cnt;
  int16_t *pil,*rxF,*dl_ch,*fl,*fm,*fr;
  int ch_offset,symbol_offset;
  //int slot_pbch;

  uint8_t nushift;
   int **rxdataF=ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;

  nushift =  ue->frame_parms.Nid_cell%4;
  ue->frame_parms.nushift = nushift;
  unsigned int  ssb_offset = ue->frame_parms.first_carrier_offset + ue->frame_parms.ssb_start_subcarrier;
  if (ssb_offset>= ue->frame_parms.ofdm_symbol_size) ssb_offset-=ue->frame_parms.ofdm_symbol_size;

  ch_offset     = ue->frame_parms.ofdm_symbol_size*symbol;

  AssertFatal(dmrss >= 0 && dmrss < 3,
	      "symbol %d is illegal for PBCH DM-RS \n",
	      dmrss);

  symbol_offset = ue->frame_parms.ofdm_symbol_size*symbol;


  k = nushift;

#ifdef DEBUG_CH
  printf("PBCH Channel Estimation : ThreadId %d, gNB_id %d ch_offset %d, OFDM size %d, Ncp=%d, Ns=%d, k=%d symbol %d\n",proc->thread_id, gNB_id,ch_offset,ue->frame_parms.ofdm_symbol_size,
         ue->frame_parms.Ncp,Ns,k, symbol);
#endif

  switch (k) {
  case 0:
    fl = filt16a_l0;
    fm = filt16a_m0;
    fr = filt16a_r0;
    break;

  case 1:
    fl = filt16a_l1;
    fm = filt16a_m1;
    fr = filt16a_r1;
    break;

  case 2:
    fl = filt16a_l2;
    fm = filt16a_m2;
    fr = filt16a_r2;
    break;

  case 3:
    fl = filt16a_l3;
    fm = filt16a_m3;
    fr = filt16a_r3;
    break;

  default:
    msg("pbch_channel_estimation: k=%d -> ERROR\n",k);
    return(-1);
    break;
  }

  idft_size_idx_t idftsizeidx;
  
  switch (ue->frame_parms.ofdm_symbol_size) {
  case 128:
    idftsizeidx = IDFT_128;
    break;
    
  case 256:
    idftsizeidx = IDFT_256;
    break;
    
  case 512:
    idftsizeidx = IDFT_512;
    break;
    
  case 1024:
    idftsizeidx = IDFT_1024;
    break;
    
  case 1536:
    idftsizeidx = IDFT_1536;
    break;
    
  case 2048:
    idftsizeidx = IDFT_2048;
    break;
    
  case 3072:
    idftsizeidx = IDFT_3072;
    break;
    
  case 4096:
    idftsizeidx = IDFT_4096;
    break;
    
  default:
    printf("unsupported ofdm symbol size \n");
    assert(0);
  }
  
  // generate pilot
  nr_pbch_dmrs_rx(dmrss,ue->nr_gold_pbch[n_hf][ssb_index], &pilot[0]);

  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++) {

    int re_offset = ssb_offset;
    pil   = (int16_t *)&pilot[0];
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    dl_ch = (int16_t *)&dl_ch_estimates[aarx][ch_offset];

    memset(dl_ch,0,sizeof(struct complex16)*(ue->frame_parms.ofdm_symbol_size));

#ifdef DEBUG_CH
    printf("pbch ch est pilot addr %p RB_DL %d\n",&pilot[0], ue->frame_parms.N_RB_DL);
    printf("k %d, first_carrier %d\n",k,ue->frame_parms.first_carrier_offset);
    printf("rxF addr %p\n", rxF);
    printf("dl_ch addr %p\n",dl_ch);
#endif

    // Treat first 2 pilots specially (left edge)
    int16_t ch[2];
    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
    printf("ch 0 %d\n",((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1]));
    printf("pilot 0 : rxF - > (%d,%d) addr %p  ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],&rxF[0],ch[0],ch[1],pil[0],pil[1]);
#endif
    multadd_real_vector_complex_scalar(fl,
				       ch,
				       dl_ch,
				       16);
    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];

    //for (int i= 0; i<8; i++)
    //printf("dl_ch addr %p %d\n", dl_ch+i, *(dl_ch+i));

    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);


#ifdef DEBUG_CH
    printf("pilot 1 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
    multadd_real_vector_complex_scalar(fm,
				       ch,
				       dl_ch,
				       16);
    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];

    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
    printf("pilot 2 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

    multadd_real_vector_complex_scalar(fr,
				       ch,
				       dl_ch,
				       16);
    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    dl_ch += 24;

    for (pilot_cnt=3; pilot_cnt<(3*20); pilot_cnt += 3) {

      //	if (pilot_cnt == 30)
      //	  rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k)];

      // in 2nd symbol, skip middle  REs (48 with DMRS,  144 for SSS, and another 48 with DMRS) 
      if (dmrss == 1 && pilot_cnt == 12) {
        pilot_cnt=48;
        re_offset = (re_offset+144) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
        dl_ch += 288;
      }
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fl,
					 ch,
					 dl_ch,
					 16);

      //for (int i= 0; i<8; i++)
      //            printf("pilot_cnt %d dl_ch %d %d\n", pilot_cnt, dl_ch+i, *(dl_ch+i));

      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
        
  
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+1,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fm,
					 ch,
					 dl_ch,
					 16);
      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
        

      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+2,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

      multadd_real_vector_complex_scalar(fr,
					 ch,
					 dl_ch,
					 16);
      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
      dl_ch += 24;

    }

    if( dmrss == 2) // update time statistics for last PBCH symbol
    {
      // do ifft of channel estimate
      LOG_D(PHY,"Channel Impulse Computation Slot %d Symbol %d ch_offset %d\n", Ns, symbol, ch_offset);
      idft(idftsizeidx,
	   (int16_t*) &dl_ch_estimates[aarx][ch_offset],
	   (int16_t*) dl_ch_estimates_time[aarx],
	   1);
    }
  }

  if (dmrss == 2)
    UEscopeCopy(ue, pbchDlChEstimateTime, (void*)dl_ch_estimates_time, sizeof(struct complex16), ue->frame_parms.nb_antennas_rx, idftsizeidx);

  return(0);
}

int nr_psbch_channel_estimation(PHY_VARS_NR_UE *ue,
                               int estimateSz,
                               struct complex16 dl_ch_estimates [][estimateSz],
                               struct complex16 dl_ch_estimates_time [][ue->frame_parms.ofdm_symbol_size],
                               UE_nr_rxtx_proc_t *proc,
                               uint8_t gNB_id,
                               unsigned char Ns,
                               unsigned char symbol,
                               int dmrss,
                               uint8_t ssb_index,
                               uint8_t n_hf)
{
  int pilot[200] __attribute__((aligned(16)));
  unsigned short k;
  unsigned int pilot_cnt;
  int16_t *pil,*rxF,*dl_ch,*fl,*fm,*fr;
  int ch_offset,symbol_offset;

  uint8_t nushift;
   int **rxdataF=ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;

  nushift = 0;// ue->frame_parms.Nid_cell%4;
  ue->frame_parms.nushift = nushift;
  unsigned int  ssb_offset = ue->frame_parms.first_carrier_offset + ue->frame_parms.ssb_start_subcarrier;
  if (ssb_offset>= ue->frame_parms.ofdm_symbol_size) ssb_offset-=ue->frame_parms.ofdm_symbol_size;

  ch_offset     = ue->frame_parms.ofdm_symbol_size*symbol;

  AssertFatal(dmrss >= 0 && dmrss <= 13,
	      "symbol %d is illegal for PSBCH DM-RS \n",
	      dmrss);

  symbol_offset = ue->frame_parms.ofdm_symbol_size*symbol;

  k = nushift;

#ifdef DEBUG_CH
  printf("PSBCH Channel Estimation : ThreadId %d, gNB_id %d ch_offset %d, OFDM size %d, Ncp=%d, Ns=%d, k=%d symbol %d\n",proc->thread_id, gNB_id,ch_offset,ue->frame_parms.ofdm_symbol_size,
         ue->frame_parms.Ncp,Ns,k, symbol);
#endif

  fl = filt16a_l0;
  fm = filt16a_m0;
  fr = filt16a_r0;

  idft_size_idx_t idftsizeidx;
  
  switch (ue->frame_parms.ofdm_symbol_size) {
  case 128:
    idftsizeidx = IDFT_128;
    break;
    
  case 256:
    idftsizeidx = IDFT_256;
    break;
    
  case 512:
    idftsizeidx = IDFT_512;
    break;
    
  case 1024:
    idftsizeidx = IDFT_1024;
    break;
    
  case 1536:
    idftsizeidx = IDFT_1536;
    break;
    
  case 2048:
    idftsizeidx = IDFT_2048;
    break;
    
  case 3072:
    idftsizeidx = IDFT_3072;
    break;
    
  case 4096:
    idftsizeidx = IDFT_4096;
    break;
    
  default:
    printf("unsupported ofdm symbol size \n");
    assert(0);
  }
  
  // generate pilot
  nr_psbch_dmrs_rx(dmrss, &ue->nr_gold_psbch[ssb_index], &pilot[0]);

  for (int aarx = 0; aarx < ue->frame_parms.nb_antennas_rx; aarx++) {

    int re_offset = ssb_offset;
    pil   = (int16_t *)&pilot[0];
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    dl_ch = (int16_t *)&dl_ch_estimates[aarx][ch_offset];

    memset(dl_ch,0,sizeof(struct complex16)*(ue->frame_parms.ofdm_symbol_size));

#ifdef DEBUG_CH
    printf("psbch ch est pilot addr %p RB_DL %d\n",&pilot[0], ue->frame_parms.N_RB_DL);
    printf("k %d, first_carrier %d\n",k,ue->frame_parms.first_carrier_offset);
    printf("rxF addr %p\n", rxF);
    printf("dl_ch addr %p\n",dl_ch);
#endif

    // Treat first 2 pilots specially (left edge)
    int16_t ch[2];
    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
    printf("ch 0 %d\n",((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1]));
    printf("pilot 0 : rxF - > (%d,%d) addr %p  ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],&rxF[0],ch[0],ch[1],pil[0],pil[1]);
#endif
    multadd_real_vector_complex_scalar(fl, ch, dl_ch, 16);
    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];

    //for (int i= 0; i<8; i++)
    //printf("dl_ch addr %p %d\n", dl_ch+i, *(dl_ch+i));

    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
    printf("pilot 1 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
    multadd_real_vector_complex_scalar(fm, ch, dl_ch, 16);
    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
    printf("pilot 2 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

    multadd_real_vector_complex_scalar(fr,
				       ch,
				       dl_ch,
				       16);
    pil += 2;
    re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
    dl_ch += 24;

    for (pilot_cnt = 3; pilot_cnt < (3 * 11); pilot_cnt += 3) {
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fl,
					 ch,
					 dl_ch,
					 16);

      //for (int i= 0; i<8; i++)
      //            printf("pilot_cnt %d dl_ch %d %d\n", pilot_cnt, dl_ch+i, *(dl_ch+i));

      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
        
  
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+1,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fm,
					 ch,
					 dl_ch,
					 16);
      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
        

      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_CH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+2,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

      multadd_real_vector_complex_scalar(fr,
					 ch,
					 dl_ch,
					 16);
      pil += 2;
      re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+re_offset)];
      dl_ch += 24;

    }

    if( dmrss == 12) // update time statistics for last PSBCH symbol
    {
      // do ifft of channel estimate
      LOG_D(PHY,"Channel Impulse Computation Slot %d Symbol %d ch_offset %d\n", Ns, symbol, ch_offset);
      idft(idftsizeidx,
	   (int16_t*) &dl_ch_estimates[aarx][ch_offset],
	   (int16_t*) dl_ch_estimates_time[aarx],
	   1);
    }
  }

  if (dmrss == 12)
    UEscopeCopy(ue, psbchDlChEstimateTime, (void*)dl_ch_estimates_time, sizeof(struct complex16), ue->frame_parms.nb_antennas_rx, idftsizeidx);

  return(0);
}

int nr_pssch_channel_estimation(PHY_VARS_NR_UE *ue,
                                unsigned char Ns,
                                unsigned short p,
                                unsigned char symbol,
                                int ul_id,
                                unsigned short bwp_start_subcarrier,
                                nfapi_nr_pssch_pdu_t *pssch_pdu) {
  c16_t pilot[3280] __attribute__((aligned(16)));
  int16_t *fl,*fm,*fr,*fml,*fmr,*fmm,*fdcl,*fdcr,*fdclh,*fdcrh;

  const int chest_freq = ue->chest_freq;

#ifdef DEBUG_CH
  FILE *debug_ch_est;
  debug_ch_est = fopen("debug_ch_est.txt","w");
#endif
  //uint16_t Nid_cell = (eNB_offset == 0) ? ue->frame_parms.Nid_cell : ue->measurements.adj_cell_id[eNB_offset-1];
  c16_t **sl_ch_estimates  = (c16_t **) ue->pssch_vars[ul_id]->sl_ch_estimates;
  const int symbolSize = ue->frame_parms.ofdm_symbol_size;
  const int soffset = (Ns&3)*ue->frame_parms.symbols_per_slot*symbolSize;
  const int nushift = (p>>1)&1;
  ue->frame_parms.nushift = nushift;
  int ch_offset     = symbolSize*symbol;
  const int symbol_offset = symbolSize*symbol;

  const int k0 = bwp_start_subcarrier;
  const int nb_rb_pusch = pssch_pdu->rb_size;

  LOG_D(PHY, "In %s: ch_offset %d, soffset %d, symbol_offset %d, OFDM size %d, Ns = %d, k0 = %d, symbol %d\n",
        __FUNCTION__,
        ch_offset, soffset,
        symbol_offset,
        symbolSize,
        Ns,
        k0,
        symbol);

  switch (nushift) {
    case 0:
      fl = filt8_l0;
      fm = filt8_m0;
      fr = filt8_r0;
      fmm = filt8_mm0;
      fml = filt8_m0;
      fmr = filt8_mr0;
      fdcl = filt8_dcl0;
      fdcr = filt8_dcr0;
      fdclh = filt8_dcl0_h;
      fdcrh = filt8_dcr0_h;
      break;

    case 1:
      fl = filt8_l1;
      fm = filt8_m1;
      fr = filt8_r1;
      fmm = filt8_mm1;
      fml = filt8_ml1;
      fmr = filt8_mm1;
      fdcl = filt8_dcl1;
      fdcr = filt8_dcr1;
      fdclh = filt8_dcl1_h;
      fdcrh = filt8_dcr1_h;
      break;

    default:
#ifdef DEBUG_CH
      if (debug_ch_est)
        fclose(debug_ch_est);

#endif
      return(-1);
      break;
  }

  //------------------generate DMRS------------------//

  if(pssch_pdu->ul_dmrs_scrambling_id != ue->pssch_gold_init_rx)  {
    ue->pssch_gold_init_rx = pssch_pdu->ul_dmrs_scrambling_id;
    nr_gold_pssch(ue, pssch_pdu->ul_dmrs_scrambling_id);
  }

  if (pssch_pdu->transform_precoding == transformPrecoder_disabled) {
    nr_pssch_dmrs_rx(ue, Ns, ue->nr_gold_pssch_dmrs[Ns][symbol], (int32_t *)pilot, (1000+p), 0, nb_rb_pusch,
                     (pssch_pdu->bwp_start + pssch_pdu->rb_start)*NR_NB_SC_PER_RB, pssch_pdu->dmrs_config_type);
  }
#if 0
  else { // if transform precoding or SC-FDMA is enabled in Uplink
    // NR_SC_FDMA supports type1 DMRS so only 6 DMRS REs per RB possible
    const uint16_t index = get_index_for_dmrs_lowpapr_seq(nb_rb_pusch * (NR_NB_SC_PER_RB/2));
    const uint8_t u = pssch_pdu->dfts_ofdm.low_papr_group_number;
    const uint8_t v = pssch_pdu->dfts_ofdm.low_papr_sequence_number;
    int16_t *dmrs_seq = gNB_dmrs_lowpaprtype1_sequence[u][v][index];
    AssertFatal(index >= 0, "Num RBs not configured according to 3GPP 38.211 section 6.3.1.4. For PUSCH with transform precoding, num RBs cannot be multiple of any other primenumber other than 2,3,5\n");
    AssertFatal(dmrs_seq != NULL, "DMRS low PAPR seq not found, check if DMRS sequences are generated");
    LOG_D(PHY,"Transform Precoding params. u: %d, v: %d, index for dmrsseq: %d\n", u, v, index);
    nr_pusch_lowpaprtype1_dmrs_rx(ue, Ns, dmrs_seq, (int32_t *)pilot, 1000, 0, nb_rb_pusch, 0, pssch_pdu->dmrs_config_type);
#ifdef DEBUG_PUSCH
    printf ("NR_UL_CHANNEL_EST: index %d, u %d,v %d\n", index, u, v);
    LOG_M("gNb_DMRS_SEQ.m","gNb_DMRS_SEQ", dmrs_seq,6*nb_rb_pusch,1,1);
#endif
  }
#endif
  //------------------------------------------------//

#ifdef DEBUG_PUSCH

  for (int i = 0; i < (6 * nb_rb_pusch); i++) {
    LOG_I(PHY, "In %s: %d + j*(%d)\n", __FUNCTION__, pilot[i].r,pilot[i].i);
  }

#endif
  const uint8_t b_shift = pssch_pdu->nrOfLayers == 1;

  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++) {
    c16_t *rxdataF = (c16_t *)&ue->common_vars.common_vars_rx_data_per_thread[ul_id].rxdataF[aarx][symbol_offset];
    c16_t *ul_ch = &sl_ch_estimates[p*ue->frame_parms.nb_antennas_rx+aarx][ch_offset];

    memset(ul_ch,0,sizeof(*ul_ch)*symbolSize);
#ifdef DEBUG_PUSCH
    LOG_I(PHY, "In %s symbol_offset %d, nushift %d\n", __FUNCTION__, symbol_offset, nushift);
    LOG_I(PHY, "In %s ch est pilot, N_RB_UL %d\n", __FUNCTION__, ue->frame_parms.N_RB_UL);
    LOG_I(PHY, "In %s bwp_start_subcarrier %d, k0 %d, first_carrier %d, nb_rb_pusch %d\n", __FUNCTION__, bwp_start_subcarrier, k0, ue->frame_parms.first_carrier_offset, nb_rb_pusch);
    LOG_I(PHY, "In %s ul_ch addr %p nushift %d\n", __FUNCTION__, ul_ch, nushift);
#endif

    if (pssch_pdu->dmrs_config_type == pusch_dmrs_type1 && chest_freq == 0) {
      c16_t *pil   = pilot;
      int re_offset = k0;
      LOG_D(PHY,"PUSCH estimation DMRS type 1, Freq-domain interpolation");
      // For configuration type 1: k = 4*n + 2*k' + delta,
      // where k' is 0 or 1, and delta is in Table 6.4.1.1.3-1 from TS 38.211
      int pilot_cnt = 0;
      int delta = nr_pusch_dmrs_delta(pusch_dmrs_type1, p);

      for (int n = 0; n < 3*nb_rb_pusch; n++) {
        // LS estimation
        c32_t ch = {0};

        for (int k_line = 0; k_line <= 1; k_line++) {
          re_offset = (k0 + (n << 2) + (k_line << 1) + delta) % symbolSize;
          ch=c32x16maddShift(*pil,
                             rxdataF[soffset + re_offset],
                             ch,
                             15+b_shift);
          pil++;
        }

        c16_t ch16= {.r=(int16_t)ch.r, .i=(int16_t)ch.i};

        // Channel interpolation
        for (int k_line = 0; k_line <= 1; k_line++) {
#ifdef DEBUG_PUSCH
          re_offset = (k0 + (n << 2) + (k_line << 1)) % symbolSize;
          c16_t *rxF = &rxdataF[soffset + re_offset];
          printf("pilot %4u: pil -> (%6d,%6d), rxF -> (%4d,%4d), ch -> (%4d,%4d)\n",
                 pilot_cnt, pil->r, pil->i, rxF->r, rxF->i, ch.r, ch.i);
#endif
          
          if (pilot_cnt == 0) {
            c16multaddVectRealComplex(fl, &ch16, ul_ch, 8);
          } else if (pilot_cnt == 1) {
            c16multaddVectRealComplex(fml, &ch16, ul_ch, 8);
          } else if (pilot_cnt == (6*nb_rb_pusch-2)) {
            c16multaddVectRealComplex(fmr, &ch16, ul_ch, 8);
            ul_ch+=4;
          } else if (pilot_cnt == (6*nb_rb_pusch-1)) {
            c16multaddVectRealComplex(fr, &ch16, ul_ch, 8);
          } else if (pilot_cnt%2 == 0) {
            c16multaddVectRealComplex(fmm, &ch16, ul_ch, 8);
            ul_ch+=4;
          } else {
            c16multaddVectRealComplex(fm, &ch16, ul_ch, 8);
          }
          
          pilot_cnt++;
        }
      }

      // check if PRB crosses DC and improve estimates around DC
      if ((bwp_start_subcarrier < symbolSize) && (bwp_start_subcarrier+nb_rb_pusch*12 >= symbolSize)) {
        ul_ch = &sl_ch_estimates[p*ue->frame_parms.nb_antennas_rx+aarx][ch_offset];
        const uint16_t idxDC = symbolSize - bwp_start_subcarrier;
        re_offset = k0;
        pil = pilot + idxDC / 2 - 1;
        ul_ch += idxDC - 2 ;
        ul_ch = memset(ul_ch, 0, sizeof(*ul_ch)*5);
        re_offset = (re_offset + idxDC - 2) % symbolSize;
        const c16_t ch=c16mulShift(*pil,
                                   rxdataF[soffset+nushift+re_offset],
                                   15);

        // for proper alignment of SIMD vectors
        if((ue->frame_parms.N_RB_UL&1)==0) {
          c16multaddVectRealComplex(fdcl, &ch, ul_ch-2, 8);
          pil += 2;
          re_offset = (re_offset+4) % symbolSize;
          const c16_t ch_tmp=c16mulShift(*pil,
                                         rxdataF[nushift+re_offset],
                                         15);
          c16multaddVectRealComplex(fdcr, &ch_tmp, ul_ch-2, 8);
        } else {
          c16multaddVectRealComplex(fdclh, &ch, ul_ch, 8);
          pil += 2;
          re_offset = (re_offset+4) % symbolSize;
          const c16_t ch_tmp=c16mulShift(*pil,
                                         rxdataF[soffset+nushift+re_offset],
                                         15);
          c16multaddVectRealComplex(fdcrh, &ch_tmp, ul_ch, 8);
        }
      }

#ifdef DEBUG_PUSCH
      ul_ch = &sl_ch_estimates[p*ue->frame_parms.nb_antennas_rx+aarx][ch_offset];

      for(uint16_t idxP=0; idxP<ceil((float)nb_rb_pusch*12/8); idxP++) {
        printf("(%3d)\t",idxP);

        for(uint8_t idxI=0; idxI<8; idxI++) {
          printf("%d\t%d\t",ul_ch[idxP*8+idxI].r,ul_ch[idxP*8+idxI].i);
        }

        printf("\n");
      }

#endif
    } else if (pssch_pdu->dmrs_config_type == pusch_dmrs_type2 && chest_freq == 0) { //pusch_dmrs_type2  |p_r,p_l,d,d,d,d,p_r,p_l,d,d,d,d|
      LOG_D(PHY,"PUSCH estimation DMRS type 2, Freq-domain interpolation");
      c16_t *pil   = pilot;
      int re_offset = k0;
      // Treat first DMRS specially (left edge)
      *ul_ch=c16mulShift(*pil,
                         rxdataF[soffset+nushift+re_offset],
                         15);
      pil++;
      ul_ch++;
      re_offset = (re_offset + 1)%symbolSize;
      ch_offset++;

      // TO verify: used after the loop, likely a piece of code is missing for ch_r
      c16_t ch_r;
      for (int re_cnt = 1; re_cnt < (nb_rb_pusch*NR_NB_SC_PER_RB) - 5; re_cnt += 6) {
        c16_t ch_l=c16mulShift(*pil,
                               rxdataF[soffset+nushift+re_offset],
                               15);
        *ul_ch = ch_l;
        pil++;
        ul_ch++;
        ch_offset++;
        multadd_real_four_symbols_vector_complex_scalar(filt8_ml2,
                                                      &ch_l,
                                                      ul_ch);
        re_offset = (re_offset+5)%symbolSize;
        ch_r=c16mulShift(*pil,
                         rxdataF[soffset+nushift+re_offset],
                         15);
        multadd_real_four_symbols_vector_complex_scalar(filt8_mr2,
                                                        &ch_r,
                                                        ul_ch);
        //for (int re_idx = 0; re_idx < 8; re_idx += 2)
        //printf("ul_ch = %d + j*%d\n", ul_ch[re_idx], ul_ch[re_idx+1]);
        ul_ch += 4;
        ch_offset += 4;
        *ul_ch = ch_r;
        pil++;
        ul_ch++;
        ch_offset++;
        re_offset = (re_offset + 1)%symbolSize;
      }
      
      // Treat last pilot specially (right edge)
      c16_t ch_l=c16mulShift(*pil,
                             rxdataF[soffset+nushift+re_offset],
                             15);
      *ul_ch = ch_l;
      ul_ch++;
      ch_offset++;
      multadd_real_four_symbols_vector_complex_scalar(filt8_rr1,
                                                      &ch_l,
                                                      ul_ch);
      multadd_real_four_symbols_vector_complex_scalar(filt8_rr2,
                                                      &ch_r,
                                                      ul_ch);
      __m128i *ul_ch_128 = (__m128i *)&sl_ch_estimates[p*ue->frame_parms.nb_antennas_rx+aarx][ch_offset];
      ul_ch_128[0] = _mm_slli_epi16 (ul_ch_128[0], 2);
    } 

    else if (pssch_pdu->dmrs_config_type == pusch_dmrs_type1) { // this is case without frequency-domain linear interpolation, just take average of LS channel estimates of 6 DMRS REs and use a common value for the whole PRB
      LOG_D(PHY,"PUSCH estimation DMRS type 1, no Freq-domain interpolation\n");
      c16_t *rxF   =  &rxdataF[soffset + nushift];
      int pil_offset = 0;
      int re_offset = k0;
      c16_t ch;
      
      // First PRB
      ch=c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);

#if NO_INTERP
      for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
        *ul_ch=ch;
#else
      c16multaddVectRealComplex(filt8_avlip0, ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip1, ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip2, ch, ul_ch, 8);
      ul_ch -= 12;
#endif

      for (int pilot_cnt=6; pilot_cnt<6*(nb_rb_pusch-1); pilot_cnt += 6) {
        ch=c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);

#if NO_INTERP
      for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
          *ul_ch=ch;
#else
        ul_ch[3].r += (ch.r * 1365)>>15; // 1/12*16384
        ul_ch[3].i += (ch.i * 1365)>>15; // 1/12*16384

        ul_ch += 4;
        multadd_real_vector_complex_scalar(filt8_avlip3, ch, ul_ch, 8);

        ul_ch += 8;
        multadd_real_vector_complex_scalar(filt8_avlip4, ch, ul_ch, 8);

        ul_ch += 8;
        multadd_real_vector_complex_scalar(filt8_avlip5, ch, ul_ch, 8);
        ul_ch -= 8;
#endif
      }
      // Last PRB
      ch=c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);
      
#if NO_INTERP
      for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
        *ul_ch=ch;
#else
      ul_ch[3].r += (ch.r * 1365)>>15; // 1/12*16384
      ul_ch[3].i += (ch.i * 1365)>>15; // 1/12*16384
      
      ul_ch += 4;
      c16multaddVectRealComplex(filt8_avlip3,
                                         ch,
                                         ul_ch,
                                         8);
      
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip6,
                                         ch,
                                         ul_ch,
                                         8);
#endif
    } else  { // this is case without frequency-domain linear interpolation, just take average of LS channel estimates of 4 DMRS REs and use a common value for the whole PRB
      LOG_D(PHY,"PUSCH estimation DMRS type 2, no Freq-domain interpolation");
      c16_t *pil   = pilot;
      int re_offset = k0;
      c32_t ch0={0};
      //First PRB
      ch0=c32x16mulShift(*pil,
                         rxdataF[soffset + nushift + re_offset],
                         15);
      pil++;
      re_offset = (re_offset+1) % symbolSize;
      ch0=c32x16maddShift(*pil,
                          rxdataF[nushift+re_offset],
                          ch0,
                          15);
      pil++;
      re_offset = (re_offset+5) % symbolSize;
      ch0=c32x16maddShift(*pil,
                          rxdataF[nushift+re_offset],
                          ch0,
                          15);
      re_offset = (re_offset+1) % symbolSize;
      ch0=c32x16maddShift(*pil,
                          rxdataF[nushift+re_offset],
                          ch0,
                          15);
      pil++;
      re_offset = (re_offset+5) % symbolSize;

      c16_t ch=c16x32div(ch0, 4);
#if NO_INTERP
      for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
        *ul_ch=ch;
#else
      c16multaddVectRealComplex(filt8_avlip0, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip1, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip2, &ch, ul_ch, 8);
      ul_ch -= 12;
#endif

      for (int pilot_cnt=4; pilot_cnt<4*(nb_rb_pusch-1); pilot_cnt += 4) {
        c32_t ch0;
        ch0=c32x16mulShift(*pil, rxdataF[nushift+re_offset], 15);
        pil++;
        re_offset = (re_offset+1) % symbolSize;

        ch0=c32x16maddShift(*pil, rxdataF[nushift+re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset+5) % symbolSize;

        ch0=c32x16maddShift(*pil, rxdataF[nushift+re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset+1) % symbolSize;

        ch0=c32x16maddShift(*pil, rxdataF[nushift+re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset+5) % symbolSize;

        ch=c16x32div(ch0, 4);

#if NO_INTERP
        for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
          *ul_ch=ch;
#else
        ul_ch[3]=c16maddShift(ch,(c16_t) {1365,1365},15); // 1365 = 1/12*16384 (full range is +/- 32768)
        ul_ch += 4;
        c16multaddVectRealComplex(filt8_avlip3, &ch, ul_ch, 8);
        ul_ch += 8;
        c16multaddVectRealComplex(filt8_avlip4, &ch, ul_ch, 8);
        ul_ch += 8;
        c16multaddVectRealComplex(filt8_avlip5, &ch, ul_ch, 8);
        ul_ch -= 8;
#endif
      }

      // Last PRB
      ch0=c32x16mulShift(*pil, rxdataF[nushift+re_offset], 15);
      pil++;
      re_offset = (re_offset+1) % symbolSize;
      
      ch0=c32x16maddShift(*pil, rxdataF[nushift+re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset+5) % symbolSize;
      
      ch0=c32x16maddShift(*pil, rxdataF[nushift+re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset+1) % symbolSize;
      
      ch0=c32x16maddShift(*pil, rxdataF[nushift+re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset+5) % symbolSize;
      
      ch=c16x32div(ch0, 4);
#if NO_INTERP
      for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
          *ul_ch=ch;
#else
      ul_ch[3]=c16maddShift(ch, c16_t {1365,1365},15);// 1365 = 1/12*16384 (full range is +/- 32768)
      ul_ch += 4;
      c16multaddVectRealComplex(filt8_avlip3, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip6, &ch, ul_ch, 8);
#endif
    }
    
#ifdef DEBUG_PUSCH
    ul_ch = &sl_ch_estimates[p*ue->frame_parms.nb_antennas_rx+aarx][ch_offset];

    for(int idxP=0; idxP<ceil((float)nb_rb_pusch*12/8); idxP++) {
      for(int idxI=0; idxI<8; idxI++) {
        printf("%d\t%d\t",ul_ch[idxP*8+idxI].r,ul_ch[idxP*8+idxI].i);
      }

      printf("%d\n",idxP);
    }

#endif
    // Convert to time domain
    freq2time(symbolSize,
              (int16_t *) &sl_ch_estimates[aarx][symbol_offset],
              (int16_t *) ue->pssch_vars[ul_id]->sl_ch_estimates_time[aarx]);
  }

#ifdef DEBUG_CH
  fclose(debug_ch_est);
#endif
  return(0);
}

void nr_pdcch_channel_estimation(PHY_VARS_NR_UE *ue,
                                 UE_nr_rxtx_proc_t *proc,
                                 uint8_t gNB_id,
                                 unsigned char Ns,
                                 unsigned char symbol,
                                 fapi_nr_coreset_t *coreset,
                                 uint16_t first_carrier_offset,
                                 uint16_t BWPStart,
                                 int32_t pdcch_est_size,
                                 int32_t pdcch_dl_ch_estimates[][pdcch_est_size])
{

  unsigned char aarx;
  unsigned short k;
  unsigned int pilot_cnt;
  int16_t ch[2],*pil,*rxF,*dl_ch;
  int ch_offset,symbol_offset;

  int **rxdataF=ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;

  ch_offset     = ue->frame_parms.ofdm_symbol_size*symbol;

  symbol_offset = ue->frame_parms.ofdm_symbol_size*symbol;

  int nb_rb_coreset=0;
  int coreset_start_rb=0;
  get_coreset_rballoc(coreset->frequency_domain_resource,&nb_rb_coreset,&coreset_start_rb);
  if(nb_rb_coreset==0) return;

#ifdef DEBUG_PDCCH
  printf(PHY, "pdcch_channel_estimation: first_carrier_offset %d, BWPStart %d, coreset_start_rb %d, coreset_nb_rb %d\n",
         first_carrier_offset, BWPStart, coreset_start_rb, nb_rb_coreset);
#endif

  unsigned short coreset_start_subcarrier = first_carrier_offset+(BWPStart + coreset_start_rb)*12;

#ifdef DEBUG_PDCCH
  printf("PDCCH Channel Estimation : ThreadId %d, gNB_id %d ch_offset %d, OFDM size %d, Ncp=%d, Ns=%d, symbol %d\n",
         proc->thread_id, gNB_id,ch_offset,ue->frame_parms.ofdm_symbol_size,ue->frame_parms.Ncp,Ns,symbol);
#endif

#if CH_INTERP
  int16_t *fl = filt16a_l1;
  int16_t *fm = filt16a_m1;
  int16_t *fr = filt16a_r1;
#endif

  unsigned short scrambling_id = coreset->pdcch_dmrs_scrambling_id;
  // checking if re-initialization of scrambling IDs is needed (should be done here but scrambling ID for PDCCH is not taken from RRC)
  if (scrambling_id != ue->scramblingID_pdcch){
    ue->scramblingID_pdcch = scrambling_id;
    nr_gold_pdcch(ue,ue->scramblingID_pdcch);
  }

  int dmrs_ref = 0;
  if (coreset->CoreSetType == NFAPI_NR_CSET_CONFIG_PDCCH_CONFIG)
    dmrs_ref = BWPStart;

  // generate pilot
  int pilot[(nb_rb_coreset + dmrs_ref) * 3] __attribute__((aligned(16)));
  nr_pdcch_dmrs_rx(ue,Ns,ue->nr_gold_pdcch[gNB_id][Ns][symbol], &pilot[0],2000,(nb_rb_coreset+dmrs_ref));

  for (aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++) {

    k = coreset_start_subcarrier;
    pil   = (int16_t *)&pilot[dmrs_ref*3];
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+k+1)];
    dl_ch = (int16_t *)&pdcch_dl_ch_estimates[aarx][ch_offset];

    memset(dl_ch,0,4*(ue->frame_parms.ofdm_symbol_size));

#ifdef DEBUG_PDCCH
    printf("pdcch ch est pilot addr %p RB_DL %d\n",&pilot[dmrs_ref*3], ue->frame_parms.N_RB_DL);
    printf("k %d, first_carrier %d\n",k,ue->frame_parms.first_carrier_offset);
    printf("rxF addr %p\n", rxF);

    printf("dl_ch addr %p\n",dl_ch);
#endif
  #if CH_INTERP
    //    if ((ue->frame_parms.N_RB_DL&1)==0) {
    // Treat first 2 pilots specially (left edge)
    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDCCH
    printf("ch 0 %d\n",((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1]));
    printf("pilot 0 : rxF - > (%d,%d) addr %p  ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],&rxF[0],ch[0],ch[1],pil[0],pil[1]);
#endif
    multadd_real_vector_complex_scalar(fl,
				       ch,
				       dl_ch,
				       16);
    pil += 2;
    rxF += 8;
    k   += 4;

    if (k >= ue->frame_parms.ofdm_symbol_size) {
      k  -= ue->frame_parms.ofdm_symbol_size;
      rxF = (int16_t *)&rxdataF[aarx][(symbol_offset+k+1)];
    }

    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDCCH
    printf("pilot 1 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
    multadd_real_vector_complex_scalar(fm,
				       ch,
				       dl_ch,
				       16);
    pil += 2;
    rxF += 8;
    k   += 4;

    if (k >= ue->frame_parms.ofdm_symbol_size) {
      k  -= ue->frame_parms.ofdm_symbol_size;
      rxF = (int16_t *)&rxdataF[aarx][(symbol_offset+k+1)];
    }

    ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
    ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_PDCCH
    printf("pilot 2 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

    multadd_real_vector_complex_scalar(fr,
				       ch,
				       dl_ch,
				       16);
#ifdef DEBUG_PDCCH
    for (int m =0; m<12; m++)
      printf("data :  dl_ch -> (%d,%d)\n",dl_ch[0+2*m],dl_ch[1+2*m]);
#endif
    dl_ch += 24;

    pil += 2;
    rxF += 8;
    k   += 4;

    for (pilot_cnt=3; pilot_cnt<(3*nb_rb_coreset); pilot_cnt += 3) {

      if (k >= ue->frame_parms.ofdm_symbol_size) {
        k  -= ue->frame_parms.ofdm_symbol_size;
        rxF = (int16_t *)&rxdataF[aarx][(symbol_offset+k+1)];
      }

      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDCCH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fl,
					 ch,
					 dl_ch,
					 16);

      //for (int i= 0; i<8; i++)
      //            printf("pilot_cnt %d dl_ch %d %d\n", pilot_cnt, dl_ch+i, *(dl_ch+i));

      pil += 2;
      rxF += 8;
      k   += 4;

      if (k >= ue->frame_parms.ofdm_symbol_size) {
        k  -= ue->frame_parms.ofdm_symbol_size;
        rxF = (int16_t *)&rxdataF[aarx][(symbol_offset+k+1)];
      }

      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDCCH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+1,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fm,
					 ch,
					 dl_ch,
					 16);
      pil += 2;
      rxF += 8;
      k   += 4;

      if (k >= ue->frame_parms.ofdm_symbol_size) {
        k  -= ue->frame_parms.ofdm_symbol_size;
        rxF = (int16_t *)&rxdataF[aarx][(symbol_offset+k+1)];
      }

      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_PDCCH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+2,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif

      multadd_real_vector_complex_scalar(fr,
					 ch,
					 dl_ch,
					 16);
#ifdef DEBUG_PDCCH
    for (int m =0; m<12; m++)
      printf("data :  dl_ch -> (%d,%d)\n",dl_ch[0+2*m],dl_ch[1+2*m]);
#endif
      dl_ch += 24;

      pil += 2;
      rxF += 8;
      k   += 4;
    }
  #else //ELSE CH_INTERP
    int ch_sum[2] = {0, 0};

    for (pilot_cnt = 0; pilot_cnt < 3*nb_rb_coreset; pilot_cnt++) {
      if (k >= ue->frame_parms.ofdm_symbol_size) {
        k  -= ue->frame_parms.ofdm_symbol_size;
        rxF = (int16_t *)&rxdataF[aarx][(symbol_offset+k+1)];
      }
#ifdef DEBUG_PDCCH
      printf("pilot[%d] = (%d, %d)\trxF[%d] = (%d, %d)\n", pilot_cnt, pil[0], pil[1], k+1, rxF[0], rxF[1]);
#endif
      ch_sum[0] += (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch_sum[1] += (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
      pil += 2;
      rxF += 8;
      k   += 4;

      if (pilot_cnt % 3 == 2) {
        ch[0] = ch_sum[0] / 3;
        ch[1] = ch_sum[1] / 3;
        multadd_real_vector_complex_scalar(filt16a_1, ch, dl_ch, 16);
        dl_ch += 24;
        ch_sum[0] = 0;
        ch_sum[1] = 0;
      }
    }
  #endif //END CH_INTERP


    //}

  }
}

int nr_pdsch_channel_estimation(PHY_VARS_NR_UE *ue,
                                UE_nr_rxtx_proc_t *proc,
                                uint8_t gNB_id,
                                bool is_SI,
                                unsigned char Ns,
                                unsigned short p,
                                unsigned char symbol,
                                unsigned char nscid,
                                unsigned short scrambling_id,
                                unsigned short BWPStart,
                                uint8_t config_type,
                                unsigned short bwp_start_subcarrier,
                                unsigned short nb_rb_pdsch)
{
  int pilot[3280] __attribute__((aligned(16)));
  unsigned char aarx;
  unsigned short k;
  unsigned int pilot_cnt;
  int16_t ch_l[2],ch_r[2],ch[2],*pil,*rxF,*dl_ch;
  int16_t *fl=NULL,*fm=NULL,*fr=NULL,*fml=NULL,*fmr=NULL,*fmm=NULL,*fdcl=NULL,*fdcr=NULL,*fdclh=NULL,*fdcrh=NULL, *frl=NULL, *frr=NULL;
  int ch_offset,symbol_offset;

  uint8_t nushift;
  int **dl_ch_estimates = ue->pdsch_vars[proc->thread_id][gNB_id]->dl_ch_estimates;
  int **rxdataF=ue->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;

  ch_offset     = ue->frame_parms.ofdm_symbol_size*symbol;

  symbol_offset = ue->frame_parms.ofdm_symbol_size*symbol;

  k = bwp_start_subcarrier;
  int re_offset = k;

#ifdef DEBUG_PDSCH
  printf("PDSCH Channel Estimation : ThreadId %d, gNB_id %d ch_offset %d, symbol_offset %d OFDM size %d, Ncp=%d, Ns=%d, k=%d symbol %d\n",proc->thread_id, gNB_id,ch_offset,symbol_offset,ue->frame_parms.ofdm_symbol_size,
         ue->frame_parms.Ncp,Ns,k, symbol);
#endif

  // generate pilot for gNB port number 1000+p
  uint16_t rb_offset = (bwp_start_subcarrier - ue->frame_parms.first_carrier_offset) / 12;
  if (is_SI) {
    rb_offset -= BWPStart;
  }
  int8_t delta = get_delta(p, config_type);

  // checking if re-initialization of scrambling IDs is needed
  if (scrambling_id != ue->scramblingID_dlsch[nscid]){
    ue->scramblingID_dlsch[nscid] = scrambling_id;
    nr_gold_pdsch(ue, nscid, scrambling_id);
  }

  nr_pdsch_dmrs_rx(ue, Ns, ue->nr_gold_pdsch[gNB_id][Ns][symbol][0], &pilot[0], 1000+p, 0, nb_rb_pdsch+rb_offset, config_type);

  if (config_type == NFAPI_NR_DMRS_TYPE1){
    nushift = (p>>1)&1;
    if (p<4) ue->frame_parms.nushift = nushift;
    switch (delta) {

    case 0://port 0,1
      fl = filt8_l0;//left interpolation Filter for DMRS config. 1
      fm = filt8_m0;//left middle interpolation Filter
      fr = filt8_r0;//right interpolation Filter
      fmm = filt8_mm0;;//middle middle interpolation Filter
      fml = filt8_m0;//left middle interpolation Filter
      fmr = filt8_mr0;//middle right interpolation Filter
      fdcl = filt8_dcl0;//left DC interpolation Filter (even RB)
      fdcr = filt8_dcr0;//right DC interpolation Filter (even RB)
      fdclh = filt8_dcl0_h;//left DC interpolation Filter (odd RB)
      fdcrh = filt8_dcr0_h;//right DC interpolation Filter (odd RB)
      frl = NULL;
      frr = NULL;
      break;

    case 1://port2,3
      fl = filt8_l1;
      fm = filt8_m1;
      fr = filt8_r1;
      fmm = filt8_mm1;
      fml = filt8_ml1;
      fmr = filt8_m1;
      fdcl = filt8_dcl1;
      fdcr = filt8_dcr1;
      fdclh = filt8_dcl1_h;
      fdcrh = filt8_dcr1_h;
      frl = NULL;
      frr = NULL;
      break;

    default:
      LOG_E(PHY,"pdsch_channel_estimation: nushift=%d -> ERROR\n",nushift);
      return -1;
      break;
    }
  } else {//NFAPI_NR_DMRS_TYPE2
    nushift = delta;
    if (p<6) ue->frame_parms.nushift = nushift;
    switch (delta) {
    case 0://port 0,1
      fl = filt8_l2;//left interpolation Filter should be fml
      fr = filt8_r2;//right interpolation Filter should be fmr
      fm = filt8_l2;
      fmm = filt8_r2;
      fml = filt8_ml2;
      fmr = filt8_mr2;
      frl = filt8_rl2;
      frr = filt8_rm2;
      fdcl = filt8_dcl1;
      fdcr = filt8_dcr1;
      fdclh = filt8_dcl1_h;
      fdcrh = filt8_dcr1_h;
      break;

    case 2://port2,3
      fl = filt8_l3;
      fm = filt8_m2;
      fr = filt8_r3;
      fmm = filt8_mm2;
      fml = filt8_l2;
      fmr = filt8_r2;
      frl = filt8_rl3;
      frr = filt8_rr3;
      fdcl = NULL;
      fdcr = NULL;
      fdclh = NULL;
      fdcrh = NULL;
      break;

    default:
      LOG_E(PHY,"pdsch_channel_estimation: nushift=%d -> ERROR\n",nushift);
      return -1;
      break;
    }
  }

  for (aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++) {
    pil   = (int16_t *)&pilot[rb_offset*((config_type == NFAPI_NR_DMRS_TYPE1) ? 6:4)];
    k     = k % ue->frame_parms.ofdm_symbol_size;
    re_offset = k;
    rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+re_offset+nushift)];
    dl_ch = (int16_t *)&dl_ch_estimates[p*ue->frame_parms.nb_antennas_rx+aarx][ch_offset];

    memset(dl_ch,0,4*(ue->frame_parms.ofdm_symbol_size));
#ifdef DEBUG_PDSCH
    printf("ch est pilot addr %p RB_DL %d\n",&pilot[0], ue->frame_parms.N_RB_DL);
    printf("k %d, first_carrier %d\n",k,ue->frame_parms.first_carrier_offset);
    printf("rxF addr %p p %d\n", rxF,p);
    printf("dl_ch addr %p nushift %d\n",dl_ch,nushift);
#endif
    if (config_type == NFAPI_NR_DMRS_TYPE1 && ue->chest_freq == 0) {

      // Treat first 2 pilots specially (left edge)
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDSCH
      printf("ch 0 %d\n",((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1]));
      printf("pilot 0 : rxF - > (%d,%d) addr %p  ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],&rxF[0],ch[0],ch[1],pil[0],pil[1]);
      printf("data 0 : rxF - > (%d,%d) addr %p  ch -> (%d,%d), pil -> (%d,%d) \n",rxF[2],rxF[3],&rxF[2],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fl,
                                         ch,
                                         dl_ch,
                                         8);
      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
      //for (int i= 0; i<8; i++)
      //printf("dl_ch addr %p %d\n", dl_ch+i, *(dl_ch+i));

      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDSCH
      printf("pilot 1 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fml,
                                         ch,
                                         dl_ch,
                                         8);
      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
      //printf("dl_ch addr %p\n",dl_ch);
      
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDSCH
      printf("pilot 2 : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fmm,
                                         ch,
                                         dl_ch,
                                         8);
                                         
      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
      dl_ch += 8;

      for (pilot_cnt=3; pilot_cnt<(6*nb_rb_pdsch-3); pilot_cnt += 2) {

        ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
        ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDSCH
	printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
        multadd_real_vector_complex_scalar(fm,
                                           ch,
                                           dl_ch,
                                           8);

        pil += 2;
        re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
      
        ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
        ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDSCH
	printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+1,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
        multadd_real_vector_complex_scalar(fmm,
                                           ch,
                                           dl_ch,
                                           8);
        pil += 2;
        re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
        dl_ch += 8;

      }
      
      // Treat first 2 pilots specially (right edge)
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDSCH
      printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fm,
                                         ch,
                                         dl_ch,
                                         8);
                                         
      //for (int i= 0; i<8; i++)
      //printf("dl_ch addr %p %d\n", dl_ch+i, *(dl_ch+i));

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
             
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDSCH
      printf("ch 0 %d\n",((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1]));
      printf("pilot %u: rxF - > (%d,%d) addr %p  ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+1,rxF[0],rxF[1],&rxF[0],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fmr,
                                         ch,
                                         dl_ch,
                                         8);
                                         
      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
      dl_ch += 8;
      
      ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
#ifdef DEBUG_PDSCH
      printf("pilot %u: rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+2,rxF[0],rxF[1],ch[0],ch[1],pil[0],pil[1]);
#endif
      multadd_real_vector_complex_scalar(fr,
                                         ch,
                                         dl_ch,
                                         8);
    
      // check if PRB crosses DC and improve estimates around DC
      if ((bwp_start_subcarrier < ue->frame_parms.ofdm_symbol_size) && (bwp_start_subcarrier+nb_rb_pdsch*12 >= ue->frame_parms.ofdm_symbol_size)) {
        dl_ch = (int16_t *)&dl_ch_estimates[aarx][ch_offset];
        uint16_t idxDC = 2*(ue->frame_parms.ofdm_symbol_size - bwp_start_subcarrier);
        uint16_t idxPil = idxDC/2;
        re_offset = k;
        pil = (int16_t *)&pilot[rb_offset*((config_type == NFAPI_NR_DMRS_TYPE1) ? 6:4)];
        pil += (idxPil-2);
        dl_ch += (idxDC-4);
        dl_ch = memset(dl_ch, 0, sizeof(int16_t)*10);
        re_offset = (re_offset+idxDC/2-2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
        ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
        ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
          
        // for proper allignment of SIMD vectors
        if((ue->frame_parms.N_RB_DL&1) == 0) {
              
          multadd_real_vector_complex_scalar(fdcl,
                     ch,
                     dl_ch-4,
                     8);
              
          pil += 4;
          re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
          rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
          ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
          ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
              
          multadd_real_vector_complex_scalar(fdcr,
                     ch,
                     dl_ch-4,
                     8);
        } else {

          multadd_real_vector_complex_scalar(fdclh,
                     ch,
                     dl_ch,
                     8);
              
          pil += 4;
          re_offset = (re_offset+4) % ue->frame_parms.ofdm_symbol_size;
          rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
          ch[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
          ch[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);
              
          multadd_real_vector_complex_scalar(fdcrh,
                     ch,
                     dl_ch,
                     8);
        }
      }
    } else if (config_type == NFAPI_NR_DMRS_TYPE2 && ue->chest_freq == 0){ //pdsch_dmrs_type2  |dmrs_r,dmrs_l,0,0,0,0,dmrs_r,dmrs_l,0,0,0,0|

      // Treat first 4 pilots specially (left edge)
      ch_l[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch_l[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_PDSCH
      printf("ch 0 %d\n",((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1]));
      printf("pilot 0 : rxF - > (%d,%d) addr %p  ch -> (%d,%d), pil -> (%d,%d) \n",rxF[0],rxF[1],&rxF[0],ch_l[0],ch_l[1],pil[0],pil[1]);
#endif

      pil += 2;
      re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
      ch_r[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch_r[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

      ch[0] = (ch_l[0]+ch_r[0])>>1;
      ch[1] = (ch_l[1]+ch_r[1])>>1;

      dl_ch[(0+2*nushift)] = ch[0];
      dl_ch[(1+2*nushift)] = ch[1];
      dl_ch[2+2*nushift] = ch[0];
      dl_ch[3+2*nushift] = ch[1];

      multadd_real_vector_complex_scalar(fl,
                                         ch,
                                         dl_ch,
                                         8);

      pil += 2;
      re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
      ch_l[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch_l[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

      pil += 2;
      re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
      ch_r[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
      ch_r[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

      ch[0] = (ch_l[0]+ch_r[0])>>1;
      ch[1] = (ch_l[1]+ch_r[1])>>1;

      multadd_real_vector_complex_scalar(fr,
                                         ch,
                                         dl_ch,
                                         8);

      dl_ch += 12;
      dl_ch[0+2*nushift] = ch[0];
      dl_ch[1+2*nushift] = ch[1];
      dl_ch[2+2*nushift] = ch[0];
      dl_ch[3+2*nushift] = ch[1];
      dl_ch += 4;

      for (pilot_cnt=4; pilot_cnt<4*nb_rb_pdsch; pilot_cnt += 4) {

        multadd_real_vector_complex_scalar(fml,
                                           ch,
                                           dl_ch,
                                           8);
        pil += 2;
        re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
        ch_l[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
        ch_l[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_PDSCH
        printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt,rxF[0],rxF[1],ch_l[0],ch_l[1],pil[0],pil[1]);
#endif

        pil += 2;
        re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
        ch_r[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
        ch_r[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

        ch[0] = (ch_l[0]+ch_r[0])>>1;
        ch[1] = (ch_l[1]+ch_r[1])>>1;

#ifdef DEBUG_PDSCH
        printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+1,rxF[0],rxF[1],ch_r[0],ch_r[1],pil[0],pil[1]);
#endif

        multadd_real_vector_complex_scalar(fmr,
                                           ch,
                                           dl_ch,
                                           8);

        dl_ch += 8;
        dl_ch[0+2*nushift] = ch[0];
        dl_ch[1+2*nushift] = ch[1];
        dl_ch[2+2*nushift] = ch[0];
        dl_ch[3+2*nushift] = ch[1];

        multadd_real_vector_complex_scalar(fm,
                                           ch,
                                           dl_ch,
                                           8);

        pil += 2;
        re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
        ch_l[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
        ch_l[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

        pil += 2;
        re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];
        ch_r[0] = (int16_t)(((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15);
        ch_r[1] = (int16_t)(((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15);

#ifdef DEBUG_PDSCH
        printf("pilot %u : rxF - > (%d,%d) ch -> (%d,%d), pil -> (%d,%d) \n",pilot_cnt+1,rxF[0],rxF[1],ch_r[0],ch_r[1],pil[0],pil[1]);
#endif

        ch[0] = (ch_l[0]+ch_r[0])>>1;
        ch[1] = (ch_l[1]+ch_r[1])>>1;

        multadd_real_vector_complex_scalar(fmm,
                                           ch,
                                           dl_ch,
                                           8);

        dl_ch += 12;
        dl_ch[0+2*nushift] = ch[0];
        dl_ch[1+2*nushift] = ch[1];
        dl_ch[2+2*nushift] = ch[0];
        dl_ch[3+2*nushift] = ch[1];
        dl_ch += 4;
      }

      // Treat last 2 pilots specially (right edge)
      // dl_ch-2+nushift<<1
      multadd_real_vector_complex_scalar(frl,
                                         dl_ch-2+2*nushift,
                                         dl_ch,
                                         8);

      multadd_real_vector_complex_scalar(frr,
                                         dl_ch-14+2*nushift,/*14*/
                                         dl_ch,
                                         8);

      // check if PRB crosses DC and improve estimates around DC
      if ((bwp_start_subcarrier < ue->frame_parms.ofdm_symbol_size) && (bwp_start_subcarrier+nb_rb_pdsch*12 >= ue->frame_parms.ofdm_symbol_size) && (p<2)) {

        dl_ch = (int16_t *)&dl_ch_estimates[p*ue->frame_parms.nb_antennas_rx+aarx][ch_offset];
        uint16_t idxDC = 2*(ue->frame_parms.ofdm_symbol_size - bwp_start_subcarrier);
        dl_ch += (idxDC-8);
        dl_ch = memset(dl_ch, 0, sizeof(int16_t)*20);

        dl_ch -= 2;

        ch_r[0] = dl_ch[0];
        ch_r[1]= dl_ch[1] ;
        dl_ch += 22;
        ch_l[0] = dl_ch[0];
        ch_l[1]= dl_ch[1] ;

        // for proper allignment of SIMD vectors
        if((ue->frame_parms.N_RB_DL&1) == 0) {
          dl_ch -= 20;
          //Interpolate fdcrl1 with ch_r
          multadd_real_vector_complex_scalar(filt8_dcrl1,
                                             ch_r,
                                             dl_ch,
                                             8);
          //Interpolate fdclh1 with ch_l
          multadd_real_vector_complex_scalar(filt8_dclh1,
                                             ch_l,
                                             dl_ch,
                                             8);
          dl_ch += 16;
          //Interpolate fdcrh1 with ch_r
          multadd_real_vector_complex_scalar(filt8_dcrh1,
                                             ch_r,
                                             dl_ch,
                                             8);
          //Interpolate fdcll1 with ch_l
          multadd_real_vector_complex_scalar(filt8_dcll1,
                                             ch_l,
                                             dl_ch,
                                             8);
        } else {
          dl_ch -= 28;
          //Interpolate fdcrl1 with ch_r
          multadd_real_vector_complex_scalar(filt8_dcrl2,
                                             ch_r,
                                             dl_ch,
                                             8);
          //Interpolate fdclh1 with ch_l
          multadd_real_vector_complex_scalar(filt8_dclh2,
                                             ch_l,
                                             dl_ch,
                                             8);
          dl_ch += 16;
          //Interpolate fdcrh1 with ch_r
          multadd_real_vector_complex_scalar(filt8_dcrh2,
                                             ch_r,
                                             dl_ch,
                                             8);
          //Interpolate fdcll1 with ch_l
          multadd_real_vector_complex_scalar(filt8_dcll2,
                                             ch_l,
                                             dl_ch,
                                             8);
        }
      }
    }
    else if (config_type == NFAPI_NR_DMRS_TYPE1) { // this is case without frequency-domain linear interpolation, just take average of LS channel estimates of 6 DMRS REs and use a common value for the whole PRB
      int32_t ch_0, ch_1;
      
      ch_0 = ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 = ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch[0] = ch_0 / 6;
      ch[1] = ch_1 / 6;

#if NO_INTERP
      for (int i=0;i<12;i++) ((int32_t*)dl_ch)[i] = *(int32_t*)ch;
      dl_ch+=24;
#else
      multadd_real_vector_complex_scalar(filt8_avlip0,
                                         ch,
                                         dl_ch,
                                         8);

      dl_ch += 16;
      multadd_real_vector_complex_scalar(filt8_avlip1,
                                         ch,
                                         dl_ch,
                                         8);

      dl_ch += 16;
      multadd_real_vector_complex_scalar(filt8_avlip2,
                                         ch,
                                         dl_ch,
                                         8);
      dl_ch -= 24;
#endif

      for (pilot_cnt=6; pilot_cnt<6*(nb_rb_pdsch-1); pilot_cnt += 6) {

        ch_0 = ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 = ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch[0] = ch_0 / 6;
        ch[1] = ch_1 / 6;

#if NO_INTERP
        for (int i=0;i<12;i++) ((int32_t*)dl_ch)[i] = *(int32_t*)ch;
        dl_ch+=24;
#else
        dl_ch[6] += (ch[0] * 1365)>>15; // 1/12*16384
        dl_ch[7] += (ch[1] * 1365)>>15; // 1/12*16384

        dl_ch += 8;
        multadd_real_vector_complex_scalar(filt8_avlip3,
                                           ch,
                                           dl_ch,
                                           8);

        dl_ch += 16;
        multadd_real_vector_complex_scalar(filt8_avlip4,
                                           ch,
                                           dl_ch,
                                           8);

        dl_ch += 16;
        multadd_real_vector_complex_scalar(filt8_avlip5,
                                           ch,
                                           dl_ch,
                                           8);
        dl_ch -= 16;
#endif
      }
      ch_0 = ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 = ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+2) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch[0] = ch_0 / 6;
      ch[1] = ch_1 / 6;

#if NO_INTERP
      for (int i=0;i<12;i++) ((int32_t*)dl_ch)[i] = *(int32_t*)ch;
      dl_ch+=24;
#else
      dl_ch[6] += (ch[0] * 1365)>>15; // 1/12*16384
      dl_ch[7] += (ch[1] * 1365)>>15; // 1/12*16384

      dl_ch += 8;
      multadd_real_vector_complex_scalar(filt8_avlip3,
                                         ch,
                                         dl_ch,
                                         8);

      dl_ch += 16;
      multadd_real_vector_complex_scalar(filt8_avlip6,
                                         ch,
                                         dl_ch,
                                         8);
#endif
    }
    else  { // this is case without frequency-domain linear interpolation, just take average of LS channel estimates of 4 DMRS REs and use a common value for the whole PRB
      int32_t ch_0, ch_1;

      ch_0 = ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 = ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch[0] = ch_0 / 4;
      ch[1] = ch_1 / 4;

#if NO_INTERP
      for (int i=0;i<12;i++) ((int32_t*)dl_ch)[i] = *(int32_t*)ch;
      dl_ch+=24;
#else
      multadd_real_vector_complex_scalar(filt8_avlip0,
                                         ch,
                                         dl_ch,
                                         8);

      dl_ch += 16;
      multadd_real_vector_complex_scalar(filt8_avlip1,
                                         ch,
                                         dl_ch,
                                         8);

      dl_ch += 16;
      multadd_real_vector_complex_scalar(filt8_avlip2,
                                         ch,
                                         dl_ch,
                                         8);
      dl_ch -= 24;
#endif

      for (pilot_cnt=4; pilot_cnt<4*(nb_rb_pdsch-1); pilot_cnt += 4) {
        int32_t ch_0, ch_1;

        ch_0 = ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 = ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
        ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

        pil += 2;
        re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
        rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

        ch[0] = ch_0 / 4;
        ch[1] = ch_1 / 4;

#if NO_INTERP
        for (int i=0;i<12;i++) ((int32_t*)dl_ch)[i] = *(int32_t*)ch;
        dl_ch+=24;
#else
        dl_ch[6] += (ch[0] * 1365)>>15; // 1/12*16384
        dl_ch[7] += (ch[1] * 1365)>>15; // 1/12*16384

        dl_ch += 8;
        multadd_real_vector_complex_scalar(filt8_avlip3,
                                           ch,
                                           dl_ch,
                                           8);

        dl_ch += 16;
        multadd_real_vector_complex_scalar(filt8_avlip4,
                                           ch,
                                           dl_ch,
                                           8);

        dl_ch += 16;
        multadd_real_vector_complex_scalar(filt8_avlip5,
                                           ch,
                                           dl_ch,
                                           8);
        dl_ch -= 16;
#endif
      }

      ch_0 = ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 = ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+1) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch_0 += ((int32_t)pil[0]*rxF[0] - (int32_t)pil[1]*rxF[1])>>15;
      ch_1 += ((int32_t)pil[0]*rxF[1] + (int32_t)pil[1]*rxF[0])>>15;

      pil += 2;
      re_offset = (re_offset+5) % ue->frame_parms.ofdm_symbol_size;
      rxF   = (int16_t *)&rxdataF[aarx][(symbol_offset+nushift+re_offset)];

      ch[0] = ch_0 / 4;
      ch[1] = ch_1 / 4;

#if NO_INTERP
      for (int i=0;i<12;i++) ((int32_t*)dl_ch)[i] = *(int32_t*)ch;
      dl_ch+=24;
#else
      dl_ch[6] += (ch[0] * 1365)>>15; // 1/12*16384
      dl_ch[7] += (ch[1] * 1365)>>15; // 1/12*16384

      dl_ch += 8;
      multadd_real_vector_complex_scalar(filt8_avlip3,
                                         ch,
                                         dl_ch,
                                         8);

      dl_ch += 16;
      multadd_real_vector_complex_scalar(filt8_avlip6,
                                         ch,
                                         dl_ch,
                                         8);
#endif
    }
#ifdef DEBUG_PDSCH
    dl_ch = (int16_t *)&dl_ch_estimates[p*ue->frame_parms.nb_antennas_rx+aarx][ch_offset];
    for(uint16_t idxP=0; idxP<ceil((float)nb_rb_pdsch*12/8); idxP++) {
      for(uint8_t idxI=0; idxI<16; idxI += 2) {
        printf("%d\t%d\t",dl_ch[idxP*16+idxI],dl_ch[idxP*16+idxI+1]);
      }
      printf("%d\n",idxP);
    }
#endif
  }
  return(0);
}

/*******************************************************************
 *
 * NAME :         nr_pdsch_ptrs_processing
 *
 * PARAMETERS :   PHY_VARS_NR_UE    : ue data structure
 *                NR_UE_PDSCH       : pdsch_vars pointer
 *                NR_DL_FRAME_PARMS : frame_parms pointer
 *                NR_DL_UE_HARQ_t   : dlsch0_harq pointer
 *                NR_DL_UE_HARQ_t   : dlsch1_harq pointer
 *                uint8_t           : gNB_id,
 *                uint8_t           : nr_slot_rx,
 *                unsigned char     : symbol,
 *                uint32_t          : nb_re_pdsch,
 *                uint16_t          : rnti
 *                RX_type_t         : rx_type
 * RETURN : Nothing
 *
 * DESCRIPTION :
 *  If ptrs is enabled process the symbol accordingly
 *  1) Estimate common phase error per PTRS symbol
 *  2) Interpolate PTRS estimated value in TD after all PTRS symbols
 *  3) Compensate signal with PTRS estimation for slot
 *********************************************************************/
void nr_pdsch_ptrs_processing(PHY_VARS_NR_UE *ue,
                              NR_UE_PDSCH **pdsch_vars,
                              NR_DL_FRAME_PARMS *frame_parms,
                              NR_DL_UE_HARQ_t *dlsch0_harq,
                              NR_DL_UE_HARQ_t *dlsch1_harq,
                              uint8_t gNB_id,
                              uint8_t nr_slot_rx,
                              unsigned char symbol,
                              uint32_t nb_re_pdsch,
                              uint16_t rnti,
                              RX_type_t rx_type)
{
  //#define DEBUG_DL_PTRS 1
  int32_t *ptrs_re_symbol = NULL;
  int8_t   ret = 0;
  /* harq specific variables */
  uint8_t  symbInSlot       = 0;
  uint16_t *startSymbIndex  = NULL;
  uint16_t *nbSymb          = NULL;
  uint8_t  *L_ptrs          = NULL;
  uint8_t  *K_ptrs          = NULL;
  uint16_t *dmrsSymbPos     = NULL;
  uint16_t *ptrsSymbPos     = NULL;
  uint8_t  *ptrsSymbIdx     = NULL;
  uint8_t  *ptrsReOffset    = NULL;
  uint8_t  *dmrsConfigType  = NULL;
  uint16_t *nb_rb           = NULL;

  if(dlsch0_harq->status == ACTIVE) {
    symbInSlot      = dlsch0_harq->start_symbol + dlsch0_harq->nb_symbols;
    startSymbIndex  = &dlsch0_harq->start_symbol;
    nbSymb          = &dlsch0_harq->nb_symbols;
    L_ptrs          = &dlsch0_harq->PTRSTimeDensity;
    K_ptrs          = &dlsch0_harq->PTRSFreqDensity;
    dmrsSymbPos     = &dlsch0_harq->dlDmrsSymbPos;
    ptrsSymbPos     = &dlsch0_harq->ptrs_symbols;
    ptrsSymbIdx     = &dlsch0_harq->ptrs_symbol_index;
    ptrsReOffset    = &dlsch0_harq->PTRSReOffset;
    dmrsConfigType  = &dlsch0_harq->dmrsConfigType;
    nb_rb           = &dlsch0_harq->nb_rb;
  }
  if(dlsch1_harq) {
    symbInSlot      = dlsch1_harq->start_symbol + dlsch0_harq->nb_symbols;
    startSymbIndex  = &dlsch1_harq->start_symbol;
    nbSymb          = &dlsch1_harq->nb_symbols;
    L_ptrs          = &dlsch1_harq->PTRSTimeDensity;
    K_ptrs          = &dlsch1_harq->PTRSFreqDensity;
    dmrsSymbPos     = &dlsch1_harq->dlDmrsSymbPos;
    ptrsSymbPos     = &dlsch1_harq->ptrs_symbols;
    ptrsSymbIdx     = &dlsch1_harq->ptrs_symbol_index;
    ptrsReOffset    = &dlsch1_harq->PTRSReOffset;
    dmrsConfigType  = &dlsch1_harq->dmrsConfigType;
    nb_rb           = &dlsch1_harq->nb_rb;
  }
  /* loop over antennas */
  for (int aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
    c16_t *phase_per_symbol = (c16_t*)pdsch_vars[gNB_id]->ptrs_phase_per_slot[aarx];
    ptrs_re_symbol = (int32_t*)pdsch_vars[gNB_id]->ptrs_re_per_slot[aarx];
    ptrs_re_symbol[symbol] = 0;
    phase_per_symbol[symbol].i = 0; // Imag
    /* set DMRS estimates to 0 angle with magnitude 1 */
    if(is_dmrs_symbol(symbol,*dmrsSymbPos)) {
      /* set DMRS real estimation to 32767 */
      phase_per_symbol[symbol].r=INT16_MAX; // 32767
#ifdef DEBUG_DL_PTRS
      printf("[PHY][PTRS]: DMRS Symbol %d -> %4d + j*%4d\n", symbol, phase_per_symbol[symbol].r,phase_per_symbol[symbol].i);
#endif
    }
    else { // real ptrs value is set to 0
      phase_per_symbol[symbol].r = 0; // Real
    }

    if(dlsch0_harq->status == ACTIVE) {
      if(symbol == *startSymbIndex) {
        *ptrsSymbPos = 0;
        set_ptrs_symb_idx(ptrsSymbPos,
                          *nbSymb,
                          *startSymbIndex,
                          1<< *L_ptrs,
                          *dmrsSymbPos);
      }
      /* if not PTRS symbol set current ptrs symbol index to zero*/
      *ptrsSymbIdx = 0;
      /* Check if current symbol contains PTRS */
      if(is_ptrs_symbol(symbol, *ptrsSymbPos)) {
        *ptrsSymbIdx = symbol;
        /*------------------------------------------------------------------------------------------------------- */
        /* 1) Estimate common phase error per PTRS symbol                                                                */
        /*------------------------------------------------------------------------------------------------------- */
        nr_ptrs_cpe_estimation(*K_ptrs,*ptrsReOffset,*dmrsConfigType,*nb_rb,
                               rnti,
                               nr_slot_rx,
                               symbol,frame_parms->ofdm_symbol_size,
                               (int16_t*)&pdsch_vars[gNB_id]->rxdataF_comp0[aarx][(symbol * nb_re_pdsch)],
                               ue->nr_gold_pdsch[gNB_id][nr_slot_rx][symbol][0],
                               (int16_t*)&phase_per_symbol[symbol],
                               &ptrs_re_symbol[symbol]);
      }
    }// HARQ 0

    /* For last OFDM symbol at each antenna perform interpolation and compensation for the slot*/
    if(symbol == (symbInSlot -1)) {
      /*------------------------------------------------------------------------------------------------------- */
      /* 2) Interpolate PTRS estimated value in TD */
      /*------------------------------------------------------------------------------------------------------- */
      /* If L-PTRS is > 0 then we need interpolation */
      if(*L_ptrs > 0) {
        ret = nr_ptrs_process_slot(*dmrsSymbPos, *ptrsSymbPos, (int16_t*)phase_per_symbol, *startSymbIndex, *nbSymb);
        if(ret != 0) {
          LOG_W(PHY,"[PTRS] Compensation is skipped due to error in PTRS slot processing !!\n");
        }
      }
#ifdef DEBUG_DL_PTRS
      LOG_M("ptrsEst.m","est",pdsch_vars[gNB_id]->ptrs_phase_per_slot[aarx],frame_parms->symbols_per_slot,1,1 );
      LOG_M("rxdataF_bf_ptrs_comp.m","bf_ptrs_cmp",
            &pdsch_vars[gNB_id]->rxdataF_comp0[aarx][(*startSymbIndex) * NR_NB_SC_PER_RB * (*nb_rb) ],
            (*nb_rb) * NR_NB_SC_PER_RB * (*nbSymb),1,1);
#endif
      /*------------------------------------------------------------------------------------------------------- */
      /* 3) Compensated DMRS based estimated signal with PTRS estimation                                        */
      /*--------------------------------------------------------------------------------------------------------*/
      for(uint8_t i = *startSymbIndex; i< symbInSlot ;i++) {
        /* DMRS Symbol has 0 phase so no need to rotate the respective symbol */
        /* Skip rotation if the slot processing is wrong */
        if((!is_dmrs_symbol(i,*dmrsSymbPos)) && (ret == 0)) {
#ifdef DEBUG_DL_PTRS
          printf("[PHY][DL][PTRS]: Rotate Symbol %2d with  %d + j* %d\n", i, phase_per_symbol[i].r,phase_per_symbol[i].i);
#endif
          rotate_cpx_vector((c16_t*)&pdsch_vars[gNB_id]->rxdataF_comp0[aarx][(i * (*nb_rb) * NR_NB_SC_PER_RB)],
                            &phase_per_symbol[i],
                            (c16_t*)&pdsch_vars[gNB_id]->rxdataF_comp0[aarx][(i * (*nb_rb) * NR_NB_SC_PER_RB)],
                            ((*nb_rb) * NR_NB_SC_PER_RB), 15);
        }// if not DMRS Symbol
      }// symbol loop
    }// last symbol check
  }//Antenna loop
}//main function
