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

/*! \file PHY/LTE_TRANSPORT/psbch.c
* \brief Top-level routines for generating and decoding  the PSBCH/BCH physical/transport channel V8.6 2009-03
* \author R. Knopp, F. Kaltenberger
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr,florian.kaltenberger.fr
* \note
* \warning
*/
#include "PHY/defs_nr_UE.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/phy_extern_nr_ue.h"
#include "PHY/sse_intrin.h"
#include "PHY/LTE_REFSIG/lte_refsig.h"
#include "PHY/INIT/phy_init.h"
#include "openair1/SCHED_NR_UE/defs.h"
#include <openair1/PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h>
#include <openair1/PHY/TOOLS/phy_scope_interface.h>

//#define DEBUG_PSBCH
//#define DEBUG_PSBCH_ENCODING

//#include "PHY_INTERFACE/defs.h"


#define PSBCH_A 32
#define PSBCH_MAX_RE_PER_SYMBOL (11*12)
#define PSBCH_MAX_RE (PSBCH_MAX_RE_PER_SYMBOL*14)
#define print_shorts(s,x) printf("%s : %d,%d,%d,%d,%d,%d,%d,%d\n",s,((int16_t*)x)[0],((int16_t*)x)[1],((int16_t*)x)[2],((int16_t*)x)[3],((int16_t*)x)[4],((int16_t*)x)[5],((int16_t*)x)[6],((int16_t*)x)[7])

static void nr_psbch_quantize(int16_t *psbch_llr8,
                             int16_t *psbch_llr,
                             uint16_t len) {
  for (int i = 0; i < len; i++) {
    if (psbch_llr[i] > 31)
      psbch_llr8[i] = 32;
    else if (psbch_llr[i] < -31)
      psbch_llr8[i] = -32;
    else
      psbch_llr8[i] = psbch_llr[i];
  }
}

static uint16_t nr_psbch_extract(int **rxdataF,
                                const int estimateSz,
                                struct complex16 dl_ch_estimates[][estimateSz],
                                struct complex16 rxdataF_ext[][PSBCH_MAX_RE_PER_SYMBOL],
                                struct complex16 dl_ch_estimates_ext[][PSBCH_MAX_RE_PER_SYMBOL],
                                uint32_t symbol,
                                uint32_t s_offset,
                                NR_DL_FRAME_PARMS *frame_parms) {
  uint16_t rb;
  uint8_t i,j,aarx;
  int nushiftmod4 = 0;//frame_parms->nushift;
  AssertFatal(symbol == 0 || symbol < 14,
              "symbol %d illegal for PSBCH extraction\n",
              symbol);

  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
    unsigned int rx_offset = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier;
    rx_offset = (rx_offset)%(frame_parms->ofdm_symbol_size);
    struct complex16 *rxF        = (struct complex16 *)&rxdataF[aarx][(symbol+s_offset)*frame_parms->ofdm_symbol_size];
    struct complex16 *rxF_ext    = rxdataF_ext[aarx];
#ifdef DEBUG_PSBCH
    printf("extract_rbs (nushift %d): rx_offset=%d, symbol %u\n",frame_parms->nushift,
           (rx_offset + ((symbol+s_offset)*(frame_parms->ofdm_symbol_size))),symbol);
    int16_t *p = (int16_t *)rxF;

    for (int i =0; i<8; i++) {
      printf("rxF [%d]= %d\n",i,rxF[i]);
      printf("psbch extract rxF  %d %d addr %p\n", p[2*i], p[2*i+1], &p[2*i]);
    }

#endif

    for (rb=0; rb<11; rb++) {
      j=0;

      if (symbol == 0 || ((symbol > 4) && (symbol <= 13))) {
        for (i=0; i<12; i++) {
          if ((i!=nushiftmod4) &&
              (i!=(nushiftmod4+4)) &&
              (i!=(nushiftmod4+8))) {
            rxF_ext[j]=rxF[rx_offset];
#ifdef DEBUG_PSBCH
            printf("rxF ext[%d] = (%d,%d) rxF [%u]= (%d,%d)\n",  (9*rb) + j,
                   rxF_ext[j].r, rxF_ext[j].i,
                   rx_offset,
                   rxF[rx_offset].r,rxF[rx_offset].i);
#endif
            j++;
          }

          rx_offset=(rx_offset+1)%(frame_parms->ofdm_symbol_size);
          //rx_offset = (rx_offset >= frame_parms->ofdm_symbol_size) ? (rx_offset - frame_parms->ofdm_symbol_size + 1) : (rx_offset+1);
        }

        rxF_ext+=9;
      }
    }

    struct complex16 *dl_ch0 = &dl_ch_estimates[aarx][((symbol+s_offset)*(frame_parms->ofdm_symbol_size))];

    //printf("dl_ch0 addr %p\n",dl_ch0);
    struct complex16 *dl_ch0_ext = dl_ch_estimates_ext[aarx];

    for (rb=0; rb<11; rb++) {
      j=0;

      if (symbol == 0 || ((symbol > 4) && (symbol <= 13))) {
        for (i=0; i<12; i++) {
          if ((i!=nushiftmod4) &&
              (i!=(nushiftmod4+4)) &&
              (i!=(nushiftmod4+8))) {
            dl_ch0_ext[j]=dl_ch0[i];
#ifdef DEBUG_PSBCH
            if ((rb==0) && (i<2))
              printf("dl ch0 ext[%d] = (%d,%d)  dl_ch0 [%d]= (%d,%d)\n",j,
                     dl_ch0_ext[j].r, dl_ch0_ext[j].i,
                     i,
                     dl_ch0[j].r, dl_ch0[j].i);
#endif
            j++;
          }
        }

        dl_ch0+=12;
        dl_ch0_ext+=9;
      }
    }
  }

  return(0);
}

//__m128i avg128;

//compute average channel_level on each (TX,RX) antenna pair
int nr_psbch_channel_level(struct complex16 dl_ch_estimates_ext[][PSBCH_MAX_RE_PER_SYMBOL],
                          NR_DL_FRAME_PARMS *frame_parms,
			  int nb_re) {
  int16_t nb_rb=nb_re/12;
#if defined(__x86_64__) || defined(__i386__)
  __m128i avg128;
  __m128i *dl_ch128;
#elif defined(__arm__)
  int32x4_t avg128;
  int16x8_t *dl_ch128;
#endif
  int avg1=0,avg2=0;

  for (int aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
    //clear average level
#if defined(__x86_64__) || defined(__i386__)
    avg128 = _mm_setzero_si128();
    dl_ch128=(__m128i *)dl_ch_estimates_ext[aarx];
#elif defined(__arm__)
    avg128 = vdupq_n_s32(0);
    dl_ch128=(int16x8_t *)dl_ch_estimates_ext[aarx];
#endif

    for (int rb=0; rb<nb_rb; rb++) {
#if defined(__x86_64__) || defined(__i386__)
      avg128 = _mm_add_epi32(avg128,_mm_madd_epi16(dl_ch128[0],dl_ch128[0]));
      avg128 = _mm_add_epi32(avg128,_mm_madd_epi16(dl_ch128[1],dl_ch128[1]));
      avg128 = _mm_add_epi32(avg128,_mm_madd_epi16(dl_ch128[2],dl_ch128[2]));
#elif defined(__arm__)
      abort();
      // to be filled in
#endif
      dl_ch128+=3;
      /*
      if (rb==0) {
      print_shorts("dl_ch128",&dl_ch128[0]);
      print_shorts("dl_ch128",&dl_ch128[1]);
      print_shorts("dl_ch128",&dl_ch128[2]);
      }*/
    }

    avg1 = (((int *)&avg128)[0] +
            ((int *)&avg128)[1] +
            ((int *)&avg128)[2] +
            ((int *)&avg128)[3])/(nb_rb*12);

    if (avg1>avg2)
      avg2 = avg1;

    //LOG_I(PHY,"Channel level : %d, %d\n",avg1, avg2);
  }

  return(avg2);
}

static void nr_psbch_channel_compensation(struct complex16 rxdataF_ext[][PSBCH_MAX_RE_PER_SYMBOL],
					 struct complex16 dl_ch_estimates_ext[][PSBCH_MAX_RE_PER_SYMBOL],
					 int nb_re,
					 struct complex16 rxdataF_comp[][PSBCH_MAX_RE_PER_SYMBOL],
					 NR_DL_FRAME_PARMS *frame_parms,
					 uint8_t output_shift) {
  for (int aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
    vect128 *dl_ch128          = (vect128 *)dl_ch_estimates_ext[aarx];
    vect128 *rxdataF128        = (vect128 *)rxdataF_ext[aarx];
    vect128 *rxdataF_comp128   = (vect128 *)rxdataF_comp[aarx];

    for (int re=0; re<nb_re; re+=12) {
      *rxdataF_comp128++ = mulByConjugate128(rxdataF128++, dl_ch128++, output_shift);
      *rxdataF_comp128++ = mulByConjugate128(rxdataF128++, dl_ch128++, output_shift);
      *rxdataF_comp128++ = mulByConjugate128(rxdataF128++, dl_ch128++, output_shift);
    }
  }
}

static void nr_psbch_unscrambling(NR_UE_PSBCH *psbch,
                                 int16_t *demod_psbch_e,
                                 uint16_t Nid,
                                 uint8_t nushift,
                                 uint16_t M,
                                 uint16_t length,
                                 uint8_t bitwise,
                                 uint32_t unscrambling_mask,
                                 uint32_t psbch_a_prime,
                                 uint32_t *psbch_a_interleaved) {
  uint8_t reset, offset;
  uint32_t x1, x2, s=0;
  uint8_t k=0;
  reset = 1;
  // x1 is set in first call to lte_gold_generic
  x2 = Nid; //this is c_init

  // The Gold sequence is shifted by nushift* M, so we skip (nushift*M /32) double words
  for (int i=0; i<(uint16_t)ceil(((float)M)/32); i++) {
    s = lte_gold_generic(&x1, &x2, reset);
    reset = 0;
  }

  // Scrambling is now done with offset (nushift*M)%32
  offset = 0; //(nushift*M)&0x1f;

  for (int i=0; i<length; i++) {
    /*if (((i+offset)&0x1f)==0) {
      s = lte_gold_generic(&x1, &x2, reset);
      reset = 0;
    }*/
    if (bitwise) {
      if (((k+offset)&0x1f)==0 && (!((unscrambling_mask>>i)&1))) {
        s = lte_gold_generic(&x1, &x2, reset);
        reset = 0;
      }

      *psbch_a_interleaved ^= ((unscrambling_mask>>i)&1)? ((psbch_a_prime>>i)&1)<<i : (((psbch_a_prime>>i)&1) ^ ((s>>((k+offset)&0x1f))&1))<<i;
      k += (!((unscrambling_mask>>i)&1));
#ifdef DEBUG_PSBCH_ENCODING
      printf("i %d k %d offset %d (unscrambling_mask>>i)&1) %d s: %08x\t  psbch_a_interleaved 0x%08x (!((unscrambling_mask>>i)&1)) %d\n", i, k, offset, (unscrambling_mask>>i)&1, s, *psbch_a_interleaved,
             (!((unscrambling_mask>>i)&1)));
#endif
    } else {
      if (((i+offset)&0x1f)==0) {
        s = lte_gold_generic(&x1, &x2, reset);
        reset = 0;
      }

      if (((s>>((i+offset)&0x1f))&1)==1)
        demod_psbch_e[i] = -demod_psbch_e[i];

#ifdef DEBUG_PSBCH_ENCODING

      if (i<8)
        printf("s %d demod_psbch_e[i] %d\n", ((s>>((i+offset)&0x1f))&1), demod_psbch_e[i]);

#endif
    }
  }
}

void nr_sl_common_signal_procedures(PHY_VARS_NR_UE *ue, int frame, int slot)
{
  NR_DL_FRAME_PARMS *fp = &ue->frame_parms;
  int **txdataF = ue->common_vars.txdataF;
  uint8_t ssb_index = 0; //TODO: Need update to get 0 or 1 from parameter in case of mu = 1.
  int txdataF_offset = slot * fp->samples_per_slot_wCP;

  int ssb_start_symbol_abs = (ue->slss->sl_timeoffsetssb_r16 + ue->slss->sl_timeinterval_r16 * ssb_index) * fp->symbols_per_slot;
  uint16_t ssb_start_symbol = ssb_start_symbol_abs % fp->symbols_per_slot;
  LOG_D(NR_PHY, "common_signal_procedures: frame %d, slot %d ssb index %d, ssb_start_symbol %d\n", frame, slot, ssb_index, ssb_start_symbol);

  const int prb_offset = 0; //TODO: Need to properly get these values.
  const int sc_offset = 0; //TODO: Need to properly get these values.
  fp->ssb_start_subcarrier = prb_offset * 12 + sc_offset;
  LOG_D(NR_PHY, "SSB first subcarrier %d (%d,%d)\n", fp->ssb_start_subcarrier, prb_offset, sc_offset);

  nr_sl_generate_pss(&txdataF[0][txdataF_offset], AMP, ssb_start_symbol, fp);
  nr_sl_generate_sss(&txdataF[0][txdataF_offset], AMP, ssb_start_symbol, fp);
  nr_sl_generate_psbch_dmrs(&ue->nr_gold_psbch[ssb_index & 7], &txdataF[0][txdataF_offset], AMP, ssb_start_symbol, fp);
  uint8_t n_hf = 0;
  nr_generate_sl_psbch(ue, &txdataF[0][txdataF_offset], AMP, ssb_start_symbol, n_hf, frame, fp);
}

int nr_rx_psbch( PHY_VARS_NR_UE *ue,
                UE_nr_rxtx_proc_t *proc,
                int estimateSz, struct complex16 dl_ch_estimates [][estimateSz],
                NR_UE_PSBCH *nr_ue_psbch_vars,
                NR_DL_FRAME_PARMS *frame_parms,
                uint8_t gNB_id,
                uint8_t i_ssb,
                MIMO_mode_t mimo_mode,
                NR_UE_PDCCH_CONFIG *phy_pdcch_config,
                fapiPsbch_t *result) {

  NR_UE_COMMON *nr_ue_common_vars = &ue->common_vars;
  uint8_t ssb_index = 0; //TODO: Need update to get 0 or 1 from parameter in case of mu = 1.
  int symbol_offset = ue->is_synchronized > 0 ? (ue->slss->sl_timeoffsetssb_r16 + ue->slss->sl_timeinterval_r16 * ssb_index) * frame_parms->symbols_per_slot : 0;
  int psbch_e_rx_idx = 0;
  int16_t psbch_unClipped[NR_POLAR_PSBCH_E] = {0};
  int16_t psbch_e_rx[NR_POLAR_PSBCH_E] = {0};
  // symbol refers to symbol within SSB. symbol_offset is the offset of the SSB wrt start of slot
  for (int symbol = 0; symbol < 13; symbol++) {
    const uint16_t nb_re = 99;
    __attribute__ ((aligned(32))) struct complex16 rxdataF_ext[frame_parms->nb_antennas_rx][PSBCH_MAX_RE_PER_SYMBOL];
    __attribute__ ((aligned(32))) struct complex16 dl_ch_estimates_ext[frame_parms->nb_antennas_rx][PSBCH_MAX_RE_PER_SYMBOL];
    memset(dl_ch_estimates_ext,0, sizeof  dl_ch_estimates_ext);
    nr_psbch_extract(nr_ue_common_vars->common_vars_rx_data_per_thread[proc->thread_id].rxdataF,
                    estimateSz,
                    dl_ch_estimates,
                    rxdataF_ext,
                    dl_ch_estimates_ext,
                    symbol,
                    symbol_offset,
                    frame_parms);

    double log2_maxh = 0;
    if (symbol == 0) {
      int max_h = nr_psbch_channel_level(dl_ch_estimates_ext,
                                         frame_parms,
                                         nb_re);
      log2_maxh = 3 + (log2_approx(max_h) / 2);
#ifdef DEBUG_PSBCH
      LOG_I(NR_PHY, "PSBCH Symbol %d ofdm size %d\n", symbol, frame_parms->ofdm_symbol_size );
      LOG_I(NR_PHY,"PSBCH log2_maxh = %d (%d)\n", log2_maxh, max_h);
#endif
    }

    __attribute__ ((aligned(32))) struct complex16 rxdataF_comp[frame_parms->nb_antennas_rx][PSBCH_MAX_RE_PER_SYMBOL];
    nr_psbch_channel_compensation(rxdataF_ext,
                                  dl_ch_estimates_ext,
                                  nb_re,
                                  rxdataF_comp,
                                  frame_parms,
                                  log2_maxh); // log2_maxh+I0_shift

    int nb = 198; // QPSK 2 bits 99*2 (m from TX side)
    nr_psbch_quantize(psbch_e_rx + psbch_e_rx_idx, (short *)rxdataF_comp[0], nb);
    memcpy(psbch_unClipped + psbch_e_rx_idx, rxdataF_comp[0], nb*sizeof(int16_t));
    psbch_e_rx_idx += nb;

    if (symbol == 0)
      symbol += 4;  // skip to accommodate PSS and SSS
  }
  // legacy code use int16, but it is complex16
  UEscopeCopy(ue, psbchRxdataF_comp, psbch_unClipped, sizeof(struct complex16), frame_parms->nb_antennas_rx, psbch_e_rx_idx/2);
  UEscopeCopy(ue, psbchLlr, psbch_e_rx, sizeof(int16_t), frame_parms->nb_antennas_rx, psbch_e_rx_idx);
#ifdef DEBUG_PSBCH
  for (int cnt = 0; cnt < NR_POLAR_PSBCH_E; cnt++)
    printf("psbch rx llr %d\n",*(psbch_e_rx + cnt));

#endif
  //un-scrambling
  uint16_t M =  NR_POLAR_PSBCH_E;
  uint8_t nushift = 0; //(Lmax==4)? i_ssb&3 : i_ssb&7;
  uint32_t psbch_a_interleaved = 0;
  uint32_t psbch_a_prime = 0;
  nr_psbch_unscrambling(nr_ue_psbch_vars, psbch_e_rx, frame_parms->Nid_SL, nushift, M, NR_POLAR_PSBCH_E,
                        0, 0,  psbch_a_prime, &psbch_a_interleaved);
  //polar decoding de-rate matching
  uint64_t tmp = 0;
  uint32_t decoderState = polar_decoder_int16(psbch_e_rx, (uint64_t *)&tmp, 0,
                                     NR_POLAR_PSBCH_MESSAGE_TYPE,
                                     NR_POLAR_PSBCH_PAYLOAD_BITS,
                                     NR_POLAR_PSBCH_AGGREGATION_LEVEL);
  psbch_a_prime = tmp;
  if(decoderState)
    return(decoderState);

  uint32_t payload = 0;
  for (int i = 0; i < NR_POLAR_PSBCH_PAYLOAD_BITS; i++)
    payload |= (((uint64_t)psbch_a_prime >> i) & 1) << (31 - i);
  printf("PSBCH payload received 0x%x \n", payload);

  for (int i = 0; i < 4; i++)
    result->decoded_output[i] = (uint8_t)((payload >> ((3 - i) << 3)) & 0xff);

  frame_parms->half_frame_bit = (result->xtra_byte >> 4) & 0x01; // computing the half frame index from the extra byte
  frame_parms->ssb_index = i_ssb;  // ssb index corresponds to i_ssb for Lmax = 4,8

  if (frame_parms->Lmax == 64) {   // for Lmax = 64 ssb index 4th,5th and 6th bits are in extra byte
    for (int i = 0; i < 3; i++)
      frame_parms->ssb_index += (((result->xtra_byte >> (7 - i)) & 0x01) << (3 + i));
  }

  ue->symbol_offset = 0; 
  if (frame_parms->half_frame_bit)
  ue->symbol_offset += (frame_parms->slots_per_frame>>1)*frame_parms->symbols_per_slot;

  uint8_t frame_number_4lsb = 0;

  for (int i = 0; i < 4; i++)
    frame_number_4lsb |= ((result->xtra_byte >> i) & 1) << (3 - i);

  proc->decoded_frame_rx = frame_number_4lsb;
#ifdef DEBUG_PSBCH
  printf("xtra_byte %x payload %x\n", result->xtra_byte, payload);

  for (int i=0; i<(NR_POLAR_PSBCH_PAYLOAD_BITS>>3); i++) {
    //     printf("unscrambling psbch_a[%d] = %x \n", i,psbch_a[i]);
    printf("[PSBCH] decoder payload[%d] = %x\n",i,result->decoded_output[i]);
  }

#endif
  nr_downlink_indication_t dl_indication;
  fapi_nr_rx_indication_t *rx_ind = calloc(1, sizeof(*rx_ind));
  uint16_t number_pdus = 1;
  //TODO: Needs to validate following function calls are required or not
  nr_fill_dl_indication(&dl_indication, NULL, rx_ind, proc, ue, gNB_id, phy_pdcch_config);
  nr_fill_rx_indication(rx_ind, FAPI_NR_RX_PDU_TYPE_SSB, gNB_id, ue, NULL, NULL, number_pdus, proc,(void *)result);

  if (ue->if_inst && ue->if_inst->dl_indication)
    ue->if_inst->dl_indication(&dl_indication, NULL);
  else
    free(rx_ind); // dl_indication would free(), so free() here if not called

  return 0;
}

