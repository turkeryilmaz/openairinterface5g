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

/*! \file PHY/LTE_TRANSPORT/pucch.c
* \brief Top-level routines for generating and decoding the PUCCH physical channel V8.6 2009-03
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr
* \note
* \warning
*/
#include "PHY/defs_UE.h"
#include "PHY/phy_extern_ue.h" 
#include "PHY/LTE_REFSIG/lte_refsig.h"
#include "PHY/LTE_TRANSPORT/transport_vars.h"

#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"

#include "T.h"

#include "../LTE_TRANSPORT/pucch_extern.h"

void generate_pucch1x(int32_t **txdataF,
		      LTE_DL_FRAME_PARMS *frame_parms,
		      uint8_t ncs_cell[20][7],
		      PUCCH_FMT_t fmt,
		      PUCCH_CONFIG_DEDICATED *pucch_config_dedicated,
		      uint16_t n1_pucch,
		      uint8_t shortened_format,
		      uint8_t *payload,
		      int16_t amp,
		      uint8_t subframe)
{

  uint32_t u,v,n;
  c16_t z[12 * 14], *zptr;
  int16_t d0;
  uint8_t ns,N_UL_symb,nsymb,n_oc,n_oc0,n_oc1;
  uint8_t c = (frame_parms->Ncp==0) ? 3 : 2;
  uint16_t nprime,nprime0,nprime1;
  uint16_t i,j,re_offset,thres,h;
  uint8_t Nprime_div_deltaPUCCH_Shift,Nprime,d;
  uint8_t m,l,refs;
  uint8_t n_cs, S, alpha_ind, rem;
  uint32_t symbol_offset;

  uint8_t deltaPUCCH_Shift          = frame_parms->pucch_config_common.deltaPUCCH_Shift;
  uint8_t NRB2                      = frame_parms->pucch_config_common.nRB_CQI;
  uint8_t Ncs1                      = frame_parms->pucch_config_common.nCS_AN;
  uint8_t Ncs1_div_deltaPUCCH_Shift = Ncs1/deltaPUCCH_Shift;

  LOG_D(PHY,"generate_pucch Start [deltaPUCCH_Shift %d, NRB2 %d, Ncs1_div_deltaPUCCH_Shift %d, n1_pucch %d]\n", deltaPUCCH_Shift, NRB2, Ncs1_div_deltaPUCCH_Shift,n1_pucch);


  uint32_t u0 = (frame_parms->Nid_cell + frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.grouphop[subframe<<1]) % 30;
  uint32_t u1 = (frame_parms->Nid_cell + frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.grouphop[1+(subframe<<1)]) % 30;
  uint32_t v0=frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[subframe<<1];
  uint32_t v1=frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[1+(subframe<<1)];

  if ((deltaPUCCH_Shift==0) || (deltaPUCCH_Shift>3)) {
    printf("[PHY] generate_pucch: Illegal deltaPUCCH_shift %d (should be 1,2,3)\n",deltaPUCCH_Shift);
    return;
  }

  if (Ncs1_div_deltaPUCCH_Shift > 7) {
    printf("[PHY] generate_pucch: Illegal Ncs1_div_deltaPUCCH_Shift %d (should be 0...7)\n",Ncs1_div_deltaPUCCH_Shift);
    return;
  }

  zptr = z;
  thres = (c*Ncs1_div_deltaPUCCH_Shift);
  Nprime_div_deltaPUCCH_Shift = (n1_pucch < thres) ? Ncs1_div_deltaPUCCH_Shift : (12/deltaPUCCH_Shift);
  Nprime = Nprime_div_deltaPUCCH_Shift * deltaPUCCH_Shift;

#ifdef DEBUG_PUCCH_TX
  printf("[PHY] PUCCH: cNcs1/deltaPUCCH_Shift %d, Nprime %d, n1_pucch %d\n",thres,Nprime,n1_pucch);
#endif

  LOG_D(PHY,"[PHY] PUCCH: n1_pucch %d, thres %d Ncs1_div_deltaPUCCH_Shift %d (12/deltaPUCCH_Shift) %d Nprime_div_deltaPUCCH_Shift %d \n",
		              n1_pucch, thres, Ncs1_div_deltaPUCCH_Shift, (int)(12/deltaPUCCH_Shift), Nprime_div_deltaPUCCH_Shift);
  LOG_D(PHY,"[PHY] PUCCH: deltaPUCCH_Shift %d, Nprime %d\n",deltaPUCCH_Shift,Nprime);


  N_UL_symb = (frame_parms->Ncp==0) ? 7 : 6;

  if (n1_pucch < thres)
    nprime0=n1_pucch;
  else
    nprime0 = (n1_pucch - thres)%(12*c/deltaPUCCH_Shift);

  if (n1_pucch >= thres)
    nprime1= ((c*(nprime0+1))%((12*c/deltaPUCCH_Shift)+1))-1;
  else {
    d = (frame_parms->Ncp==0) ? 2 : 0;
    h= (nprime0+d)%(c*Nprime_div_deltaPUCCH_Shift);
#ifdef DEBUG_PUCCH_TX
    printf("[PHY] PUCCH: h %d, d %d\n",h,d);
#endif
    nprime1 = (h/c) + (h%c)*Nprime_div_deltaPUCCH_Shift;
  }

#ifdef DEBUG_PUCCH_TX
  printf("[PHY] PUCCH: nprime0 %d nprime1 %d, %s, payload (%d,%d)\n",nprime0,nprime1,pucch_format_string[fmt],payload[0],payload[1]);
#endif

  n_oc0 = nprime0/Nprime_div_deltaPUCCH_Shift;

  if (frame_parms->Ncp==1)
    n_oc0<<=1;

  n_oc1 = nprime1/Nprime_div_deltaPUCCH_Shift;

  if (frame_parms->Ncp==1)  // extended CP
    n_oc1<<=1;

#ifdef DEBUG_PUCCH_TX
  printf("[PHY] PUCCH: noc0 %d noc1 %d\n",n_oc0,n_oc1);
#endif

  nprime=nprime0;
  n_oc  =n_oc0;

  c16_t W = {};
  // loop over 2 slots
  for (ns=(subframe<<1),u=u0,v=v0; ns<(2+(subframe<<1)); ns++,u=u1,v=v1) {

    if ((nprime&1) == 0)
      S=0;  // 1
    else
      S=1;  // j

    //loop over symbols in slot
    for (l=0; l<N_UL_symb; l++) {
      // Compute n_cs (36.211 p. 18)
      n_cs = ncs_cell[ns][l];

      if (frame_parms->Ncp==0) { // normal CP
        n_cs = ((uint16_t)n_cs + (nprime*deltaPUCCH_Shift + (n_oc%deltaPUCCH_Shift))%Nprime)%12;
      } else {
        n_cs = ((uint16_t)n_cs + (nprime*deltaPUCCH_Shift + (n_oc>>1))%Nprime)%12;
      }

      refs = 0;
      // Comput W_noc(m) (36.211 p. 19)
      if ((ns == (1 + (subframe << 1))) && (shortened_format == 1)) { // second slot and shortened format
        if (l<2) {                                         // data
          W = W3[n_oc][l];
        } else if ((l<N_UL_symb-2)&&(frame_parms->Ncp==0)) { // reference and normal CP
          W = W3[n_oc][l - 2];
          refs=1;
        } else if ((l<N_UL_symb-2)&&(frame_parms->Ncp==1)) { // reference and extended CP
          W = (c16_t){W4[n_oc][l - 2], 0};
          refs=1;
        } else if ((l>=N_UL_symb-2)) {                      // data
          W = W3[n_oc][l - N_UL_symb + 4];
        }
      } else {
        if (l<2) {                                         // data
          W = (c16_t){W4[n_oc][l], 0};
        } else if ((l<N_UL_symb-2)&&(frame_parms->Ncp==0)) { // reference and normal CP
          W = W3[n_oc][l - 2];
          refs=1;
        } else if ((l<N_UL_symb-2)&&(frame_parms->Ncp==1)) { // reference and extended CP
          W = (c16_t){W4[n_oc][l - 2], 0};
          refs=1;
        } else if ((l>=N_UL_symb-2)) {                     // data
          W = (c16_t){W4[n_oc][l - N_UL_symb + 4], 0};
        }
      }

      // multiply W by S(ns) (36.211 p.17). only for data, reference symbols do not have this factor
      if ((S==1)&&(refs==0)) {
        W = (c16_t){-W.i, W.r};
      }

#ifdef DEBUG_PUCCH_TX
      printf("[PHY] PUCCH: ncs[%d][%d]=%d, W_re %d, W_im %d, S %d, refs %d\n",ns,l,n_cs,W_re,W_im,S,refs);
#endif
      alpha_ind=0;
      // compute output sequence

      for (n = 0; n < 12; n++) {
        // this is r_uv^alpha(n)
        c16_t tmp = c16mulShift(alphaTBL[alpha_ind], ul_ref_sigs[u][v][0][n], 15);
        // this is S(ns)*w_noc(m)*r_uv^alpha(n)
        c16_t ref = c16mulShift(tmp, W, 15);
        if ((l<2)||(l>=(N_UL_symb-2))) { //these are PUCCH data symbols
          switch (fmt) {
          case pucch_format1:   //OOK 1-bit
            zptr[n] = c16mulRealShift(ref, amp, 15);
            break;

          case pucch_format1a:  //BPSK 1-bit
            d0 = (payload[0]&1)==0 ? amp : -amp;
            zptr[n] = c16mulRealShift(ref, d0, 15);
            break;

          case pucch_format1b:  //QPSK 2-bits (Table 5.4.1-1 from 36.211, pg. 18)
            if (((payload[0]&1)==0) && ((payload[1]&1)==0))  {// 1
              zptr[n] = c16mulRealShift(ref, amp, 15);
            } else if (((payload[0]&1)==0) && ((payload[1]&1)==1))  { // -j
              zptr[n] = c16mulRealShift((c16_t){ref.i, -ref.r}, amp, 15);
            } else if (((payload[0]&1)==1) && ((payload[1]&1)==0))  { // j
              zptr[n] = c16mulRealShift((c16_t){-ref.i, ref.r}, amp, 15);
            } else  { // -1
              zptr[n] = c16mulRealShift((c16_t){-ref.r, -ref.i}, amp, 15);
            }

            break;
	  case pucch_format1b_csA2:
	  case pucch_format1b_csA3:
	  case pucch_format1b_csA4:
	    AssertFatal(1==0,"PUCCH format 1b_csX not supported yet\n");
	    break;
          case pucch_format2:
          case pucch_format2a:
          case pucch_format2b:
            AssertFatal(1==0,"should not go here\n");
            break;

          case pucch_format3:
            fprintf(stderr, "PUCCH format 3 not handled\n");
            abort();
          } // switch fmt
        } else { // These are PUCCH reference symbols
          zptr[n] = c16mulRealShift(ref, amp, 15);
        }

#ifdef DEBUG_PUCCH_TX
        printf("[PHY] PUCCH subframe %d z(%d,%u) => %d,%d, alpha(%d) => %d,%d\n",subframe,l,n,((int16_t *)&zptr[n])[0],((int16_t *)&zptr[n])[1],
            alpha_ind,alpha_re[alpha_ind],alpha_im[alpha_ind]);
#endif
        alpha_ind = (alpha_ind + n_cs)%12;
      } // n

      zptr+=12;
    } // l

    nprime=nprime1;
    n_oc  =n_oc1;
  } // ns

  rem = ((((12*Ncs1_div_deltaPUCCH_Shift)>>3)&7)>0) ? 1 : 0;

  m = (n1_pucch < thres) ? NRB2 : (((n1_pucch-thres)/(12*c/deltaPUCCH_Shift))+NRB2+((deltaPUCCH_Shift*Ncs1_div_deltaPUCCH_Shift)>>3)+rem);

#ifdef DEBUG_PUCCH_TX
  printf("[PHY] PUCCH: m %d\n",m);
#endif
  nsymb = N_UL_symb<<1;

  //for (j=0,l=0;l<(nsymb-1);l++) {
  for (j=0,l=0; l<(nsymb); l++) {
    if ((l<(nsymb>>1)) && ((m&1) == 0))
      re_offset = (m*6) + frame_parms->first_carrier_offset;
    else if ((l<(nsymb>>1)) && ((m&1) == 1))
      re_offset = frame_parms->first_carrier_offset + (frame_parms->N_RB_DL - (m>>1) - 1)*12;
    else if ((m&1) == 0)
      re_offset = frame_parms->first_carrier_offset + (frame_parms->N_RB_DL - (m>>1) - 1)*12;
    else
      re_offset = ((m-1)*6) + frame_parms->first_carrier_offset;

    if (re_offset > frame_parms->ofdm_symbol_size)
      re_offset -= (frame_parms->ofdm_symbol_size);

    symbol_offset = (unsigned int)frame_parms->ofdm_symbol_size*(l+(subframe*nsymb));
    c16_t *txptr = (c16_t *)&txdataF[0][symbol_offset];

    for (i=0; i<12; i++,j++) {
      txptr[re_offset++] = z[j];

      if (re_offset==frame_parms->ofdm_symbol_size)
        re_offset = 0;

#ifdef DEBUG_PUCCH_TX
      printf("[PHY] PUCCH subframe %d (%d,%d,%d,%d) => %d,%d\n",subframe,l,i,re_offset-1,m,((int16_t *)&z[j])[0],((int16_t *)&z[j])[1]);
#endif
    }
  }

}

static inline void pucch2x_scrambling_modulation(LTE_DL_FRAME_PARMS *fp,
                                                 int subframe,
                                                 uint16_t rnti,
                                                 uint32_t B,
                                                 int16_t amp,
                                                 int16_t *d)
{
  uint32_t x1, x2;
  x2 = (rnti) + ((uint32_t)(1 + subframe) << 16) * (1 + (fp->Nid_cell << 1)); // this is c_init in 36.211 Sec 6.3.1
  uint32_t s = lte_gold_generic(&x1, &x2, 1);
  for (int i = 0; i < 20; i++) {
    d[i] = ((B ^ s) >> i) & 1 ? -amp : amp;
  }
}

uint32_t pucch_code[13] = {0xFFFFF,0x5A933,0x10E5A,0x6339C,0x73CE0,
			   0xFFC00,0xD8E64,0x4F6B0,0x218EC,0x1B746,
			   0x0FFFF,0x33FFF,0x3FFFC};

void generate_pucch2x(c16_t **txdataF,
                      LTE_DL_FRAME_PARMS *fp,
                      uint8_t ncs_cell[20][7],
                      PUCCH_FMT_t fmt,
                      PUCCH_CONFIG_DEDICATED *pucch_config_dedicated,
                      uint16_t n2_pucch,
                      uint8_t *payload,
                      int A,
                      int B2,
                      int16_t amp,
                      uint8_t subframe,
                      uint16_t rnti)
{
  int i,j;
  uint32_t B = 0;
  c16_t d[11];
  uint8_t deltaPUCCH_Shift          = fp->pucch_config_common.deltaPUCCH_Shift;
  uint8_t NRB2                      = fp->pucch_config_common.nRB_CQI;
  uint8_t Ncs1                      = fp->pucch_config_common.nCS_AN;

  uint32_t u0 = fp->pucch_config_common.grouphop[subframe<<1];
  uint32_t u1 = fp->pucch_config_common.grouphop[1+(subframe<<1)];
  uint32_t v0 = fp->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[subframe<<1];
  uint32_t v1 = fp->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[1+(subframe<<1)];

  c16_t z[12 * 14], *zptr;
  uint32_t u,v,n;
  uint8_t ns,N_UL_symb,nsymb_slot0,nsymb_pertti;
  uint32_t nprime,l,n_cs;
  int alpha_ind, data_ind;
  int m, re_offset, symbol_offset;

  if ((deltaPUCCH_Shift==0) || (deltaPUCCH_Shift>3)) {
    printf("[PHY] generate_pucch: Illegal deltaPUCCH_shift %d (should be 1,2,3)\n",deltaPUCCH_Shift);
    return;
  }

  if (Ncs1 > 7) {
    printf("[PHY] generate_pucch: Illegal Ncs1 %d (should be 0...7)\n",Ncs1);
    return;
  }

  // pucch2x_encoding
  for (i=0;i<A;i++)
    if ((*payload & (1<<i)) > 0)
      B=B^pucch_code[i];

  // scrambling
  pucch2x_scrambling_modulation(fp, subframe, rnti, B, amp, (int16_t *)d);

  // add extra symbol for 2a/2b
  d[10] = (c16_t){};
  if (fmt==pucch_format2a)
    d[10] = (B2 == 0) ? (c16_t){amp, 0} : (c16_t){-amp, 0};
  else if (fmt==pucch_format2b) {
    switch (B2) {
    case 0:
      d[10] = (c16_t){amp, 0};
      break;
    case 1:
      d[10] = (c16_t){0, -amp};
      break;
    case 2:
      d[10] = (c16_t){0, amp};
      break;
    case 3:
      d[10] = (c16_t){-amp, 0};
      break;
    default:
      AssertFatal(1==0,"Illegal modulation symbol %d for PUCCH %s\n",B2,pucch_format_string[fmt]);
      break;
    }
  }


#ifdef DEBUG_PUCCH_TX
  printf("[PHY] PUCCH2x: n2_pucch %d\n",n2_pucch);
#endif

  N_UL_symb = (fp->Ncp==0) ? 7 : 6;
  data_ind  = 0;
  zptr      = z;
  nprime    = 0;
  for (ns=(subframe<<1),u=u0,v=v0; ns<(2+(subframe<<1)); ns++,u=u1,v=v1) {

    if ((ns&1) == 0)
        nprime = (n2_pucch < 12*NRB2) ?
                n2_pucch % 12 :
                (n2_pucch+Ncs1 + 1)%12;
    else {
        nprime = (n2_pucch < 12*NRB2) ?
                ((12*(nprime+1)) % 13)-1 :
                (10-n2_pucch)%12;
    }
    //loop over symbols in slot
    for (l=0; l<N_UL_symb; l++) {
      // Compute n_cs (36.211 p. 18)
      n_cs = (ncs_cell[ns][l]+nprime)%12;

      alpha_ind = 0;
      for (n=0; n<12; n++)
      {
          // this is r_uv^alpha(n)
          c16_t ref = c16mulShift(alphaTBL[alpha_ind], ul_ref_sigs[u][v][0][n], 15);
          if ((l != 1) && (l != 5)) { // these are PUCCH data symbols
            zptr[n] = c16mulShift(d[data_ind], ref, 15);
          } else {
            if ((l == 1) || ((l == 5) && (fmt == pucch_format2))) {
              zptr[n] = c16mulRealShift(ref, amp, 15);
            }
            // l == 5 && pucch format 2a
            else if (fmt == pucch_format2a) {
              zptr[n] = (c16_t){(d[10].r * ref.r) >> 15, (d[10].i * ref.i) >> 15};
            }
            // l == 5 && pucch format 2b
            else if (fmt == pucch_format2b) {
              zptr[n] = (c16_t){(d[10].r * ref.r) >> 15, (d[10].i * ref.i) >> 15};
            }
          } // l==1 || l==5
      alpha_ind = (alpha_ind + n_cs)%12;
      } // n
      zptr+=12;

      if ((l!=1)&&(l!=5))  //these are PUCCH data symbols so increment data index
        data_ind++;
    } // l
  } //ns

  m = n2_pucch/12;

#ifdef DEBUG_PUCCH_TX
  LOG_D(PHY,"[PHY] PUCCH: n2_pucch %d m %d\n",n2_pucch,m);
#endif

  nsymb_slot0  = ((fp->Ncp==0) ? 7 : 6);
  nsymb_pertti = nsymb_slot0 << 1;

  //nsymb = nsymb_slot0<<1;

  //for (j=0,l=0;l<(nsymb-1);l++) {
  for (j=0,l=0; l<(nsymb_pertti); l++) {

    if ((l<nsymb_slot0) && ((m&1) == 0))
      re_offset = (m*6) + fp->first_carrier_offset;
    else if ((l<nsymb_slot0) && ((m&1) == 1))
      re_offset = fp->first_carrier_offset + (fp->N_RB_DL - (m>>1) - 1)*12;
    else if ((m&1) == 0)
      re_offset = fp->first_carrier_offset + (fp->N_RB_DL - (m>>1) - 1)*12;
    else
      re_offset = ((m-1)*6) + fp->first_carrier_offset;

    if (re_offset > fp->ofdm_symbol_size)
      re_offset -= (fp->ofdm_symbol_size);



    symbol_offset = (unsigned int)fp->ofdm_symbol_size*(l+(subframe*nsymb_pertti));
    c16_t *txptr = &txdataF[0][symbol_offset];

    //LOG_I(PHY,"ofdmSymb %d/%d, firstCarrierOffset %d, symbolOffset[sfn %d] %d, reOffset %d, &txptr: %x \n", l, nsymb, fp->first_carrier_offset, subframe, symbol_offset, re_offset, &txptr[0]);

    for (i=0; i<12; i++,j++) {
      txptr[re_offset] = z[j];

      re_offset++;

      if (re_offset==fp->ofdm_symbol_size)
          re_offset -= (fp->ofdm_symbol_size);

#ifdef DEBUG_PUCCH_TX
      LOG_D(PHY,"[PHY] PUCCH subframe %d (%d,%d,%d,%d) => %d,%d\n",subframe,l,i,re_offset-1,m,((int16_t *)&z[j])[0],((int16_t *)&z[j])[1]);
#endif
    }
  }
}

/* PUCCH format3 >> */
/* DFT */
static void pucchfmt3_Dft(c16_t *x, c16_t *y)
{
  for (int i = 0; i < D_NSC1RB; i++) {
    c32_t calctmp = {0};
    for (int k = 0; k < D_NSC1RB; k++) {
      c16_t tmp = c16mulShift(x[k], alphaTBL[(12 - ((i * k) % 12)) % 12], 15);
      calctmp = (c32_t){calctmp.r + tmp.r, calctmp.i + tmp.i};
    }
    y[i] = (c16_t){calctmp.r / sqrt(D_NSC1RB), calctmp.i / sqrt(D_NSC1RB)};
  }
}

void generate_pucch3x(c16_t **txdataF,
                      LTE_DL_FRAME_PARMS *frame_parms,
                      uint8_t ncs_cell[20][7],
                      PUCCH_FMT_t fmt,
                      PUCCH_CONFIG_DEDICATED *pucch_config_dedicated,
                      uint16_t n3_pucch,
                      uint8_t shortened_format,
                      uint8_t *payload,
                      int16_t amp,
                      uint8_t subframe,
                      uint16_t rnti)
{

  uint32_t u, v;
  uint16_t i, j, re_offset;
  c16_t z[12 * 14], *zptr;
  c16_t y_tilda[12 * 14] = {}, *y_tilda_ptr;
  uint8_t ns, nsymb, n_oc, n_oc0, n_oc1;
  uint8_t N_UL_symb = (frame_parms->Ncp==0) ? 7 : 6;
  uint8_t m, l;
  uint8_t n_cs;
  c16_t *txptr;
  uint32_t symbol_offset;
  
  uint32_t u0 = (frame_parms->Nid_cell + frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.grouphop[subframe<<1]) % 30;
  uint32_t u1 = (frame_parms->Nid_cell + frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.grouphop[1+(subframe<<1)]) % 30;
  uint32_t v0=frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[subframe<<1];
  uint32_t v1=frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[1+(subframe<<1)];

  // variables for channel coding
  uint8_t chcod_tbl_idx = 0;
  //uint8_t chcod_dt[48] = {};

  // variables for Scrambling
  uint32_t cinit = 0;
  uint32_t x1;
  uint32_t s,s0,s1;
  uint8_t  C[48] ={};
  uint8_t  scr_dt[48]={};

  // variables for Modulation
  c16_t d[24] = {};

  // variables for orthogonal sequence selection
  uint8_t N_PUCCH_SF0 = 5;
  uint8_t N_PUCCH_SF1 = (shortened_format==0)? 5:4;
  uint8_t first_slot = 0;
  uint8_t dt_offset;
  uint8_t sym_offset;
  c16_t W = {};
  // DMRS
  uint8_t alpha_idx=0;
  uint8_t m_alpha_idx=0;

  // TODO
  // "SR+ACK/NACK" length is only 7 bits.
  // This restriction will be lifted in the future.
  // "CQI/PMI/RI+ACK/NACK" will be supported in the future.

    // Channel Coding
    for (uint8_t i=0; i<7; i++) {
      chcod_tbl_idx += (payload[i]<<i); 
    }

    // Scrambling
    cinit = (subframe + 1) * ((2 * frame_parms->Nid_cell + 1)<<16) + rnti;
    s0 = lte_gold_generic(&x1,&cinit,1);
    s1 = lte_gold_generic(&x1,&cinit,0);

    for (i=0; i<48; i++) {
      s = (i<32)? s0:s1;
      j = (i<32)? i:(i-32);
      C[i] = ((s>>j)&1);
    }

    for (i=0; i<48; i++) {
      scr_dt[i] = chcod_tbl[chcod_tbl_idx][i] ^ C[i];
    }

    // Modulation
    for (uint8_t i=0; i<48; i+=2){
      if (scr_dt[i]==0 && scr_dt[i+1]==0){
        d[i >> 1].r = ((ONE_OVER_SQRT2_Q15 * amp) >> 15);
        d[i >> 1].i = ((ONE_OVER_SQRT2_Q15 * amp) >> 15);
      } else if (scr_dt[i]==0 && scr_dt[i+1]==1) {
        d[i >> 1].r = ((ONE_OVER_SQRT2_Q15 * amp) >> 15);
        d[i >> 1].i = -1 * ((ONE_OVER_SQRT2_Q15 * amp) >> 15);
      } else if (scr_dt[i]==1 && scr_dt[i+1]==0) {
        d[i >> 1].r = -1 * ((ONE_OVER_SQRT2_Q15 * amp) >> 15);
        d[i >> 1].i = ((ONE_OVER_SQRT2_Q15 * amp) >> 15);
      } else if (scr_dt[i]==1 && scr_dt[i+1]==1) {
        d[i >> 1].r = -1 * ((ONE_OVER_SQRT2_Q15 * amp) >> 15);
        d[i >> 1].i = -1 * ((ONE_OVER_SQRT2_Q15 * amp) >> 15);
      } else {
        //***log Modulation Error!
      }
    }

    // Calculate Orthogonal Sequence index
    n_oc0 = n3_pucch % N_PUCCH_SF1;
    if (N_PUCCH_SF1 == 5) {
      n_oc1 = (3 * n_oc0) % N_PUCCH_SF1;
    } else {
      n_oc1 = n_oc0 % N_PUCCH_SF1;
    }

    y_tilda_ptr = y_tilda;
    zptr = z;

    // loop over 2 slots
    for (ns=(subframe<<1), u=u0, v=v0; ns<(2+(subframe<<1)); ns++, u=u1, v=v1) {
      first_slot = (ns==(subframe<<1))?1:0;

      //loop over symbols in slot
      for (l=0; l<N_UL_symb; l++) {
        c16_t rot = RotTBL[ncs_cell[ns][l] / 64];

        // Comput W_noc(m) (36.211 p. 19)
        if ( first_slot == 0 && shortened_format==1) {  // second slot and shortened format
          n_oc = n_oc1;

          if (l<1) {                                         // data
            W = (c16_t){W4_fmt3[n_oc][l], 0};
          } else if (l==1) {                                  // DMRS
            W = (c16_t){W2[0], 0};
          } else if (l>=2 && l<5) {                          // data
            W = (c16_t){W4_fmt3[n_oc][l - 1], 0};
          } else if (l==5) {                                 // DMRS
            W = (c16_t){W2[1], 0};
          } else if ((l>=N_UL_symb-2)) {                      // data
            ;
          } else {
            //***log W Select Error!
          }
        } else {
          if (first_slot == 1) {                       // 1st slot or 2nd slot and not shortened
            n_oc=n_oc0;
          } else {
            n_oc=n_oc1;
          }

          if (l<1) {                                         // data
            W = (c16_t){W5_fmt3_re[n_oc][l], W5_fmt3_im[n_oc][l]};
          } else if (l==1) {                                  // DMRS
            W = (c16_t){W2[0], 0};
          } else if (l>=2 && l<5) {                          // data
            W = (c16_t){W5_fmt3_re[n_oc][l - 1], W5_fmt3_im[n_oc][l - 1]};
          } else if (l==5) {                                 // DMRS
            W = (c16_t){W2[1], 0};
          } else if ((l>=N_UL_symb-1)) {                     // data
            W = (c16_t){W5_fmt3_re[n_oc][l - N_UL_symb + 5], W5_fmt3_im[n_oc][l - N_UL_symb + 5]};
          } else {
            //***log W Select Error!
          }
        }  // W Selection end

        // Compute n_cs (36.211 p. 18)
        n_cs = ncs_cell[ns][l];
        if (N_PUCCH_SF1 == 5) {
          alpha_idx = (n_cs + Np5_TBL[n_oc]) % 12;
        } else {
          alpha_idx = (n_cs + Np4_TBL[n_oc]) % 12;
        }

        // generate pucch data
        dt_offset = (first_slot == 1) ? 0:12;
        sym_offset = (first_slot == 1) ? 0:7;

        for (i=0; i<12; i++) {
          // Calculate yn(i)
          c16_t tmp = c16mulShift(W, rot, 15);
          c16_t yn = c16mulShift(tmp, d[i + dt_offset], 15);
          // cyclic shift
          y_tilda_ptr[(l + sym_offset) * 12 + (i - (ncs_cell[ns][l] % 12) + 12) % 12] = yn;
          // DMRS
          m_alpha_idx = (alpha_idx * i) % 12;
          if (l==1 || l==5) {
            zptr[(l + sym_offset) * 12 + i] = c16mulShift(alphaTBL[m_alpha_idx], ul_ref_sigs[u][v][0][i], 15);
          }
        }

      } // l loop
    } // ns

    // DFT for pucch-data
    for (l=0; l<14; l++) {
      if (l==1 || l==5 || l==8 || l==12) {
        ;
      } else {
        pucchfmt3_Dft(&y_tilda_ptr[l * 12], &zptr[l * 12]);
      }
    }


    // Mapping
    m = n3_pucch / N_PUCCH_SF0;

    if (shortened_format == 1) {
      nsymb = (N_UL_symb<<1) - 1;
    } else {
      nsymb = (N_UL_symb<<1);
   }

    for (j=0,l=0; l<(nsymb); l++) {

      if ((l<7) && ((m&1) == 0))
        re_offset = (m*6) + frame_parms->first_carrier_offset;
      else if ((l<7) && ((m&1) == 1))
        re_offset = frame_parms->first_carrier_offset + (frame_parms->N_RB_DL - (m>>1) - 1)*12;
      else if ((m&1) == 0)
        re_offset = frame_parms->first_carrier_offset + (frame_parms->N_RB_DL - (m>>1) - 1)*12;
      else
        re_offset = ((m-1)*6) + frame_parms->first_carrier_offset;

      if (re_offset > frame_parms->ofdm_symbol_size)
        re_offset -= (frame_parms->ofdm_symbol_size);

      symbol_offset = (unsigned int)frame_parms->ofdm_symbol_size*(l+(subframe*14));
      txptr = &txdataF[0][symbol_offset];

      for (i=0; i<12; i++,j++) {
          txptr[re_offset++] = z[j];

        if (re_offset==frame_parms->ofdm_symbol_size)
          re_offset = 0;

#ifdef DEBUG_PUCCH_TX
        msg("[PHY] PUCCH subframe %d (%d,%d,%d,%d) => %d,%d\n",subframe,l,i,re_offset-1,m,((int16_t *)&z[j])[0],((int16_t *)&z[j])[1]);
#endif
      }
    }

}



