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

/* file: nr_rate_matching.c
   purpose: Procedures for rate matching/interleaving for NR LDPC
   author: hongzhi.wang@tcl.com
*/

#include "PHY/defs_gNB.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/sse_intrin.h"

//#define RM_DEBUG 1

static const uint8_t index_k0[2][4] = {{0, 17, 33, 56}, {0, 13, 25, 43}};

void nr_interleaving_ldpc(uint32_t E, uint8_t Qm, uint8_t *e,uint8_t *f, int start_idx) {
  uint32_t EQm = (E + Qm - 1) / Qm;
  uint32_t bit_pos = start_idx;
  uint32_t byte_idx = bit_pos / 8;
  uint32_t bit_idx = bit_pos % 8;
  uint8_t *fp = &f[byte_idx];
  for (uint32_t i = 0; i < EQm; i++) {
    for (uint32_t j = 0; j < Qm; j++) {
      uint32_t e_idx = j * EQm + i;
      uint8_t value = e[e_idx] & 0x01;
      *fp |= (value << bit_idx);
      bit_idx++;
      if (bit_idx == 8) {
        bit_idx = 0;
        fp++;
      }
    }
  }
}

void nr_deinterleaving_ldpc(uint32_t E, uint8_t Qm, int16_t *e,int16_t *f)
{ 

  
  switch(Qm) {
  case 2:
    {
      AssertFatal(E%2==0,"");
      int16_t *e1=e+(E/2);
      int16_t *end=f+E-1;
      while( f<end ){
        *e++  = *f++;
        *e1++ = *f++;
      }
    }
    break;
  case 4:
    {
      AssertFatal(E%4==0,"");
      int16_t *e1=e+(E/4);
      int16_t *e2=e1+(E/4);
      int16_t *e3=e2+(E/4);
      int16_t *end=f+E-3;
      while( f<end ){ 
        *e++  = *f++;
        *e1++ = *f++;
        *e2++ = *f++;
        *e3++ = *f++;
      }
    }
    break;
  case 6:
    {
      AssertFatal(E%6==0,"");
      int16_t *e1=e+(E/6);
      int16_t *e2=e1+(E/6);
      int16_t *e3=e2+(E/6);
      int16_t *e4=e3+(E/6);
      int16_t *e5=e4+(E/6);
      int16_t *end=f+E-5;
     while( f<end ){ 
        *e++  = *f++;
        *e1++ = *f++;
        *e2++ = *f++;
        *e3++ = *f++;
        *e4++ = *f++;
        *e5++ = *f++;
      }
    }
    break;
  case 8:
    {
      AssertFatal(E%8==0,"");
      int16_t *e1=e+(E/8);
      int16_t *e2=e1+(E/8);
      int16_t *e3=e2+(E/8);
      int16_t *e4=e3+(E/8);
      int16_t *e5=e4+(E/8);
      int16_t *e6=e5+(E/8);
      int16_t *e7=e6+(E/8);
      int16_t *end=f+E-7;
      while( f<end ){
        *e++  = *f++;
        *e1++ = *f++;
        *e2++ = *f++;
        *e3++ = *f++;
        *e4++ = *f++;
        *e5++ = *f++;
        *e6++ = *f++;
        *e7++ = *f++;
      }
    }
    break;
  default:
    AssertFatal(1==0,"Should not get here : Qm %d\n",Qm);
    break;
  }

}

int nr_get_R_ldpc_decoder(int rvidx,
                          int E,
                          int BG,
                          int Z,
                          int *llrLen,
                          int round) {
  AssertFatal(BG == 1 || BG == 2, "Unknown BG %d\n", BG);

  int Ncb = (BG==1)?(66*Z):(50*Z);
  int infoBits = (index_k0[BG-1][rvidx] * Z + E);

  if (round == 0) *llrLen = infoBits;
  if (infoBits > Ncb) infoBits = Ncb;
  if (infoBits > *llrLen) *llrLen = infoBits;

  int sysBits = (BG==1)?(22*Z):(10*Z);
  float decoderR = (float)sysBits/(infoBits + 2*Z);

  if (BG == 2)
    if (decoderR < 0.3333)
      return 15;
    else if (decoderR < 0.6667)
      return 13;
    else
      return 23;
  else
    if (decoderR < 0.6667)
      return 13;
    else if (decoderR < 0.8889)
      return 23;
    else
      return 89;
}

int nr_rate_matching_ldpc(uint32_t Tbslbrm,
                          uint8_t BG,
                          uint16_t Z,
                          uint8_t *w,
                          uint8_t *e,
                          uint8_t C,
			  uint32_t F,
			  uint32_t Foffset,
                          uint8_t rvidx,
                          uint32_t E)
{
  uint32_t Ncb,ind,k=0,Nref,N;

  if (C==0) {
    LOG_E(PHY,"nr_rate_matching: invalid parameters (C %d\n",C);
    return -1;
  }

  //Bit selection
  N = (BG==1)?(66*Z):(50*Z);

  if (Tbslbrm == 0)
      Ncb = N;
  else {
      Nref = 3*Tbslbrm/(2*C); //R_LBRM = 2/3
      Ncb = min(N, Nref);
  }

  ind = (index_k0[BG-1][rvidx]*Ncb/N)*Z;

#ifdef RM_DEBUG
  printf("nr_rate_matching_ldpc: E %u, F %u, Foffset %u, k0 %u, Ncb %u, rvidx %d, Tbslbrm %u\n", E, F, Foffset, ind, Ncb, rvidx, Tbslbrm);
#endif

  if (Foffset > E) {
    LOG_E(PHY,"nr_rate_matching: invalid parameters (Foffset %d > E %d) F %d, k0 %d, Ncb %d, rvidx %d, Tbslbrm %d\n",Foffset,E,F, ind, Ncb, rvidx, Tbslbrm);
    return -1;
  }
  if (Foffset > Ncb) {
    LOG_E(PHY,"nr_rate_matching: invalid parameters (Foffset %d > Ncb %d)\n",Foffset,Ncb);
    return -1;
  }

  if (ind >= Foffset && ind < (F+Foffset)) ind = F+Foffset;

  if (ind < Foffset) { // case where we have some bits before the filler and the rest after
    memcpy((void*)e,(void*)(w+ind),Foffset-ind);

    if (E + F <= Ncb-ind) { // E+F doesn't contain all coded bits
      memcpy((void*)(e+Foffset-ind),(void*)(w+Foffset+F),E-Foffset+ind);
      k=E;
    }
    else {
      memcpy((void*)(e+Foffset-ind),(void*)(w+Foffset+F),Ncb-Foffset-F);
      k=Ncb-F-ind;
    }
  }
  else {
    if (E <= Ncb-ind) { //E+F doesn't contain all coded bits
      memcpy((void*)(e),(void*)(w+ind),E);
      k=E;
    }
    else {
      memcpy((void*)(e),(void*)(w+ind),Ncb-ind);
      k=Ncb-ind;
    }
  }

  while(k<E) { // case where we do repetitions (low mcs)
    for (ind=0; (ind<Ncb)&&(k<E); ind++) {

#ifdef RM_DEBUG
      printf("RM_TX k%u Ind: %u (%d)\n",k,ind,w[ind]);
#endif

      if (w[ind] != NR_NULL) e[k++]=w[ind];
    }
  }


  return 0;
}

int nr_rate_matching_ldpc_rx(uint32_t Tbslbrm,
                             uint8_t BG,
                             uint16_t Z,
                             int16_t *w,
                             int16_t *soft_input,
                             uint8_t C,
                             uint8_t rvidx,
                             uint8_t clear,
                             uint32_t E,
                             uint32_t F,
                             uint32_t Foffset)
{
  uint32_t Ncb,ind,k,Nref,N;

#ifdef RM_DEBUG
  int nulled=0;
#endif

  if (C==0) {
    LOG_E(PHY,"nr_rate_matching: invalid parameters (C %d\n",C);
    return -1;
  }

  //Bit selection
  N = (BG==1)?(66*Z):(50*Z);

  if (Tbslbrm == 0)
    Ncb = N;
  else {
    Nref = (3*Tbslbrm/(2*C)); //R_LBRM = 2/3
    Ncb = min(N, Nref);
  }

  ind = (index_k0[BG-1][rvidx]*Ncb/N)*Z;
  if (Foffset > E) {
    LOG_E(PHY,"nr_rate_matching: invalid parameters (Foffset %d > E %d)\n",Foffset,E);
    return -1;
  }
  if (Foffset > Ncb) {
    LOG_E(PHY,"nr_rate_matching: invalid parameters (Foffset %d > Ncb %d)\n",Foffset,Ncb);
    return -1;
  }

#ifdef RM_DEBUG
  printf("nr_rate_matching_ldpc_rx: Clear %d, E %u, Foffset %u, k0 %u, Ncb %u, rvidx %d, Tbslbrm %u\n", clear, E, Foffset, ind, Ncb, rvidx, Tbslbrm);
#endif

  if (clear == 1)
    memset(w, 0, Ncb * sizeof(int16_t));

  k=0;

  if (ind < Foffset)
    for (; (ind<Foffset)&&(k<E); ind++) {
#ifdef RM_DEBUG
      printf("RM_RX k%u Ind %u(before filler): %d (%d)=>",k,ind,w[ind],soft_input[k]);
#endif
      w[ind]+=soft_input[k++];
#ifdef RM_DEBUG
      printf("%d\n",w[ind]);
#endif
    }
  if (ind >= Foffset && ind < Foffset+F) ind=Foffset+F;

  for (; (ind<Ncb)&&(k<E); ind++) {
#ifdef RM_DEBUG
    printf("RM_RX k%u Ind %u(after filler) %d (%d)=>",k,ind,w[ind],soft_input[k]);
#endif
      w[ind] += soft_input[k++];
#ifdef RM_DEBUG
      printf("%d\n",w[ind]);
#endif
  }

  while(k<E) {
    for (ind=0; (ind<Foffset)&&(k<E); ind++) {
#ifdef RM_DEBUG
      printf("RM_RX k%u Ind %u(before filler) %d(%d)=>",k,ind,w[ind],soft_input[k]);
#endif
      w[ind]+=soft_input[k++];
#ifdef RM_DEBUG
      printf("%d\n",w[ind]);
#endif
    }
    for (ind=Foffset+F; (ind<Ncb)&&(k<E); ind++) {
#ifdef RM_DEBUG
      printf("RM_RX (after filler) k%u Ind: %u (%d)(soft in %d)=>",k,ind,w[ind],soft_input[k]);
#endif
      w[ind] += soft_input[k++];
#ifdef RM_DEBUG
      printf("%d\n",w[ind]);
#endif
    }
  }

  return 0;
}

