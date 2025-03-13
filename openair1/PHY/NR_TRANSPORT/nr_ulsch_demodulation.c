#include "PHY/defs_gNB.h"
#include "PHY/phy_extern.h"
#include "nr_transport_proto.h"
#include "PHY/impl_defs_top.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "PHY/defs_nr_common.h"
#include "common/utils/nr/nr_common.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "executables/nr-uesoftmodem.h"
#include "SCHED_NR_UE/defs.h"
#include "openair1/PHY/MODULATION/modulation_UE.h"

//#define DEBUG_CH_COMP
//#define DEBUG_RB_EXT
//#define DEBUG_CH_MAG
//#define ML_DEBUG

#define INVALID_VALUE 255

void nr_idft(int32_t *z, uint32_t Msc_PUSCH) {


#if defined(__x86_64__) || defined(__i386__)
  __m128i idft_in128[1][3240], idft_out128[1][3240];
  __m128i norm128;
#elif defined(__arm__) || defined(__aarch64__)
  int16x8_t idft_in128[1][3240], idft_out128[1][3240];
  int16x8_t norm128;
#endif
  int16_t *idft_in0 = (int16_t*)idft_in128[0], *idft_out0 = (int16_t*)idft_out128[0];

  int i, ip;

  LOG_T(PHY,"Doing lte_idft for Msc_PUSCH %d\n",Msc_PUSCH);

  if ((Msc_PUSCH % 1536) > 0) {
    // conjugate input
    for (i = 0; i < (Msc_PUSCH>>2); i++) {
#if defined(__x86_64__)||defined(__i386__)
      *&(((__m128i*)z)[i]) = _mm_sign_epi16(*&(((__m128i*)z)[i]), *(__m128i*)&conjugate2[0]);
#elif defined(__arm__) || defined(__aarch64__)
      *&(((int16x8_t*)z)[i]) = vmulq_s16(*&(((int16x8_t*)z)[i]), *(int16x8_t*)&conjugate2[0]);
#endif
    }
    for (i = 0, ip = 0; i < Msc_PUSCH; i++, ip+=4)
      ((uint32_t*)idft_in0)[ip+0] = z[i];
  }

  switch (Msc_PUSCH) {
    case 12:
      dft(DFT_12,(int16_t *)idft_in0, (int16_t *)idft_out0,0);

#if defined(__x86_64__)||defined(__i386__)
      norm128 = _mm_set1_epi16(9459);
#elif defined(__arm__) || defined(__aarch64__)
      norm128 = vdupq_n_s16(9459);
#endif

      for (i = 0; i < 12; i++) {
#if defined(__x86_64__)||defined(__i386__)
        ((__m128i*)idft_out0)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)idft_out0)[i], norm128), 1);
#elif defined(__arm__) || defined(__aarch64__)
        ((int16x8_t*)idft_out0)[i] = vqdmulhq_s16(((int16x8_t*)idft_out0)[i], norm128);
#endif
      }

      break;

    case 24:
      dft(DFT_24,idft_in0, idft_out0, 1);
      break;

    case 36:
      dft(DFT_36,idft_in0, idft_out0, 1);
      break;

    case 48:
      dft(DFT_48,idft_in0, idft_out0, 1);
      break;

    case 60:
      dft(DFT_60,idft_in0, idft_out0, 1);
      break;

    case 72:
      dft(DFT_72,idft_in0, idft_out0, 1);
      break;

    case 96:
      dft(DFT_96,idft_in0, idft_out0, 1);
      break;

    case 108:
      dft(DFT_108,idft_in0, idft_out0, 1);
      break;

    case 120:
      dft(DFT_120,idft_in0, idft_out0, 1);
      break;

    case 144:
      dft(DFT_144,idft_in0, idft_out0, 1);
      break;

    case 180:
      dft(DFT_180,idft_in0, idft_out0, 1);
      break;

    case 192:
      dft(DFT_192,idft_in0, idft_out0, 1);
      break;

    case 216:
      dft(DFT_216,idft_in0, idft_out0, 1);
      break;

    case 240:
      dft(DFT_240,idft_in0, idft_out0, 1);
      break;

    case 288:
      dft(DFT_288,idft_in0, idft_out0, 1);
      break;

    case 300:
      dft(DFT_300,idft_in0, idft_out0, 1);
      break;

    case 324:
      dft(DFT_324,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 360:
      dft(DFT_360,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 384:
      dft(DFT_384,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 432:
      dft(DFT_432,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 480:
      dft(DFT_480,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 540:
      dft(DFT_540,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 576:
      dft(DFT_576,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 600:
      dft(DFT_600,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 648:
      dft(DFT_648,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 720:
      dft(DFT_720,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 768:
      dft(DFT_768,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 864:
      dft(DFT_864,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 900:
      dft(DFT_900,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 960:
      dft(DFT_960,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 972:
      dft(DFT_972,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1080:
      dft(DFT_1080,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1152:
      dft(DFT_1152,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1200:
      dft(DFT_1200,idft_in0, idft_out0, 1);
      break;

    case 1296:
      dft(DFT_1296,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1440:
      dft(DFT_1440,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1500:
      dft(DFT_1500,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1536:
      //dft(DFT_1536,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      idft(IDFT_1536,(int16_t*)z, (int16_t*)z, 1);
      break;

    case 1620:
      dft(DFT_1620,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1728:
      dft(DFT_1728,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1800:
      dft(DFT_1800,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1920:
      dft(DFT_1920,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 1944:
      dft(DFT_1944,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 2160:
      dft(DFT_2160,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 2304:
      dft(DFT_2304,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 2400:
      dft(DFT_2400,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 2592:
      dft(DFT_2592,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 2700:
      dft(DFT_2700,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 2880:
      dft(DFT_2880,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 2916:
      dft(DFT_2916,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 3000:
      dft(DFT_3000,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    case 3072:
      //dft(DFT_3072,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      idft(IDFT_3072,(int16_t*)z, (int16_t*)z, 1);
      break;

    case 3240:
      dft(DFT_3240,(int16_t*)idft_in0, (int16_t*)idft_out0, 1);
      break;

    default:
      // should not be reached
      LOG_E( PHY, "Unsupported Msc_PUSCH value of %"PRIu16"\n", Msc_PUSCH );
      return;
  }

  if ((Msc_PUSCH % 1536) > 0) {
    for (i = 0, ip = 0; i < Msc_PUSCH; i++, ip+=4)
      z[i] = ((uint32_t*)idft_out0)[ip];

    // conjugate output
    for (i = 0; i < (Msc_PUSCH>>2); i++) {
#if defined(__x86_64__) || defined(__i386__)
      ((__m128i*)z)[i] = _mm_sign_epi16(((__m128i*)z)[i], *(__m128i*)&conjugate2[0]);
#elif defined(__arm__) || defined(__aarch64__)
      *&(((int16x8_t*)z)[i]) = vmulq_s16(*&(((int16x8_t*)z)[i]), *(int16x8_t*)&conjugate2[0]);
#endif
    }
  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif

}


void nr_ulsch_extract_rbs(int rxFSz,
                          c16_t rxdataF[][rxFSz],
                          NR_gNB_PUSCH *pusch_vars,
                          int slot,
                          unsigned char symbol,
                          uint8_t is_dmrs_symbol,
                          uint8_t is_csirs_symbol,
                          uint32_t bwp_start,
                          uint32_t rb_start,
                          uint32_t rb_size,
                          uint32_t nrOfLayers,
                          uint32_t num_dmrs_cdm_grps_no_data,
                          uint32_t dmrs_config_type,
                          NR_DL_FRAME_PARMS *frame_parms,
                          nfapi_nr_dl_tti_csi_rs_pdu_rel15_t *csi_params) {
 
  unsigned short start_re, re, nb_re_pusch;
  unsigned char aarx, aatx;
  uint32_t rxF_ext_index = 0;
  uint32_t ul_ch0_ext_index = 0;
  uint32_t ul_ch0_index = 0;
  int16_t *rxF,*rxF_ext;
  int *ul_ch0,*ul_ch0_ext;
  int soffset = 0; /*(slot&3)*frame_parms->symbols_per_slot*frame_parms->ofdm_symbol_size;*/

#ifdef DEBUG_RB_EXT
  printf("--------------------symbol = %d-----------------------\n", symbol);
  printf("--------------------ch_ext_index = %d-----------------------\n", symbol*NR_NB_SC_PER_RB * rb_size);
#endif

  uint8_t is_data_re;
  start_re = (frame_parms->first_carrier_offset + (rb_start + bwp_start) * NR_NB_SC_PER_RB)%frame_parms->ofdm_symbol_size;
  nb_re_pusch = NR_NB_SC_PER_RB * rb_size;

  int nb_re_pusch2 = nb_re_pusch + (nb_re_pusch&7);

  for (aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {

    rxF = (int16_t *)&rxdataF[aarx][soffset+(symbol * frame_parms->ofdm_symbol_size)];
    rxF_ext = (int16_t *)&pusch_vars->rxdataF_ext[aarx][symbol * nb_re_pusch2]; // [hna] rxdataF_ext isn't contiguous in order to solve an alignment problem ib llr computation in case of mod_order = 4, 6
    AssertFatal(soffset + (symbol * frame_parms->ofdm_symbol_size) + start_re < rxFSz, "rxF offset is greater than the buffer size\n");
    AssertFatal(symbol * nb_re_pusch2 + nb_re_pusch < nb_re_pusch2 * frame_parms->symbols_per_slot, "Copied PUSCH data is more than rxF_ext size\n");
    LOG_D(NR_PHY,"symbol %d : rxF energy %d\n",symbol,dB_fixed(signal_energy_nodc((int32_t*)rxF,frame_parms->ofdm_symbol_size))); 
    if (is_dmrs_symbol == 0) {
      if (is_csirs_symbol == 0) {
        if (start_re + nb_re_pusch <= frame_parms->ofdm_symbol_size) {
          memcpy1((void*)rxF_ext, (void*)&rxF[start_re*2], nb_re_pusch*sizeof(int32_t));
        } else {
          int neg_length = frame_parms->ofdm_symbol_size-start_re;
          int pos_length = nb_re_pusch-neg_length;
          memcpy1((void*)rxF_ext, (void*)&rxF[start_re*2], neg_length*sizeof(int32_t));
          memcpy1((void*)&rxF_ext[2*neg_length], (void*)rxF, pos_length*sizeof(int32_t));
        }

        for (aatx = 0; aatx < nrOfLayers; aatx++) {
          ul_ch0 = &pusch_vars->ul_ch_estimates[aatx*frame_parms->nb_antennas_rx+aarx][pusch_vars->dmrs_symbol*frame_parms->ofdm_symbol_size]; // update channel estimates if new dmrs symbol are available
          ul_ch0_ext = &pusch_vars->ul_ch_estimates_ext[aatx*frame_parms->nb_antennas_rx+aarx][symbol*nb_re_pusch2];
          memcpy1((void*)ul_ch0_ext, (void*)ul_ch0,nb_re_pusch*sizeof(int32_t));
        }
      } else {
        int16_t csi_rs_rb = csi_params->start_rb;
        for (aatx = 0; aatx < nrOfLayers; aatx++) {
          ul_ch0 = &pusch_vars->ul_ch_estimates[aatx*frame_parms->nb_antennas_rx+aarx][pusch_vars->dmrs_symbol*frame_parms->ofdm_symbol_size]; // update channel estimates if new dmrs symbol are available
          ul_ch0_ext = &pusch_vars->ul_ch_estimates_ext[aatx*frame_parms->nb_antennas_rx+aarx][symbol*nb_re_pusch2];

          rxF_ext_index = 0;
          ul_ch0_ext_index = 0;
          ul_ch0_index = 0;
          for (re = 0; re < nb_re_pusch; re++) {
            uint8_t is_csi_rs = 0;
            uint16_t k = start_re + re;
            if ((k >= csi_params->start_rb * NR_NB_SC_PER_RB) && (re % NR_NB_SC_PER_RB == 0) && (csi_rs_rb < csi_params->nr_of_rbs)) {
              csi_rs_params_t table_params;
              get_csi_rs_params_from_table(csi_params, &table_params);
              port_freq_indices_t *port_freq_indices = (port_freq_indices_t *)malloc(table_params.ports*sizeof(port_freq_indices));
              get_csi_rs_freq_ind_sl(frame_parms, csi_rs_rb, csi_params, &table_params, port_freq_indices);
              if (k == port_freq_indices[aatx].k) {
                is_csi_rs = 1;
                csi_rs_rb++;
              }
              free(port_freq_indices);
              port_freq_indices = NULL;
            }

            if (++k >= frame_parms->ofdm_symbol_size) {
              k -= frame_parms->ofdm_symbol_size;
            }

            // save only data and respective channel estimates
            if (is_csi_rs == 0) {
              if (aatx == 0) {
                rxF_ext[rxF_ext_index]     = (rxF[ ((start_re + re)*2)      % (frame_parms->ofdm_symbol_size*2)]);
                rxF_ext[rxF_ext_index + 1] = (rxF[(((start_re + re)*2) + 1) % (frame_parms->ofdm_symbol_size*2)]);
                rxF_ext_index +=2;
              }

              ul_ch0_ext[ul_ch0_ext_index] = ul_ch0[ul_ch0_index];
              ul_ch0_ext_index++;

            }
            ul_ch0_index++;
          }
        }
      }
    } else {

      for (aatx = 0; aatx < nrOfLayers; aatx++) {
        ul_ch0 = &pusch_vars->ul_ch_estimates[aatx*frame_parms->nb_antennas_rx+aarx][pusch_vars->dmrs_symbol*frame_parms->ofdm_symbol_size]; // update channel estimates if new dmrs symbol are available
        ul_ch0_ext = &pusch_vars->ul_ch_estimates_ext[aatx*frame_parms->nb_antennas_rx+aarx][symbol*nb_re_pusch2];

        rxF_ext_index = 0;
        ul_ch0_ext_index = 0;
        ul_ch0_index = 0;
        for (re = 0; re < nb_re_pusch; re++) {
          uint16_t k = start_re + re;
          is_data_re = allowed_xlsch_re_in_dmrs_symbol(k, start_re, frame_parms->ofdm_symbol_size, num_dmrs_cdm_grps_no_data, dmrs_config_type);
          if (++k >= frame_parms->ofdm_symbol_size) {
            k -= frame_parms->ofdm_symbol_size;
          }

          #ifdef DEBUG_RB_EXT
          printf("re = %d, is_dmrs_symbol = %d, symbol = %d\n", re, is_dmrs_symbol, symbol);
          #endif

          // save only data and respective channel estimates
          if (is_data_re == 1) {
            if (aatx == 0) {
              rxF_ext[rxF_ext_index]     = (rxF[ ((start_re + re)*2)      % (frame_parms->ofdm_symbol_size*2)]);
              rxF_ext[rxF_ext_index + 1] = (rxF[(((start_re + re)*2) + 1) % (frame_parms->ofdm_symbol_size*2)]);
              rxF_ext_index +=2;
            }

            ul_ch0_ext[ul_ch0_ext_index] = ul_ch0[ul_ch0_index];
            ul_ch0_ext_index++;

            #ifdef DEBUG_RB_EXT
            printf("dmrs symb %d: rxF_ext[%u] = (%d,%d), ul_ch0_ext[%u] = (%d,%d)\n",
                 is_dmrs_symbol,rxF_ext_index>>1, rxF_ext[rxF_ext_index],rxF_ext[rxF_ext_index+1],
                 ul_ch0_ext_index,  ((int16_t*)&ul_ch0_ext[ul_ch0_ext_index])[0],  ((int16_t*)&ul_ch0_ext[ul_ch0_ext_index])[1]);
            #endif          
          }
          ul_ch0_index++;
        }
      }
    }
  }
}

void nr_ulsch_scale_channel(int **ul_ch_estimates_ext,
                            NR_DL_FRAME_PARMS *frame_parms,
                            NR_gNB_ULSCH_t *ulsch_gNB,
                            uint8_t symbol,
                            uint8_t is_dmrs_symbol,
                            uint32_t len,
                            uint8_t nrOfLayers,
                            unsigned short nb_rb,
                            int shift_ch_ext)
{

#if defined(__x86_64__)||defined(__i386__)

  // Determine scaling amplitude based the symbol
  int b = 3;
  short ch_amp = 1024 * 8;
  if (shift_ch_ext > 3) {
    b = 0;
    ch_amp >>= (shift_ch_ext - 3);
    if (ch_amp == 0) {
      ch_amp = 1;
    }
  } else {
    b -= shift_ch_ext;
  }
  __m128i ch_amp128 = _mm_set1_epi16(ch_amp); // Q3.13
  LOG_D(PHY, "Scaling PUSCH Chest in OFDM symbol %d by %d, pilots %d nb_rb %d NCP %d symbol %d\n", symbol, ch_amp, is_dmrs_symbol, nb_rb, frame_parms->Ncp, symbol);

  uint32_t nb_rb_0 = len / 12 + ((len % 12) ? 1 : 0);
  int off = ((nb_rb & 1) == 1) ? 4 : 0;
  for (int aatx = 0; aatx < nrOfLayers; aatx++) {
    for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
      __m128i *ul_ch128 = (__m128i *)&ul_ch_estimates_ext[aatx * frame_parms->nb_antennas_rx + aarx][symbol * (off + (nb_rb * NR_NB_SC_PER_RB))];
      for (int rb = 0; rb < nb_rb_0; rb++) {
        ul_ch128[0] = _mm_mulhi_epi16(ul_ch128[0], ch_amp128);
        ul_ch128[0] = _mm_slli_epi16(ul_ch128[0], b);

        ul_ch128[1] = _mm_mulhi_epi16(ul_ch128[1], ch_amp128);
        ul_ch128[1] = _mm_slli_epi16(ul_ch128[1], b);

        ul_ch128[2] = _mm_mulhi_epi16(ul_ch128[2], ch_amp128);
        ul_ch128[2] = _mm_slli_epi16(ul_ch128[2], b);
        ul_ch128 += 3;
      }
    }
  }
#endif
}

//compute average channel_level on each (TX,RX) antenna pair
void nr_ulsch_channel_level(int **ul_ch_estimates_ext,
                            NR_DL_FRAME_PARMS *frame_parms,
                            int32_t *avg,
                            uint8_t symbol,
                            uint32_t len,
                            uint8_t nrOfLayers,
                            unsigned short nb_rb)
{

#if defined(__x86_64__)||defined(__i386__)

  short rb;
  unsigned char aatx, aarx;
  __m128i *ul_ch128, avg128U;

  int16_t x = factor2(len);
  int16_t y = (len)>>x;
  
  uint32_t nb_rb_0 = len/12 + ((len%12)?1:0);

  int off = ((nb_rb&1) == 1)? 4:0;

  for (aatx = 0; aatx < nrOfLayers; aatx++) {
    for (aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
      //clear average level
      avg128U = _mm_setzero_si128();

      ul_ch128=(__m128i *)&ul_ch_estimates_ext[aatx*frame_parms->nb_antennas_rx+aarx][symbol*(off+(nb_rb*12))];

      for (rb = 0; rb < nb_rb_0; rb++) {
        avg128U = _mm_add_epi32(avg128U, _mm_srai_epi32(_mm_madd_epi16(ul_ch128[0], ul_ch128[0]), x));
        avg128U = _mm_add_epi32(avg128U, _mm_srai_epi32(_mm_madd_epi16(ul_ch128[1], ul_ch128[1]), x));
        avg128U = _mm_add_epi32(avg128U, _mm_srai_epi32(_mm_madd_epi16(ul_ch128[2], ul_ch128[2]), x));
        ul_ch128+=3;
      }

      avg[aatx*frame_parms->nb_antennas_rx+aarx] = (((int32_t*)&avg128U)[0] +
                                                    ((int32_t*)&avg128U)[1] +
                                                    ((int32_t*)&avg128U)[2] +
                                                    ((int32_t*)&avg128U)[3]) / y;
    }
  }

  _mm_empty();
  _m_empty();

#elif defined(__arm__) || defined(__aarch64__)

  short rb;
  unsigned char aatx, aarx, nre = 12, symbol_mod;
  int32x4_t avg128U;
  int16x4_t *ul_ch128;

  symbol_mod = (symbol>=(7-frame_parms->Ncp)) ? symbol-(7-frame_parms->Ncp) : symbol;
  uint32_t nb_rb_0 = len/12 + ((len%12)?1:0);
  for (aatx=0; aatx<nrOfLayers; aatx++) {
    for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
      //clear average level
      avg128U = vdupq_n_s32(0);
      // 5 is always a symbol with no pilots for both normal and extended prefix

      ul_ch128 = (int16x4_t *)&ul_ch_estimates_ext[aatx*frame_parms->nb_antennas_rx+aarx][symbol*frame_parms->N_RB_UL*12];

      for (rb = 0; rb < nb_rb_0; rb++) {
        //  printf("rb %d : ",rb);
        //  print_shorts("ch",&ul_ch128[0]);
        avg128U = vqaddq_s32(avg128U, vmull_s16(ul_ch128[0], ul_ch128[0]));
        avg128U = vqaddq_s32(avg128U, vmull_s16(ul_ch128[1], ul_ch128[1]));
        avg128U = vqaddq_s32(avg128U, vmull_s16(ul_ch128[2], ul_ch128[2]));
        avg128U = vqaddq_s32(avg128U, vmull_s16(ul_ch128[3], ul_ch128[3]));

        if (((symbol_mod == 0) || (symbol_mod == (frame_parms->Ncp-1)))&&(nrOfLayers!=1)) {
          ul_ch128+=4;
        } else {
          avg128U = vqaddq_s32(avg128U, vmull_s16(ul_ch128[4], ul_ch128[4]));
          avg128U = vqaddq_s32(avg128U, vmull_s16(ul_ch128[5], ul_ch128[5]));
          ul_ch128+=6;
        }

        /*
          if (rb==0) {
          print_shorts("ul_ch128",&ul_ch128[0]);
          print_shorts("ul_ch128",&ul_ch128[1]);
          print_shorts("ul_ch128",&ul_ch128[2]);
          }
        */
      }

      if (symbol==2) //assume start symbol 2
          nre=6;
      else
          nre=12;

      avg[aatx*frame_parms->nb_antennas_rx+aarx] = (((int32_t*)&avg128U)[0] +
                                                    ((int32_t*)&avg128U)[1] +
                                                    ((int32_t*)&avg128U)[2] +
                                                    ((int32_t*)&avg128U)[3]) / (nb_rb*nre);
    }
  }
#endif
}

__m128i a_mult_conjb(__m128i a, __m128i b, unsigned char output_shift)
{
  __m128i mmtmpD0 = _mm_madd_epi16(b, a);
  __m128i mmtmpD1 = _mm_shufflelo_epi16(b, _MM_SHUFFLE(2, 3, 0, 1));
  mmtmpD1 = _mm_shufflehi_epi16(mmtmpD1, _MM_SHUFFLE(2, 3, 0, 1));
  mmtmpD1 = _mm_sign_epi16(mmtmpD1, *(__m128i *)&conjugate[0]);
  mmtmpD1 = _mm_madd_epi16(mmtmpD1, a);
  mmtmpD0 = _mm_srai_epi32(mmtmpD0, output_shift);
  mmtmpD1 = _mm_srai_epi32(mmtmpD1, output_shift);
  __m128i mmtmpD2 = _mm_unpacklo_epi32(mmtmpD0, mmtmpD1);
  __m128i mmtmpD3 = _mm_unpackhi_epi32(mmtmpD0, mmtmpD1);
  return _mm_packs_epi32(mmtmpD2, mmtmpD3);
}

//==============================================================================================
// Pre-processing for LLR computation
//==============================================================================================
void nr_ulsch_channel_compensation(int **rxdataF_ext,
                                   int **ul_ch_estimates_ext,
                                   int **ul_ch_mag,
                                   int **ul_ch_magb,
                                   int **ul_ch_magc,
                                   int **rxdataF_comp,
                                   int ***rho,
                                   NR_DL_FRAME_PARMS *frame_parms,
                                   unsigned char symbol,
                                   int length,
                                   uint8_t is_dmrs_symbol,
                                   unsigned char mod_order,
                                   uint8_t  nrOfLayers,
                                   unsigned short nb_rb,
                                   unsigned char output_shift) {

  int off = ((nb_rb&1) == 1)? 4:0;

#ifdef DEBUG_CH_COMP
  int16_t *rxF, *ul_ch;
  int prnt_idx;
  for (int nl=0; nl<nrOfLayers; nl++) {
    for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
      rxF = (int16_t *) &rxdataF_ext[aarx][symbol * (off + (nb_rb * 12))];
      ul_ch = (int16_t *) &ul_ch_estimates_ext[nl * frame_parms->nb_antennas_rx + aarx][symbol * (off + (nb_rb * 12))];

      LOG_I(NR_PHY,"--------symbol = %d, mod_order = %d, output_shift = %d, layer %i, antenna rx = %d -----------\n",
             symbol, mod_order, output_shift, nl, aarx);
      LOG_I(NR_PHY,"----------------Before compensation------------------\n");

      for (prnt_idx = 0; prnt_idx < 12 * 5 * 2; prnt_idx += 2) {
        LOG_I(NR_PHY,"rxF[%d] = (%d,%d)\n", prnt_idx >> 1, rxF[prnt_idx], rxF[prnt_idx + 1]);
        LOG_I(NR_PHY,"ul_ch[%d] = (%d,%d)\n", prnt_idx >> 1, ul_ch[prnt_idx], ul_ch[prnt_idx + 1]);
      }
    }
  }
#endif

#ifdef DEBUG_CH_MAG
  int16_t *ch_mag;
  int print_idx;


  for (int ant=0; ant<frame_parms->nb_antennas_rx; ant++) {
    ch_mag   = (int16_t *)&ul_ch_mag[ant][symbol*(off+(nb_rb*12))];

    printf("--------------------symbol = %d, mod_order = %d-----------------------\n", symbol, mod_order);
    printf("----------------Before computation------------------\n");

    for (print_idx=0;print_idx<5;print_idx++){

      printf("ch_mag[%d] = %d\n", print_idx, ch_mag[print_idx]);

    }
  }

#endif

#if defined(__i386) || defined(__x86_64__)

  unsigned short rb;
  unsigned char aatx,aarx;
  __m128i *ul_ch128,*ul_ch128_2,*ul_ch_mag128,*ul_ch_mag128b,*ul_ch_mag128c,*rxdataF128,*rxdataF_comp128,*rho128;
  __m128i mmtmpD0,mmtmpD1,mmtmpD2,mmtmpD3,QAM_amp128={0},QAM_amp128b={0},QAM_amp128c={0};
  QAM_amp128b = _mm_setzero_si128();

  uint32_t nb_rb_0 = length/12 + ((length%12)?1:0);
  for (aatx=0; aatx<nrOfLayers; aatx++) {
    if (mod_order == 4) {
      QAM_amp128 = _mm_set1_epi16(QAM16_n1);  // 2/sqrt(10)
      QAM_amp128b = _mm_setzero_si128();
      QAM_amp128c = _mm_setzero_si128();
    }
    else if (mod_order == 6) {
      QAM_amp128  = _mm_set1_epi16(QAM64_n1); //
      QAM_amp128b = _mm_set1_epi16(QAM64_n2);
      QAM_amp128c = _mm_setzero_si128();
    }
    else if (mod_order == 8) {
      QAM_amp128  = _mm_set1_epi16(QAM256_n1); //
      QAM_amp128b = _mm_set1_epi16(QAM256_n2);
      QAM_amp128c = _mm_set1_epi16(QAM256_n3);
    }

    //    printf("comp: rxdataF_comp %p, symbol %d\n",rxdataF_comp[0],symbol);

    for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++)  {
      ul_ch128          = (__m128i *)&ul_ch_estimates_ext[aatx*frame_parms->nb_antennas_rx+aarx][symbol*(off+(nb_rb*12))];
      ul_ch_mag128      = (__m128i *)&ul_ch_mag[aatx*frame_parms->nb_antennas_rx+aarx][symbol*(off+(nb_rb*12))];
      ul_ch_mag128b     = (__m128i *)&ul_ch_magb[aatx*frame_parms->nb_antennas_rx+aarx][symbol*(off+(nb_rb*12))];
      ul_ch_mag128c     = (__m128i *)&ul_ch_magc[aatx*frame_parms->nb_antennas_rx+aarx][symbol*(off+(nb_rb*12))];
      rxdataF128        = (__m128i *)&rxdataF_ext[aarx][symbol*(off+(nb_rb*12))];
      rxdataF_comp128   = (__m128i *)&rxdataF_comp[aatx*frame_parms->nb_antennas_rx+aarx][symbol*(off+(nb_rb*12))];


      for (rb=0; rb<nb_rb_0; rb++) {
        if (mod_order>2) {
          // get channel amplitude if not QPSK

          //print_shorts("ch:",(int16_t*)&ul_ch128[0]);

          mmtmpD0 = _mm_madd_epi16(ul_ch128[0],ul_ch128[0]);
          mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift);

          mmtmpD1 = _mm_madd_epi16(ul_ch128[1],ul_ch128[1]);
          mmtmpD1 = _mm_srai_epi32(mmtmpD1,output_shift);

          mmtmpD0 = _mm_packs_epi32(mmtmpD0,mmtmpD1);

          // store channel magnitude here in a new field of ulsch

          ul_ch_mag128[0] = _mm_unpacklo_epi16(mmtmpD0,mmtmpD0);
          ul_ch_mag128b[0] = ul_ch_mag128[0];
          ul_ch_mag128c[0] = ul_ch_mag128[0];
          ul_ch_mag128[0] = _mm_mulhrs_epi16(ul_ch_mag128[0],QAM_amp128);
          ul_ch_mag128b[0] = _mm_mulhrs_epi16(ul_ch_mag128b[0],QAM_amp128b);
          ul_ch_mag128c[0] = _mm_mulhrs_epi16(ul_ch_mag128c[0],QAM_amp128c);
          // print_ints("ch: = ",(int32_t*)&mmtmpD0);
          // print_shorts("QAM_amp:",(int16_t*)&QAM_amp128);
          // print_shorts("mag:",(int16_t*)&ul_ch_mag128[0]);

          ul_ch_mag128[1]  = _mm_unpackhi_epi16(mmtmpD0,mmtmpD0);
          ul_ch_mag128b[1] = ul_ch_mag128[1];
          ul_ch_mag128c[1] = ul_ch_mag128[1];
          ul_ch_mag128[1]  = _mm_mulhrs_epi16(ul_ch_mag128[1],QAM_amp128);
          ul_ch_mag128b[1] = _mm_mulhrs_epi16(ul_ch_mag128b[1],QAM_amp128b);
          ul_ch_mag128c[1] = _mm_mulhrs_epi16(ul_ch_mag128c[1],QAM_amp128c);

          mmtmpD0 = _mm_madd_epi16(ul_ch128[2],ul_ch128[2]);
          mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift);
          mmtmpD1 = _mm_packs_epi32(mmtmpD0,mmtmpD0);

          ul_ch_mag128[2]  = _mm_unpacklo_epi16(mmtmpD1,mmtmpD1);
          ul_ch_mag128b[2] = ul_ch_mag128[2];
          ul_ch_mag128c[2] = ul_ch_mag128[2];

          ul_ch_mag128[2]  = _mm_mulhrs_epi16(ul_ch_mag128[2],QAM_amp128);
          ul_ch_mag128b[2] = _mm_mulhrs_epi16(ul_ch_mag128b[2],QAM_amp128b);
          ul_ch_mag128c[2] = _mm_mulhrs_epi16(ul_ch_mag128c[2],QAM_amp128c);
        }

        // Multiply received data by conjugated channel
        rxdataF_comp128[0] = a_mult_conjb(rxdataF128[0], ul_ch128[0], output_shift);
        rxdataF_comp128[1] = a_mult_conjb(rxdataF128[1], ul_ch128[1], output_shift);
        rxdataF_comp128[2] = a_mult_conjb(rxdataF128[2], ul_ch128[2], output_shift);

        ul_ch128 += 3;
        ul_ch_mag128 += 3;
        ul_ch_mag128b += 3;
        ul_ch_mag128c += 3;
        rxdataF128 += 3;
        rxdataF_comp128 += 3;
      }
    }
  }

  if (rho) {
    //we compute the Tx correlation matrix for each Rx antenna
    //As an example the 2x2 MIMO case requires
    //rho[aarx][nb_aatx*nb_aatx] = [cov(H_aarx_0,H_aarx_0) cov(H_aarx_0,H_aarx_1)
    //                              cov(H_aarx_1,H_aarx_0) cov(H_aarx_1,H_aarx_1)], aarx=0,...,nb_antennas_rx-1

    int avg_rho_re[frame_parms->nb_antennas_rx][nrOfLayers*nrOfLayers];
    int avg_rho_im[frame_parms->nb_antennas_rx][nrOfLayers*nrOfLayers];

    for (aarx=0; aarx < frame_parms->nb_antennas_rx; aarx++) {
      for (aatx=0; aatx < nrOfLayers; aatx++) {
        for (int atx=0; atx< nrOfLayers; atx++) {

          avg_rho_re[aarx][aatx*nrOfLayers+atx] = 0;
          avg_rho_im[aarx][aatx*nrOfLayers+atx] = 0;
          rho128        = (__m128i *)&rho[aarx][aatx*nrOfLayers+atx][symbol*(off+(nb_rb*12))];
          ul_ch128      = (__m128i *)&ul_ch_estimates_ext[aatx*frame_parms->nb_antennas_rx+aarx][symbol*(off+(nb_rb*12))];
          ul_ch128_2    = (__m128i *)&ul_ch_estimates_ext[atx*frame_parms->nb_antennas_rx+aarx][symbol*(off+(nb_rb*12))];

          for (rb=0; rb<nb_rb_0; rb++) {
            // multiply by conjugated channel
            mmtmpD0 = _mm_madd_epi16(ul_ch128[0],ul_ch128_2[0]);
            //  print_ints("re",&mmtmpD0);

            // mmtmpD0 contains real part of 4 consecutive outputs (32-bit)
            mmtmpD1 = _mm_shufflelo_epi16(ul_ch128[0],_MM_SHUFFLE(2,3,0,1));
            mmtmpD1 = _mm_shufflehi_epi16(mmtmpD1,_MM_SHUFFLE(2,3,0,1));
            mmtmpD1 = _mm_sign_epi16(mmtmpD1,*(__m128i*)&conjugate[0]);
            //  print_ints("im",&mmtmpD1);
            mmtmpD1 = _mm_madd_epi16(mmtmpD1,ul_ch128_2[0]);
            // mmtmpD1 contains imag part of 4 consecutive outputs (32-bit)
            mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift);
            //  print_ints("re(shift)",&mmtmpD0);
            mmtmpD1 = _mm_srai_epi32(mmtmpD1,output_shift);
            //  print_ints("im(shift)",&mmtmpD1);
            mmtmpD2 = _mm_unpacklo_epi32(mmtmpD0,mmtmpD1);
            mmtmpD3 = _mm_unpackhi_epi32(mmtmpD0,mmtmpD1);
            //        print_ints("c0",&mmtmpD2);
            //  print_ints("c1",&mmtmpD3);
            rho128[0] = _mm_packs_epi32(mmtmpD2,mmtmpD3);

            //print_shorts("rx:",ul_ch128_2);
            //print_shorts("ch:",ul_ch128);
            //print_shorts("pack:",rho128);

            avg_rho_re[aarx][aatx*nrOfLayers+atx] +=(((int16_t*)&rho128[0])[0]+
              ((int16_t*)&rho128[0])[2] +
              ((int16_t*)&rho128[0])[4] +
              ((int16_t*)&rho128[0])[6])/16;//

            avg_rho_im[aarx][aatx*nrOfLayers+atx] +=(((int16_t*)&rho128[0])[1]+
              ((int16_t*)&rho128[0])[3] +
              ((int16_t*)&rho128[0])[5] +
              ((int16_t*)&rho128[0])[7])/16;//
            // multiply by conjugated channel
            mmtmpD0 = _mm_madd_epi16(ul_ch128[1],ul_ch128_2[1]);
            // mmtmpD0 contains real part of 4 consecutive outputs (32-bit)
            mmtmpD1 = _mm_shufflelo_epi16(ul_ch128[1],_MM_SHUFFLE(2,3,0,1));
            mmtmpD1 = _mm_shufflehi_epi16(mmtmpD1,_MM_SHUFFLE(2,3,0,1));
            mmtmpD1 = _mm_sign_epi16(mmtmpD1,*(__m128i*)conjugate);
            mmtmpD1 = _mm_madd_epi16(mmtmpD1,ul_ch128_2[1]);
            // mmtmpD1 contains imag part of 4 consecutive outputs (32-bit)
            mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift);
            mmtmpD1 = _mm_srai_epi32(mmtmpD1,output_shift);
            mmtmpD2 = _mm_unpacklo_epi32(mmtmpD0,mmtmpD1);
            mmtmpD3 = _mm_unpackhi_epi32(mmtmpD0,mmtmpD1);
            rho128[1] =_mm_packs_epi32(mmtmpD2,mmtmpD3);
            //print_shorts("rx:",ul_ch128_2+1);
            //print_shorts("ch:",ul_ch128+1);
            //print_shorts("pack:",rho128+1);

            // multiply by conjugated channel
            avg_rho_re[aarx][aatx*nrOfLayers+atx] +=(((int16_t*)&rho128[1])[0]+
              ((int16_t*)&rho128[1])[2] +
              ((int16_t*)&rho128[1])[4] +
              ((int16_t*)&rho128[1])[6])/16;

            avg_rho_im[aarx][aatx*nrOfLayers+atx] +=(((int16_t*)&rho128[1])[1]+
              ((int16_t*)&rho128[1])[3] +
              ((int16_t*)&rho128[1])[5] +
              ((int16_t*)&rho128[1])[7])/16;

            mmtmpD0 = _mm_madd_epi16(ul_ch128[2],ul_ch128_2[2]);
            // mmtmpD0 contains real part of 4 consecutive outputs (32-bit)
            mmtmpD1 = _mm_shufflelo_epi16(ul_ch128[2],_MM_SHUFFLE(2,3,0,1));
            mmtmpD1 = _mm_shufflehi_epi16(mmtmpD1,_MM_SHUFFLE(2,3,0,1));
            mmtmpD1 = _mm_sign_epi16(mmtmpD1,*(__m128i*)conjugate);
            mmtmpD1 = _mm_madd_epi16(mmtmpD1,ul_ch128_2[2]);
            // mmtmpD1 contains imag part of 4 consecutive outputs (32-bit)
            mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift);
            mmtmpD1 = _mm_srai_epi32(mmtmpD1,output_shift);
            mmtmpD2 = _mm_unpacklo_epi32(mmtmpD0,mmtmpD1);
            mmtmpD3 = _mm_unpackhi_epi32(mmtmpD0,mmtmpD1);

            rho128[2] = _mm_packs_epi32(mmtmpD2,mmtmpD3);
            //print_shorts("rx:",ul_ch128_2+2);
            //print_shorts("ch:",ul_ch128+2);
            //print_shorts("pack:",rho128+2);
            avg_rho_re[aarx][aatx*nrOfLayers+atx] +=(((int16_t*)&rho128[2])[0]+
              ((int16_t*)&rho128[2])[2] +
              ((int16_t*)&rho128[2])[4] +
              ((int16_t*)&rho128[2])[6])/16;

            avg_rho_im[aarx][aatx*nrOfLayers+atx] +=(((int16_t*)&rho128[2])[1]+
              ((int16_t*)&rho128[2])[3] +
              ((int16_t*)&rho128[2])[5] +
              ((int16_t*)&rho128[2])[7])/16;

            ul_ch128+=3;
            ul_ch128_2+=3;
            rho128+=3;
          }
          if (is_dmrs_symbol==1) {
            //measurements->rx_correlation[0][0][aarx] = signal_energy(&rho[aarx][aatx*nb_aatx+atx][symbol*nb_rb*12],rb*12);
            avg_rho_re[aarx][aatx*nrOfLayers+atx] = 16*avg_rho_re[aarx][aatx*nrOfLayers+atx]/(nb_rb*12);
            avg_rho_im[aarx][aatx*nrOfLayers+atx] = 16*avg_rho_im[aarx][aatx*nrOfLayers+atx]/(nb_rb*12);
            //printf("rho[rx]%d tx%d tx%d = Re: %d Im: %d\n",aarx, aatx,atx, avg_rho_re[aarx][aatx*nb_aatx+atx], avg_rho_im[aarx][aatx*nb_aatx+atx]);
          }
        }
      }
    }
  }

  _mm_empty();
  _m_empty();

#elif defined(__arm__) || defined(__aarch64__)

  unsigned short rb;
  unsigned char aatx,aarx,symbol_mod,is_dmrs_symbol=0;

  int16x4_t *ul_ch128,*ul_ch128_2,*rxdataF128;
  int32x4_t mmtmpD0,mmtmpD1,mmtmpD0b,mmtmpD1b;
  int16x8_t *ul_ch_mag128,*ul_ch_mag128b,mmtmpD2,mmtmpD3,mmtmpD4;
  int16x8_t QAM_amp128,QAM_amp128b;
  int16x4x2_t *rxdataF_comp128,*rho128;

  int16_t conj[4]__attribute__((aligned(16))) = {1,-1,1,-1};
  int32x4_t output_shift128 = vmovq_n_s32(-(int32_t)output_shift);

  symbol_mod = (symbol>=(7-frame_parms->Ncp)) ? symbol-(7-frame_parms->Ncp) : symbol;

  if ((symbol_mod == 0) || (symbol_mod == (4-frame_parms->Ncp))) {
    if (nrOfLayers==1) { // 10 out of 12 so don't reduce size
      nb_rb=1+(5*nb_rb/6);
    }
    else {
      is_dmrs_symbol=1;
    }
  }

  for (aatx=0; aatx<nrOfLayers; aatx++) {
    if (mod_order == 4) {
      QAM_amp128  = vmovq_n_s16(QAM16_n1);  // 2/sqrt(10)
      QAM_amp128b = vmovq_n_s16(0);
    } else if (mod_order == 6) {
      QAM_amp128  = vmovq_n_s16(QAM64_n1); //
      QAM_amp128b = vmovq_n_s16(QAM64_n2);
    }
    //    printf("comp: rxdataF_comp %p, symbol %d\n",rxdataF_comp[0],symbol);

    for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
      ul_ch128          = (int16x4_t*)&ul_ch_estimates_ext[aatx*frame_parms->nb_antennas_rx+aarx][symbol*frame_parms->N_RB_UL*12];
      ul_ch_mag128      = (int16x8_t*)&ul_ch_mag[aatx*frame_parms->nb_antennas_rx+aarx][symbol*frame_parms->N_RB_UL*12];
      ul_ch_mag128b     = (int16x8_t*)&ul_ch_magb[aatx*frame_parms->nb_antennas_rx+aarx][symbol*frame_parms->N_RB_UL*12];
      rxdataF128        = (int16x4_t*)&rxdataF_ext[aarx][symbol*frame_parms->N_RB_UL*12];
      rxdataF_comp128   = (int16x4x2_t*)&rxdataF_comp[aatx*frame_parms->nb_antennas_rx+aarx][symbol*frame_parms->N_RB_UL*12];

      for (rb=0; rb<nb_rb; rb++) {
  if (mod_order>2) {
    // get channel amplitude if not QPSK
    mmtmpD0 = vmull_s16(ul_ch128[0], ul_ch128[0]);
    // mmtmpD0 = [ch0*ch0,ch1*ch1,ch2*ch2,ch3*ch3];
    mmtmpD0 = vqshlq_s32(vqaddq_s32(mmtmpD0,vrev64q_s32(mmtmpD0)),output_shift128);
    // mmtmpD0 = [ch0*ch0 + ch1*ch1,ch0*ch0 + ch1*ch1,ch2*ch2 + ch3*ch3,ch2*ch2 + ch3*ch3]>>output_shift128 on 32-bits
    mmtmpD1 = vmull_s16(ul_ch128[1], ul_ch128[1]);
    mmtmpD1 = vqshlq_s32(vqaddq_s32(mmtmpD1,vrev64q_s32(mmtmpD1)),output_shift128);
    mmtmpD2 = vcombine_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));
    // mmtmpD2 = [ch0*ch0 + ch1*ch1,ch0*ch0 + ch1*ch1,ch2*ch2 + ch3*ch3,ch2*ch2 + ch3*ch3,ch4*ch4 + ch5*ch5,ch4*ch4 + ch5*ch5,ch6*ch6 + ch7*ch7,ch6*ch6 + ch7*ch7]>>output_shift128 on 16-bits
    mmtmpD0 = vmull_s16(ul_ch128[2], ul_ch128[2]);
    mmtmpD0 = vqshlq_s32(vqaddq_s32(mmtmpD0,vrev64q_s32(mmtmpD0)),output_shift128);
    mmtmpD1 = vmull_s16(ul_ch128[3], ul_ch128[3]);
    mmtmpD1 = vqshlq_s32(vqaddq_s32(mmtmpD1,vrev64q_s32(mmtmpD1)),output_shift128);
    mmtmpD3 = vcombine_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));
    if (is_dmrs_symbol==0) {
      mmtmpD0 = vmull_s16(ul_ch128[4], ul_ch128[4]);
      mmtmpD0 = vqshlq_s32(vqaddq_s32(mmtmpD0,vrev64q_s32(mmtmpD0)),output_shift128);
      mmtmpD1 = vmull_s16(ul_ch128[5], ul_ch128[5]);
      mmtmpD1 = vqshlq_s32(vqaddq_s32(mmtmpD1,vrev64q_s32(mmtmpD1)),output_shift128);
      mmtmpD4 = vcombine_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));
    }

    ul_ch_mag128b[0] = vqdmulhq_s16(mmtmpD2,QAM_amp128b);
    ul_ch_mag128b[1] = vqdmulhq_s16(mmtmpD3,QAM_amp128b);
    ul_ch_mag128[0] = vqdmulhq_s16(mmtmpD2,QAM_amp128);
    ul_ch_mag128[1] = vqdmulhq_s16(mmtmpD3,QAM_amp128);

    if (is_dmrs_symbol==0) {
      ul_ch_mag128b[2] = vqdmulhq_s16(mmtmpD4,QAM_amp128b);
      ul_ch_mag128[2]  = vqdmulhq_s16(mmtmpD4,QAM_amp128);
    }
  }

  mmtmpD0 = vmull_s16(ul_ch128[0], rxdataF128[0]);
  //mmtmpD0 = [Re(ch[0])Re(rx[0]) Im(ch[0])Im(ch[0]) Re(ch[1])Re(rx[1]) Im(ch[1])Im(ch[1])]
  mmtmpD1 = vmull_s16(ul_ch128[1], rxdataF128[1]);
  //mmtmpD1 = [Re(ch[2])Re(rx[2]) Im(ch[2])Im(ch[2]) Re(ch[3])Re(rx[3]) Im(ch[3])Im(ch[3])]
  mmtmpD0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0),vget_high_s32(mmtmpD0)),
             vpadd_s32(vget_low_s32(mmtmpD1),vget_high_s32(mmtmpD1)));
  //mmtmpD0 = [Re(ch[0])Re(rx[0])+Im(ch[0])Im(ch[0]) Re(ch[1])Re(rx[1])+Im(ch[1])Im(ch[1]) Re(ch[2])Re(rx[2])+Im(ch[2])Im(ch[2]) Re(ch[3])Re(rx[3])+Im(ch[3])Im(ch[3])]

  mmtmpD0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[0],*(int16x4_t*)conj)), rxdataF128[0]);
  //mmtmpD0 = [-Im(ch[0])Re(rx[0]) Re(ch[0])Im(rx[0]) -Im(ch[1])Re(rx[1]) Re(ch[1])Im(rx[1])]
  mmtmpD1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[1],*(int16x4_t*)conj)), rxdataF128[1]);
  //mmtmpD0 = [-Im(ch[2])Re(rx[2]) Re(ch[2])Im(rx[2]) -Im(ch[3])Re(rx[3]) Re(ch[3])Im(rx[3])]
  mmtmpD1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0b),vget_high_s32(mmtmpD0b)),
             vpadd_s32(vget_low_s32(mmtmpD1b),vget_high_s32(mmtmpD1b)));
  //mmtmpD1 = [-Im(ch[0])Re(rx[0])+Re(ch[0])Im(rx[0]) -Im(ch[1])Re(rx[1])+Re(ch[1])Im(rx[1]) -Im(ch[2])Re(rx[2])+Re(ch[2])Im(rx[2]) -Im(ch[3])Re(rx[3])+Re(ch[3])Im(rx[3])]

  mmtmpD0 = vqshlq_s32(mmtmpD0,output_shift128);
  mmtmpD1 = vqshlq_s32(mmtmpD1,output_shift128);
  rxdataF_comp128[0] = vzip_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));
  mmtmpD0 = vmull_s16(ul_ch128[2], rxdataF128[2]);
  mmtmpD1 = vmull_s16(ul_ch128[3], rxdataF128[3]);
  mmtmpD0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0),vget_high_s32(mmtmpD0)),
             vpadd_s32(vget_low_s32(mmtmpD1),vget_high_s32(mmtmpD1)));
  mmtmpD0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[2],*(int16x4_t*)conj)), rxdataF128[2]);
  mmtmpD1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[3],*(int16x4_t*)conj)), rxdataF128[3]);
  mmtmpD1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0b),vget_high_s32(mmtmpD0b)),
             vpadd_s32(vget_low_s32(mmtmpD1b),vget_high_s32(mmtmpD1b)));
  mmtmpD0 = vqshlq_s32(mmtmpD0,output_shift128);
  mmtmpD1 = vqshlq_s32(mmtmpD1,output_shift128);
  rxdataF_comp128[1] = vzip_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));

  if (is_dmrs_symbol==0) {
    mmtmpD0 = vmull_s16(ul_ch128[4], rxdataF128[4]);
    mmtmpD1 = vmull_s16(ul_ch128[5], rxdataF128[5]);
    mmtmpD0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0),vget_high_s32(mmtmpD0)),
         vpadd_s32(vget_low_s32(mmtmpD1),vget_high_s32(mmtmpD1)));

    mmtmpD0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[4],*(int16x4_t*)conj)), rxdataF128[4]);
    mmtmpD1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[5],*(int16x4_t*)conj)), rxdataF128[5]);
    mmtmpD1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0b),vget_high_s32(mmtmpD0b)),
         vpadd_s32(vget_low_s32(mmtmpD1b),vget_high_s32(mmtmpD1b)));


    mmtmpD0 = vqshlq_s32(mmtmpD0,output_shift128);
    mmtmpD1 = vqshlq_s32(mmtmpD1,output_shift128);
    rxdataF_comp128[2] = vzip_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));


    ul_ch128+=6;
    ul_ch_mag128+=3;
    ul_ch_mag128b+=3;
    rxdataF128+=6;
    rxdataF_comp128+=3;

  } else { // we have a smaller PUSCH in symbols with pilots so skip last group of 4 REs and increment less
    ul_ch128+=4;
    ul_ch_mag128+=2;
    ul_ch_mag128b+=2;
    rxdataF128+=4;
    rxdataF_comp128+=2;
  }
      }
    }
  }

  if (rho) {
    for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
      rho128        = (int16x4x2_t*)&rho[aarx][symbol*frame_parms->N_RB_UL*12];
      ul_ch128      = (int16x4_t*)&ul_ch_estimates_ext[aarx][symbol*frame_parms->N_RB_UL*12];
      ul_ch128_2    = (int16x4_t*)&ul_ch_estimates_ext[2+aarx][symbol*frame_parms->N_RB_UL*12];
      for (rb=0; rb<nb_rb; rb++) {
  mmtmpD0 = vmull_s16(ul_ch128[0], ul_ch128_2[0]);
  mmtmpD1 = vmull_s16(ul_ch128[1], ul_ch128_2[1]);
  mmtmpD0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0),vget_high_s32(mmtmpD0)),
             vpadd_s32(vget_low_s32(mmtmpD1),vget_high_s32(mmtmpD1)));
  mmtmpD0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[0],*(int16x4_t*)conj)), ul_ch128_2[0]);
  mmtmpD1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[1],*(int16x4_t*)conj)), ul_ch128_2[1]);
  mmtmpD1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0b),vget_high_s32(mmtmpD0b)),
             vpadd_s32(vget_low_s32(mmtmpD1b),vget_high_s32(mmtmpD1b)));

  mmtmpD0 = vqshlq_s32(mmtmpD0,output_shift128);
  mmtmpD1 = vqshlq_s32(mmtmpD1,output_shift128);
  rho128[0] = vzip_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));

  mmtmpD0 = vmull_s16(ul_ch128[2], ul_ch128_2[2]);
  mmtmpD1 = vmull_s16(ul_ch128[3], ul_ch128_2[3]);
  mmtmpD0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0),vget_high_s32(mmtmpD0)),
             vpadd_s32(vget_low_s32(mmtmpD1),vget_high_s32(mmtmpD1)));
  mmtmpD0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[2],*(int16x4_t*)conj)), ul_ch128_2[2]);
  mmtmpD1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[3],*(int16x4_t*)conj)), ul_ch128_2[3]);
  mmtmpD1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0b),vget_high_s32(mmtmpD0b)),
             vpadd_s32(vget_low_s32(mmtmpD1b),vget_high_s32(mmtmpD1b)));

  mmtmpD0 = vqshlq_s32(mmtmpD0,output_shift128);
  mmtmpD1 = vqshlq_s32(mmtmpD1,output_shift128);
  rho128[1] = vzip_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));

  mmtmpD0 = vmull_s16(ul_ch128[0], ul_ch128_2[0]);
  mmtmpD1 = vmull_s16(ul_ch128[1], ul_ch128_2[1]);
  mmtmpD0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0),vget_high_s32(mmtmpD0)),
             vpadd_s32(vget_low_s32(mmtmpD1),vget_high_s32(mmtmpD1)));
  mmtmpD0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[4],*(int16x4_t*)conj)), ul_ch128_2[4]);
  mmtmpD1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[5],*(int16x4_t*)conj)), ul_ch128_2[5]);
  mmtmpD1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpD0b),vget_high_s32(mmtmpD0b)),
             vpadd_s32(vget_low_s32(mmtmpD1b),vget_high_s32(mmtmpD1b)));

  mmtmpD0 = vqshlq_s32(mmtmpD0,output_shift128);
  mmtmpD1 = vqshlq_s32(mmtmpD1,output_shift128);
  rho128[2] = vzip_s16(vmovn_s32(mmtmpD0),vmovn_s32(mmtmpD1));


  ul_ch128+=6;
  ul_ch128_2+=6;
  rho128+=3;
      }
    }
  }
#endif


#ifdef DEBUG_CH_COMP
  for (int nl2=0; nl2<nrOfLayers; nl2++) {
    for (int aarx2=0; aarx2<frame_parms->nb_antennas_rx; aarx2++) {
      rxF   = (int16_t *)&rxdataF_comp[nl2*frame_parms->nb_antennas_rx+aarx2][(symbol*(off+(nb_rb*12)))];

      LOG_I(NR_PHY,"--------After compansation, layer %i, antenna rx %i----------\n", nl2, aarx2);

      for (prnt_idx=0;prnt_idx<12*5*2;prnt_idx+=2){
        LOG_I(NR_PHY,"rxF[%d] = (%d,%d)\n", prnt_idx>>1, rxF[prnt_idx],rxF[prnt_idx+1]);
      }
    }
  }
#endif

#ifdef DEBUG_CH_MAG


  for (int ant=0; ant<frame_parms->nb_antennas_rx; ant++) {
    ch_mag   = (int16_t *)&ul_ch_mag[ant][(symbol*(off+(nb_rb*12)))];

    printf("----------------After computation------------------\n");

    for (print_idx=0;print_idx<12*5*2;print_idx+=2){

      printf("ch_mag[%d] = (%d,%d)\n", print_idx>>1, ch_mag[print_idx],ch_mag[print_idx+1]);

    }
  }

#endif

}

void nr_ulsch_detection_mrc(NR_DL_FRAME_PARMS *frame_parms,
                int32_t **rxdataF_comp,
                int32_t **ul_ch_mag,
                int32_t **ul_ch_magb,
                int32_t **ul_ch_magc,
                int32_t ***rho,
                uint8_t  nrOfLayers,
                uint8_t symbol,
                uint16_t nb_rb,
                int length) {
  int n_rx = frame_parms->nb_antennas_rx;
#if defined(__x86_64__) || defined(__i386__)
  __m128i *rxdataF_comp128[2],*ul_ch_mag128[2],*ul_ch_mag128b[2],*ul_ch_mag128c[2];
#elif defined(__arm__) || defined(__aarch64__)
  int16x8_t *rxdataF_comp128_0,*ul_ch_mag128_0,*ul_ch_mag128_0b;
  int16x8_t *rxdataF_comp128_1,*ul_ch_mag128_1,*ul_ch_mag128_1b;
#endif
  int32_t i;
  uint32_t nb_rb_0 = length/12 + ((length%12)?1:0);

  int off = ((nb_rb&1) == 1)? 4:0;

  if (n_rx > 1) {
    #if defined(__x86_64__) || defined(__i386__)

    int nb_re = nb_rb * 12;

    for (int aatx = 0; aatx < nrOfLayers; aatx++) {

      rxdataF_comp128[0]   = (__m128i *)&rxdataF_comp[aatx*frame_parms->nb_antennas_rx][(symbol*(nb_re + off))];
      ul_ch_mag128[0]      = (__m128i *)&ul_ch_mag[aatx*frame_parms->nb_antennas_rx][(symbol*(nb_re + off))];
      ul_ch_mag128b[0]     = (__m128i *)&ul_ch_magb[aatx*frame_parms->nb_antennas_rx][(symbol*(nb_re + off))];
      ul_ch_mag128c[0]     = (__m128i *)&ul_ch_magc[aatx*frame_parms->nb_antennas_rx][(symbol*(nb_re + off))];

      for (int aa=1;aa < n_rx;aa++) {
        rxdataF_comp128[1]   = (__m128i *)&rxdataF_comp[aatx*frame_parms->nb_antennas_rx+aa][(symbol*(nb_re + off))];
        ul_ch_mag128[1]      = (__m128i *)&ul_ch_mag[aatx*frame_parms->nb_antennas_rx+aa][(symbol*(nb_re + off))];
        ul_ch_mag128b[1]     = (__m128i *)&ul_ch_magb[aatx*frame_parms->nb_antennas_rx+aa][(symbol*(nb_re + off))];
        ul_ch_mag128c[1]     = (__m128i *)&ul_ch_magc[aatx*frame_parms->nb_antennas_rx+aa][(symbol*(nb_re + off))];

        // MRC on each re of rb, both on MF output and magnitude (for 16QAM/64QAM llr computation)
        for (i=0; i<nb_rb_0*3; i++) {
            rxdataF_comp128[0][i] = _mm_adds_epi16(rxdataF_comp128[0][i],rxdataF_comp128[1][i]);
            ul_ch_mag128[0][i]    = _mm_adds_epi16(ul_ch_mag128[0][i],ul_ch_mag128[1][i]);
            ul_ch_mag128b[0][i]   = _mm_adds_epi16(ul_ch_mag128b[0][i],ul_ch_mag128b[1][i]);
            ul_ch_mag128c[0][i]   = _mm_adds_epi16(ul_ch_mag128c[0][i],ul_ch_mag128c[1][i]);
            //rxdataF_comp128[0][i] = _mm_add_epi16(rxdataF_comp128_0[i],(*(__m128i *)&jitterc[0]));
        }
      }

      if (rho) {
        __m128i *rho128[2];
        for (int aatx2 = 0; aatx2 < nrOfLayers; aatx2++) {
          rho128[0] = (__m128i *) &rho[0][aatx * nrOfLayers + aatx2][(symbol * (nb_re + off))];
          for (int aa = 1; aa < n_rx; aa++) {
            rho128[1] = (__m128i *) &rho[aa][aatx * nrOfLayers + aatx2][(symbol * (nb_re + off))];
            for (i = 0; i < nb_rb_0 * 3; i++) {
              rho128[0][i] = _mm_adds_epi16(rho128[0][i], rho128[1][i]);
            }
          }
        }
      }

    }
    #elif defined(__arm__) || defined(__aarch64__)
    rxdataF_comp128_0   = (int16x8_t *)&rxdataF_comp[0][symbol*frame_parms->N_RB_DL*12];
    rxdataF_comp128_1   = (int16x8_t *)&rxdataF_comp[1][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_0      = (int16x8_t *)&ul_ch_mag[0][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_1      = (int16x8_t *)&ul_ch_mag[1][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_0b     = (int16x8_t *)&ul_ch_magb[0][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_1b     = (int16x8_t *)&ul_ch_magb[1][symbol*frame_parms->N_RB_DL*12];
      
    // MRC on each re of rb, both on MF output and magnitude (for 16QAM/64QAM llr computation)
    for (i=0; i<nb_rb*3; i++) {
      rxdataF_comp128_0[i] = vhaddq_s16(rxdataF_comp128_0[i],rxdataF_comp128_1[i]);
      ul_ch_mag128_0[i]    = vhaddq_s16(ul_ch_mag128_0[i],ul_ch_mag128_1[i]);
      ul_ch_mag128_0b[i]   = vhaddq_s16(ul_ch_mag128_0b[i],ul_ch_mag128_1b[i]);
      rxdataF_comp128_0[i] = vqaddq_s16(rxdataF_comp128_0[i],(*(int16x8_t *)&jitterc[0]));
    }
    #endif
  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif
}

/* Zero Forcing Rx function: nr_det_HhH()
 *
 *
 * */
void nr_ulsch_det_HhH(int32_t *after_mf_00,//a
                int32_t *after_mf_01,//b
                int32_t *after_mf_10,//c
                int32_t *after_mf_11,//d
                int32_t *det_fin,//1/ad-bc
                unsigned short nb_rb,
                unsigned char symbol,
                int32_t shift)
{
  int16_t nr_conjug2[8]__attribute__((aligned(16))) = {1,-1,1,-1,1,-1,1,-1} ;
  unsigned short rb;
  __m128i *after_mf_00_128,*after_mf_01_128, *after_mf_10_128, *after_mf_11_128, ad_re_128, bc_re_128; //ad_im_128, bc_im_128;
  __m128i *det_fin_128, det_re_128; //det_im_128, tmp_det0, tmp_det1;

  after_mf_00_128 = (__m128i *)after_mf_00;
  after_mf_01_128 = (__m128i *)after_mf_01;
  after_mf_10_128 = (__m128i *)after_mf_10;
  after_mf_11_128 = (__m128i *)after_mf_11;

  det_fin_128 = (__m128i *)det_fin;

  for (rb=0; rb<3*nb_rb; rb++) {

    //complex multiplication (I_a+jQ_a)(I_d+jQ_d) = (I_aI_d - Q_aQ_d) + j(Q_aI_d + I_aQ_d)
    //The imag part is often zero, we compute only the real part
    ad_re_128 = _mm_sign_epi16(after_mf_00_128[0],*(__m128i*)&nr_conjug2[0]);
    ad_re_128 = _mm_madd_epi16(ad_re_128,after_mf_11_128[0]); //Re: I_a0*I_d0 - Q_a1*Q_d1
    //ad_im_128 = _mm_shufflelo_epi16(after_mf_00_128[0],_MM_SHUFFLE(2,3,0,1));//permutes IQs for the low 64 bits as [I_a0 Q_a1 I_a2 Q_a3]_64bits to [Q_a1 I_a0 Q_a3 I_a2]_64bits
    //ad_im_128 = _mm_shufflehi_epi16(ad_im_128,_MM_SHUFFLE(2,3,0,1));//permutes IQs for the high 64 bits as [I_a0 Q_a1 I_a2 Q_a3]_64bits to [Q_a1 I_a0 Q_a3 I_a2]_64bits
    //ad_im_128 = _mm_madd_epi16(ad_im_128,after_mf_11_128[0]);//Im: (Q_aI_d + I_aQ_d)

    //complex multiplication (I_b+jQ_b)(I_c+jQ_c) = (I_bI_c - Q_bQ_c) + j(Q_bI_c + I_bQ_c)
    //The imag part is often zero, we compute only the real part
    bc_re_128 = _mm_sign_epi16(after_mf_01_128[0],*(__m128i*)&nr_conjug2[0]);
    bc_re_128 = _mm_madd_epi16(bc_re_128,after_mf_10_128[0]); //Re: I_b0*I_c0 - Q_b1*Q_c1
    //bc_im_128 = _mm_shufflelo_epi16(after_mf_01_128[0],_MM_SHUFFLE(2,3,0,1));//permutes IQs for the low 64 bits as [I_b0 Q_b1 I_b2 Q_b3]_64bits to [Q_b1 I_b0 Q_b3 I_b2]_64bits
    //bc_im_128 = _mm_shufflehi_epi16(bc_im_128,_MM_SHUFFLE(2,3,0,1));//permutes IQs for the high 64 bits as [I_b0 Q_b1 I_b2 Q_b3]_64bits to [Q_b1 I_b0 Q_b3 I_b2]_64bits
    //bc_im_128 = _mm_madd_epi16(bc_im_128,after_mf_10_128[0]);//Im: (Q_bI_c + I_bQ_c)

    det_re_128 = _mm_sub_epi32(ad_re_128, bc_re_128);
    //det_im_128 = _mm_sub_epi32(ad_im_128, bc_im_128);

    //det in Q30 format
    det_fin_128[0] = _mm_abs_epi32(det_re_128);


#ifdef DEBUG_DLSCH_DEMOD
     printf("\n Computing det_HhH_inv \n");
     //print_ints("det_re_128:",(int32_t*)&det_re_128);
     //print_ints("det_im_128:",(int32_t*)&det_im_128);
     print_ints("det_fin_128:",(int32_t*)&det_fin_128[0]);
#endif
    det_fin_128+=1;
    after_mf_00_128+=1;
    after_mf_01_128+=1;
    after_mf_10_128+=1;
    after_mf_11_128+=1;
  }
  _mm_empty();
  _m_empty();
}

/* Zero Forcing Rx function: nr_inv_comp_muli
 * Complex number multi: z = x*y
 *                         = (x_re*y_re - x_im*y_im) + j(x_im*y_re + x_re*y_im)
 * */
__m128i nr_ulsch_inv_comp_muli(__m128i input_x,
                         __m128i input_y)
{
  int16_t nr_conjug2[8]__attribute__((aligned(16))) = {1,-1,1,-1,1,-1,1,-1} ;

  __m128i xy_re_128, xy_im_128;
  __m128i output_z, tmp_z0, tmp_z1;

  // complex multiplication (x_re + jx_im)*(y_re + jy_im) = (x_re*y_re - x_im*y_im) + j(x_im*y_re + x_re*y_im)

  // the real part
  xy_re_128 = _mm_sign_epi16(input_x,*(__m128i*)&nr_conjug2[0]);
  xy_re_128 = _mm_madd_epi16(xy_re_128,input_y); //Re: (x_re*y_re - x_im*y_im)

  // the imag part
  xy_im_128 = _mm_shufflelo_epi16(input_x,_MM_SHUFFLE(2,3,0,1));//permutes IQs for the low 64 bits as [I_a0 Q_a1 I_a2 Q_a3]_64bits to [Q_a1 I_a0 Q_a3 I_a2]_64bits
  xy_im_128 = _mm_shufflehi_epi16(xy_im_128,_MM_SHUFFLE(2,3,0,1));//permutes IQs for the high 64 bits as [I_a0 Q_a1 I_a2 Q_a3]_64bits to [Q_a1 I_a0 Q_a3 I_a2]_64bits
  xy_im_128 = _mm_madd_epi16(xy_im_128,input_y);//Im: (x_im*y_re + x_re*y_im)

  //convert back to Q15 before packing
  xy_re_128 = _mm_srai_epi32(xy_re_128,4);//(2^15/64*2*16)
  xy_im_128 = _mm_srai_epi32(xy_im_128,4);

  tmp_z0  = _mm_unpacklo_epi32(xy_re_128,xy_im_128);
  //print_ints("unpack lo:",&tmp_z0[0]);
  tmp_z1  = _mm_unpackhi_epi32(xy_re_128,xy_im_128);
  //print_ints("unpack hi:",&tmp_z1[0]);
  output_z = _mm_packs_epi32(tmp_z0,tmp_z1);

  _mm_empty();
  _m_empty();
  return(output_z);
}

/* Zero Forcing Rx function: nr_conjch0_mult_ch1()
 *
 *
 * */
void nr_ulsch_conjch0_mult_ch1(int *ch0,
                         int *ch1,
                         int32_t *ch0conj_ch1,
                         unsigned short nb_rb,
                         unsigned char output_shift0)
{
  //This function is used to compute multiplications in H_hermitian * H matrix
  short nr_conjugate[8]__attribute__((aligned(16))) = {-1,1,-1,1,-1,1,-1,1};
  unsigned short rb;
  __m128i *dl_ch0_128,*dl_ch1_128, *ch0conj_ch1_128, mmtmpD0,mmtmpD1,mmtmpD2,mmtmpD3;

  dl_ch0_128 = (__m128i *)ch0;
  dl_ch1_128 = (__m128i *)ch1;

  ch0conj_ch1_128 = (__m128i *)ch0conj_ch1;

  for (rb=0; rb<3*nb_rb; rb++) {

    mmtmpD0 = _mm_madd_epi16(dl_ch0_128[0],dl_ch1_128[0]);
    mmtmpD1 = _mm_shufflelo_epi16(dl_ch0_128[0],_MM_SHUFFLE(2,3,0,1));
    mmtmpD1 = _mm_shufflehi_epi16(mmtmpD1,_MM_SHUFFLE(2,3,0,1));
    mmtmpD1 = _mm_sign_epi16(mmtmpD1,*(__m128i*)&nr_conjugate[0]);
    mmtmpD1 = _mm_madd_epi16(mmtmpD1,dl_ch1_128[0]);
    mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift0);
    mmtmpD1 = _mm_srai_epi32(mmtmpD1,output_shift0);
    mmtmpD2 = _mm_unpacklo_epi32(mmtmpD0,mmtmpD1);
    mmtmpD3 = _mm_unpackhi_epi32(mmtmpD0,mmtmpD1);

    ch0conj_ch1_128[0] = _mm_packs_epi32(mmtmpD2,mmtmpD3);

    /*printf("\n Computing conjugates \n");
    print_shorts("ch0:",(int16_t*)&dl_ch0_128[0]);
    print_shorts("ch1:",(int16_t*)&dl_ch1_128[0]);
    print_shorts("pack:",(int16_t*)&ch0conj_ch1_128[0]);*/

    dl_ch0_128+=1;
    dl_ch1_128+=1;
    ch0conj_ch1_128+=1;
  }
  _mm_empty();
  _m_empty();
}
__m128i nr_ulsch_comp_muli_sum(__m128i input_x,
                         __m128i input_y,
                         __m128i input_w,
                         __m128i input_z,
                         __m128i det)
{
  int16_t nr_conjug2[8]__attribute__((aligned(16))) = {1,-1,1,-1,1,-1,1,-1} ;

  __m128i xy_re_128, xy_im_128, wz_re_128, wz_im_128;
  __m128i output, tmp_z0, tmp_z1;

  // complex multiplication (x_re + jx_im)*(y_re + jy_im) = (x_re*y_re - x_im*y_im) + j(x_im*y_re + x_re*y_im)
  // the real part
  xy_re_128 = _mm_sign_epi16(input_x,*(__m128i*)&nr_conjug2[0]);
  xy_re_128 = _mm_madd_epi16(xy_re_128,input_y); //Re: (x_re*y_re - x_im*y_im)

  // the imag part
  xy_im_128 = _mm_shufflelo_epi16(input_x,_MM_SHUFFLE(2,3,0,1));//permutes IQs for the low 64 bits as [I_a0 Q_a1 I_a2 Q_a3]_64bits to [Q_a1 I_a0 Q_a3 I_a2]_64bits
  xy_im_128 = _mm_shufflehi_epi16(xy_im_128,_MM_SHUFFLE(2,3,0,1));//permutes IQs for the high 64 bits as [I_a0 Q_a1 I_a2 Q_a3]_64bits to [Q_a1 I_a0 Q_a3 I_a2]_64bits
  xy_im_128 = _mm_madd_epi16(xy_im_128,input_y);//Im: (x_im*y_re + x_re*y_im)

  // complex multiplication (w_re + jw_im)*(z_re + jz_im) = (w_re*z_re - w_im*z_im) + j(w_im*z_re + w_re*z_im)
  // the real part
  wz_re_128 = _mm_sign_epi16(input_w,*(__m128i*)&nr_conjug2[0]);
  wz_re_128 = _mm_madd_epi16(wz_re_128,input_z); //Re: (w_re*z_re - w_im*z_im)

  // the imag part
  wz_im_128 = _mm_shufflelo_epi16(input_w,_MM_SHUFFLE(2,3,0,1));//permutes IQs for the low 64 bits as [I_a0 Q_a1 I_a2 Q_a3]_64bits to [Q_a1 I_a0 Q_a3 I_a2]_64bits
  wz_im_128 = _mm_shufflehi_epi16(wz_im_128,_MM_SHUFFLE(2,3,0,1));//permutes IQs for the high 64 bits as [I_a0 Q_a1 I_a2 Q_a3]_64bits to [Q_a1 I_a0 Q_a3 I_a2]_64bits
  wz_im_128 = _mm_madd_epi16(wz_im_128,input_z);//Im: (w_im*z_re + w_re*z_im)


  xy_re_128 = _mm_sub_epi32(xy_re_128, wz_re_128);
  xy_im_128 = _mm_sub_epi32(xy_im_128, wz_im_128);
  //print_ints("rx_re:",(int32_t*)&xy_re_128[0]);
  //print_ints("rx_Img:",(int32_t*)&xy_im_128[0]);
  //divide by matrix det and convert back to Q15 before packing
  int sum_det =0;
  for (int k=0; k<4;k++) {
    sum_det += ((((int *)&det)[k])>>2);
    //printf("det_%d = %d log2 =%d \n",k,(((int *)&det[0])[k]),log2_approx(((int *)&det[0])[k]));
    }

  int b = log2_approx(sum_det) - 8;
  if (b > 0) {
    xy_re_128 = _mm_srai_epi32(xy_re_128, b);
    xy_im_128 = _mm_srai_epi32(xy_im_128, b);
  } else {
    xy_re_128 = _mm_slli_epi32(xy_re_128, -b);
    xy_im_128 = _mm_slli_epi32(xy_im_128, -b);
  }

  tmp_z0  = _mm_unpacklo_epi32(xy_re_128,xy_im_128);
  //print_ints("unpack lo:",&tmp_z0[0]);
  tmp_z1  = _mm_unpackhi_epi32(xy_re_128,xy_im_128);
  //print_ints("unpack hi:",&tmp_z1[0]);
  output = _mm_packs_epi32(tmp_z0,tmp_z1);

  _mm_empty();
  _m_empty();
  return(output);
}
/* Zero Forcing Rx function: nr_construct_HhH_elements()
 *
 *
 * */
void nr_ulsch_construct_HhH_elements(int *conjch00_ch00,
                               int *conjch01_ch01,
                               int *conjch11_ch11,
                               int *conjch10_ch10,//
                               int *conjch20_ch20,
                               int *conjch21_ch21,
                               int *conjch30_ch30,
                               int *conjch31_ch31,
                               int *conjch00_ch01,//00_01
                               int *conjch01_ch00,//01_00
                               int *conjch10_ch11,//10_11
                               int *conjch11_ch10,//11_10
                               int *conjch20_ch21,
                               int *conjch21_ch20,
                               int *conjch30_ch31,
                               int *conjch31_ch30,
                               int32_t *after_mf_00,
                               int32_t *after_mf_01,
                               int32_t *after_mf_10,
                               int32_t *after_mf_11,
                               unsigned short nb_rb,
                               unsigned char symbol)
{
  //This function is used to construct the (H_hermitian * H matrix) matrix elements
  unsigned short rb;
  __m128i *conjch00_ch00_128, *conjch01_ch01_128, *conjch11_ch11_128, *conjch10_ch10_128;
  __m128i *conjch20_ch20_128, *conjch21_ch21_128, *conjch30_ch30_128, *conjch31_ch31_128;
  __m128i *conjch00_ch01_128, *conjch01_ch00_128, *conjch10_ch11_128, *conjch11_ch10_128;
  __m128i *conjch20_ch21_128, *conjch21_ch20_128, *conjch30_ch31_128, *conjch31_ch30_128;
  __m128i *after_mf_00_128, *after_mf_01_128, *after_mf_10_128, *after_mf_11_128;

  conjch00_ch00_128 = (__m128i *)conjch00_ch00;
  conjch01_ch01_128 = (__m128i *)conjch01_ch01;
  conjch11_ch11_128 = (__m128i *)conjch11_ch11;
  conjch10_ch10_128 = (__m128i *)conjch10_ch10;

  conjch20_ch20_128 = (__m128i *)conjch20_ch20;
  conjch21_ch21_128 = (__m128i *)conjch21_ch21;
  conjch30_ch30_128 = (__m128i *)conjch30_ch30;
  conjch31_ch31_128 = (__m128i *)conjch31_ch31;

  conjch00_ch01_128 = (__m128i *)conjch00_ch01;
  conjch01_ch00_128 = (__m128i *)conjch01_ch00;
  conjch10_ch11_128 = (__m128i *)conjch10_ch11;
  conjch11_ch10_128 = (__m128i *)conjch11_ch10;

  conjch20_ch21_128 = (__m128i *)conjch20_ch21;
  conjch21_ch20_128 = (__m128i *)conjch21_ch20;
  conjch30_ch31_128 = (__m128i *)conjch30_ch31;
  conjch31_ch30_128 = (__m128i *)conjch31_ch30;

  after_mf_00_128 = (__m128i *)after_mf_00;
  after_mf_01_128 = (__m128i *)after_mf_01;
  after_mf_10_128 = (__m128i *)after_mf_10;
  after_mf_11_128 = (__m128i *)after_mf_11;

  for (rb=0; rb<3*nb_rb; rb++) {

    after_mf_00_128[0] =_mm_adds_epi16(conjch00_ch00_128[0],conjch10_ch10_128[0]);//00_00 + 10_10
    if (conjch20_ch20 != NULL) after_mf_00_128[0] =_mm_adds_epi16(after_mf_00_128[0],conjch20_ch20_128[0]);
    if (conjch30_ch30 != NULL) after_mf_00_128[0] =_mm_adds_epi16(after_mf_00_128[0],conjch30_ch30_128[0]);

    after_mf_11_128[0] =_mm_adds_epi16(conjch01_ch01_128[0], conjch11_ch11_128[0]); //01_01 + 11_11
    if (conjch21_ch21 != NULL) after_mf_11_128[0] =_mm_adds_epi16(after_mf_11_128[0],conjch21_ch21_128[0]);
    if (conjch31_ch31 != NULL) after_mf_11_128[0] =_mm_adds_epi16(after_mf_11_128[0],conjch31_ch31_128[0]);

    after_mf_01_128[0] =_mm_adds_epi16(conjch00_ch01_128[0], conjch10_ch11_128[0]);//00_01 + 10_11
    if (conjch20_ch21 != NULL) after_mf_01_128[0] =_mm_adds_epi16(after_mf_01_128[0],conjch20_ch21_128[0]);
    if (conjch30_ch31 != NULL) after_mf_01_128[0] =_mm_adds_epi16(after_mf_01_128[0],conjch30_ch31_128[0]);

    after_mf_10_128[0] =_mm_adds_epi16(conjch01_ch00_128[0], conjch11_ch10_128[0]);//01_00 + 11_10
    if (conjch21_ch20 != NULL) after_mf_10_128[0] =_mm_adds_epi16(after_mf_10_128[0],conjch21_ch20_128[0]);
    if (conjch31_ch30 != NULL) after_mf_10_128[0] =_mm_adds_epi16(after_mf_10_128[0],conjch31_ch30_128[0]);

#ifdef DEBUG_DLSCH_DEMOD
    if ((rb<=30))
    {
      printf(" \n construct_HhH_elements \n");
      print_shorts("after_mf_00_128:",(int16_t*)&after_mf_00_128[0]);
      print_shorts("after_mf_01_128:",(int16_t*)&after_mf_01_128[0]);
      print_shorts("after_mf_10_128:",(int16_t*)&after_mf_10_128[0]);
      print_shorts("after_mf_11_128:",(int16_t*)&after_mf_11_128[0]);
    }
#endif
    conjch00_ch00_128+=1;
    conjch10_ch10_128+=1;
    conjch01_ch01_128+=1;
    conjch11_ch11_128+=1;

    if (conjch20_ch20 != NULL) conjch20_ch20_128+=1;
    if (conjch21_ch21 != NULL) conjch21_ch21_128+=1;
    if (conjch30_ch30 != NULL) conjch30_ch30_128+=1;
    if (conjch31_ch31 != NULL) conjch31_ch31_128+=1;

    conjch00_ch01_128+=1;
    conjch01_ch00_128+=1;
    conjch10_ch11_128+=1;
    conjch11_ch10_128+=1;

    if (conjch20_ch21 != NULL) conjch20_ch21_128+=1;
    if (conjch21_ch20 != NULL) conjch21_ch20_128+=1;
    if (conjch30_ch31 != NULL) conjch30_ch31_128+=1;
    if (conjch31_ch30 != NULL) conjch31_ch30_128+=1;

    after_mf_00_128 += 1;
    after_mf_01_128 += 1;
    after_mf_10_128 += 1;
    after_mf_11_128 += 1;
  }
  _mm_empty();
  _m_empty();
}

/*
 * MMSE Rx function: nr_ulsch_mmse_2layers()
 */
uint8_t nr_ulsch_mmse_2layers(NR_DL_FRAME_PARMS *frame_parms,
                              int **rxdataF_comp,
                              int **ul_ch_mag,
                              int **ul_ch_magb,
                              int **ul_ch_magc,
                              int **ul_ch_estimates_ext,
                              unsigned short nb_rb,
                              unsigned char n_rx,
                              unsigned char mod_order,
                              int shift,
                              unsigned char symbol,
                              int length,
                              uint32_t noise_var)
{
  int *ch00, *ch01, *ch10, *ch11;
  int *ch20, *ch30, *ch21, *ch31;
  uint32_t nb_rb_0 = length/12 + ((length%12)?1:0);

  int off = ((nb_rb&1) == 1)? 4:0;

  /* we need at least alignment to 16 bytes, let's put 32 to be sure
   * (maybe not necessary but doesn't hurt)
   */
  int32_t conjch00_ch01[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch01_ch00[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch10_ch11[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch11_ch10[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch00_ch00[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch01_ch01[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch10_ch10[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch11_ch11[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch20_ch20[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch21_ch21[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch30_ch30[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch31_ch31[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch20_ch21[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch30_ch31[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch21_ch20[12*nb_rb] __attribute__((aligned(32)));
  int32_t conjch31_ch30[12*nb_rb] __attribute__((aligned(32)));

  int32_t af_mf_00[12*nb_rb] __attribute__((aligned(32)));
  int32_t af_mf_01[12*nb_rb] __attribute__((aligned(32)));
  int32_t af_mf_10[12*nb_rb] __attribute__((aligned(32)));
  int32_t af_mf_11[12*nb_rb] __attribute__((aligned(32)));
  int32_t determ_fin[12*nb_rb] __attribute__((aligned(32)));

  switch (n_rx) {
    case 2://
      ch00 = (int *)&ul_ch_estimates_ext[0][symbol*(off+nb_rb*12)];
      ch01 = (int *)&ul_ch_estimates_ext[2][symbol*(off+nb_rb*12)];
      ch10 = (int *)&ul_ch_estimates_ext[1][symbol*(off+nb_rb*12)];
      ch11 = (int *)&ul_ch_estimates_ext[3][symbol*(off+nb_rb*12)];
      ch20 = NULL;
      ch21 = NULL;
      ch30 = NULL;
      ch31 = NULL;
      break;

    case 4://
      ch00 = (int *)&ul_ch_estimates_ext[0][symbol*(off+nb_rb*12)];
      ch01 = (int *)&ul_ch_estimates_ext[4][symbol*(off+nb_rb*12)];
      ch10 = (int *)&ul_ch_estimates_ext[1][symbol*(off+nb_rb*12)];
      ch11 = (int *)&ul_ch_estimates_ext[5][symbol*(off+nb_rb*12)];
      ch20 = (int *)&ul_ch_estimates_ext[2][symbol*(off+nb_rb*12)];
      ch21 = (int *)&ul_ch_estimates_ext[6][symbol*(off+nb_rb*12)];
      ch30 = (int *)&ul_ch_estimates_ext[3][symbol*(off+nb_rb*12)];
      ch31 = (int *)&ul_ch_estimates_ext[7][symbol*(off+nb_rb*12)];
      break;

    default:
      return -1;
      break;
  }

  /* 1- Compute the rx channel matrix after compensation: (1/2^log2_max)x(H_herm x H)
   * for n_rx = 2
   * |conj_H_00       conj_H_10|    | H_00         H_01|   |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
   * |                         |  x |                  | = |                                                                 |
   * |conj_H_01       conj_H_11|    | H_10         H_11|   |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
   *
   */

  if (n_rx>=2){
    // (1/2^log2_maxh)*conj_H_00xH_00: (1/(64*2))conjH_00*H_00*2^15
    nr_ulsch_conjch0_mult_ch1(ch00,
                        ch00,
                        conjch00_ch00,
                        nb_rb_0,
                        shift);
    // (1/2^log2_maxh)*conj_H_10xH_10: (1/(64*2))conjH_10*H_10*2^15
    nr_ulsch_conjch0_mult_ch1(ch10,
                        ch10,
                        conjch10_ch10,
                        nb_rb_0,
                        shift);
    // conj_H_00xH_01
    nr_ulsch_conjch0_mult_ch1(ch00,
                        ch01,
                        conjch00_ch01,
                        nb_rb_0,
                        shift); // this shift is equal to the channel level log2_maxh
    // conj_H_10xH_11
    nr_ulsch_conjch0_mult_ch1(ch10,
                        ch11,
                        conjch10_ch11,
                        nb_rb_0,
                        shift);
    // conj_H_01xH_01
    nr_ulsch_conjch0_mult_ch1(ch01,
                        ch01,
                        conjch01_ch01,
                        nb_rb_0,
                        shift);
    // conj_H_11xH_11
    nr_ulsch_conjch0_mult_ch1(ch11,
                        ch11,
                        conjch11_ch11,
                        nb_rb_0,
                        shift);
    // conj_H_01xH_00
    nr_ulsch_conjch0_mult_ch1(ch01,
                        ch00,
                        conjch01_ch00,
                        nb_rb_0,
                        shift);
    // conj_H_11xH_10
    nr_ulsch_conjch0_mult_ch1(ch11,
                        ch10,
                        conjch11_ch10,
                        nb_rb_0,
                        shift);
  }
  if (n_rx==4){
    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2*16))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch20,
                        ch20,
                        conjch20_ch20,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_30xH_30: (1/(64*2*4))conjH_30*H_30*2^15
    nr_ulsch_conjch0_mult_ch1(ch30,
                        ch30,
                        conjch30_ch30,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch20,
                        ch21,
                        conjch20_ch21,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch30,
                        ch31,
                        conjch30_ch31,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch21,
                        ch21,
                        conjch21_ch21,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch31,
                        ch31,
                        conjch31_ch31,
                        nb_rb_0,
                        shift);

    // (1/2^log2_maxh)*conj_H_20xH_20: (1/(64*2))conjH_20*H_20*2^15
    nr_ulsch_conjch0_mult_ch1(ch21,
                        ch20,
                        conjch21_ch20,
                        nb_rb_0,
                        shift);

    nr_ulsch_conjch0_mult_ch1(ch31,
                        ch30,
                        conjch31_ch30,
                        nb_rb_0,
                        shift);

    nr_ulsch_construct_HhH_elements(conjch00_ch00,
                              conjch01_ch01,
                              conjch11_ch11,
                              conjch10_ch10,//
                              conjch20_ch20,
                              conjch21_ch21,
                              conjch30_ch30,
                              conjch31_ch31,
                              conjch00_ch01,
                              conjch01_ch00,
                              conjch10_ch11,
                              conjch11_ch10,//
                              conjch20_ch21,
                              conjch21_ch20,
                              conjch30_ch31,
                              conjch31_ch30,
                              af_mf_00,
                              af_mf_01,
                              af_mf_10,
                              af_mf_11,
                              nb_rb_0,
                              symbol);
  }
  if (n_rx==2){
    nr_ulsch_construct_HhH_elements(conjch00_ch00,
                              conjch01_ch01,
                              conjch11_ch11,
                              conjch10_ch10,//
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              conjch00_ch01,
                              conjch01_ch00,
                              conjch10_ch11,
                              conjch11_ch10,//
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              af_mf_00,
                              af_mf_01,
                              af_mf_10,
                              af_mf_11,
                              nb_rb_0,
                              symbol);
  }

  // Add noise_var such that: H^h * H + noise_var * I
  if (noise_var != 0) {
    __m128i nvar_128i = simde_mm_set1_epi32(noise_var);
    __m128i *af_mf_00_128i = (__m128i *)af_mf_00;
    __m128i *af_mf_11_128i = (__m128i *)af_mf_11;
    for (int k = 0; k < 3 * nb_rb_0; k++) {
      af_mf_00_128i[0] = simde_mm_add_epi32(af_mf_00_128i[0], nvar_128i);
      af_mf_11_128i[0] = simde_mm_add_epi32(af_mf_11_128i[0], nvar_128i);
      af_mf_00_128i++;
      af_mf_11_128i++;
    }
  }

  //det_HhH = ad -bc
  nr_ulsch_det_HhH(af_mf_00,//a
             af_mf_01,//b
             af_mf_10,//c
             af_mf_11,//d
             determ_fin,
             nb_rb_0,
             symbol,
             shift);
  /* 2- Compute the channel matrix inversion **********************************
   *
     *    |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
     * A= |                                                                 |
     *    |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
     *
     *
     *
     *inv(A) =(1/det)*[d  -b
     *                 -c  a]
     *
     *
     **************************************************************************/
  __m128i *ul_ch_mag128_0 = NULL, *ul_ch_mag128b_0 = NULL, *ul_ch_mag128c_0 = NULL; // Layer 0
  __m128i *ul_ch_mag128_1 = NULL, *ul_ch_mag128b_1 = NULL, *ul_ch_mag128c_1 = NULL; // Layer 1
  __m128i mmtmpD0, mmtmpD1, mmtmpD2, mmtmpD3;
  __m128i QAM_amp128 = {0}, QAM_amp128b = {0}, QAM_amp128c = {0};

  __m128i *determ_fin_128 = (__m128i *)&determ_fin[0];

  __m128i *rxdataF_comp128_0 = (__m128i *)&rxdataF_comp[0][symbol * (off + nb_rb * 12)]; // aatx=0 @ aarx =0
  __m128i *rxdataF_comp128_1 = (__m128i *)&rxdataF_comp[n_rx][symbol * (off + nb_rb * 12)]; // aatx=1 @ aarx =0

  __m128i *after_mf_a_128 = (__m128i *)af_mf_00;
  __m128i *after_mf_b_128 = (__m128i *)af_mf_01;
  __m128i *after_mf_c_128 = (__m128i *)af_mf_10;
  __m128i *after_mf_d_128 = (__m128i *)af_mf_11;

  if (mod_order > 2) {
    if (mod_order == 4) {
      QAM_amp128 = _mm_set1_epi16(QAM16_n1); // 2/sqrt(10)
      QAM_amp128b = _mm_setzero_si128();
      QAM_amp128c = _mm_setzero_si128();
    } else if (mod_order == 6) {
      QAM_amp128 = _mm_set1_epi16(QAM64_n1); // 4/sqrt{42}
      QAM_amp128b = _mm_set1_epi16(QAM64_n2); // 2/sqrt{42}
      QAM_amp128c = _mm_setzero_si128();
    } else if (mod_order == 8) {
      QAM_amp128 = _mm_set1_epi16(QAM256_n1);
      QAM_amp128b = _mm_set1_epi16(QAM256_n2);
      QAM_amp128c = _mm_set1_epi16(QAM256_n3);
    }
    ul_ch_mag128_0 = (__m128i *)&ul_ch_mag[0][symbol * (off + nb_rb * 12)];
    ul_ch_mag128b_0 = (__m128i *)&ul_ch_magb[0][symbol * (off + nb_rb * 12)];
    ul_ch_mag128c_0 = (__m128i *)&ul_ch_magc[0][symbol * (off + nb_rb * 12)];
    ul_ch_mag128_1 = (__m128i *)&ul_ch_mag[frame_parms->nb_antennas_rx][symbol * (off + nb_rb * 12)];
    ul_ch_mag128b_1 = (__m128i *)&ul_ch_magb[frame_parms->nb_antennas_rx][symbol * (off + nb_rb * 12)];
    ul_ch_mag128c_1 = (__m128i *)&ul_ch_magc[frame_parms->nb_antennas_rx][symbol * (off + nb_rb * 12)];
  }

  for (int rb = 0; rb < 3 * nb_rb_0; rb++) {

    // Magnitude computation
    if (mod_order > 2) {

      int sum_det = 0;
      for (int k = 0; k < 4; k++) {
        sum_det += ((((int *)&determ_fin_128[0])[k]) >> 2);
      }

      int b = log2_approx(sum_det) - 8;
      if (b > 0) {
        mmtmpD2 = _mm_srai_epi32(determ_fin_128[0], b);
      } else {
        mmtmpD2 = _mm_slli_epi32(determ_fin_128[0], -b);
      }
      mmtmpD3 = _mm_unpacklo_epi32(mmtmpD2, mmtmpD2);
      mmtmpD2 = _mm_unpackhi_epi32(mmtmpD2, mmtmpD2);
      mmtmpD2 = _mm_packs_epi32(mmtmpD3, mmtmpD2);

      // Layer 0
      ul_ch_mag128_0[0] = mmtmpD2;
      ul_ch_mag128b_0[0] = mmtmpD2;
      ul_ch_mag128c_0[0] = mmtmpD2;
      ul_ch_mag128_0[0] = _mm_mulhi_epi16(ul_ch_mag128_0[0], QAM_amp128);
      ul_ch_mag128_0[0] = _mm_slli_epi16(ul_ch_mag128_0[0], 1);
      ul_ch_mag128b_0[0] = _mm_mulhi_epi16(ul_ch_mag128b_0[0], QAM_amp128b);
      ul_ch_mag128b_0[0] = _mm_slli_epi16(ul_ch_mag128b_0[0], 1);
      ul_ch_mag128c_0[0] = _mm_mulhi_epi16(ul_ch_mag128c_0[0], QAM_amp128c);
      ul_ch_mag128c_0[0] = _mm_slli_epi16(ul_ch_mag128c_0[0], 1);

      // Layer 1
      ul_ch_mag128_1[0] = mmtmpD2;
      ul_ch_mag128b_1[0] = mmtmpD2;
      ul_ch_mag128c_1[0] = mmtmpD2;
      ul_ch_mag128_1[0] = _mm_mulhi_epi16(ul_ch_mag128_1[0], QAM_amp128);
      ul_ch_mag128_1[0] = _mm_slli_epi16(ul_ch_mag128_1[0], 1);
      ul_ch_mag128b_1[0] = _mm_mulhi_epi16(ul_ch_mag128b_1[0], QAM_amp128b);
      ul_ch_mag128b_1[0] = _mm_slli_epi16(ul_ch_mag128b_1[0], 1);
      ul_ch_mag128c_1[0] = _mm_mulhi_epi16(ul_ch_mag128c_1[0], QAM_amp128c);
      ul_ch_mag128c_1[0] = _mm_slli_epi16(ul_ch_mag128c_1[0], 1);
    }

    // multiply by channel Inv
    //rxdataF_zf128_0 = rxdataF_comp128_0*d - b*rxdataF_comp128_1
    //rxdataF_zf128_1 = rxdataF_comp128_1*a - c*rxdataF_comp128_0
    //printf("layer_1 \n");
    mmtmpD0 = nr_ulsch_comp_muli_sum(rxdataF_comp128_0[0],
                               after_mf_d_128[0],
                               rxdataF_comp128_1[0],
                               after_mf_b_128[0],
                               determ_fin_128[0]);

    //printf("layer_2 \n");
    mmtmpD1 = nr_ulsch_comp_muli_sum(rxdataF_comp128_1[0],
                               after_mf_a_128[0],
                               rxdataF_comp128_0[0],
                               after_mf_c_128[0],
                               determ_fin_128[0]);

    rxdataF_comp128_0[0] = mmtmpD0;
    rxdataF_comp128_1[0] = mmtmpD1;

#ifdef DEBUG_DLSCH_DEMOD
    printf("\n Rx signal after ZF l%d rb%d\n",symbol,rb);
    print_shorts(" Rx layer 1:",(int16_t*)&rxdataF_comp128_0[0]);
    print_shorts(" Rx layer 2:",(int16_t*)&rxdataF_comp128_1[0]);
#endif
    determ_fin_128 += 1;
    ul_ch_mag128_0 += 1;
    ul_ch_mag128_1 += 1;
    ul_ch_mag128b_0 += 1;
    ul_ch_mag128b_1 += 1;
    ul_ch_mag128c_0 += 1;
    ul_ch_mag128c_1 += 1;
    rxdataF_comp128_0 += 1;
    rxdataF_comp128_1 += 1;
    after_mf_a_128 += 1;
    after_mf_b_128 += 1;
    after_mf_c_128 += 1;
    after_mf_d_128 += 1;
  }
  _mm_empty();
  _m_empty();
   return(0);
}

//==============================================================================================

/* Main Function */
void nr_rx_pusch(PHY_VARS_gNB *gNB,
                 PHY_VARS_NR_UE *ue,
                 UE_nr_rxtx_proc_t *proc,
                 nr_phy_data_t *phy_data,
                 int rxFSz,
                 c16_t rxdataF[][rxFSz],
                 uint8_t ulsch_id,
                 uint32_t frame,
                 uint8_t slot,
                 uint8_t nb_antennas_tx,
                 void (* _nr_ue_csi_rs_procedures)(PHY_VARS_NR_UE *ue, UE_nr_rxtx_proc_t *proc, c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP]),
                 unsigned char harq_pid,
                 bool *is_csi_rs_slot)
{

  uint8_t aarx, aatx;
  uint32_t nb_re_pusch, bwp_start_subcarrier;
  int avgs = 0;

  nfapi_nr_dl_tti_csi_rs_pdu_rel15_t *csi_params = NULL;
  AssertFatal((gNB && !ue) || (!gNB && ue),"Both gNB and UE cannot be non-null\n");
  NR_DL_FRAME_PARMS *frame_parms = gNB ? &gNB->frame_parms : &ue->SL_UE_PHY_PARAMS.sl_frame_params;
  NR_gNB_ULSCH_t *ulsch = gNB ? &gNB->ulsch[ulsch_id] : &ue->slsch[ulsch_id];
  nfapi_nr_pusch_pdu_t *rel15_ul = gNB ? &ulsch->harq_process->ulsch_pdu : NULL;
  sl_nr_rx_config_pssch_sci_pdu_t *pssch_pdu = ue ? ulsch->harq_process->pssch_pdu : NULL;
  uint32_t nrOfLayers = pssch_pdu ? pssch_pdu->num_layers : rel15_ul->nrOfLayers;
  uint32_t rb_start = pssch_pdu ? pssch_pdu->startrb : rel15_ul->rb_start;
  uint32_t bwp_start = pssch_pdu ? 0 : rel15_ul->bwp_start;
  uint32_t rnti = pssch_pdu ? 0 : rel15_ul->rnti;

  uint32_t rb_size                   = pssch_pdu ? pssch_pdu->num_subch*pssch_pdu->subchannel_size : rel15_ul->rb_size;
  uint32_t qam_mod_order             = pssch_pdu ? pssch_pdu->mod_order                                   : rel15_ul->qam_mod_order;
  uint32_t start_symbol_index        = pssch_pdu ? 1                                                     : rel15_ul->start_symbol_index;
  uint32_t nr_of_symbols             = pssch_pdu ? pssch_pdu->pssch_numsym                               : rel15_ul->nr_of_symbols;
  uint32_t dmrs_config_type          = pssch_pdu ? 0                                                     : rel15_ul->dmrs_config_type;
  uint32_t num_dmrs_cdm_grps_no_data = pssch_pdu ? 1                                                     : rel15_ul->num_dmrs_cdm_grps_no_data;
  uint32_t ul_dmrs_symb_pos          = pssch_pdu ? pssch_pdu->dmrs_symbol_position                       : rel15_ul->ul_dmrs_symb_pos;
  uint32_t dmrs_ports                = pssch_pdu ? pssch_pdu->num_layers                                 : rel15_ul->dmrs_ports;
  int sci1_re_per_symb = pssch_pdu ? (pssch_pdu->pscch_numrbs*NR_NB_SC_PER_RB) : 0; 
  int sci2_re = pssch_pdu ? get_NREsci2_2(pssch_pdu->sci2_alpha_times_100,
                                          pssch_pdu->sci2_len,
                                          pssch_pdu->sci2_beta_offset,
                                          pssch_pdu->pssch_numsym,
                                          pssch_pdu->pscch_numsym,
                                          pssch_pdu->pscch_numrbs,
                                          pssch_pdu->l_subch,
                                          pssch_pdu->subchannel_size,
                                          pssch_pdu->targetCodeRate,
                                          0) : 0;
  int16_t sci2_llrs[(sci2_re*2)] __attribute__((aligned(16)));
  int16_t unscrambled_sci2_llrs[(sci2_re*2)] __attribute__((aligned(16)));
  int sci2_cnt=0;
  int sci2_left = sci2_re;

  int avg[frame_parms->nb_antennas_rx*nrOfLayers];
  int16_t *temp_llr = (int16_t *)malloc16_clear((8 * ((3 * 8 * 6144) + 12)) * sizeof(int16_t));
  int32_t *temp_symbol = (int32_t *) malloc16_clear(rb_size * NR_NB_SC_PER_RB * sizeof(int32_t));
  NR_gNB_PUSCH *pusch_vars = gNB ? &gNB->pusch_vars[ulsch_id] : &ue->pssch_vars[ulsch_id];
  pusch_vars->dmrs_symbol = INVALID_VALUE;
  pusch_vars->cl_done = 0;

  bwp_start_subcarrier = ((rb_start + bwp_start)*NR_NB_SC_PER_RB + frame_parms->first_carrier_offset) % frame_parms->ofdm_symbol_size;
  LOG_D(PHY,"pusch %d.%d : bwp_start_subcarrier %d, rb_start %d, first_carrier_offset %d\n", frame,slot,bwp_start_subcarrier, rb_start, frame_parms->first_carrier_offset);
  LOG_D(PHY,"pusch %d.%d : ul_dmrs_symb_pos %x\n",frame,slot,ul_dmrs_symb_pos);
  LOG_D(PHY,"ulsch RX %x : start_rb %d nb_rb %d Nl %d Tpmi %d bwp_start %d start_sc %d start_symbol %d num_symbols %d cdmgrpsnodata %d num_dmrs %d dmrs_ports %d\n",
          rnti,rb_start,rb_size,
          nrOfLayers,0,bwp_start,0,start_symbol_index,nr_of_symbols,
          num_dmrs_cdm_grps_no_data,ul_dmrs_symb_pos,dmrs_ports);
  //----------------------------------------------------------
  //--------------------- Channel estimation ---------------------
  //----------------------------------------------------------
  if (gNB) start_meas(&gNB->ulsch_channel_estimation_stats);
  int max_ch = 0;
  uint32_t nvar = 0;
  for(uint8_t symbol = start_symbol_index; symbol < (start_symbol_index + nr_of_symbols); symbol++) {
    uint8_t dmrs_symbol_flag = (ul_dmrs_symb_pos >> symbol) & 0x01;
    LOG_D(PHY, "symbol %d, dmrs_symbol_flag :%d\n", symbol, dmrs_symbol_flag);
    
    if (dmrs_symbol_flag == 1) {
      if (pusch_vars->dmrs_symbol == INVALID_VALUE)
        pusch_vars->dmrs_symbol = symbol;

      for (int nl=0; nl<nrOfLayers; nl++) {
        uint32_t nvar_tmp = 0;
	int dmrs_port = get_dmrs_port(nl,dmrs_ports);
	if (dmrs_port<0) return;
        nr_pusch_channel_estimation(gNB,ue,rxFSz,rxdataF,
                                    slot,
                                    dmrs_port,
                                    symbol,
                                    ulsch_id,
                                    bwp_start_subcarrier,
                                    rel15_ul,
				    pssch_pdu,
                                    &max_ch,
                                    &nvar_tmp);
        nvar += nvar_tmp;
      }

      PHY_MEASUREMENTS_gNB *meas = gNB ? &gNB->measurements : ue->sl_measurements; 
      nr_gnb_measurements(meas, frame_parms,ulsch, pusch_vars, symbol, nrOfLayers);
      allocCast2D(n0_subband_power,
                  unsigned int,
                  meas->n0_subband_power,
                  frame_parms->nb_antennas_rx,
                  frame_parms->N_RB_UL,
                  false);
      for (aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
        if (symbol == start_symbol_index) {
          pusch_vars->ulsch_power[aarx] = 0;
          pusch_vars->ulsch_noise_power[aarx] = 0;
        }
        for (aatx = 0; aatx < nrOfLayers; aatx++) {
          pusch_vars->ulsch_power[aarx] += signal_energy_nodc(
              &pusch_vars->ul_ch_estimates[aatx * frame_parms->nb_antennas_rx + aarx][symbol * frame_parms->ofdm_symbol_size],
              rb_size * 12);
        }
        for (int rb = 0; rb < rb_size; rb++) {
          pusch_vars->ulsch_noise_power[aarx] +=
              n0_subband_power[aarx][bwp_start + rb_start + rb] / rb_size;
        }
        LOG_D(NR_PHY,
              "aa %d, symbol %d, bwp_start%d, rb_start %d, rb_size %d: ulsch_power %d, ulsch_noise_power %d\n",
              aarx,symbol,
              bwp_start,
              rb_start,
              rb_size,
              pusch_vars->ulsch_power[aarx],
              pusch_vars->ulsch_noise_power[aarx]);
      }
    }
  }

  nvar /= (nr_of_symbols * nrOfLayers * frame_parms->nb_antennas_rx);

  if (gNB && gNB->chest_time == 1) { // averaging time domain channel estimates
    nr_chest_time_domain_avg(frame_parms,
                             pusch_vars->ul_ch_estimates,
                             nr_of_symbols,
                             start_symbol_index,
                             ul_dmrs_symb_pos,
                             rb_size);

    pusch_vars->dmrs_symbol =
        get_next_dmrs_symbol_in_slot(ul_dmrs_symb_pos, start_symbol_index, nr_of_symbols);
  }
  if (gNB) stop_meas(&gNB->ulsch_channel_estimation_stats);

  int off = ((rb_size&1) == 1)? 4:0;
  uint32_t rxdataF_ext_offset = 0;
  uint8_t shift_ch_ext = nrOfLayers > 1 ? log2_approx(max_ch >> 11) : 0;

  // Flag to select the receiver: (true) Nonlinear ML receiver, (false) Linear MMSE receiver
  // By default, we are using the Nonlinear ML receiver, except
  //  - for 256QAM as Nonlinear ML receiver is not implemented for 256QAM
  //  - for 64QAM as Nonlinear ML receiver requires more processing time than MMSE, and many machines are not powerful enough
  bool ml_rx = true;
  if (nrOfLayers != 2 || qam_mod_order >= 6) {
    ml_rx = false;
  }

  int ad_shift = 0;
  if (nrOfLayers == 1) {
    ad_shift = 1 + log2_approx(frame_parms->nb_antennas_rx >> 2);
  } else if (ml_rx == false) {
    ad_shift = -3; // For 2-layers, we are already doing a bit shift in the nr_ulsch_mmse_2layers() function, so we can use more bits
  }

  for(uint8_t symbol = start_symbol_index; symbol < (start_symbol_index + nr_of_symbols); symbol++) {

    uint8_t csi_rs_symbol_flag = 0;
    if (phy_data->sl_rx_action == SL_NR_CONFIG_TYPE_RX_PSSCH_SLSCH_CSI_RS) {
      *is_csi_rs_slot = true;
      csi_params = (nfapi_nr_dl_tti_csi_rs_pdu_rel15_t *)&ue->csirs_vars[0]->csirs_config_pdu;
    } else {
      *is_csi_rs_slot = false;
    }
    if (*is_csi_rs_slot && (csi_params->symb_l0 == symbol)) {
      csi_rs_symbol_flag = 1;
      AssertFatal(csi_params->freq_density > 0, "freq_density MUST be greater than zero");
      AssertFatal(csi_params->nr_of_rbs > 0, "nr_of_rbs MUST be greater than zero");
      LOG_D(NR_PHY, "%d.%d symbol %i, freq_density %i symb_l0 %i csi_type %i power_control_offset %i power_control_offset_ss %i measurement_bitmap %i cdm_type %i row %i freq_domain %i start_rb %i nr_of_rbs %i\n",
            frame,
            slot,
            symbol,
            csi_params->freq_density,
            csi_params->symb_l0,
            csi_params->csi_type,
            csi_params->power_control_offset,
            csi_params->power_control_offset_ss,
            csi_params->measurement_bitmap,
            csi_params->cdm_type,
            csi_params->row,
            csi_params->freq_domain,
            csi_params->start_rb,
            csi_params->nr_of_rbs);
      if (phy_data->sl_rx_action == SL_NR_CONFIG_TYPE_RX_PSSCH_SLSCH_CSI_RS) {
        // FIXIT: Reconsider index of csirs_vars[0] for multiple connected UEs case
        if (ue->csirs_vars[0]->active == 1) {
          LOG_D(NR_PHY, "%d.%d Received CSI-RS\n", proc->frame_rx, proc->nr_slot_rx);
          nr_slot_fep(ue, frame_parms, proc, symbol, rxdataF, link_type_sl_pc5);
          _nr_ue_csi_rs_procedures(ue, proc, rxdataF);
          ue->csirs_vars[0]->active = 0;
        }
      }
    }

    uint8_t dmrs_symbol_flag = (ul_dmrs_symb_pos >> symbol) & 0x01;
    int sci2_cnt_thissymb=0;
    if (csi_rs_symbol_flag) {
      uint8_t freq_subcarriers_per_rb = 12;
      uint8_t nr_rbs_w_csi_rs = csi_params->nr_of_rbs / csi_params->freq_density;
      uint8_t nr_rbs_wo_csi_rs = (rb_size - nr_rbs_w_csi_rs);
      // Actually, kprime + 1 sub-carriers are used by csi-rs. kprime can be 0 or 1 but nb_antennas_tx can be greater than 2.
      uint8_t subcarriers_used = nb_antennas_tx > 2 ? 2 : nb_antennas_tx;
      nb_re_pusch = nr_rbs_wo_csi_rs * freq_subcarriers_per_rb  + nr_rbs_w_csi_rs * (freq_subcarriers_per_rb - subcarriers_used);
    } else if (dmrs_symbol_flag == 1) {
      if ((ul_dmrs_symb_pos >> ((symbol + 1) % frame_parms->symbols_per_slot)) & 0x01)
        AssertFatal(1==0,"Double DMRS configuration is not yet supported\n");

      if (ue || gNB->chest_time == 0) // Non averaging time domain channel estimates
        pusch_vars->dmrs_symbol = symbol;

      if (dmrs_config_type == 0) {
        // if no data in dmrs cdm group is 1 only even REs have no data
        // if no data in dmrs cdm group is 2 both odd and even REs have no data
        nb_re_pusch = rb_size *(12 - (num_dmrs_cdm_grps_no_data*6));
      }
      else {
        nb_re_pusch = rb_size *(12 - (num_dmrs_cdm_grps_no_data*4));
      }
    } 
    else {
      nb_re_pusch = rb_size * NR_NB_SC_PER_RB;
    }

    pusch_vars->ul_valid_re_per_slot[symbol] = nb_re_pusch;
    LOG_D(PHY, "symbol %d: nb_re_pusch %d, DMRS symbl used for Chest :%d \n", symbol, nb_re_pusch, pusch_vars->dmrs_symbol);
    //----------------------------------------------------------
    //--------------------- RBs extraction ---------------------
    //----------------------------------------------------------
    if (nb_re_pusch > 0) {
      LOG_D(NR_PHY,"extract RBs : frame   %d, slot %d symbol %d nb_re_pusch %d\n", frame,slot,symbol, nb_re_pusch);
      if (gNB) start_meas(&gNB->ulsch_rbs_extraction_stats);
      nr_ulsch_extract_rbs(rxFSz, rxdataF, pusch_vars, slot, symbol, dmrs_symbol_flag, csi_rs_symbol_flag, bwp_start, rb_start, rb_size, nrOfLayers, num_dmrs_cdm_grps_no_data, dmrs_config_type, frame_parms, csi_params);
      if (gNB) stop_meas(&gNB->ulsch_rbs_extraction_stats);

      //----------------------------------------------------------
      //--------------------- Channel Scaling --------------------
      //----------------------------------------------------------
      nr_ulsch_scale_channel(pusch_vars->ul_ch_estimates_ext,
                             frame_parms,
                             ulsch,
                             symbol,
                             dmrs_symbol_flag,
                             nb_re_pusch,
                             nrOfLayers,
                             rb_size,
                             shift_ch_ext);

      if (pusch_vars->cl_done == 0) {
        nr_ulsch_channel_level(pusch_vars->ul_ch_estimates_ext,
                               frame_parms,
                               avg,
                               symbol,
                               nb_re_pusch,
                               nrOfLayers,
                               rb_size);

        avgs = 0;

        for (aatx=0;aatx<nrOfLayers;aatx++)
          for (aarx=0;aarx<frame_parms->nb_antennas_rx;aarx++)
            avgs = cmax(avgs,avg[aatx*frame_parms->nb_antennas_rx+aarx]);

        pusch_vars->log2_maxh = (log2_approx(avgs) >> 1) + ad_shift;
        if (pusch_vars->log2_maxh < 0) {
          pusch_vars->log2_maxh = 0;
        }
        pusch_vars->cl_done = 1;
      }

      //----------------------------------------------------------
      //--------------------- Channel Compensation ---------------
      //----------------------------------------------------------
      if (gNB) start_meas(&gNB->ulsch_channel_compensation_stats);
      //LOG_I(PHY, "Doing channel compensations log2_maxh %d, avgs %d (%d,%d)\n" ,pusch_vars->log2_maxh, avgs,avg[0], avg[1]);
      nr_ulsch_channel_compensation(pusch_vars->rxdataF_ext,
                                    pusch_vars->ul_ch_estimates_ext,
                                    pusch_vars->ul_ch_mag0,
                                    pusch_vars->ul_ch_magb0,
                                    pusch_vars->ul_ch_magc0,
                                    pusch_vars->rxdataF_comp,
                                    (nrOfLayers > 1) ? pusch_vars->rho : NULL,
                                    frame_parms,
                                    symbol,
                                    nb_re_pusch,
                                    dmrs_symbol_flag,
                                    qam_mod_order,
                                    nrOfLayers,
                                    rb_size,
                                    pusch_vars->log2_maxh);
      if (gNB) stop_meas(&gNB->ulsch_channel_compensation_stats);

      if (gNB) start_meas(&gNB->ulsch_mrc_stats);
      nr_ulsch_detection_mrc(frame_parms,
                             pusch_vars->rxdataF_comp,
                             pusch_vars->ul_ch_mag0,
                             pusch_vars->ul_ch_magb0,
                             pusch_vars->ul_ch_magc0,
                             (nrOfLayers > 1) ? pusch_vars->rho : NULL,
                             nrOfLayers,
                             symbol,
                             rb_size,
                             nb_re_pusch);

      // Apply MMSE for 2 Tx layers
      if (ml_rx == false && nrOfLayers == 2) {
        nr_ulsch_mmse_2layers(frame_parms,
                              pusch_vars->rxdataF_comp,
                              pusch_vars->ul_ch_mag0,
                              pusch_vars->ul_ch_magb0,
                              pusch_vars->ul_ch_magc0,
                              pusch_vars->ul_ch_estimates_ext,
                              rb_size,
                              frame_parms->nb_antennas_rx,
                              qam_mod_order,
                              pusch_vars->log2_maxh,
                              symbol,
                              nb_re_pusch,
                              nvar);
      }

      if (gNB) stop_meas(&gNB->ulsch_mrc_stats);

      if (gNB && rel15_ul->transform_precoding == transformPrecoder_enabled) {
        // For odd number of resource blocks need byte alignment to multiple of 8
        int nb_re_pusch2 = nb_re_pusch + (nb_re_pusch&7);

        // perform IDFT operation on the compensated rxdata if transform precoding is enabled
        nr_idft(&pusch_vars->rxdataF_comp[0][symbol * nb_re_pusch2], nb_re_pusch);
        LOG_D(PHY,"Transform precoding being done on data- symbol: %d, nb_re_pusch: %d\n", symbol, nb_re_pusch);
      }

      //----------------------------------------------------------
      //--------------------- PTRS Processing --------------------
      //----------------------------------------------------------
      /* In case PTRS is enabled then LLR will be calculated after PTRS symbols are processed *
      * otherwise LLR are calculated for each symbol based upon DMRS channel estimates only. */
      if (gNB && rel15_ul->pdu_bit_map & PUSCH_PDU_BITMAP_PUSCH_PTRS) {
        if (gNB) start_meas(&gNB->ulsch_ptrs_processing_stats);
        nr_pusch_ptrs_processing(gNB,ue,
                                 frame_parms,
                                 rel15_ul,
				 pssch_pdu, 
                                 ulsch_id,
                                 slot,
                                 symbol,
                                 nb_re_pusch);
        if (gNB) stop_meas(&gNB->ulsch_ptrs_processing_stats);

        /*  Subtract total PTRS RE's in the symbol from PUSCH RE's */
        pusch_vars->ul_valid_re_per_slot[symbol] -= pusch_vars->ptrs_re_per_slot;
      }

      /*---------------------------------------------------------------------------------------------------- */
      /*--------------------  LLRs computation  -------------------------------------------------------------*/
      /*-----------------------------------------------------------------------------------------------------*/
      if (gNB) start_meas(&gNB->ulsch_llr_stats);
      int sci1_offset=0;
      if (symbol <= pssch_pdu->pscch_numsym) { 
        pusch_vars->ul_valid_re_per_slot[symbol] -= sci1_re_per_symb;
        sci1_offset=sci1_re_per_symb;
      }
      if (ml_rx == false || nrOfLayers == 1) {       
        if (pssch_pdu && sci2_left>0){
	  LOG_D(NR_PHY, "valid_re_per_slot[%d] %d\n", symbol, pusch_vars->ul_valid_re_per_slot[symbol]);
	  int available_sci2_res_in_symb = pusch_vars->ul_valid_re_per_slot[symbol];
	  int slsch_res_in_symbol;
	  LOG_D(NR_PHY,"available_sci2_res_in_symb[%d] %d (sci1_re %d)\n",symbol,available_sci2_res_in_symb,sci1_re_per_symb);
	  int sci2_cnt_prev = sci2_cnt;
	  if (available_sci2_res_in_symb < sci2_left) {
	     sci2_cnt += available_sci2_res_in_symb; // take all of the PSSCH REs for SCI2
	     memcpy(&sci2_llrs[2*sci2_cnt_prev],&pusch_vars->rxdataF_comp[0][(symbol * (off + rb_size * NR_NB_SC_PER_RB))+sci1_offset],
	                       available_sci2_res_in_symb*sizeof(int32_t));
             sci2_left-= available_sci2_res_in_symb;
	     LOG_D(NR_PHY,"SCI2 taking all available REs. sci2_left %d\n",sci2_left);
	     pusch_vars->ul_valid_re_per_slot[symbol] = 0;
	     sci2_cnt_thissymb=available_sci2_res_in_symb;
	  }
	  else { // we finish SCI2 off here
	       memcpy(&sci2_llrs[2*sci2_cnt_prev],&pusch_vars->rxdataF_comp[0][(symbol * (off + rb_size * NR_NB_SC_PER_RB))+sci1_re_per_symb],
			         sci2_left*sizeof(int32_t));
	       slsch_res_in_symbol=available_sci2_res_in_symb-sci2_left;
	       LOG_D(NR_PHY, "SCI2 taking %d REs, SLSCH taking %d\n", sci2_left, slsch_res_in_symbol);
	       pusch_vars->ul_valid_re_per_slot[symbol]=slsch_res_in_symbol;
	       sci2_cnt_thissymb=sci2_left;
               sci2_left=0;
	       //for (int i=0;i<sci2_re;i++) LOG_I(NR_PHY,"sci2_llrs [%d] %d,%d\n",i,sci2_llrs[i<<1],sci2_llrs[1+(i<<1)]);
	       //unscramble the SCI2 payload
	       nr_pdcch_unscrambling(sci2_llrs, 1010,sci2_re*2,pssch_pdu->Nid,unscrambled_sci2_llrs,1);
	  //     for (int i=0;i<sci2_re;i++) LOG_I(NR_PHY,"sci2_llrs [%d] %d,%d\n",i,unscrambled_sci2_llrs[i<<1],unscrambled_sci2_llrs[1+(i<<1)]);

	       uint64_t sci_estimation[2]={0};
         uint16_t dummy;
         uint16_t crc = polar_decoder_int16(unscrambled_sci2_llrs,
                                            sci_estimation,
                                            &dummy,
                                            1,
                                            NR_POLAR_SCI2_MESSAGE_TYPE,
                                            pssch_pdu->sci2_len,
                                            sci2_re);
	       // send SCI indication with SCI2 payload and get SLSCH information if CRC is OK
	       LOG_D(NR_PHY,"SCI indication (crc %x)\n",crc);
	       if (crc==0) ue->SL_UE_PHY_PARAMS.pssch.rx_sci2_ok++;   
	       else        ue->SL_UE_PHY_PARAMS.pssch.rx_sci2_errors++;   
	       sl_nr_sci_indication_t sci_ind={0}; 
               sci_ind.sfn = frame;
               sci_ind.slot = slot;
               sci_ind.sensing_result = 0;
               sci_ind.pssch_rsrp = 0; // setting this flag to zero; measuring from sci1
               sci_ind.sci_pdu[sci_ind.number_of_SCIs].sci_format_type = SL_SCI_FORMAT_2_ON_PSSCH;
               sci_ind.sci_pdu[sci_ind.number_of_SCIs].subch_index = 0;
               sci_ind.sci_pdu[sci_ind.number_of_SCIs].pscch_rsrp = 0; // setting this flag to zero; measuring from sci1
               sci_ind.sci_pdu[sci_ind.number_of_SCIs].sci_payloadlen = pssch_pdu->sci2_len;
               sci_ind.sci_pdu[sci_ind.number_of_SCIs].Nid = dummy&65535;
 
               memcpy(sci_ind.sci_pdu[sci_ind.number_of_SCIs].sci_payloadBits,&sci_estimation,8);
               sci_ind.number_of_SCIs++;
	       nr_sidelink_indication_t sl_indication;
	       nr_fill_sl_indication(&sl_indication, NULL, &sci_ind, proc, ue, phy_data);
	       ue->if_inst->sl_indication(&sl_indication);
	       LOG_D(NR_PHY,"Returning from SCI2 SL indication\n");
               //
	  }
        } // (not ML || nrOfLayers==1 ) AND pssch and sci2 REs to handle	
	if (pssch_pdu) LOG_D(NR_PHY, "symbol %d: PSSCH REs %d (sci1 %d,sci2 %d)\n", symbol, pusch_vars->ul_valid_re_per_slot[symbol], sci1_offset, sci2_cnt_thissymb);
        for (aatx=0; aatx < nrOfLayers; aatx++) {
          if ((sci1_offset > 0 || sci2_cnt_thissymb > 0) && (qam_mod_order > 2)) {
            memset(temp_symbol, 0, (sci1_offset + sci2_cnt_thissymb) * sizeof(int32_t));
            memcpy(temp_symbol + sci1_offset + sci2_cnt_thissymb,
                  &pusch_vars->rxdataF_comp[aatx * frame_parms->nb_antennas_rx][symbol * (off + rb_size * NR_NB_SC_PER_RB) + sci1_offset+sci2_cnt_thissymb],
                  (rb_size * NR_NB_SC_PER_RB - (sci1_offset + sci2_cnt_thissymb)) * sizeof(int32_t));
            nr_ulsch_compute_llr(temp_symbol,
                                 pusch_vars->ul_ch_mag0[aatx * frame_parms->nb_antennas_rx],
                                 pusch_vars->ul_ch_magb0[aatx * frame_parms->nb_antennas_rx],
                                 pusch_vars->ul_ch_magc0[aatx * frame_parms->nb_antennas_rx],
                                 temp_llr,
                                 rb_size,
                                 rb_size * NR_NB_SC_PER_RB,
                                 symbol,
                                 qam_mod_order);
            memcpy(&pusch_vars->llr_layers[aatx][rxdataF_ext_offset * qam_mod_order],
                   temp_llr + (sci1_offset + sci2_cnt_thissymb) * qam_mod_order,
                   (rb_size * NR_NB_SC_PER_RB - (sci1_offset + sci2_cnt_thissymb)) * 2 * qam_mod_order);
          } else {
            nr_ulsch_compute_llr(&pusch_vars->rxdataF_comp[aatx * frame_parms->nb_antennas_rx][symbol * (off + rb_size * NR_NB_SC_PER_RB) + sci1_offset + sci2_cnt_thissymb],
                                pusch_vars->ul_ch_mag0[aatx * frame_parms->nb_antennas_rx],
                                pusch_vars->ul_ch_magb0[aatx * frame_parms->nb_antennas_rx],
                                pusch_vars->ul_ch_magc0[aatx * frame_parms->nb_antennas_rx],
                                &pusch_vars->llr_layers[aatx][rxdataF_ext_offset * qam_mod_order],
                                rb_size,
                                pusch_vars->ul_valid_re_per_slot[symbol],
                                symbol,
                                qam_mod_order);
          }
        }
      } else { // this is MIMO case with ML
	if (pssch_pdu) AssertFatal(1==0,"We need to handle the MIMO case for SCI2\n");      
        nr_ulsch_compute_ML_llr(pusch_vars->rxdataF_comp,
                                pusch_vars->ul_ch_mag0,
                                pusch_vars->rho,
                                pusch_vars->llr_layers,
                                frame_parms->nb_antennas_rx,
                                rb_size,
                                nb_re_pusch,
                                symbol,
                                rxdataF_ext_offset,
                                qam_mod_order);

        if (qam_mod_order == 2) {
          nr_ulsch_shift_llr(pusch_vars->llr_layers, nb_re_pusch, rxdataF_ext_offset, qam_mod_order, 4);
        }

#ifdef ML_DEBUG
        c16_t *llr_layers0 = (c16_t *)&pusch_vars->llr_layers[0][rxdataF_ext_offset * qam_mod_order];
        c16_t *llr_layers1 = (c16_t *)&pusch_vars->llr_layers[1][rxdataF_ext_offset * qam_mod_order];
        printf("===============================\n");
        printf("AFTER nr_ulsch_compute_ML_llr()\n");
        printf("===============================\n");
        for (int k = 0; k < nb_re_pusch; k++) {
          printf("[%3i] llr_layers0 = (%6i, %6i), llr_layers1 = (%6i, %6i)\n",
                 k, llr_layers0[k].r, llr_layers0[k].i, llr_layers1[k].r, llr_layers1[k].i);
        }
        printf("\n");
#endif
      }
      stop_meas(&gNB->ulsch_llr_stats);
      rxdataF_ext_offset += pusch_vars->ul_valid_re_per_slot[symbol];
    }
  } // symbol loop
  free(temp_llr);
  free(temp_symbol);
}
