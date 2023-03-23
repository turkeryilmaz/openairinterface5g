#include "PHY/defs_gNB.h"
#include "PHY/phy_extern.h"
#include "nr_transport_proto.h"
#include "PHY/impl_defs_top.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "PHY/defs_nr_common.h"
#include "common/utils/nr/nr_common.h"
#include "openair1/PHY/NR_TRANSPORT/nr_transport_common_proto.h"

//#define DEBUG_CH_COMP
//#define DEBUG_RB_EXT
//#define DEBUG_CH_MAG

#define INVALID_VALUE 255

void nr_idft(int32_t *z, uint32_t Msc_PUSCH)
{

#if defined(__x86_64__) || defined(__i386__)
  __m128i idft_in128[1][3240], idft_out128[1][3240];
  __m128i norm128;
#elif defined(__arm__)
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
#elif defined(__arm__)
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
#elif defined(__arm__)
      norm128 = vdupq_n_s16(9459);
#endif

      for (i = 0; i < 12; i++) {
#if defined(__x86_64__)||defined(__i386__)
        ((__m128i*)idft_out0)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)idft_out0)[i], norm128), 1);
#elif defined(__arm__)
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
#elif defined(__arm__)
      *&(((int16x8_t*)z)[i]) = vmulq_s16(*&(((int16x8_t*)z)[i]), *(int16x8_t*)&conjugate2[0]);
#endif
    }
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
    sum_det += ((((int *)&det[0])[k])>>2);
    //printf("det_%d = %d log2 =%d \n",k,(((int *)&det[0])[k]),log2_approx(((int *)&det[0])[k]));
    }

  xy_re_128 = _mm_slli_epi32(xy_re_128,5);
  xy_re_128 = _mm_srai_epi32(xy_re_128,log2_approx(sum_det));
  xy_re_128 = _mm_slli_epi32(xy_re_128,5);

  xy_im_128 = _mm_slli_epi32(xy_im_128,5);
  xy_im_128 = _mm_srai_epi32(xy_im_128,log2_approx(sum_det));
  xy_im_128 = _mm_slli_epi32(xy_im_128,5);

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
