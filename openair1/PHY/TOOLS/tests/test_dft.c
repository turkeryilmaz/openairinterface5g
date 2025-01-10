#include <stdio.h>
#include <math.h>
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "common/utils/utils.h"

// should be FOREACH_DFTSZ but we restrict to tested DFT sizes
#define FOREACH_DFTSZ_working(SZ_DEF) \
  SZ_DEF(12)                          \
  SZ_DEF(64)                          \
  SZ_DEF(128)                         \
  SZ_DEF(256)                         \
  SZ_DEF(512)                         \
  SZ_DEF(768)                         \
  SZ_DEF(1024)                        \
  SZ_DEF(1536)                        \
  SZ_DEF(2048)                        \
  SZ_DEF(3072)                        \
  SZ_DEF(4096)                        \
  SZ_DEF(6144)                        \
  SZ_DEF(8192)                        \
  SZ_DEF(12288)

#define SZ_PTR(Sz) {Sz},
struct {
  int size;
} const dftFtab[] = {FOREACH_DFTSZ_working(SZ_PTR)};

#define SZ_iPTR(Sz) {Sz},
struct {
  int size;
} const idftFtab[] = {FOREACH_IDFTSZ(SZ_iPTR)};

bool error(c16_t v16, cd_t vd, double percent)
{
  cd_t err = {abs(v16.r - vd.r), abs(v16.i - vd.i)};
  if (err.r < 10 && err.i < 10)
    return false; // ignore quantization noise
  int denomr = min(abs(v16.r), abs(vd.r));
  if (denomr && err.r / denomr > percent / 100)
    return true;
  int denomi = min(abs(v16.i), abs(vd.i));
  if (denomi && err.i / denomi > percent / 100)
    return true;
  return false;
}

void math_dft(cd_t *in, cd_t *out, int len,int dir)
{
  for (int k = 0; k < len; k++) {
    cd_t tmp = {0};
    // wrote this way to help gcc to generate SIMD
    double phi[len], sint[len], cost[len];
    for (int n = 0; n < len; n++)
      if (dir ==0) phi[n] = -2 * M_PI * ((double)k / len) * n;
      else         phi[n] =  2* M_PI * ((double)k/len)*n;
    for (int n = 0; n < len; n++)
      sint[n] = sin(phi[n]);
    for (int n = 0; n < len; n++)
      cost[n] = cos(phi[n]);
    for (int n = 0; n < len; n++) {
      cd_t coeff = {.r = cost[n], .i = sint[n]};
      cd_t component = cdMul(coeff, in[n]);
      tmp.r += component.r;
      tmp.i += component.i;
    }
    out[k].r = tmp.r / sqrt(len);
    out[k].i = tmp.i / sqrt(len);
  }
}

void fill_qam(int n, cd_t *x, int mod) {
  int size;
  if (mod < 0 || mod >1) {
    printf("Illegal modulation %d\n",mod);
    exit(-1);
  }
  double sqrt170 = 1.0/sqrt(170);
  memset((void*)&x[0],0,n*sizeof(cd_t));
  switch (n) {
    case 128:  size=72;   break;
    case 256:  size=180;  break;
    case 512:  size=300;  break;
    case 768:  size=612;  break;
    case 1024: size=612;  break;
    case 1536: size=900;  break;
    case 2048: size=1596; break;
    case 3072: size=2556; break;
    case 4096: size=3276; break;
    default:   printf("Illegal FFT length %d\n",n); exit(-1);;
  }
  for (int i=0;i<size/2;i++) {
    if (mod==0) {
      int rv=taus()&1;
      x[i].r = (1/sqrt(2.0)) * ((rv<<1) - 1); 
      rv=taus()&1;
      x[i].i = (1/sqrt(2.0)) * ((rv<<1) - 1);
    }
    else {
      int rvi=taus()&15;
      int rvq=taus()&15;
      x[i].r   = ((1-2*(rvi&1))*(8-(1-2*((rvi>>1)&1))*(4-(1-2*((rvi>>2)&1))*(2-(1-2*((rvi>>3)&1))))))*sqrt170;
      x[i].i   = ((1-2*(rvq&1))*(8-(1-2*((rvq>>1)&1))*(4-(1-2*((rvq>>2)&1))*(2-(1-2*((rvq>>3)&1))))))*sqrt170;
    }
  } 
  for (int i=n-(size/2);i<n;i++) {
    if (mod==0) {
      int rv=taus()&1;
      x[i].r = (1/sqrt(2.0)) * ((rv<<1) - 1); 
      rv=taus()&1;
      x[i].i = (1/sqrt(2.0)) * ((rv<<1) - 1);
    }
    else {
      int rvi=taus()&15;
      int rvq=taus()&15;
      x[i].r   = ((1-2*(rvi&1))*(8-(1-2*((rvi>>1)&1))*(4-(1-2*((rvi>>2)&1))*(2-(1-2*((rvi>>3)&1))))))*sqrt170;
      x[i].i   = ((1-2*(rvq&1))*(8-(1-2*((rvq>>1)&1))*(4-(1-2*((rvq>>2)&1))*(2-(1-2*((rvq>>3)&1))))))*sqrt170;
    }
  }
}


int main(void)
{
  int ret = 0;
  load_dftslib();
  c16_t *d16 = malloc16(12 * dftFtab[sizeofArray(dftFtab) - 1].size * sizeof(*d16));
  c16_t *o16 = malloc16(12 * dftFtab[sizeofArray(dftFtab) - 1].size * sizeof(*d16));
  for (int sz = 0; sz < sizeofArray(dftFtab); sz++) {
    const int n = dftFtab[sz].size;
    cd_t data[n];
    double coeffs[] = {0.25, 0.5, 1, 1.5, 2, 2.5, 3};
    printf("Testing size %d\n",n);
    cd_t out[n];
    for (int i = 0; i < n; i++) {
      data[i].r = gaussZiggurat(0, 1.0); // gaussZiggurat not used paramters, to fix
      data[i].i = gaussZiggurat(0, 1.0);
    }
    math_dft(data, out, n,0);
    double evm[sizeofArray(coeffs)] = {0};
    double sqnr[sizeofArray(coeffs)] = {0};
    double samples[sizeofArray(coeffs)] = {0};
    for (int coeff = 0; coeff < sizeofArray(coeffs); coeff++) {
      double expand = coeffs[coeff] * SHRT_MAX / sqrt(n);
      if (n == 12) {
        for (int i = 0; i < n; i++)
          for (int j = 0; j < 4; j++) {
            d16[i * 4 + j].r = expand * data[i].r;
            d16[i * 4 + j].i = expand * data[i].i;
          }
      } else {
        for (int i = 0; i < n; i++) {
          d16[i].r = expand * data[i].r;
          d16[i].i = expand * data[i].i;
        }
      }
      dft(get_dft(n), (int16_t *)d16, (int16_t *)o16,get_dft_scaling(n,(int32_t)(10*log10(expand))));
      if (n == 12) {
        for (int i = 0; i < n; i++) {
          cd_t error = {.r = o16[i * 4].r / (expand * sqrt(n)) - out[i].r, .i = o16[i * 4].i / (expand * sqrt(n)) - out[i].i};
          evm[coeff] += sqrt(squaredMod(error)) / sqrt(squaredMod(out[i]));
          samples[coeff] += sqrt(squaredMod(d16[i]));
        }
      } else {
        for (int i = 0; i < n; i++) {
          cd_t error2 = {.r = o16[i].r / expand - out[i].r , .i = o16[i].i / expand - out[i].i};
          evm[coeff] += sqrt(squaredMod(error2)) / sqrt(squaredMod(out[i]));
          sqnr[coeff] += squaredMod(out[i]) / (squaredMod(error2));
          samples[coeff] += sqrt(squaredMod(d16[i]));
      /*    if (n==64){ 
            if (error(o16[i], out[i], 5))
            printf("Error in dft %d at %d, coeff %d, expand %f, (%f, %f) != %f, %f)\n", n, i, coeff, expand,o16[i].r/expand, o16[i].i/expand, out[i].r, out[i].i);
          }*/
        }
      }
    }
    printf("done DFT size %d (evm (%%), SQNRdB, avg samples amplitude) = ", n);
    for (int coeff = 0; coeff < sizeofArray(coeffs); coeff++)
      printf("dBFS %f (%.2f, %f, %.0f) ", 20*log10(coeffs[coeff] / sqrt(n)),(evm[coeff] / n) * 100, 10*log10(sqnr[coeff]/n),samples[coeff] / n);
    printf("\n");
    int i;
    for (i = 0; i < sizeofArray(coeffs); i++)
      if (evm[i] / n < 0.01)
        break;
    if (i == sizeofArray(coeffs)) {
      printf("DFT size: %d, minimum error is more than 1%%, setting the test as failed\n", n);
      ret = 1;
    }
    fflush(stdout);
  }

  for (int sz = 0; sz < sizeofArray(dftFtab); sz++) {
    const int n = dftFtab[sz].size;
    cd_t data[n];
    cd_t in[n];
    if (n > 4096) break;
    if (n < 128) continue;
    printf("Testing IDFT size %d\n",n);
    cd_t out[n];
    for (int mod=0;mod<2;mod++) {
      fill_qam(n,data,mod);
      int16_t amp=512;
      for (int i = 0; i < n; i++) {
        d16[i].r = amp*data[i].r; 
        d16[i].i = amp*data[i].i;
      }
      idft(get_idft(n), (int16_t *)d16, (int16_t *)o16,get_idft_scaling(n));
      for (int i =0; i < n; i++) {
        in[i].r = (double)o16[i].r; 
        in[i].i = (double)o16[i].i;
      }
      math_dft(in, out, n,0);
      double evm = 0;
      double sqnr = 0;
      double samples = 0;
      int nz=0;
      for (int i = 0; i < n; i++) {
        if (data[i].r != 0) {
            cd_t error2 = {.r = d16[i].r - out[i].r, .i = d16[i].i - out[i].i};
            evm += sqrt(squaredMod(error2)) / sqrt(squaredMod(out[i]));
            sqnr += squaredMod(out[i]) / (squaredMod(error2));
            samples += sqrt(squaredMod(d16[i]));
            nz++;
        }
      }
      printf("done IDFT size %d nz %d mod %s (evm (%%), SQNRdB, avg samples amplitude) = ", n,nz, mod==0?"QPSK":"256QAM");
      printf("(%.2f, %f, %.0f) ", (evm / n) * 100, 10*log10(sqnr/nz),samples/ nz);
      printf("\n");
      if (evm / nz > 0.01){
        printf("IDFT size: %d/ mod %s, minimum error is more than 1%%, setting the test as failed\n", n, mod==0?"QPSK":"256QAM");
        ret = 1;
        break;
      }
    }
    fflush(stdout);
  }
  free(d16);
  free(o16);
  return ret;
}
