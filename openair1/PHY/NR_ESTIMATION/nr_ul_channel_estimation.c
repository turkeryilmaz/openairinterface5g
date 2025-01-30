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
#include <lapacke.h>
#include <complex.h>
#include <float.h>

#include "nr_ul_estimation.h"
#include "PHY/sse_intrin.h"
#include "PHY/NR_REFSIG/nr_refsig.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_UE_TRANSPORT/srs_modulation_nr.h"
#include "PHY/NR_UE_ESTIMATION/filt16a_32.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/NR_REFSIG/ul_ref_seq_nr.h"
#include "executables/softmodem-common.h"
#include "nr_phy_common.h"

//#define DEBUG_CH
//#define DEBUG_PUSCH
//#define SRS_DEBUG

#define NO_INTERP 1
#define  NR_SRS_IDFT_OVERSAMP_FACTOR 1
#define dBc(x,y) (dB_fixed(((int32_t)(x))*(x) + ((int32_t)(y))*(y)))

#define N_AP 1 // Number of antenna ports
#define N_FREQ 2048 // Number of frequency bins
void rootmusic_toa(int source_count, double *eigval, double complex **eigvec, 
                   double *source_delays, double *source_powers);

int rissanen_mdl_epsilon(double *eigenvalues, int M_len, int L_len, int use_fbcm);
void save_text(const char *filename, double complex **matrix, int rows, int cols);
void save_eigenvalues_text(const char *filename, double *eigval, int size);

void save_text(const char *filename, double complex **matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%d %d\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.6f %.6fi ", creal(matrix[i][j]), cimag(matrix[i][j]));
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void save_eigenvalues_text(const char *filename, double *eigval, int size) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%.6f\n", eigval[i]);
    }
    fclose(file);
}

// Function to estimate the number of multipath components using MDL
int rissanen_mdl_epsilon(double *eigenvalues, int M_len, int L_len, int use_fbcm) {
    double epsilon = 1e-10; // Small value to prevent log(0)
    double *sorted_eigenvalues = (double *)malloc(N_FREQ * sizeof(double));
    for (int i = 0; i < N_FREQ; i++) {
        sorted_eigenvalues[i] = eigenvalues[N_FREQ - 1 - i]+epsilon;  // Correct full reversal
    }
    save_eigenvalues_text("eigenvalues_desc.txt", sorted_eigenvalues, N_FREQ);
    int M = M_len;
    double mdl[L_len];
    int estimated_order = 0;
    double min_mdl = DBL_MAX;

    for (int k = 0; k < L_len; k++) {
        double sum_log_eigenvalues = 0.0, log_sum_eigenvalues = 0.0, total_sum = 0.0;

        for (int i = k; i < L_len; i++) {
            sum_log_eigenvalues += log(sorted_eigenvalues[i]);
            total_sum += sorted_eigenvalues[i];
        }

        sum_log_eigenvalues /= (L_len - k);
        log_sum_eigenvalues = log(total_sum / (L_len - k));

        mdl[k] = -M * (L_len - k) * (sum_log_eigenvalues - log_sum_eigenvalues);

        if (use_fbcm) {
            mdl[k] += (0.25) * k * (2 * L_len - k + 1) * log(M);
        } else {
            mdl[k] += (0.5) * k * (2 * L_len - k) * log(M);
        }

        if (mdl[k] < min_mdl) {
            min_mdl = mdl[k];
            estimated_order = k;
        }
    }

    free(sorted_eigenvalues);
    return estimated_order;
}

// Function to compute Root-MUSIC delays and powers
void rootmusic_toa(int source_count, double *eigval, double complex **eigvec, 
                   double *source_delays, double *source_powers) {
    
    // Step 1: Compute noise subspace Qn (last columns of eigvec)
    int noise_dim = N_FREQ - source_count;
    double complex **Qn = (double complex **)malloc(N_FREQ * sizeof(double complex *));
    for (int i = 0; i < N_FREQ; i++) {
        Qn[i] = (double complex *)malloc(noise_dim * sizeof(double complex));
        for (int j = 0; j < noise_dim; j++) {
            Qn[i][j] = eigvec[i][source_count + j];  // Select last noise_dim columns
        }
    }

    // Step 2: Compute C = Qn * Qn^H
    double complex **C = (double complex **)malloc(N_FREQ * sizeof(double complex *));
    for (int i = 0; i < N_FREQ; i++) {
        C[i] = (double complex *)malloc(N_FREQ * sizeof(double complex));
        for (int j = 0; j < N_FREQ; j++) {
            double complex sum = 0.0 + 0.0 * I;
            for (int k = 0; k < noise_dim; k++) {
                sum += Qn[i][k] * conj(Qn[j][k]);
            }
            C[i][j] = sum;
        }
    }

    // Step 3: Compute coefficients for polynomial equation
    double complex *coeffs = (double complex *)malloc((N_FREQ + 1) * sizeof(double complex));
    if (!coeffs) {
      printf("Memory allocation failed for coeffs\n");
      exit(EXIT_FAILURE);
    }
    for (int diag = 1; diag < N_FREQ; diag++) {
        double complex trace_sum = 0.0 + 0.0 * I;
        for (int i = 0; i < N_FREQ - diag; i++) {
            trace_sum += C[i][i + diag];
        }
        coeffs[N_FREQ - diag] = trace_sum;
    }
    coeffs[N_FREQ] = 1.0;  // Middle coefficient
    for (int i = 1; i < N_FREQ; i++) {
        coeffs[i - 1] = conj(coeffs[N_FREQ - i]);  // Symmetric polynomial
    }

    // Step 4: Solve for polynomial roots using LAPACK
    lapack_int info;
    lapack_int degree = N_FREQ;
    double complex *roots = (double complex *)malloc(N_FREQ * sizeof(double complex));
    
    info = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'N', degree, coeffs, degree, roots, NULL, degree, NULL, degree);
    if (info > 0) {
        printf("Eigenvalue computation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Step 5: Compute powers and filter valid roots
    double *powers = (double *)malloc(N_FREQ * sizeof(double));
    int valid_root_count = 0;
    for (int i = 0; i < N_FREQ; i++) {
        if (cabs(roots[i]) < 1.0) {  // Filter stable roots
            powers[valid_root_count] = 1.0 / (1.0 - cabs(roots[i]));
            source_delays[valid_root_count] = -N_FREQ * carg(roots[i]) / (2.0 * M_PI);
            valid_root_count++;
        }
    }

    // Step 6: Sort by power (strongest to weakest)
    for (int i = 0; i < source_count - 1; i++) {
        for (int j = i + 1; j < source_count; j++) {
            if (powers[i] < powers[j]) {
                double temp_power = powers[i];
                double temp_delay = source_delays[i];
                powers[i] = powers[j];
                source_delays[i] = source_delays[j];
                powers[j] = temp_power;
                source_delays[j] = temp_delay;
            }
        }
    }

    // Copy sorted powers to output
    for (int i = 0; i < source_count; i++) {
        source_powers[i] = powers[i];
    }

    // Free memory
    free(coeffs);
    free(roots);
    free(powers);
    for (int i = 0; i < N_FREQ; i++) {
        free(Qn[i]);
        free(C[i]);
    }
    free(Qn);
    free(C);
}

// Allocate Rxx dynamically
double complex **allocate_matrix(int rows, int cols) {
    double complex **matrix = (double complex **)malloc(rows * sizeof(double complex *));
    if (!matrix) {
        printf("Memory allocation failed for Rxx rows.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        matrix[i] = (double complex *)malloc(cols * sizeof(double complex));
        if (!matrix[i]) {
            printf("Memory allocation failed for Rxx[%d].\n", i);
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

// Free memory
void free_matrix(double complex **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void compute_covariance_matrix(double complex chF_reconstructed[N_AP][N_FREQ], double complex R[N_FREQ][N_FREQ]) {
    memset(R, 0, sizeof(double complex) * N_FREQ * N_FREQ); // Initialize to zero

    for (int f1 = 0; f1 < N_FREQ; f1++) {
        for (int f2 = 0; f2 < N_FREQ; f2++) {
            double complex sum = 0.0 + 0.0 * I; // Initialize sum as complex number
            for (int ap = 0; ap < N_AP; ap++) {
                sum += chF_reconstructed[ap][f1] * conj(chF_reconstructed[ap][f2]);
            }
            R[f1][f2] = sum; // Store the covariance result
        }
    }
}

int nr_est_toa_ns_srs_music(NR_DL_FRAME_PARMS *frame_parms,
		          uint8_t N_arx,
		          uint8_t N_ap,
              uint8_t N_symb_srs,
			        int32_t srs_estimated_channel_freq[N_arx][N_ap][frame_parms->ofdm_symbol_size * N_symb_srs],
			        int32_t *srs_toa_ns)
{

  int32_t chF_interpol[N_ap][NR_SRS_IDFT_OVERSAMP_FACTOR*frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
  int32_t chT_interpol[N_ap][NR_SRS_IDFT_OVERSAMP_FACTOR*frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
  int32_t chT_interpol_mag_squ_avg[NR_SRS_IDFT_OVERSAMP_FACTOR*frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
  memset(chF_interpol,0,sizeof(chF_interpol));
  memset(chT_interpol,0,sizeof(chT_interpol));

  int16_t start_offset = NR_SRS_IDFT_OVERSAMP_FACTOR*frame_parms->ofdm_symbol_size - (frame_parms->ofdm_symbol_size>>1);

  for (int arx_index = 0; arx_index < N_arx; arx_index++) {

    memset(chT_interpol_mag_squ_avg,0,sizeof(chT_interpol_mag_squ_avg));

    for (int symb = 0; symb < N_symb_srs; symb++){

      for (int ap_index = 0; ap_index < N_ap; ap_index++) {

        // Place SRS channel estimates in FFT shifted format for oversampling
        memcpy((int16_t *)&chF_interpol[ap_index][0], &srs_estimated_channel_freq[arx_index][ap_index][symb*frame_parms->ofdm_symbol_size], (frame_parms->ofdm_symbol_size>>1) * sizeof(int32_t));
        memcpy((int16_t *)&chF_interpol[ap_index][start_offset], &srs_estimated_channel_freq[arx_index][ap_index][symb*frame_parms->ofdm_symbol_size + (frame_parms->ofdm_symbol_size>>1)], (frame_parms->ofdm_symbol_size>>1) * sizeof(int32_t));

        // Convert to time domain oversampled
        freq2time(frame_parms->ofdm_symbol_size*NR_SRS_IDFT_OVERSAMP_FACTOR,
        (int16_t*) chF_interpol[ap_index],
        (int16_t*) chT_interpol[ap_index]);

        for(int k = 0; k < NR_SRS_IDFT_OVERSAMP_FACTOR*frame_parms->ofdm_symbol_size; k++) {

         chT_interpol_mag_squ_avg[k] += squaredMod(((c16_t*)chT_interpol[ap_index])[k]);

        } // Loop over samples
      } // antenna port loop
    } // SRS OFDM symbol loop


    // average over SRS symbols
    for(int k = 0; k < NR_SRS_IDFT_OVERSAMP_FACTOR*frame_parms->ofdm_symbol_size; k++) {
      chT_interpol_mag_squ_avg[k] /= N_symb_srs;
    }

    int32_t chF_reconstructed[N_ap][NR_SRS_IDFT_OVERSAMP_FACTOR*frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
    for (int ap_index = 0; ap_index < N_ap; ap_index++) {
        time2freq(frame_parms->ofdm_symbol_size * NR_SRS_IDFT_OVERSAMP_FACTOR,
                  (int16_t*) chT_interpol[ap_index],
                  (int16_t*) chF_reconstructed[ap_index]);
    }
    
    size_t num_elements = sizeof(chF_reconstructed) / sizeof(chF_reconstructed[0][0]);
    printf("size chF_reconstructed: %lu\n", num_elements);

    // Convert chF_reconstructed to complex format

    double complex chF_complex[N_AP][N_FREQ];
    for (int ap = 0; ap < N_AP; ap++) {
        for (int f = 0; f < N_FREQ; f++) {
            int16_t *real_part = (int16_t *)&chF_reconstructed[ap][f];
            int16_t *imag_part = real_part + 1;
            chF_complex[ap][f] = *real_part + (*imag_part) * I;
        }
    }
    printf("chF_complex[0][0] = %.2f + %.2fi\n", creal(chF_complex[0][0]), cimag(chF_complex[0][0]));

    double complex **Rxx = allocate_matrix(N_FREQ, N_FREQ);
    for (int i = 0; i < N_FREQ; i++) {
        for (int j = 0; j < N_FREQ; j++) {
            Rxx[i][j] = 0.0 + 0.0 * I;
        }
    }

    // Compute covariance matrix Rxx
    for (int f1 = 0; f1 < N_FREQ; f1++) {
        for (int f2 = 0; f2 < N_FREQ; f2++) {
            double complex sum = 0.0 + 0.0 * I;
            for (int ap = 0; ap < N_AP; ap++) {
                sum += chF_complex[ap][f1] * conj(chF_complex[ap][f2]);
            }
            Rxx[f1][f2] = sum;
        }
    }
    
    // FBCM (Forward Backward Covariance Matrix) to compute a complex symmetric matrix Rxx from the covariance matrix
    double complex **Rxx_fbcm = allocate_matrix(N_FREQ, N_FREQ); 
    for (int i = 0; i < frame_parms->ofdm_symbol_size; i++) {
      for (int j = 0; j < frame_parms->ofdm_symbol_size; j++) {
        double complex flipped_conj = conj(Rxx[frame_parms->ofdm_symbol_size - 1 - i][frame_parms->ofdm_symbol_size - 1 - j]); // Flip and conjugate
        Rxx_fbcm[i][j] = (Rxx[i][j] + flipped_conj) / 2.0; // Average
        }
      }

    // Debug: Print some values from covariance matrix
    printf("Rxx[0][0] = %.2f + %.2fi\n", creal(Rxx[0][0]), cimag(Rxx[0][0]));
    
    
    double complex *Rxx_flat = (double complex *)malloc(N_FREQ * N_FREQ * sizeof(double complex));
    if (!Rxx_flat) {
      printf("Memory allocation failed for Rxx_flat\n");
      exit(EXIT_FAILURE);
    }
    
    // Fill Rxx_flat from Rxx because LAPACK requires a 1D array
    for (int i = 0; i < N_FREQ; i++) {
        for (int j = 0; j < N_FREQ; j++) {
          Rxx_flat[i + j * N_FREQ] = Rxx_fbcm[i][j]; // Column-major format for LAPACK
        }
    }
    printf("Rxx_flat[0] = %.2f + %.2fi\n", creal(Rxx_flat[0]), cimag(Rxx_flat[0]));
    
    // Allocate arrays for eigenvalues and eigenvectors
    double *eigval = (double *)malloc(N_FREQ * sizeof(double));
    double complex **eigvec = (double complex **)malloc(N_FREQ * sizeof(double complex *));
    for (int i = 0; i < N_FREQ; i++) {
        eigvec[i] = (double complex *)malloc(N_FREQ * sizeof(double complex));
    }


    // LAPACK parameters
    char jobz = 'V';  // Compute both eigenvalues and eigenvectors
    char uplo = 'L';  // Lower triangular part is stored
    int lda = N_FREQ;      // Leading dimension of matrix
    int info1;            // Stores error info

    //info1 = LAPACKE_zheev(LAPACK_COL_MAJOR, jobz, uplo, N_FREQ, Rxx_flat, lda, eigval);
    // info1 = LAPACKE_zheevx() to be tested on N largest eigenvalues
    info1 = LAPACKE_zheevd(LAPACK_COL_MAJOR, jobz, uplo, N_FREQ, Rxx_flat, lda, eigval); //the Divide-and-Conquer version of LAPACKE_zheev() is faster
    if (info1 > 0) {
    printf("Eigenvalue computation failed: LAPACK zheev did not converge\n");
    exit(EXIT_FAILURE);
    }

    // Returning eigenvectors overwritten in Rxx_flat into 2D
    for (int i = 0; i < N_FREQ; i++) {
        for (int j = 0; j < N_FREQ; j++) {
            eigvec[i][j] = Rxx_flat[i + j * N_FREQ]; 
        }
    }    
    
    double *eigval_desc = (double *)malloc(N_FREQ * sizeof(double));
    for (int i = 0; i < N_FREQ / 2; i++) {
        double temp = eigval[i];
        eigval_desc[i] = eigval[N_FREQ - 1 - i];
        eigval_desc[N_FREQ - 1 - i] = temp;
    }

    printf("Eigenvalues:\n");
    for (int i = 0; i < 10; i++) {
        printf("λ[%d] = %.6f\n", i, eigval_desc[i]);
    }
    save_eigenvalues_text("eigenvalues.txt", eigval_desc, N_FREQ);
    int L_len = N_FREQ/2;
    int M_len = N_FREQ - L_len + 1;
    int use_fbcm =1;    
    int mpc = rissanen_mdl_epsilon(eigval,M_len,L_len,use_fbcm);
    printf("Estimated MPC: %d\n", mpc);
    /*
    int source_count = 1;

    // Step 1: Compute noise subspace Qn (last columns of eigvec)
    int noise_dim = N_FREQ - source_count;
    double complex **Qn = (double complex **)malloc(N_FREQ * sizeof(double complex *));
    for (int i = 0; i < N_FREQ; i++) {
        Qn[i] = (double complex *)malloc(noise_dim * sizeof(double complex));
        for (int j = 0; j < noise_dim; j++) {
            Qn[i][j] = eigvec[i][source_count + j];  // Select last noise_dim columns
        }
    }

    printf("Qn[0][0] = %.2f + %.2fi\n", creal(Qn[0][0]), cimag(Qn[0][0]));

    // Step 2: Compute C = Qn * Qn^H
    double complex **C = (double complex **)malloc(N_FREQ * sizeof(double complex *));
    for (int i = 0; i < N_FREQ; i++) {
        C[i] = (double complex *)malloc(N_FREQ * sizeof(double complex));
        for (int j = 0; j < N_FREQ; j++) {
            double complex sum = 0.0 + 0.0 * I;
            for (int k = 0; k < noise_dim; k++) {
                sum += Qn[i][k] * conj(Qn[j][k]);
            }
            C[i][j] = sum;
        }
    }
    printf("C[0][0] = %.2f + %.2fi\n", creal(C[0][0]), cimag(C[0][0]));
   
    // Step 3: Compute coefficients for polynomial equation
    double complex *coeffs = (double complex *)malloc((N_FREQ + 1) * sizeof(double complex));
    if (!coeffs) {
      printf("Memory allocation failed for coeffs\n");
      exit(EXIT_FAILURE);
    }
    for (int diag = 1; diag < N_FREQ; diag++) {
        double complex trace_sum = 0.0 + 0.0 * I;
        for (int i = 0; i < N_FREQ - diag; i++) {
            trace_sum += C[i][i + diag];
        }
        coeffs[N_FREQ - diag] = trace_sum;
    }
    coeffs[N_FREQ] = 1.0;  // Middle coefficient
    for (int i = 1; i < N_FREQ; i++) {
        coeffs[i - 1] = conj(coeffs[N_FREQ - i]);  // Symmetric polynomial
    }
    printf("coeffs[0] = %.2f + %.2fi\n", creal(coeffs[0]), cimag(coeffs[0]));

    // Step 4: Solve for polynomial roots using LAPACK
    lapack_int info2;
    lapack_int degree = N_FREQ;
    double complex *roots = (double complex *)malloc(N_FREQ * sizeof(double complex));
    // Allocate space for eigenvalues
    double complex *eigval2 = (double complex *)malloc(degree * sizeof(double complex));
    if (!eigval2) {
        printf("Memory allocation failed for eigval2\n");
        exit(EXIT_FAILURE);
    }

    // Allocate space for coefficient matrix (2D array in column-major format)
    double complex *coeffs_matrix = (double complex *)malloc(degree * degree * sizeof(double complex));
    if (!coeffs_matrix) {
        printf("Memory allocation failed for coeffs_matrix\n");
        free(eigval2);
        exit(EXIT_FAILURE);
    }
    // Copy `coeffs` (1D polynomial coefficients) into `coeffs_matrix`
    for (int i = 0; i < degree; i++) {
        for (int j = 0; j < degree; j++) {
            coeffs_matrix[i + j * degree] = coeffs[i * degree + j]; // Column-major order
        }
    }

    // Compute eigenvalues using LAPACK
    info2 = LAPACKE_zgeev(LAPACK_COL_MAJOR, 'N', 'N', degree, coeffs_matrix, degree, eigval2, NULL, 1, NULL, 1);
    if (info2 > 0) {
        printf("Eigenvalue computation failed: LAPACK zgeev did not converge\n");
        free(eigval2);
        free(coeffs_matrix);
        exit(EXIT_FAILURE);
        }
    // Debug Output
    printf("Eigenvalue[0] = %.6f + %.6fi\n", creal(eigval2[0]), cimag(eigval2[0]));   
    
    
    // Step 5: Compute powers and filter valid roots
    double *source_delays = (double *)malloc(N_FREQ * sizeof(double));
    double *powers = (double *)malloc(N_FREQ * sizeof(double));
    double *source_powers = (double *)malloc(N_FREQ * sizeof(double));

    if (!source_delays || !powers || !source_powers) {
        printf("Memory allocation failed for source_delays, powers, or source_powers\n");
        free(source_delays);
        free(powers);
        free(source_powers);
        exit(EXIT_FAILURE);
    }

    int valid_root_count = 0;
    for (int i = 0; i < N_FREQ; i++) {
        if (cabs(roots[i]) < 1.0) {  // Filter stable roots
            powers[valid_root_count] = 1.0 / (1.0 - cabs(roots[i]));
            source_delays[valid_root_count] = -N_FREQ * carg(roots[i]) / (2.0 * M_PI);
            valid_root_count++;
        }
    }

    // Resize dynamically to match `valid_root_count`
    source_delays = (double *)realloc(source_delays, valid_root_count * sizeof(double));
    source_powers = (double *)realloc(source_powers, valid_root_count * sizeof(double));

    if (!source_delays || !source_powers) {
        printf("Reallocation failed for source_delays or source_powers\n");
        free(source_delays);
        free(source_powers);
        free(powers);
        exit(EXIT_FAILURE);
    }

    // Step 6: Sort by power (strongest to weakest)
    for (int i = 0; i < source_count - 1; i++) {
        for (int j = i + 1; j < source_count; j++) {
            if (powers[i] < powers[j]) {
                double temp_power = powers[i];
                double temp_delay = source_delays[i];
                powers[i] = powers[j];
                source_delays[i] = source_delays[j];
                powers[j] = temp_power;
                source_delays[j] = temp_delay;
            }
        }
    }

    // Copy sorted powers to output
    for (int i = 0; i < source_count; i++) {
        source_powers[i] = powers[i];
    }

    // Debug: Print some values
    printf("powers[0] = %.2f\n", powers[0]);
    printf("source_delays[0] = %.2f\n", source_delays[0]);
    printf("source_powers[0] = %.2f\n", source_powers[0]);

    // Free allocated memory
    free(source_delays);
    free(source_powers);
    free(powers);
    
    free(Qn);
    free(C);
    */

    // Free covariance and eigenvalue data
    free(Rxx_flat);
    free_matrix(Rxx, N_FREQ);
    free(eigval);
    free_matrix(eigvec, N_FREQ);


  }
  return 0;
}

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

int nr_pusch_channel_estimation(PHY_VARS_gNB *gNB,
                                unsigned char Ns,
                                int nl,
                                unsigned short p,
                                unsigned char symbol,
                                int ul_id,
                                unsigned short bwp_start_subcarrier,
                                nfapi_nr_pusch_pdu_t *pusch_pdu,
                                int *max_ch,
                                uint32_t *nvar)
{
  c16_t pilot[3280] __attribute__((aligned(32)));
  const int chest_freq = gNB->chest_freq;

#ifdef DEBUG_CH
  FILE *debug_ch_est;
  debug_ch_est = fopen("debug_ch_est.txt","w");
#endif
  NR_gNB_PUSCH *pusch_vars = &gNB->pusch_vars[ul_id];
  c16_t **ul_ch_estimates = (c16_t **)pusch_vars->ul_ch_estimates;
  const int symbolSize = gNB->frame_parms.ofdm_symbol_size;
  const int slot_offset = (Ns & 3) * gNB->frame_parms.symbols_per_slot * symbolSize;
  const int delta = get_delta(p, pusch_pdu->dmrs_config_type);
  const int symbol_offset = symbolSize * symbol;
  const int k0 = bwp_start_subcarrier;
  const int nb_rb_pusch = pusch_pdu->rb_size;

  LOG_D(PHY, "symbol_offset %d, slot_offset %d, OFDM size %d, Ns = %d, k0 = %d, symbol %d\n",
        symbol_offset,
        slot_offset,
        symbolSize,
        Ns,
        k0,
        symbol);

  //------------------generate DMRS------------------//

  if (pusch_pdu->transform_precoding == transformPrecoder_disabled) {
    // Note: pilot returned by the following function is already the complex conjugate of the transmitted DMRS
    NR_DL_FRAME_PARMS *fp = &gNB->frame_parms;
    const uint32_t *gold = nr_gold_pusch(fp->N_RB_UL,
                                         fp->symbols_per_slot,
                                         gNB->gNB_config.cell_config.phy_cell_id.value,
                                         pusch_pdu->scid,
                                         Ns,
                                         symbol);
    nr_pusch_dmrs_rx(gNB,
                     Ns,
                     gold,
                     pilot,
                     (1000 + p),
                     0,
                     nb_rb_pusch,
                     (pusch_pdu->bwp_start + pusch_pdu->rb_start) * NR_NB_SC_PER_RB,
                     pusch_pdu->dmrs_config_type);
  } else { // if transform precoding or SC-FDMA is enabled in Uplink
    // NR_SC_FDMA supports type1 DMRS so only 6 DMRS REs per RB possible
    const int index = get_index_for_dmrs_lowpapr_seq(nb_rb_pusch * (NR_NB_SC_PER_RB / 2));
    const uint8_t u = pusch_pdu->dfts_ofdm.low_papr_group_number;
    const uint8_t v = pusch_pdu->dfts_ofdm.low_papr_sequence_number;
    c16_t *dmrs_seq = gNB_dmrs_lowpaprtype1_sequence[u][v][index];
    LOG_D(PHY,"Transform Precoding params. u: %d, v: %d, index for dmrsseq: %d\n", u, v, index);
    AssertFatal(index >= 0, "Num RBs not configured according to 3GPP 38.211 section 6.3.1.4. For PUSCH with transform precoding, num RBs cannot be multiple of any other primenumber other than 2,3,5\n");
    AssertFatal(dmrs_seq != NULL, "DMRS low PAPR seq not found, check if DMRS sequences are generated");
    nr_pusch_lowpaprtype1_dmrs_rx(gNB, Ns, dmrs_seq, pilot, 1000, 0, nb_rb_pusch, 0, pusch_pdu->dmrs_config_type);
#ifdef DEBUG_PUSCH
    printf ("NR_UL_CHANNEL_EST: index %d, u %d,v %d\n", index, u, v);
    LOG_M("gNb_DMRS_SEQ.m","gNb_DMRS_SEQ", dmrs_seq,6*nb_rb_pusch,1,1);
#endif
  }
  //------------------------------------------------//

#ifdef DEBUG_PUSCH

  for (int i = 0; i < (6 * nb_rb_pusch); i++) {
    LOG_I(PHY, "In %s: %d + j*(%d)\n", __FUNCTION__, pilot[i].r,pilot[i].i);
  }

#endif

  int nest_count = 0;
  uint64_t noise_amp2 = 0;
  c16_t ul_ls_est[symbolSize] __attribute__((aligned(32)));
  memset(ul_ls_est, 0, sizeof(c16_t) * symbolSize);
  delay_t *delay = &gNB->ulsch[ul_id].delay;
  memset(delay, 0, sizeof(*delay));

  for (int aarx=0; aarx<gNB->frame_parms.nb_antennas_rx; aarx++) {
    c16_t *rxdataF = (c16_t *)&gNB->common_vars.rxdataF[aarx][symbol_offset + slot_offset];
    c16_t *ul_ch = &ul_ch_estimates[nl * gNB->frame_parms.nb_antennas_rx + aarx][symbol_offset];

    memset(ul_ch, 0, sizeof(*ul_ch) * symbolSize);
#ifdef DEBUG_PUSCH
    LOG_I(PHY, "symbol_offset %d, delta %d\n", symbol_offset, delta);
    LOG_I(PHY, "ch est pilot, N_RB_UL %d\n", gNB->frame_parms.N_RB_UL);
    LOG_I(PHY,
          "bwp_start_subcarrier %d, k0 %d, first_carrier %d, nb_rb_pusch %d\n",
          bwp_start_subcarrier,
          k0,
          gNB->frame_parms.first_carrier_offset,
          nb_rb_pusch);
    LOG_I(PHY, "ul_ch addr %p \n", ul_ch);
#endif

    if (pusch_pdu->dmrs_config_type == pusch_dmrs_type1 && chest_freq == 0) {
      c16_t *pil   = pilot;
      int re_offset = k0;
      LOG_D(PHY,"PUSCH estimation DMRS type 1, Freq-domain interpolation");
      int pilot_cnt = 0;

      for (int n = 0; n < 3 * nb_rb_pusch; n++) {
        // LS estimation
        c32_t ch = {0};

        for (int k_line = 0; k_line <= 1; k_line++) {
          re_offset = (k0 + (n << 2) + (k_line << 1) + delta) % symbolSize;
          ch = c32x16maddShift(*pil, rxdataF[re_offset], ch, 16);
          pil++;
        }

        c16_t ch16 = {.r = (int16_t)ch.r, .i = (int16_t)ch.i};
        *max_ch = max(*max_ch, max(abs(ch.r), abs(ch.i)));
        for (int k = pilot_cnt << 1; k < (pilot_cnt << 1) + 4; k++) {
          ul_ls_est[k] = ch16;
        }
        pilot_cnt += 2;
      }

      nr_est_delay(gNB->frame_parms.ofdm_symbol_size, ul_ls_est, (c16_t *)pusch_vars->ul_ch_estimates_time[aarx], delay);
      int delay_idx = get_delay_idx(delay->est_delay, MAX_DELAY_COMP);
      c16_t *ul_delay_table = gNB->frame_parms.delay_table[delay_idx];

#ifdef DEBUG_PUSCH
      printf("Estimated delay = %i\n", delay->est_delay >> 1);
#endif

      pilot_cnt = 0;
      for (int n = 0; n < 3*nb_rb_pusch; n++) {

        // Channel interpolation
        for (int k_line = 0; k_line <= 1; k_line++) {

          // Apply delay
          int k = pilot_cnt << 1;
          c16_t ch16 = c16mulShift(ul_ls_est[k], ul_delay_table[k], 8);

#ifdef DEBUG_PUSCH
          re_offset = (k0 + (n << 2) + (k_line << 1)) % symbolSize;
          c16_t *rxF = &rxdataF[re_offset];
          printf("pilot %4d: pil -> (%6d,%6d), rxF -> (%4d,%4d), ch -> (%4d,%4d)\n",
                 pilot_cnt, pil->r, pil->i, rxF->r, rxF->i, ch16.r, ch16.i);
#endif

          if (pilot_cnt == 0) {
            c16multaddVectRealComplex(filt16_ul_p0, &ch16, ul_ch, 16);
          } else if (pilot_cnt == 1 || pilot_cnt == 2) {
            c16multaddVectRealComplex(filt16_ul_p1p2, &ch16, ul_ch, 16);
          } else if (pilot_cnt == (6 * nb_rb_pusch - 1)) {
            c16multaddVectRealComplex(filt16_ul_last, &ch16, ul_ch, 16);
          } else {
            c16multaddVectRealComplex(filt16_ul_middle, &ch16, ul_ch, 16);
            if (pilot_cnt % 2 == 0) {
              ul_ch += 4;
            }
          }

          pilot_cnt++;
        }
      }

      // Revert delay
      pilot_cnt = 0;
      ul_ch = &ul_ch_estimates[nl * gNB->frame_parms.nb_antennas_rx + aarx][symbol_offset];
      int inv_delay_idx = get_delay_idx(-delay->est_delay, MAX_DELAY_COMP);
      c16_t *ul_inv_delay_table = gNB->frame_parms.delay_table[inv_delay_idx];
      for (int n = 0; n < 3 * nb_rb_pusch; n++) {
        for (int k_line = 0; k_line <= 1; k_line++) {
          int k = pilot_cnt << 1;
          ul_ch[k] = c16mulShift(ul_ch[k], ul_inv_delay_table[k], 8);
          ul_ch[k + 1] = c16mulShift(ul_ch[k + 1], ul_inv_delay_table[k + 1], 8);
          noise_amp2 += c16amp2(c16sub(ul_ls_est[k], ul_ch[k]));
          noise_amp2 += c16amp2(c16sub(ul_ls_est[k + 1], ul_ch[k + 1]));

#ifdef DEBUG_PUSCH
          re_offset = (k0 + (n << 2) + (k_line << 1)) % symbolSize;
          printf("ch -> (%4d,%4d), ch_inter -> (%4d,%4d)\n", ul_ls_est[k].r, ul_ls_est[k].i, ul_ch[k].r, ul_ch[k].i);
#endif
          pilot_cnt++;
          nest_count += 2;
        }
      }

    } else if (pusch_pdu->dmrs_config_type == pusch_dmrs_type2 && chest_freq == 0) { // pusch_dmrs_type2  |p_r,p_l,d,d,d,d,p_r,p_l,d,d,d,d|
      LOG_D(PHY, "PUSCH estimation DMRS type 2, Freq-domain interpolation\n");
      c16_t *pil = pilot;
      c16_t *rx = &rxdataF[delta];
      for (int n = 0; n < nb_rb_pusch * NR_NB_SC_PER_RB; n += 6) {
        c16_t ch0 = c16mulShift(*pil, rx[(k0 + n) % symbolSize], 15);
        pil++;
        c16_t ch1 = c16mulShift(*pil, rx[(k0 + n + 1) % symbolSize], 15);
        pil++;
        c16_t ch = c16addShift(ch0, ch1, 1);
        *max_ch = max(*max_ch, max(abs(ch.r), abs(ch.i)));
        multadd_real_four_symbols_vector_complex_scalar(filt8_rep4, &ch, &ul_ls_est[n]);
        ul_ls_est[n + 4] = ch;
        ul_ls_est[n + 5] = ch;
        noise_amp2 += c16amp2(c16sub(ch0, ch));
        nest_count++;
      }

      // Delay compensation
      nr_est_delay(gNB->frame_parms.ofdm_symbol_size, ul_ls_est, (c16_t *)pusch_vars->ul_ch_estimates_time[aarx], delay);
      int delay_idx = get_delay_idx(-delay->est_delay, MAX_DELAY_COMP);
      c16_t *ul_delay_table = gNB->frame_parms.delay_table[delay_idx];
      for (int n = 0; n < nb_rb_pusch * NR_NB_SC_PER_RB; n++) {
        ul_ch[n] = c16mulShift(ul_ls_est[n], ul_delay_table[n % 6], 8);
      }

    }
    // this is case without frequency-domain linear interpolation, just take average of LS channel estimates of 6 DMRS REs and use a common value for the whole PRB
    else if (pusch_pdu->dmrs_config_type == pusch_dmrs_type1) {
      LOG_D(PHY,"PUSCH estimation DMRS type 1, no Freq-domain interpolation\n");
      c16_t *rxF = &rxdataF[delta];
      int pil_offset = 0;
      int re_offset = k0;
      c16_t ch;

      // First PRB
      ch = c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);

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

      for (int pilot_cnt=6; pilot_cnt<6*(nb_rb_pusch-1); pilot_cnt += 6) {
        ch = c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);
        *max_ch = max(*max_ch, max(abs(ch.r), abs(ch.i)));

#if NO_INTERP
      for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
          *ul_ch=ch;
#else
        ul_ch[3].r += (ch.r * 1365)>>15; // 1/12*16384
        ul_ch[3].i += (ch.i * 1365)>>15; // 1/12*16384

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
      ch=c32x16cumulVectVectWithSteps(pilot, &pil_offset, 1, rxF, &re_offset, 2, symbolSize, 6);

#if NO_INTERP
      for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
        *ul_ch=ch;
#else
      ul_ch[3].r += (ch.r * 1365)>>15; // 1/12*16384
      ul_ch[3].i += (ch.i * 1365)>>15; // 1/12*16384

      ul_ch += 4;
      c16multaddVectRealComplex(filt8_avlip3, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip6, &ch, ul_ch, 8);
#endif
    } else  { // this is case without frequency-domain linear interpolation, just take average of LS channel estimates of 4 DMRS REs and use a common value for the whole PRB
      LOG_D(PHY,"PUSCH estimation DMRS type 2, no Freq-domain interpolation");
      c16_t *pil = pilot;
      int re_offset = (k0 + delta) % symbolSize;
      c32_t ch0 = {0};
      //First PRB
      ch0 = c32x16mulShift(*pil, rxdataF[re_offset], 15);
      pil++;
      re_offset = (re_offset + 1) % symbolSize;
      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 5) % symbolSize;
      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      re_offset = (re_offset + 1) % symbolSize;
      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 5) % symbolSize;

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
        ch0 = c32x16mulShift(*pil, rxdataF[re_offset], 15);
        pil++;
        re_offset = (re_offset + 1) % symbolSize;

        ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset + 5) % symbolSize;

        ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset + 1) % symbolSize;

        ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
        pil++;
        re_offset = (re_offset+5) % symbolSize;

        ch=c16x32div(ch0, 4);
        *max_ch = max(*max_ch, max(abs(ch.r), abs(ch.i)));

#if NO_INTERP
        for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
          *ul_ch=ch;
#else
        ul_ch[3] = c16maddShift(ch, (c16_t){1365, 1365}, (c16_t){0, 0}, 15); // 1365 = 1/12*16384 (full range is +/- 32768)
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
      ch0 = c32x16mulShift(*pil, rxdataF[re_offset], 15);
      pil++;
      re_offset = (re_offset + 1) % symbolSize;

      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 5) % symbolSize;

      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 1) % symbolSize;

      ch0 = c32x16maddShift(*pil, rxdataF[re_offset], ch0, 15);
      pil++;
      re_offset = (re_offset + 5) % symbolSize;

      ch=c16x32div(ch0, 4);
#if NO_INTERP
      for (c16_t *end=ul_ch+12; ul_ch<end; ul_ch++)
          *ul_ch=ch;
#else
      ul_ch[3] = c16maddShift(ch, (c16_t){1365, 1365}, (c16_t){0, 0}, 15); // 1365 = 1/12*16384 (full range is +/- 32768)
      ul_ch += 4;
      c16multaddVectRealComplex(filt8_avlip3, &ch, ul_ch, 8);
      ul_ch += 8;
      c16multaddVectRealComplex(filt8_avlip6, &ch, ul_ch, 8);
#endif
    }

#ifdef DEBUG_PUSCH
    ul_ch = &ul_ch_estimates[nl * gNB->frame_parms.nb_antennas_rx + aarx][symbol_offset];
    for (int idxP = 0; idxP < ceil((float)nb_rb_pusch * 12 / 8); idxP++) {
      for (int idxI = 0; idxI < 8; idxI++) {
          printf("%d\t%d\t", ul_ch[idxP * 8 + idxI].r, ul_ch[idxP * 8 + idxI].i);
      }
      printf("%d\n", idxP);
    }
#endif

  }

#ifdef DEBUG_CH
  fclose(debug_ch_est);
#endif

  if (nvar && nest_count > 0) {
    *nvar = (uint32_t)(noise_amp2 / nest_count);
  }

  return 0;
}


/*******************************************************************
 *
 * NAME :         nr_pusch_ptrs_processing
 *
 * PARAMETERS :   gNB         : gNB data structure
 *                rel15_ul    : UL parameters
 *                UE_id       : UE ID
 *                nr_tti_rx   : slot rx TTI
 *            dmrs_symbol_flag: DMRS Symbol Flag
 *                symbol      : OFDM Symbol
 *                nb_re_pusch : PUSCH RE's
 *                nb_re_pusch : PUSCH RE's
 *
 * RETURN :       nothing
 *
 * DESCRIPTION :
 *  If ptrs is enabled process the symbol accordingly
 *  1) Estimate phase noise per PTRS symbol
 *  2) Interpolate PTRS estimated value in TD after all PTRS symbols
 *  3) Compensated DMRS based estimated signal with PTRS estimation for slot
 *********************************************************************/
// #define DEBUG_UL_PTRS
void nr_pusch_ptrs_processing(PHY_VARS_gNB *gNB,
                              NR_DL_FRAME_PARMS *frame_parms,
                              nfapi_nr_pusch_pdu_t *rel15_ul,
                              uint8_t ulsch_id,
                              uint8_t nr_tti_rx,
                              unsigned char symbol,
                              uint32_t nb_re_pusch)
{
  NR_gNB_PUSCH *pusch_vars = &gNB->pusch_vars[ulsch_id];
  int32_t *ptrs_re_symbol   = NULL;
  int8_t   ret = 0;
  uint8_t  symbInSlot       = rel15_ul->start_symbol_index + rel15_ul->nr_of_symbols;
  uint8_t *startSymbIndex   = &rel15_ul->start_symbol_index;
  uint8_t *nbSymb           = &rel15_ul->nr_of_symbols;
  uint8_t  *L_ptrs          = &rel15_ul->pusch_ptrs.ptrs_time_density;
  uint8_t  *K_ptrs          = &rel15_ul->pusch_ptrs.ptrs_freq_density;
  uint16_t *dmrsSymbPos     = &rel15_ul->ul_dmrs_symb_pos;
  uint16_t *ptrsSymbPos = &pusch_vars->ptrs_symbols;
  uint8_t *ptrsSymbIdx = &pusch_vars->ptrs_symbol_index;
  uint16_t *nb_rb           = &rel15_ul->rb_size;
  uint8_t  *ptrsReOffset    = &rel15_ul->pusch_ptrs.ptrs_ports_list[0].ptrs_re_offset;

  /* loop over antennas */
  for (int aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
    c16_t *phase_per_symbol = (c16_t *)pusch_vars->ptrs_phase_per_slot[aarx];
    ptrs_re_symbol = &pusch_vars->ptrs_re_per_slot;
    *ptrs_re_symbol = 0;
    phase_per_symbol[symbol].i = 0;
    /* set DMRS estimates to 0 angle with magnitude 1 */
    if(is_dmrs_symbol(symbol,*dmrsSymbPos)) {
      /* set DMRS real estimation to 32767 */
      phase_per_symbol[symbol].r=INT16_MAX; // 32767
#ifdef DEBUG_UL_PTRS
      printf("[PHY][PTRS]: DMRS Symbol %d -> %4d + j*%4d\n", symbol, phase_per_symbol[symbol].r,phase_per_symbol[symbol].i);
#endif
    }
    else {// real ptrs value is set to 0
      phase_per_symbol[symbol].r = 0;
    }

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
      const uint32_t *gold = nr_gold_pusch(frame_parms->N_RB_UL,
                                           frame_parms->symbols_per_slot,
                                           gNB->gNB_config.cell_config.phy_cell_id.value,
                                           rel15_ul->scid,
                                           nr_tti_rx,
                                           symbol);
      nr_ptrs_cpe_estimation(*K_ptrs,
                             *ptrsReOffset,
                             *nb_rb,
                             rel15_ul->rnti,
                             nr_tti_rx,
                             symbol,
                             frame_parms->ofdm_symbol_size,
                             (int16_t *)&pusch_vars->rxdataF_comp[aarx][(symbol * nb_re_pusch)],
                             gold,
                             (int16_t *)&phase_per_symbol[symbol],
                             ptrs_re_symbol);
    }

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

      /*------------------------------------------------------------------------------------------------------- */
      /* 3) Compensated DMRS based estimated signal with PTRS estimation                                        */
      /*--------------------------------------------------------------------------------------------------------*/
      for(uint8_t i = *startSymbIndex; i < symbInSlot; i++) {
        /* DMRS Symbol has 0 phase so no need to rotate the respective symbol */
        /* Skip rotation if the slot processing is wrong */
        if((!is_dmrs_symbol(i,*dmrsSymbPos)) && (ret == 0)) {
#ifdef DEBUG_UL_PTRS
          printf("[PHY][UL][PTRS]: Rotate Symbol %2d with  %d + j* %d\n", i, phase_per_symbol[i].r,phase_per_symbol[i].i);
#endif
          rotate_cpx_vector((c16_t *)&pusch_vars->rxdataF_comp[aarx][i * nb_re_pusch],
                            &phase_per_symbol[i],
                            (c16_t *)&pusch_vars->rxdataF_comp[aarx][i * nb_re_pusch],
                            ((*nb_rb) * NR_NB_SC_PER_RB),
                            15);
        } // if not DMRS Symbol
      } // symbol loop
    } // last symbol check
  } // Antenna loop
}

uint32_t calc_power(const int16_t *x, const uint32_t size) {
  int64_t sum_x = 0;
  int64_t sum_x2 = 0;
  for(int k = 0; k<size; k++) {
    sum_x = sum_x + x[k];
    sum_x2 = sum_x2 + x[k]*x[k];
  }

  return sum_x2/size - (sum_x/size)*(sum_x/size);
}

int nr_srs_channel_estimation(
    const PHY_VARS_gNB *gNB,
    const int frame,
    const int slot,
    const nfapi_nr_srs_pdu_t *srs_pdu,
    const nr_srs_info_t *nr_srs_info,
    const c16_t **srs_generated_signal,
    int32_t srs_received_signal[][gNB->frame_parms.ofdm_symbol_size * (1 << srs_pdu->num_symbols)],
    int32_t srs_estimated_channel_freq[][1 << srs_pdu->num_ant_ports]
                                      [gNB->frame_parms.ofdm_symbol_size * (1 << srs_pdu->num_symbols)],
    int32_t srs_estimated_channel_time[][1 << srs_pdu->num_ant_ports][gNB->frame_parms.ofdm_symbol_size],
    int32_t srs_estimated_channel_time_shifted[][1 << srs_pdu->num_ant_ports][gNB->frame_parms.ofdm_symbol_size],
    int8_t *snr_per_rb,
    int8_t *snr)
{
#ifdef SRS_DEBUG
  LOG_I(NR_PHY,"Calling %s function\n", __FUNCTION__);
#endif

  const NR_DL_FRAME_PARMS *frame_parms = &gNB->frame_parms;
  const uint64_t subcarrier_offset = frame_parms->first_carrier_offset + srs_pdu->bwp_start*NR_NB_SC_PER_RB;

  const uint8_t N_ap = 1<<srs_pdu->num_ant_ports;
  const uint8_t K_TC = 2<<srs_pdu->comb_size;
  const uint16_t m_SRS_b = srs_bandwidth_config[srs_pdu->config_index][srs_pdu->bandwidth_index][0];
  const uint16_t M_sc_b_SRS = m_SRS_b * NR_NB_SC_PER_RB/K_TC;
  uint8_t fd_cdm = N_ap;
  if (N_ap == 4 && ((K_TC == 2 && srs_pdu->cyclic_shift >= 4) || (K_TC == 4 && srs_pdu->cyclic_shift >= 6))) {
    fd_cdm = 2;
  }

  c16_t srs_ls_estimated_channel[frame_parms->ofdm_symbol_size*(1<<srs_pdu->num_symbols)];
  uint32_t noise_power_per_rb[srs_pdu->bwp_size];

  const uint32_t arr_len = frame_parms->nb_antennas_rx * N_ap * M_sc_b_SRS;

  int16_t ch_real[arr_len];
  memset(ch_real, 0, arr_len * sizeof(int16_t));
  
  int16_t ch_imag[arr_len];
  memset(ch_imag, 0, arr_len * sizeof(int16_t));
 
  int16_t noise_real[arr_len];
  memset(noise_real, 0, arr_len * sizeof(int16_t));
 
  int16_t noise_imag[arr_len];
  memset(noise_imag, 0, arr_len * sizeof(int16_t));

  int16_t ls_estimated[2];

  uint8_t mem_offset = ((16 - ((long)&srs_estimated_channel_freq[0][0][subcarrier_offset + nr_srs_info->k_0_p[0][0]])) & 0xF) >> 2; // >> 2 <=> /sizeof(int32_t)

  // filt16_end is {4096,8192,8192,8192,12288,16384,16384,16384,0,0,0,0,0,0,0,0}
  // The End of OFDM symbol corresponds to the position of last 16384 in the filter
  // The multadd_real_vector_complex_scalar applies the remaining 8 zeros of filter, therefore, to avoid a buffer overflow,
  // we added 8 in the array size
  int32_t srs_est[frame_parms->ofdm_symbol_size*(1<<srs_pdu->num_symbols) + mem_offset + 8] __attribute__ ((aligned(32)));

  for (int ant = 0; ant < frame_parms->nb_antennas_rx; ant++) {

    for (int p_index = 0; p_index < N_ap; p_index++) {

      memset(srs_ls_estimated_channel, 0, frame_parms->ofdm_symbol_size*(1<<srs_pdu->num_symbols)*sizeof(c16_t));
      memset(srs_est, 0, (frame_parms->ofdm_symbol_size*(1<<srs_pdu->num_symbols) + mem_offset)*sizeof(int32_t));

#ifdef SRS_DEBUG
      LOG_I(NR_PHY,"====================== UE port %d --> gNB Rx antenna %i ======================\n", p_index, ant);
#endif

      // Estimate the SRS channel over all OFDM symbols
      for (int srs_symb = 0; srs_symb<(1<<srs_pdu->num_symbols); srs_symb++) {
        uint16_t srs_symbol_offset =srs_symb*frame_parms->ofdm_symbol_size;
        uint16_t subcarrier = subcarrier_offset + nr_srs_info->k_0_p[p_index][srs_symb];
        if (subcarrier>frame_parms->ofdm_symbol_size) {
          subcarrier -= frame_parms->ofdm_symbol_size;
        }

        int16_t *srs_estimated_channel16 = (int16_t *)&srs_est[subcarrier + srs_symbol_offset + mem_offset];
        
        for (int k = 0; k < M_sc_b_SRS; k++) {

          if (k%fd_cdm==0) {

            ls_estimated[0] = 0;
            ls_estimated[1] = 0;
            uint16_t subcarrier_cdm = subcarrier;

            for (int cdm_idx = 0; cdm_idx < fd_cdm; cdm_idx++) {
              int16_t generated_real = srs_generated_signal[p_index][subcarrier_cdm + srs_symbol_offset].r;
              int16_t generated_imag = srs_generated_signal[p_index][subcarrier_cdm + srs_symbol_offset].i;

              int16_t received_real = ((c16_t*)srs_received_signal[ant])[subcarrier_cdm + srs_symbol_offset].r;
              int16_t received_imag = ((c16_t*)srs_received_signal[ant])[subcarrier_cdm + srs_symbol_offset].i;

              // We know that nr_srs_info->srs_generated_signal_bits bits are enough to represent the generated_real and generated_imag.
              // So we only need a nr_srs_info->srs_generated_signal_bits shift to ensure that the result fits into 16 bits.
              ls_estimated[0] += (int16_t)(((int32_t)generated_real*received_real + (int32_t)generated_imag*received_imag)>>nr_srs_info->srs_generated_signal_bits);
              ls_estimated[1] += (int16_t)(((int32_t)generated_real*received_imag - (int32_t)generated_imag*received_real)>>nr_srs_info->srs_generated_signal_bits);

              // Subcarrier increment
              subcarrier_cdm += K_TC;
              if (subcarrier_cdm >= frame_parms->ofdm_symbol_size) {
                subcarrier_cdm=subcarrier_cdm-frame_parms->ofdm_symbol_size;
              }
            }
          }

          srs_ls_estimated_channel[subcarrier + srs_symbol_offset].r = ls_estimated[0];
          srs_ls_estimated_channel[subcarrier + srs_symbol_offset].i = ls_estimated[1];

  #ifdef SRS_DEBUG
          int subcarrier_log = subcarrier-subcarrier_offset;
          if(subcarrier_log < 0) {
            subcarrier_log = subcarrier_log + frame_parms->ofdm_symbol_size;
          }
          if(subcarrier_log%12 == 0) {
            LOG_I(NR_PHY,"------------------------------------ %d ------------------------------------\n", subcarrier_log/12);
            LOG_I(NR_PHY,"\t  __genRe________genIm__|____rxRe_________rxIm__|____lsRe________lsIm_\n");
          }
          LOG_I(NR_PHY,"(%4i) %6i\t%6i  |  %6i\t%6i  |  %6i\t%6i\n",
                subcarrier_log,
                ((c16_t*)srs_generated_signal[p_index])[subcarrier].r, ((c16_t*)srs_generated_signal[p_index])[subcarrier].i,
                ((c16_t*)srs_received_signal[ant])[subcarrier].r, ((c16_t*)srs_received_signal[ant])[subcarrier].i,
                ls_estimated[0], ls_estimated[1]);
  #endif

          const uint16_t sc_offset = subcarrier + mem_offset;

          // Channel interpolation
          if(srs_pdu->comb_size == 0) {
            if(k == 0) { // First subcarrier case
              // filt8_start is {12288,8192,4096,0,0,0,0,0}
              multadd_real_vector_complex_scalar(filt8_start, ls_estimated, srs_estimated_channel16, 8);
            } else if(subcarrier < K_TC) { // Start of OFDM symbol case
              // filt8_start is {12288,8192,4096,0,0,0,0,0}
              srs_estimated_channel16 = (int16_t *)&srs_est[subcarrier + srs_symbol_offset];
              const short *filter = mem_offset == 0 ? filt8_start : filt8_start_shift2;
              multadd_real_vector_complex_scalar(filter, ls_estimated, srs_estimated_channel16, 8);
            } else if((subcarrier+K_TC)>=frame_parms->ofdm_symbol_size || k == (M_sc_b_SRS-1)) { // End of OFDM symbol or last subcarrier cases
              // filt8_end is {4096,8192,12288,16384,0,0,0,0}
              const short *filter = mem_offset == 0 || k == (M_sc_b_SRS - 1) ? filt8_end : filt8_end_shift2;
              multadd_real_vector_complex_scalar(filter, ls_estimated, srs_estimated_channel16, 8);
            } else if(k%2 == 1) { // 1st middle case
              // filt8_middle2 is {4096,8192,8192,8192,4096,0,0,0}
              multadd_real_vector_complex_scalar(filt8_middle2, ls_estimated, srs_estimated_channel16, 8);
            } else if(k%2 == 0) { // 2nd middle case
              // filt8_middle4 is {0,0,4096,8192,8192,8192,4096,0}
              multadd_real_vector_complex_scalar(filt8_middle4, ls_estimated, srs_estimated_channel16, 8);
              srs_estimated_channel16 = (int16_t *)&srs_est[sc_offset + srs_symbol_offset];
            }
          } else {
            if(k == 0) { // First subcarrier case
              // filt16_start is {12288,8192,8192,8192,4096,0,0,0,0,0,0,0,0,0,0,0}
              multadd_real_vector_complex_scalar(filt16_start, ls_estimated, srs_estimated_channel16, 16);
            } else if(subcarrier < K_TC) { // Start of OFDM symbol case
              srs_estimated_channel16 = (int16_t *)&srs_est[sc_offset + srs_symbol_offset];
              // filt16_start is {12288,8192,8192,8192,4096,0,0,0,0,0,0,0,0,0,0,0}
              multadd_real_vector_complex_scalar(filt16_start, ls_estimated, srs_estimated_channel16, 16);
            } else if((subcarrier+K_TC)>=frame_parms->ofdm_symbol_size || k == (M_sc_b_SRS-1)) { // End of OFDM symbol or last subcarrier cases
              // filt16_end is {4096,8192,8192,8192,12288,16384,16384,16384,0,0,0,0,0,0,0,0}
              multadd_real_vector_complex_scalar(filt16_end, ls_estimated, srs_estimated_channel16, 16);
            } else { // Middle case
              // filt16_middle4 is {4096,8192,8192,8192,8192,8192,8192,8192,4096,0,0,0,0,0,0,0}
              multadd_real_vector_complex_scalar(filt16_middle4, ls_estimated, srs_estimated_channel16, 16);
              srs_estimated_channel16 = (int16_t *)&srs_est[sc_offset + srs_symbol_offset];
            }
          }

          // Subcarrier increment
          subcarrier += K_TC;
          if (subcarrier >= frame_parms->ofdm_symbol_size) {
            subcarrier=subcarrier-frame_parms->ofdm_symbol_size;
          }

        } // for (int k = 0; k < M_sc_b_SRS; k++)
      } // for (int srs_symb = 0; srs_symb<(1<<srs_pdu->num_symbols); srs_symb++)

       memcpy(&srs_estimated_channel_freq[ant][p_index][0],
              &srs_est[mem_offset],
              ((1<<srs_pdu->num_symbols)*frame_parms->ofdm_symbol_size)*sizeof(int32_t));

      // Compute noise
      uint16_t subcarrier = subcarrier_offset + nr_srs_info->k_0_p[p_index][0];
      if (subcarrier>frame_parms->ofdm_symbol_size) {
        subcarrier -= frame_parms->ofdm_symbol_size;
      }
      uint16_t base_idx = ant*N_ap*M_sc_b_SRS + p_index*M_sc_b_SRS;
      for (int k = 0; k < M_sc_b_SRS; k++) {
        ch_real[base_idx+k] = ((c16_t*)srs_estimated_channel_freq[ant][p_index])[subcarrier].r;
        ch_imag[base_idx+k] = ((c16_t*)srs_estimated_channel_freq[ant][p_index])[subcarrier].i;
        noise_real[base_idx+k] = abs(srs_ls_estimated_channel[subcarrier].r - ch_real[base_idx+k]);
        noise_imag[base_idx+k] = abs(srs_ls_estimated_channel[subcarrier].i - ch_imag[base_idx+k]);
        subcarrier += K_TC;
        if (subcarrier >= frame_parms->ofdm_symbol_size) {
          subcarrier=subcarrier-frame_parms->ofdm_symbol_size;
        }
      }

      // Compute signal power
      uint32_t signal_power_ant = calc_power(&ch_real[base_idx], M_sc_b_SRS) + calc_power(&ch_imag[base_idx], M_sc_b_SRS);
      
//#ifdef SRS_DEBUG
      LOG_D(NR_PHY,"signal_power(p_index %d, ant %d) = %d dB\n", p_index, ant, dB_fixed(signal_power_ant));
//#endif
      


#ifdef SRS_DEBUG
      subcarrier = subcarrier_offset + nr_srs_info->k_0_p[p_index][0];
      if (subcarrier>frame_parms->ofdm_symbol_size) {
        subcarrier -= frame_parms->ofdm_symbol_size;
      }

      for (int k = 0; k < K_TC*M_sc_b_SRS; k++) {

        int subcarrier_log = subcarrier-subcarrier_offset;
        if(subcarrier_log < 0) {
          subcarrier_log = subcarrier_log + frame_parms->ofdm_symbol_size;
        }

        if(subcarrier_log%12 == 0) {
          LOG_I(NR_PHY,"------------------------------------- %d -------------------------------------\n", subcarrier_log/12);
          LOG_I(NR_PHY,"\t  __lsRe__________lsIm__|____intRe_______intIm__|____noiRe_______noiIm__\n");
        }

        LOG_I(NR_PHY,"(%4i) %6i\t%6i  |  %6i\t%6i  |  %6i\t%6i\n",
              subcarrier_log,
              srs_ls_estimated_channel[subcarrier].r,
              srs_ls_estimated_channel[subcarrier].i,
              ((c16_t*)srs_estimated_channel_freq[ant][p_index])[subcarrier].r,
              ((c16_t*)srs_estimated_channel_freq[ant][p_index])[subcarrier].i,
              noise_real[base_idx+(k/K_TC)], noise_imag[base_idx+(k/K_TC)]);

        // Subcarrier increment
        subcarrier++;
        if (subcarrier >= frame_parms->ofdm_symbol_size) {
          subcarrier=subcarrier-frame_parms->ofdm_symbol_size;
        }
      }
#endif

      // Convert to time domain
      freq2time(gNB->frame_parms.ofdm_symbol_size,
                (int16_t*) srs_estimated_channel_freq[ant][p_index],
                (int16_t*) srs_estimated_channel_time[ant][p_index]);

      memcpy(&srs_estimated_channel_time_shifted[ant][p_index][0],
             &srs_estimated_channel_time[ant][p_index][gNB->frame_parms.ofdm_symbol_size>>1],
             (gNB->frame_parms.ofdm_symbol_size>>1)*sizeof(int32_t));

      memcpy(&srs_estimated_channel_time_shifted[ant][p_index][gNB->frame_parms.ofdm_symbol_size>>1],
             &srs_estimated_channel_time[ant][p_index][0],
             (gNB->frame_parms.ofdm_symbol_size>>1)*sizeof(int32_t));
    } // for (int p_index = 0; p_index < N_ap; p_index++)
  } // for (int ant = 0; ant < frame_parms->nb_antennas_rx; ant++)


      // Compute signal power
      uint32_t signal_power = calc_power(ch_real, arr_len) + calc_power(ch_imag, arr_len);

#ifdef SRS_DEBUG
      LOG_I(NR_PHY,"signal_power(p_index %d, ant %d) = %d dB\n", p_index, ant, dB_fixed(signal_power));
#endif

      if (signal_power == 0) {
	LOG_W(NR_PHY, "Received SRS signal power is 0\n");
	return -1;
      }


  // Compute noise power
  const uint8_t signal_power_bits = log2_approx(signal_power);
  const uint8_t factor_bits = signal_power_bits < 32 ? 32 - signal_power_bits : 0; // 32 due to input of dB_fixed(uint32_t x)
  const int32_t factor_dB = dB_fixed(1<<factor_bits);

  const uint8_t srs_symbols_per_rb = srs_pdu->comb_size == 0 ? 6 : 3;
  const uint8_t n_noise_est = frame_parms->nb_antennas_rx*N_ap*srs_symbols_per_rb;
  uint64_t sum_re = 0;
  uint64_t sum_re2 = 0;
  uint64_t sum_im = 0;
  uint64_t sum_im2 = 0;

  for (int rb = 0; rb < m_SRS_b; rb++) {

    sum_re = 0;
    sum_re2 = 0;
    sum_im = 0;
    sum_im2 = 0;

    for (int ant = 0; ant < frame_parms->nb_antennas_rx; ant++) {
      for (int p_index = 0; p_index < N_ap; p_index++) {
        uint16_t base_idx = ant*N_ap*M_sc_b_SRS + p_index*M_sc_b_SRS + rb*srs_symbols_per_rb;
        for (int srs_symb = 0; srs_symb < srs_symbols_per_rb; srs_symb++) {
          sum_re = sum_re + noise_real[base_idx+srs_symb];
          sum_re2 = sum_re2 + noise_real[base_idx+srs_symb]*noise_real[base_idx+srs_symb];
          sum_im = sum_im + noise_imag[base_idx+srs_symb];
          sum_im2 = sum_im2 + noise_imag[base_idx+srs_symb]*noise_imag[base_idx+srs_symb];
        } // for (int srs_symb = 0; srs_symb < srs_symbols_per_rb; srs_symb++)
      } // for (int p_index = 0; p_index < N_ap; p_index++)
    } // for (int ant = 0; ant < frame_parms->nb_antennas_rx; ant++)

    noise_power_per_rb[rb] = max(sum_re2 / n_noise_est - (sum_re / n_noise_est) * (sum_re / n_noise_est) +
                                 sum_im2 / n_noise_est - (sum_im / n_noise_est) * (sum_im / n_noise_est), 1);
    snr_per_rb[rb] = dB_fixed((int32_t)((signal_power<<factor_bits)/noise_power_per_rb[rb])) - factor_dB;

#ifdef SRS_DEBUG
    LOG_I(NR_PHY,"noise_power_per_rb[%i] = %i, snr_per_rb[%i] = %i dB\n", rb, noise_power_per_rb[rb], rb, snr_per_rb[rb]);
#endif

  } // for (int rb = 0; rb < m_SRS_b; rb++)

  const uint32_t noise_power = max(calc_power(noise_real, arr_len) + calc_power(noise_imag, arr_len), 1);

  *snr = dB_fixed((int32_t)((signal_power<<factor_bits)/(noise_power))) - factor_dB;

#ifdef SRS_DEBUG
  LOG_I(NR_PHY,"noise_power = %u, SNR = %i dB\n", noise_power, *snr);
#endif
  return 0;
}
