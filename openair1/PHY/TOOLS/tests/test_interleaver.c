#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <cuda_runtime.h>  // Move this BEFORE oai_cuda.h

#include "PHY/TOOLS/tools_defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "SIMULATION/TOOLS/oai_cuda.h"  // Move this AFTER cuda_runtime.h
// #include "common/utils/LOG/log.h"
#include "common/utils/utils.h"

// Forward declaration for our new GPU function
void interleave_channel_output_cuda(float **rx_sig_re,
                                    float **rx_sig_im,
                                    float2 **output_interleaved,
                                    int nb_rx,
                                    int num_samples);

// The original CPU version for comparison
void interleave_channel_output(float **rx_sig_re,
                               float **rx_sig_im,
                               float **output_interleaved,
                               int nb_rx,
                               int num_samples)
{
    for (int i = 0; i < nb_rx; i++) {
        for (int j = 0; j < num_samples; j++) {
            output_interleaved[i][2 * j]     = rx_sig_re[i][j];
            output_interleaved[i][2 * j + 1] = rx_sig_im[i][j];
        }
    }
}

/**
 * @brief Display sample input data for the first few antennas and samples
 */
void display_input_samples(float **rx_sig_re, float **rx_sig_im, int nb_rx, int num_samples) {
    int max_antennas = (nb_rx > 4) ? 4 : nb_rx;  // Show max 4 antennas
    int max_samples = (num_samples > 8) ? 8 : num_samples;  // Show max 8 samples
    
    printf("--- Input Samples (first %d antennas, first %d samples) ---\n", max_antennas, max_samples);
    for (int i = 0; i < max_antennas; i++) {
        printf("Antenna %d:\n", i);
        printf("  Real:     ");
        for (int j = 0; j < max_samples; j++) {
            printf("%8.4f ", rx_sig_re[i][j]);
        }
        printf("\n");
        printf("  Imag:     ");
        for (int j = 0; j < max_samples; j++) {
            printf("%8.4f ", rx_sig_im[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Display sample output data for CPU and GPU results
 */
void display_output_samples(float2 **output_cpu, float2 **output_gpu, int nb_rx, int num_samples) {
    int max_antennas = (nb_rx > 4) ? 4 : nb_rx;  // Show max 4 antennas
    int max_samples = (num_samples > 8) ? 8 : num_samples;  // Show max 8 samples
    
    printf("--- Output Samples (first %d antennas, first %d samples) ---\n", max_antennas, max_samples);
    for (int i = 0; i < max_antennas; i++) {
        printf("Antenna %d CPU Output:\n", i);
        printf("  Interleaved: ");
        for (int j = 0; j < max_samples; j++) {
            printf("(%10.6f,%10.6f) ", output_cpu[i][j].x, output_cpu[i][j].y);  // Show 6 decimals
        }
        printf("\n");
        
        printf("Antenna %d GPU Output:\n", i);
        printf("  Interleaved: ");
        for (int j = 0; j < max_samples; j++) {
            printf("(%10.6f,%10.6f) ", output_gpu[i][j].x, output_gpu[i][j].y);  // Show 6 decimals
        }
        printf("\n");
        
        printf("Antenna %d Difference:\n", i);
        printf("  Error:       ");
        for (int j = 0; j < max_samples; j++) {
            double err_re = output_cpu[i][j].x - output_gpu[i][j].x;
            double err_im = output_cpu[i][j].y - output_gpu[i][j].y;
            printf("(%6.3e,%6.3e) ", err_re, err_im);
        }
        printf("\n\n");
    }
}

/**
 * @brief Verifies that the CPU and GPU interleaved outputs are identical.
 * @return 0 on success (PASSED), 1 on failure (FAILED).
 */
int verify_interleaved_results(float2 **output_cpu, float2 **output_gpu, int nb_rx, int num_samples) {
    double total_error = 0.0;
    double max_error = 0.0;
    for (int i = 0; i < nb_rx; i++) {
        for (int j = 0; j < num_samples; j++) {
            double err_re = output_cpu[i][j].x - output_gpu[i][j].x;
            double err_im = output_cpu[i][j].y - output_gpu[i][j].y;
            double sample_error = (err_re * err_re) + (err_im * err_im);
            total_error += sample_error;
            if (sample_error > max_error) {
                max_error = sample_error;
            }
        }
    }
    double mse = total_error / (nb_rx * num_samples);
    printf("--- Error Statistics ---\n");
    printf("  Mean Squared Error (MSE): %e\n", mse);
    printf("  Maximum Error:            %e\n", sqrt(max_error));
    printf("  Total Samples Compared:   %d\n", nb_rx * num_samples);
    
    if (mse > 1e-12) {
        printf("  [ERROR] MSE is too high: %e\n", mse);
        return 1; // FAILED
    }
    return 0; // PASSED
}

int main(int argc, char **argv) {
    // logInit();/
    randominit(0);

    int nb_rx = 8;
    int num_samples = 30720; // A typical slot length

    printf("--- Testing Output Interleaver (CPU vs. GPU) ---\n");
    printf("Configuration: %d RX Antennas, %d Samples\n\n", nb_rx, num_samples);

    // --- 1. Allocate Host Memory ---
    float **rx_sig_re = malloc(nb_rx * sizeof(float *));
    float **rx_sig_im = malloc(nb_rx * sizeof(float *));
    
    // Note: The CPU version outputs to a float**, which we will cast to float2** for verification
    float **output_cpu_temp = malloc(nb_rx * sizeof(float *));
    float2 **output_gpu = malloc(nb_rx * sizeof(float2 *));

    for (int i = 0; i < nb_rx; i++) {
        rx_sig_re[i] = malloc(num_samples * sizeof(float));
        rx_sig_im[i] = malloc(num_samples * sizeof(float));
        output_cpu_temp[i] = malloc(num_samples * 2 * sizeof(float)); // float** format
        output_gpu[i] = malloc(num_samples * sizeof(float2));      // float2** format
    }

    // --- 2. Generate Random Input Data ---
    printf("Generating random de-interleaved input data...\n");
    for (int i = 0; i < nb_rx; i++) {
        for (int j = 0; j < num_samples; j++) {
            rx_sig_re[i][j] = (float)rand() / (float)RAND_MAX;
            rx_sig_im[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }

    // --- 2.5. Display Sample Input Data ---
    display_input_samples(rx_sig_re, rx_sig_im, nb_rx, num_samples);

    // --- 3. Run CPU Version (Golden Reference) ---
    printf("Running CPU interleaver...\n");
    interleave_channel_output(rx_sig_re, rx_sig_im, output_cpu_temp, nb_rx, num_samples);
    // Cast the float** output to float2** for direct comparison. This is safe because the memory layout is identical.
    float2 **output_cpu = (float2**)output_cpu_temp;

    // --- 4. Run GPU Version ---
    printf("Running GPU interleaver...\n");
    interleave_channel_output_cuda(rx_sig_re, rx_sig_im, output_gpu, nb_rx, num_samples);

    // --- 4.5. Display Sample Output Data ---
    display_output_samples(output_cpu, output_gpu, nb_rx, num_samples);

    // --- 5. Verify Results ---
    printf("Verifying results...\n");
    int result = verify_interleaved_results(output_cpu, output_gpu, nb_rx, num_samples);

    if (result == 0) {
        printf("\n  +--------+\n");
        printf("  | PASSED |\n");
        printf("  +--------+\n");
        printf("  GPU output matches the CPU reference.\n\n");
    } else {
        printf("\n  +--------+\n");
        printf("  | FAILED |\n");
        printf("  +--------+\n");
        printf("  GPU output does NOT match the CPU reference.\n\n");
    }
    
    // --- 6. Cleanup ---
    for (int i = 0; i < nb_rx; i++) {
        free(rx_sig_re[i]);
        free(rx_sig_im[i]);
        free(output_cpu_temp[i]);
        free(output_gpu[i]);
    }
    free(rx_sig_re);
    free(rx_sig_im);
    free(output_cpu_temp);
    free(output_gpu);

    return result;
}
