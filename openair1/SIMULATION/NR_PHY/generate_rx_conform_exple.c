#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>



void construct_file(const char *filename, 
                    int16_t n_rnti, 
                    int16_t nb_rb, 
                    int16_t start_rb,
                    int16_t nb_symb_sch, 
                    int16_t start_symbol, 
                    int16_t Imcs, 
                    int16_t rv_index, 
                    int16_t dummy,
                    int16_t rxdata[][2], 
                    int rxdata_length, 
                    int slot_length) 
{
    FILE *fd = fopen(filename, "wb");
    if(fd == NULL) {
        perror("Failed to open file for writing");
        return;
    }

    // Offset at the beginning
    int file_offset = 0; // set this value based on your requirements.
    int start_offset = file_offset * ((slot_length << 2) + 4000 + 16);
    fseek(fd, start_offset, SEEK_SET);

    fwrite(&n_rnti, sizeof(int16_t), 1, fd);
    fwrite(&nb_rb, sizeof(int16_t), 1, fd);
    fwrite(&start_rb, sizeof(int16_t), 1, fd);
    fwrite(&nb_symb_sch, sizeof(int16_t), 1, fd);
    fwrite(&start_symbol, sizeof(int16_t), 1, fd);
    fwrite(&Imcs, sizeof(int16_t), 1, fd);
    fwrite(&rv_index, sizeof(int16_t), 1, fd);
    fwrite(&dummy, sizeof(int16_t), 1, fd);

    // Offset after parameters
    fseek(fd, file_offset * sizeof(int16_t) * 2, SEEK_CUR);

    // Now write the rxdata
    for(int i = 0; i < rxdata_length; i++) {
        fwrite(&rxdata[i][0], sizeof(int16_t), 1, fd); // Real part
        fwrite(&rxdata[i][1], sizeof(int16_t), 1, fd); // Imaginary part
    }

    fclose(fd);
}


void generate_signal(int16_t rxdata[][2], int rxdata_length) {
    // Seed the random number generator with current time
    srand((unsigned int) time(NULL));

    for(int i = 0; i < rxdata_length; i++) {
        // Populate the real and imaginary parts with random values
        rxdata[i][0] = (int16_t)(rand() % (INT16_MAX + 1 - INT16_MIN) + INT16_MIN);
        rxdata[i][1] = (int16_t)(rand() % (INT16_MAX + 1 - INT16_MIN) + INT16_MIN);
    }
}

int main() {
    // Example parameters
    int16_t n_rnti = 0x1234;
    int16_t nb_rb = 273;
    int16_t start_rb = 5;
    int16_t nb_symb_sch = 14;
    int16_t start_symbol = 0;
    int16_t Imcs = 1; 
    int16_t rv_index = 1;
    int16_t dummy = 0;

    int rxdata_length = 1000; // Number of complex samples ( can be changed based on requirement)
    int16_t rxdata[rxdata_length][2];

    // Generate the signal
    generate_signal(rxdata, rxdata_length);

    // Call the function to construct the file
    int slot_length = 1000; // adjust this based on your data
    construct_file("rx_conform.bin", n_rnti, nb_rb, start_rb, nb_symb_sch, start_symbol, Imcs, rv_index, dummy, rxdata, rxdata_length, slot_length);

    return 0;
}