/*! \file bap_cpdu.c
 * \brief BAP Control PDU per RLC channel
 * \author Surajit Dey, Danny Nsouli, Michael Bundas
 * \date NOV 2023
 * \version 1.0
 * \company MITRE
 * \email: sdey@mitre.org, dnsouli@mitre.org, mbundas@mitre.org
 * 
 * NOTICE
 * 
 * This software was produced for the U. S. Government
 * under Basic Contract No. W56KGU-18-D-0004, and is
 * subject to the Rights in Noncommercial Computer Software
 * and Noncommercial Computer Software Documentation
 * Clause 252.227-7014 (FEB 2014)
 * 
 * (C) 2024 The MITRE Corporation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// BAP header
struct bap_control {
    unsigned char oct1;
    unsigned char oct2;
    unsigned char oct3;
    unsigned char oct4;
    unsigned char oct5;
    unsigned char oct6;
};

uint64_t bap_control_func(unsigned short bh_rlc_ch_id, unsigned int avail_buffer_size) {

    struct bap_control bapc;

    // Set header fields
    bapc.oct1 = 0;

    printf("Oct 1 0x%x\n", bapc.oct1);

    bapc.oct2 = (bh_rlc_ch_id >> 8) & 0xFF;
    bapc.oct3 = (bh_rlc_ch_id) & 0xFF;

    printf("Oct 2 0x%X\n", bapc.oct2);
    printf("Oct 3 0x%X\n", bapc.oct3);

    bapc.oct4 = (avail_buffer_size >> 16) & 0xFF;
    bapc.oct5 = (avail_buffer_size >> 8) & 0xFF;
    bapc.oct6 = (avail_buffer_size) & 0xFF;

    printf("Oct 4 0x%X\n", bapc.oct4);
    printf("Oct 5 0x%X\n", bapc.oct5);
    printf("Oct 6 0x%X\n", bapc.oct6);

    unsigned char controlPDU[6];
    memcpy(&controlPDU[0], (unsigned char *) &bapc, sizeof(bapc));

    // Print assembled PDU
    for(int i=0; i<6; i++) {
        printf("%02X ", controlPDU[i]);
    }
    printf("\n");

    uint64_t octets = 0;

    octets |= ((uint64_t)bapc.oct6 << 40);
    octets |= ((uint64_t)bapc.oct5 << 32);
    octets |= ((uint64_t)bapc.oct4 << 24);
    octets |= ((uint64_t)bapc.oct3 << 16);
    octets |= ((uint64_t)bapc.oct2 << 8);
    octets |= bapc.oct1;

    return octets;

}

uint64_t main(int argc, char *argv[]){
    int i; 

    if (argc != 3) {
        printf("Please pass in exactly two arguments\n");
        return 1;
    }

    printf("The following arguments were passed to main(): ");
    for(i=1; i<argc; i++) printf("%s ", argv[i]);
    printf("\n");

    short bh_rlc_ch_id = (short)strtol(argv[1], NULL, 16);
    int avail_buffer_size = (int)strtol(argv[2], NULL, 16);

    printf("bh_rlc_ch_id: %d\n", bh_rlc_ch_id);
    printf("avail_buffer_size: %d\n", avail_buffer_size);

    if (bh_rlc_ch_id < 0 || bh_rlc_ch_id > 0xFFFF || avail_buffer_size < 0 || avail_buffer_size > 0xFFFFFF) {
        printf("ERROR: One or both inputted values is out of range.\n");
        return -1;
    }

    uint64_t octets = bap_control_func(bh_rlc_ch_id, avail_buffer_size);

    printf("Octets: 0x%llX\n", octets);

    return octets;
}