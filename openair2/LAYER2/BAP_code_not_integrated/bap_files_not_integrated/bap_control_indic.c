/*! \file bap_control_indic.c
 * \brief BAP Control PDU format for BH RLF indication
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
struct bap_control_indic {
    unsigned char oct1;
};

// Flow control fields
unsigned char PDU_Type = 0b0011;

uint64_t bap_control_indic_func() {

    struct bap_control_indic bap_ci;

    // Set header fields
    bap_ci.oct1 = 0;
    bap_ci.oct1 |= (PDU_Type << 3);
    // DC=0, PDU Type=0011

    printf("Oct 1 0x%x\n", bap_ci.oct1);

    unsigned char controlPDU[1];
    memcpy(&controlPDU[0], (unsigned char *) &bap_ci, sizeof(bap_ci));

    // Print assembled PDU
    for(int i=0; i<6; i++) {
        printf("%02X ", controlPDU[i]);
    }
    printf("\n");

    uint64_t octets = 0;

    octets |= bap_ci.oct1;

    return octets;

}

uint64_t main(int argc, char *argv[]){

    if (argc != 1) {
        printf("Please pass in exactly two arguments\n");
        return 1;
    }

    uint64_t octets = bap_control_indic_func();

    printf("Octets: 0x%llX\n", octets);

    return octets;
}