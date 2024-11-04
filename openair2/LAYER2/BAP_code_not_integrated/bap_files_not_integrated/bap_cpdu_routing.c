/*! \file bap_cpdu_routing.c
 * \brief BAP Control PDU per BAP Routing ID
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
#include <stdint.h>

// BAP header
struct bap_control {
    unsigned char oct1;
    unsigned char oct2;
    unsigned char oct3;
    unsigned char oct4;
    unsigned char oct5;
    unsigned char oct6;
    unsigned char oct7;
};

unsigned char PDU_Type = 0b0001;

uint64_t bap_control_routing_func(unsigned int bap_routing_id, unsigned int avail_buffer_size) {

    struct bap_control bapc;

    // Set header fields
    bapc.oct1 = 0;
    bapc.oct1 |= (PDU_Type << 3);
    // bapc.oct1 = 0x01; // DC=0, PDU Type=0001

    printf("Oct 1 0x%X\n", bapc.oct1);

    bapc.oct2 = (bap_routing_id >> 12) & 0xFF;
    bapc.oct3 = (bap_routing_id >> 4) & 0xFF;
    //Takes into account the 4 reserved bits
    bapc.oct4 = (bap_routing_id) & 0xF;
    bapc.oct4 = (bapc.oct4 << 4) & 0xF0;

    printf("Oct 2 0x%X\n", bapc.oct2);
    printf("Oct 3 0x%X\n", bapc.oct3);
    printf("Oct 4 0x%X\n", bapc.oct4);

    bapc.oct5 = (avail_buffer_size >> 16) & 0xFF;
    bapc.oct6 = (avail_buffer_size >> 8) & 0xFF;
    bapc.oct7 = (avail_buffer_size) & 0xFF;

    printf("Oct 5 0x%X\n", bapc.oct5);
    printf("Oct 6 0x%X\n", bapc.oct6);
    printf("Oct 7 0x%X\n", bapc.oct7);

    unsigned char controlPDU[7];
    memcpy(&controlPDU[0], (unsigned char *) &bapc, sizeof(bapc));

    // Print assembled PDU
    for(int i=0; i<7; i++) {
        printf("%02X ", controlPDU[i]);
    }
    printf("\n");

    uint64_t octets = 0;

    octets |= ((uint64_t)bapc.oct7 << 56);
    octets |= ((uint64_t)bapc.oct6 << 48); 
    octets |= ((uint64_t)bapc.oct5 << 40);
    octets |= ((uint64_t)bapc.oct4 << 32);
    octets |= ((uint64_t)bapc.oct3 << 24);
    octets |= ((uint64_t)bapc.oct2 << 16);
    octets |= ((uint64_t)bapc.oct1 << 8);

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

    int bap_routing_id = (int)strtol(argv[1], NULL, 16);;
    int avail_buffer_size = (int)strtol(argv[2], NULL, 16);

    printf("bap_routing_id: %d\n", bap_routing_id);
    printf("avail_buffer_size: %d\n", avail_buffer_size);

    if (bap_routing_id < 0 || bap_routing_id > 0xFFFFF || avail_buffer_size < 0 || avail_buffer_size > 0xFFFFFF) {
        printf("ERROR: One or both inputted values is out of range.\n");
        return -1;
    }

    uint64_t octets = bap_control_routing_func(bap_routing_id, avail_buffer_size);

    printf("Octets: 0x%llX\n", octets);

    return octets;
}