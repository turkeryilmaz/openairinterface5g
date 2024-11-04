/*! \file bap_pdu.c
 * \brief BAP PDU
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

#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>


/****************************************

ipv4 packet structure

****************************************/

struct ipv4_packet {

    struct iphdr iph;

    struct tcphdr tcph;

    void *data;
};


/****************************************

BAP Data PDU

****************************************/

struct bap_data
{
    unsigned char oct1;
    unsigned char oct2;
    unsigned char oct3;
//    void *data;
};

int bap_pdu_func(unsigned short dest, unsigned short path) {

    struct bap_data bapd;
    unsigned short dest_high, dest_low;
    unsigned char dest_high_char, dest_low_char;
    unsigned short path_high, path_low;
    unsigned char path_high_char, path_low_char;
    unsigned char packet[2048];
    char buffer[1024] = "ABCDEFGH";
    int i, len=8;
    unsigned char* cp;
    unsigned char bapx[3];

    /* Prepare BAP header */

    /* 1st octet */
    bapd.oct1 = 0;
    bapd.oct1 |= 0x80;  /* D/C bit 7=1 for data, 3 R bits 4-6 = 0 */
    dest_high = dest & 0x03C0;  /* bits 6-9 of destination=dest high */
    dest_high = dest_high >> 6;
    dest_high_char = (unsigned char) dest_high;
    bapd.oct1 |= (dest_high_char & 0x0F);   /* bits 0-3 of oct 1 = dest high */
    printf("Oct 1 0x%x\n", bapd.oct1);

    /* 2nd octet */
    bapd.oct2 = 0;
    dest_low = dest & 0x003F;   /* bits 0-5 of destination=dest low */
    dest_low = dest_low << 2;
    dest_low_char = (unsigned char) dest_low;
    bapd.oct2 |= (dest_low_char & 0xFC);  /* bits 2-7 of oct 2 = dest low */

    path_high = path & 0x0300;  /* bits 8,9 of path = path high */
    path_high = path_high >> 8;
    path_high_char = (unsigned char) path_high;
    bapd.oct2 |= (path_high_char & 0x03);  /* bits 0-1 of oct 2 = path high */
    printf("Oct 2 0x%x\n", bapd.oct2);


    /* 3rd octet */
    path_low = path & 0x00FF;   /* bits 0-7 of oct 3 = path low */
    path_low_char = (unsigned char) path_low;
    bapd.oct3 = path_low_char;
    printf("Oct 3 0x%x\n", bapd.oct3);

    /*Assemble packet*/

    memcpy(&packet[0], (unsigned char *) &bapd, sizeof(bapd));

    printf("Packet 1 BAP hdr len=%d\n", sizeof(bapd));
    cp = &packet[0];
    for ( ; *cp != '\0'; ++cp )
    {
       printf("%x", *cp);
    }

    memcpy(packet + sizeof(bapd), buffer, len);

    printf("\nPacket 2 BAP hdr + data len=%d\n", (sizeof(bapd)+len));
    cp = &packet[0];
    for ( ; *cp != '\0'; ++cp )
    {
       printf("%x", *cp);
    }
    printf("\n");

    int octets = 0;
  
    octets = (bapd.oct3 << 16) | (bapd.oct2 << 8) | bapd.oct1;

    return octets;

}


int main (int argc, char *argv[] ) {

    int i; 

    if (argc != 3) {
        printf("Please pass in exactly two arguments\n");
        return 1;
    }

    printf("The following arguments were passed to main(): ");
    for(i=1; i<argc; i++) printf("%s ", argv[i]);
    printf("\n");

    short dest = (short)strtol(argv[1], NULL, 16);;
    short path = (short)strtol(argv[2], NULL, 16);

    printf("dest: %d\n", dest);
    printf("path: %d\n", path);

    if (dest < 0 || dest > 0x03FF || path < 0 || path > 0x03FF) {
        printf("ERROR: One or both inputted values is out of range.\n");
        return -1;
    }

    int octets = bap_pdu_func(dest, path);

    printf("Octets: 0x%x\n", octets);;

    return octets;

}


