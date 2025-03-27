#include "srap_header.h"
#include "softmodem-common.h"

// Function to encode SRAP Header
void encode_srap_header(void* header, uint8_t* buffer) {
    if (get_softmodem_params()->relay_type == U2N) {
        U2NHeader_t* u2n_header = (U2NHeader_t*)header;
        buffer[0] = u2n_header->octet1;
        buffer[1] = u2n_header->octet2;
    } else if (get_softmodem_params()->relay_type == U2U) {
        U2UHeader_t* u2u_header = (U2UHeader_t*)header;
        buffer[0] = u2u_header->octet1;
        buffer[1] = u2u_header->octet2;
        buffer[2] = u2u_header->octet3;
    } else {
        //LOG_E(); // Incorrect relay type!!!
    }
}

// Function to decode SRAP Header
void decode_srap_header(void* header, uint8_t* buffer) {
    if (get_softmodem_params()->relay_type == U2N) {
        U2NHeader_t* u2n_header = (U2NHeader_t*)header;
        u2n_header->octet1 = buffer[0];
        u2n_header->octet2 = buffer[1];
    } else if (get_softmodem_params()->relay_type == U2U) {
        U2UHeader_t* u2u_header = (U2UHeader_t*)header;
        u2u_header->octet1 = buffer[0];
        u2u_header->octet2 = buffer[1];
        u2u_header->octet3 = buffer[2];
    } else {
        //LOG_E(); // Incorrect relay type!!!
    }
}

// Function to construct SRAP data PDU header
// Inputs: SRAP data PDU BEARER ID (5 bits) and UE ID (8 bits)
// Output: SRAP data PDU header
void create_header(relay_type_t r_type, uint8_t bearer_id, uint8_t src_ue_id, uint8_t dest_ue_id, void* header) {
    if (r_type == U2N) {
        U2NHeader_t* u2n_header = (U2NHeader_t*) header;
        /* 1st octet */
        u2n_header->octet1 = 0;
        u2n_header->octet1 |= 0x80; /* D/C bit 7=1 for data, 2 R bits 5-6 = 0 */
        u2n_header->octet1 |= (bearer_id & 0x1F);

        /* 2nd octet */
        u2n_header->octet2 = src_ue_id;
    } else if (r_type == U2U) {
        U2UHeader_t* u2u_header = (U2UHeader_t*) header;
        /* 1st octet */
        u2u_header->octet1 = 0;
        u2u_header->octet1 |= 0x80; /* D/C bit 7=1 for data, 2 R bits 5-6 = 0 */
        u2u_header->octet1 |= (bearer_id & 0x1F);

        /* 2nd octet */
        u2u_header->octet2 = src_ue_id;

        /* 3rd octet */
        u2u_header->octet3 = dest_ue_id;
    }
}
