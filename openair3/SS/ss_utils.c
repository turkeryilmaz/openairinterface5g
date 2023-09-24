#include "ss_utils.h"

void bits_copy_from_array(unsigned char *dst, int off, const unsigned char* src, int len)
{
    while (len-- > 0)
    {
        int bit = *src++ ? 1 : 0;
        dst[off / 8] |= bit << (7 - off % 8);
        off++;
    }
}

void bits_copy_to_array(unsigned char *dst, int off, const unsigned char* src, int len)
{
    while (len-- > 0)
    {
        int bit = src[off / 8] & (1 << (7 - off % 8));
        *dst++ = bit ? 0x01 : 0x00;
        off++;
    }
}
