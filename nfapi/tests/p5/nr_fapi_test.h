//
// Created by user on 14-03-2024.
//

#ifndef OPENAIRINTERFACE_NR_FAPI_TEST_H
#define OPENAIRINTERFACE_NR_FAPI_TEST_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef _STANDALONE_TESTING_
#include "common/utils/LOG/log.h"
#endif
#include "nr_fapi_p5.h"
#include "nr_fapi.h"
#include "nfapi_nr_interface.h"
#include "nfapi_nr_interface_scf.h"

uint8_t rand8()
{
  return (rand() & 0xff);
}
uint16_t rand16()
{
  return rand8() << 8 | rand8();
}
uint32_t rand24()
{
  return rand16() << 8 | rand8();
}
uint32_t rand32()
{
  return rand24() << 8 | rand8();
}
uint64_t rand64()
{
  return (uint64_t )rand32() << 32 | rand32();
}

int main(int n, char *v[]);
#endif // OPENAIRINTERFACE_NR_FAPI_TEST_H
