/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

#ifndef _NR_PDCP_CONFIGURATION_H_
#define _NR_PDCP_CONFIGURATION_H_

#ifndef INT_LIST_SIZE
#define INT_LIST_SIZE(x) (sizeof((int[]){x})/sizeof(int))
#endif

/* values taken from 3GPP TS 38.331 version 17.3.0 */

#define VALUES_NR_PDCP_SN_SIZE \
    12, 18

#define VALUES_NR_PDCP_SN_SIZE_STR \
    "len12bits", "len18bits"

#define SIZEOF_NR_PDCP_SN_SIZE INT_LIST_SIZE(VALUES_NR_PDCP_SN_SIZE)

#define VALUES_NR_PDCP_T_REORDERING \
    0,   1,   2,   4,    5,    8,    10,   15,   20,   30,   40,   50, \
    60,  80,  100, 120,  140,  160,  180,  200,  220,  240,  260,  280, \
    300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, \
    -1 /* -1 means infinity */ \

#define VALUES_NR_PDCP_T_REORDERING_STR \
    "ms0", "ms1", "ms2", "ms4", "ms5", "ms8", "ms10", "ms15", "ms20", "ms30", "ms40", "ms50", \
    "ms60", "ms80", "ms100", "ms120", "ms140", "ms160", "ms180", "ms200", "ms220", "ms240", "ms260", "ms280", \
    "ms300", "ms500", "ms750", "ms1000", "ms1250", "ms1500", "ms1750", "ms2000", "ms2250", "ms2500", "ms2750", "ms3000", \
    "infinity"

#define SIZEOF_NR_PDCP_T_REORDERING INT_LIST_SIZE(VALUES_NR_PDCP_T_REORDERING)

#define VALUES_NR_PDCP_DISCARD_TIMER \
    10, 20, 30, 40, 50, 60, 75, 100, 150, 200, 250, 300, 500, 750, 1500, -1 /* -1 means infinity */

#define VALUES_NR_PDCP_DISCARD_TIMER_STR \
    "ms10", "ms20", "ms30", "ms40", "ms50", "ms60", "ms75", "ms100", "ms150", "ms200", "ms250", "ms300", "ms500", "ms750", "ms1500", "infinity"

#define SIZEOF_NR_PDCP_DISCARD_TIMER INT_LIST_SIZE(VALUES_NR_PDCP_DISCARD_TIMER)

/* configuration of NR PDCP, coming from configuration file */
typedef struct {
  struct {
    int sn_size;
    int t_reordering;
    int discard_timer;
  } drb;
} nr_pdcp_configuration_t;

#endif /* _NR_PDCP_CONFIGURATION_H_ */
