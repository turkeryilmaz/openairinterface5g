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

#include <pthread.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <net/if.h>

#include <netinet/in.h>
#include <netinet/sctp.h>

#include <arpa/inet.h>

#include "assertions.h"
#include "common/utils/system.h"
#include "queue.h"
#include "sctp_common.h"

#include "intertask_interface.h"
#include "common/ran_context.h"

#include "ss_eNB_context.h"
#include "ss_eNB_multicell_helper.h"

int get_cell_index(uint16_t cell_id, SS_Cell_Context_t SSCell_list[]){
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if((SSCell_list[Cell_idx].cell_configured_flag == true)&&(SSCell_list[Cell_idx].eutra_cellId == cell_id)){
      return Cell_idx;
    }
  }
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if(SSCell_list[Cell_idx].cell_configured_flag == false){
      SSCell_list[Cell_idx].cell_configured_flag = true;
      return Cell_idx;
    }
  }
  return -1;
}

int get_cell_index_pci(uint16_t physCellId, SS_Cell_Context_t SSCell_list[]){
  for(int Cell_idx = 0; Cell_idx < 8; Cell_idx++){
    if(SSCell_list[Cell_idx].PhysicalCellId == physCellId){
      return Cell_idx;
    }
  }
  return -1;
}
