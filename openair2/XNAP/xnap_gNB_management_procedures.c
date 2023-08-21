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
 *      conmnc_digit_lengtht@openairinterface.org
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "intertask_interface.h"
#include "assertions.h"
#include "conversions.h"
#include "xnap_common.h"
#include "xnap_gNB_defs.h"
#include "xnap_gNB_task.h"


#define XNAP_DEBUG_LIST
#ifdef XNAP_DEBUG_LIST
#  define XNAP_gNB_LIST_OUT(x, args...) LOG_I(XNAP, "[gNB]%*s"x"\n", 4*indent, "", ##args)
#else
#  define XNAP_gNB_LIST_OUT(x, args...)
#endif

static int                  indent = 0;


xnap_gNB_internal_data_t xnap_gNB_internal_data;

RB_GENERATE(xnap_gnb_map, xnap_gNB_data_s, entry, xnap_gNB_compare_assoc_id);

int xnap_gNB_compare_assoc_id(
  struct xnap_gNB_data_s *p1, struct xnap_gNB_data_s *p2)
{
  if (p1->assoc_id == -1) {
    if (p1->cnx_id < p2->cnx_id) {
      return -1;
    }

    if (p1->cnx_id > p2->cnx_id) {
      return 1;
    }
  } else {
    if (p1->assoc_id < p2->assoc_id) {
      return -1;
    }

    if (p1->assoc_id > p2->assoc_id) {
      return 1;
    }
  }

  /* Matching reference */
  return 0;
}

uint16_t xnap_gNB_fetch_add_global_cnx_id(void)
{
  return ++xnap_gNB_internal_data.global_cnx_id;
}

void xnap_gNB_prepare_internal_data(void)
{
  memset(&xnap_gNB_internal_data, 0, sizeof(xnap_gNB_internal_data));
  STAILQ_INIT(&xnap_gNB_internal_data.xnap_gNB_instances_head);
}

void xnap_gNB_insert_new_instance(xnap_gNB_instance_t *new_instance_p)
{
  DevAssert(new_instance_p != NULL);

  STAILQ_INSERT_TAIL(&xnap_gNB_internal_data.xnap_gNB_instances_head,
                     new_instance_p, xnap_gNB_entries);
}

void xnap_dump_tree(xnap_gNB_data_t *t)
{
  if (t == NULL) return;
  printf("-----------------------\n");
  printf("gNB id %d %s\n", t->gNB_id, t->gNB_name);
  printf("state %d\n", t->state);
  printf("nextstream %d\n", t->nextstream);
  printf("in_streams %d out_streams %d\n", t->in_streams, t->out_streams);
  printf("cnx_id %d assoc_id %d\n", t->cnx_id, t->assoc_id);
  xnap_dump_tree(t->entry.rbe_left);
  xnap_dump_tree(t->entry.rbe_right);
}

void xnap_dump_trees(void)
{
xnap_gNB_instance_t *zz;
STAILQ_FOREACH(zz, &xnap_gNB_internal_data.xnap_gNB_instances_head,
               xnap_gNB_entries) {
printf("here comes the tree (instance %ld):\n---------------------------------------------\n", zz->instance);
xnap_dump_tree(zz->xnap_gnb_head.rbh_root);
printf("---------------------------------------------\n");
}
}

struct xnap_gNB_data_s *xnap_get_gNB(xnap_gNB_instance_t *instance_p,
				     int32_t assoc_id,
				     uint16_t cnx_id)
{
  struct xnap_gNB_data_s  temp;
  struct xnap_gNB_data_s *found;

  memset(&temp, 0, sizeof(struct xnap_gNB_data_s));
  temp.assoc_id = assoc_id;
  temp.cnx_id   = cnx_id;

  if (instance_p == NULL) {
    STAILQ_FOREACH(instance_p, &xnap_gNB_internal_data.xnap_gNB_instances_head,
                   xnap_gNB_entries) {
      found = RB_FIND(xnap_gnb_map, &instance_p->xnap_gnb_head, &temp);

      if (found != NULL) {
        return found;
      }
    }
  } else {
    return RB_FIND(xnap_gnb_map, &instance_p->xnap_gnb_head, &temp);
  }

  return NULL;
}


xnap_gNB_instance_t *xnap_gNB_get_instance(instance_t instance)
{
  xnap_gNB_instance_t *temp = NULL;

  STAILQ_FOREACH(temp, &xnap_gNB_internal_data.xnap_gNB_instances_head,
                 xnap_gNB_entries) {
    if (temp->instance == instance) {
      /* Matching occurence */
      return temp;
    }
  }

  return NULL;
}

/// utility functions

void xnap_dump_gNB (xnap_gNB_data_t  * gNB_ref);

void
xnap_dump_gNB_list (void) {
   xnap_gNB_instance_t *inst = NULL;
   struct xnap_gNB_data_s *found = NULL;
   struct xnap_gNB_data_s temp;

   memset(&temp, 0, sizeof(struct xnap_gNB_data_s));

  STAILQ_FOREACH (inst, &xnap_gNB_internal_data.xnap_gNB_instances_head,  xnap_gNB_entries) {
    found = RB_FIND(xnap_gnb_map, &inst->xnap_gnb_head, &temp);
    xnap_dump_gNB (found);
  }
}

void xnap_dump_gNB (xnap_gNB_data_t  * gNB_ref) {

  if (gNB_ref == NULL) {
    return;
  }

  XNAP_gNB_LIST_OUT ("");
  XNAP_gNB_LIST_OUT ("gNB name:          %s", gNB_ref->gNB_name == NULL ? "not present" : gNB_ref->gNB_name);
  XNAP_gNB_LIST_OUT ("gNB STATE:         %07x", gNB_ref->state);
  XNAP_gNB_LIST_OUT ("gNB ID:            %07x", gNB_ref->gNB_id);
  indent++;
  XNAP_gNB_LIST_OUT ("SCTP cnx id:     %d", gNB_ref->cnx_id);
  XNAP_gNB_LIST_OUT ("SCTP assoc id:     %d", gNB_ref->assoc_id);
  XNAP_gNB_LIST_OUT ("SCTP instreams:    %d", gNB_ref->in_streams);
  XNAP_gNB_LIST_OUT ("SCTP outstreams:   %d", gNB_ref->out_streams);
  indent--;
}

xnap_gNB_data_t  * xnap_is_gNB_pci_in_list (const uint32_t pci)
{
  xnap_gNB_instance_t    *inst;
  struct xnap_gNB_data_s *elm;

  STAILQ_FOREACH(inst, &xnap_gNB_internal_data.xnap_gNB_instances_head, xnap_gNB_entries) {
    RB_FOREACH(elm, xnap_gnb_map, &inst->xnap_gnb_head) {
    if (elm->Nid_cell== pci) {
        return elm;
        }
    }
  }
  return NULL;
}

xnap_gNB_data_t  * xnap_is_gNB_id_in_list (const uint32_t gNB_id)
{
  xnap_gNB_instance_t    *inst;
  struct xnap_gNB_data_s *elm;

  STAILQ_FOREACH(inst, &xnap_gNB_internal_data.xnap_gNB_instances_head, xnap_gNB_entries) {
    RB_FOREACH(elm, xnap_gnb_map, &inst->xnap_gnb_head) {
      if (elm->gNB_id == gNB_id)
        return elm;
    }
  }
  return NULL;
}

xnap_gNB_data_t  * xnap_is_gNB_assoc_id_in_list (const uint32_t sctp_assoc_id)
{
  xnap_gNB_instance_t    *inst;
  struct xnap_gNB_data_s *found;
  struct xnap_gNB_data_s temp;

  temp.assoc_id = sctp_assoc_id;
  temp.cnx_id = -1;

  STAILQ_FOREACH(inst, &xnap_gNB_internal_data.xnap_gNB_instances_head, xnap_gNB_entries) {
    found = RB_FIND(xnap_gnb_map, &inst->xnap_gnb_head, &temp);
    if (found != NULL){
      if (found->assoc_id == sctp_assoc_id) {
	return found;
      }
    }
  }
  return NULL;
}
