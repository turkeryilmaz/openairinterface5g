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

#include "intertask_interface.h"
#include "common/ran_context.h"
#include "ss_gNB_context.h"
#include "ss_gNB_proxy_iface.h"
#include "ss_gNB_vt_timer_task.h"
#include <assert.h>

extern SSConfigContext_t SS_context;

static inline uint32_t _vt_sfn_slot(uint16_t sfn, uint32_t slot)
{
  return ((sfn << 8) | slot);
}

/*
 * Function : _vt_subtract_slot
 * Description: Helper function to substract offset to SFN_SLOT
 */
static void _vt_subtract_slot(uint16_t *frameP, uint32_t *slotP, int offset)
{
  if (*slotP < offset)
  {
    *frameP = (*frameP+1024-1)%1024;
  }
  uint8_t slotsPerFrame = 10 *(1<<SS_context.mu);
  *slotP = (*slotP+slotsPerFrame-offset)%slotsPerFrame;
}

/*
 * Function : _vt_add_slot
 * Description: Helper function to add offset to SFN_SLOT
 */

static void _vt_add_slot(uint16_t *frameP, uint32_t *slotP, int offset)
{
	if (offset > 0) {
      uint8_t slotsPerFrame = 10 *(1<<SS_context.mu);
		*frameP    = (*frameP + ((*slotP + offset) / slotsPerFrame)) % 1024;
		*slotP = ((*slotP + offset) % slotsPerFrame);
	} else {
		_vt_subtract_slot(frameP, slotP, 0-offset);
	}
}

/*
 * Function : _nr_msg_can_be_queued
 * Description: Helper function to check if the received MSG shall be queued
 */
uint8_t _nr_msg_can_be_queued(ss_nrset_timinfo_t * req_tinfo)
{
	ss_nrset_timinfo_t curr_tinfo;
	curr_tinfo.sfn = SS_context.sfn;
	curr_tinfo.slot = SS_context.slot;

	LOG_D(GNB_APP,"VT_TIMER Enter msg_can_be_queued for  SFN %d , SLOT %d\n",req_tinfo->sfn,req_tinfo->slot);

   /*It is nonsense to check req_tinfo is after curr_tinfo */
	if(req_tinfo->sfn != curr_tinfo.sfn || ((req_tinfo->sfn == curr_tinfo.sfn) && (req_tinfo->slot - curr_tinfo.slot) > 0) )
	{
	   /* The actual DL message would be transfered in next slot after receiving SS_NRUPD_TIM_INFO(last DL slot TIME update) */
		_vt_subtract_slot(&req_tinfo->sfn, &req_tinfo->slot, 1);
		LOG_D(GNB_APP,"VT_TIMER MSG to be queued  TRUE for  SFN %d , SLOT %d\n",req_tinfo->sfn,req_tinfo->slot);
		return true;
	}

	return false;
}

/*
 * Function : _nr_vt_timer_setup
 * Description: Function to set upt the VT timer for the SFN_SLOT
 * and store the message to received
 */
static uint8_t _nr_vt_timer_setup(ss_nrset_timinfo_t* tinfo, task_id_t task_id,instance_t instance, void *msg)
{
	uint32_t sfnSlotKey =_vt_sfn_slot(tinfo->sfn, tinfo->slot);

	vt_timer_elm_t *timer_ele_p = calloc(1, sizeof(vt_timer_elm_t));
	assert(timer_ele_p);
	timer_ele_p->instance = instance;
	timer_ele_p->task_id = task_id;
	timer_ele_p->msg = msg;

	if (hashtable_insert(SS_context.vt_timer_table,
	   (hash_key_t)sfnSlotKey, (void *)timer_ele_p) == HASH_TABLE_OK)
	{
		LOG_D(GNB_APP,"VT_TIMER setup for  SFN %d , SLOT %d\n", tinfo->sfn,tinfo->slot);
		return 1;
	}
	return 0;
}

void nr_vt_add_slot(struct TimingInfo_Type* at, int offset)
{
  if (at!=NULL && at->d == TimingInfo_Type_SubFrame)
  {
    uint16_t sfn = SS_context.sfn;
    uint32_t slot = SS_context.slot;
    uint8_t slotsPerSubFrame = 1<<SS_context.mu;

    if (at->v.SubFrame.SFN.d == SystemFrameNumberInfo_Type_Number)
    {
      sfn = at->v.SubFrame.SFN.v.Number;
    }

    if (at->v.SubFrame.Subframe.d == SubFrameInfo_Type_Number)
    {
      slot = at->v.SubFrame.Subframe.v.Number * slotsPerSubFrame;
    }

    if(at->v.SubFrame.Slot.d == SlotTimingInfo_Type_SlotOffset)
    {
      if(at->v.SubFrame.Slot.v.SlotOffset.d >= SlotOffset_Type_Numerology1){
        slot += at->v.SubFrame.Slot.v.SlotOffset.v.Numerology1;
      }
    }
    if(offset !=0){
      _vt_add_slot(&sfn, &slot, offset);
    }
    at->d = TimingInfo_Type_SubFrame;
    at->v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
    at->v.SubFrame.SFN.v.Number = sfn;
    at->v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
    at->v.SubFrame.Subframe.v.Number = slot/slotsPerSubFrame;
    at->v.SubFrame.Slot.d = SlotTimingInfo_Type_SlotOffset;
    at->v.SubFrame.Slot.v.SlotOffset.d = (enum SlotOffset_Type_Sel)(SS_context.mu+1);
    if(at->v.SubFrame.Slot.v.SlotOffset.d > SlotOffset_Type_Numerology0){
      at->v.SubFrame.Slot.v.SlotOffset.v.Numerology1 = slot%slotsPerSubFrame;
    }else {
      at->v.SubFrame.Slot.v.SlotOffset.v.Numerology0 = true;
    }
  }
}

int nr_vt_timer_push_msg(struct TimingInfo_Type* at, int32_t slotOffset, task_id_t task_id,instance_t instance, MessageDef *msg_p)
{
  int msg_queued = 0;
  if (at != NULL && at->d == TimingInfo_Type_SubFrame)
  {
    ss_nrset_timinfo_t timer_tinfo;
    if (at->v.SubFrame.SFN.d == SystemFrameNumberInfo_Type_Number)
    {
      timer_tinfo.sfn = at->v.SubFrame.SFN.v.Number;
    }

    if (at->v.SubFrame.Subframe.d == SubFrameInfo_Type_Number)
    {
      uint8_t slotsPerSubFrame = 1<<SS_context.mu;
      timer_tinfo.slot = at->v.SubFrame.Subframe.v.Number * slotsPerSubFrame;
    }
    else
    {
      return 0;
    }

    if(at->v.SubFrame.Slot.d == SlotTimingInfo_Type_SlotOffset)
    {
      if(at->v.SubFrame.Slot.v.SlotOffset.d >= SlotOffset_Type_Numerology1){
        timer_tinfo.slot += at->v.SubFrame.Slot.v.SlotOffset.v.Numerology1;
      }
    }

    _vt_add_slot(&timer_tinfo.sfn, &timer_tinfo.slot, slotOffset);
    msg_queued = _nr_msg_can_be_queued(&timer_tinfo);
    if(msg_queued)
    {
      msg_queued = _nr_vt_timer_setup(&timer_tinfo, task_id, instance ,msg_p);
      LOG_D(GNB_APP, "Message Queued as the scheduled SFN is %d SLOT: %d and curr SFN %d , SLOT %d, msg_queued: %d\r\n",
      timer_tinfo.sfn,timer_tinfo.slot, SS_context.sfn,SS_context.slot, msg_queued);
    }
  }
  return msg_queued;
}

/*
 * Function : ss_nr_vt_timer_check
 * Description: Function to check if any SFN_SF is timed out and forward
 * the stored message to requested TASK.
 */
static inline void ss_nr_vt_timer_check(ss_nrset_timinfo_t tinfo)
{
  vt_timer_elm_t *timer_ele_p;

  uint32_t sfnSlotKey =_vt_sfn_slot(tinfo.sfn, tinfo.slot);
  while (hashtable_is_key_exists(SS_context.vt_timer_table, (hash_key_t)sfnSlotKey) == HASH_TABLE_OK)
  {
    LOG_D(GNB_APP,"VT_TIMER  Timeout sending  curr SFN %d SLOT %d\n",SS_context.sfn,SS_context.slot);

    hashtable_get(SS_context.vt_timer_table, (hash_key_t)sfnSlotKey, (void **)&timer_ele_p);
    AssertFatal(timer_ele_p, "VT Timer - timer element is NULL");
    LOG_D(GNB_APP,"VT_TIMER Enter check SFN %d , SLOT %d taskID %d timer_ele.task_id instance %ld \n",
        tinfo.sfn,tinfo.slot, timer_ele_p->task_id,timer_ele_p->instance);

    int send_res = itti_send_msg_to_task(timer_ele_p->task_id,timer_ele_p->instance, (MessageDef *)timer_ele_p->msg);
    if (send_res < 0)
    {
      LOG_E(GNB_APP, "[VT_TIMER] Error in SS_VT_TIME_OUT itti_send_msg_to_task");
    }
    else
    {
      LOG_A(GNB_APP,"VT_TIMER Sent message to  taskID %d timer_ele.task_id instance %ld \n",
			  timer_ele_p->task_id,timer_ele_p->instance);
      hashtable_remove(SS_context.vt_timer_table, (hash_key_t)sfnSlotKey);
    }
    LOG_D(GNB_APP,"VT_TIMER  Timeout sending done curr SFN %d SLOT %d\n",
				SS_context.sfn,SS_context.slot);
  }
}

/*
 * Function : ss_gNB_vt_timer_process_itti_msg
 * Description: Function to hanle the ITTI messages for VT_TIMER_TASK
 */
void *ss_gNB_vt_timer_process_itti_msg(void *notUsed)
{
  MessageDef *received_msg = NULL;
  int result;
  itti_receive_msg(TASK_VT_TIMER, &received_msg);

  /* Check if there is a packet to handle */
  if (received_msg != NULL)
  {
    switch (ITTI_MSG_ID(received_msg))
    {
      case SS_NRUPD_TIM_INFO:
      {
        ss_nrset_timinfo_t tinfo;
        tinfo.sfn = SS_NRUPD_TIM_INFO(received_msg).sfn;
        tinfo.slot = SS_NRUPD_TIM_INFO(received_msg).slot;
        LOG_D(GNB_APP, "[VT_TIMER] received SS_NRUPD_TIM_INFO SFN: %d Slot: %d\n", tinfo.sfn, tinfo.slot);

        ss_nr_vt_timer_check(tinfo);
      }
      break;

      case TERMINATE_MESSAGE:
      {
        itti_exit_task();
        break;
      }
      default:
        LOG_E(GNB_APP, "[SS-VT_TIMER] Received unhandled message %d:%s\n",
          ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
    }
    result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
    AssertFatal(result == EXIT_SUCCESS, "[VT_TIMER] Failed to free memory (%d)!\n", result);
    received_msg = NULL;
  }

  return NULL;
}

/*
 * Function : ss_gNB_vt_timer_init
 * Description: Function to initialize the TASK_VT_TIMER
 */
int ss_gNB_vt_timer_init(void)
{
  SS_context.vt_timer_table = hashtable_create (32, NULL, NULL);
  itti_mark_task_ready(TASK_VT_TIMER);
  return 0;
}
/*
 * Function : ss_gNB_vt_timer_task
 * Description: Function to create the TASK_VT_TIMER
 */
void* ss_gNB_vt_timer_task(void *arg) {
  int retVal = ss_gNB_vt_timer_init();

  if (retVal != -1) {
    LOG_A(GNB_APP, "[SS-VT_TIMER] Enabled TASK_VT_TIMER starting the itti_msg_handler \n");

    while (1) {
      (void) ss_gNB_vt_timer_process_itti_msg(NULL);
    }
  } else {
    LOG_A(GNB_APP, "[SS-VT_TIMER] TASK_VT_TIMER port disabled at gNB \n");
    sleep(10);
  }
  return NULL;
}
