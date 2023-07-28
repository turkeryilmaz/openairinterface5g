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
 *
 *
 * AUTHOR  : Vijay Chadachan
 * COMPANY : Firecell
 * EMAIL   : Vijay.chadachan@firecell.io
 */

#include "intertask_interface.h"
#include "common/ran_context.h"

#include "ss_eNB_context.h"

#include "ss_eNB_proxy_iface.h"
#include "ss_eNB_vt_timer_task.h"
#include <assert.h>


extern SSConfigContext_t SS_context;
extern RAN_CONTEXT_t RC;

/*
 * Function : _vt_subtract_sf
 * Description: Helper function to substract offset to SFN_SF
 */
static void _vt_subtract_sf(uint16_t *frameP, uint8_t *subframeP, int offset)
{
  if (*subframeP < offset)
  {
    *frameP = (*frameP+1024-1)%1024;
  }
  *subframeP = (*subframeP+10-offset)%10;
}

/*
 * Function : vt_add_sf
 * Description: Helper function to add offset to SFN_SF
 */

static void _vt_add_sf(uint16_t *frameP, uint8_t *subframeP, int offset)
{
	if (offset > 0) {
		*frameP    = (*frameP + ((*subframeP + offset) / 10)) % 1024;
		*subframeP = ((*subframeP + offset) % 10);
	} else {
	   _vt_subtract_sf(frameP,subframeP,0-offset);
	}
}

/*
 * Function : msg_can_be_queued
 * Description: Helper function to check if the received MSG shall be queued
 */
static uint8_t _msg_can_be_queued(ss_set_timinfo_t req_tinfo, ss_set_timinfo_t *timer_tinfo)
{
	ss_set_timinfo_t curr_tinfo;
	curr_tinfo.sfn = SS_context.sfn;
	curr_tinfo.sf = SS_context.sf;

	LOG_A(ENB_SS_VT_TIMER,"VT_TIMER Enter msg_can_be_queued for  SFN %d , SF %d\n",req_tinfo.sfn,req_tinfo.sf);

   /*It is nonsense to check req_tinfo is after curr_tinfo */
	if(req_tinfo.sfn != curr_tinfo.sfn || ((req_tinfo.sfn == curr_tinfo.sfn) && (req_tinfo.sf - curr_tinfo.sf) > 0) )
	{
		_vt_subtract_sf(&timer_tinfo->sfn,&timer_tinfo->sf, 4); /* queued ahead of 4 subframes because of mac schedule 4 subframes ahead when processing */
		LOG_A(ENB_SS_VT_TIMER,"VT_TIMER MSG to be queued  TRUE for  SFN %d , SF %d\n",timer_tinfo->sfn,timer_tinfo->sf);
		return true;
	}

	return false;
}

/*
 * Function : vt_timer_setup
 * Description: Function to set upt the VT timer for the SFN_SF
 * and store the message to received
 */
static uint8_t _vt_timer_setup(ss_set_timinfo_t* tinfo, task_id_t task_id,instance_t instance, void *msg)
{
	uint32_t sfnSfKey = (tinfo->sfn << 4) | tinfo->sf;

	vt_timer_elm_t *timer_ele_p = calloc(1, sizeof(vt_timer_elm_t));
	assert(timer_ele_p);
	timer_ele_p->instance = instance;
	timer_ele_p->task_id = task_id;
	timer_ele_p->msg = msg;

	if (hashtable_insert(SS_context.vt_timer_table,
	   (hash_key_t)sfnSfKey, (void *)timer_ele_p) == HASH_TABLE_OK)
	{
		LOG_A(ENB_SS_VT_TIMER,"VT_TIMER setup for  SFN %d , SF %d\n",tinfo->sfn,tinfo->sf);
		return 1;
	}

	return false;
}

void vt_add_sf(struct TimingInfo_Type* at, int offset)
{
	if (at != NULL && at->d == TimingInfo_Type_SubFrame)
	{
		uint16_t sfn = SS_context.sfn;
		uint8_t sf = SS_context.sf;
		if (at->v.SubFrame.SFN.d == SystemFrameNumberInfo_Type_Number)
		{
			sfn = at->v.SubFrame.SFN.v.Number;
		}

		if (at->v.SubFrame.Subframe.d == SubFrameInfo_Type_Number)
		{
			sf = at->v.SubFrame.Subframe.v.Number;
		}
		LOG_D(ENB_SS_VT_TIMER,"vt_add_sf SFN %d SF %d offset %d\n",sfn,sf,offset);
		_vt_add_sf(&sfn, &sf, offset);
		at->v.SubFrame.SFN.d = SystemFrameNumberInfo_Type_Number;
		at->v.SubFrame.SFN.v.Number = sfn;
		at->v.SubFrame.Subframe.d = SubFrameInfo_Type_Number;
		at->v.SubFrame.Subframe.v.Number = sf;
	}
}


int vt_timer_push_msg(struct TimingInfo_Type* at, task_id_t task_id,instance_t instance, MessageDef *msg_p)
{
	int msg_queued = 0;
	if (at != NULL && at->d == TimingInfo_Type_SubFrame)
	{
		ss_set_timinfo_t timer_tinfo = {
			.sfn = SS_context.sfn,
			.sf = SS_context.sf + 6,
		};

		if (at->v.SubFrame.SFN.d == SystemFrameNumberInfo_Type_Number)
		{
			timer_tinfo.sfn = at->v.SubFrame.SFN.v.Number;
		}

		if (at->v.SubFrame.Subframe.d == SubFrameInfo_Type_Number)
		{
			timer_tinfo.sf = at->v.SubFrame.Subframe.v.Number;
		}
		else
		{
			return 0;
		}

		msg_queued = _msg_can_be_queued(timer_tinfo,&timer_tinfo);
		if(msg_queued)
		{
			msg_queued = _vt_timer_setup(&timer_tinfo, task_id, instance ,msg_p);
			LOG_A(ENB_SS_VT_TIMER, "Message Queued as the scheduled SFN is %d SF: %d and curr SFN %d , SF %d, msg_queued: %d\r\n",
						timer_tinfo.sfn,timer_tinfo.sf, SS_context.sfn,SS_context.sf, msg_queued);
		}
	}
	return msg_queued;
}

/*
 * Function : ss_vt_timer_check
 * Description: Function to check if any SFN_SF is timed out and forward
 * the stored message to requested TASK.
 */
static inline void ss_vt_timer_check(ss_set_timinfo_t tinfo)
{
	  vt_timer_elm_t *timer_ele_p;


	  uint32_t sfnSfKey = (tinfo.sfn << 4) | tinfo.sf;
	  //printf("VT_TIMER foudn queued SFN %d , SF %d\n",tinfo.sfn,tinfo.sf);
	  while (hashtable_is_key_exists(SS_context.vt_timer_table, (hash_key_t)sfnSfKey) == HASH_TABLE_OK)
	  {
		  LOG_D(ENB_SS_VT_TIMER,"VT_TIMER  Timeout sending  curr SFN %d SF %d\n",
		  					SS_context.sfn,SS_context.sf);

		  hashtable_get(SS_context.vt_timer_table, (hash_key_t)sfnSfKey, (void **)&timer_ele_p);
		  AssertFatal(timer_ele_p, "VT Timer - timer element is NULL");
		  LOG_A(ENB_SS_VT_TIMER,"VT_TIMER Enter check SFN %d , SF %d taskID %d timer_ele.task_id instance %ld \n",

						  tinfo.sfn,tinfo.sf, timer_ele_p->task_id,timer_ele_p->instance);

		  int send_res = itti_send_msg_to_task(timer_ele_p->task_id,timer_ele_p->instance, (MessageDef *)timer_ele_p->msg);
		  if (send_res < 0)
		  {
				LOG_E(ENB_SS_VT_TIMER, "[VT_TIMER] Error in SS_VT_TIME_OUT itti_send_msg_to_task");
		  }
		  else
		  {
			  LOG_A(ENB_SS_VT_TIMER,"VT_TIMER Sent message to  taskID %d timer_ele.task_id instance %ld \n",
			  						  timer_ele_p->task_id,timer_ele_p->instance);
			  hashtable_remove(SS_context.vt_timer_table, (hash_key_t)sfnSfKey);
		  }
		  LOG_D(ENB_SS_VT_TIMER,"VT_TIMER  Timeout sending done curr SFN %d SF %d\n",
		 		  					SS_context.sfn,SS_context.sf);
	  }
}

/*
 * Function : ss_eNB_vt_timer_process_itti_msg
 * Description: Function to hanle the ITTI messages for VT_TIMER_TASK
 */
void *ss_eNB_vt_timer_process_itti_msg(void *notUsed)
{
    MessageDef *received_msg = NULL;
    int result;
    itti_receive_msg(TASK_VT_TIMER, &received_msg);

    /* Check if there is a packet to handle */
    if (received_msg != NULL)
    {
        switch (ITTI_MSG_ID(received_msg))
        {
        case SS_UPD_TIM_INFO:
        {
            ss_set_timinfo_t tinfo;
            tinfo.sf = SS_UPD_TIM_INFO(received_msg).sf;
            tinfo.sfn = SS_UPD_TIM_INFO(received_msg).sfn;
            LOG_D(ENB_SS_VT_TIMER, "[VT_TIMER] received_UPD_TIM_INFO SFN: %d SF: %d\n", tinfo.sfn, tinfo.sf);

            ss_vt_timer_check(tinfo);
        }
        break;
        case TERMINATE_MESSAGE:
        {
            itti_exit_task();
            break;
        }
        default:
            LOG_E(ENB_SS_VT_TIMER, "[SS-VT_TIMER] Received unhandled message %d:%s\n",
                  ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
        }
        result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
        AssertFatal(result == EXIT_SUCCESS, "[VT_TIMER] Failed to free memory (%d)!\n", result);
        received_msg = NULL;
    }

    return NULL;
}

/*
 * Function : ss_eNB_vt_timer_init
 * Description: Function to initialize the TASK_VT_TIMER
 */
int ss_eNB_vt_timer_init(void)
{
    SS_context.vt_timer_table = hashtable_create (32, NULL, NULL);

    itti_mark_task_ready(TASK_VT_TIMER);
    return 0;
}
/*
 * Function : ss_eNB_vt_timer_task
 * Description: Function to create the TASK_VT_TIMER
 */
void* ss_eNB_vt_timer_task(void *arg) {

	int retVal = ss_eNB_vt_timer_init();

	if (retVal != -1) {
		LOG_A(ENB_SS_VT_TIMER, "[SS-VT_TIMER] Enabled TASK_VT_TIMER starting the itti_msg_handler \n");

		while (1) {
			(void) ss_eNB_vt_timer_process_itti_msg(NULL);
		}
	} else {

		LOG_A(ENB_SS_VT_TIMER, "[SS-VT_TIMER] TASK_VT_TIMER port disabled at eNB \n");
		sleep(10);
	}

	return NULL;
}
