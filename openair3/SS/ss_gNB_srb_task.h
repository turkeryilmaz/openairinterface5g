#ifndef SS_GNB_SRB_TASK_H_
#define SS_GNB_SRB_TASK_H_

void  ss_gNB_srb_init(void);
void *ss_gNB_srb_process_itti_msg(void *);
void *ss_gNB_srb_task(void *arg);
void *ss_gNB_srb_acp_task(void *arg);

#endif /* SS_GNB_SRB_TASK_H_ */

