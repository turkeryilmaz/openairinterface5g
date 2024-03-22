#ifndef SS_GNB_DRB_TASK_H_
#define SS_GNB_DRB_TASK_H_

void  ss_gNB_drb_init(void);
void *ss_gNB_drb_process_itti_msg(void *);
void *ss_gNB_drb_task(void *arg);
void *ss_gNB_drb_acp_task(void *arg);

#endif /* SS_GNB_DRB_TASK_H_ */
