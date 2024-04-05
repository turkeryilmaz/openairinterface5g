#ifndef SS_ENB_DRB_TASK_H_
#define SS_ENB_DRB_TASK_H_

void  ss_eNB_drb_init(void);
void *ss_eNB_drb_process_itti_msg(void *);
void *ss_eNB_drb_task(void *arg);
void *ss_eNB_drb_acp_task(void *arg);

#endif /* SS_ENB_DRB_TASK_H_ */

