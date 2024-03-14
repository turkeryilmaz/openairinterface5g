#ifndef SS_ENB_SRB_TASK_H_
#define SS_ENB_SRB_TASK_H_

void  ss_eNB_srb_init(void);
void *ss_eNB_srb_process_itti_msg(void *);
void *ss_eNB_srb_task(void *arg);
void *ss_eNB_srb_acp_task(void *arg);
#endif /* SS_ENB_SRB_TASK_H_ */
