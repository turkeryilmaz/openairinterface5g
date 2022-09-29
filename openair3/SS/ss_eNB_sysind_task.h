#ifndef SS_ENB_SYSIND_TASK_H_
#define SS_ENB_SYSIND_TASK_H_

void  ss_eNB_sysind_init(void);
void *ss_eNB_sysind_process_itti_msg(void *);
void *ss_eNB_sysind_task(void *arg);
void *ss_eNB_sysind_acp_task(void *arg);

#endif /* SS_ENB_SYSIND_TASK_H_ */                                
