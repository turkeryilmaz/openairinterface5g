#define _GNU_SOURCE
#include <unistd.h>

#include "task_manager.h"

#include <assert.h> 
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/sysinfo.h>

#include <fcntl.h>

#include <linux/futex.h>      /* Definition of FUTEX_* constants */
#include <sys/syscall.h>      /* Definition of SYS_* constants */
#include <unistd.h>

#if defined (__i386__) || defined(__x86_64__)
  #define pause_or_yield  __builtin_ia32_pause
#elif __aarch64__
  #define pause_or_yield() asm volatile("yield" ::: "memory")
#else
    static_assert(0!=0, "Unknown CPU architecture");
#endif

/*
static
int64_t time_now_us(void)
{
  struct timespec tms;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &tms)) {
    return -1;
  }
  int64_t micros = tms.tv_sec * 1000000;
  int64_t const tv_nsec = tms.tv_nsec;
  micros += tv_nsec/1000;
  if (tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return micros;
}

static
void pin_thread_to_core(int core_num)
{
  cpu_set_t set = {0};
  CPU_ZERO(&set);
  CPU_SET(core_num, &set);
  int ret = sched_setaffinity(gettid(), sizeof(set), &set);
  assert(ret != -1); 
  printf("Pining into core %d id %ld \n", core_num, pthread_self());
}
*/

//////////////////////////////
//////////////////////////////
////////// RING //
//////////////////////////////
//////////////////////////////
//////////////////////////////



typedef struct seq_ring_buf_s
{
//  const size_t elt_size;
  task_t* array;

  size_t cap;
  uint32_t head;
  uint32_t tail;

  _Atomic uint64_t sz;

} seq_ring_task_t;

typedef void (*seq_free_func)(task_t*); 

// For working correctly, maintain the default elements to a multiple of 2
#define DEFAULT_ELM 32

static
size_t size_seq_ring_task(seq_ring_task_t* r)
{
  assert(r != NULL);

  return r->head - r->tail;
}

inline static
uint32_t mask(uint32_t cap, uint32_t val)
{
  return val & (cap-1);
}

static 
bool full(seq_ring_task_t* r)
{
  return size_seq_ring_task(r) == r->cap -1;
}

static
void enlarge_buffer(seq_ring_task_t* r)
{
  assert(r != NULL);
  assert(full(r));

  const uint32_t factor = 2;
  task_t* tmp_buffer = calloc(r->cap * factor, sizeof(task_t) );
  assert(tmp_buffer != NULL);

  const uint32_t head_pos = mask(r->cap, r->head);
  const uint32_t tail_pos = mask(r->cap, r->tail);

  if(head_pos > tail_pos){
    memcpy(tmp_buffer, r->array + tail_pos , (head_pos-tail_pos)*sizeof(task_t) );
  } else {
    memcpy(tmp_buffer, r->array + tail_pos, (r->cap-tail_pos)*sizeof(task_t));
    memcpy(tmp_buffer + (r->cap-tail_pos), r->array, head_pos*sizeof(task_t));
  }
  r->cap *= factor;
  free(r->array);
  r->array = tmp_buffer;
  r->tail = 0;
  r->head = r->cap/2 - 1;
}

static
void init_seq_ring_task(seq_ring_task_t* r)
{
  assert(r != NULL);
  task_t* tmp_buffer = calloc(DEFAULT_ELM, sizeof(task_t)); 
  assert(tmp_buffer != NULL);
  seq_ring_task_t tmp = {.array = tmp_buffer, .head = 0, .tail = 0, .cap = DEFAULT_ELM};
  memcpy(r, &tmp, sizeof(seq_ring_task_t));
  r->sz = 0;
}

static
void free_seq_ring_task(seq_ring_task_t* r, seq_free_func fp)
{
  assert(r != NULL);
  assert(fp == NULL);
  free(r->array);
}


static
void push_back_seq_ring_task(seq_ring_task_t* r, task_t t)
{
  assert(r != NULL);

  if(full(r))
    enlarge_buffer(r);
  
  const uint32_t pos = mask(r->cap, r->head);
  r->array[pos] = t;
  r->head += 1;
  r->sz += 1;
}

static
task_t pop_seq_ring_task(seq_ring_task_t* r )
{
  assert(r != NULL);
  assert(size_seq_ring_task(r) > 0);

  const uint32_t pos = mask(r->cap, r->tail);
  task_t t = r->array[pos];
  r->tail += 1; 
  r->sz -= 1;
  return t;
}

//////////////////////////////
//////////////////////////////
////////// END RING //
//////////////////////////////
//////////////////////////////
//////////////////////////////



//////////////////////////////
//////////////////////////////
////////// Start Notification Queue //
//////////////////////////////
//////////////////////////////
//////////////////////////////

typedef struct {
  pthread_mutex_t mtx;
  pthread_cond_t cv;
  seq_ring_task_t r;
  _Atomic int32_t* futex;
  //_Atomic bool* waiting;
  _Atomic int done;
} not_q_t;

typedef struct{
  task_t t;
  bool success;
} ret_try_t;


static
void init_not_q(not_q_t* q, _Atomic int32_t* futex /*, _Atomic bool* waiting */)
{
  assert(q != NULL);

  q->done = 0;
  //q->waiting = waiting;
  init_seq_ring_task(&q->r);

  pthread_mutexattr_t attr = {0};
#ifdef _DEBUG
  int const rc_mtx = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
  assert(rc_mtx == 0);
#endif
  int rc = pthread_mutex_init(&q->mtx, &attr);
  assert(rc == 0 && "Error while creating the mtx");

  pthread_condattr_t* c_attr = NULL; 
  rc = pthread_cond_init(&q->cv, c_attr);
  assert(rc == 0);

  q->futex = futex;
}

static
void free_not_q(not_q_t* q, void (*clean)(task_t*) )
{
  assert(q != NULL);
  assert(q->done == 1);

  free_seq_ring_task(&q->r, clean);

  int rc = pthread_mutex_destroy(&q->mtx);
  assert(rc == 0);

  rc = pthread_cond_destroy(&q->cv);
  assert(rc == 0);
}

static
bool try_push_not_q(not_q_t* q, task_t t)
{
  assert(q != NULL);
  assert(q->done == 0 || q->done ==1);
  assert(t.func != NULL);
  assert(t.args != NULL);

  if(pthread_mutex_trylock(&q->mtx ) != 0)
    return false;

  push_back_seq_ring_task(&q->r, t);

  int const rc = pthread_mutex_unlock(&q->mtx);
  assert(rc == 0);

  return true;
}

static
void push_not_q(not_q_t* q, task_t t)
{
  assert(q != NULL);
  assert(q->done == 0 || q->done ==1);
  assert(t.func != NULL);
  
  int const rc = pthread_mutex_lock(&q->mtx);
  assert(rc == 0);

  push_back_seq_ring_task(&q->r, t);

  pthread_mutex_unlock(&q->mtx);
}


static
ret_try_t try_pop_not_q(not_q_t* q)
{
  assert(q != NULL);

  ret_try_t ret = {.success = false}; 

  int rc = pthread_mutex_trylock(&q->mtx);
  assert(rc == 0 || rc == EBUSY);

  if(rc == EBUSY)
    return ret;

  assert(q->done == 0 || q->done ==1);

  size_t sz = size_seq_ring_task(&q->r); 
  if(sz == 0){
    rc = pthread_mutex_unlock(&q->mtx);
    assert(rc == 0);

    return ret;
  }

  ret.t = pop_seq_ring_task(&q->r);

  rc = pthread_mutex_unlock(&q->mtx);
  assert(rc == 0);
  ret.success = true; 

  return ret;
}

static
bool pop_not_q(not_q_t* q, ret_try_t* out)
{
  assert(q != NULL);
  assert(out != NULL);
  assert(q->done == 0 || q->done ==1);

label:
  // Let's be conservative and not use memory_order_relaxed
 // while (atomic_load_explicit(q->waiting, memory_order_seq_cst) == true){ //
      // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
      // hyper-threads
 //     pause_or_yield();
 //   }

  pthread_mutex_lock(&q->mtx);

  if(size_seq_ring_task(&q->r) == 0 && q->done == 0){
    int rc = pthread_mutex_unlock(&q->mtx);
    assert(rc == 0);

    int val = *q->futex; // atomic_load_explicit(q->futex, memory_order_acquire);
    long r = syscall(SYS_futex, q->futex, FUTEX_WAIT_PRIVATE, val, NULL, 0);
    assert(r != -1);
    goto label;
  }

  assert(q->done == 0 || q->done ==1);

  //printf("Waking %ld id %ld \n", time_now_us(), pthread_self());

  if(q->done == 1){
    //printf("Done, returning \n");
    int rc = pthread_mutex_unlock(&q->mtx);
    assert(rc == 0);
    return false;
  }

  out->t = pop_seq_ring_task(&q->r);

  int rc = pthread_mutex_unlock(&q->mtx);
  assert(rc == 0);

  return true;
}

static
void done_not_q(not_q_t* q)
{
  assert(q != NULL);

  int rc = pthread_mutex_lock(&q->mtx);
  assert(rc == 0);
  
  q->done = 1;
  long r = syscall(SYS_futex, q->futex, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
  assert(r != -1);
  rc = pthread_mutex_unlock(&q->mtx);
  assert(rc == 0);

//  rc = pthread_cond_signal(&q->cv);
//  assert(rc == 0);

//  q->futex++;
}


//////////////////////////////
//////////////////////////////
////////// END Notification Queue //
//////////////////////////////
//////////////////////////////
//////////////////////////////


//static int marker_fd;

typedef struct{
  task_manager_t* man;
  int idx;
} task_thread_args_t;


// Just for debugging purposes, it is very slow!!!!
//static
//_Atomic int cnt_out = 0;

//static
//_Atomic int cnt_in = 0;

static
void* worker_thread(void* arg)
{
  assert(arg != NULL);

  task_thread_args_t* args = (task_thread_args_t*)arg; 
  int const idx = args->idx;

  int const log_cores = get_nprocs_conf();
  assert(log_cores > 0);
  // Assuming: 2 x Physical cores = Logical cores
  //pin_thread_to_core(idx+log_cores/2);

  task_manager_t* man = args->man;

  uint32_t const len = man->len_thr;
  uint32_t const num_it = 2*(man->len_thr + idx); 

  not_q_t* q_arr = (not_q_t*)man->q_arr;

  int acc_num_task = 0;
  for(;;){

    ret_try_t ret = {.success = false}; 

    for(uint32_t i = idx; i < num_it; ++i){
      ret = try_pop_not_q(&q_arr[i%len]);
      if(ret.success == true){
        break;
      } 
    }

    if(ret.success == false){
      man->num_task -= acc_num_task;
      acc_num_task = 0;
      if(pop_not_q(&q_arr[idx], &ret) == false)
        break;
    }
        //int64_t now = time_now_us();
    //printf("Calling fuinc \n");
    ret.t.func(ret.t.args); 
    //printf("Returning from func \n");
    //int64_t stop = time_now_us(); 

    //cnt_out++;
    //printf("Tasks out %d %ld \n", cnt_out, time_now_us());

    acc_num_task +=1;
  }

  free(args);
  return NULL;
}

void init_task_manager(task_manager_t* man, uint32_t num_threads)
{
  assert(man != NULL);
//  assert(num_threads > 0); // && num_threads < 33 && "Do you have zero or more than 32 processors??");
  if(num_threads == 0 || num_threads > 8){
    printf("[MIR]: number of threads set to 8. %d number of threads asked \n", num_threads);
    num_threads = 8;
  }

  man->q_arr = calloc(num_threads, sizeof(not_q_t));
  assert(man->q_arr != NULL && "Memory exhausted");
  man->futex = 0; 

 // man->waiting = false;

  not_q_t* q_arr = (not_q_t*)man->q_arr;
  for(uint32_t i = 0; i < num_threads; ++i){
    init_not_q(&q_arr[i], &man->futex /*, &man->waiting */);   
  }

  man->t_arr = calloc(num_threads, sizeof(pthread_t));
  assert(man->t_arr != NULL && "Memory exhausted" );
  man->len_thr = num_threads;

  for(uint32_t i = 0; i < num_threads; ++i){
    task_thread_args_t* args = malloc(sizeof(task_thread_args_t) ); 
    args->idx = i;
    args->man = man;

    /*
    pthread_attr_t attr = {0};
    int ret=pthread_attr_init(&attr);
    assert(ret == 0);
    ret=pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    assert(ret == 0);
    ret=pthread_attr_setschedpolicy(&attr, SCHED_RR);
    assert(ret == 0);
    struct sched_param sparam={0};
    sparam.sched_priority = 94;
    ret=pthread_attr_setschedparam(&attr, &sparam);
    */

    int rc = pthread_create(&man->t_arr[i], NULL, worker_thread, args);
    assert(rc == 0);
  }

  man->index = 0;

  pthread_mutexattr_t attr = {0};
#ifdef _DEBUG
  int const rc_mtx = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
  assert(rc_mtx == 0);
#endif
  int rc = pthread_mutex_init(&man->wait_mtx, &attr);
  assert(rc == 0 && "Error while creating the mtx");

  pthread_condattr_t* c_attr = NULL; 
  rc = pthread_cond_init(&man->wait_cv, c_attr);
  assert(rc == 0);

  //pin_thread_to_core(3);
}

void free_task_manager(task_manager_t* man, void (*clean)(task_t*))
{
  not_q_t* q_arr = (not_q_t*)man->q_arr;
  //atomic_store(&man->waiting, false);

  for(uint32_t i = 0; i < man->len_thr; ++i){
    done_not_q(&q_arr[i]);
  }

  for(uint32_t i = 0; i < man->len_thr; ++i){
    int rc = pthread_join(man->t_arr[i], NULL); 
    assert(rc == 0);
  }

  for(uint32_t i = 0; i < man->len_thr; ++i){
    free_not_q(&q_arr[i], clean); 
  }

  free(man->q_arr);

  free(man->t_arr);

  int rc = pthread_mutex_destroy(&man->wait_mtx);
  assert(rc == 0);

  rc = pthread_cond_destroy(&man->wait_cv);
  assert(rc == 0);
}

void async_task_manager(task_manager_t* man, task_t t)
{
  assert(man != NULL);
  assert(man->len_thr > 0);
  assert(t.func != NULL);
  //assert(t.args != NULL);

  uint64_t const index = man->index++;
  const uint32_t len_thr = man->len_thr;

  not_q_t* q_arr = (not_q_t*)man->q_arr;
  for(uint32_t i = 0; i < len_thr ; ++i){
    if(try_push_not_q(&q_arr[(i+index) % len_thr], t)){
      man->num_task +=1;

      // Debbugging purposes
      //cnt_in++;
      //printf("Tasks in %d %ld \n", cnt_in, time_now_us());

      return;
    }
  }

  push_not_q(&q_arr[index%len_thr], t);
  man->num_task +=1;

  // Debbugging purposes
  //cnt_in++;
  //printf("Tasks in %d %ld \n", cnt_in, time_now_us());
}

/*
void trigger_and_spin_task_manager(task_manager_t* man)
{
  assert(man != NULL);
  
  atomic_store(&man->waiting, true);
  trigger_all_task_manager(man);
}

void stop_spining_task_manager(task_manager_t* man)
{
  assert(man != NULL);
  atomic_store(&man->waiting, false);
}

void trigger_and_wait_all_task_manager(task_manager_t* man)
{
  assert(man != NULL);

  trigger_and_spin_task_manager(man);
  stop_spining_task_manager(man);

  // atomic_store(&man->waiting, false);

  // GCC thread sanitizer complains, while Clang does not. Let's be conservative and pay some cycles
  // memory_order_relaxed
  //printf("Main func waiting \n");
  while (atomic_load_explicit(&man->num_task,  memory_order_relaxed)){
      // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
      // hyper-threads
      pause_or_yield();
    }

  //printf("Main func returnning \n");

  atomic_store(&man->waiting, true);
}

*/

void trigger_all_task_manager(task_manager_t* man)
{
  assert(man != NULL);
  //  FUTEX_WAKE_PRIVATE
  man->futex = 0;
  long r = syscall(SYS_futex, &man->futex, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
  //printf("Number of threads woken %ld \n", r);
  if(r == -1){
    printf("%d errno \n", errno);
  }
  assert(r != -1);
}

/*
void wait_all_task_manager(task_manager_t* man)
{
  assert(man != NULL);

  while (atomic_load_explicit(&man->num_task, memory_order_relaxed)){
    // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
    // hyper-threads
    pause_or_yield();
  }
}
*/


// This function does not belong here logically
void wait_task_status_completed(size_t len, task_status_t* arr)
{
  assert(len > 0);
  assert(arr != NULL);

  // We are believing Fedor
  const struct timespec ns = {0,1};
  int i = 0;
  for(;;){
    int cnt = 0;
    for(int j = len -1; j > -1; --j){
      if(atomic_load(&arr[j].completed) == 1){
        cnt++; 
      } else 
        break;
    }
    if(i == 8){
      i = 0;
      nanosleep(&ns, NULL); 
    }
    if(cnt == len)
      break;
    ++i;
  }
}

#undef pause_or_yield

