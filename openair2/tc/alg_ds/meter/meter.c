#include "meter.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>

#include "../alg/defer.h"
#include "../alg/accumulate.h"
#include "../alg/lower_bound.h"

/*
static
int64_t time_now_us()
{
  struct timespec tms;

  // The C11 way 
  // if (! timespec_get(&tms, TIME_UTC))  

  // POSIX.1-2008 way 
  if (clock_gettime(CLOCK_REALTIME,&tms)) {
    return -1;
  }
  // seconds, multiplied with 1 million 
  int64_t micros = tms.tv_sec * 1000000;
  // Add full microseconds 
  micros += tms.tv_nsec/1000;
  // round up if necessary 
  if (tms.tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return micros;
}

static inline
bool meter_invariants(mtr_t* m)
{
  assert(m != NULL);
  assert(seq_size(&m->tstamps) == seq_size(&m->val));
  assert(!(m->avg < 0.0));
  assert(m->window_ms > 0.0);

  return true;
}
*/

void mtr_init(mtr_t* m, float window_ms)
{
  assert(m != NULL);
  assert(window_ms > 0);

  mv_avg_wnd_init(&m->avg_wnd, window_ms);

/*
  defer({ assert(meter_invariants(m)); } );

  seq_init(&m->tstamps, sizeof(int64_t));
  seq_init(&m->val, sizeof(uint32_t));

  m->avg = 0.0;
  m->window_ms = window_ms;
*/
}

void mtr_free(mtr_t* m)
{
  assert(m != NULL);

  mv_avg_wnd_free(&m->avg_wnd);

/*
  assert(meter_invariants(m) == true);

  void* value_semantic = NULL;
  seq_free(&m->tstamps, value_semantic);
  seq_free(&m->val, value_semantic);
*/
}

void mtr_push_back(mtr_t* m, int64_t tstamp, uint32_t val)
{
  assert(m != NULL);
  assert(!(tstamp < 0));
  assert(val != 0);

  
  mv_avg_wnd_push_back(&m->avg_wnd, tstamp, val);

/*
  assert(meter_invariants(m) == true);
  defer({ assert(meter_invariants(m)); } );

  if(tstamp == 0)
    tstamp = time_now_us();

  const size_t sz = seq_size(&m->tstamps);
  void* l = seq_at(&m->tstamps, sz -1);

  assert(tstamp >= *(int64_t*)l && "Time is a monotonically increasing function");

  size_t const elm = seq_size(&m->tstamps);
  seq_push_back(&m->tstamps, (uint8_t*)&tstamp, sizeof(tstamp));
  seq_push_back(&m->val, (uint8_t*)&val, sizeof(val));

  m->avg = m->avg*elm/(elm+1) + (float)val/(elm+1);  
  */
}

/*
static inline
bool cmp_int64(void* val, void* it)
{
  int64_t* a = (int64_t*)val;
  int64_t* b = (int64_t*)it;

  return *a > *b;
}
*/

float mtr_bndwdth_kbps(mtr_t* m)
{
  assert(m != NULL);


  double val = mv_avg_wnd_val(&m->avg_wnd);
  size_t const sz = seq_size(&m->avg_wnd.tstamps); 
  return (val*sz / m->avg_wnd.wnd_ms);

/*

  assert(meter_invariants(m) == true);
  defer({assert(meter_invariants(m)); }); 

  size_t const elm = seq_size(&m->tstamps);
  if(elm == 0) 
    return 0.0;

  int64_t const now_us = time_now_us();

  void* f_ts = seq_front(&m->tstamps);
  void* l_ts = seq_end(&m->tstamps); 

  int64_t const limit = now_us - m->window_ms*1000;
  void* it = lower_bound_ring(&m->tstamps, f_ts, l_ts, (void*)&limit, cmp_int64);

  uint32_t n = 0;
  if(it != f_ts){
    n = seq_distance(&m->tstamps, f_ts, it);
    int64_t const last_tstamp = *(int64_t*)seq_at(&m->tstamps, n -1);
    assert(last_tstamp < limit);
  }
  assert(n <= elm);
  //printf("num of n = %d \n", n);

  seq_erase(&m->tstamps, f_ts, it);

  void* f_val = seq_front(&m->val);
  void* l_val = seq_at(&m->val, n); 

  uint64_t const acc = accumulate_ring(&m->val,f_val, l_val);
  //printf("accumulated number of bytes = %lu \n", acc);

  seq_erase(&m->val, f_val, l_val);

  if(n < elm)
    m->avg = (m->avg*elm - acc)/(elm-n);
  else // n == elm
    m->avg = 0.0;

  size_t const sz = seq_size(&m->tstamps); 
  return (m->avg*sz / m->window_ms) ;
  */
}

