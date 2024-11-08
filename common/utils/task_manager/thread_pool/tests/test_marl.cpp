#include "marl/defer.h"
#include "marl/event.h"
#include "marl/scheduler.h"
#include "marl/waitgroup.h"

#include <cstdio>

#define NUM_THREADS 8
#define NUM_JOBS 1024 * 100

int64_t time_now_us(void)
{
  struct timespec tms;

  if (clock_gettime(CLOCK_MONOTONIC_RAW, &tms)) {
    return -1;
  }
  /* seconds, multiplied with 1 million */
  int64_t micros = tms.tv_sec * 1000000;
  /* Add full microseconds */
  int64_t const tv_nsec = tms.tv_nsec;
  micros += tv_nsec / 1000;
  /* round up if necessary */
  if (tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return micros;
}

static inline int64_t naive_fibonnacci(int64_t a)
{
  if (a < 2)
    return a;

  return naive_fibonnacci(a - 1) + naive_fibonnacci(a - 2);
}

int main() {
  // Create a marl scheduler using all the logical processors available to the process.
  // Bind this scheduler to the main thread so we can call marl::schedule()
  marl::Scheduler scheduler(marl::Scheduler::Config::allCores().setWorkerThreadCount(NUM_THREADS));
  scheduler.bind();
  defer(scheduler.unbind());  // Automatically unbind before returning.

  constexpr int numTasks = NUM_JOBS;

  // Create an event that is manually reset.
  marl::Event sayHello(marl::Event::Mode::Manual);

  // Create a WaitGroup with an initial count of numTasks.
  marl::WaitGroup saidHello(numTasks);

  int64_t now = time_now_us();
  // Schedule some tasks to run asynchronously.
  for (int i = 0; i < numTasks; i++) {
    // Each task will run on one of the 4 worker threads.
    marl::schedule([=] {  // All marl primitives are capture-by-value.
      // Decrement the WaitGroup counter when the task has finished.
      defer(saidHello.done());
      naive_fibonnacci(23);
    });
  }
  printf("Waiting %ld \n", time_now_us());


  saidHello.wait();  // Wait for all tasks to complete.
  int64_t end = time_now_us();
  printf("Done %ld \n", end);
  printf("Total elapsed %ld \n", end - now);

  printf("All tasks said hello.\n");
}
