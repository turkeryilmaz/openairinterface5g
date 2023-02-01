#include "cells_failed_to_activate.h"

#include <assert.h>
#include <stdlib.h>

void free_cells_failed_to_activate( cells_failed_to_activate_t* src)
{
  // No memory allocated from this object
  (void)src; 
}

bool eq_cells_failed_to_activate(cells_failed_to_activate_t const* m0, cells_failed_to_activate_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;

  // Mandatory
  // NR CGI 9.3.1.12
  if(eq_nr_cgi(&m0->nr_cgi, &m1->nr_cgi) == false)
    return false;

  // Mandatory
  // Cause 9.3.1.2
  if(eq_cause_f1ap(&m0->cause, &m1->cause) == false)
    return false;

  return true;
}

