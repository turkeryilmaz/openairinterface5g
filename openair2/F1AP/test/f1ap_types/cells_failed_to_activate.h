#ifndef F1AP_CELLS_FAILED_TO_BE_ACTIVATED_LIST_H
#define F1AP_CELLS_FAILED_TO_BE_ACTIVATED_LIST_H

#include <stdbool.h>

#include "cause_f1ap.h"
#include "nr_cgi.h"

typedef struct{

  // Mandatory
  // NR CGI 9.3.1.12
  nr_cgi_t nr_cgi;

  // Mandatory
  // Cause 9.3.1.2
  cause_f1ap_t cause;

} cells_failed_to_activate_t;

void free_cells_failed_to_activate(cells_failed_to_activate_t* src);

bool eq_cells_failed_to_activate( cells_failed_to_activate_t const* m0, cells_failed_to_activate_t const* m1);

#endif

