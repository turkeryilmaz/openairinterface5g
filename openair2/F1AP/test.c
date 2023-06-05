/* compile with:
 * gcc -o test -Wall test.c f1ap_ids.c  -I ../../ -I ../../openair2/COMMON/ ../../common/utils/hashtable/hashtable.c -I ../../common/utils/ -fsanitize=address
 */

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include "f1ap_ids.h"

void exit_function(void) {}

int main()
{
  du_init_f1_ue_data();
  int rnti = 13;
  f1_ue_data_t data = {.secondary_ue = 1};
  bool ret = du_add_f1_ue_data(rnti, &data);
  assert(ret);
  ret = du_add_f1_ue_data(rnti, &data);
  assert(!ret);
  const f1_ue_data_t *rdata = du_get_f1_ue_data(rnti);
  assert(rdata != NULL);
  assert(rdata->secondary_ue == data.secondary_ue);
}
