#include <papi.h>
#include <stdio.h>
#include <stdlib.h>

void handle_error (int retval)
{
    printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
    exit(1);
}
	
int main()
{
    int retval;
			
    retval = PAPI_hl_region_begin("computation");
    if ( retval != PAPI_OK )
        handle_error(retval);
		
    /* Do some computation here */
    printf("Some computation\n");
		
    retval = PAPI_hl_region_end("computation");
    if ( retval != PAPI_OK )
        handle_error(retval);

     /* Executes if all low-level PAPI
    function calls returned PAPI_OK */
    printf("\033[0;32mPASSED\n\033[0m");
    exit(0); 
}
