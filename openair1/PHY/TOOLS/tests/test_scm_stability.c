#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common/utils/LOG/log.h"
#include "common/config/config_userapi.h"
#include "SIMULATION/TOOLS/sim.h"

configmodule_interface_t *uniqCfg = NULL;

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert_not_exit) {
    fprintf(stderr, "Exit function called from %s:%d in %s(). Message: %s\n", file, line, function, s);
    exit(1);
}

int main(int argc, char **argv) {
    logInit();
    randominit(0);

    printf("Starting SCM stability test...\n");

    int num_channels_to_test = 44;
    if (argc > 1) {
        num_channels_to_test = atoi(argv[1]);
        if (num_channels_to_test <= 0) {
            printf("Invalid number of channels specified. Using default: 44\n");
            num_channels_to_test = 44;
        }
    } else {
        printf("Enter number of channels to test: ");
        if (scanf("%d", &num_channels_to_test) != 1 || num_channels_to_test <= 0) {
            printf("Invalid input. Using default: 44\n");
            num_channels_to_test = 44;
        }
    }
    int error_count = 0;
    channel_desc_t **channels = malloc(num_channels_to_test * sizeof(channel_desc_t*));

    for (int i = 0; i < num_channels_to_test; i++) {
        channels[i] = new_channel_desc_scm(
            4, 4, TDL_A, 122.88e6, 3.5e9, 100e6, 30e-9, 0.0,
            CORR_LEVEL_LOW, 0.0, 0, 0.0, -100.0
        );

        if (channels[i] == NULL || channels[i]->channel_length == 0) {
            printf("ERROR: Failed to create channel descriptor on iteration %d.\n", i);
            error_count++;
        }
    }

    for (int i = 0; i < num_channels_to_test; i++) {
        if (channels[i] != NULL) {
            free_channel_desc_scm(channels[i]);
        }
    }
    free(channels);


    if (error_count == 0) {
        printf("SUCCESS: All %d channel descriptors were created and freed without errors.\n", num_channels_to_test);
    } else {
        printf("FAILED: Encountered %d errors during the test.\n", error_count);
    }

    return error_count > 0;
}