#!/bin/bash

remote_host_ip=10.1.1.80 # to test on N310
user_id=$USER

message_to_send=EpiScience
num_repeat=1

atten_list=(0) #(0 5 10 15 20 25 30 35 39 40)
mcs_list=(9)
repeat=1

TEST_FOLDER=$HOME/test_usrp
[ -d $TEST_FOLDER ] || mkdir $TEST_FOLDER
for i in ${!atten_list[@]}; do
    for j in ${!mcs_list[@]}; do
        python3 run_sl_test.py --repeat $repeat \
                            -u $user_id --host $remote_host_ip \
                            --att ${atten_list[$i]} \
                            --mcs ${mcs_list[$j]} \
                            --save $TEST_FOLDER/att_${atten_list[$i]}_mcs_${mcs_list[$j]}.txt \
                            # -m $message_to_send
    done
done

