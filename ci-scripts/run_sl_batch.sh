#!/bin/bash

remote_host_ip=10.1.1.61
user_id=$USER

message_to_send=EpiScience
num_repeat=1

atten_list=(21) #(0 5 10 15 20 25 30 35 39 40)

for i in ${!atten_list[@]}; do
    python3 run_sl_test.py -m $message_to_send -r $num_repeat \
                           -u $user_id --host $remote_host_ip \
                           --att ${atten_list[$i]} \
                           --save test_att_${atten_list[$i]}.txt
done

