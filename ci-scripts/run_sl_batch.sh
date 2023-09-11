#!/bin/bash

remote_host_ip=10.1.1.80 # to test on N310
user_id=$USER

message_to_send=EpiScience
num_repeat=1

# change test type accordingly
test_type='rfsim'
test_type_cmd="--test "$test_type

if [ $test_type != 'rfsim' ]; then
    atten_list=(0 5 10 15 20 25 30 35 39 40)
    test_params=("${atten_list[@]}")
    param_name='att'
else
    param_name='snr'
fi

start_mcs=0
end_mcs=9
step_mcs=1
mcs_list=()

for i in $(seq $start_mcs $step_mcs $end_mcs); do
    mcs_list+=($i)
done

step_snr=0.2
start_snr=-10

TEST_FOLDER=$HOME/test_$test_type
[ -d $TEST_FOLDER ] || mkdir $TEST_FOLDER

for j in ${!mcs_list[@]}; do
    if [ $test_type = 'rfsim' ]; then
        snr_list=()
        if [ ${mcs_list[$j]} -ge 0 ] && [ ${mcs_list[$j]} -le 5 ]; then
            end_snr=6
        elif [ ${mcs_list[$j]} -ge 6 ] && [ ${mcs_list[$j]} -le 8 ]; then
            end_snr=10
        else
            start_snr=-6
            end_snr=14
        fi
        for k in $(seq $start_snr $step_snr $end_snr); do
            snr_list+=($k)
        done
        test_params=("${snr_list[@]}")
    fi
    for k in ${!test_params[@]}; do
        if [ $test_type = 'rfsim' ]; then
            param_cmd='--snr '${test_params[$k]}
        else
            param_cmd='--att '${test_params[$k]}
        fi
            python3 run_sl_test.py --repeat $num_repeat \
                    -u $user_id --host $remote_host_ip \
                    --mcs ${mcs_list[$j]} \
                    $param_cmd \
                    --save $TEST_FOLDER/mcs_${mcs_list[$j]}_${param_name}_${test_params[$k]}.txt \
                    $test_type_cmd \
                    --duration 20
    done
done