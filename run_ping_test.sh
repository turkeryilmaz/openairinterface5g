#!/bin/bash
set -e
trap finalize SIGABRT SIGINT ERR
divider="============================="

section()
{
    echo ${divider}
    echo $1
    echo ${divider}
}

finalize()
{
    set +e
    echo ${divider}
    echo "Script exited unexpectedly"
    echo ${divider}
    docker logs local-oai-gnb >& ${directory}/gnb.log
    docker logs local-oai-nr-ue >& ${directory}/ue.log
    docker compose -f ping_docker_compose.yml down
    echo "Test result saves in ${directory}"
    exit 1
}
date=date=$(date '+%Y-%m-%d')
directory="logs/"$(date '+%Y-%m-%d_%H-%M-%S')"/"
mkdir -p ${directory}
ln -s "$(pwd)/${directory}" logs/latest
docker compose -f ping_docker_compose.yml up --wait

section "running ping test from ue1 to gnb"

docker exec local-oai-nr-ue bash -c "ping -I oaitun_ue1 10.0.1.1 -c 10" | tee ${directory}/ping_ue_gnb_1.log

section "running ping test from ue2 to gnb"

docker exec local-oai-nr-ue bash -c "ping -I oaitun_ue2 10.0.1.1 -c 10" | tee ${directory}/ping_ue_gnb_2.log

section "running ping test from gnb to ue1"

docker exec local-oai-gnb bash -c "ping -I oaitun_enb1 10.0.1.2 -c 10" | tee ${directory}/ping_gnb_ue_1.log

section "running ping test from gnb to ue2"

docker exec local-oai-gnb bash -c "ping -I oaitun_enb1 10.0.1.3 -c 10" | tee ${directory}/ping_gnb_ue_2.log

docker logs local-oai-gnb >& ${directory}/gnb.log
docker logs local-oai-nr-ue >& ${directory}/ue.log
docker compose -f ping_docker_compose.yml down

section "error summary: UE. Warnings & errors: "$(cat ${directory}/ue.log  | grep "\] [WE] " | wc -l)
cat ${directory}/ue.log  | grep "\] [WE] "

section "error summary: gNB. Warnings & errors: "$(cat ${directory}/gnb.log  | grep "\] [WE] " | wc -l)
cat ${directory}/gnb.log  | grep "\] [WE] "


exit 0
