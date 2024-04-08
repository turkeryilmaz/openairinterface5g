#!/bin/bash

echo "sleeping for all the others NFs to be ready"
sleep 3

set -euo pipefail

echo "error point 1"

# we need to resolve
HSS_IP_ADDRESS=`getent hosts hss | awk '{print $1}'`
echo "error point 2"
MME_IP_ADDRESS=`getent hosts mme | awk '{print $1}'`
echo "error point 3"
REDIS_IP_ADDRESS=`getent hosts redis_4g | awk '{print $1}'`
echo "error point 4"

SPGWC_IP_ADDRESS=`getent hosts spgwc | awk '{print $1}'`
echo "error point 5"

set +euo
echo "error point 6"

echo "all NFs should be resolved"
echo $HSS_IP_ADDRESS
echo $MME_IP_ADDRESS
echo $REDIS_IP_ADDRESS
echo $SPGWC_IP_ADDRESS

INSTANCE=1
PREFIX='/magma-mme/etc'
MY_REALM='openairinterface.org'

declare -A MME_CONF

pushd $PREFIX
MME_CONF[@MME_S6A_IP_ADDR@]="$MME_IP_ADDRESS"
MME_CONF[@INSTANCE@]=$INSTANCE
MME_CONF[@PREFIX@]=$PREFIX
MME_CONF[@REALM@]=$MY_REALM
MME_CONF[@MME_FQDN@]="mme.${MME_CONF[@REALM@]}"
MME_CONF[@HSS_HOSTNAME@]='hss'
MME_CONF[@HSS_FQDN@]="${MME_CONF[@HSS_HOSTNAME@]}.${MME_CONF[@REALM@]}"
MME_CONF[@HSS_IP_ADDR@]="$HSS_IP_ADDRESS"
MME_CONF[@SPGWC_IP_ADDR@]="$SPGWC_IP_ADDRESS"
MME_CONF[@REDIS_IP_ADDR@]="$REDIS_IP_ADDRESS"

cp $PREFIX/mme_fd.conf.tmplt $PREFIX/mme_fd.conf
cp $PREFIX/mme.conf.tmplt $PREFIX/mme.conf

for K in "${!MME_CONF[@]}"; do 
  egrep -lRZ "$K" $PREFIX/mme_fd.conf | xargs -0 -l sed -i -e "s|$K|${MME_CONF[$K]}|g"
  ret=$?;[[ ret -ne 0 ]] && echo "Could not replace $K with ${MME_CONF[$K]}"
  egrep -lRZ "$K" $PREFIX/mme.conf | xargs -0 -l sed -i -e "s|$K|${MME_CONF[$K]}|g"
  ret=$?;[[ ret -ne 0 ]] && echo "Could not replace $K with ${MME_CONF[$K]}"
done

sed -i -e "s@etc/freeDiameter@etc@" /magma-mme/etc/mme_fd.conf
sed -i -e "s@bind: 127.0.0.1@bind: $REDIS_IP_ADDRESS@" /etc/magma/redis.yml

cat /magma-mme/etc/mme_fd.conf
cat $PREFIX/mme.conf
cat /etc/magma/redis.yml

# Generate freeDiameter certificate
popd
cd /magma-mme/scripts
./check_mme_s6a_certificate $PREFIX mme.${MME_CONF[@REALM@]}

cd /magma-mme
nohup /magma-mme/bin/sctpd > /var/log/sctpd.log 2>&1 &
sleep 5
/magma-mme/bin/oai_mme -c /magma-mme/etc/mme.conf || true

sleep 5
cat /var/log/mme.log
sleep infinity
