#!/bin/bash
#export SPGWC0_IP_ADDRESS=`getent hosts $SPGWC0_HOST | awk '{print $1}'`
export SPGWC0_IP_ADDRESS=`getent hosts spgwc | awk '{print $1}'`
echo $SPGWC0_IP_ADDRESS

# Enabling NET_ADMIN and SYS_ADMIN for SPGWU executable
ls -lst /openair-spgwu-tiny/bin/oai_spgwu
getcap /openair-spgwu-tiny/bin/oai_spgwu
echo "changing capabilities on oai_spgwu"
#setcap 'cap_net_admin=ep cap_sys_admin=ep' /openair-spgwu-tiny/bin/oai_spgwu
echo "checking capabilities on oai_spgwu"
ls -lst /openair-spgwu-tiny/bin/oai_spgwu
getcap /openair-spgwu-tiny/bin/oai_spgwu

# Command to run the Python entrypoint script
python3 /openair-spgwu-tiny/bin/entrypoint.py /openair-spgwu-tiny/bin/oai_spgwu -c /openair-spgwu-tiny/etc/spgw_u.conf -o
# If the user does not specify a command at runtime and ENTRYPOINT is defined, the command specified by ENTRYPOINT is executed, with CMD as its arguments.