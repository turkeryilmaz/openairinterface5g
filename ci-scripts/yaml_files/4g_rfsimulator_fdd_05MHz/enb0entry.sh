#!/bin/bash
echo "Running the custom entry point *******************************************************************************************************************************"
echo "waiting till mme is resolved"

# Function to check if hostname is resolved
check_resolution() {
    local hostname="$1"
    getent hosts "$hostname" >/dev/null 2>&1
}

# Wait for hostname resolution
while ! check_resolution "mme"; do
    echo "Waiting for 'mme' hostname resolution..."
    sleep 1
done


# Resolve hostnames using getent
mme_ip_address_ipv4=$(getent hosts mme | awk '{print $1}')
enb_ipv4_address=$(getent hosts enb0 | awk '{print $1}')


# Resolve placeholders and create new configuration file
sed -e "s/enb0_placeholder/${enb_ipv4_address}/g" \
    -e "s/mme_placeholder/${mme_ip_address_ipv4}/g" \
    /opt/oai-enb/etc/enb.conf.tmpt > /opt/oai-enb/etc/enb.conf


echo "Updated config"
cat /opt/oai-enb/etc/enb.conf
# trying to execute the original entrypoint
exec /tini -v -- /opt/oai-enb/bin/entrypoint.sh /opt/oai-enb/bin/lte-softmodem -O /opt/oai-enb/etc/enb.conf
echo "Slepping to infinity"
sleep infinity
