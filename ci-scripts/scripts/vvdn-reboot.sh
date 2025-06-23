#!/bin/sh

set -e

RU_IP="10.10.0.111"
RU_HOST="root@10.10.0.111"
SETUP_SCRIPT="/home/root/disabled-mplane/setup-ru-4x4-100-no-mplane.sh"
CHECK_CMD="sh -c /etc/scripts/lpru_configuration.sh | grep -q 'CU_PLANE_VLAN[[:space:]]*A0050010[[:space:]]*:[[:space:]]*00000043'"


ssh_exec() {
    ssh $RU_HOST $@ </dev/null
}

echo "→ Checking for PTP fluctuations in the last 60 minutes..."

logs=$(journalctl -u ptp4l.service --since "60 minutes ago")

check_ptp=$(echo "$logs" | awk '
/rms/ {
    rms=$8
    if (rms > 100) {
        print "✗ PTP FLUCTUATION DETECTED → " $0
	exit 0
    }
}
')

if [ -z "$check_ptp" ]; then
    echo "✓ No PTP fluctuations detected."
    if ssh_exec $CHECK_CMD; then
        echo "✓ TX/RX carriers activated."
        echo "→ Skipping RU reboot..."
        exit 0
    else
        echo "✗ TX/RX carriers not activated"
    fi
else
    echo "$check_ptp"
fi

echo "→ Rebooting VVDN LPRU..."
ssh_exec 'reboot'

sleep 60

echo "→ Waiting for $RU_IP SSH port 22 to be open..."
while ! nc -zv $RU_IP 22 >/dev/null 2>&1; do
    echo "SSH port 22 not open, waiting 5 seconds..."
    sleep 5
done
echo "✓ SSH port 22 is open on $RU_IP"

echo "→ Checking if VVDN LPRU is PTP synchronized (timeout 5 min)..."

if ssh_exec 'timeout 5m sh -c '\''tail -F /var/log/synctimingptp2.log | grep -m 1 -F ", synchronized"'\'''; then
    echo "✓ VVDN LPRU synchronized"
else
    echo "✗ VVDN LPRU not synchronized in 5 min"
    sleep infinity
fi
sleep 20

echo "→ Running configuration script $SETUP_SCRIPT on RU..."
ssh_exec $SETUP_SCRIPT >/dev/null

for attempt in 1 2; do
    echo "→ CU_PLANE_VLAN check attempt $attempt..."

    if ssh_exec $CHECK_CMD; then
        echo "✓ TX/RX carriers activated"
        exit 0
    fi

    if [ $attempt -eq 1 ]; then
        echo "✗ TX/RX carriers not activated, re-running configuration script..."
        ssh_exec $SETUP_SCRIPT
        sleep 2
    fi
done

echo "✗ Failed to activate TX/RX carriers"
sleep infinity
