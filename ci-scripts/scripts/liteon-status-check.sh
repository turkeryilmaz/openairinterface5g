#!/bin/sh

set -e

RU_IP="172.21.18.101"
RU_HOST="root@172.21.18.101"

# Expected values
EXPECTED_SYNC="SYNCHRONIZED"
EXPECTED_RFINIT="Ready"
EXPECTED_DPD="Ready"

ssh_exec() {
    ssh $RU_HOST $@ </dev/null
}

check_status() {
  local sync rfinit dpd
  sync=$(ssh_exec "uci get oru_state.status.Sync")
  rfinit=$(ssh_exec "uci get oru_state.status.RFinit")
  dpd=$(ssh_exec "uci get oru_state.status.DPD")
  echo "$sync" "$rfinit" "$dpd"
}

wait_for_ssh() {
  echo "→ Waiting for $RU_IP SSH port 22 to be open..."
  until ssh_exec "echo OK" &>/dev/null; do
    sleep 5
  done
  echo "✓ SSH port 22 is open on $RU_IP"
}

echo "→ Checking LiteON RU status..."
read -r sync_val rfinit_val dpd_val <<< "$(check_status)"
echo "Sync State  : $sync_val
RF State    : $rfinit_val
DPD         : $dpd_val"

if [[ "$sync_val" == "$EXPECTED_SYNC" ]] &&
   [[ "$rfinit_val" == "$EXPECTED_RFINIT" ]] &&
   [[ "$dpd_val" == "$EXPECTED_DPD" ]]; then
  echo "✓ LiteON RU Status OK."
  echo "→ Skipping RU reboot..."
else
  echo "✗ LiteON RU status NOK."
  echo "→ Rebooting LiteON RU..."
  ssh_exec "reboot"
  sleep 60
  wait_for_ssh
  echo "→ Waiting for LiteON RU to reach expected status..."
  sleep 60
  retries=20
  for ((i=1; i<=retries; i++)); do
    read -r sync_val rfinit_val dpd_val <<< "$(check_status)"
    echo "[$i/$retries] Sync State : $sync_val, RF State : $rfinit_val, DPD : $dpd_val"
    if [[ "$sync_val" == "$EXPECTED_SYNC" ]] &&
       [[ "$rfinit_val" == "$EXPECTED_RFINIT" ]] &&
       [[ "$dpd_val" == "$EXPECTED_DPD" ]]; then
      echo "✓ LiteON RU has reached expected status."
      echo "Sync State  : $sync_val
RF State    : $rfinit_val
DPD         : $dpd_val"
      exit 0
    fi
    sleep 10
  done
  # if we reach here, timeout happened
  echo "✗ Timeout reached (5 min). LiteON RU did not reach expected status."
  exit 1
fi
