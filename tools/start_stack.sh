#!/bin/bash
# Full stack startup + jammer sweep automation
# Run from Instance-2: bash ~/5g-oran-msjc/tools/start_stack.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

UE_IP="10.111.143.165"
JAMMER_IP="10.111.143.61"

echo "=== [1/6] Checking Open5GS ==="
sudo systemctl is-active open5gs-amfd open5gs-smfd open5gs-upfd open5gs-nrfd
echo "5GC OK"

echo ""
echo "=== [2/6] Starting FlexRIC nearRT-RIC ==="
kill $(pgrep nearRT-RIC) 2>/dev/null || true
sleep 1
cd /home/ubuntu/flexric/build/examples/ric && taskset -c 6-7 nohup ./nearRT-RIC > /tmp/ric.log 2>&1 &
sleep 2
echo "RIC PID: $(pgrep nearRT-RIC)"

echo ""
echo "=== [3/6] Starting gNB ==="
sudo pkill -9 gnb 2>/dev/null || true
sleep 1
cd /home/ubuntu/srsRAN_Project/build && sudo taskset -c 0-3 chrt -f 50 ./apps/gnb/gnb -c "$REPO_DIR/srsran/gnb_msjc.yaml" > /tmp/gnb.log 2>&1 &
# Wait for E2 setup
for i in $(seq 1 15); do
    sleep 1
    if grep -q "E2 Setup procedure successful" /tmp/gnb.log 2>/dev/null; then
        echo "gNB ready (E2 connected) at +${i}s"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "WARNING: gNB E2 not connected after 15s"
        tail -5 /tmp/gnb.log
    fi
done

echo ""
echo "=== [4/6] Starting UE (retry loop) ==="
MAX_ATTEMPTS=20
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$UE_IP \
        'sudo pkill -9 srsue 2>/dev/null; sudo rm -f /tmp/srsue_usrp_fdd.log; sleep 1; sudo /home/ubuntu/srsRAN_4G/build/srsue/src/srsue /home/ubuntu/5g-oran-msjc/srsran/ue_msjc.conf > /tmp/srsue_stdout.log 2>&1 &' 2>/dev/null

    sleep 15

    result=$(sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$UE_IP \
        'sudo grep "PDU Session Establishment successful" /tmp/srsue_usrp_fdd.log 2>/dev/null' 2>/dev/null)

    if [ -n "$result" ]; then
        echo "UE ATTACHED on attempt $attempt!"
        echo "$result"
        break
    fi

    status=$(sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$UE_IP \
        'sudo grep -oE "snr=[+0-9.-]+|SIB1 received|Random Access Complete|Couldn.t select|Completed with failure" /tmp/srsue_usrp_fdd.log 2>/dev/null | head -3' 2>/dev/null)
    echo "  Attempt $attempt/$MAX_ATTEMPTS: $status"

    if [ $attempt -eq $MAX_ATTEMPTS ]; then
        echo "FAILED: UE could not attach after $MAX_ATTEMPTS attempts"
        exit 1
    fi
done

echo ""
echo "=== [5/6] Starting keepalive traffic ==="
# Start ping to keep UE connected
sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$UE_IP \
    'UE_IP=$(sudo grep "PDU Session Establishment successful" /tmp/srsue_usrp_fdd.log | grep -oE "10\.[0-9.]+"); nohup sudo ping -I tun_srsue -i 0.5 10.45.0.1 > /tmp/ping_keepalive.log 2>&1 &; echo "Keepalive ping started, UE IP=$UE_IP"' 2>/dev/null

# Also start iperf3 server locally
sudo pkill iperf3 2>/dev/null || true
iperf3 -s -B 10.45.0.1 -D 2>/dev/null || true
sleep 2

# Verify ping works
sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$UE_IP \
    'sudo ping -c 3 -I tun_srsue 10.45.0.1 2>&1 | tail -3' 2>/dev/null

echo ""
echo "=== [6/6] Collecting Normal baseline (30s) ==="
rm -f /tmp/kpm_data.csv
cp "$REPO_DIR/srsran/xapp_kpm.conf" /tmp/xapp_kpm_hello.conf
python3 "$REPO_DIR/tools/kpm_collector.py" Normal 30 /tmp/kpm_data.csv 2>&1 | grep -E "\[Normal\]|Done|E2 nodes"

echo ""
echo "=== Stack ready! ==="
echo "Normal baseline saved to /tmp/kpm_data.csv"
echo "Next: run ~/5g-oran-msjc/tools/jammer_sweep.sh for jammer sweet spot search"
