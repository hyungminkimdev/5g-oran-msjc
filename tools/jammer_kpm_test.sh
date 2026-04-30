#!/bin/bash
# Jammer Constant 모드 — gain별 KPM 수집 자동화
# Usage: ./jammer_kpm_test.sh <gain> <duration_sec> <output_csv>

GAIN="${1:?Usage: $0 <gain> <duration_sec> <output_csv>}"
DURATION="${2:-10}"
OUTCSV="${3:-/tmp/kpm_jammer_test.csv}"
JAMMER_IP="10.111.143.61"
JAMMER_PW="CCI@2025"
FREQ="1842.5e6"  # gNB DL frequency (Band 3)

echo "=== Jammer Constant Test: gain=${GAIN} dB, duration=${DURATION}s ==="

# 1. Start jammer on Instance-3
echo "[1/4] Starting jammer (gain=${GAIN})..."
sshpass -p "$JAMMER_PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@${JAMMER_IP} \
  "echo '$JAMMER_PW' | sudo -S nohup python3 ~/5g-oran-msjc/tools/jammer.py --mode constant --freq ${FREQ} --gain ${GAIN} > /tmp/jammer.log 2>&1 &
   sleep 3; echo 'Jammer started'; ps aux | grep jammer.py | grep -v grep | wc -l" 2>&1

# 2. Wait for jammer to stabilize
echo "[2/4] Waiting 3s for jammer stabilization..."
sleep 3

# 3. Collect KPM
echo "[3/4] Collecting KPM for ${DURATION}s (label=constant_gain${GAIN})..."
cd ~/5g-oran-msjc && sudo taskset -c 6,7 python3 tools/kpm_collector.py \
  "constant_gain${GAIN}" "${DURATION}" "${OUTCSV}" 2>&1 | grep -E "Collecting|Done|\[constant"

# 4. Kill jammer
echo "[4/4] Killing jammer..."
sshpass -p "$JAMMER_PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@${JAMMER_IP} \
  "echo '$JAMMER_PW' | sudo -S killall -9 python3 2>/dev/null; echo 'Jammer killed'" 2>&1

echo "=== Done: gain=${GAIN} ==="
