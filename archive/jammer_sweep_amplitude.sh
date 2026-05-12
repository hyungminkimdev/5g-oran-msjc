#!/bin/bash
# Jammer amplitude sweep — gain=0 고정, amplitude 변화
# UE가 끊기면 자동 재시작
OUTCSV="/tmp/kpm_jammer_sweep.csv"
JAMMER_IP="10.111.143.61"
UE_IP="10.111.143.165"
PW="CCI@2025"
FREQ="1842.5e6"
DURATION=10

rm -f "$OUTCSV"

# Baseline 수집
echo "========================================="
echo "[BASELINE] No jammer, ${DURATION}s"
echo "========================================="
cd ~/5g-oran-msjc
sudo taskset -c 6,7 python3 tools/kpm_collector.py "baseline" "$DURATION" "$OUTCSV" 2>&1 | grep -v "Loading\|Opening\|LibConf\|xApp\|NEAR-RIC\|E2-AGENT\|generating"
echo ""

# Amplitude 레벨별 테스트
for AMP in 0.01 0.05 0.1 0.3 0.5 1.0; do
  echo "========================================="
  echo "[TEST] amplitude=${AMP}, gain=0, ${DURATION}s"
  echo "========================================="

  # UE 연결 확인
  PING_OK=$(sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} "sudo ping -I tun_srsue 10.45.0.1 -c 1 -W 2 2>/dev/null | grep -c '1 received'" 2>/dev/null)
  if [ "$PING_OK" != "1" ]; then
    echo "[WARN] UE disconnected, restarting..."
    sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} "sudo pkill -9 srsue; sleep 5; sudo /home/ubuntu/srsRAN_4G/build/srsue/src/srsue /home/ubuntu/5g-oran-msjc/srsran/ue_msjc.conf > /tmp/srsue_safe.log 2>&1 & sleep 30; grep 'PDU Session' /tmp/srsue_safe.log" 2>/dev/null
    sleep 3
  fi

  # Start jammer
  sshpass -p "$PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@${JAMMER_IP} \
    "echo '$PW' | sudo -S nohup python3 ~/5g-oran-msjc/tools/jammer.py --mode constant --freq ${FREQ} --gain 0 --amplitude ${AMP} > /tmp/jammer.log 2>&1 &
     sleep 4; echo 'Jammer started (amp=${AMP})'" 2>&1 | tail -1

  sleep 2

  # Collect KPM
  sudo taskset -c 6,7 python3 tools/kpm_collector.py "constant_amp${AMP}" "$DURATION" "$OUTCSV" 2>&1 | grep -v "Loading\|Opening\|LibConf\|xApp\|NEAR-RIC\|E2-AGENT\|generating"

  # Kill jammer
  sshpass -p "$PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@${JAMMER_IP} \
    "echo '$PW' | sudo -S killall -9 python3 2>/dev/null; echo 'Jammer killed'" 2>&1 | tail -1

  sleep 3
  echo ""
done

echo "========================================="
echo "All tests complete → $OUTCSV"
echo "========================================="
wc -l "$OUTCSV"
