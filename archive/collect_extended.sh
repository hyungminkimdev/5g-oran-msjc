#!/bin/bash
# 확장 데이터 수집 — 모드당 10분 (600초)
# 기존 kpm_fdd_alldata.csv는 보존, 새 파일에 수집 후 병합
#
# 사용법: bash tools/collect_extended.sh [output_csv] [duration_seconds]

OUTCSV="${1:-/home/ubuntu/5g-oran-msjc/kpm_fdd_extended.csv}"
DURATION="${2:-600}"
JAMMER_IP="10.111.143.61"
UE_IP="10.111.143.165"
PW="CCI@2025"
FREQ="1842.5e6"

echo "=== 확장 데이터 수집 (모드당 ${DURATION}s) ==="
echo "  출력: $OUTCSV"
echo "  예상 시간: $((DURATION * 8 / 60 + 10))분"
echo ""

# 기존 파일 있으면 백업
if [ -f "$OUTCSV" ]; then
  cp "$OUTCSV" "${OUTCSV}.bak.$(date +%s)"
fi

rm -f "$OUTCSV"

restore_ue() {
  echo "[restore] UE 재시작 중..."
  sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} "sudo pkill -9 srsue" 2>/dev/null
  sudo pkill -9 gnb 2>/dev/null
  sleep 1
  > /tmp/gnb.log
  cd ~/srsRAN_Project/build && sudo taskset -c 0-5 chrt -f 90 ./apps/gnb/gnb -c ~/5g-oran-msjc/srsran/gnb_msjc.yaml > /tmp/gnb.log 2>&1 &
  sleep 5
  sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} \
    "sudo /home/ubuntu/srsRAN_4G/build/srsue/src/srsue /home/ubuntu/5g-oran-msjc/srsran/ue_msjc.conf > /tmp/srsue_safe.log 2>&1 &" 2>/dev/null

  for attempt in 1 2 3; do
    sleep 25
    RESULT=$(sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} \
      "grep 'PDU Session' /tmp/srsue_safe.log 2>/dev/null" 2>/dev/null)
    if [ -n "$RESULT" ]; then
      echo "[restore] UE attach 성공 (attempt $attempt)"
      sleep 3
      return 0
    fi
    echo "[restore] 시도 $attempt 실패, 재시작..."
    sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} \
      "sudo pkill -9 srsue; sleep 5; sudo /home/ubuntu/srsRAN_4G/build/srsue/src/srsue /home/ubuntu/5g-oran-msjc/srsran/ue_msjc.conf > /tmp/srsue_safe.log 2>&1 &" 2>/dev/null
  done
  echo "[restore] UE 복구 실패!"
  return 1
}

check_ue() {
  NUES=$(grep "MSJC KPM" /tmp/gnb.log | tail -1 | grep -oP 'nof_ues=\K[0-9]+' 2>/dev/null)
  [ "$NUES" = "1" ] && return 0 || return 1
}

run_test() {
  local MODE="$1"
  local LABEL="$2"

  echo ""
  echo "========================================="
  echo "[$(date +%H:%M:%S)] mode=$MODE (label=$LABEL) — ${DURATION}s"
  echo "========================================="

  # UE 확인
  if ! check_ue; then
    echo "[WARN] UE 끊김 — 복구 중..."
    restore_ue || return 1
  fi

  if [ "$MODE" != "baseline" ]; then
    case "$MODE" in
      constant)  GAIN=0;  AMP=0.5 ;;  # amp=0.5 안정 (장시간 UE 유지)
      random)    GAIN=0;  AMP=0.6 ;;
      reactive)  GAIN=0;  AMP=0.6 ;;
      deceptive) GAIN=0;  AMP=0.6 ;;
      pss)       GAIN=10; AMP=1.0 ;;
      pdcch)     GAIN=5;  AMP=1.0 ;;
      dmrs)      GAIN=10; AMP=1.0 ;;
      *)         GAIN=0;  AMP=0.6 ;;
    esac
    # Jammer 시작
    sshpass -p "$PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@${JAMMER_IP} \
      "echo '$PW' | sudo -S nohup python3 ~/5g-oran-msjc/tools/jammer.py --mode $MODE --freq $FREQ --gain $GAIN --amplitude $AMP > /tmp/jammer.log 2>&1 &
       sleep 4; echo 'Jammer $MODE started (gain=$GAIN, amp=$AMP)'" 2>&1 | tail -1
    sleep 2
  fi

  # KPM 수집
  cd ~/5g-oran-msjc
  bash tools/gnb_kpm_parser.sh "$LABEL" "$DURATION" "$OUTCSV" 2>&1

  if [ "$MODE" != "baseline" ]; then
    # Jammer 종료
    sshpass -p "$PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@${JAMMER_IP} \
      "echo '$PW' | sudo -S killall -9 python3 2>/dev/null; echo 'Jammer killed'" 2>&1 | tail -1
    sleep 5

    # UE 생존 확인 — 강한 재밍 후 UE 탈락 가능
    if ! check_ue; then
      echo "[WARN] 재밍 후 UE 탈락 — 복구 진행"
      restore_ue
    fi
  fi

  SAMPLES=$(grep -c "$LABEL" "$OUTCSV" 2>/dev/null || echo 0)
  echo "[DONE] $LABEL — ${SAMPLES}개 수집"
}

# 실행
echo "=== gNB + UE 상태 확인 ==="
if ! check_ue; then
  echo "[INFO] UE 미연결 — 스택 시작 필요"
  restore_ue || { echo "[FATAL] 스택 시작 실패"; exit 1; }
fi
echo "[OK] UE 연결 확인됨"
echo ""

run_test "baseline" "Normal"
run_test "constant" "Constant"
run_test "random" "Random"
run_test "reactive" "Reactive"
run_test "deceptive" "Deceptive"
run_test "pss" "PSS"
run_test "pdcch" "PDCCH"
run_test "dmrs" "DMRS"

# 두 번째 Normal (다른 시간대)
echo ""
echo "[INFO] 추가 Normal baseline (session 2)"
run_test "baseline" "Normal"

echo ""
echo "========================================="
echo "=== 확장 수집 완료 ==="
echo "========================================="
wc -l "$OUTCSV"
echo "파일: $OUTCSV"

# RF 프로세스 정리
echo ""
echo "[CLEANUP] RF 프로세스 정리..."
sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${JAMMER_IP} \
  "echo '$PW' | sudo -S killall -9 python3 2>/dev/null" 2>/dev/null
echo "[DONE] 정리 완료"
