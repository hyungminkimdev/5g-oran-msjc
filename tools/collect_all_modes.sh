#!/bin/bash
# 7개 모드 + baseline KPM 수집 자동화
# Phase 4: 실데이터 재수집 (모드당 120초)
# Constant: gain=0, amp=0.6 (sweet spot)
# 나머지: gain=5, amp=1.0 (강한 재밍)
OUTCSV="${1:-/home/ubuntu/5g-oran-msjc/kpm_fdd_alldata.csv}"
JAMMER_IP="10.111.143.61"
UE_IP="10.111.143.165"
PW="CCI@2025"
FREQ="1842.5e6"
DURATION=120

rm -f "$OUTCSV"

restore_ue() {
  echo "[restore] UE 재시작 중..."
  sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} "sudo pkill -9 srsue" 2>/dev/null
  # gNB도 재시작 (PRACH buffer 문제 방지)
  sudo pkill -9 gnb 2>/dev/null
  sleep 1
  > /tmp/gnb.log
  cd ~/srsRAN_Project/build && sudo taskset -c 0-5 chrt -f 90 ./apps/gnb/gnb -c ~/5g-oran-msjc/srsran/gnb_msjc.yaml > /tmp/gnb.log 2>&1 &
  sleep 5
  sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} \
    "sudo /home/ubuntu/srsRAN_4G/build/srsue/src/srsue /home/ubuntu/5g-oran-msjc/srsran/ue_msjc.conf > /tmp/srsue_safe.log 2>&1 &" 2>/dev/null

  # 최대 3번 시도
  for attempt in 1 2 3; do
    sleep 25
    RESULT=$(sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@${UE_IP} \
      "grep 'PDU Session' /tmp/srsue_safe.log 2>/dev/null" 2>/dev/null)
    if [ -n "$RESULT" ]; then
      echo "[restore] UE attach 성공: $RESULT"
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
  # 마지막 MSJC KPM 라인에서 nof_ues 확인 (PHY 로그가 섞여도 정확)
  NUES=$(grep "MSJC KPM" /tmp/gnb.log | tail -1 | grep -oP 'nof_ues=\K[0-9]+' 2>/dev/null)
  [ "$NUES" = "1" ] && return 0 || return 1
}

run_test() {
  local MODE="$1"
  local LABEL="$2"

  echo ""
  echo "========================================="
  echo "[TEST] mode=$MODE (label=$LABEL)"
  echo "========================================="

  # UE 확인
  if ! check_ue; then
    echo "[WARN] UE 끊김 — 복구 중..."
    restore_ue || return 1
  fi

  if [ "$MODE" != "baseline" ]; then
    # 모드별 gain/amplitude — 실측 검증 완료 파라미터
    case "$MODE" in
      constant)  GAIN=0;  AMP=0.6 ;;  # 광대역 지속, 표준 수치
      random)    GAIN=0;  AMP=0.6 ;;  # ON/OFF 반복, 안정적
      reactive)  GAIN=0;  AMP=0.6 ;;  # 트래픽 반응형, 연결 유지
      deceptive) GAIN=0;  AMP=0.6 ;;  # 프로토콜 교란, 안정적
      pss)       GAIN=10; AMP=1.0 ;;  # 협대역/낮은 duty, 고출력 필요
      pdcch)     GAIN=5;  AMP=1.0 ;;  # CORESET 타겟팅
      dmrs)      GAIN=10; AMP=1.0 ;;  # 참조 신호 타겟팅, 고출력 권장
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
    sleep 3
  fi

  echo "[DONE] $LABEL"
}

# 실행
echo "=== 7개 모드 KPM 수집 시작 (${DURATION}s/mode) ==="
echo "  Constant/Random/Reactive/Deceptive: g0/a0.6 | PSS/DMRS: g10/a1.0 | PDCCH: g5/a1.0"

run_test "baseline" "Normal"
run_test "constant" "Constant"
run_test "random" "Random"
run_test "reactive" "Reactive"
run_test "deceptive" "Deceptive"
run_test "pss" "PSS"
run_test "pdcch" "PDCCH"
run_test "dmrs" "DMRS"

echo ""
echo "========================================="
echo "=== 전체 수집 완료 ==="
echo "========================================="
wc -l "$OUTCSV"
echo "파일: $OUTCSV"
