#!/bin/bash
# Phase 6: Closed-Loop MLOps E2E 데모
#
# 전체 사이클:
#   1. xApp 시작 (MockFlexRIC) → InfluxDB 로깅
#   2. Labeler rApp → GMM 레이블링
#   3. Training Manager → 재학습 + A1 알림
#   4. xApp hot-reload 확인
#
# 사용법: bash tools/demo_closedloop.sh

set -e
cd "$(dirname "$0")/.."

XAPP_LOG="/tmp/xapp_closedloop.log"
LABELER_LOG="/tmp/labeler_closedloop.log"
TRAINER_LOG="/tmp/trainer_closedloop.log"

cleanup() {
    echo ""
    echo "=== 정리 ==="
    kill $XAPP_PID 2>/dev/null && echo "xApp 종료" || true
    sleep 1
}
trap cleanup EXIT

echo "=============================================="
echo "  MSJC Closed-Loop MLOps E2E 데모"
echo "=============================================="
echo ""

# ── Step 0: 이전 프로세스 정리 ─────────────────
pkill -f "xapp_msjc" 2>/dev/null || true
sleep 2

# ── Step 1: xApp 시작 ──────────────────────────
echo "[Step 1] xApp 시작 (MockFlexRIC, Stage 1+2)..."
python3 -u xapp_msjc.py --mock-ric --no-stage3 > "$XAPP_LOG" 2>&1 &
XAPP_PID=$!
echo "  PID=$XAPP_PID"
sleep 8

# A1 health 확인
A1_STATUS=$(curl -s http://127.0.0.1:5000/a1/health 2>/dev/null)
echo "  A1 Health: $A1_STATUS"

# xApp 로그 확인
XAPP_LINES=$(wc -l < "$XAPP_LOG")
echo "  xApp 로그: ${XAPP_LINES}줄"
echo ""

# ── Step 2: 데이터 축적 대기 ──────────────────
echo "[Step 2] InfluxDB 데이터 축적 (20초)..."
sleep 20

# 축적된 데이터 확인
TOTAL=$(curl -s -X POST http://127.0.0.1:5000/a1/accuracy_report 2>/dev/null)
echo "  xApp 통계: $TOTAL"
echo ""

# ── Step 3: Labeler rApp ──────────────────────
echo "[Step 3] Labeler rApp (GMM 레이블링)..."
python3 labeler_rapp.py --once > "$LABELER_LOG" 2>&1
cat "$LABELER_LOG"
echo ""

# ── Step 4: Training Manager → 재학습 + A1 ────
echo "[Step 4] Training Manager (강제 재학습 + A1 알림)..."
python3 training_manager_rapp.py --force-retrain > "$TRAINER_LOG" 2>&1
cat "$TRAINER_LOG"
echo ""

# ── Step 5: xApp A1 수신 확인 ─────────────────
echo "[Step 5] xApp A1 MODEL_UPDATE 수신 확인..."
sleep 2
if grep -q "MODEL_UPDATE" "$XAPP_LOG"; then
    echo "  ✅ A1 MODEL_UPDATE 수신 확인!"
    grep "MODEL_UPDATE" "$XAPP_LOG"
else
    echo "  ⚠️ A1 MODEL_UPDATE 미수신 (hot-reload fallback 사용)"
fi
echo ""

# ── Step 6: 결과 보고 ──────────────────────────
echo "=============================================="
echo "  Closed-Loop E2E 결과"
echo "=============================================="

FINAL_STATS=$(curl -s -X POST http://127.0.0.1:5000/a1/accuracy_report 2>/dev/null)
echo "  xApp 최종 통계: $FINAL_STATS"

echo ""
echo "  [사이클 요약]"
echo "  1. xApp → KPM 수신 → Stage 1/2 추론 → InfluxDB 로깅  ✅"

if grep -q "레이블 완료" "$LABELER_LOG"; then
    echo "  2. Labeler rApp → GMM 레이블링 → InfluxDB 저장      ✅"
else
    echo "  2. Labeler rApp → (데이터 부족 또는 오류)              ⚠️"
fi

if grep -q "재학습 완료" "$TRAINER_LOG"; then
    echo "  3. Training Manager → 재학습 완료                     ✅"
else
    echo "  3. Training Manager → (오류)                          ⚠️"
fi

if grep -q "MODEL_UPDATE" "$XAPP_LOG"; then
    echo "  4. A1 → xApp → hot-reload                           ✅"
else
    echo "  4. A1 미연결 → mtime hot-reload fallback             ⚠️"
fi

echo ""
echo "  Grafana: http://localhost:3000/d/msjc-main"
echo "=============================================="
