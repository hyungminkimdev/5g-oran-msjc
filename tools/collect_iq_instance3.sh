#!/bin/bash
# Instance-3 Full-Duplex I/Q 캡처 자동화
# X310 Radio#0 TX(재밍) + Radio#1 RX(캡처) 동시 수행
# gNB 실행 중이면 gNB DL + Jammer 혼합 캡처, 미실행이면 Jammer만 캡처
#
# 사용법: bash tools/collect_iq_instance3.sh [n_snapshots]
#   기본 50 snapshots/mode × 8 modes = 400개

set -e

N_SNAPSHOTS=${1:-50}
JAMMER_IP="10.111.143.61"
PW="CCI@2025"
REMOTE_OUTDIR="/tmp/iq_captures"
LOCAL_OUTDIR="/home/ubuntu/5g-oran-msjc/iq_data"

echo "=== Instance-3 Full-Duplex I/Q 캡처 ==="
echo "  Snapshots/mode: $N_SNAPSHOTS"
echo ""

# 최신 스크립트 Instance-3에 전송
echo "[*] 스크립트 전송..."
sshpass -p "$PW" scp -o StrictHostKeyChecking=no \
    ~/5g-oran-msjc/tools/iq_capture_fullduplex.py \
    ~/5g-oran-msjc/tools/jammer.py \
    ubuntu@$JAMMER_IP:~/5g-oran-msjc/tools/

# config.yaml도 전송 (jammer.py import 시 필요할 수 있음)
sshpass -p "$PW" scp -o StrictHostKeyChecking=no \
    ~/5g-oran-msjc/tools/config.yaml \
    ubuntu@$JAMMER_IP:~/5g-oran-msjc/tools/ 2>/dev/null || true

# 기존 캡처 삭제
sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@$JAMMER_IP \
    "rm -rf $REMOTE_OUTDIR; mkdir -p $REMOTE_OUTDIR" 2>/dev/null

# 모드별 설정 (CLAUDE.md 8.1 기준)
declare -a MODES=("Normal" "Constant" "Random" "Reactive" "Deceptive" "PSS" "PDCCH" "DMRS")
declare -A TX_GAIN AMP
TX_GAIN[Normal]=0;     AMP[Normal]=0
TX_GAIN[Constant]=0;   AMP[Constant]=0.6
TX_GAIN[Random]=0;     AMP[Random]=0.6
TX_GAIN[Reactive]=0;   AMP[Reactive]=0.6
TX_GAIN[Deceptive]=0;  AMP[Deceptive]=0.6
TX_GAIN[PSS]=10;       AMP[PSS]=1.0
TX_GAIN[PDCCH]=5;      AMP[PDCCH]=1.0
TX_GAIN[DMRS]=10;      AMP[DMRS]=1.0

for MODE in "${MODES[@]}"; do
    echo ""
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] $MODE (tx_gain=${TX_GAIN[$MODE]}, amp=${AMP[$MODE]})"
    echo "=========================================="

    sshpass -p "$PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@$JAMMER_IP \
        "echo '$PW' | sudo -S python3 ~/5g-oran-msjc/tools/iq_capture_fullduplex.py \
         --mode $MODE --n-snapshots $N_SNAPSHOTS \
         --tx-gain ${TX_GAIN[$MODE]} --amplitude ${AMP[$MODE]} \
         --outdir $REMOTE_OUTDIR" 2>&1

    echo "[$(date +%H:%M:%S)] $MODE 완료"
    sleep 2
done

# Instance-3 → Instance-2 전송
echo ""
echo "[*] I/Q 파일 전송..."
mkdir -p "$LOCAL_OUTDIR"
sshpass -p "$PW" scp -r -o StrictHostKeyChecking=no \
    "ubuntu@$JAMMER_IP:$REMOTE_OUTDIR/*" "$LOCAL_OUTDIR/"

echo ""
echo "=== I/Q 캡처 완료 ==="
TOTAL=0
for d in "$LOCAL_OUTDIR"/*/; do
    [ -d "$d" ] || continue
    mode=$(basename "$d")
    count=$(ls "$d"/*.npy 2>/dev/null | wc -l)
    echo "  $mode: ${count}개"
    TOTAL=$((TOTAL + count))
done
echo "  합계: ${TOTAL}개"
