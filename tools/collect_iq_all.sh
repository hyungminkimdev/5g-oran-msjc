#!/bin/bash
# Phase 5: 전모드 I/Q 캡처 자동화
# Instance-2(로컬)에서 실행.
# Instance-1: UE USRP passive RX (srsUE 중지 상태)
# Instance-3: Jammer 제어
#
# 사전조건:
#   - gNB 실행 중 (Instance-2) — DL 신호 송출
#   - srsUE 중지 (Instance-1) — USRP 해제
#   - Instance-1/3에 git pull 완료
#
# 사용법: bash tools/collect_iq_all.sh [n_snapshots] [outdir_on_instance1]

set -e

N_SNAPSHOTS=${1:-50}
REMOTE_OUTDIR=${2:-/tmp/iq_captures}
LOCAL_OUTDIR="${3:-/home/ubuntu/5g-oran-msjc/iq_data}"

UE_IP="10.111.143.165"
JAMMER_IP="10.111.143.61"
PW="CCI@2025"

# 모드별 gain/amplitude (CLAUDE.md 8.1)
declare -A JAM_GAIN JAM_AMP
JAM_GAIN[constant]=0;   JAM_AMP[constant]=0.6
JAM_GAIN[random]=0;     JAM_AMP[random]=0.6
JAM_GAIN[reactive]=0;   JAM_AMP[reactive]=0.6
JAM_GAIN[deceptive]=0;  JAM_AMP[deceptive]=0.6
JAM_GAIN[pss]=10;       JAM_AMP[pss]=1.0
JAM_GAIN[pdcch]=5;      JAM_AMP[pdcch]=1.0
JAM_GAIN[dmrs]=10;      JAM_AMP[dmrs]=1.0

MODES="Normal constant random reactive deceptive pss pdcch dmrs"

ssh_ue() {
    sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@$UE_IP "$@" 2>/dev/null
}
ssh_jam() {
    sshpass -p "$PW" ssh -o StrictHostKeyChecking=no ubuntu@$JAMMER_IP "$@" 2>/dev/null
}

start_jammer() {
    local mode=$1
    local gain=${JAM_GAIN[$mode]}
    local amp=${JAM_AMP[$mode]}
    echo "[*] Jammer 시작: $mode (gain=$gain, amp=$amp)"
    sshpass -p "$PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@$JAMMER_IP \
        "echo '$PW' | sudo -S nohup python3 ~/5g-oran-msjc/tools/jammer.py --mode $mode --gain $gain --amplitude $amp > /tmp/jammer.log 2>&1 &
         sleep 4; echo 'Jammer started'" 2>&1 | tail -1
}

stop_jammer() {
    sshpass -p "$PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@$JAMMER_IP \
        "echo '$PW' | sudo -S killall -9 python3 2>/dev/null; echo done" 2>&1 | tail -1
    sleep 1
}

capture_iq() {
    local label=$1
    echo "[*] I/Q 캡처: $label × $N_SNAPSHOTS"
    sshpass -p "$PW" ssh -tt -o StrictHostKeyChecking=no ubuntu@$UE_IP \
        "echo '$PW' | sudo -S python3 ~/5g-oran-msjc/tools/iq_capture.py \
         --mode $label --n-snapshots $N_SNAPSHOTS --outdir $REMOTE_OUTDIR" 2>&1
}

echo "=== Phase 5: 전모드 I/Q 캡처 ==="
echo "  Snapshots/mode: $N_SNAPSHOTS"
echo "  Remote dir: $REMOTE_OUTDIR (Instance-1)"
echo "  Local dir: $LOCAL_OUTDIR"
echo ""

# srsUE 중지 확인
echo "[*] srsUE 중지..."
ssh_ue "sudo pkill -9 srsue 2>/dev/null; echo done"
sleep 2

# 기존 캡처 디렉토리 초기화
ssh_ue "rm -rf $REMOTE_OUTDIR; mkdir -p $REMOTE_OUTDIR"

for MODE in $MODES; do
    LABEL=$(echo "$MODE" | sed 's/^./\U&/')
    [ "$MODE" = "Normal" ] && LABEL="Normal"

    echo ""
    echo "=========================================="
    echo "[$(date +%H:%M:%S)] $LABEL"
    echo "=========================================="

    if [ "$MODE" != "Normal" ]; then
        start_jammer "$MODE"
        sleep 2
    fi

    capture_iq "$LABEL"

    if [ "$MODE" != "Normal" ]; then
        stop_jammer
    fi

    echo "[$(date +%H:%M:%S)] $LABEL 완료"
    sleep 3
done

# 결과 전송 Instance-1 → Instance-2
echo ""
echo "[*] I/Q 파일 전송 Instance-1 → Instance-2..."
mkdir -p "$LOCAL_OUTDIR"
scp -r -o StrictHostKeyChecking=no "ubuntu@$UE_IP:$REMOTE_OUTDIR/*" "$LOCAL_OUTDIR/" 2>/dev/null

echo ""
echo "=== I/Q 캡처 완료 ==="
for d in "$LOCAL_OUTDIR"/*/; do
    mode=$(basename "$d")
    count=$(ls "$d"/*.npy 2>/dev/null | wc -l)
    echo "  $mode: ${count}개"
done
