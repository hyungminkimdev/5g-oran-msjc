#!/bin/bash
# Safe srsUE launcher: timeout + reject monitoring
# Usage: ./run-ue-safe.sh <config_file> [timeout_seconds]

CONF="${1:?Usage: $0 <config_file> [timeout_seconds]}"
TIMEOUT="${2:-30}"
LOG="/tmp/srsue_safe.log"

echo "[safe-ue] Starting srsUE with config=$CONF timeout=${TIMEOUT}s"

# Launch srsUE in background
sudo /home/ubuntu/srsRAN_4G/build/srsue/src/srsue "$CONF" > "$LOG" 2>&1 &
UE_PID=$!

echo "[safe-ue] srsUE PID=$UE_PID, monitoring..."

START=$(date +%s)
ATTACHED=0

while kill -0 $UE_PID 2>/dev/null; do
    ELAPSED=$(( $(date +%s) - START ))

    # Check for NAS reject (not GW failures)
    if grep -qi 'Reject\|NAS.*failed\|attach.*failed\|registration.*reject' "$LOG" 2>/dev/null; then
        echo "[safe-ue] NAS REJECT detected in log! Killing srsUE."
        sudo kill -9 $UE_PID 2>/dev/null
        tail -20 "$LOG"
        exit 1
    fi

    # Check for successful attach
    if grep -qi 'RRC Connected\|PDU Session' "$LOG" 2>/dev/null; then
        if [ $ATTACHED -eq 0 ]; then
            echo "[safe-ue] Attach SUCCESS detected at ${ELAPSED}s"
            ATTACHED=1
        fi
    fi

    # Timeout check (only if not yet attached)
    if [ $ATTACHED -eq 0 ] && [ $ELAPSED -ge $TIMEOUT ]; then
        echo "[safe-ue] TIMEOUT (${TIMEOUT}s) without attach. Killing srsUE."
        sudo kill -9 $UE_PID 2>/dev/null
        tail -20 "$LOG"
        exit 2
    fi

    sleep 1
done

echo "[safe-ue] srsUE process exited."
tail -20 "$LOG"
