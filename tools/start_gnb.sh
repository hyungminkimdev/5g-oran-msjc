#!/bin/bash
# Start gNB with proper CPU affinity and RT priority
GNB_BIN=/home/ubuntu/srsRAN_Project/build/apps/gnb/gnb
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/../srsran/gnb_msjc.yaml"
LOG=/tmp/gnb_fdd.log

# Launch gNB on cores 0-5
taskset -c 0-5 $GNB_BIN -c $CONFIG > $LOG 2>&1 &
GNB_PID=$!
echo "gNB PID: $GNB_PID"

# Apply RT priority immediately
sleep 1
chrt -f -p 90 $GNB_PID 2>/dev/null
echo "RT priority applied: $(chrt -p $GNB_PID 2>&1)"

wait $GNB_PID
