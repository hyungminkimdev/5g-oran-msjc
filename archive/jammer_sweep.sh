#!/bin/bash
# Jammer sweep: find degradation sweet spot
# Usage: ./jammer_sweep.sh
# Requires: UE attached, iperf3 server running on 10.45.0.1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

JAMMER_IP="10.111.143.61"
UE_IP="10.111.143.165"
JAMMER_FREQ="1842.5e6"
JAMMER_RATE="23.04e6"
CSV="/tmp/kpm_sweep.csv"
rm -f "$CSV"

echo "=== Jammer Sweep: Finding sweet spot ==="

# Sweep configurations: gain amplitude
CONFIGS=(
    "0 0.001"
    "0 0.005"
    "0 0.01"
    "0 0.05"
    "0 0.1"
    "0 0.5"
    "5 0.01"
    "5 0.05"
    "5 0.1"
)

for cfg in "${CONFIGS[@]}"; do
    read -r GAIN AMP <<< "$cfg"
    LABEL="constant_g${GAIN}_a${AMP}"

    echo ""
    echo "--- Testing: gain=$GAIN amplitude=$AMP ---"

    # Start jammer
    sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$JAMMER_IP \
        "nohup python3 ~/5g-oran-msjc/tools/jammer.py --mode constant --freq $JAMMER_FREQ --gain $GAIN --amplitude $AMP --rate $JAMMER_RATE > /tmp/jammer.log 2>&1 &" 2>/dev/null

    # Wait for jammer to initialize
    sleep 3

    # Check if UE is still connected
    UE_ALIVE=$(sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$UE_IP \
        'ps aux | grep srsue | grep -v grep | wc -l' 2>/dev/null)

    if [ "$UE_ALIVE" -eq 0 ]; then
        echo "  UE DROPPED! Stopping sweep at gain=$GAIN amp=$AMP"
        sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$JAMMER_IP \
            'sudo pkill -f jammer.py' 2>/dev/null
        break
    fi

    # Collect KPM for 10 seconds
    echo "  Collecting KPM..."
    python3 "$REPO_DIR/tools/kpm_collector.py" "$LABEL" 10 "$CSV" 2>&1 | grep -E "\[|Done"

    # Kill jammer
    sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$JAMMER_IP \
        'sudo pkill -f jammer.py' 2>/dev/null

    # Brief recovery pause
    sleep 2

    # Check UE again
    UE_ALIVE=$(sshpass -p 'CCI@2025' ssh -o StrictHostKeyChecking=no ubuntu@$UE_IP \
        'ps aux | grep srsue | grep -v grep | wc -l' 2>/dev/null)

    if [ "$UE_ALIVE" -eq 0 ]; then
        echo "  UE dropped after jammer off. Stopping."
        break
    fi
done

echo ""
echo "=== Sweep complete. Results in $CSV ==="
echo "Summary:"
python3 -c "
import csv
from collections import defaultdict
data = defaultdict(list)
with open('$CSV') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row['label']
        data[label].append({
            'CQI': float(row.get('CQI', 0)),
            'SINR': float(row.get('PUCCH.SINR', 0)),
            'BLER': float(row.get('DL.BLER', 0)),
        })

for label, rows in data.items():
    n = len(rows)
    avg_cqi = sum(r['CQI'] for r in rows) / n
    avg_sinr = sum(r['SINR'] for r in rows) / n
    avg_bler = sum(r['BLER'] for r in rows) / n
    print(f'  {label}: CQI={avg_cqi:.1f} SINR={avg_sinr:.1f} BLER={avg_bler:.2f} (n={n})')
"
