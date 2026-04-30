#!/bin/bash
# gNB 로그에서 KPM 메트릭을 직접 파싱하여 CSV로 출력
# Usage: ./gnb_kpm_parser.sh <label> <duration_sec> <output_csv>
LABEL="${1:-test}"
DURATION="${2:-10}"
OUTCSV="${3:-/tmp/kpm_gnb_parsed.csv}"

# Header if file doesn't exist
if [ ! -s "$OUTCSV" ]; then
  echo "timestamp,label,nof_ues,cqi,pusch_snr,pucch_snr,dl_ok,dl_nok,dl_bler" > "$OUTCSV"
fi

echo "[parser] Collecting '$LABEL' for ${DURATION}s from gNB log..."
END=$(($(date +%s) + DURATION))
COUNT=0

tail -f /tmp/gnb.log | grep -a --line-buffered "MSJC KPM report_metrics" | while read -r line; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$END" ]; then
    break
  fi

  # Parse: nof_ues=1 cqi_obs=50 pusch_snr=0.0 pucch_snr=4.0 dl_nof_ok=0 dl_nof_nok=0
  TS=$(echo "$line" | grep -oP '^\S+')
  NUES=$(echo "$line" | grep -oP 'nof_ues=\K[0-9]+')
  CQI=$(echo "$line" | grep -oP 'cqi_obs=\K[0-9.]+')
  PUSCH=$(echo "$line" | grep -oP 'pusch_snr=\K[-0-9.]+')
  PUCCH=$(echo "$line" | grep -oP 'pucch_snr=\K[-0-9.]+')
  DL_OK=$(echo "$line" | grep -oP 'dl_nof_ok=\K[0-9]+')
  DL_NOK=$(echo "$line" | grep -oP 'dl_nof_nok=\K[0-9]+')

  # Compute BLER
  TOTAL=$((DL_OK + DL_NOK))
  if [ "$TOTAL" -gt 0 ]; then
    BLER=$(echo "scale=4; $DL_NOK / $TOTAL" | bc)
  else
    BLER="0.0"
  fi

  # Convert CQI observation count to actual CQI (cqi_obs is sum of CQI values / period)
  # In gNB, cqi_obs is the raw CQI accumulator; 15 means CQI=15 for 1 sample
  # If cqi_obs > 15, it means multiple samples accumulated
  echo "$(date +%s.%N),$LABEL,$NUES,$CQI,$PUSCH,$PUCCH,$DL_OK,$DL_NOK,$BLER" >> "$OUTCSV"

  COUNT=$((COUNT + 1))
  if [ $((COUNT % 5)) -eq 0 ]; then
    echo "  [$LABEL] #$COUNT CQI_obs=$CQI PUCCH_SNR=$PUCCH DL_BLER=$BLER"
  fi
done

echo "[parser] Done: $COUNT samples → $OUTCSV"
