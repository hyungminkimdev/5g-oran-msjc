# MSJC — Multi-Stage Jamming Classification xApp for 5G O-RAN

> Hierarchical jamming detection and protocol-aware classification on a real 5G O-RAN testbed with closed-loop MLOps.

[![srsRAN](https://img.shields.io/badge/srsRAN-Project-blue)](https://github.com/srsran/srsRAN_Project)
[![FlexRIC](https://img.shields.io/badge/FlexRIC-Near--RT%20RIC-green)](https://gitlab.eurecom.fr/mosaic5g/flexric)
[![Open5GS](https://img.shields.io/badge/Open5GS-5GC-orange)](https://open5gs.org/)

---

## Overview

MSJC is an O-RAN Near-RT RIC **xApp** that detects and classifies 7 types of RF jamming attacks against a live 5G NR link using a 3-stage ML cascade:

```
E2SM-KPM (1s) → [Stage 1: MLP 5-class] → Constant/Random/Reactive → ATTACK
                                        → Deceptive → [Stage 3: MobileNetV3] → PSS/PDCCH/DMRS/Generic
                                        → Normal → [Stage 2: KSVM Window] → Attack → FN_CAUGHT
                                                                           → Normal → CLEAN
```

| Metric | Value |
|--------|-------|
| **Detection Rate** | 100% (all 7 modes, S1+S2 combined) |
| **False Alarm** | 3.1% |
| **Latency** | ≤ 73 ms (within Near-RT RIC 100 ms budget) |
| **Protocol Classification** | PSS 86%, PDCCH 90%, DMRS 76%, Deceptive 100% |

## Architecture

```
Instance-1 (UE)               Instance-3 (Jammer)
┌──────────────┐              ┌──────────────────┐
│ srsUE        │              │ jammer.py        │
│ USRP X310    │              │ USRP X310        │
└──────┬───────┘              └────────┬─────────┘
       │ RF (Band 3 FDD, 20 MHz)      │ RF jamming
       └────────────┬─────────────────┘
                    ▼
      Instance-2 (gNB + 5GC + RIC)
┌──────────────────────────────────────────────┐
│  srsRAN gNB ──── Open5GS 5GC                │
│       │ E2 (SCTP)                            │
│  FlexRIC Near-RT RIC                         │
│  ├── MSJC xApp (xapp_msjc.py)               │
│  │   ├── Stage 1: MLP (8→256→128→64→5)      │
│  │   ├── Stage 2: KSVM (W=15, 12-dim stats) │
│  │   └── Stage 3: MobileNetV3 (224×224)      │
│  │                                           │
│  ├── Labeler rApp (GMM auto-labeling)        │
│  ├── Training Manager rApp (MLOps)           │
│  ├── InfluxDB v2 + Grafana                   │
│  └── A1 HTTP API (:5000)                     │
└──────────────────────────────────────────────┘
```

## Jamming Modes

| Mode | Type | Gain | Amp | Effect |
|------|------|------|-----|--------|
| **Constant** | Broadband AWGN | 0 | 0.5 | BLER ~75%, SINR -1 dB |
| **Random** | Burst AWGN (30% duty) | 0 | 0.6 | BLER ~33%, intermittent |
| **Reactive** | Periodic (40% duty) | 0 | 0.6 | BLER ~28%, periodic |
| **Deceptive** | OFDM-like | 0 | 0.6 | Minimal KPM impact |
| **PSS** | SSB-targeted narrowband | 10 | 1.0 | Sync disruption |
| **PDCCH** | CORESET-targeted | 5 | 1.0 | Control channel attack |
| **DMRS** | Comb-pattern pilot | 10 | 1.0 | Channel estimation attack |

## Quick Start

### Prerequisites

- CCI xG Testbed (3 instances with USRP X310)
- Ubuntu 24.04, Python 3.12+, UHD 4.6.0
- srsRAN Project + srsRAN 4G (patched), Open5GS 2.7+, FlexRIC

### 1. Clone & Configure

```bash
git clone https://github.com/hyungminkimdev/5g-oran-msjc.git
cd 5g-oran-msjc
cp config.template.yaml config.yaml
# Edit config.yaml: set InfluxDB token, network IPs
```

### 2. Start 5G Stack

```bash
# Instance-2: gNB (taskset+chrt required!)
sudo taskset -c 0-5 chrt -f 90 \
  ~/srsRAN_Project/build/apps/gnb/gnb -c srsran/gnb_msjc.yaml

# Instance-2: Near-RT RIC
sudo taskset -c 6,7 ~/flexric/build/examples/ric/nearRT-RIC

# Instance-1: UE
sudo ~/srsRAN_4G/build/srsue/src/srsue srsran/ue_msjc.conf
```

### 3. Train Models

```bash
# Stage 1: MLP (synthetic + real data)
python3 stage1_mlp.py --retrain --n-per-class 1000 --epochs 100 \
  --real-csv kpm_fdd_alldata.csv

# Stage 2: KSVM (trains on real sliding windows)
python3 stage2_ksvm.py --retrain

# Stage 3: MobileNetV3 (synthetic, then fine-tune with real I/Q)
python3 stage3_mobilenet.py --retrain --n-per-class 300 --epochs 30
```

### 4. Run xApp

```bash
# With real FlexRIC:
python3 xapp_msjc.py

# With MockFlexRIC (no RIC needed):
python3 xapp_msjc.py --mock-ric

# A1 API available at http://localhost:5000/a1/health
```

### 5. Run Jammer (Instance-3)

```bash
python3 tools/jammer.py --mode constant --gain 0 --amplitude 0.5
```

### 6. Closed-Loop MLOps Demo

```bash
bash tools/demo_closedloop.sh
# Runs: xApp → Labeler (GMM) → Training Manager (retrain) → A1 → hot-reload
```

## Project Structure

```
5g-oran-msjc/
├── xapp_msjc.py               # Main xApp (E2SM-KPM + 3-stage + A1 API)
├── stage1_mlp.py               # Stage 1: MLP 5-class (44.7K params)
├── stage2_ksvm.py              # Stage 2: KSVM sliding window (W=15, 12-dim)
├── stage3_mobilenet.py         # Stage 3: MobileNetV3-Small (1.5M params)
├── kpi_feature_extractor.py    # E2SM-KPM → 8-dim feature vector
├── influx_logger.py            # InfluxDB async logging
├── iq_snapshot.py              # On-demand I/Q capture
├── labeler_rapp.py             # Non-RT RIC: GMM auto-labeling rApp
├── training_manager_rapp.py    # Non-RT RIC: ML lifecycle rApp
│
├── srsran/
│   ├── gnb_msjc.yaml           # gNB config (Band 3 FDD, 23.04 MHz)
│   ├── ue_msjc.conf            # UE config
│   └── xapp_kpm.conf           # FlexRIC KPM subscription
│
├── grafana/
│   ├── datasource.yaml         # InfluxDB Flux datasource
│   └── dashboard.json          # 5-panel monitoring dashboard
│
├── tools/
│   ├── jammer.py               # 7-mode RF jammer
│   ├── collect_all_modes.sh    # Automated KPM data collection
│   ├── iq_capture.py           # Passive RX I/Q capture (Instance-1)
│   ├── collect_iq_all.sh       # Automated I/Q collection
│   ├── demo_closedloop.sh      # Closed-loop E2E demo
│   ├── generate_paper_figures.py # Paper figures (PDF)
│   └── ...
│
├── config.template.yaml        # Config template (copy to config.yaml)
├── CLAUDE.md                   # Detailed project documentation
└── PROJECT_ROADMAP.md          # Phase 1-7 roadmap
```

## RF Configuration (Golden Config)

| Parameter | Value |
|-----------|-------|
| Band | 3 (FDD) |
| DL Frequency | 1842.5 MHz (ARFCN 368500) |
| Bandwidth | 20 MHz (106 PRB) |
| Subcarrier Spacing | 15 kHz (μ=0) |
| Sample Rate | 23.04 MHz |
| FFT Size | 1536 |
| KPM Report Period | 1000 ms |
| gNB TX Gain | ≤ 20 dB |
| UE TX Gain | ≤ 5 dB |

## Data

Data files are `.gitignore`d (too large for git). Reproduce with:

```bash
# KPM data collection (all 7 modes + Normal):
bash tools/collect_all_modes.sh kpm_fdd_alldata.csv

# I/Q snapshot collection:
bash tools/collect_iq_all.sh 50 /tmp/iq_captures ~/5g-oran-msjc/iq_data
```

Model files (`.pth`, `.pkl`) are also gitignored. Retrain with commands above.

## Key Results

### Detection (Stage 1 + Stage 2 Combined)

| Mode | Stage 1 | Stage 2 | Combined |
|------|---------|---------|----------|
| Normal (FA) | 2.3% | 3.1% | **~3%** |
| Constant | 100% | 100% | **100%** |
| Random | 68% | 100% | **~100%** |
| Reactive | 64% | 100% | **~100%** |
| Deceptive | 0% | 100% | **100%** |
| PSS | 1% | 100% | **100%** |
| PDCCH | 0% | 100% | **100%** |
| DMRS | 0% | 100% | **100%** |

### Latency

| Path | Latency |
|------|---------|
| CLEAN (S1→S2) | 1–4 ms |
| ATTACK_CONFIRMED (S1) | 1–2 ms |
| PROTOCOL_AWARE (S1→S3) | 40–73 ms |
| Near-RT RIC Budget | **≤ 100 ms** ✅ |

## References

1. M. Hachimi et al., "Multi-stage Jamming Attacks Detection using Deep Learning Combined with Kernelized Support Vector Machine in 5G Cloud-RAN," *IEEE GLOBECOM*, 2020.
2. M. Rahman et al., "SAJD: Self-Adaptive Jamming Attack Detection in AI/ML Integrated 5G O-RAN Networks," *arXiv:2511.17519*, 2025.

## License

This project is part of a master's thesis at Virginia Tech / CCI xG Testbed.
