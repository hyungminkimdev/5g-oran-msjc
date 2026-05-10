# MSJC — Multi-Stage Jamming Classification for 5G O-RAN

> Multi-stage jamming detection and protocol-aware classification on a real 5G O-RAN testbed with closed-loop MLOps.

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
| **Detection Rate** | 100% (all 7 modes, S1+S2 combined, real data hold-out) |
| **False Alarm** | 1.25% (held-out real testbed data, N=325 windows) |
| **Latency** | ≤ 23 ms p95 (well within Near-RT RIC 100 ms budget) |
| **Protocol Classification** | PSS 94%, PDCCH 72%, DMRS 84%, Deceptive 90% (85% overall, real I/Q) |

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
│   ├── eval_real_crossval.py    # Real data cross-validation
│   ├── collect_extended.sh      # Extended KPM collection (10min/mode)
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
| Normal (FA) | 1.25% | 0% | **1.25%** |
| Constant | 100% | 100% | **100%** |
| Random | 66.4% | 100% | **100%** |
| Reactive | 63.0% | 100% | **100%** |
| Deceptive | 0% | 100% | **100%** |
| PSS | 0% | 100% | **100%** |
| PDCCH | 0% | 100% | **100%** |
| DMRS | 0% | 100% | **100%** |

### Latency

| Path | Stages | Latency (p5–p95) |
|------|--------|-----------------|
| CLEAN | S1→S2 | 0.2–3.0 ms |
| ATTACK_CONFIRMED | S1 | < 0.5 ms |
| FN_CAUGHT | S1→S2→S3 | 8.8–22.5 ms |
| PROTOCOL_AWARE | S1→S3 | 8.2–20.7 ms |
| Near-RT RIC Budget | — | **≤ 100 ms** |

## References

### Core — MSJC Pipeline Basis
1. M. Hachimi, G. Kaddoum, and G. Gagnon, "Multi-stage Jamming Attacks Detection using Deep Learning Combined with Kernelized Support Vector Machine in 5G Cloud-RAN," *IEEE GLOBECOM*, 2020. doi:[10.1109/GLOBECOM42002.2020.9348244](https://doi.org/10.1109/GLOBECOM42002.2020.9348244)
2. M. A. Rahman, M. A. K. Azad, A. Beheshti, *et al.*, "SAJD: Self-Adaptive Jamming Attack Detection in AI/ML Integrated 5G O-RAN Networks," *arXiv:2511.17519*, 2025.

### Jamming Surveys & Classification
3. H. Pirayesh and H. Zeng, "Jamming Attacks and Anti-Jamming Strategies in Wireless Networks: A Comprehensive Survey," *IEEE Commun. Surveys Tuts.*, vol. 24, no. 2, pp. 767–809, 2022. doi:[10.1109/COMST.2022.3159185](https://doi.org/10.1109/COMST.2022.3159185)
4. S. Savadatti, S. K. Dhariwal, S. Krishnamoorthy, and R. Delhibabu, "An Extensive Classification of 5G Network Jamming Attacks," *Security and Communication Networks*, vol. 2024, 2024. doi:[10.1155/2024/2883082](https://doi.org/10.1155/2024/2883082)
5. M. Harvanek, J. Bolcek, J. Kufa, L. Polak, M. Simka, and R. Marsalek, "Survey on 5G Physical Layer Security Threats and Countermeasures," *Sensors*, vol. 24, no. 17, p. 5523, 2024. doi:[10.3390/s24175523](https://doi.org/10.3390/s24175523)
6. M. F. Shahid, K. Mehmood, M. Mohsin, A. Saleem, S. Yaqoob, and W. Bashir, "Taxonomy of Physical Layer Jamming Techniques and Strategies for Security Enhancement in Wireless Communication: A Comprehensive Survey," *TechRxiv*, 2024. doi:[10.36227/techrxiv.172425846.66605015/v1](https://doi.org/10.36227/techrxiv.172425846.66605015/v1)

### ML-based Jamming Detection
7. S. Jere, Y. Wang, I. Aryendu, S. Dayekh, and L. Liu, "Bayesian Inference-Assisted Machine Learning for Near Real-Time Jamming Detection and Classification in 5G NR," *IEEE Trans. Wireless Commun.*, vol. 23, no. 7, pp. 7043–7058, 2024. doi:[10.1109/TWC.2023.3337058](https://doi.org/10.1109/TWC.2023.3337058)

### O-RAN Architecture
8. M. Polese, L. Bonati, S. D'Oro, S. Basagni, and T. Melodia, "Understanding O-RAN: Architecture, Interfaces, Algorithms, Security, and Research Challenges," *IEEE Commun. Surveys Tuts.*, vol. 25, no. 2, pp. 1376–1411, 2023. doi:[10.1109/COMST.2023.3239220](https://doi.org/10.1109/COMST.2023.3239220)
9. A. S. Abdalla, P. S. Upadhyaya, V. K. Shah, and V. Marojevic, "Toward Next Generation Open Radio Access Networks: What O-RAN Can and Cannot Do!," *IEEE Network*, vol. 36, no. 6, pp. 206–213, 2022. doi:[10.1109/MNET.108.2100659](https://doi.org/10.1109/MNET.108.2100659)
10. S. Marinova and A. Leon-Garcia, "Intelligent O-RAN Beyond 5G: Architecture, Use Cases, Challenges, and Opportunities," *IEEE Access*, vol. 12, pp. 27088–27121, 2024. doi:[10.1109/ACCESS.2024.3367289](https://doi.org/10.1109/ACCESS.2024.3367289)

### O-RAN xApp Design & Testbeds
11. J. F. Santos, A. Huff, D. Campos, K. V. Cardoso, C. B. Both, and L. A. DaSilva, "Managing O-RAN Networks: xApp Development from Zero to Hero," *IEEE Commun. Surveys Tuts.*, 2025. doi:[10.1109/COMST.2025.3539687](https://doi.org/10.1109/COMST.2025.3539687)
12. M. Hoffmann, S. Janji, A. Samorzewski, *et al.*, "Open RAN xApps Design and Evaluation: Lessons Learnt and Identified Challenges," *IEEE J. Sel. Areas Commun.*, vol. 42, no. 2, pp. 473–491, 2024. doi:[10.1109/JSAC.2023.3336190](https://doi.org/10.1109/JSAC.2023.3336190)
13. A. da Silva, M. Roy Chowdhury, A. Sathish, A. Tripathi, S. F. Midkiff, and L. A. da Silva, "CCI xG Testbed: An O-RAN Based Platform for Future Wireless Network Experimentation," *IEEE Commun. Mag.*, vol. 63, no. 2, pp. 62–68, 2025. doi:[10.1109/MCOM.001.2400322](https://doi.org/10.1109/MCOM.001.2400322)
14. N. H. Stephenson, A. J. Chiejina, N. B. Kabigting, and V. K. Shah, "Demonstration of Closed Loop AI-Driven RAN Controllers Using O-RAN SDR Testbed," *IEEE MILCOM*, 2023. doi:[10.1109/MILCOM58377.2023.10356330](https://doi.org/10.1109/MILCOM58377.2023.10356330)

### O-RAN Jamming / Anomaly Detection xApps
15. P. Kryszkiewicz and M. Hoffmann, "Open RAN for Detection of a Jamming Attack in a 5G Network," *IEEE VTC2023-Spring*, 2023. doi:[10.1109/VTC2023-Spring57618.2023.10201067](https://doi.org/10.1109/VTC2023-Spring57618.2023.10201067)
16. J. Moore, A. S. Abdalla, C. Ueltschey, and V. Marojevic, "Demonstrating Jamming Mitigation in O-RAN via AI enabled Intrusion Detection and Secure Slicing xApps," *IEEE MILCOM*, 2025. doi:[10.1109/MILCOM64451.2025.11310542](https://doi.org/10.1109/MILCOM64451.2025.11310542)
17. H. Bogucka, M. Hoffmann, P. Kryszkiewicz, and Ł. Kułacz, "An Open-RAN Testbed for Detecting and Mitigating Radio-Access Anomalies," *IEEE Commun. Mag.*, 2025. doi:[10.1109/MCOM.003.2400513](https://doi.org/10.1109/MCOM.003.2400513)
18. S. Dimou and G. Noubir, "ARGOS: Anomaly Recognition and Guarding through O-RAN Sensing," *arXiv:2506.06916*, 2025.
19. A. Paz-Pérez, J. Suárez Gómez, F. J. Valera Sánchez, and J. J. Escudero-Garzás, "Design and Implementation of an xApp-based System for Jamming and Interference Mitigation," *IEEE ISNCC*, 2025. doi:[10.1109/ISNCC66965.2025.11250457](https://doi.org/10.1109/ISNCC66965.2025.11250457)

### Spectrogram / I/Q ML
20. G. Reus-Muns, P. S. Upadhyaya, U. Demir, N. Stephenson, N. Soltani, V. K. Shah, and K. R. Chowdhury, "SenseORAN: O-RAN-Based Radar Detection in the CBRS Band," *IEEE J. Sel. Areas Commun.*, vol. 42, no. 2, pp. 326–340, 2024. doi:[10.1109/JSAC.2023.3336152](https://doi.org/10.1109/JSAC.2023.3336152)

### ML Models
21. A. Howard, M. Sandler, G. Chu, *et al.*, "Searching for MobileNetV3," *IEEE/CVF ICCV*, pp. 1314–1324, 2019. doi:[10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140)

### O-RAN Security
22. P. K. Kakani, M. A. Habibi, M. R. Chavva Balannagari, X. Costa-Pérez, and H. D. Schotten, "Mitigating ML-Driven Adversarial Attacks on xApps Using Dynamic Defense Mechanisms," *IEEE Open J. Commun. Soc.*, vol. 6, pp. 6912–6930, 2025. doi:[10.1109/OJCOMS.2025.3602200](https://doi.org/10.1109/OJCOMS.2025.3602200)
23. Y. Rumesh, D. Attanayaka, P. Porambage, J. Pinola, J. Groen, and K. Chowdhury, "Federated Learning for Anomaly Detection in Open RAN: Security Architecture Within a Digital Twin," *EuCNC & 6G Summit*, 2024.

## Paper

**MSJC: Multi-Stage Jamming Classification for 5G O-RAN with Closed-Loop MLOps**
Hyungmin Kim and Eric W. Burger, Virginia Tech.
Target venue: IEEE MILCOM.

Paper source in `paper/main.tex`.

## License

This project is part of a master's thesis at Virginia Tech / CCI xG Testbed.
