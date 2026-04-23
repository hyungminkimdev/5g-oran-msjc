# MSJC Project: Hierarchical Jamming Classification xApp on 5G O-RAN (srsRAN + FlexRIC)

## 1. Project Context & Goal

This project implements the MSJC (Multi-Stage Jamming Classification) framework as a **real O-RAN xApp** running on the CCI xG Testbed.
The classifier receives live 5G KPIs via the **E2 interface (E2SM-KPM)** from a srsRAN gNB, detects and classifies jamming attacks, and optionally issues control actions via **E2SM-RC**.

**Target accuracy:** 94.51% (per Hachimi et al. 2020)
**Latency budget:** ≤ 100 ms end-to-end (within Near-RT RIC timing window)

### What makes this genuinely 5G O-RAN (not just an SDR experiment)
- **srsRAN Project** runs a real 5G NR gNB (PHY/MAC/RLC/PDCP/RRC/SDAP) on Instance-2 USRP
- **srsUE** (srsRAN 4G repo) runs a real 5G NR UE on Instance-1 USRP
- **Open5GS** provides the 5G Core (AMF/SMF/UPF/NRF) on Instance-2
- **FlexRIC** Near-RT RIC connects to the gNB over **E2 (SCTP)** and hosts the MSJC xApp
- **E2SM-KPM** (KPI Monitoring Service Model) delivers per-cell/per-UE 5G metrics to the xApp
- **E2SM-RC** (RAN Control Service Model) allows the xApp to trigger mitigations (beam steering, power control)
- The **Jammer** (Instance-3 USRP) injects protocol-aware RF attacks into the live 5G link
- The **MSJC xApp** classifies jamming from real 5G KPIs — not from raw I/Q

---

## 2. Infrastructure & Hardware Mapping

> **ZeroTier 직접 연결** (ProxyJump/Gateway 미사용) — 물리적 순서: Classifier → Jammer → UE

| Role | Instance | ZeroTier IP | ZeroTier Net ID | USRP IP | USRP Serial |
|---|---|---|---|---|---|
| **UE (srsUE)** | Instance-1 | 10.111.143.165 | b92226be69 | 192.168.114.2 | 323DF47 |
| **gNB + 5GC + Near-RT RIC** | Instance-2 | 10.111.143.143 | d5cc70b949 | 192.168.116.2 | 323DF42 |
| **Jammer** | Instance-3 | 10.111.143.61 | e51276b157 | 192.168.115.2 | 323EE55 |

**Network interfaces:** All USRP I/Q streams use `ens5` with **MTU 9000** (jumbo frames).

---

## 3. Full Stack Architecture

```
Instance-1 (UE)                Instance-3 (Jammer)
┌──────────────┐               ┌──────────────────┐
│ srsUE        │               │ jammer.py        │
│ USRP X310    │               │ USRP X310        │
│ 192.168.114.2│               │ 192.168.115.2    │
└──────┬───────┘               └────────┬─────────┘
       │ RF (3.5 GHz, 20 MHz BW)        │ RF jamming
       └──────────────┬─────────────────┘
                      ▼
         Instance-2 (gNB + RIC)
┌──────────────────────────────────────────────┐
│                                              │
│  ┌─────────────────────┐                     │
│  │  srsRAN gNB          │  ←── USRP X310     │
│  │  PHY/MAC/RLC/PDCP    │      192.168.116.2 │
│  │  RRC/SDAP/F1/E1      │                    │
│  └─────────┬───────────┘                     │
│            │ N2/N3 (NGAP/GTP-U)              │
│  ┌─────────▼───────────┐                     │
│  │  Open5GS 5GC        │                     │
│  │  AMF / SMF / UPF    │                     │
│  │  NRF / AUSF / UDM   │                     │
│  └─────────────────────┘                     │
│            │ E2 (SCTP, port 36421)            │
│  ┌─────────▼───────────────────────────────┐ │
│  │  FlexRIC Near-RT RIC                    │ │
│  │  ┌──────────────────────────────────┐   │ │
│  │  │  MSJC xApp (xapp_msjc.py)        │   │ │
│  │  │  E2SM-KPM subscription (100ms)   │   │ │
│  │  │  Stage1 MLP → Stage2 KSVM        │   │ │
│  │  │  → Stage3 MobileNetV3            │   │ │
│  │  │  → InfluxDB KPI logging          │   │ │
│  │  │  → E2SM-RC control (optional)    │   │ │
│  │  └──────────────────────────────────┘   │ │
│  └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

---

## 4. Technology Stack

| Layer | Component | Version / Notes |
|---|---|---|
| **5G UE** | srsRAN 4G (srsUE) | `release_23_11` branch, USRP UHD |
| **5G gNB** | srsRAN Project | `main` branch, ZMQ disabled, UHD backend |
| **5G Core** | Open5GS | v2.7+, local install |
| **Near-RT RIC** | FlexRIC | EURECOM, `br_flexric_stable` branch |
| **E2 Service Model** | E2SM-KPM v2.0 | KPI subscription; E2SM-RC v1.0 for control |
| **ML Pipeline** | PyTorch 2.x, Scikit-learn | Stage1 MLP, Stage2 KSVM, Stage3 MobileNetV3 |
| **KPI Transport** | E2SM-KPM Indication | 100 ms report period |
| **MLOps** | ClearML | Experiment tracking, model versioning, retraining |
| **Observability** | InfluxDB v2 + Grafana | KPI time-series, detection events |
| **SDR Driver** | UHD 4.x | USRP X310, `ens5` MTU 9000 |

---

## 5. MSJC Pipeline — Input Layer Redesign

### 5.1 Feature Vector: E2SM-KPM KPIs (not raw I/Q)

The xApp receives **8-dimensional KPI vectors** from the gNB via E2SM-KPM, replacing the previous raw I/Q feature extraction.

```
Feature Index | KPI Name          | O-RAN KPM Measurement Name      | Jamming Signature
──────────────|───────────────────|─────────────────────────────────|──────────────────────────────
[0]  RSRP     | Ref Signal RX Pwr | NR.RSRP.Avg                     | Drops under Constant/PSS
[1]  RSRQ     | Ref Signal RX Qty | NR.RSRQ.Avg                     | Degrades under all jamming
[2]  SINR     | Signal/Interf+Noise| NR.SINR.Avg                   | Goes negative under jamming
[3]  BLER     | Block Error Rate  | DRB.RlcSduDelayDl (proxy)       | Spikes: 0→0.8+ under attack
[4]  UCI NACK | Uplink Ctrl NACK  | PUSCH.NackRatio                 | ~1.0 under Constant jamming
[5]  DL Tput  | DL Throughput     | DRB.UEThpDl                     | Collapses; periodic dip=Reactive
[6]  CQI      | Channel Quality   | L1M.CQI.Avg                     | →0 under PSS/SSS jamming
[7]  HARQ RTx | HARQ Retransmit   | HARQ.RetxRatio                  | Proportional to jam intensity
```

### 5.2 Classification Logic (unchanged cascade)

```
E2SM-KPM report (100ms) → extract 8-dim KPI vector
    │
    ▼
[Stage 1: MLP — 5-class]
    ├─ Constant / Random / Reactive  → ATTACK confirmed (log + optional RC mitigation)
    ├─ Deceptive (OFDM-like)        → [Stage 3: MobileNetV3 spectrogram] → PSS/PDCCH/DMRS/Generic
    └─ Normal                        → [Stage 2: KSVM binary recheck]
                                           ├─ Normal → CLEAN
                                           └─ Attack  → FN caught → [Stage 3]
```

### 5.3 Stage 3 Spectrogram Source

Stage 3 MobileNetV3 still uses **2D spectrograms**, but sourced from:
- **Primary:** raw I/Q snapshot triggered on-demand via srsRAN's monitoring interface (only when Stage 1 reports Deceptive or Stage 2 catches FN)
- **Fallback:** KPI time-series converted to a 2D heatmap (8 KPIs × sliding window of 28 samples = 224 time steps, resized to 224×224)

---

## 6. Key Files & Responsibilities

```
5g-oran-msjc/
├── CLAUDE.md                  ← This file
├── config.yaml                ← Runtime config (gitignored, copy from template)
├── config.template.yaml       ← Template (safe to commit)
│
├── srsran/
│   ├── gnb_msjc.yaml          ← srsRAN gNB config (USRP, E2, 5GC addresses)
│   └── ue_msjc.conf           ← srsUE config (USRP, PLMN, USIM)
│
├── open5gs/
│   └── setup_open5gs.sh       ← 5GC install + subscriber registration script
│
├── xapp_msjc.py               ← Main O-RAN xApp (FlexRIC Python SDK)
│                                 Subscribes E2SM-KPM, runs MSJC pipeline,
│                                 logs to InfluxDB, optionally sends E2SM-RC
│
├── stage1_mlp.py              ← MLP 5-class classifier (KPI features)
├── stage2_ksvm.py             ← KSVM binary False-Negative recheck (KPI features)
├── stage3_mobilenet.py        ← MobileNetV3 spectrogram (Protocol-Aware)
│
├── kpi_feature_extractor.py   ← E2SM-KPM report → 8-dim numpy feature vector
├── iq_snapshot.py             ← On-demand I/Q capture for Stage 3 (via srsRAN API)
│
├── jammer.py                  ← 7-mode jammer (NR-timing-aware: pss/pdcch/dmrs/...)
├── influx_logger.py           ← Async InfluxDB KPI + detection event logger
│
├── pipeline_runner.py         ← LEGACY: standalone raw-IQ runner (kept for reference)
└── ue_transmitter.py          ← LEGACY: raw USRP TX (replaced by srsUE)
```

---

## 7. O-RAN Interfaces

| Interface | Endpoints | Protocol | Purpose |
|---|---|---|---|
| **E2** | gNB ↔ Near-RT RIC | SCTP / ASN.1 | KPM subscription + RC control |
| **O1** | Near-RT RIC ↔ SMO | NETCONF/YANG | Configuration & fault management |
| **A1** | Non-RT RIC ↔ Near-RT RIC | HTTP/JSON | Policy injection (future) |
| **N2** | gNB ↔ AMF | NGAP / SCTP | UE registration, handover |
| **N3** | UPF ↔ gNB | GTP-U / UDP | User-plane data |

---

## 8. Development Constraints

- **E2 Latency:** E2SM-KPM report period = 100 ms. Full MSJC pipeline must complete within 100 ms.
- **Stage 3 trigger:** Invoke MobileNetV3 only for Deceptive/FN cases to stay within latency budget.
- **MTU:** `sudo ip link set ens5 mtu 9000` must be set on all three instances before UHD streaming.
- **Atomic model update:** Hot-reload model weights (threading.Lock + state_dict swap) without restarting the xApp process.
- **Synthetic bootstrap:** All three ML models must train on synthetic KPI data before real srsRAN data is available. Simulator functions must faithfully reproduce KPI signatures per jamming type.
- **Graceful degradation:** If FlexRIC E2 connection drops, xApp must fall back to raw-IQ mode (legacy pipeline_runner.py logic) automatically.

---

## 9. Build & Run Commands

### Environment Setup (Instance-2, one-time)
```bash
# MTU 9000 on all instances
sudo ip link set ens5 mtu 9000

# srsRAN Project (gNB)
git clone https://github.com/srsran/srsRAN_Project.git
cd srsRAN_Project && mkdir build && cd build
cmake -DENABLE_EXPORT=ON .. && make -j$(nproc)

# srsRAN 4G (srsUE) — Instance-1
git clone https://github.com/srsran/srsRAN_4G.git
cd srsRAN_4G && mkdir build && cd build
cmake .. && make -j$(nproc) srsue

# Open5GS (5GC) — Instance-2
sudo apt install open5gs
# Register test subscriber: IMSI 001010123456780

# FlexRIC (Near-RT RIC) — Instance-2
git clone https://gitlab.eurecom.fr/mosaic5g/flexric.git
cd flexric && mkdir build && cd build
cmake -DXAPP_DB=SQLITE3_XAPP_DB .. && make -j$(nproc) && sudo make install
```

### Runtime (in order)
```bash
# 1. Instance-2: Start 5GC
sudo systemctl start open5gs-mmed  # (or open5gs-amfd for 5G SA)

# 2. Instance-2: Start gNB
cd srsRAN_Project/build && sudo ./apps/gnb/gnb -c /path/to/gnb_msjc.yaml

# 3. Instance-2: Start Near-RT RIC
cd flexric/build && ./nearRT-RIC

# 4. Instance-2: Start MSJC xApp
cd 5g-oran-msjc && python3 xapp_msjc.py

# 5. Instance-1: Start UE
cd srsRAN_4G/build && sudo ./srsue/src/srsue /path/to/ue_msjc.conf

# 6. Instance-3: Start Jammer (optional)
cd 5g-oran-msjc && python3 jammer.py --mode pss
```

### ML Model Training
```bash
# Stage 1 (KPI-based MLP)
python3 stage1_mlp.py --retrain --n-per-class 1000 --epochs 100

# Stage 2 (KPI-based KSVM)
python3 stage2_ksvm.py --retrain --n-per-class 1000

# Stage 3 (Spectrogram MobileNetV3 — unchanged)
python3 stage3_mobilenet.py --retrain --n-per-class 500 --epochs 50
```

### Legacy Raw-IQ Pipeline (fallback / reference only)
```bash
python3 pipeline_runner.py [--no-stage2] [--no-stage3]
```

---

## 10. ClearML MLOps Integration

All three model training runs must log to ClearML:
- **Task name format:** `msjc-stage{1,2,3}-{timestamp}`
- **Tracked artifacts:** model weights (`.pth`, `.pkl`), scaler, confusion matrix, per-class accuracy
- **Retraining trigger:** When MSJC xApp FN rate > 5% over a 1-hour window → ClearML retraining task auto-queued

---

## 11. InfluxDB Schema

```
Measurement: "detection"
Tags:    s1_label, final_verdict, s3_label, node_id
Fields:  rsrp, rsrq, sinr, bler, uci_nack_rate, dl_throughput_mbps,
         cqi_mean, harq_retx_rate,
         s1_confidence, s2_confidence, s3_confidence,
         latency_ms, fn_caught (bool)
```

---

## 12. Paper Reference

Hachimi et al. (2020) — "Multi-stage Jamming Attacks Detection using Deep Learning Combined with Kernelized Support Vector Machine in 5G Wireless Network."
- Section III-A: KSVM for physical jamming signatures
- Section III-B: MLP for fast binary/multi-class detection
- Section IV: MobileNetV3 for spectrogram-based protocol-aware classification
- **Note:** Original paper uses simulated data. This project extends it to a live 5G O-RAN testbed with real E2SM-KPM KPIs.
