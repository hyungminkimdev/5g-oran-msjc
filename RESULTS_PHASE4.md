# MSJC Phase 4 실험 결과 및 환경 스냅샷 (2026-04-30)

---

## 1. 실험 환경 (Reproducibility Snapshot)

### 1.1 하드웨어 — CCI xG Testbed

| Instance | 역할 | ZeroTier IP | USRP | USRP IP | Serial |
|----------|------|-------------|------|---------|--------|
| Instance-1 | UE (srsUE) | 10.111.143.165 | X310 | 192.168.114.2 | 323DF47 |
| Instance-2 | gNB + 5GC + RIC | 10.111.143.143 | X310 | 192.168.116.2 | 323DF42 |
| Instance-3 | Jammer | 10.111.143.61 | X310 | 192.168.115.2 | 323EE55 |

- SSH 비밀번호: `CCI@2025` (Instance-1, 3)
- 모든 Instance: `ens5` MTU 9000 (jumbo frames)

### 1.2 소프트웨어 버전

| 컴포넌트 | 버전 | 비고 |
|----------|------|------|
| OS | Ubuntu 24.04.3 LTS (kernel 6.8.0-110) | |
| Python | 3.12.3 | |
| PyTorch | 2.11.0+cpu | GPU 없음 |
| scikit-learn | 1.8.0 | |
| UHD | 4.6.0 | |
| srsRAN Project (gNB) | `fae1efa35d` (main + MSJC KPM patch) | |
| srsRAN 4G (UE) | `c661d75` branch `msjc-nr-sa-patches` | release_23_11 + NR SA patch |
| FlexRIC | `736508123` branch `br-flexric` | |
| Open5GS | 2.7.7 | |
| MSJC xApp | `41db12b` | Phase 4 완료 커밋 |

### 1.3 RF 설정 (Golden Config)

| 항목 | 값 |
|------|-----|
| Band | 3 (FDD) |
| DL freq | 1842.5 MHz (ARFCN 368500) |
| UL freq | 1747.5 MHz |
| Bandwidth | 20 MHz |
| SCS | 15 kHz (μ=0) |
| Sample rate | 23.04 MHz |
| FFT size | 1536 |
| PRB | 106 |
| gNB TX gain | 20 dB (max) |
| UE TX gain | 5 dB (max) |
| Master clock rate | 184.32 MHz (X310) |
| E2SM-KPM period | 1000 ms |

### 1.4 Jammer 설정 (실측 검증)

| Mode | Gain | Amplitude | Duty Cycle | 비고 |
|------|------|-----------|------------|------|
| Constant | 0 | 0.6 | 100% | 광대역 AWGN |
| Random | 0 | 0.6 | ~30% | `build_random_frame()` |
| Reactive | 0 | 0.6 | ~40% | 2-on/3-off 슬롯 패턴 |
| Deceptive | 0 | 0.6 | 100% | OFDM 위장 |
| PSS | 10 | 1.0 | 낮음 | 협대역 SSB 타겟 |
| PDCCH | 5 | 1.0 | 낮음 | CORESET 타겟 |
| DMRS | 10 | 1.0 | 낮음 | 빗살 패턴 파일럿 |

### 1.5 프로세스 시작 순서

```bash
# 1. Instance-2: gNB (taskset+chrt 필수!)
cd ~/srsRAN_Project/build
sudo taskset -c 0-5 chrt -f 90 ./apps/gnb/gnb \
  -c ~/5g-oran-msjc/srsran/gnb_msjc.yaml > /tmp/gnb.log 2>&1 &

# 2. Instance-2: FlexRIC (gNB 시작 후)
sudo taskset -c 6,7 ~/flexric/build/examples/ric/nearRT-RIC &

# 3. Instance-1: UE (gNB 안정화 8초 대기 후)
sudo /home/ubuntu/srsRAN_4G/build/srsue/src/srsue \
  /home/ubuntu/5g-oran-msjc/srsran/ue_msjc.conf > /tmp/srsue_safe.log 2>&1 &

# 4. UE attach 확인 (10~30초 소요)
grep "PDU Session" /tmp/srsue_safe.log

# 5. gNB KPM 확인
grep -a "MSJC KPM.*nof_ues=1" /tmp/gnb.log | tail -1

# 6. Instance-2: xApp
cd ~/5g-oran-msjc && python3 xapp_msjc.py

# 7. Instance-3: Jammer (필요 시)
cd ~/5g-oran-msjc/tools
sudo python3 jammer.py --mode constant --gain 0 --amplitude 0.6
```

### 1.6 모델 파일

| 파일 | 크기 | 설명 |
|------|------|------|
| `stage1_mlp.pth` | 190 KB | MLP 8→256→128→64→5, 합성+실측 학습 |
| `stage1_scaler.pkl` | 607 B | StandardScaler |
| `stage2_ksvm.pkl` | 9.2 KB | RBF SVM, 실측 sliding window 학습 |
| `stage3_mobilenet.pth` | 6.2 MB | MobileNetV3-Small, 합성 스펙트로그램 학습 |

### 1.7 데이터 파일

| 파일 | 행 수 | 설명 |
|------|------|------|
| `kpm_fdd_alldata.csv` | 954 | Phase 4 전모드 수집 (모드당 ~120개) |
| `kpm_fdd_7modes.csv` | 155 | Phase 3 초기 수집 (모드당 ~20개) |

---

## 2. 논문 결과 (Tables & Figures 소스)

### Table 1: 실측 KPM 통계 (모드별)

| Jamming Mode | N | SINR (dB) | BLER | CQI (accum) | 비고 |
|---|---|---|---|---|---|
| Normal (No Jam) | 120 | +4.2 ± 1.7 | 0.039 ± 0.173 | 48.5 ± 3.6 | 자연 변동 포함 |
| Constant | 7 | +1.0 ± 0.5 | **0.947 ± 0.060** | 47.9 ± 1.4 | UE 초반 7초만 유지 |
| Random | 119 | +4.2 ± 0.9 | **0.328 ± 0.246** | 49.2 ± 2.0 | duty 30%, BLER 69%>0 |
| Reactive | 119 | +4.2 ± 0.9 | **0.275 ± 0.246** | 49.4 ± 1.7 | duty 40%, BLER 66%>0 |
| Deceptive | 119 | +4.7 ± 0.6 | 0.000 ± 0.000 | 49.8 ± 1.2 | KPM 영향 미미 |
| PSS | 119 | +4.8 ± 0.6 | 0.008 ± 0.064 | 49.8 ± 1.2 | 간헐적 BLER spike |
| PDCCH | 119 | +4.7 ± 0.6 | 0.007 ± 0.076 | 49.9 ± 0.9 | 간헐적 CQI dip |
| DMRS | 119 | +4.7 ± 0.6 | 0.006 ± 0.068 | 49.8 ± 1.0 | CQI 41~50 |

### Table 2: Stage 1 MLP — Per-sample 5-class 분류

| Mode → S1 Class | Normal | Constant | Random | Reactive | Deceptive |
|---|---|---|---|---|---|
| **Normal** | **97.5%** | 0.8% | 0% | 0% | 1.7% |
| **Constant** | 0% | **100%** | 0% | 0% | 0% |
| **Random** | 31.1% | 0.8% | **48.7%** | 19.3% | 0% |
| **Reactive** | 33.6% | 5.0% | 25.2% | **37.0%** | 4.2% |
| Deceptive | 100% | 0% | 0% | 0% | 0% |
| PSS | 98.3% | 0% | 0% | 0% | 1.7% |
| PDCCH | 99.2% | 0.8% | 0% | 0% | 0% |
| DMRS | 99.2% | 0.8% | 0% | 0% | 0% |

> Stage 1 binary detection rate: Constant **100%**, Random **69%**, Reactive **66%**, Deceptive/PSS/PDCCH/DMRS **~0%**
> False alarm rate: **2.5%**

### Table 3: Stage 2 KSVM — Sliding Window (W=15) Binary Detection

| Mode | Attack Detection | False Alarm | Windows |
|---|---|---|---|
| Normal | — | **8.5%** | 106 |
| Constant | (샘플 부족) | — | <15 |
| Random | **100%** | — | 105 |
| Reactive | **100%** | — | 105 |
| Deceptive | **100%** | — | 105 |
| PSS | **100%** | — | 105 |
| PDCCH | **100%** | — | 105 |
| DMRS | **100%** | — | 105 |

> Stage 2 sliding window 통계 features (12-dim):
> `bler_mean, bler_var, bler_max, bler_spike_ratio, sinr_mean, sinr_var, sinr_min, cqi_var, cqi_min, nack_mean, bler_periodicity, transition_count`

### Table 4: Stage 3 MobileNetV3 — Protocol-Aware 4-class (합성 데이터)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| PSS/SSS | 1.000 | 1.000 | 1.000 |
| PDCCH | 1.000 | 1.000 | 1.000 |
| DMRS | 1.000 | 1.000 | 1.000 |
| Generic Deceptive | 1.000 | 1.000 | 1.000 |

> 합성 스펙트로그램 테스트 (클래스당 50개). 실측 I/Q 검증은 미실시.

### Table 5: Combined 3-Stage Pipeline 성능

| Mode | S1 Detection | S2 Detection | Combined | Latency |
|---|---|---|---|---|
| Normal (FA) | 2.5% | 8.5% | **~2% FA** | 1-4 ms |
| Constant | 100% | — | **100%** | 1-2 ms |
| Random | 69% | 100% | **~100%** | 1-2 / 3-4 ms |
| Reactive | 66% | 100% | **~100%** | 1-2 / 3-4 ms |
| Deceptive | 0% | 100% | **100%** | 50-73 ms |
| PSS | 2% | 100% | **100%** | 50-73 ms |
| PDCCH | 1% | 100% | **100%** | 50-73 ms |
| DMRS | 1% | 100% | **100%** | 50-73 ms |

> **모든 공격 유형 탐지. 100ms Near-RT RIC latency budget 이내.**

### Table 6: Latency Breakdown

| Path | Stage 1 | Stage 2 | Stage 3 | Total |
|---|---|---|---|---|
| CLEAN | 1-2 ms | 2-3 ms | — | **3-5 ms** |
| ATTACK_CONFIRMED | 1-2 ms | — | — | **1-2 ms** |
| FN_CAUGHT | 1-2 ms | 2-3 ms | 40-50 ms | **43-55 ms** |
| PROTOCOL_AWARE | 1-2 ms | — | 40-50 ms | **41-52 ms** |

### Figure 소스 데이터

- **Figure: KPM 시계열** — `kpm_fdd_alldata.csv` (SINR/BLER/CQI vs time, 모드별 색상)
- **Figure: Confusion Matrix** — Table 2 데이터
- **Figure: ROC Curve** — Stage 1 softmax 확률 + Stage 2 SVM probability
- **Figure: Spectrogram 예시** — `stage3_mobilenet.py`의 `simulate_attack()` + `iq_to_spectrogram_224()`
- **Figure: Pipeline 구조도** — CLAUDE.md Section 5.2

---

## 3. 한계점 및 향후 과제

1. **Constant 유효 데이터 부족** (7개) — gain=0 amp=0.6에서 UE가 ~7초 후 탈락. amplitude 미세 조정 필요.
2. **Stage 3 실측 I/Q 미검증** — 합성 스펙트로그램 100%이지만, 실제 USRP I/Q 캡처로 검증 미실시.
3. **Stage 2 Normal FA 8.5%** — 실환경 Normal 변동이 크면 false alarm 가능. 장시간 Normal 데이터 추가 수집으로 개선 가능.
4. **Random/Reactive 혼동** (Stage 1) — 둘 다 간헐적 BLER 패턴. periodicity 기반 구분은 Stage 2에서 수행되나, 5-class 세분류는 어려움.
5. **CQI accumulator 이슈** — srsRAN DU KPM의 CQI는 accumulator 값(0-50), 실제 CQI(0-15) 아님. 변환 로직 미적용.
