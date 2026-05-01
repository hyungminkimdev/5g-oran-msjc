# MSJC 프로젝트 전체 로드맵 (Phase 1~7)

> 최종 업데이트: 2026-05-01
> 프로젝트: Hierarchical Jamming Classification xApp on 5G O-RAN (석사 논문)
> 타깃: IEEE MILCOM

---

## Phase 1: 인프라 구축 ✅

5G Full Stack(gNB↔UE↔5GC) + O-RAN RIC + E2 연동 구축.

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 1.1 | CCI xG Testbed Instance 3개 할당 + ZeroTier 네트워크 | ✅ | Instance-1(UE), 2(gNB+RIC), 3(Jammer) |
| 1.2 | USRP X310 3대 IP + MTU 9000 | ✅ | ens5, UHD 4.6.0 |
| 1.3 | srsRAN Project (gNB) 빌드 + MSJC KPM 패치 | ✅ | E2SM-KPM report_metrics 로깅 |
| 1.4 | srsRAN 4G (srsUE) 빌드 + NR SA 패치 | ✅ | branch `msjc-nr-sa-patches` |
| 1.5 | Open5GS 5GC + 테스트 가입자 등록 | ✅ | AMF/SMF/UPF |
| 1.6 | FlexRIC Near-RT RIC + E2AP 연동 | ✅ | branch `br-flexric` |
| 1.7 | Band 3 FDD 20MHz Golden Config 확정 | ✅ | srate=23.04MHz, FFT=1536 |
| 1.8 | gNB↔UE RF 링크 + PDU Session + ping | ✅ | |
| 1.9 | E2SM-KPM 구독 (1초 주기) | ✅ | 100ms는 gNB PRACH buffer 소진 |
| 1.10 | gNB RT 안정성 (taskset + chrt) | ✅ | 필수 |

---

## Phase 2: xApp ML 파이프라인 구현 ✅

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 2.1 | `kpi_feature_extractor.py` — 8-dim feature vector | ✅ | KPM mapping + CQI scale 보정 |
| 2.2 | `stage1_mlp.py` — MLP 5-class | ✅ | 8→256→128→64→5, 44.7K params |
| 2.3 | `stage2_ksvm.py` — KSVM sliding window | ✅ | W=15, 12-dim 통계 feature |
| 2.4 | `stage3_mobilenet.py` — MobileNetV3 4-class | ✅ | 224×224 spectrogram, 1.5M params |
| 2.5 | `xapp_msjc.py` — FlexRIC xApp | ✅ | MockFlexRIC, hot-reload, A1 API |
| 2.6 | `influx_logger.py` — InfluxDB 로깅 | ✅ | |
| 2.7 | `iq_snapshot.py` — I/Q 캡처 | ✅ | |
| 2.8 | 합성 데이터 학습 | ✅ | |
| 2.9 | ClearML 연동 | ✅ | 미설치 시 graceful skip |
| 2.10 | 모델 핫 리로드 | ✅ | threading.Lock + mtime |

---

## Phase 3: Jammer + KPM 수집 ✅

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 3.1 | `jammer.py` 7-mode FDD Band 3 | ✅ | Constant/Random/Reactive/Deceptive/PSS/PDCCH/DMRS |
| 3.2 | OFDM 구조 gNB 정합 | ✅ | FFT=1536, CP=108, master_clock=184.32MHz |
| 3.3 | Constant sweet spot (g0 a0.5~0.6) | ✅ | BLER 75~92%, UE 유지 |
| 3.4 | 초기 KPM 수집 (모드당 ~20개) | ✅ | `kpm_fdd_7modes.csv` (155개) |
| 3.5 | `gnb_kpm_parser.sh` | ✅ | binary log `-a` 대응 |
| 3.6 | `collect_all_modes.sh` 자동 수집 | ✅ | UE 복구, check_ue 수정 |
| 3.7 | 모드별 Gain/Amplitude 확정 | ✅ | CLAUDE.md 8.1 반영 |
| 3.8 | KPM 100ms 시도 → 1초 유지 | ✅ | gNB 한계 |

---

## Phase 4: 실데이터 재학습 + E2E 검증 ✅

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 4.1 | 실측 KPI range 분석 | ✅ | CQI accumulator(0-50) vs E2SM(0-15) 발견 |
| 4.2 | `_ATTACK_PROFILES` 실측 재조정 | ✅ | |
| 4.3 | Random jammer 수정 | ✅ | `build_random_frame(duty=30%)`, BLER 0→0.33 |
| 4.4 | Stage 2 → sliding window 개편 | ✅ | 12-dim 통계 feature |
| 4.5 | xApp KPM 히스토리 deque 연동 | ✅ | |
| 4.6 | Stage 1 재학습 (합성+실측) | ✅ | clean-looking 필터링 |
| 4.7 | 전모드 KPM 재수집 (954개) | ✅ | `kpm_fdd_alldata.csv` |
| 4.8 | Stage 2 실측 직접 학습 | ✅ | FA 8.5% → 3.1%(Phase 5) |
| 4.9 | Stage 3 합성 학습 | ✅ | 4-class 100% (합성) |
| 4.10 | 3-stage E2E MockFlexRIC 검증 | ✅ | latency ≤73ms |

---

## Phase 5: 실측 I/Q + 데이터 보강 ✅

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 5.1 | `iq_capture.py` Instance-1 passive RX | ✅ | gNB USRP 점유 → UE USRP 사용 |
| 5.2 | 모드별 I/Q 수집 (8×50=400개) | ✅ | .npy, 16384 samples/snapshot |
| 5.3 | Stage 3 실측 평가 → domain mismatch 발견 | ✅ | 합성만: PDCCH 100%, 나머지 0-14% |
| 5.4 | Stage 3 fine-tuning (합성200+실측200) | ✅ | PSS 86%, PDCCH 90%, DMRS 76%, Deceptive 100% |
| 5.5 | Constant burst 수집 100개 | ✅ | amp=0.5, BLER 74.9% |
| 5.6 | Normal 장시간 678개 수집 | ✅ | Stage 2 FA 8.5%→3.1% |
| 5.7 | RealFlexRIC E2E 연결 + CQI scale 버그 수정 | ✅ | E2SM CQI(0-15) → ×3.33 보정 |

---

## Phase 6: Closed-Loop MLOps ✅

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 6.1 | InfluxDB KPI 로깅 검증 | ✅ | xApp E2E에서 확인 |
| 6.2 | Grafana 대시보드 (5패널) | ✅ | Grafana 13.0.1 + InfluxDB v2.8.0 |
| 6.3 | `labeler_rapp.py` GMM 레이블링 | ✅ | Moving Avg → ARC(τ=0.0004) → GMM 2-class |
| 6.4 | Labeler CSV 데모 검증 | ✅ | Random 99%, Reactive 100% Attack |
| 6.5 | ClearML: Standalone fallback 구현 | ✅ | ClearML 미설치 시 직접 stage1_mlp.py 호출 |
| 6.6 | `training_manager_rapp.py` | ✅ | 정확도<30% 트리거, A1 알림 |
| 6.7 | A1 HTTP API (port 5000) | ✅ | MODEL_UPDATE / ACCURACY_REPORT / health |
| 6.8 | xApp A1 수신 → hot-reload 연동 | ✅ | |
| 6.9 | 폐루프 E2E 풀사이클 검증 | ✅ | xApp→Labeler→TrainMgr→A1→hot-reload |
| 6.10 | `demo_closedloop.sh` 데모 스크립트 | ✅ | |

---

## Phase 7: 논문 작성 (IEEE MILCOM) 🔶

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 7.1 | Figure 7개 + LaTeX Table 4개 생성 | ✅ | `tools/generate_paper_figures.py` |
| 7.2 | Hachimi et al. 대비 비교표 | ✅ | Table IV (LaTeX) |
| 7.3 | Rahman et al. 대비 차별점 | ✅ | Table IV 포함 |
| 7.4 | 스펙트로그램 시각화 | ✅ | `fig_spectrograms.pdf` |
| 7.5 | Latency CDF Figure | ✅ | `fig_latency_cdf.pdf` |
| 7.6 | E2SM-RC 제어 시연 | 📋 선택 | stub 수준, contribution 강화 |
| 7.7 | README + 재현성 패키지 | 📋 예정 | GitHub repo 정리 |
| 7.8 | 논문 초안 (Overleaf) | 📋 예정 | IEEE MILCOM 6페이지 |
| 7.9 | 지도교수 리뷰 + 수정 | 📋 예정 | |
| 7.10 | 최종 제출 | 📋 예정 | |

---

## 진행률 요약

| Phase | 설명 | 완료율 | 상태 |
|-------|------|--------|------|
| **Phase 1** | 인프라 구축 | 10/10 | ✅ |
| **Phase 2** | xApp ML 파이프라인 | 10/10 | ✅ |
| **Phase 3** | Jammer + KPM 수집 | 8/8 | ✅ |
| **Phase 4** | 실데이터 재학습 + E2E | 10/10 | ✅ |
| **Phase 5** | 실측 I/Q + 데이터 보강 | 7/7 | ✅ |
| **Phase 6** | Closed-Loop MLOps | 10/10 | ✅ |
| **Phase 7** | 논문 작성 | 5/10 | 🔶 진행 중 |

**전체: 60/65 태스크 완료 (92%)**

---

## 최종 성능 수치

| 지표 | 값 |
|------|-----|
| S1+S2 탐지율 | 전 7모드 **100%** |
| False Alarm | **3.1%** |
| S3 분류 (실측 I/Q) | PSS 86%, PDCCH 90%, DMRS 76%, Deceptive 100% |
| Latency | CLEAN 1-4ms, PROTOCOL_AWARE 40-73ms (**≤100ms**) |
| 데이터 | KPM 1,733개 + I/Q 400개 |
| Closed-loop | xApp→Labeler→TrainMgr→A1→hot-reload 검증 |
