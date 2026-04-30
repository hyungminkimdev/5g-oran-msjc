# MSJC 프로젝트 전체 로드맵 (Phase 1~7)

> 최종 업데이트: 2026-04-30
> 프로젝트: Hierarchical Jamming Classification xApp on 5G O-RAN (석사 논문)

---

## Phase 1: 인프라 구축 ✅

5G Full Stack(gNB↔UE↔5GC) + O-RAN RIC + E2 연동 구축.

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 1.1 | CCI xG Testbed Instance 3개 할당 및 ZeroTier 네트워크 구성 | ✅ 완료 | Instance-1(UE), 2(gNB+RIC), 3(Jammer) |
| 1.2 | USRP X310 3대 IP 설정 + MTU 9000 (jumbo frames) | ✅ 완료 | ens5 인터페이스, UHD 4.6.0 |
| 1.3 | srsRAN Project (gNB) 빌드 + MSJC KPM 패치 적용 | ✅ 완료 | E2SM-KPM report_metrics 로깅 추가 |
| 1.4 | srsRAN 4G (srsUE) 빌드 + NR SA 패치 적용 | ✅ 완료 | branch `msjc-nr-sa-patches` (release_23_11 + band n78/cell-search) |
| 1.5 | Open5GS 5GC 설치 + 테스트 가입자 등록 | ✅ 완료 | AMF/SMF/UPF, IMSI 001010123456780 |
| 1.6 | FlexRIC Near-RT RIC 빌드 + E2AP 연동 확인 | ✅ 완료 | branch `br-flexric`, SCTP port 36421 |
| 1.7 | Band 3 FDD 20MHz Golden Config 확정 | ✅ 완료 | srate=23.04MHz, FFT=1536, SCS=15kHz, ARFCN=368500 |
| 1.8 | gNB↔UE RF 링크 확립 + PDU Session + User Plane ping 확인 | ✅ 완료 | `ping -I tun_srsue 10.45.0.1` 성공 |
| 1.9 | E2SM-KPM 구독 확인 (gNB→RIC 1초 주기 KPI 보고) | ✅ 완료 | 100ms는 gNB PRACH buffer 소진 → 1초 유지 |
| 1.10 | gNB RT 안정성 확보 (taskset + chrt) | ✅ 완료 | `taskset -c 0-5 chrt -f 90` 필수, E2와 동시 실행 시 |

---

## Phase 2: xApp ML 파이프라인 구현 ✅

MSJC 3-Stage 분류기 코드 + xApp 프레임워크 구현.

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 2.1 | `kpi_feature_extractor.py` — E2SM-KPM → 8-dim feature vector 변환 | ✅ 완료 | KPM key mapping, simulate_kpi_chunk(), CSV 로더 |
| 2.2 | `stage1_mlp.py` — MLP 5-class 분류기 (Normal/Constant/Random/Reactive/Deceptive) | ✅ 완료 | 8→256→128→64→5, dual-mode (KPI/I/Q) |
| 2.3 | `stage2_ksvm.py` — KSVM binary False Negative 재검사 | ✅ 완료 | Sliding window 15샘플 12-dim 통계 feature |
| 2.4 | `stage3_mobilenet.py` — MobileNetV3 Protocol-Aware 4-class 스펙트로그램 분류 | ✅ 완료 | PSS/PDCCH/DMRS/Generic, 224×224 spectrogram |
| 2.5 | `xapp_msjc.py` — FlexRIC xApp 메인 (E2SM-KPM 콜백 + 3-stage 파이프라인) | ✅ 완료 | MockFlexRIC fallback, hot-reload, KPM history deque |
| 2.6 | `influx_logger.py` — InfluxDB v2 KPI + detection event 로깅 | ✅ 완료 | 비동기 로깅 |
| 2.7 | `iq_snapshot.py` — Stage 3용 on-demand I/Q 캡처 | ✅ 완료 | USRP 직접 캡처, KPI→fake I/Q fallback |
| 2.8 | 합성 데이터 기반 Stage 1/2/3 초기 학습 | ✅ 완료 | 합성 프로파일 기반 bootstrap |
| 2.9 | ClearML 연동 (Stage 1/2 학습 추적, FN rate 재학습 트리거) | ✅ 완료 | ClearML 미설치 시 graceful skip |
| 2.10 | 모델 핫 리로드 (threading.Lock + state_dict swap) | ✅ 완료 | 파일 mtime 감시, 60초 주기 |

---

## Phase 3: Jammer 구현 + KPM 데이터 수집 ✅

7-mode FDD jammer 구현 + 실측 KPM 데이터 첫 수집.

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 3.1 | `jammer.py` — 7-mode FDD Band 3 jammer 구현 | ✅ 완료 | Constant/Random/Reactive/Deceptive/PSS/PDCCH/DMRS |
| 3.2 | Jammer OFDM 구조 gNB 정합 (FFT=1536, CP=108, SCS=15kHz) | ✅ 완료 | master_clock_rate=184.32MHz 필수 |
| 3.3 | Constant sweet spot 탐색 (gain=0, amp=0.6) | ✅ 완료 | SINR 4.7→1.0dB, BLER 0→92%, UE 유지 |
| 3.4 | 7-mode KPM 초기 수집 (모드당 ~20개) | ✅ 완료 | `kpm_fdd_7modes.csv` (155개) |
| 3.5 | `gnb_kpm_parser.sh` — gNB stdout KPM 파싱 CSV 출력 | ✅ 완료 | binary log `-a` 플래그 대응 |
| 3.6 | `collect_all_modes.sh` — 전모드 자동 수집 + UE 복구 로직 | ✅ 완료 | 모드별 gain/amp, check_ue 버그 수정 |
| 3.7 | 모드별 Gain/Amplitude 실측 확정 | ✅ 완료 | CLAUDE.md 8.1 반영. 광대역 g0/a0.6, PSS/DMRS g10/a1.0, PDCCH g5/a1.0 |
| 3.8 | KPM 100ms 시도 → 실패 → 1초 유지 결정 | ✅ 완료 | gNB PRACH buffer 소진 문제 |

---

## Phase 4: 실데이터 기반 재학습 + 파이프라인 검증 ✅

합성 프로파일을 실측 KPM range에 재조정하고, 전체 파이프라인 end-to-end 검증.

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 4.1 | 실측 KPI range 분석 (CQI accumulator, PUCCH.SINR 매핑) | ✅ 완료 | CQI 50 = accumulator (실제 CQI 15), 4개 feature만 유효 |
| 4.2 | `_ATTACK_PROFILES` 실측 기반 재조정 | ✅ 완료 | Normal BLER 0-0.15, Random/Reactive burst 확률 실측 반영 |
| 4.3 | Random jammer 로직 리뷰 + 수정 | ✅ 완료 | `build_random_frame()`: duty 2%→30%, sleep→frame 단위, OFDM→AWGN |
| 4.4 | Stage 2 KSVM → sliding window 12-dim 전면 개편 | ✅ 완료 | bler_var/max, sinr_var/min, cqi_var, periodicity, transition_count |
| 4.5 | xapp_msjc.py KPM 히스토리 deque + Stage 2 window 연동 | ✅ 완료 | `deque(maxlen=15)` |
| 4.6 | Stage 1 MLP 재학습 (합성 + 실측 혼합, clean-looking 필터링) | ✅ 완료 | Normal 98%, Constant 100%, Random 49% |
| 4.7 | 전모드 KPM 재수집 (모드당 120초, 총 954개) | ✅ 완료 | `kpm_fdd_alldata.csv`, Random BLER 0→0.33 효과 확인 |
| 4.8 | Stage 2 KSVM 실측 데이터 직접 학습 | ✅ 완료 | Normal FA 8.5%, 전 공격 100% |
| 4.9 | Stage 3 MobileNetV3 학습 (합성 스펙트로그램) | ✅ 완료 | 4-class 100% (합성) |
| 4.10 | 3-stage 파이프라인 end-to-end MockFlexRIC 검증 | ✅ 완료 | CLEAN 1-4ms, ATTACK 1-2ms, PROTOCOL_AWARE 40-73ms |
| 4.11 | 논문 결과 정리 + 환경 스냅샷 | ✅ 완료 | `RESULTS_PHASE4.md` |

---

## Phase 5: 실측 I/Q 기반 Stage 3 검증 + 추가 데이터 🔶

Stage 3 MobileNetV3를 실제 USRP I/Q 캡처로 검증하고, 부족한 데이터를 보강.

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 5.1 | I/Q 캡처 도구 검증 — `iq_snapshot.py`로 gNB USRP에서 I/Q 스냅샷 캡처 | 📋 예정 | UHD RX streamer 동작 확인 필요 |
| 5.2 | 모드별 실측 I/Q 수집 (Jammer ON 상태에서 gNB USRP 캡처) | 📋 예정 | 모드당 50+ 스냅샷 목표 |
| 5.3 | 실측 I/Q → 스펙트로그램 변환 + Stage 3 정확도 평가 | 📋 예정 | 합성 100% → 실측 ?% |
| 5.4 | Stage 3 fine-tuning (합성 + 실측 I/Q 혼합 학습) | 📋 예정 | 정확도 저하 시 수행 |
| 5.5 | Constant 데이터 보강 (amplitude 미세 조정, 안정적 UE 유지) | 📋 예정 | 현재 7개 유효, amp=0.5 시도 |
| 5.6 | 장시간 Normal 데이터 수집 (1시간+) | 📋 예정 | Stage 2 FA 개선용, baseline 변동 프로파일링 |
| 5.7 | RealFlexRIC 모드로 xApp E2E 테스트 (Mock 아닌 실제 E2) | 📋 예정 | KPM→S1→S2→S3 실시간 경로 검증 |

---

## Phase 6: Closed-Loop MLOps (rApps + ClearML) 🔶

Rahman et al. (2025) 기반 자동 라벨링 + 재학습 폐루프 구현.

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 6.1 | InfluxDB v2 KPI 로깅 검증 — xApp → InfluxDB 실시간 기록 확인 | 📋 예정 | `influx_logger.py` 동작 확인 |
| 6.2 | Grafana 대시보드 — KPI 시계열 + detection event 시각화 | 📋 예정 | 논문 Figure 소스 |
| 6.3 | `labeler_rapp.py` — GMM 기반 비지도 자동 라벨링 구현 | 📋 예정 | InfluxDB fetch → smooth → ARC → GMM clustering |
| 6.4 | Labeler rApp 검증 — 실측 KPM에 대해 auto-label 정확도 평가 | 📋 예정 | 수동 레이블 vs GMM 레이블 비교 |
| 6.5 | ClearML 설치 + 에이전트 구성 (Instance-2) | 📋 예정 | ClearML server + agent + model registry |
| 6.6 | `training_manager_rapp.py` — ClearML 기반 재학습 오케스트레이션 | 📋 예정 | Task clone → enqueue → monitor → A1 notify |
| 6.7 | A1 인터페이스 구현 (HTTP REST, MODEL_UPDATE / ACCURACY_REPORT) | 📋 예정 | Training Manager ↔ xApp |
| 6.8 | xApp A1 수신 → 모델 핫 리로드 연동 | 📋 예정 | 기존 hot-reload + A1 trigger 합치기 |
| 6.9 | 폐루프 E2E 테스트 — Jammer 모드 변경 → xApp 감지 → FN rate 상승 → 자동 재학습 → 모델 업데이트 | 📋 예정 | 전체 사이클 검증 |
| 6.10 | Drift 시나리오 테스트 — Jammer 파라미터 점진 변경 시 자동 적응 확인 | 📋 예정 | ARC threshold τ 튜닝 |

---

## Phase 7: 논문 작성 + 최종 실험 🔶

석사 논문 완성을 위한 결과 정리, 추가 실험, 문서화.

| # | 태스크 | 상태 | 비고 |
|---|--------|------|------|
| 7.1 | 논문 Table/Figure 최종 확정 — Confusion Matrix, ROC, KPM 시계열 | 📋 예정 | RESULTS_PHASE4.md 기반 |
| 7.2 | Hachimi et al. (2020) 대비 성능 비교표 작성 | 📋 예정 | 목표 94.51% vs 실측 결과 |
| 7.3 | Rahman et al. (2025) 대비 아키텍처 차별점 정리 | 📋 예정 | Binary→5-class, 3-stage cascade, sliding window |
| 7.4 | 스펙트로그램 시각화 (4 attack type 비교 Figure) | 📋 예정 | stage3_mobilenet.py의 simulate_attack → spectrogram |
| 7.5 | Latency 분석 Figure — Stage별 추론 시간 분포 | 📋 예정 | MockFlexRIC 로그에서 추출 |
| 7.6 | E2SM-RC 제어 시연 (optional) — 공격 감지 시 beam/power 조정 | 📋 예정 | 논문 contribution 강화, 구현은 stub 수준 |
| 7.7 | 재현성 패키지 — 코드 + 데이터 + 모델 + 환경 세팅 문서화 | 📋 예정 | GitHub repo 정리, README |
| 7.8 | 논문 초안 작성 (Introduction ~ Conclusion) | 📋 예정 | |
| 7.9 | 지도교수 리뷰 + 수정 | 📋 예정 | |
| 7.10 | 최종 제출 | 📋 예정 | |

---

## 진행률 요약

| Phase | 설명 | 상태 | 완료율 |
|-------|------|------|--------|
| **Phase 1** | 인프라 구축 | ✅ 완료 | 10/10 |
| **Phase 2** | xApp ML 파이프라인 | ✅ 완료 | 10/10 |
| **Phase 3** | Jammer + KPM 수집 | ✅ 완료 | 8/8 |
| **Phase 4** | 실데이터 재학습 + E2E | ✅ 완료 | 11/11 |
| **Phase 5** | 실측 I/Q + 추가 데이터 | 🔶 예정 | 0/7 |
| **Phase 6** | Closed-Loop MLOps | 🔶 예정 | 0/10 |
| **Phase 7** | 논문 작성 | 🔶 예정 | 0/10 |

**전체: 39/66 태스크 완료 (59%)**
