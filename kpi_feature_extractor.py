"""
KPI Feature Extractor — MSJC Project
Converts E2SM-KPM indication messages to 8-dim feature vectors for the MSJC pipeline.

Feature order (CLAUDE.md Section 5.1):
  [0] rsrp_dbm          — Reference Signal Received Power (dBm)
  [1] rsrq_db           — Reference Signal Received Quality (dB)
  [2] sinr_db           — Signal to Interference + Noise Ratio (dB)
  [3] bler              — Block Error Rate (0.0 ~ 1.0)
  [4] uci_nack_rate     — Uplink Control NACK ratio (0.0 ~ 1.0)
  [5] dl_throughput_mbps — Downlink throughput (Mbps)
  [6] cqi_mean          — Channel Quality Indicator mean (0 ~ 15)
  [7] harq_retx_rate    — HARQ retransmission rate (0.0 ~ 1.0)

Also provides simulate_kpi_chunk() for synthetic dataset generation.
"""

import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NUM_FEATURES = 8

FEATURE_NAMES = [
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "bler",
    "uci_nack_rate",
    "dl_throughput_mbps",
    "cqi_mean",
    "harq_retx_rate",
]

# Safe defaults: values that look "Normal" when a KPM key is missing
_DEFAULTS = {
    "rsrp_dbm":          -80.0,
    "rsrq_db":           -12.0,
    "sinr_db":            20.0,
    "bler":                0.01,
    "uci_nack_rate":       0.01,
    "dl_throughput_mbps": 100.0,
    "cqi_mean":           12.0,
    "harq_retx_rate":      0.02,
}

# FlexRIC E2SM-KPM measurement name → our feature name mapping
# Includes pass-through entries for FEATURE_NAMES themselves (used by MockFlexRIC)
_KPM_KEY_MAP = {
    # O-RAN / srsRAN KPM measurement names (swig_wrapper.cpp MSJC_KPM_MEAS 순서와 일치)
    "DRB.UEThpDl":       "dl_throughput_mbps",
    "L1M.DL-SS-SINR":    "sinr_db",
    "L1M.SS-RSRP":       "rsrp_dbm",
    "MR.SS-RSRQ":        "rsrq_db",
    # TB.ErrTotNbrDl / TB.TotNbrDl → bler는 _KPMCallback에서 직접 계산
    "bler":              "bler",
    "DRB.RlcSduDelayDl": "bler",       # latency proxy (BLER 대리값)
    # 추가 별칭 (표준 O-RAN / 3GPP 이름)
    "NR.RSRP.Avg":       "rsrp_dbm",
    "NR.RSRP":           "rsrp_dbm",
    "rsrp":              "rsrp_dbm",
    "NR.RSRQ.Avg":       "rsrq_db",
    "NR.RSRQ":           "rsrq_db",
    "rsrq":              "rsrq_db",
    "NR.SINR.Avg":       "sinr_db",
    "L1M.RS-SINR":       "sinr_db",
    "sinr":              "sinr_db",
    "PUSCH.NackRatio":   "uci_nack_rate",
    "uci_nack_rate":     "uci_nack_rate",
    "dl_throughput":     "dl_throughput_mbps",
    "L1M.CQI.Avg":       "cqi_mean",
    "cqi":               "cqi_mean",
    "HARQ.RetxRatio":    "harq_retx_rate",
    "harq_retx_rate":    "harq_retx_rate",
    # srsRAN DU KPM 메트릭 직접 매핑 (e2sm_kpm_du_meas_provider_impl.cpp)
    "CQI":               "cqi_mean",
    "RSRP":              "rsrp_dbm",     # srsRAN: PUSCH SNR (dB) — RSRP placeholder
    "RSRQ":              "rsrq_db",      # srsRAN: PUSCH SNR (dB) — RSRQ placeholder
    "DL.BLER":           "bler",
    "UL.BLER":           "uci_nack_rate", # UL BLER → UCI NACK rate proxy
    "PUCCH.SINR":        "sinr_db",      # PUCCH SINR → SINR feature
    # Pass-through: FEATURE_NAMES used directly as keys (MockFlexRIC, internal use)
    "rsrp_dbm":          "rsrp_dbm",
    "rsrq_db":           "rsrq_db",
    "sinr_db":           "sinr_db",
    "dl_throughput_mbps": "dl_throughput_mbps",
    "cqi_mean":          "cqi_mean",
}


# ─────────────────────────────────────────────
# 1. E2SM-KPM → Feature Vector
# ─────────────────────────────────────────────
def extract_from_kpm(kpm_msg: dict) -> np.ndarray:
    """
    Convert a FlexRIC E2SM-KPM indication message (Python dict) to
    an 8-dimensional float32 feature vector.

    The dict may use either FlexRIC E2SM-KPM measurement names or
    our internal feature names as keys. Missing keys get safe Normal defaults.

    Args:
        kpm_msg: dict with KPI values (e.g. {"L1M.RS-SINR": 25.3, "bler": 0.01})

    Returns:
        np.ndarray of shape (8,), dtype float32
    """
    values = dict(_DEFAULTS)

    for raw_key, raw_val in kpm_msg.items():
        feat_name = _KPM_KEY_MAP.get(raw_key)
        if feat_name is not None:
            try:
                values[feat_name] = float(raw_val)
            except (ValueError, TypeError):
                pass

    # E2SM-KPM CQI 보정: srsRAN DU KPM은 평균 CQI(0-15)를 보내지만,
    # gNB stdout은 accumulator(0-50)를 출력. 모델은 accumulator scale로 학습.
    # CQI ≤ 15이면 accumulator scale로 변환 (×3.33)
    if values["cqi_mean"] <= 15.0:
        values["cqi_mean"] = values["cqi_mean"] * (50.0 / 15.0)

    # dl_throughput=0 (idle)이면 default 유지 (모델이 0에 민감)
    if values["dl_throughput_mbps"] < 0.01:
        values["dl_throughput_mbps"] = _DEFAULTS["dl_throughput_mbps"]

    vec = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    return vec


# ─────────────────────────────────────────────
# 2. Synthetic KPI Simulation
#    *** Phase 4: 실측 FDD KPM 데이터 기반 프로파일 (2026-04-30) ***
#    kpm_fdd_7modes.csv에서 측정된 실제 range로 재조정
#
#    Feature 매핑 (srsRAN DU KPM → 8-dim):
#      [0] rsrp_dbm  ← PUSCH SNR (bimodal: 0 or ~30dB, 65-80% 가 0)
#      [1] rsrq_db   ← 미제공 (항상 default -12.0)
#      [2] sinr_db   ← PUCCH.SINR (주요 판별 지표)
#      [3] bler      ← DL.BLER (주요 판별 지표)
#      [4] uci_nack  ← dl_nok/(dl_ok+dl_nok) 파생
#      [5] dl_tput   ← DRB.UEThpDl (CSV에 없음, default 근방)
#      [6] cqi_mean  ← CQI accumulator (Normal=50, NOT 0-15 scale)
#      [7] harq_retx ← BLER과 유사하게 파생
# ─────────────────────────────────────────────
_ATTACK_PROFILES = {
    #                   rsrp(pusch)    rsrq(fixed)    sinr(pucch)  bler         nack         tput(fixed)  cqi(accum)   harq
    # Normal: 자연 변동 포함 (실측: 6/120 BLER spike, 20/120 CQI dip)
    "Normal":         [(-1, 32),     (-12.5,-11.5), (-1.0,6.0),  (0.0,0.15),  (0.0,0.15),  (95,105),    (38,50),     (0.0,0.15)],
    "Constant":       [(-5, 31),     (-12.5,-11.5), (-2.5, 2.0), (0.80,1.0),  (0.80,1.0),  (95,105),    (38,50),     (0.80,1.0)],
    # Deceptive: 명확한 KPM 열화만 (clean-looking Deceptive은 Stage 2/3 담당)
    "Deceptive":      [(-1, 32),     (-12.5,-11.5), (-5.0, 3.5), (0.02,0.5),  (0.02,0.5),  (95,105),    (36,49),     (0.02,0.5)],
    # Stage 3 subtypes (backward compat)
    "PSS/SSS":        [(-1, 32),     (-12.5,-11.5), (-0.5, 6.0), (0.0, 0.05), (0.0, 0.05), (95,105),    (36,50),     (0.0, 0.05)],
    "PDCCH":          [(-26, 31),    (-12.5,-11.5), (0.5, 5.5),  (0.0, 0.85), (0.0, 0.85), (95,105),    (40,50),     (0.0, 0.85)],
    "DMRS":           [(-1, 32),     (-12.5,-11.5), (1.0, 6.0),  (0.0, 0.05), (0.0, 0.05), (95,105),    (40,50),     (0.0, 0.05)],
}
# Burst/intermittent attacks: (jammed_profile, clean_profile, jam_probability)
_BURST_PROFILES = {
    "Random":   ("Random_jam",   "Random_clean",   0.65),   # 실측: 69% BLER>0
    "Reactive": ("Reactive_jam", "Reactive_clean",  0.60),   # 실측: 66% BLER>0
}
# Random: duty 30% → BLER 0~1.0, mean=0.33 (실측)
_ATTACK_PROFILES["Random_jam"]    = [(-1, 32),    (-12.5,-11.5), (-1.0, 6.0), (0.10,1.0),  (0.10,1.0),  (95,105),   (38,50),     (0.10,1.0)]
_ATTACK_PROFILES["Random_clean"]  = [(-1, 32),    (-12.5,-11.5), (0.0, 6.0),  (0.0,0.05),  (0.0,0.05),  (95,105),   (40,50),     (0.0,0.05)]
# Reactive: duty 40% → BLER 0~1.0, mean=0.28 (실측), CQI 40 dip
_ATTACK_PROFILES["Reactive_jam"]  = [(-1, 32),    (-12.5,-11.5), (0.0, 6.0),  (0.10,1.0),  (0.10,1.0),  (95,105),   (40,50),     (0.10,1.0)]
_ATTACK_PROFILES["Reactive_clean"]= [(-1, 32),    (-12.5,-11.5), (0.0, 6.0),  (0.0,0.05),  (0.0,0.05),  (95,105),   (40,50),     (0.0,0.05)]


def _sample_profile(profile: list) -> np.ndarray:
    return np.array([np.random.uniform(lo, hi) for lo, hi in profile], dtype=np.float32)


def simulate_kpi_chunk(label: str) -> np.ndarray:
    """Generate a synthetic 8-dim KPI feature vector for a given jamming label."""
    if label in _BURST_PROFILES:
        jam_key, clean_key, prob = _BURST_PROFILES[label]
        key = jam_key if np.random.rand() < prob else clean_key
        return _sample_profile(_ATTACK_PROFILES[key])

    if label not in _ATTACK_PROFILES:
        raise ValueError(f"Unknown KPI simulation label: {label}")
    return _sample_profile(_ATTACK_PROFILES[label])


# ─────────────────────────────────────────────
# 3. 실측 CSV → Feature Vector 변환
# ─────────────────────────────────────────────
# 7-mode jammer label → Stage 1 5-class 매핑
_LABEL_TO_STAGE1 = {
    "Normal": "Normal",
    "Constant": "Constant",
    "Random": "Random",
    "Reactive": "Reactive",
    "Deceptive": "Deceptive",
    "PSS": "Deceptive",
    "PDCCH": "Deceptive",
    "DMRS": "Deceptive",
}


def csv_row_to_features(row: dict) -> np.ndarray:
    """
    실측 CSV 한 줄 → 8-dim feature vector.
    두 가지 CSV 형식 지원:
      (A) kpm_fdd_7modes.csv: pusch_snr, pucch_snr, dl_ok, dl_nok, dl_bler, cqi
      (B) kpm_collector.py: RSRP, PUCCH.SINR, DL.BLER, UL.BLER, CQI, DRB.UEThpDl
    """
    # 형식 감지: KPM 메트릭 이름이 있으면 (B)
    if "PUCCH.SINR" in row or "CQI" in row:
        # kpm_collector.py 형식 → extract_from_kpm과 동일 경로
        return extract_from_kpm(row)

    # (A) 기존 형식 — 빈 문자열 대응
    def _f(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    dl_ok = _f(row.get("dl_ok", 0))
    dl_nok = _f(row.get("dl_nok", 0))
    total = dl_ok + dl_nok
    nack_rate = dl_nok / total if total > 0 else 0.0

    return np.array([
        _f(row.get("pusch_snr", 0)),      # [0] rsrp_dbm (PUSCH SNR placeholder)
        -12.0,                             # [1] rsrq_db (미제공)
        _f(row.get("pucch_snr", 0)),      # [2] sinr_db (PUCCH.SINR)
        _f(row.get("dl_bler", 0)),        # [3] bler
        nack_rate,                         # [4] uci_nack_rate (파생)
        100.0,                             # [5] dl_throughput_mbps (미제공)
        _f(row.get("cqi", 50), 50),       # [6] cqi_mean (accumulator)
        nack_rate,                         # [7] harq_retx_rate (NACK과 동일)
    ], dtype=np.float32)


def load_real_csv(csv_path: str):
    """실측 CSV 로드 → (X: N×8 float32, labels: list[str] Stage1 5-class)"""
    import csv as _csv
    X_list, labels = [], []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            raw_label = row["label"]
            s1_label = _LABEL_TO_STAGE1.get(raw_label, raw_label)
            X_list.append(csv_row_to_features(row))
            labels.append(s1_label)
    return np.array(X_list, dtype=np.float32), labels
