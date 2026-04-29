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

    vec = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    return vec


# ─────────────────────────────────────────────
# 2. Synthetic KPI Simulation
# ─────────────────────────────────────────────
_ATTACK_PROFILES = {
    #                   rsrp          rsrq          sinr         bler          nack          tput         cqi          harq
    "Normal":         [(-70,-85),   (-8,-15),    (15,35),     (0.0,0.05),  (0.0,0.03),  (50,200),    (10,15),     (0.0,0.05)],
    "Constant":       [(-100,-130), (-28,-40),   (-15,0),     (0.7,1.0),   (0.8,1.0),   (0,3),       (0,2),       (0.8,1.0)],
    "Deceptive":      [(-78,-95),   (-14,-22),   (3,15),      (0.1,0.4),   (0.05,0.25), (15,60),     (4,9),       (0.1,0.4)],
    "PSS/SSS":        [(-110,-140), (-30,-43),   (-20,-5),    (0.85,1.0),  (0.9,1.0),   (0,1),       (0,1),       (0.9,1.0)],
    "PDCCH":          [(-80,-100),  (-15,-25),   (0,12),      (0.6,0.95),  (0.7,0.95),  (0,10),      (2,6),       (0.6,0.9)],
    "DMRS":           [(-82,-100),  (-14,-24),   (2,14),      (0.3,0.7),   (0.2,0.5),   (5,40),      (3,8),       (0.5,0.85)],
}
# Burst/intermittent attacks: (jammed_profile, clean_profile, jam_probability)
_BURST_PROFILES = {
    "Random":   ("Random_jam",   "Random_clean",   0.4),
    "Reactive": ("Reactive_jam", "Reactive_clean",  0.4),
}
_ATTACK_PROFILES["Random_jam"]    = [(-95,-115), (-22,-35), (-5,8),   (0.4,0.9),  (0.3,0.8),  (2,30),   (1,6),   (0.5,0.9)]
_ATTACK_PROFILES["Random_clean"]  = [(-75,-90),  (-10,-18), (10,25),  (0.02,0.15),(0.02,0.1), (30,120), (7,13),  (0.03,0.15)]
_ATTACK_PROFILES["Reactive_jam"]  = [(-95,-120), (-25,-38), (-10,3),  (0.6,1.0),  (0.7,1.0),  (0,8),    (0,3),   (0.7,1.0)]
_ATTACK_PROFILES["Reactive_clean"]= [(-72,-88),  (-9,-16),  (12,28),  (0.01,0.08),(0.01,0.05),(40,150), (9,14),  (0.01,0.08)]


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
