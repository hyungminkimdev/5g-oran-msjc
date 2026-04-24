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
def simulate_kpi_chunk(label: str) -> np.ndarray:
    """
    Generate a synthetic 8-dim KPI feature vector for a given jamming label.

    Labels:
      "Normal"     — clean channel, good KPIs
      "Constant"   — wideband AWGN jammer → all KPIs degraded
      "Random"     — intermittent bursts → moderate degradation with high variance
      "Reactive"   — TDD-synchronized jammer → periodic KPI dips
      "Deceptive"  — OFDM-mimicking jammer → subtle CQI/SINR degradation
      "PSS/SSS"    — sync channel jammer → CQI→0, RSRP drop, cell search fails
      "PDCCH"      — control channel jammer → high BLER, NACK spike
      "DMRS"       — pilot jammer → HARQ spikes, moderate SINR drop

    Returns:
        np.ndarray of shape (8,), dtype float32
    """
    if label == "Normal":
        return np.array([
            np.random.uniform(-70, -85),     # rsrp: typical good range
            np.random.uniform(-8, -15),      # rsrq: normal quality
            np.random.uniform(15, 35),       # sinr: good link
            np.random.uniform(0.0, 0.05),    # bler: very low
            np.random.uniform(0.0, 0.03),    # uci_nack: very low
            np.random.uniform(50, 200),      # dl_tput: healthy throughput
            np.random.uniform(10, 15),       # cqi: high quality
            np.random.uniform(0.0, 0.05),    # harq_retx: minimal
        ], dtype=np.float32)

    elif label == "Constant":
        return np.array([
            np.random.uniform(-100, -130),   # rsrp: severe drop
            np.random.uniform(-28, -40),     # rsrq: terrible
            np.random.uniform(-15, 0),       # sinr: negative
            np.random.uniform(0.7, 1.0),     # bler: very high
            np.random.uniform(0.8, 1.0),     # uci_nack: almost all NACK
            np.random.uniform(0, 3),         # dl_tput: near zero
            np.random.uniform(0, 2),         # cqi: minimum
            np.random.uniform(0.8, 1.0),     # harq_retx: maximum
        ], dtype=np.float32)

    elif label == "Random":
        # Intermittent bursts → high variance, moderate average degradation
        burst_on = np.random.rand() < 0.4
        if burst_on:
            return np.array([
                np.random.uniform(-95, -115),
                np.random.uniform(-22, -35),
                np.random.uniform(-5, 8),
                np.random.uniform(0.4, 0.9),
                np.random.uniform(0.3, 0.8),
                np.random.uniform(2, 30),
                np.random.uniform(1, 6),
                np.random.uniform(0.5, 0.9),
            ], dtype=np.float32)
        else:
            return np.array([
                np.random.uniform(-75, -90),
                np.random.uniform(-10, -18),
                np.random.uniform(10, 25),
                np.random.uniform(0.02, 0.15),
                np.random.uniform(0.02, 0.1),
                np.random.uniform(30, 120),
                np.random.uniform(7, 13),
                np.random.uniform(0.03, 0.15),
            ], dtype=np.float32)

    elif label == "Reactive":
        # TDD-synchronized — jams during specific slots
        # Simulates periodic degradation: ~40% of the time is jammed
        jammed_slot = np.random.rand() < 0.4
        if jammed_slot:
            return np.array([
                np.random.uniform(-95, -120),
                np.random.uniform(-25, -38),
                np.random.uniform(-10, 3),
                np.random.uniform(0.6, 1.0),
                np.random.uniform(0.7, 1.0),
                np.random.uniform(0, 8),
                np.random.uniform(0, 3),
                np.random.uniform(0.7, 1.0),
            ], dtype=np.float32)
        else:
            return np.array([
                np.random.uniform(-72, -88),
                np.random.uniform(-9, -16),
                np.random.uniform(12, 28),
                np.random.uniform(0.01, 0.08),
                np.random.uniform(0.01, 0.05),
                np.random.uniform(40, 150),
                np.random.uniform(9, 14),
                np.random.uniform(0.01, 0.08),
            ], dtype=np.float32)

    elif label == "Deceptive":
        # OFDM-mimicking: looks like legitimate signal but with subtle anomalies
        # SINR is moderately degraded, CQI drops, but RSRP/RSRQ look semi-normal
        return np.array([
            np.random.uniform(-78, -95),     # rsrp: slight drop (jammer adds power)
            np.random.uniform(-14, -22),     # rsrq: moderate degradation
            np.random.uniform(3, 15),        # sinr: reduced but not crashed
            np.random.uniform(0.1, 0.4),     # bler: elevated
            np.random.uniform(0.05, 0.25),   # uci_nack: somewhat elevated
            np.random.uniform(15, 60),       # dl_tput: reduced
            np.random.uniform(4, 9),         # cqi: mid-range drop
            np.random.uniform(0.1, 0.4),     # harq_retx: elevated
        ], dtype=np.float32)

    elif label == "PSS/SSS":
        # Sync channel jamming → UE can't find/maintain cell
        # CQI → 0, RSRP erratic, throughput zero, high BLER
        return np.array([
            np.random.uniform(-110, -140),   # rsrp: can't decode PSS → very low
            np.random.uniform(-30, -43),     # rsrq: worst case
            np.random.uniform(-20, -5),      # sinr: very poor in sync band
            np.random.uniform(0.85, 1.0),    # bler: nearly total
            np.random.uniform(0.9, 1.0),     # uci_nack: all NACK (can't schedule)
            np.random.uniform(0, 1),         # dl_tput: effectively zero
            np.random.uniform(0, 1),         # cqi: zero (can't measure channel)
            np.random.uniform(0.9, 1.0),     # harq_retx: max
        ], dtype=np.float32)

    elif label == "PDCCH":
        # Control channel jammer → can't decode DCI → scheduling destroyed
        # BLER spikes, NACK spikes, but RSRP may look OK (jammer targets CORESET only)
        return np.array([
            np.random.uniform(-80, -100),    # rsrp: moderate (jammer narrowband)
            np.random.uniform(-15, -25),     # rsrq: degraded
            np.random.uniform(0, 12),        # sinr: reduced in CORESET region
            np.random.uniform(0.6, 0.95),    # bler: very high (DCI decode fails)
            np.random.uniform(0.7, 0.95),    # uci_nack: high (no grants decoded)
            np.random.uniform(0, 10),        # dl_tput: near zero (no scheduling)
            np.random.uniform(2, 6),         # cqi: low but not zero (data symbols OK)
            np.random.uniform(0.6, 0.9),     # harq_retx: high
        ], dtype=np.float32)

    elif label == "DMRS":
        # Pilot signal jammer → channel estimation ruined
        # HARQ retransmission spikes, SINR drops moderately, CQI unreliable
        return np.array([
            np.random.uniform(-82, -100),    # rsrp: moderate degradation
            np.random.uniform(-14, -24),     # rsrq: degraded
            np.random.uniform(2, 14),        # sinr: moderate drop
            np.random.uniform(0.3, 0.7),     # bler: elevated (bad channel est)
            np.random.uniform(0.2, 0.5),     # uci_nack: moderate
            np.random.uniform(5, 40),        # dl_tput: reduced
            np.random.uniform(3, 8),         # cqi: unreliable mid-range
            np.random.uniform(0.5, 0.85),    # harq_retx: high (retransmit everything)
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown KPI simulation label: {label}")
