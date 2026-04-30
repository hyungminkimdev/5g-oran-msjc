"""
MSJC xApp — 5G O-RAN Near-RT RIC xApp
FlexRIC E2SM-KPM 기반 재밍 탐지 및 분류

동작 흐름:
  FlexRIC RIC ←E2SM-KPM→ srsRAN gNB
      │
      └─ on_kpm_indication()
            → kpi_feature_extractor.extract_from_kpm()
            → MSJC Pipeline (Stage1 → Stage2 → Stage3)
            → InfluxDB logging
            → (optional) E2SM-RC control

FlexRIC 미연결 시:
  - MockFlexRIC: 합성 KPI 이벤트를 100ms 간격으로 재생

실행:
  python3 xapp_msjc.py [--no-stage2] [--no-stage3] [--no-rc] [--mock-ric]
"""

import argparse
import os
import sys
import time
import threading
import yaml
import numpy as np
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ─────────────────────────────────────────────
# FlexRIC import (graceful)
# xapp_sdk: FlexRIC Python SDK (SWIG-generated)
# KPM SM Python 바인딩 미지원 → MAC SM 지표로 KPI 벡터 구성
# ─────────────────────────────────────────────
try:
    sys.path.insert(0, '/usr/local/lib/python3/dist-packages/xapp_sdk/')
    import xapp_sdk as flexric
    _FLEXRIC_AVAILABLE = True
except ImportError:
    _FLEXRIC_AVAILABLE = False


# ─────────────────────────────────────────────
# MockFlexRIC — synthetic KPM event replay
# ─────────────────────────────────────────────
class MockFlexRIC:
    """
    FlexRIC 미설치 환경에서 합성 KPM Indication을 100ms 간격으로 재생.
    10-event 사이클:
      Normal, Normal, Constant, Reactive, Deceptive,
      Normal, Normal, PDCCH, Normal, Normal
    """

    _CYCLE = [
        "Normal", "Normal", "Constant", "Reactive", "Deceptive",
        "Normal", "Normal", "PDCCH", "Normal", "Normal",
    ]

    def __init__(self, report_period_ms: int = 100):
        self._period = report_period_ms / 1000.0
        self._callback = None
        self._running  = False
        self._thread   = None
        self._idx      = 0

    def subscribe_kpm(self, callback, e2_node_id: str = "mock-gnb-001"):
        self._callback = callback
        self._e2_node  = e2_node_id

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[{_ts()}] [MockFlexRIC] 시작 — KPM 이벤트 재생 100ms 간격")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        from kpi_feature_extractor import simulate_kpi_chunk, FEATURE_NAMES

        while self._running:
            label = self._CYCLE[self._idx % len(self._CYCLE)]
            self._idx += 1

            feat_vec = simulate_kpi_chunk(label)
            kpm_msg  = {name: float(feat_vec[i]) for i, name in enumerate(FEATURE_NAMES)}
            kpm_msg["_mock_label"] = label  # debug tag

            hdr = {"e2_node_id": self._e2_node, "timestamp_ms": int(time.time() * 1000)}

            if self._callback:
                self._callback(hdr, kpm_msg)

            time.sleep(self._period)

    def send_rc_control(self, e2_node_id: str, action: str):
        print(f"[{_ts()}] [MockFlexRIC] RC 제어 (stub): node={e2_node_id} action={action}")


# ─────────────────────────────────────────────
# RealFlexRIC wrapper (FlexRIC SDK가 있을 때)
# ─────────────────────────────────────────────
class _KPMCallback(flexric.kpm_cb if _FLEXRIC_AVAILABLE else object):
    """
    KPM SM indication 콜백 — xapp_sdk SWIG director 파생 클래스.
    br-flexric API: ind.msg.frm_1 (FORMAT_1_INDICATION_MESSAGE)
    """
    def __init__(self, user_fn, node_id: str):
        if _FLEXRIC_AVAILABLE:
            flexric.kpm_cb.__init__(self)
        self._fn      = user_fn
        self._node_id = node_id

    def handle(self, ind):
        msg = {}
        if ind.msg.type == flexric.FORMAT_1_INDICATION_MESSAGE:
            frm1 = ind.msg.frm_1
            for meas_data in frm1.meas_data_lst:
                for j, rec in enumerate(meas_data.meas_record_lst):
                    if j < len(frm1.meas_info_lst):
                        info = frm1.meas_info_lst[j]
                        name = info.meas_type.name if info.meas_type.type == flexric.NAME_MEAS_TYPE else str(info.meas_type.id)
                    else:
                        name = f"metric_{j}"
                    if rec.value == flexric.REAL_MEAS_VALUE:
                        msg[name] = rec.real_val
                    elif rec.value == flexric.INTEGER_MEAS_VALUE:
                        msg[name] = rec.int_val

        hdr = {"e2_node_id": self._node_id,
               "timestamp_ms": int(time.time() * 1000)}
        try:
            self._fn(hdr, msg)
        except Exception as e:
            print(f"[{_ts()}] [KPM CB] 파이프라인 오류: {e}")


class RealFlexRIC:
    """
    FlexRIC Python SDK (xapp_sdk) 래퍼.
    br-flexric API: report_kpm_sm(node.id, interval, actions, callback)
    Action definition format 1 (cell-level), E2SM-KPM v2.03/v3.
    """

    # srsRAN DU가 지원하는 KPM 메트릭 (PHY 확장 포함)
    KPM_ACTIONS = ['DRB.UEThpDl', 'DRB.UEThpUl', 'RRU.PrbTotDl', 'RRU.PrbTotUl',
                   'CQI', 'RSRP', 'DL.BLER', 'UL.BLER', 'PUCCH.SINR']

    def __init__(self, ric_addr: str, ric_port: int, report_period_ms: int,
                 conf_path: str = "/tmp/xapp_kpm_msjc.conf"):
        self._ric_addr   = ric_addr
        self._ric_port   = ric_port
        self._period_ms  = report_period_ms
        self._conf_path  = conf_path
        self._callback   = None
        self._stop_evt   = threading.Event()
        self._handles    = []
        self._cb_refs    = []

    def _write_conf(self):
        """FlexRIC xApp config 파일 생성"""
        actions_str = ",\n".join(f'            {{ name = "{a}" }}' for a in self.KPM_ACTIONS)
        conf = f'''SM_DIR = "/usr/local/lib/flexric/"
Name = "xApp"
NearRT_RIC_IP = "{self._ric_addr}"
E42_Port = {self._ric_port}

Sub_ORAN_SM_List = (
    {{ name = "KPM", time = {self._period_ms},
      format = 1,
      ran_type = "ngran_gNB_DU",
      actions = (
{actions_str}
            )
    }}
)
'''
        with open(self._conf_path, 'w') as f:
            f.write(conf)

    def subscribe_kpm(self, callback, e2_node_id: str = ""):
        self._callback = callback

    def start(self):
        self._write_conf()

        # 1. nearRT-RIC 연결 (br-flexric API: init with config)
        flexric.init([sys.argv[0], '-c', self._conf_path])

        # 2. E2 노드(gNB) 연결 대기
        conn = []
        wait_printed = False
        while not conn and not self._stop_evt.is_set():
            conn = flexric.conn_e2_nodes()
            if not conn:
                if not wait_printed:
                    print(f"[{_ts()}] [FlexRIC] E2 노드 대기 중 (gNB E2AP 연결 필요)...")
                    wait_printed = True
                time.sleep(2.0)

        if not conn:
            return

        # 3. Interval 매핑 (ms → enum)
        interval = flexric.Interval_ms_1000  # default
        try:
            interval_map = {10: flexric.Interval_ms_10, 100: flexric.Interval_ms_100, 1000: flexric.Interval_ms_1000}
            if self._period_ms in interval_map:
                interval = interval_map[self._period_ms]
        except AttributeError:
            pass

        # 4. 각 E2 노드에 KPM 구독
        for node in conn:
            mcc     = node.id.plmn.mcc
            mnc     = node.id.plmn.mnc
            node_id = f"gnb-mcc{mcc}-mnc{mnc}"
            ntype   = flexric.get_e2ap_ngran_name(node.id.type)
            print(f"[{_ts()}] [FlexRIC] E2 노드: {ntype} PLMN MCC={mcc} MNC={mnc}")

            cb = _KPMCallback(self._callback, node_id)
            self._cb_refs.append(cb)

            # br-flexric API: report_kpm_sm(node_id, interval, actions, callback)
            handle = flexric.report_kpm_sm(node.id, interval, self.KPM_ACTIONS, cb)
            self._handles.append(handle)
            print(f"[{_ts()}] [FlexRIC] KPM 구독 완료: handle={handle}, "
                  f"metrics={self.KPM_ACTIONS}, period={self._period_ms}ms")

        if not self._handles:
            raise RuntimeError("[FlexRIC] 모든 E2 노드의 KPM 구독 실패")

        print(f"[{_ts()}] [FlexRIC] E2SM-KPM 수신 대기 중 ({self._period_ms}ms 주기)...")

        # 5. 종료 신호 대기
        try:
            self._stop_evt.wait()
        finally:
            # os._exit로 종료 — rm_report_kpm_sm의 subscription delete timeout assertion 회피
            # (FlexRIC br-flexric 버그: sync_ui timeout on delete response)
            import os
            os._exit(0)

    def stop(self):
        self._stop_evt.set()

    def send_rc_control(self, e2_node_id: str, action: str):
        print(f"[{_ts()}] [FlexRIC] RC 제어 (stub): node={e2_node_id} action={action}")


# ─────────────────────────────────────────────
# MSJC xApp
# ─────────────────────────────────────────────
class MSJCxApp:

    def __init__(self, config: dict, args: argparse.Namespace):
        self._cfg      = config
        self._args     = args
        self._lock     = threading.RLock()
        self._stop_evt = threading.Event()

        oran_cfg = config.get("oran", {})
        e2_cfg   = oran_cfg.get("e2", {})

        self._ric_addr        = e2_cfg.get("ric_addr", "127.0.0.1")
        self._ric_port        = e2_cfg.get("ric_port", 36422)  # E42 port
        self._report_period   = e2_cfg.get("report_period_ms", 100)
        self._rc_enabled      = e2_cfg.get("e2sm_rc_enabled", False) and not args.no_rc

        self._use_mock        = args.mock_ric or not _FLEXRIC_AVAILABLE
        self._has_stage2      = not args.no_stage2
        self._has_stage3      = not args.no_stage3

        sdr_cfg = config.get("sdr", {})
        self._snapshot_on_s3  = sdr_cfg.get("snapshot_on_stage3", True)

        upd_cfg = config.get("model_update", {})
        self._reload_interval = upd_cfg.get("check_interval_sec", 60)

        # Model file mtimes for hot-reload
        self._model_mtimes = {}
        self._model_paths  = {
            "stage1": os.path.join(os.path.dirname(__file__), "stage1_mlp.pth"),
            "scaler":  os.path.join(os.path.dirname(__file__), "stage1_scaler.pkl"),
            "stage2": os.path.join(os.path.dirname(__file__), "stage2_ksvm.pkl"),
            "stage3": os.path.join(os.path.dirname(__file__), "stage3_mobilenet.pth"),
        }

        # KPM 히스토리 버퍼 (Stage 2 sliding window용)
        from stage2_ksvm import WINDOW_SIZE
        self._kpm_window_size = WINDOW_SIZE
        self._kpm_history = deque(maxlen=WINDOW_SIZE)

        # Runtime stats
        self._stats = {
            "total": 0, "attacks": 0, "normals": 0,
            "fn_caught": 0, "fn_window": [],
        }

        self._s1_model = self._s1_scaler = self._s1_device = None
        self._s2_pipeline = None
        self._s3_model = self._s3_device = None
        self._influx_logger = None
        self._ric = None
        self._iq_snap = None

    # ── 모델 로드 ──────────────────────────────
    def _load_models(self):
        from stage1_mlp import load_model as load_s1
        self._s1_model, self._s1_scaler, self._s1_device = load_s1()
        self._model_mtimes["stage1"] = self._mtime("stage1")
        self._model_mtimes["scaler"]  = self._mtime("scaler")

        if self._has_stage2:
            from stage2_ksvm import load_model as load_s2
            self._s2_pipeline = load_s2()
            self._model_mtimes["stage2"] = self._mtime("stage2")

        if self._has_stage3:
            from stage3_mobilenet import load_model as load_s3
            self._s3_model, self._s3_device = load_s3()
            self._model_mtimes["stage3"] = self._mtime("stage3")

    def _mtime(self, key: str) -> float:
        p = self._model_paths.get(key, "")
        return os.path.getmtime(p) if os.path.exists(p) else 0.0

    # ── 핫 리로드 ──────────────────────────────
    def _reload_models_if_updated(self):
        changed = False

        s1_mtime = self._mtime("stage1")
        sc_mtime = self._mtime("scaler")
        if (s1_mtime != self._model_mtimes.get("stage1") or
                sc_mtime != self._model_mtimes.get("scaler")):
            changed = True
            with self._lock:
                import torch
                from stage1_mlp import JammingMLP, NUM_FEATURES
                import pickle
                device = self._s1_device
                model  = JammingMLP().to(device)
                model.load_state_dict(
                    torch.load(self._model_paths["stage1"],
                               map_location=device, weights_only=True))
                model.eval()
                with open(self._model_paths["scaler"], "rb") as f:
                    scaler = pickle.load(f)
                self._s1_model  = model
                self._s1_scaler = scaler
                self._model_mtimes["stage1"] = s1_mtime
                self._model_mtimes["scaler"]  = sc_mtime
            print(f"[{_ts()}] [xApp] Stage1 MLP 핫 리로드 완료")

        if self._has_stage2:
            s2_mtime = self._mtime("stage2")
            if s2_mtime != self._model_mtimes.get("stage2"):
                changed = True
                with self._lock:
                    import pickle
                    with open(self._model_paths["stage2"], "rb") as f:
                        self._s2_pipeline = pickle.load(f)
                    self._model_mtimes["stage2"] = s2_mtime
                print(f"[{_ts()}] [xApp] Stage2 KSVM 핫 리로드 완료")

        if self._has_stage3:
            s3_mtime = self._mtime("stage3")
            if s3_mtime != self._model_mtimes.get("stage3"):
                changed = True
                with self._lock:
                    import torch
                    from stage3_mobilenet import build_model, NUM_CLASSES
                    device = self._s3_device
                    model  = build_model().to(device)
                    model.load_state_dict(
                        torch.load(self._model_paths["stage3"],
                                   map_location=device, weights_only=True))
                    model.eval()
                    self._s3_model = model
                    self._model_mtimes["stage3"] = s3_mtime
                print(f"[{_ts()}] [xApp] Stage3 MobileNetV3 핫 리로드 완료")

        return changed

    def _hot_reload_loop(self):
        while not self._stop_evt.is_set():
            self._stop_evt.wait(self._reload_interval)
            if not self._stop_evt.is_set():
                self._reload_models_if_updated()

    # ── IQ Snapshot 초기화 ─────────────────────
    def _init_iq_snapshot(self):
        if not self._snapshot_on_s3:
            return
        usrp_addr = (self._cfg.get("network", {})
                         .get("nodes", {})
                         .get("instance2_classifier", {})
                         .get("usrp", {})
                         .get("ip", "192.168.116.2"))
        from iq_snapshot import IQSnapshot
        self._iq_snap = IQSnapshot(usrp_addr)

    # ── InfluxDB 로거 초기화 ───────────────────
    def _init_influx(self):
        influx_cfg = self._cfg.get("data", {}).get("influxdb", {})
        from influx_logger import InfluxLogger
        self._influx_logger = InfluxLogger(influx_cfg)

    # ── E2SM-KPM Indication 콜백 ───────────────
    def _on_kpm_indication(self, hdr: dict, msg: dict):
        t0 = time.perf_counter()

        from kpi_feature_extractor import extract_from_kpm
        features = extract_from_kpm(msg)

        # KPM 히스토리 버퍼에 저장 (Stage 2 sliding window용)
        self._kpm_history.append(features.copy())

        verdict = self._run_pipeline(features)
        verdict["latency_ms"] = (time.perf_counter() - t0) * 1000

        # 상태 업데이트
        self._stats["total"] += 1
        if verdict["final_verdict"] == "CLEAN":
            self._stats["normals"] += 1
        else:
            self._stats["attacks"] += 1
            if verdict.get("fn_caught"):
                self._stats["fn_caught"] += 1

        # FN rate 윈도우 (1시간 = 36,000 샘플 @ 100ms)
        self._stats["fn_window"].append(1 if verdict.get("fn_caught") else 0)
        if len(self._stats["fn_window"]) > 36000:
            self._stats["fn_window"].pop(0)

        # 출력
        sinr = float(features[2])
        bler = float(features[3])
        s1_label = verdict.get("s1_label", "?")
        s1_conf  = verdict.get("s1_confidence", 0.0)
        fv       = verdict["final_verdict"]
        lat      = verdict["latency_ms"]
        print(f"[{_ts()}] SINR:{sinr:+.1f}dB BLER:{bler:.3f} | "
              f"S1:{s1_label}({s1_conf:.2f}) | {fv} | {lat:.1f}ms")

        # InfluxDB 로깅
        if self._influx_logger:
            log_kw = {
                "rssi":              float(features[0]),
                "spectral_flatness": float(features[3]),
                "s1_label":          verdict.get("s1_label", ""),
                "s1_confidence":     verdict.get("s1_confidence", 0.0),
                "s2_confidence":     verdict.get("s2_confidence", 0.0),
                "s3_label":          verdict.get("s3_label", ""),
                "s3_confidence":     verdict.get("s3_confidence", 0.0),
                "final_verdict":     verdict["final_verdict"],
                "fn_caught":         bool(verdict.get("fn_caught", False)),
                "latency_ms":        verdict["latency_ms"],
            }
            self._influx_logger.log_detection(**log_kw)

        # E2SM-RC 제어
        if (self._rc_enabled and
                verdict["final_verdict"] != "CLEAN" and
                self._ric is not None):
            e2_node = hdr.get("e2_node_id", "unknown")
            action  = self._verdict_to_rc_action(verdict)
            self._send_rc_control(e2_node, action)

        # 500샘플마다 통계 출력
        if self._stats["total"] % 500 == 0:
            self._print_stats()

        # ClearML FN rate 모니터링
        self._check_fn_rate_retrain()

    # ── MSJC 파이프라인 ────────────────────────
    def _run_pipeline(self, features: np.ndarray) -> dict:
        with self._lock:
            from stage1_mlp import classify as s1_classify
            s1_label, s1_conf, _ = s1_classify(
                features, self._s1_model, self._s1_scaler, self._s1_device)

            if s1_label in ("Constant", "Random", "Reactive"):
                return {
                    "s1_label": s1_label, "s1_confidence": s1_conf,
                    "final_verdict": "ATTACK_CONFIRMED",
                }

            if s1_label == "Deceptive":
                if self._has_stage3:
                    s3_label, s3_conf = self._run_stage3(features)
                    return {
                        "s1_label": s1_label, "s1_confidence": s1_conf,
                        "s3_label": s3_label, "s3_confidence": s3_conf,
                        "final_verdict": "PROTOCOL_AWARE",
                    }
                return {
                    "s1_label": s1_label, "s1_confidence": s1_conf,
                    "final_verdict": "ATTACK_CONFIRMED",
                }

            # s1_label == "Normal" → Stage 2 sliding window 재검사
            if self._has_stage2 and len(self._kpm_history) >= self._kpm_window_size:
                from stage2_ksvm import recheck as s2_recheck
                window = np.array(list(self._kpm_history))  # (W, 8)
                is_attack, s2_conf, _ = s2_recheck(window, self._s2_pipeline)

                if is_attack:
                    if self._has_stage3:
                        s3_label, s3_conf = self._run_stage3(features)
                        return {
                            "s1_label": s1_label, "s1_confidence": s1_conf,
                            "s2_confidence": s2_conf,
                            "s3_label": s3_label, "s3_confidence": s3_conf,
                            "final_verdict": "FN_CAUGHT", "fn_caught": True,
                        }
                    return {
                        "s1_label": s1_label, "s1_confidence": s1_conf,
                        "s2_confidence": s2_conf,
                        "final_verdict": "FN_CAUGHT", "fn_caught": True,
                    }

                return {
                    "s1_label": s1_label, "s1_confidence": s1_conf,
                    "s2_confidence": s2_conf,
                    "final_verdict": "CLEAN",
                }

            # Stage 2 비활성화 또는 window 미충족 → Normal 확정
            return {
                "s1_label": s1_label, "s1_confidence": s1_conf,
                "final_verdict": "CLEAN",
            }

    def _run_stage3(self, features: np.ndarray) -> tuple:
        """Stage 3 실행. I/Q 스냅샷 우선, 없으면 KPI 히트맵 사용."""
        from stage3_mobilenet import classify as s3_classify

        iq = None
        if self._iq_snap is not None:
            try:
                iq = self._iq_snap.capture()
            except Exception as e:
                print(f"[{_ts()}] [xApp] IQ 캡처 실패: {e} — KPI 히트맵 fallback")

        if iq is not None:
            s3_label, s3_conf, _ = s3_classify(iq, self._s3_model, self._s3_device)
        else:
            iq_fake = self._kpi_to_fake_iq(features)
            s3_label, s3_conf, _ = s3_classify(iq_fake, self._s3_model, self._s3_device)

        return s3_label, s3_conf

    def _kpi_to_fake_iq(self, features: np.ndarray) -> np.ndarray:
        """
        KPI 벡터를 Stage3용 I/Q 유사 신호로 변환.
        sinr, bler, cqi 등을 신호 특성에 매핑해 스펙트로그램 입력 생성.
        """
        n = 128 * 128
        sinr    = float(features[2])
        bler    = float(features[3])
        cqi     = float(features[6])

        noise_level = max(0.01, 1.0 - (sinr + 20) / 60.0)
        sig = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
        sig *= noise_level

        if cqi < 3:
            n_fft = 1024
            center_filt = np.zeros(n_fft, dtype=np.complex64)
            c = n_fft // 2
            half = 72
            center_filt[c - half: c + half] = 1.0
            frames = []
            for i in range(int(np.ceil(n / n_fft))):
                chunk = sig[i * n_fft: (i + 1) * n_fft]
                if len(chunk) < n_fft:
                    chunk = np.pad(chunk, (0, n_fft - len(chunk)))
                frames.append(np.fft.ifft(np.fft.fft(chunk) * center_filt).astype(np.complex64))
            sig = np.concatenate(frames)[:n]
        elif bler > 0.5:
            period = 10000
            slot_len = int(period * 0.3)
            for start in range(0, n, period):
                end = min(start + slot_len, n)
                if end > start:
                    sig[start:end] *= 5.0

        return sig.astype(np.complex64)

    # ── E2SM-RC 제어 ───────────────────────────
    def _verdict_to_rc_action(self, verdict: dict) -> str:
        fv = verdict.get("final_verdict", "")
        s3 = verdict.get("s3_label", "")
        if fv == "ATTACK_CONFIRMED":
            return "INCREASE_TX_POWER"
        if s3 == "PSS/SSS":
            return "CELL_RESELECT"
        if s3 == "PDCCH":
            return "CORESET_RECONFIGURE"
        if s3 == "DMRS":
            return "PILOT_BOOST"
        return "INCREASE_TX_POWER"

    def _send_rc_control(self, e2_node_id: str, action: str):
        if self._ric is not None:
            self._ric.send_rc_control(e2_node_id, action)

    # ── 통계 출력 ──────────────────────────────
    def _print_stats(self):
        fn_window = self._stats["fn_window"]
        fn_rate   = sum(fn_window) / max(len(fn_window), 1)
        total     = self._stats["total"]
        attacks   = self._stats["attacks"]
        normals   = self._stats["normals"]
        fn_caught = self._stats["fn_caught"]
        print(f"\n  [STATS] 총: {total} | 공격: {attacks} | 정상: {normals} | "
              f"FN 포착: {fn_caught} | FN율(1h): {fn_rate:.2%}\n")

    def _check_fn_rate_retrain(self):
        fn_window = self._stats["fn_window"]
        if len(fn_window) < 100:
            return
        fn_rate = sum(fn_window) / len(fn_window)
        threshold = (self._cfg.get("clearml", {})
                         .get("fn_rate_retrain_threshold", 0.05))
        if fn_rate > threshold:
            self._trigger_clearml_retrain(fn_rate)
            self._stats["fn_window"].clear()

    def _trigger_clearml_retrain(self, fn_rate: float):
        try:
            from clearml import Task
            cfg = self._cfg.get("clearml", {})
            if not cfg.get("enabled", False):
                return
            task = Task.create(
                project_name=cfg.get("project_name", "msjc-5g-oran"),
                task_name=f"msjc-retrain-fn{fn_rate:.3f}-{int(time.time())}",
                task_type=Task.TaskTypes.training,
            )
            task.set_parameter("trigger_fn_rate", fn_rate)
            queue = cfg.get("queue_name", "default")
            Task.enqueue(task, queue_name=queue)
            print(f"[{_ts()}] [ClearML] 재학습 큐 등록: FN율={fn_rate:.2%} → {queue}")
        except ImportError:
            pass
        except Exception as e:
            print(f"[{_ts()}] [ClearML] 재학습 큐 등록 실패: {e}")

    # ── 시작 / 종료 ────────────────────────────
    def start(self):
        print(f"[{_ts()}] [xApp] MSJC xApp 시작")
        print(f"[{_ts()}] [xApp] Stage2: {'활성화' if self._has_stage2 else '비활성화'} | "
              f"Stage3: {'활성화' if self._has_stage3 else '비활성화'} | "
              f"RC: {'활성화' if self._rc_enabled else '비활성화'}")

        self._load_models()
        self._init_influx()
        self._init_iq_snapshot()

        # 핫 리로드 백그라운드 스레드
        reload_thread = threading.Thread(
            target=self._hot_reload_loop, daemon=True, name="hot-reload")
        reload_thread.start()

        if self._use_mock:
            print(f"[{_ts()}] [xApp] MockFlexRIC 모드 사용 "
                  f"({'--mock-ric 강제' if self._args.mock_ric else 'FlexRIC 미설치'})")
            ric = MockFlexRIC(self._report_period)
            ric.subscribe_kpm(self._on_kpm_indication)
            self._ric = ric
            ric.start()
            try:
                self._stop_evt.wait()
            except KeyboardInterrupt:
                pass
            ric.stop()

        else:
            ric = RealFlexRIC(self._ric_addr, self._ric_port, self._report_period)
            ric.subscribe_kpm(self._on_kpm_indication)
            self._ric = ric

            try:
                ric.start()  # blocking
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print(f"[{_ts()}] [xApp] E2 연결 실패: {e} — MockFlexRIC로 전환")
                ric.stop()
                mock = MockFlexRIC(self._report_period)
                mock.subscribe_kpm(self._on_kpm_indication)
                self._ric = mock
                mock.start()
                try:
                    self._stop_evt.wait()
                except KeyboardInterrupt:
                    pass
                mock.stop()
            else:
                ric.stop()

        self.stop()

    def stop(self):
        self._stop_evt.set()
        if self._influx_logger:
            self._influx_logger.close()
        print(f"[{_ts()}] [xApp] 종료 완료")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MSJC xApp — 5G O-RAN Near-RT RIC jamming detection xApp")
    parser.add_argument("--no-stage2", action="store_true",
                        help="Stage 2 KSVM 비활성화")
    parser.add_argument("--no-stage3", action="store_true",
                        help="Stage 3 MobileNetV3 비활성화")
    parser.add_argument("--no-rc", action="store_true",
                        help="E2SM-RC 제어 비활성화")
    parser.add_argument("--mock-ric", action="store_true",
                        help="FlexRIC 설치 여부 무관하게 MockFlexRIC 강제 사용")
    args = parser.parse_args()

    cfg = load_config()
    app = MSJCxApp(cfg, args)

    try:
        app.start()
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] [xApp] 사용자 종료 요청")
        app.stop()
