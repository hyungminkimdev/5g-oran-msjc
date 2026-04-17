"""
MSJC Pipeline Runner — Stage 1 (MLP) + Stage 2 (KSVM) + Stage 3 (MobileNetV3)
USRP X310 @ Classifier/RIC node (Instance-2)

전체 파이프라인:

  I/Q 수신 → [Stage 1 MLP: 5클래스]
                 │
                 ├─ Constant / Random / Reactive → 공격 즉시 확정 (끝)
                 ├─ Deceptive → [Stage 3 MobileNetV3] → PSS/PDCCH/DMRS/Generic 정밀 분류
                 │
                 └─ Normal → [Stage 2 KSVM: 이진 재검사]
                                 │
                                 ├─ Normal → CLEAN 확정 (끝)
                                 └─ Attack → FN 포착 → [Stage 3 MobileNetV3] → 정밀 분류

Phase 1: I/Q Sanity Check (5초간 수신 상태 확인)
Phase 2: Stage 1+2+3 계단식 탐지 루프 (Ctrl+C까지 연속)
"""

import sys
import os
import time
import yaml
import numpy as np
import uhd
from datetime import datetime

# Stage 1: MLP (5클래스)
from stage1_mlp import (
    load_model as load_stage1_model,
    classify   as stage1_classify,
    extract_features,
    LABELS as S1_LABELS,
)

# Stage 2: KSVM (이진 재검사)
from stage2_ksvm import (
    load_model as load_stage2_model,
    recheck    as stage2_recheck,
)

# Stage 3: MobileNetV3 (공격 유형 정밀 분류)
from stage3_mobilenet import (
    load_model as load_stage3_model,
    classify   as stage3_classify,
    LABELS as S3_LABELS,
)

# InfluxDB 로깅
from influx_logger import InfluxLogger

# ─────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    usrp_addr = cfg["network"]["nodes"]["instance2_classifier"]["usrp"]["ip"]
    return cfg, usrp_addr


# ─────────────────────────────────────────────
# 1. USRP 초기화
# ─────────────────────────────────────────────
RATE    = 20e6
FREQ    = 3.5e9
GAIN    = 45
ANTENNA = "RX2"
CHUNK   = 128 * 128   # 16,384 samples ≈ 0.82 ms @ 20 MHz

def init_usrp(addr):
    print(f"[USRP] 연결 중: addr={addr}")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    usrp.set_rx_rate(RATE)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(FREQ))
    usrp.set_rx_gain(GAIN)
    usrp.set_rx_antenna(ANTENNA, 0)
    time.sleep(0.2)
    print(f"[USRP] 초기화 완료 — rate={RATE/1e6:.0f} MHz, freq={FREQ/1e9:.1f} GHz, "
          f"gain={GAIN} dB, ant={ANTENNA}")
    return usrp

def make_streamer(usrp):
    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)
    return streamer

def stop_stream(streamer):
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(cmd)


# ─────────────────────────────────────────────
# 2. Phase 1 — Sanity Check
# ─────────────────────────────────────────────
SANITY_DURATION     = 5      # 초
RSSI_WARN_THRESHOLD = -80.0  # dB

def run_sanity_check(streamer):
    print(f"\n{'='*60}")
    print(f"[Phase 1] I/Q Sanity Check ({SANITY_DURATION}초)")
    print(f"{'='*60}")

    buf  = np.zeros((1, CHUNK), dtype=np.complex64)
    meta = uhd.types.RXMetadata()

    rssi_list = []
    flat_list = []
    err_count = 0
    chunk_idx = 0
    t_end     = time.time() + SANITY_DURATION

    while time.time() < t_end:
        n = streamer.recv(buf, meta)
        if meta.error_code != uhd.types.RXMetadataErrorCode.none:
            err_count += 1
            print(f"  [UHD ERR] {meta.strerror()} (총 {err_count}회)")
            continue
        if n == 0:
            continue

        feat = extract_features(buf[0])
        rssi, flat = feat[0], feat[2]
        rssi_list.append(rssi)
        flat_list.append(flat)
        chunk_idx += 1

        print(f"  [Sanity] chunk #{chunk_idx:4d} | "
              f"RSSI: {rssi:+7.2f} dB | Flatness: {flat:.4f} | errors: {err_count}")

    if not rssi_list:
        print("\n[FAIL] I/Q 데이터 수신 실패 — USRP 연결 상태를 확인하세요.")
        return False

    rssi_arr = np.array(rssi_list)
    print(f"\n{'─'*60}")
    print(f"[Sanity 요약] 청크: {chunk_idx}개 | 에러: {err_count}회")
    print(f"  RSSI     — mean: {rssi_arr.mean():+.2f} dB, "
          f"min: {rssi_arr.min():+.2f}, max: {rssi_arr.max():+.2f}")
    print(f"  Flatness — mean: {np.mean(flat_list):.4f}")

    if rssi_arr.mean() < RSSI_WARN_THRESHOLD:
        print(f"\n[WARN] 평균 RSSI({rssi_arr.mean():.1f} dB) < {RSSI_WARN_THRESHOLD} dB")
        print("       신호 미수신 가능성 — UE/Jammer 동작 여부를 확인하세요.")
    else:
        print(f"\n[OK] I/Q 데이터 수신 확인 완료")
    return True


# ─────────────────────────────────────────────
# 3. Phase 2 — Stage 1 + 2 + 3 탐지 루프
# ─────────────────────────────────────────────
def run_detection_loop(streamer,
                       s1_model, s1_scaler, s1_device,
                       s2_pipeline=None,
                       s3_model=None, s3_device=None,
                       influx_logger=None):

    has_s2 = s2_pipeline is not None
    has_s3 = s3_model is not None

    print(f"\n{'='*80}")
    print("[Phase 2] MSJC 계단식 탐지 루프 (Ctrl+C 로 종료)")
    print(f"  Stage 1 (MLP)        : {' / '.join(S1_LABELS)}")
    print(f"  Stage 2 (KSVM)       : {'Normal 재검사 활성화' if has_s2 else '비활성화'}")
    print(f"  Stage 3 (MobileNetV3): {'Protocol-Aware 정밀 분류 활성화' if has_s3 else '비활성화'}")
    if has_s3:
        print(f"                         (Deceptive / FN 포착 → {' / '.join(S3_LABELS)})")
    print(f"{'='*80}\n")

    buf  = np.zeros((1, CHUNK), dtype=np.complex64)
    meta = uhd.types.RXMetadata()

    stats = {"total": 0, "attacks": 0, "normals": 0, "fn_caught": 0}

    while True:
        t0 = time.perf_counter()
        n  = streamer.recv(buf, meta)

        if meta.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f"[UHD ERR] {meta.strerror()} — skip")
            continue
        if n == 0:
            continue

        stats["total"] += 1
        iq = buf[0]

        # ── Stage 1: MLP 5클래스 ──
        s1_label, s1_conf, _ = stage1_classify(iq, s1_model, s1_scaler, s1_device)
        feat = extract_features(iq)
        rssi = feat[0]
        ts   = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if s1_label in ("Constant", "Random", "Reactive"):
            # ── 물리적 재밍 확정 — Stage 3 불필요 ──
            stats["attacks"] += 1
            total_ms = (time.perf_counter() - t0) * 1000
            print(f"[{ts}] RSSI: {rssi:+7.2f} dB | "
                  f"S1: {s1_label:<10s}({s1_conf:.2f}) | "
                  f"ATTACK CONFIRMED | {total_ms:.1f}ms")
            if influx_logger:
                influx_logger.log_detection(
                    rssi=rssi, spectral_flatness=feat[2],
                    s1_label=s1_label, s1_confidence=s1_conf,
                    final_verdict="ATTACK_CONFIRMED", latency_ms=total_ms)

        elif s1_label == "Deceptive":
            # ── Deceptive → Stage 3 Protocol-Aware 정밀 분류 ──
            stats["attacks"] += 1

            if has_s3:
                s3_label, s3_conf, _ = stage3_classify(iq, s3_model, s3_device)
                total_ms = (time.perf_counter() - t0) * 1000
                print(f"[{ts}] RSSI: {rssi:+7.2f} dB | "
                      f"S1: Deceptive ({s1_conf:.2f}) → "
                      f"S3: {s3_label:<18s}({s3_conf:.2f}) | "
                      f"PROTOCOL-AWARE | {total_ms:.1f}ms")
                if influx_logger:
                    influx_logger.log_detection(
                        rssi=rssi, spectral_flatness=feat[2],
                        s1_label=s1_label, s1_confidence=s1_conf,
                        s3_label=s3_label, s3_confidence=s3_conf,
                        final_verdict="PROTOCOL_AWARE", latency_ms=total_ms)
            else:
                total_ms = (time.perf_counter() - t0) * 1000
                print(f"[{ts}] RSSI: {rssi:+7.2f} dB | "
                      f"S1: Deceptive ({s1_conf:.2f}) | "
                      f"ATTACK CONFIRMED | {total_ms:.1f}ms")
                if influx_logger:
                    influx_logger.log_detection(
                        rssi=rssi, spectral_flatness=feat[2],
                        s1_label=s1_label, s1_confidence=s1_conf,
                        final_verdict="ATTACK_CONFIRMED", latency_ms=total_ms)

        elif has_s2:
            # ── Normal → Stage 2 재검사 ──
            is_attack, s2_conf, _ = stage2_recheck(iq, s2_pipeline)

            if is_attack:
                # ── FN 포착 → Stage 3 정밀 분류 ──
                stats["fn_caught"] += 1
                stats["attacks"]   += 1

                if has_s3:
                    s3_label, s3_conf, _ = stage3_classify(iq, s3_model, s3_device)
                    total_ms = (time.perf_counter() - t0) * 1000
                    print(f"[{ts}] RSSI: {rssi:+7.2f} dB | "
                          f"S1: Normal   ({s1_conf:.2f}) → "
                          f"S2: ATTACK   ({s2_conf:.2f}) → "
                          f"S3: {s3_label:<18s}({s3_conf:.2f}) | "
                          f"FN → PROTOCOL-AWARE | {total_ms:.1f}ms")
                    if influx_logger:
                        influx_logger.log_detection(
                            rssi=rssi, spectral_flatness=feat[2],
                            s1_label="Normal", s1_confidence=s1_conf,
                            s2_confidence=s2_conf,
                            s3_label=s3_label, s3_confidence=s3_conf,
                            final_verdict="FN_CAUGHT", fn_caught=True,
                            latency_ms=total_ms)
                else:
                    total_ms = (time.perf_counter() - t0) * 1000
                    print(f"[{ts}] RSSI: {rssi:+7.2f} dB | "
                          f"S1: Normal   ({s1_conf:.2f}) → "
                          f"S2: ATTACK   ({s2_conf:.2f}) | "
                          f"FN CAUGHT | {total_ms:.1f}ms")
                    if influx_logger:
                        influx_logger.log_detection(
                            rssi=rssi, spectral_flatness=feat[2],
                            s1_label="Normal", s1_confidence=s1_conf,
                            s2_confidence=s2_conf,
                            final_verdict="FN_CAUGHT", fn_caught=True,
                            latency_ms=total_ms)
            else:
                stats["normals"] += 1
                total_ms = (time.perf_counter() - t0) * 1000
                print(f"[{ts}] RSSI: {rssi:+7.2f} dB | "
                      f"S1: Normal   ({s1_conf:.2f}) → "
                      f"S2: Normal   ({s2_conf:.2f}) | "
                      f"CLEAN | {total_ms:.1f}ms")
                if influx_logger:
                    influx_logger.log_detection(
                        rssi=rssi, spectral_flatness=feat[2],
                        s1_label="Normal", s1_confidence=s1_conf,
                        s2_confidence=s2_conf,
                        final_verdict="CLEAN", latency_ms=total_ms)

        else:
            # ── Stage 2/3 비활성화 ──
            verdict = "CLEAN" if s1_label == "Normal" else "ATTACK_CONFIRMED"
            if s1_label == "Normal":
                stats["normals"] += 1
            else:
                stats["attacks"] += 1
            total_ms = (time.perf_counter() - t0) * 1000
            print(f"[{ts}] RSSI: {rssi:+7.2f} dB | "
                  f"S1: {s1_label:<10s}({s1_conf:.2f}) | "
                  f"{verdict} | {total_ms:.1f}ms")
            if influx_logger:
                influx_logger.log_detection(
                    rssi=rssi, spectral_flatness=feat[2],
                    s1_label=s1_label, s1_confidence=s1_conf,
                    final_verdict=verdict, latency_ms=total_ms)

        # 500청크마다 통계
        if stats["total"] % 500 == 0:
            fn_rate = stats["fn_caught"] / max(stats["normals"] + stats["fn_caught"], 1)
            print(f"\n  [STATS] 총: {stats['total']} | "
                  f"공격: {stats['attacks']} | 정상: {stats['normals']} | "
                  f"FN 포착: {stats['fn_caught']} ({fn_rate:.2%})\n")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MSJC Pipeline — Stage 1 (MLP) + Stage 2 (KSVM) + Stage 3 (MobileNetV3)")
    parser.add_argument("--no-stage2", action="store_true",
                        help="Stage 2 비활성화")
    parser.add_argument("--no-stage3", action="store_true",
                        help="Stage 3 비활성화 (공격 유형 정밀 분류 생략)")
    args = parser.parse_args()

    cfg, usrp_addr = load_config()
    print(f"[Config] USRP addr = {usrp_addr} (Classifier/RIC node)\n")

    # Stage 1 로드
    s1_model, s1_scaler, s1_device = load_stage1_model()

    # Stage 2 로드
    s2_pipeline = None
    if not args.no_stage2:
        s2_pipeline = load_stage2_model()

    # Stage 3 로드
    s3_model, s3_device = None, None
    if not args.no_stage3:
        s3_model, s3_device = load_stage3_model()

    # InfluxDB 로거 초기화
    influx_cfg = cfg.get("data", {}).get("influxdb", {})
    influx_logger = InfluxLogger(influx_cfg)

    # USRP 초기화
    usrp     = init_usrp(usrp_addr)
    streamer = make_streamer(usrp)

    try:
        ok = run_sanity_check(streamer)
        if not ok:
            sys.exit(1)

        run_detection_loop(streamer,
                           s1_model, s1_scaler, s1_device,
                           s2_pipeline=s2_pipeline,
                           s3_model=s3_model, s3_device=s3_device,
                           influx_logger=influx_logger)

    except KeyboardInterrupt:
        print("\n\n[INFO] 사용자 종료 — 스트림 정지 중...")
        stop_stream(streamer)
        influx_logger.close()
        print("[INFO] 종료 완료.")
