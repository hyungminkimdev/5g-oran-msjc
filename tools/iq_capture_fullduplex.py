#!/usr/bin/env python3
"""
I/Q Full-Duplex Capture — Instance-3 USRP X310
Instance-1 USRP 불가 시 대안: Instance-3에서 TX(재밍)+RX(캡처) 동시 수행

  Channel 0 TX/RX: Jammer 송신 (기존 jammer.py 신호 생성 재사용)
  Channel 1 RX2:   gNB DL + 재밍 간섭 수신

사용법 (Instance-3에서 실행):
  sudo python3 iq_capture_fullduplex.py --mode PSS --n-snapshots 50
  sudo python3 iq_capture_fullduplex.py --mode Normal --n-snapshots 50  # RX only
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uhd
import numpy as np
import time
import argparse
import threading

from jammer import (
    build_constant_buffer, build_random_frame, build_reactive_frame,
    build_deceptive_frame, build_pss_frame, build_pdcch_frame, build_dmrs_frame,
    SAMPLES_PER_SLOT, SAMPLES_PER_FRAME, RATE, DL_FREQ
)

N_SAMPLES = 128 * 128  # 16384 = Stage 3 입력 크기
RX_GAIN = 20
USRP_ADDR = "192.168.115.2"

# 모드명 → jammer build 함수
BUILDERS = {
    "constant":  lambda amp: build_constant_buffer(SAMPLES_PER_SLOT, amp),
    "random":    lambda amp: build_random_frame(amp, duty_cycle=0.3),
    "reactive":  lambda amp: build_reactive_frame(amp),
    "deceptive": lambda amp: build_deceptive_frame(amp),
    "pss":       lambda amp: build_pss_frame(amp),
    "pdcch":     lambda amp: build_pdcch_frame(amp),
    "dmrs":      lambda amp: build_dmrs_frame(amp),
}


def tx_worker(tx_stream, mode, amplitude, stop_event):
    """TX 연속 전송 쓰레드"""
    metadata = uhd.types.TXMetadata()
    build_fn = BUILDERS[mode]
    count = 0
    while not stop_event.is_set():
        buf = build_fn(amplitude)
        tx_stream.send(buf, metadata)
        count += 1
    metadata.end_of_burst = True
    tx_stream.send(np.zeros(1, dtype=np.complex64), metadata)
    print(f"  [TX] 종료: {count} bursts")


def capture_snapshots(rx_stream, n_snapshots, outdir, label, interval=0.5):
    """RX 스냅샷 → .npy 저장"""
    os.makedirs(outdir, exist_ok=True)

    for i in range(n_snapshots):
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        cmd.num_samps = N_SAMPLES
        cmd.stream_now = True
        rx_stream.issue_stream_cmd(cmd)

        buf = np.zeros(N_SAMPLES, dtype=np.complex64)
        meta = uhd.types.RXMetadata()
        collected = 0
        max_chunk = rx_stream.get_max_num_samps()

        while collected < N_SAMPLES:
            remaining = N_SAMPLES - collected
            chunk = min(remaining, max_chunk)
            temp = np.zeros((1, chunk), dtype=np.complex64)
            n_recv = rx_stream.recv(temp, meta)
            if meta.error_code == uhd.types.RXMetadataErrorCode.timeout:
                break
            if n_recv > 0:
                end = min(collected + n_recv, N_SAMPLES)
                buf[collected:end] = temp[0, :end - collected]
                collected = end

        np.save(os.path.join(outdir, f"{label}_{i:04d}.npy"), buf[:collected])

        if (i + 1) % 10 == 0 or i == 0:
            rssi = 10 * np.log10(np.mean(np.abs(buf[:collected])**2) + 1e-12)
            print(f"  [{label}] {i+1}/{n_snapshots} | samples={collected} | RSSI={rssi:+.1f} dB")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Instance-3 Full-Duplex I/Q Capture")
    parser.add_argument("--mode", required=True,
                        help="Normal/Constant/Random/Reactive/Deceptive/PSS/PDCCH/DMRS")
    parser.add_argument("--n-snapshots", type=int, default=50)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--outdir", default="/tmp/iq_captures")
    parser.add_argument("--addr", default=USRP_ADDR)
    parser.add_argument("--tx-gain", type=int, default=0)
    parser.add_argument("--amplitude", type=float, default=0.6)
    parser.add_argument("--rx-gain", type=float, default=RX_GAIN)
    args = parser.parse_args()

    mode_lower = args.mode.lower()
    is_normal = mode_lower == "normal"
    mode_dir = os.path.join(args.outdir, args.mode)

    print(f"[Capture] USRP addr={args.addr}, mode={args.mode}")
    usrp = uhd.usrp.MultiUSRP(
        f"addr={args.addr},master_clock_rate=184.32e6,"
        f"num_recv_frames=512,num_send_frames=512")

    # RX: channel 1 / RX2 포트 (TX와 물리적 격리)
    rx_ch = 1
    usrp.set_rx_rate(RATE, rx_ch)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(DL_FREQ), rx_ch)
    usrp.set_rx_gain(args.rx_gain, rx_ch)
    usrp.set_rx_antenna("RX2", rx_ch)

    rx_sa = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_sa.channels = [rx_ch]
    rx_stream = usrp.get_rx_stream(rx_sa)

    print(f"  RX: ch{rx_ch}/RX2, gain={args.rx_gain}dB, freq={DL_FREQ/1e6:.1f}MHz")

    stop_event = None
    tx_thread = None

    if not is_normal:
        # TX: channel 0 / TX-RX 포트
        tx_ch = 0
        usrp.set_tx_rate(RATE, tx_ch)
        usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(DL_FREQ), tx_ch)
        usrp.set_tx_gain(args.tx_gain, tx_ch)
        usrp.set_tx_antenna("TX/RX", tx_ch)

        tx_sa = uhd.usrp.StreamArgs("fc32", "sc16")
        tx_sa.channels = [tx_ch]
        tx_stream = usrp.get_tx_stream(tx_sa)

        print(f"  TX: ch{tx_ch}/TX-RX, gain={args.tx_gain}dB, amp={args.amplitude}")

        stop_event = threading.Event()
        tx_thread = threading.Thread(
            target=tx_worker,
            args=(tx_stream, mode_lower, args.amplitude, stop_event),
            daemon=True)
        tx_thread.start()
        time.sleep(1)  # TX 안정화
    else:
        print("  TX: 없음 (Normal — RX only)")

    # RX 캡처
    capture_snapshots(rx_stream, args.n_snapshots, mode_dir, args.mode, args.interval)

    # TX 정리
    if tx_thread:
        stop_event.set()
        tx_thread.join(timeout=5)

    total = len([f for f in os.listdir(mode_dir) if f.endswith('.npy')])
    print(f"[Capture] 완료: {total}개 → {mode_dir}/")


if __name__ == "__main__":
    main()
