"""
I/Q Passive Capture — MSJC Phase 5
Instance-1 UE USRP에서 DL 주파수를 passive RX로 캡처.
gNB 신호 + Jammer 간섭이 합쳐진 raw I/Q를 수집한다.

사용법 (Instance-1에서 실행):
  # srsUE 중지 후:
  sudo python3 iq_capture.py --mode constant --n-snapshots 50 --outdir /tmp/iq_captures

  # 전 모드 자동 수집 (Instance-2에서 SSH로 제어):
  # → tools/collect_iq_all.sh 사용
"""

import uhd
import numpy as np
import time
import os
import argparse

# Band 3 FDD DL — gNB/Jammer와 동일
DL_FREQ = 1842.5e6
RATE = 23.04e6
N_SAMPLES = 128 * 128   # 16384 = Stage 3 입력 크기
RX_GAIN = 30             # passive RX gain (튜닝 가능)

def capture_snapshot(usrp, streamer, n_samples=N_SAMPLES):
    """단일 I/Q 스냅샷 캡처."""
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    cmd.num_samps = n_samples
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)

    buf = np.zeros((1, n_samples), dtype=np.complex64)
    meta = uhd.types.RXMetadata()
    collected = 0
    max_chunk = streamer.get_max_num_samps()

    while collected < n_samples:
        remaining = n_samples - collected
        chunk_size = min(remaining, max_chunk)
        temp = np.zeros((1, chunk_size), dtype=np.complex64)
        n_recv = streamer.recv(temp, meta)

        if meta.error_code == uhd.types.RXMetadataErrorCode.timeout:
            break
        if n_recv > 0:
            end = min(collected + n_recv, n_samples)
            buf[0, collected:end] = temp[0, :end - collected]
            collected = end

    return buf[0, :collected]


def main():
    parser = argparse.ArgumentParser(description="MSJC I/Q Passive Capture (Instance-1)")
    parser.add_argument("--mode", type=str, required=True,
                        help="재밍 모드 라벨 (Normal/Constant/Random/...)")
    parser.add_argument("--n-snapshots", type=int, default=50,
                        help="캡처할 스냅샷 수 (기본: 50)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="스냅샷 간 간격 초 (기본: 0.5)")
    parser.add_argument("--outdir", type=str, default="/tmp/iq_captures",
                        help="출력 디렉토리")
    parser.add_argument("--addr", type=str, default="192.168.114.2",
                        help="USRP 주소 (기본: Instance-1 UE USRP)")
    parser.add_argument("--freq", type=float, default=DL_FREQ,
                        help=f"RX 주파수 Hz (기본: {DL_FREQ/1e6:.1f} MHz)")
    parser.add_argument("--gain", type=float, default=RX_GAIN,
                        help=f"RX gain dB (기본: {RX_GAIN})")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mode_dir = os.path.join(args.outdir, args.mode)
    os.makedirs(mode_dir, exist_ok=True)

    print(f"[IQ Capture] USRP 연결: addr={args.addr}")
    usrp = uhd.usrp.MultiUSRP(
        f"addr={args.addr},master_clock_rate=184.32e6,num_recv_frames=512")
    usrp.set_rx_rate(RATE)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(args.freq))
    usrp.set_rx_gain(args.gain)
    usrp.set_rx_antenna("RX2", 0)

    actual_rate = usrp.get_rx_rate()
    actual_freq = usrp.get_rx_freq()
    print(f"[IQ Capture] rate={actual_rate/1e6:.2f} MHz, freq={actual_freq/1e6:.1f} MHz, "
          f"gain={args.gain} dB")

    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))

    print(f"[IQ Capture] {args.mode} 모드 {args.n_snapshots}개 캡처 시작...")
    for i in range(args.n_snapshots):
        iq = capture_snapshot(usrp, streamer, N_SAMPLES)
        rssi = 10 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-12)

        fname = os.path.join(mode_dir, f"{args.mode}_{i:04d}.npy")
        np.save(fname, iq)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{args.mode}] {i+1}/{args.n_snapshots} | "
                  f"samples={len(iq)} | RSSI={rssi:+.1f} dB")

        time.sleep(args.interval)

    print(f"[IQ Capture] 완료: {args.n_snapshots}개 → {mode_dir}/")


if __name__ == "__main__":
    main()
