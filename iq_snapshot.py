"""
IQ Snapshot — MSJC Project
On-demand I/Q capture for Stage 3 MobileNetV3 spectrogram analysis.

Used by xapp_msjc.py when the MSJC pipeline detects a Deceptive or FN case
and needs a raw I/Q snapshot for protocol-aware spectrogram classification.

Thread-safe: uses a Lock to prevent concurrent USRP access.
Does NOT hold the USRP open between captures.
"""

import threading
import numpy as np
from datetime import datetime


class IQSnapshot:
    """
    On-demand I/Q capture from USRP X310.

    Connects to USRP, captures exactly n_samples, then disconnects.
    Thread-safe via internal Lock.
    """

    def __init__(self, usrp_addr: str, rate: float = 20e6,
                 freq: float = 3.5e9, gain: float = 45):
        self._addr = usrp_addr
        self._rate = rate
        self._freq = freq
        self._gain = gain
        self._lock = threading.Lock()

    def capture(self, n_samples: int = 128 * 128) -> np.ndarray:
        """
        Capture exactly n_samples from the USRP.

        Returns:
            np.ndarray of shape (n_samples,), dtype complex64

        Raises:
            RuntimeError: if UHD is not available or capture fails
        """
        with self._lock:
            return self._capture_locked(n_samples)

    def _capture_locked(self, n_samples: int) -> np.ndarray:
        try:
            import uhd
        except ImportError:
            raise RuntimeError(
                "[IQSnapshot] UHD Python bindings not installed. "
                "Install via: sudo apt install python3-uhd"
            )

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] [IQSnapshot] 캡처 시작: addr={self._addr}, "
              f"n_samples={n_samples}")

        usrp = uhd.usrp.MultiUSRP(f"addr={self._addr}")
        usrp.set_rx_rate(self._rate)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self._freq))
        usrp.set_rx_gain(self._gain)
        usrp.set_rx_antenna("RX2", 0)

        streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
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
            temp_buf = np.zeros((1, chunk_size), dtype=np.complex64)
            n_recv = streamer.recv(temp_buf, meta)

            if meta.error_code == uhd.types.RXMetadataErrorCode.timeout:
                break
            if meta.error_code not in (
                uhd.types.RXMetadataErrorCode.none,
                uhd.types.RXMetadataErrorCode.overflow,
            ):
                print(f"[IQSnapshot] UHD error: {meta.strerror()}")
                break

            if n_recv > 0:
                end = min(collected + n_recv, n_samples)
                actual = end - collected
                buf[0, collected:end] = temp_buf[0, :actual]
                collected = end

        # Stop stream
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        streamer.issue_stream_cmd(stop_cmd)

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        rssi = 10 * np.log10(np.mean(np.abs(buf[0, :collected]) ** 2) + 1e-12)
        print(f"[{ts}] [IQSnapshot] 캡처 완료: {collected}/{n_samples} samples, "
              f"RSSI={rssi:+.2f} dB")

        return buf[0, :collected]


if __name__ == "__main__":
    import argparse
    import yaml
    import os

    parser = argparse.ArgumentParser(
        description="MSJC IQ Snapshot — 단발성 I/Q 캡처 테스트")
    parser.add_argument("--addr", type=str, default=None,
                        help="USRP 주소 (미지정 시 config.yaml에서 읽기)")
    parser.add_argument("--n-samples", type=int, default=128 * 128,
                        help="캡처할 샘플 수 (기본: 16384)")
    args = parser.parse_args()

    if args.addr:
        addr = args.addr
    else:
        cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        addr = cfg["network"]["nodes"]["instance2_classifier"]["usrp"]["ip"]

    snap = IQSnapshot(addr)
    iq = snap.capture(args.n_samples)

    rssi = 10 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-12)
    print(f"\n[Result] Captured {len(iq)} samples")
    print(f"[Result] RSSI: {rssi:+.2f} dB")
    print(f"[Result] Power std: {np.std(np.abs(iq)**2):.6f}")
