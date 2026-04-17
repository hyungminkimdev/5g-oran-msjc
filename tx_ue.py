import uhd
import numpy as np
import time
import yaml
import os

# ─────────────────────────────────────────────
# Config 로드
# ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_ue_addr():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["network"]["nodes"]["instance1_ue"]["usrp"]["ip"]

# ─────────────────────────────────────────────
# UE 송신기
# ─────────────────────────────────────────────
def run_transmitter(addr, freq, rate, duration=10):
    print(f"[UE] USRP 연결 중: addr={addr}")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    usrp.set_tx_rate(rate)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(freq))
    usrp.set_tx_gain(70)
    usrp.set_tx_antenna("TX/RX", 0)
    print(f"[UE] 초기화 완료 — rate={rate/1e6:.0f} MHz, freq={freq/1e9:.1f} GHz, gain=70 dB, ant=TX/RX")

    # OFDM-like 신호 생성 (Random QAM + IFFT)
    n_fft = 1024
    data = (np.random.randn(n_fft) + 1j * np.random.randn(n_fft)).astype(np.complex64)
    ifft_data = np.fft.ifft(data)
    samples = np.tile(ifft_data, int((rate * duration) / n_fft))
    print(f"[UE] 신호 생성 완료 — {len(samples):,} samples ({duration}초 분량)")

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    streamer = usrp.get_tx_stream(st_args)

    # 다음 정수 초 + 2초 후 송신 시작 (동기화)
    start_time = usrp.get_time_now().get_real_secs() + 2.0
    metadata = uhd.types.TXMetadata()
    metadata.start_of_burst = True
    metadata.time_spec = uhd.types.TimeSpec(start_time)
    metadata.has_time_spec = True

    print(f"[UE] 송신 예약 완료 — T+{start_time:.3f}s 에 시작 (2초 후)")
    print(f"[UE] >>> 송신 시작 ({duration}초간, freq={freq/1e9:.1f} GHz)")

    streamer.send(samples, metadata)

    print(f"[UE] <<< 송신 완료")

if __name__ == "__main__":
    addr = load_ue_addr()
    print(f"[Config] UE USRP addr = {addr}\n")
    run_transmitter(addr, freq=3.5e9, rate=20e6)
