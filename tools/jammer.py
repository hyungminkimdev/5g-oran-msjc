"""
MSJC Jammer — 5G O-RAN FDD 재밍 테스트 도구
CCI xG Testbed 전용 (Band 3 FDD, SCS 15kHz, 20MHz BW)

환경:
  - gNB: srate=23.04 MHz, SCS=15kHz (μ=0), FDD Band 3
  - DL: 1842.5 MHz (ARFCN 368500)
  - UL: 1747.5 MHz
  - 106 PRB, FFT=1536, CORESET0 index=6
"""

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
    "constant":  "지속 광대역 AWGN — 채널 완전 포화",
    "random":    "랜덤 펄스 재밍 — 불규칙 on/off 버스트",
    "reactive":  "주기적 재밍 — FDD DL 슬롯 타이밍 기반 간헐 공격",
    "deceptive": "OFDM 위장 재밍 — 합법 신호처럼 보이는 간섭",
    "pss":       "PSS/SSS 타겟 — SSB 위치 중앙 RB에 집중 (narrowband)",
    "pdcch":     "PDCCH 타겟 — CORESET 심볼 광대역 공격",
    "dmrs":      "DMRS 타겟 — 빗살 패턴 파일럿 재밍, 채널 추정 교란",
}

# ─────────────────────────────────────────────
# 5G NR FDD 파라미터 (Band 3, μ=0, SCS=15kHz)
# gNB srate=23.04 MHz와 동일하게 맞춤
# ─────────────────────────────────────────────
RATE               = 23.04e6                      # gNB와 동일
N_FFT              = 1536                          # 23.04 MHz / 15 kHz = 1536
CP_NORMAL          = int(N_FFT * 144 / 2048)      # Normal CP = 108 samples
CP_FIRST           = int(N_FFT * 160 / 2048)      # 첫 심볼 CP = 120 samples
SAMPLES_PER_SYMBOL = N_FFT + CP_NORMAL             # 1644 samples
SAMPLES_PER_SLOT   = 14 * SAMPLES_PER_SYMBOL       # 14 심볼/슬롯 = 23016 samples (~1ms)
SAMPLES_PER_SUBFRAME = SAMPLES_PER_SLOT            # FDD μ=0: 1 slot = 1 subframe = 1ms
SAMPLES_PER_FRAME  = 10 * SAMPLES_PER_SUBFRAME     # 10ms radio frame

# FDD: DL/UL이 다른 주파수 → 모든 슬롯이 DL (DL freq에서)
# SSB 위치: SCS=15kHz → SSB period=20ms, 프레임 내 특정 심볼
# Band 3 SSB: Case A (SCS 15kHz), 심볼 인덱스 {2,8} + {16,22} (i_ssb 0~3)
SSB_SYMBOLS_IN_SLOT = [2, 8]   # 슬롯 0 내 SSB 시작 심볼 (Case A, L=4)
SSB_SLOTS = [0, 1]             # SSB가 포함되는 슬롯 (half-frame 0)

# CORESET0: index=6, SCS=15kHz → 1 OFDM symbol, 48 RBs
CORESET0_SYMBOLS = 1
CORESET0_RBS     = 48  # 48 RBs = 576 subcarriers

# PRB/서브캐리어
N_PRB  = 106                   # 20 MHz @ 15 kHz
N_SC   = N_PRB * 12            # 1272 active subcarriers

# 주파수 설정
DL_FREQ = 1842.5e6             # Band 3 DL center
UL_FREQ = 1747.5e6             # Band 3 UL center


# ─────────────────────────────────────────────
# 신호 생성 헬퍼
# ─────────────────────────────────────────────
def awgn(n):
    """복소 AWGN 생성 (단위 분산)"""
    return (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * np.float32(0.7071)

def ofdm_symbol(subcarriers=None, amplitude=1.0):
    """
    OFDM 심볼 생성: 주파수 도메인 → IFFT → CP 추가
    subcarriers: 사용할 서브캐리어 인덱스 (None=전체)
    """
    freq = np.zeros(N_FFT, dtype=np.complex64)
    if subcarriers is None:
        # 전체 active 서브캐리어 사용
        half = N_SC // 2
        center = N_FFT // 2
        freq[center - half:center + half] = awgn(N_SC) * amplitude
    else:
        freq[subcarriers] = awgn(len(subcarriers)) * amplitude
    time_domain = np.fft.ifft(freq).astype(np.complex64)
    # CP 추가
    cp = time_domain[-CP_NORMAL:]
    return np.concatenate([cp, time_domain])

def build_slot(symbol_fn, amplitude=1.0):
    """14개 OFDM 심볼로 1슬롯 생성"""
    symbols = []
    for _ in range(14):
        symbols.append(symbol_fn(amplitude=amplitude))
    return np.concatenate(symbols)[:SAMPLES_PER_SLOT]


# ─────────────────────────────────────────────
# 모드별 프레임/버퍼 빌더
# ─────────────────────────────────────────────

def build_constant_buffer(n_samples, amplitude=1.0):
    """연속 광대역 AWGN"""
    return awgn(n_samples) * amplitude

def build_random_burst(amplitude=1.0):
    """랜덤 길이 burst (1~5 OFDM 심볼)"""
    n_sym = np.random.randint(1, 6)
    burst = []
    for _ in range(n_sym):
        burst.append(ofdm_symbol(amplitude=amplitude))
    return np.concatenate(burst)

def build_reactive_frame(amplitude=1.0):
    """
    FDD 주기적 재밍: 1ms 슬롯 중 일부 심볼만 재밍.
    2슬롯 ON → 3슬롯 OFF → 2슬롯 ON → 3슬롯 OFF (10ms 프레임)
    → duty cycle ~40%, 주기적 패턴
    """
    frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.complex64)
    on_slots = [0, 1, 5, 6]  # 40% duty, 주기적
    for slot_idx in on_slots:
        start = slot_idx * SAMPLES_PER_SLOT
        end = start + SAMPLES_PER_SLOT
        frame[start:end] = awgn(SAMPLES_PER_SLOT) * amplitude
    return frame

def build_deceptive_frame(amplitude=1.0):
    """OFDM 위장 — 정상 OFDM 심볼처럼 생성"""
    symbols = []
    for _ in range(14):  # 1 슬롯
        symbols.append(ofdm_symbol(amplitude=amplitude))
    return np.concatenate(symbols)

def build_pss_frame(amplitude=1.0):
    """
    PSS/SSS 타겟 — SSB 위치의 중앙 RB만 재밍.
    Band 3 Case A: SSB는 프레임 내 슬롯 0,1의 심볼 2,8,16,22에 위치.
    SSB는 20 RB (240 SC) 폭, 중앙 DC 주변.
    20ms 주기로 반복 → 10ms 프레임당 1세트.
    """
    frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.complex64)

    # SSB 서브캐리어: 중앙 240 SC (20 RBs)
    center = N_FFT // 2
    ssb_half = 120  # 240/2
    ssb_sc = np.arange(center - ssb_half, center + ssb_half)

    # 슬롯 0,1의 SSB 심볼 위치에 재밍
    for slot_idx in SSB_SLOTS:
        for sym_idx in SSB_SYMBOLS_IN_SLOT:
            sym_start = slot_idx * SAMPLES_PER_SLOT + sym_idx * SAMPLES_PER_SYMBOL
            # SSB 심볼: 4개 연속 (PSS+SSS+PBCH)
            for s in range(4):
                offset = sym_start + s * SAMPLES_PER_SYMBOL
                if offset + SAMPLES_PER_SYMBOL <= SAMPLES_PER_FRAME:
                    sym = ofdm_symbol(subcarriers=ssb_sc, amplitude=amplitude * 3.0)
                    frame[offset:offset + len(sym)] = sym[:min(len(sym), SAMPLES_PER_FRAME - offset)]

    return frame

def build_pdcch_frame(amplitude=1.0):
    """
    PDCCH 타겟 — 매 슬롯의 CORESET 심볼(첫 1-2개)을 광대역 재밍.
    CORESET0: 48 RBs = 576 SC, 첫 1 심볼 (index=6 설정 기준)
    모든 DL 슬롯(FDD=전부)에 적용.
    """
    frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.complex64)

    # CORESET 서브캐리어: 48 RB = 576 SC
    center = N_FFT // 2
    coreset_half = CORESET0_RBS * 12 // 2  # 288
    coreset_sc = np.arange(center - coreset_half, center + coreset_half)

    for slot_idx in range(10):  # FDD: 10 슬롯/프레임
        slot_start = slot_idx * SAMPLES_PER_SLOT
        # CORESET: 첫 CORESET0_SYMBOLS 심볼
        for sym_idx in range(CORESET0_SYMBOLS):
            offset = slot_start + sym_idx * SAMPLES_PER_SYMBOL
            if offset + SAMPLES_PER_SYMBOL <= SAMPLES_PER_FRAME:
                sym = ofdm_symbol(subcarriers=coreset_sc, amplitude=amplitude * 4.0)
                frame[offset:offset + len(sym)] = sym[:min(len(sym), SAMPLES_PER_FRAME - offset)]

    return frame

def build_dmrs_frame(amplitude=1.0):
    """
    DMRS 타겟 — PDSCH DMRS 위치의 빗살 패턴 재밍.
    Type 1 DMRS: 매 2번째 서브캐리어 (comb-2), 심볼 2,3 (추가 위치)
    모든 슬롯에 적용.
    """
    frame = np.zeros(SAMPLES_PER_FRAME, dtype=np.complex64)

    # DMRS comb-2: 짝수 또는 홀수 서브캐리어
    comb_offset = np.random.randint(0, 2)
    dmrs_sc = np.arange(comb_offset, N_FFT, 2)  # 매 2번째 SC

    # DMRS 심볼 위치: 슬롯 내 심볼 2 (단일 심볼 DMRS 기본 위치)
    dmrs_symbols = [2, 3]  # additional DMRS 포함

    for slot_idx in range(10):
        slot_start = slot_idx * SAMPLES_PER_SLOT
        for sym_idx in dmrs_symbols:
            offset = slot_start + sym_idx * SAMPLES_PER_SYMBOL
            if offset + SAMPLES_PER_SYMBOL <= SAMPLES_PER_FRAME:
                sym = ofdm_symbol(subcarriers=dmrs_sc, amplitude=amplitude * 3.0)
                frame[offset:offset + len(sym)] = sym[:min(len(sym), SAMPLES_PER_FRAME - offset)]

    return frame


# ─────────────────────────────────────────────
# 메인 재머
# ─────────────────────────────────────────────
def run_jammer(addr, mode="constant", freq=DL_FREQ, rate=RATE, gain=0,
               amplitude=1.0):
    print(f"[Jammer] USRP 연결 중: addr={addr}")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    usrp.set_tx_rate(rate)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(freq))
    usrp.set_tx_gain(gain)
    usrp.set_tx_antenna("TX/RX", 0)

    actual_rate = usrp.get_tx_rate()
    actual_freq = usrp.get_tx_freq()
    actual_gain = usrp.get_tx_gain()

    print(f"[Jammer] 초기화 완료:")
    print(f"  rate={actual_rate/1e6:.2f} MHz (요청: {rate/1e6:.2f})")
    print(f"  freq={actual_freq/1e6:.1f} MHz (요청: {freq/1e6:.1f})")
    print(f"  gain={actual_gain:.0f} dB, amplitude={amplitude:.3f}")
    print(f"  FFT={N_FFT}, SCS=15kHz, 106PRB, FDD Band 3")

    streamer = usrp.get_tx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    metadata = uhd.types.TXMetadata()

    print(f"\n[Jammer] >>> 공격 시작: mode={mode} — {MODE_DESC[mode]}")
    print("[Jammer]     종료하려면 Ctrl+C\n")

    tx_count = 0
    t_start = time.time()

    try:
        while True:
            if mode == "constant":
                buf = build_constant_buffer(SAMPLES_PER_SLOT, amplitude)
                streamer.send(buf, metadata)

            elif mode == "random":
                burst = build_random_burst(amplitude)
                streamer.send(burst, metadata)
                time.sleep(np.random.uniform(0.002, 0.02))

            elif mode == "reactive":
                frame = build_reactive_frame(amplitude)
                streamer.send(frame, metadata)

            elif mode == "deceptive":
                buf = build_deceptive_frame(amplitude)
                streamer.send(buf, metadata)

            elif mode == "pss":
                frame = build_pss_frame(amplitude)
                streamer.send(frame, metadata)

            elif mode == "pdcch":
                frame = build_pdcch_frame(amplitude)
                streamer.send(frame, metadata)

            elif mode == "dmrs":
                frame = build_dmrs_frame(amplitude)
                streamer.send(frame, metadata)

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
    parser = argparse.ArgumentParser(description="MSJC Jammer — 5G FDD Band 3 재밍 테스트")
    parser.add_argument("--mode", choices=list(MODE_DESC.keys()),
                        default=None, help="공격 모드 (미지정 시 인터랙티브 선택)")
    parser.add_argument("--freq", type=float, default=DL_FREQ,
                        help=f"TX 주파수 Hz (기본: {DL_FREQ/1e6:.1f} MHz = Band 3 DL)")
    parser.add_argument("--rate", type=float, default=RATE,
                        help=f"샘플 레이트 Hz (기본: {RATE/1e6:.2f} MHz = gNB 동일)")
    parser.add_argument("--gain", type=int, default=0,
                        help="TX 게인 dB (기본: 0)")
    parser.add_argument("--amplitude", type=float, default=1.0,
                        help="신호 amplitude 스케일링 0.0~1.0 (기본: 1.0)")
    args = parser.parse_args()

    addr = load_jammer_addr()
    print(f"[Config] Jammer USRP addr = {addr}")
    print(f"[Config] FDD Band 3: DL={DL_FREQ/1e6:.1f} MHz, UL={UL_FREQ/1e6:.1f} MHz")
    print(f"[Config] Rate={RATE/1e6:.2f} MHz, FFT={N_FFT}, SCS=15kHz")

    mode = args.mode
    if mode is None:
        print("\n[Jammer] 공격 모드를 선택하세요:")
        for key, name in MODES.items():
            print(f"  {key}) {name:<12} — {MODE_DESC[name]}")
        choice = input("\n모드 번호 또는 이름 입력: ").strip().lower()
        mode = MODES.get(choice, choice)
        if mode not in MODE_DESC:
            print(f"[ERROR] 알 수 없는 입력. constant로 시작합니다.")
            mode = "constant"

    run_jammer(addr, mode=mode, freq=args.freq, rate=args.rate, gain=args.gain,
               amplitude=args.amplitude)
