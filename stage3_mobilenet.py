"""
Stage 3 MobileNetV3 — MSJC Project
224×224 스펙트로그램 기반 Protocol-Aware 재밍 정밀 분류

진입 조건:
  - Stage 1(MLP)이 "Deceptive"로 판정한 경우
  - Stage 2(KSVM)가 False Negative를 포착한 경우
  (Constant/Random/Reactive는 Stage 1에서 확정, Stage 3 불필요)

출력 클래스 (4):
  0 = PSS/SSS Jamming    — 중앙 6-12 RB 집중, 셀 접속/동기화 차단
  1 = PDCCH Jamming      — 슬롯 첫 1-3 심볼 CORESET 타겟, 스케줄링 파괴
  2 = DMRS Jamming       — 빗살(comb) 패턴 파일럿 재밍, 채널 추정 교란
  3 = Generic Deceptive  — 프로토콜 특정 타겟 없는 광대역 OFDM 위장

스펙트로그램 시그니처:
  PSS/SSS  — 주파수축 중앙에 집중된 전력 스파이크
  PDCCH    — 시간축 초반에 걸친 수직 줄무늬 (광대역)
  DMRS     — 주파수축 빗살(comb) 패턴 (등간격 서브캐리어)
  Generic  — 전 대역에 걸친 균일한 OFDM 구조
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v3_small
from scipy import signal

from stage1_mlp import RATE, CHUNK

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stage3_mobilenet.pth")

LABELS = ["PSS/SSS", "PDCCH", "DMRS", "Generic Deceptive"]
LABEL_IDX = {name: i for i, name in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

IMG_SIZE   = 224    # 논문 스펙
N_FFT_OFDM = 1024  # 시뮬레이션용 OFDM FFT 크기


# ─────────────────────────────────────────────
# 1. I/Q → 224×224 스펙트로그램 변환
# ─────────────────────────────────────────────
def iq_to_spectrogram_224(iq: np.ndarray) -> np.ndarray:
    """
    I/Q 청크 → 224×224 dB 스펙트로그램 (정규화 0~1)
    """
    nperseg  = 256
    noverlap = max(1, min(nperseg - int(len(iq) / IMG_SIZE), nperseg - 1))

    _, _, Sxx = signal.spectrogram(
        iq, fs=RATE, nperseg=nperseg, noverlap=noverlap,
        return_onesided=False
    )
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)

    vmin, vmax = Sxx_db.min(), Sxx_db.max()
    if vmax - vmin > 1e-6:
        Sxx_norm = (Sxx_db - vmin) / (vmax - vmin)
    else:
        Sxx_norm = np.zeros_like(Sxx_db)

    tensor  = torch.from_numpy(Sxx_norm).float().unsqueeze(0).unsqueeze(0)
    resized = nn.functional.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE),
                                        mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()


# ─────────────────────────────────────────────
# 2. Protocol-Aware I/Q 시뮬레이션
#    jammer.py의 실제 공격 패턴을 재현
# ─────────────────────────────────────────────
def _awgn(n: int) -> np.ndarray:
    return (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

def _ofdm_like(n: int) -> np.ndarray:
    syms  = _awgn(N_FFT_OFDM)
    frame = np.fft.ifft(syms).astype(np.complex64)
    return np.tile(frame, int(np.ceil(n / N_FFT_OFDM)))[:n]


def _simulate_freq_domain(n: int, gain: float, fill_fn) -> np.ndarray:
    """주파수 도메인 시뮬레이션 공통 루프: fill_fn(freq_array) 호출 후 IFFT."""
    frames = []
    for _ in range(int(np.ceil(n / N_FFT_OFDM))):
        freq = np.zeros(N_FFT_OFDM, dtype=np.complex64)
        fill_fn(freq, gain)
        freq += _awgn(N_FFT_OFDM) * gain * 0.05  # 배경 잡음
        frames.append(np.fft.ifft(freq).astype(np.complex64))
    return np.concatenate(frames)[:n]


def simulate_attack(attack_type: str, gain_factor: float = 1.0) -> np.ndarray:
    """
    Protocol-Aware I/Q 시뮬레이션 (PSS/SSS, PDCCH, DMRS, Generic Deceptive).
    """
    n = CHUNK

    if attack_type == "PSS/SSS":
        def fill(freq, g):
            c = N_FFT_OFDM // 2
            n_sc = np.random.randint(72, 145)
            half = n_sc // 2
            freq[c - half: c - half + n_sc] = _awgn(n_sc) * g * 5.0
        return _simulate_freq_domain(n, gain_factor, fill)

    elif attack_type == "DMRS":
        comb = np.random.choice([4, 6])
        offset = np.random.randint(0, comb)
        def fill(freq, g):
            idx = np.arange(offset, N_FFT_OFDM, comb)
            freq[idx] = _awgn(len(idx)) * g * 5.0
        return _simulate_freq_domain(n, gain_factor, fill)

    elif attack_type == "PDCCH":
        sig = _awgn(n) * gain_factor * 0.03
        sps = int(RATE * 0.0005)          # samples per slot
        sym_len = sps // 14               # samples per OFDM symbol
        n_sym = np.random.randint(1, 4)   # 1-3 CORESET symbols
        for start in range(0, n, sps):
            end = min(start + sym_len * n_sym, n)
            if end > start:
                sig[start:end] = _ofdm_like(end - start) * gain_factor * 4.0
        return sig

    else:  # Generic Deceptive
        return _ofdm_like(n) * gain_factor


# Backward-compatible aliases
simulate_pss = lambda gf=1.0: simulate_attack("PSS/SSS", gf)
simulate_pdcch = lambda gf=1.0: simulate_attack("PDCCH", gf)
simulate_dmrs = lambda gf=1.0: simulate_attack("DMRS", gf)
simulate_generic_deceptive = lambda gf=1.0: simulate_attack("Generic Deceptive", gf)


SIMULATORS = {
    "PSS/SSS":           simulate_pss,
    "PDCCH":             simulate_pdcch,
    "DMRS":              simulate_dmrs,
    "Generic Deceptive": simulate_generic_deceptive,
}


# ─────────────────────────────────────────────
# 3. 합성 데이터셋 생성
# ─────────────────────────────────────────────
def generate_dataset(n_per_class: int = 200):
    """
    4클래스 protocol-aware 스펙트로그램 데이터셋
    Returns: X (N, 224, 224), y (N,)
    """
    X_list, y_list = [], []
    for label in LABELS:
        sim_fn = SIMULATORS[label]
        for i in range(n_per_class):
            gain = np.random.uniform(0.5, 2.5)
            iq   = sim_fn(gain_factor=gain)
            spec = iq_to_spectrogram_224(iq)
            X_list.append(spec)
            y_list.append(LABEL_IDX[label])

            if (i + 1) % 50 == 0:
                print(f"  {label}: {i+1}/{n_per_class}")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ─────────────────────────────────────────────
# 4. MobileNetV3 모델
# ─────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = False):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def spectrogram_to_tensor(spec: np.ndarray) -> torch.Tensor:
    """224×224 스펙트로그램 → (1, 3, 224, 224) 텐서"""
    t = torch.from_numpy(spec).float().unsqueeze(0)
    t = t.expand(3, -1, -1).unsqueeze(0)
    return t


# ─────────────────────────────────────────────
# 5. 학습
# ─────────────────────────────────────────────
def train_and_save(n_per_class: int = 300, epochs: int = 30, lr: float = 0.001):
    print(f"[Stage3] Protocol-Aware 합성 스펙트로그램 생성 중 (클래스당 {n_per_class}개)...")
    X, y = generate_dataset(n_per_class)

    X_tensor = torch.from_numpy(X).unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
    y_tensor = torch.from_numpy(y)
    dataset  = TensorDataset(X_tensor, y_tensor)
    loader   = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            total   += by.size(0)

        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(loader):.4f} | "
                  f"Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[Stage3] 모델 저장: {MODEL_PATH}")
    print(f"[Stage3] 최종 학습 정확도: {correct/total:.4f}")
    return model, device


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH):
        model = build_model(pretrained=False).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"[Stage3] 모델 로드 완료: {MODEL_PATH}")
        return model, device
    else:
        print(f"[Stage3] {MODEL_PATH} 없음 — 합성 데이터로 학습합니다.")
        return train_and_save()


# ─────────────────────────────────────────────
# 6. 추론 (Stage 1/2에서 호출)
# ─────────────────────────────────────────────
def classify(iq: np.ndarray, model, device) -> tuple:
    """
    공격 확정된 I/Q 청크의 protocol-aware 유형을 정밀 분류합니다.

    Returns:
        label      (str)        — "PSS/SSS" / "PDCCH" / "DMRS" / "Generic Deceptive"
        confidence (float)      — 해당 클래스 확률
        probs      (np.ndarray) — 4클래스 확률 벡터
    """
    spec = iq_to_spectrogram_224(iq)
    inp  = spectrogram_to_tensor(spec).to(device)

    model.eval()
    with torch.no_grad():
        out   = model(inp)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MSJC Stage 3 — MobileNetV3 Protocol-Aware 재밍 정밀 분류")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--n-per-class", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.retrain:
        model, device = train_and_save(args.n_per_class, args.epochs)
    else:
        model, device = load_model()

    if args.test:
        print(f"\n[Test] 합성 테스트 데이터 생성 (클래스당 50개)...")
        X_test, y_test = generate_dataset(50)
        X_t = torch.from_numpy(X_test).unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        X_t = X_t.to(device)

        model.eval()
        with torch.no_grad():
            out  = model(X_t)
            pred = out.argmax(1).cpu().numpy()

        acc = (pred == y_test).mean()
        print(f"\n[Test] 전체 정확도: {acc:.4f}")
        for i, label in enumerate(LABELS):
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = (pred[mask] == y_test[mask]).mean()
                print(f"  {label:<20s}: {class_acc:.4f} ({mask.sum()}개)")
