"""
Stage 1 MLP — MSJC Project
Multilayer Perceptron 기반 5클래스 재밍 분류 (논문 Section III-B 준수)

출력 클래스:
  0 = Normal      — 정상 트래픽
  1 = Constant    — 지속 고출력 AWGN
  2 = Random      — 랜덤 펄스 재밍
  3 = Reactive    — TDD 슬롯 모방 주기적 재밍
  4 = Deceptive   — OFDM 위장 재밍

논문에서 MLP를 선택한 이유 (p.3):
  "MLP has been chosen because of the very large number of input vectors
   ... MLP was chosen among all deep learning algorithms because of its
   flexibility which allows it to be applied to different types of data."

특징 벡터 (7차원): RSSI, Power Std, Spectral Flatness, PAPR,
                   Duty Cycle, Periodicity, Spectral Centroid
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stage1_mlp.pth")

LABELS = ["Normal", "Constant", "Random", "Reactive", "Deceptive"]
LABEL_IDX = {name: i for i, name in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

RATE  = 20e6
CHUNK = 128 * 128   # 16,384 samples

POWER_THRESHOLD_DB = -60.0
NUM_FEATURES = 7


# ─────────────────────────────────────────────
# 1. 특징 추출 (I/Q → 7차원 벡터)
# ─────────────────────────────────────────────
def extract_features(iq: np.ndarray) -> np.ndarray:
    """
    I/Q 청크 (complex64, shape N) → float32 특징 벡터 (shape 7)
    """
    power = np.abs(iq) ** 2

    # [0] RSSI (dB)
    rssi = 10 * np.log10(np.mean(power) + 1e-12)

    # [1] Power Std
    power_std = float(np.std(power))

    # [2] Spectral Flatness
    psd = np.abs(np.fft.fft(iq)) ** 2
    flatness = float(
        np.exp(np.nanmean(np.log(psd + 1e-12))) / (np.nanmean(psd) + 1e-12)
    )

    # [3] PAPR (Peak-to-Average Power Ratio)
    papr = float(np.max(power) / (np.mean(power) + 1e-12))

    # [4] Duty Cycle
    power_db = 10 * np.log10(power + 1e-12)
    duty_cycle = float(np.mean(power_db > POWER_THRESHOLD_DB))

    # [5] Periodicity (자기상관 피크)
    n_blocks = 16
    block_size = len(power) // n_blocks
    block_power = np.array([
        np.mean(power[i * block_size:(i + 1) * block_size])
        for i in range(n_blocks)
    ])
    block_power -= block_power.mean()
    if block_power.std() > 1e-12:
        autocorr = np.correlate(block_power, block_power, mode="full")
        autocorr = autocorr[n_blocks:]
        autocorr /= (autocorr[0] + 1e-12)
        periodicity = float(np.max(autocorr[1:]))
    else:
        periodicity = 0.0

    # [6] Spectral Centroid (정규화)
    freqs = np.fft.fftfreq(len(iq), d=1.0 / RATE)
    psd_half = psd[:len(psd) // 2]
    freqs_half = np.abs(freqs[:len(freqs) // 2])
    centroid = float(
        np.sum(freqs_half * psd_half) / (np.sum(psd_half) + 1e-12) / (RATE / 2)
    )

    return np.array([rssi, power_std, flatness, papr, duty_cycle, periodicity, centroid],
                    dtype=np.float32)


# ─────────────────────────────────────────────
# 2. MLP 모델 정의
# ─────────────────────────────────────────────
class JammingMLP(nn.Module):
    """
    논문 Figure 3 기반 Multilayer Perceptron
    Input(7) → 128 → 64 → 32 → Output(5)
    """
    def __init__(self, n_features: int = NUM_FEATURES, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 3. 합성 데이터 생성 (jammer.py 공격 패턴 재현)
# ─────────────────────────────────────────────
def _awgn(n: int) -> np.ndarray:
    return (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

def _ofdm_like(n: int) -> np.ndarray:
    n_fft = 1024
    syms = _awgn(n_fft)
    frame = np.fft.ifft(syms).astype(np.complex64)
    return np.tile(frame, int(np.ceil(n / n_fft)))[:n]

def simulate_chunk(label: str, gain_factor: float = 1.0) -> np.ndarray:
    """재밍 유형/정상 신호 I/Q 시뮬레이션"""
    n = CHUNK

    if label == "Normal":
        # 배경 잡음 수준의 약한 신호
        return _awgn(n) * gain_factor * 0.01

    elif label == "Constant":
        return _awgn(n) * gain_factor

    elif label == "Random":
        sig = np.zeros(n, dtype=np.complex64)
        burst_rate = np.random.uniform(0.1, 0.4)
        on_mask = np.random.rand(n) < burst_rate
        sig[on_mask] = _awgn(int(on_mask.sum())) * gain_factor * 3.0
        return sig

    elif label == "Reactive":
        sig = np.zeros(n, dtype=np.complex64)
        period = int(RATE * 0.005)
        slot   = int(period * 0.4)
        for start in range(0, n, period):
            end = min(start + slot, n)
            length = end - start
            if length > 0:
                sig[start:end] = _awgn(length) * gain_factor * 2.0
        return sig

    elif label == "Deceptive":
        return _ofdm_like(n) * gain_factor

    raise ValueError(f"Unknown label: {label}")


def generate_synthetic_dataset(n_per_class: int = 500):
    """
    5클래스 합성 데이터셋 생성
    Returns: X (N×7 float32), y (N, int64)
    """
    X_list, y_list = [], []
    for label in LABELS:
        for _ in range(n_per_class):
            gain = np.random.uniform(0.5, 2.0)
            iq   = simulate_chunk(label, gain_factor=gain)
            feat = extract_features(iq)
            X_list.append(feat)
            y_list.append(LABEL_IDX[label])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ─────────────────────────────────────────────
# 4. 학습
# ─────────────────────────────────────────────
# 특징 정규화를 위한 통계값 (학습 시 저장, 추론 시 로드)
SCALER_PATH = os.path.join(os.path.dirname(__file__), "stage1_scaler.pkl")

def train_and_save(n_per_class: int = 500, epochs: int = 50, lr: float = 0.001):
    print(f"[Stage1 MLP] 합성 데이터 생성 중 (클래스당 {n_per_class}개, 총 {n_per_class * NUM_CLASSES}개)...")
    X, y = generate_synthetic_dataset(n_per_class)

    # StandardScaler로 정규화
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Stage1 MLP] Scaler 저장: {SCALER_PATH}")

    # DataLoader
    dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y))
    loader  = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = JammingMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(loader):.4f} | "
                  f"Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[Stage1 MLP] 모델 저장: {MODEL_PATH}")
    print(f"[Stage1 MLP] 최종 학습 정확도: {correct/total:.4f}")
    return model, scaler, device


def load_model():
    """
    학습된 MLP + Scaler 로드
    없으면 합성 데이터로 자동 학습 후 저장
    Returns: (model, scaler, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = JammingMLP().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"[Stage1 MLP] 모델 로드 완료: {MODEL_PATH}")
        return model, scaler, device
    else:
        print(f"[Stage1 MLP] 저장된 모델 없음 — 합성 데이터로 학습합니다.")
        return train_and_save()


# ─────────────────────────────────────────────
# 5. 추론
# ─────────────────────────────────────────────
def classify(iq: np.ndarray, model, scaler, device) -> tuple:
    """
    I/Q 청크 → 5클래스 분류

    Returns:
        label (str)        — "Normal" / "Constant" / "Random" / "Reactive" / "Deceptive"
        confidence (float) — 해당 클래스 확률
        probs (np.ndarray) — 5클래스 확률 벡터
    """
    feat = extract_features(iq).reshape(1, -1)
    feat_scaled = scaler.transform(feat).astype(np.float32)

    inp = torch.from_numpy(feat_scaled).to(device)
    with torch.no_grad():
        out   = model(inp)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    return LABELS[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────
# Entry Point (독립 학습/테스트용)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MSJC Stage 1 — MLP 5클래스 재밍 분류기")
    parser.add_argument("--retrain", action="store_true",
                        help="기존 모델 무시하고 재학습")
    parser.add_argument("--n-per-class", type=int, default=500,
                        help="클래스당 합성 샘플 수 (기본: 500)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="학습 에포크 수 (기본: 50)")
    parser.add_argument("--test", action="store_true",
                        help="합성 테스트 데이터로 정확도 확인")
    args = parser.parse_args()

    if args.retrain:
        model, scaler, device = train_and_save(args.n_per_class, args.epochs)
    else:
        model, scaler, device = load_model()

    if args.test:
        print(f"\n[Test] 합성 테스트 데이터 생성 (클래스당 100개)...")
        X_test, y_test = generate_synthetic_dataset(100)
        X_scaled = scaler.transform(X_test).astype(np.float32)
        inp = torch.from_numpy(X_scaled).to(device)

        model.eval()
        with torch.no_grad():
            out  = model(inp)
            pred = out.argmax(1).cpu().numpy()

        acc = (pred == y_test).mean()
        print(f"[Test] 전체 정확도: {acc:.4f}")

        for i, label in enumerate(LABELS):
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = (pred[mask] == y_test[mask]).mean()
                print(f"  {label:<12s}: {class_acc:.4f} ({mask.sum()}개)")
