"""
Stage 1 MLP — MSJC Project
Multilayer Perceptron 기반 5클래스 재밍 분류 (논문 Section III-B 준수)

출력 클래스:
  0 = Normal      — 정상 트래픽
  1 = Constant    — 지속 고출력 AWGN
  2 = Random      — 랜덤 펄스 재밍
  3 = Reactive    — TDD 슬롯 모방 주기적 재밍
  4 = Deceptive   — OFDM 위장 재밍

입력 모드 (dual-mode):
  - O-RAN 모드: 8차원 E2SM-KPM KPI 벡터 (xapp_msjc.py에서 호출)
  - Legacy 모드: raw I/Q complex64 배열 (pipeline_runner.py에서 호출)

특징 벡터 (8차원):
  O-RAN: RSRP, RSRQ, SINR, BLER, UCI_NACK, DL_Throughput, CQI, HARQ_Retx
  Legacy I/Q: RSSI, Power Std, Spectral Flatness, PAPR, Duty Cycle, Periodicity, Centroid, 0.0
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from kpi_feature_extractor import (
    simulate_kpi_chunk,
    NUM_FEATURES as KPI_NUM_FEATURES,
)

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stage1_mlp.pth")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "stage1_scaler.pkl")

LABELS = ["Normal", "Constant", "Random", "Reactive", "Deceptive"]
LABEL_IDX = {name: i for i, name in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

RATE  = 20e6
CHUNK = 128 * 128   # 16,384 samples

POWER_THRESHOLD_DB = -60.0
NUM_FEATURES = 8    # 8-dim: KPI or padded I/Q features


# ─────────────────────────────────────────────
# 1. 특징 추출 (dual-mode)
# ─────────────────────────────────────────────
def _extract_iq_features(iq: np.ndarray) -> np.ndarray:
    """
    Legacy I/Q 청크 (complex64, shape N) → float32 특징 벡터 (shape 8)
    First 7 features are the original I/Q features, 8th is zero-padded.
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

    # [7] Zero-pad to match 8-dim KPI feature length
    return np.array([rssi, power_std, flatness, papr, duty_cycle, periodicity, centroid, 0.0],
                    dtype=np.float32)


def extract_features(data: np.ndarray) -> np.ndarray:
    """
    Dual-mode feature extraction.

    Args:
        data: EITHER
          - raw I/Q array (complex64/complex128, len >= 1) → legacy I/Q features (8-dim, padded)
          - pre-computed 8-dim float32 KPI vector → passed through unchanged

    Returns:
        np.ndarray of shape (8,), dtype float32
    """
    if np.iscomplexobj(data):
        return _extract_iq_features(data)

    if data.dtype in (np.float32, np.float64) and data.shape == (NUM_FEATURES,):
        return data.astype(np.float32)

    # Fallback: if it looks like a feature vector (1D, length 8), use it
    flat = data.ravel().astype(np.float32)
    if len(flat) == NUM_FEATURES:
        return flat

    # Otherwise treat as I/Q
    return _extract_iq_features(data.astype(np.complex64))


# ─────────────────────────────────────────────
# 2. MLP 모델 정의
# ─────────────────────────────────────────────
class JammingMLP(nn.Module):
    """
    논문 Figure 3 기반 Multilayer Perceptron (확장)
    Input(8) → 256 → 128 → 64 → Output(5)
    """
    def __init__(self, n_features: int = NUM_FEATURES, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 3. 합성 데이터 생성
# ─────────────────────────────────────────────
def _awgn(n: int) -> np.ndarray:
    return (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)

def _ofdm_like(n: int) -> np.ndarray:
    n_fft = 1024
    syms = _awgn(n_fft)
    frame = np.fft.ifft(syms).astype(np.complex64)
    return np.tile(frame, int(np.ceil(n / n_fft)))[:n]

def simulate_chunk(label: str, gain_factor: float = 1.0) -> np.ndarray:
    """Legacy: 재밍 유형/정상 신호 I/Q 시뮬레이션 (pipeline_runner.py 호환)"""
    n = CHUNK

    if label == "Normal":
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
    5클래스 합성 데이터셋 생성 (KPI + I/Q 혼합 50/50)
    Returns: X (N×8 float32), y (N, int64)
    """
    X_list, y_list = [], []

    for label in LABELS:
        # Half from KPI simulation
        kpi_count = n_per_class // 2
        for _ in range(kpi_count):
            feat = simulate_kpi_chunk(label)
            X_list.append(feat)
            y_list.append(LABEL_IDX[label])

        # Half from legacy I/Q simulation
        iq_count = n_per_class - kpi_count
        for _ in range(iq_count):
            gain = np.random.uniform(0.5, 2.0)
            iq = simulate_chunk(label, gain_factor=gain)
            feat = _extract_iq_features(iq)
            X_list.append(feat)
            y_list.append(LABEL_IDX[label])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ─────────────────────────────────────────────
# 4. 학습
# ─────────────────────────────────────────────
def train_and_save(n_per_class: int = 500, epochs: int = 50, lr: float = 0.001):
    # ClearML integration (graceful)
    clearml_task = None
    try:
        from clearml import Task
        clearml_task = Task.init(
            project_name="msjc-5g-oran",
            task_name=f"msjc-stage1-{int(__import__('time').time())}",
            task_type=Task.TaskTypes.training,
        )
        clearml_task.connect({
            "n_per_class": n_per_class,
            "epochs": epochs,
            "lr": lr,
            "num_features": NUM_FEATURES,
            "num_classes": NUM_CLASSES,
            "architecture": "MLP 8→256→128→64→5",
        })
        print("[Stage1 MLP] ClearML 실험 추적 활성화")
    except ImportError:
        print("[Stage1 MLP] ClearML 미설치 — 실험 추적 비활성화")

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

    # ClearML: confusion matrix artifact
    if clearml_task is not None:
        try:
            model.eval()
            with torch.no_grad():
                all_inp = torch.from_numpy(X_scaled).to(device)
                all_pred = model(all_inp).argmax(1).cpu().numpy()
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, all_pred)
            clearml_task.upload_artifact("confusion_matrix", cm)
            clearml_task.upload_artifact("model_weights", MODEL_PATH)
            clearml_task.upload_artifact("scaler", SCALER_PATH)
        except Exception:
            pass

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
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
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
def classify(data: np.ndarray, model, scaler, device) -> tuple:
    """
    Dual-mode 5클래스 분류.

    Args:
        data: raw I/Q chunk (complex64) OR 8-dim float32 KPI feature vector

    Returns:
        label (str)        — "Normal" / "Constant" / "Random" / "Reactive" / "Deceptive"
        confidence (float) — 해당 클래스 확률
        probs (np.ndarray) — 5클래스 확률 벡터
    """
    feat = extract_features(data).reshape(1, -1)
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
