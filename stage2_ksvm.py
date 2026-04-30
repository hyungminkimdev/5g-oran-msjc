"""
Stage 2 KSVM — MSJC Project (Sliding Window 통계 기반)
논문 Section III-A 기반 False Negative 재검사 (이진 분류)

역할:
  Stage 1 MLP가 "Normal"로 판정한 구간을 **시간축 통계**로 2차 검증.
  단일 샘플이 아닌 최근 N샘플(sliding window)의 통계 feature를 분석하여
  Random/Reactive 등 간헐적 공격의 FN을 포착한다.

입력:
  최근 W개 KPM 8-dim 벡터의 시계열 → 통계 feature 12-dim 추출

출력:
  0 = Normal  — 진짜 정상 (확정)
  1 = Attack  — MLP가 놓친 공격 (False Negative 포착)
"""

import os
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from kpi_feature_extractor import simulate_kpi_chunk, NUM_FEATURES

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stage2_ksvm.pkl")

S2_LABELS = ["Normal", "Attack"]

WINDOW_SIZE = 15   # 15초 (1초 KPM × 15샘플)
NUM_WINDOW_FEATURES = 12


# ─────────────────────────────────────────────
# 1. Sliding Window 통계 Feature 추출
# ─────────────────────────────────────────────
def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    최근 W개 KPM 8-dim 벡터 → 12-dim 통계 feature.

    Args:
        window: shape (W, 8), W >= 2. 각 행은 8-dim KPM feature vector.
                [rsrp, rsrq, sinr, bler, nack, tput, cqi, harq]

    Returns:
        np.ndarray shape (12,) float32 — sliding window 통계
    """
    # 주요 KPI 열 추출
    sinr = window[:, 2]   # PUCCH SINR
    bler = window[:, 3]   # DL BLER
    nack = window[:, 4]   # UCI NACK rate
    cqi  = window[:, 6]   # CQI accumulator

    # [0] bler_mean — 평균 BLER (Constant: ~0.92, Normal: ~0)
    bler_mean = np.mean(bler)

    # [1] bler_var — BLER 분산 (Random/Reactive: 높음, Normal/Constant: 낮음)
    bler_var = np.var(bler)

    # [2] bler_max — 최대 BLER (spike 탐지)
    bler_max = np.max(bler)

    # [3] bler_spike_ratio — BLER > 0.03인 샘플 비율
    bler_spike_ratio = np.mean(bler > 0.03)

    # [4] sinr_mean — 평균 SINR
    sinr_mean = np.mean(sinr)

    # [5] sinr_var — SINR 분산 (재밍 시 증가)
    sinr_var = np.var(sinr)

    # [6] sinr_min — 최소 SINR (순간 열화 감지)
    sinr_min = np.min(sinr)

    # [7] cqi_var — CQI 분산 (재밍 시 fluctuation)
    cqi_var = np.var(cqi)

    # [8] cqi_min — 최소 CQI
    cqi_min = np.min(cqi)

    # [9] nack_mean — 평균 NACK rate
    nack_mean = np.mean(nack)

    # [10] bler_periodicity — BLER 시계열 자기상관 (Reactive: 높음, Random: 낮음)
    bler_centered = bler - bler_mean
    if np.std(bler_centered) > 1e-8 and len(bler) >= 4:
        autocorr = np.correlate(bler_centered, bler_centered, mode='full')
        autocorr = autocorr[len(bler):]  # positive lags only
        autocorr /= (autocorr[0] + 1e-12)
        # 1차 피크 (lag 1 이후)
        bler_periodicity = float(np.max(autocorr[1:min(len(autocorr), len(bler)//2)])) \
            if len(autocorr) > 1 else 0.0
    else:
        bler_periodicity = 0.0

    # [11] transition_count — BLER 상태 전환 횟수 (0↔nonzero)
    #      Random/Reactive: 빈번, Normal/Constant: 0
    bler_binary = (bler > 0.03).astype(int)
    transition_count = float(np.sum(np.abs(np.diff(bler_binary))))

    return np.array([
        bler_mean, bler_var, bler_max, bler_spike_ratio,
        sinr_mean, sinr_var, sinr_min,
        cqi_var, cqi_min,
        nack_mean,
        bler_periodicity, transition_count,
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# 2. 합성 시계열 생성 (학습용)
# ─────────────────────────────────────────────
def _generate_window(label: str, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """한 개 sliding window 시계열 생성 (W × 8)"""
    samples = []
    for _ in range(window_size):
        samples.append(simulate_kpi_chunk(label))
    return np.stack(samples)


def generate_window_dataset(n_per_class: int = 500, window_size: int = WINDOW_SIZE):
    """
    이진 재검사용 sliding window 데이터셋 생성.
    - Normal 시계열: 정상 KPM W개 연속
    - Attack 시계열: 5가지 공격 유형 균등 혼합

    Returns: X (N × 12 float32), y (N, int32) — 0=Normal, 1=Attack
    """
    X_list, y_list = [], []

    # Normal windows
    for _ in range(n_per_class):
        win = _generate_window("Normal", window_size)
        X_list.append(extract_window_features(win))
        y_list.append(0)

    # Attack windows — 5가지 공격 유형 + Stage3 서브타입
    attack_types = ["Constant", "Random", "Reactive", "Deceptive",
                    "PSS/SSS", "PDCCH", "DMRS"]
    per_type = max(1, n_per_class // len(attack_types))

    for attack in attack_types:
        for _ in range(per_type):
            win = _generate_window(attack, window_size)
            X_list.append(extract_window_features(win))
            y_list.append(1)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ─────────────────────────────────────────────
# 3. 모델 학습 / 저장 / 로드
# ─────────────────────────────────────────────
def train_and_save(n_per_class: int = 500, window_size: int = WINDOW_SIZE):
    # ClearML integration (graceful)
    clearml_task = None
    try:
        from clearml import Task
        clearml_task = Task.init(
            project_name="msjc-5g-oran",
            task_name=f"msjc-stage2-{int(__import__('time').time())}",
            task_type=Task.TaskTypes.training,
        )
        clearml_task.connect({
            "n_per_class": n_per_class,
            "window_size": window_size,
            "n_window_features": NUM_WINDOW_FEATURES,
            "kernel": "rbf",
            "C": 10.0,
            "gamma": "scale",
        })
        print("[Stage2 KSVM] ClearML 실험 추적 활성화")
    except ImportError:
        print("[Stage2 KSVM] ClearML 미설치 — 실험 추적 비활성화")

    print(f"[Stage2 KSVM] Sliding window 합성 데이터 생성 (클래스당 {n_per_class}개, W={window_size})...")
    X, y = generate_window_dataset(n_per_class, window_size)

    print(f"[Stage2 KSVM] Feature shape: {X.shape}, 공격 유형 7가지 포함")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       probability=True, random_state=42)),
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"[Stage2 KSVM] 5-Fold CV 정확도: {scores.mean():.4f} ± {scores.std():.4f}")

    pipeline.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[Stage2 KSVM] 모델 저장: {MODEL_PATH}")

    # ClearML artifacts
    if clearml_task is not None:
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            preds = pipeline.predict(X)
            cm = confusion_matrix(y, preds)
            clearml_task.upload_artifact("confusion_matrix", cm)
            clearml_task.upload_artifact("model_pipeline", MODEL_PATH)
        except Exception:
            pass

    return pipeline


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        print(f"[Stage2 KSVM] 모델 로드 완료: {MODEL_PATH}")
    else:
        print(f"[Stage2 KSVM] {MODEL_PATH} 없음 — 합성 데이터로 학습합니다.")
        pipeline = train_and_save()
    return pipeline


# ─────────────────────────────────────────────
# 4. 추론
# ─────────────────────────────────────────────
def recheck(window: np.ndarray, pipeline) -> tuple:
    """
    Stage 1이 "Normal"로 판정한 구간을 재검사.

    Args:
        window: shape (W, 8) — 최근 W개 KPM feature vector
        pipeline: sklearn Pipeline (scaler + SVM)

    Returns:
        is_attack  (bool)  — True = FN 포착
        confidence (float) — 판정 확률 (0~1)
        probs      (np.ndarray) — [P(Normal), P(Attack)]
    """
    feat = extract_window_features(window).reshape(1, -1)
    probs = pipeline.predict_proba(feat)[0]
    pred = int(np.argmax(probs))

    is_attack = (pred == 1)
    confidence = float(probs[pred])
    return is_attack, confidence, probs


# ─────────────────────────────────────────────
# Entry Point (독립 학습/테스트)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MSJC Stage 2 — KSVM Sliding Window 이진 재검사")
    parser.add_argument("--retrain", action="store_true",
                        help="기존 모델 무시하고 재학습")
    parser.add_argument("--n-per-class", type=int, default=500,
                        help="클래스당 합성 window 수 (기본: 500)")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                        help=f"슬라이딩 윈도우 크기 (기본: {WINDOW_SIZE})")
    parser.add_argument("--test", action="store_true",
                        help="합성 테스트 데이터로 정확도 확인")
    parser.add_argument("--test-real", type=str, default=None,
                        help="실측 CSV로 평가 (kpm_fdd_7modes.csv)")
    args = parser.parse_args()

    if args.retrain:
        pipeline = train_and_save(args.n_per_class, args.window_size)
    else:
        pipeline = load_model()

    if args.test:
        print(f"\n[Test] 합성 테스트 (클래스당 200 windows, W={args.window_size})...")
        X_test, y_test = generate_window_dataset(200, args.window_size)

        from sklearn.metrics import classification_report
        preds = pipeline.predict(X_test)
        print(classification_report(y_test, preds, target_names=S2_LABELS))

        # Per-attack-type 분석
        print("[Test] 공격 유형별 탐지율:")
        attack_types = ["Constant", "Random", "Reactive", "Deceptive",
                        "PSS/SSS", "PDCCH", "DMRS"]
        per_type = 200 // len(attack_types)
        for i, atype in enumerate(attack_types):
            start = 200 + i * per_type
            end = start + per_type
            if end <= len(preds):
                det = (preds[start:end] == 1).mean()
                print(f"  {atype:<12s}: {det:.4f}")

    if args.test_real:
        from kpi_feature_extractor import load_real_csv
        import csv as _csv

        print(f"\n[Test-Real] 실측 데이터 평가: {args.test_real}")
        X_real, labels_real = load_real_csv(args.test_real)

        # 원본 7-mode 레이블 로드
        with open(args.test_real) as f:
            rows = list(_csv.DictReader(f))

        # 각 모드별로 sliding window 구성 (연속 샘플)
        from collections import defaultdict
        mode_indices = defaultdict(list)
        for i, r in enumerate(rows):
            mode_indices[r['label']].append(i)

        W = args.window_size
        print(f"  Window size: {W}")
        for mode in ['Normal', 'Constant', 'Random', 'Reactive',
                      'Deceptive', 'PSS', 'PDCCH', 'DMRS']:
            indices = mode_indices[mode]
            if len(indices) < W:
                print(f"  {mode:<12s}: 샘플 부족 ({len(indices)} < W={W})")
                continue

            # 가능한 모든 window
            n_windows = len(indices) - W + 1
            results = []
            for start in range(n_windows):
                win_idx = indices[start:start + W]
                win = X_real[win_idx]
                is_atk, conf, _ = recheck(win, pipeline)
                results.append(is_atk)

            det_rate = np.mean(results)
            print(f"  {mode:<12s}: Attack 탐지={det_rate:.4f} ({sum(results)}/{len(results)} windows)")
