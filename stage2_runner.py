"""
Stage 2 KSVM — MSJC Project
논문 Section III-A 기반 False Negative 재검사 (이진 분류)

역할:
  Stage 1 MLP가 "Normal"로 판정한 샘플만 입력받아 2차 검증 수행.
  MLP가 놓친 공격(False Negative)을 KSVM으로 포착합니다.

  논문 인용 (p.3):
    "If classified as normal by MLP, the traffic is processed again by KSVM.
     The motivation to add KSVM after MLP is to reduce the false negatives
     that MLP has created."

출력:
  0 = Normal  — 진짜 정상 (확정)
  1 = Attack  — MLP가 놓친 공격 (False Negative 포착)

파이프라인 흐름:
  I/Q 수신 → [Stage 1 MLP: 5클래스] → 공격 판정 시 즉시 확정
                                     → "Normal" 판정 시 → [Stage 2 KSVM: 이진 재검사]
                                                           → Normal 확정 / Attack 경보
"""

import os
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from stage1_mlp import (
    extract_features, simulate_chunk, LABELS as S1_LABELS,
    CHUNK, RATE
)

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "stage2_ksvm.pkl")

S2_LABELS = ["Normal", "Attack"]


# ─────────────────────────────────────────────
# 1. 합성 학습 데이터 생성 (이진: Normal vs Attack)
# ─────────────────────────────────────────────
def generate_binary_dataset(n_per_class: int = 500):
    """
    이진 재검사용 데이터셋 생성
    - Normal 샘플: 정상 신호
    - Attack 샘플: 4가지 공격을 균등 혼합 (Stage 1이 놓칠 수 있는 경계 케이스 포함)

    Returns: X (N×7 float32), y (N, int32) — 0=Normal, 1=Attack
    """
    X_list, y_list = [], []

    # Normal 샘플
    for _ in range(n_per_class):
        gain = np.random.uniform(0.3, 1.5)
        iq   = simulate_chunk("Normal", gain_factor=gain)
        feat = extract_features(iq)
        X_list.append(feat)
        y_list.append(0)

    # Attack 샘플 — 4가지 공격을 균등 배분
    attack_types = ["Constant", "Random", "Reactive", "Deceptive"]
    per_attack = n_per_class // len(attack_types)

    for attack in attack_types:
        for _ in range(per_attack):
            # 경계 케이스를 만들기 위해 낮은 gain도 포함 (Stage 1이 놓치기 쉬운 약한 공격)
            gain = np.random.uniform(0.2, 1.5)
            iq   = simulate_chunk(attack, gain_factor=gain)
            feat = extract_features(iq)
            X_list.append(feat)
            y_list.append(1)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ─────────────────────────────────────────────
# 2. 모델 학습 / 저장 / 로드
# ─────────────────────────────────────────────
def train_and_save(n_per_class: int = 500):
    print(f"[Stage2 KSVM] 이진 재검사 합성 데이터 생성 중 (클래스당 {n_per_class}개)...")
    X, y = generate_binary_dataset(n_per_class)

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
# 3. 추론 (Stage 1에서 호출)
# ─────────────────────────────────────────────
def recheck(iq: np.ndarray, pipeline) -> tuple:
    """
    Stage 1이 "Normal"로 판정한 I/Q 청크를 재검사합니다.

    Returns:
        is_attack  (bool)  — True = MLP가 놓친 공격 (False Negative)
        confidence (float) — 판정 확률 (0~1)
        probs      (np.ndarray) — [P(Normal), P(Attack)]
    """
    feat  = extract_features(iq).reshape(1, -1)
    probs = pipeline.predict_proba(feat)[0]
    pred  = int(np.argmax(probs))

    is_attack  = (pred == 1)
    confidence = float(probs[pred])
    return is_attack, confidence, probs


# ─────────────────────────────────────────────
# Entry Point (독립 학습/테스트)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MSJC Stage 2 — KSVM 이진 재검사 (Normal 판정 False Negative 포착)")
    parser.add_argument("--retrain", action="store_true",
                        help="기존 모델 무시하고 재학습")
    parser.add_argument("--n-per-class", type=int, default=500,
                        help="클래스당 합성 샘플 수 (기본: 500)")
    parser.add_argument("--test", action="store_true",
                        help="합성 테스트 데이터로 정확도 확인")
    args = parser.parse_args()

    if args.retrain:
        pipeline = train_and_save(args.n_per_class)
    else:
        pipeline = load_model()

    if args.test:
        print(f"\n[Test] 합성 테스트 데이터 생성 (클래스당 200개)...")
        X_test, y_test = generate_binary_dataset(200)

        from sklearn.metrics import classification_report
        preds = pipeline.predict(X_test)
        print(classification_report(y_test, preds, target_names=S2_LABELS))
