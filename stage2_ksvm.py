"""
Stage 2 KSVM — MSJC Project
논문 Section III-A 기반 False Negative 재검사 (이진 분류)

역할:
  Stage 1 MLP가 "Normal"로 판정한 샘플만 입력받아 2차 검증 수행.
  MLP가 놓친 공격(False Negative)을 KSVM으로 포착합니다.

출력:
  0 = Normal  — 진짜 정상 (확정)
  1 = Attack  — MLP가 놓친 공격 (False Negative 포착)

입력 모드 (dual-mode):
  - O-RAN 모드: 8차원 E2SM-KPM KPI 벡터 (xapp_msjc.py에서 호출)
  - Legacy 모드: raw I/Q complex64 배열 (pipeline_runner.py에서 호출)
"""

import os
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from stage1_mlp import extract_features, simulate_chunk, LABELS as S1_LABELS, CHUNK, RATE
from kpi_feature_extractor import simulate_kpi_chunk

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
    이진 재검사용 데이터셋 생성 (KPI + I/Q 혼합 50/50)
    - Normal 샘플: 정상 신호
    - Attack 샘플: 7가지 공격을 균등 혼합 (경계 케이스 포함)

    Returns: X (N×8 float32), y (N, int32) — 0=Normal, 1=Attack
    """
    X_list, y_list = [], []

    # Normal 샘플 — KPI half
    kpi_count = n_per_class // 2
    for _ in range(kpi_count):
        feat = simulate_kpi_chunk("Normal")
        X_list.append(feat)
        y_list.append(0)

    # Normal 샘플 — I/Q half
    iq_count = n_per_class - kpi_count
    for _ in range(iq_count):
        gain = np.random.uniform(0.3, 1.5)
        iq = simulate_chunk("Normal", gain_factor=gain)
        feat = extract_features(iq)
        X_list.append(feat)
        y_list.append(0)

    # Attack 샘플 — all attack types
    attack_types_kpi = ["Constant", "Random", "Reactive", "Deceptive",
                        "PSS/SSS", "PDCCH", "DMRS"]
    attack_types_iq  = ["Constant", "Random", "Reactive", "Deceptive"]

    # KPI-mode attacks
    kpi_per_type = kpi_count // len(attack_types_kpi)
    for attack in attack_types_kpi:
        for _ in range(kpi_per_type):
            feat = simulate_kpi_chunk(attack)
            X_list.append(feat)
            y_list.append(1)

    # I/Q-mode attacks
    iq_per_type = iq_count // len(attack_types_iq)
    for attack in attack_types_iq:
        for _ in range(iq_per_type):
            gain = np.random.uniform(0.2, 1.5)
            iq = simulate_chunk(attack, gain_factor=gain)
            feat = extract_features(iq)
            X_list.append(feat)
            y_list.append(1)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ─────────────────────────────────────────────
# 2. 모델 학습 / 저장 / 로드
# ─────────────────────────────────────────────
def train_and_save(n_per_class: int = 500):
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
            "kernel": "rbf",
            "C": 10.0,
            "gamma": "scale",
        })
        print("[Stage2 KSVM] ClearML 실험 추적 활성화")
    except ImportError:
        print("[Stage2 KSVM] ClearML 미설치 — 실험 추적 비활성화")

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

    # ClearML artifacts
    if clearml_task is not None:
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            preds = pipeline.predict(X)
            cm = confusion_matrix(y, preds)
            clearml_task.upload_artifact("confusion_matrix", cm)
            clearml_task.upload_artifact("model_pipeline", MODEL_PATH)
            report = classification_report(y, preds, target_names=S2_LABELS)
            clearml_task.get_logger().report_text("Classification Report", report)
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
# 3. 추론 (dual-mode)
# ─────────────────────────────────────────────
def recheck(data: np.ndarray, pipeline) -> tuple:
    """
    Stage 1이 "Normal"로 판정한 데이터를 재검사합니다.

    Args:
        data: raw I/Q chunk (complex64) OR 8-dim float32 KPI feature vector
        pipeline: sklearn Pipeline (scaler + SVM)

    Returns:
        is_attack  (bool)  — True = MLP가 놓친 공격 (False Negative)
        confidence (float) — 판정 확률 (0~1)
        probs      (np.ndarray) — [P(Normal), P(Attack)]
    """
    feat  = extract_features(data).reshape(1, -1)
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
