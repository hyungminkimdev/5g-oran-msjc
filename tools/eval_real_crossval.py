#!/usr/bin/env python3
"""
실측 데이터 기반 Stage 1+2 Cross-Validation
kpm_fdd_alldata.csv (1,733개)를 사용하여 실측 accuracy를 검증한다.

출력: per-class detection rate, confusion matrix, false alarm rate
결과를 eval_results.json으로 저장하여 generate_paper_figures.py에서 사용
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from kpi_feature_extractor import csv_row_to_features, NUM_FEATURES
from stage2_ksvm import extract_window_features, WINDOW_SIZE, NUM_WINDOW_FEATURES

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval_results.json")

# ─────────────────────────────────────────────
# Stage 1: MLP on real data
# ─────────────────────────────────────────────
def eval_stage1_real(csv_path: str):
    """Stage 1 MLP를 실측 데이터로 평가 (학습된 모델 로드)"""
    import torch
    from stage1_mlp import JammingMLP, LABELS, LABEL_IDX, load_model

    model, scaler, device = load_model()
    model.eval()

    # CSV 로드
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Stage1 label mapping (PSS/PDCCH/DMRS → Deceptive)
    LABEL_TO_S1 = {
        "Normal": "Normal", "Constant": "Constant",
        "Random": "Random", "Reactive": "Reactive",
        "Deceptive": "Deceptive", "PSS": "Deceptive",
        "PDCCH": "Deceptive", "DMRS": "Deceptive",
    }

    X_list, y_true_s1, y_true_raw = [], [], []
    for row in rows:
        raw_label = row["label"]
        s1_label = LABEL_TO_S1.get(raw_label, raw_label)
        if s1_label not in LABEL_IDX:
            continue
        X_list.append(csv_row_to_features(row))
        y_true_s1.append(LABEL_IDX[s1_label])
        y_true_raw.append(raw_label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_true_s1)

    # 추론
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        preds = logits.argmax(dim=1).numpy()

    print("=" * 60)
    print("Stage 1 MLP — 실측 데이터 평가")
    print("=" * 60)
    print(f"총 샘플: {len(X)}")
    print(classification_report(y, preds, target_names=LABELS, digits=4))

    # 원본 7-mode 기준 detection rate (binary: Normal vs Attack)
    print("\n[7-Mode Binary Detection Rate (Stage 1)]")
    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]
    s1_results = {}
    for mode in modes:
        mask = np.array([r == mode for r in y_true_raw])
        if mask.sum() == 0:
            continue
        mode_preds = preds[mask]
        if mode == "Normal":
            # False alarm = Normal을 Attack으로 오판
            fa_rate = (mode_preds != 0).mean()
            s1_results[mode] = {"fa_rate": fa_rate, "n": int(mask.sum()),
                                "preds": mode_preds, "correct": (mode_preds == 0).mean()}
            print(f"  Normal: FA rate = {fa_rate:.4f} ({(mode_preds != 0).sum()}/{mask.sum()})")
        else:
            # Detection = Attack을 Attack으로 정확히 판정 (≠ Normal)
            det_rate = (mode_preds != 0).mean()
            s1_results[mode] = {"det_rate": det_rate, "n": int(mask.sum()),
                                "preds": mode_preds}
            print(f"  {mode:<12s}: Det = {det_rate:.4f} ({(mode_preds != 0).sum()}/{mask.sum()})")

    return X, y_true_raw, preds, s1_results


# ─────────────────────────────────────────────
# Stage 2: KSVM on real data
# ─────────────────────────────────────────────
def eval_stage2_real(csv_path: str, pipeline, window_size: int = WINDOW_SIZE):
    """Stage 2 KSVM를 실측 데이터의 sliding window로 평가"""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # 모드별 인덱스
    mode_indices = defaultdict(list)
    for i, r in enumerate(rows):
        mode_indices[r['label']].append(i)

    all_features = np.array([csv_row_to_features(r) for r in rows], dtype=np.float32)

    W = window_size
    print("\n" + "=" * 60)
    print(f"Stage 2 KSVM — 실측 Sliding Window 평가 (W={W})")
    print("=" * 60)

    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]
    s2_results = {}

    for mode in modes:
        indices = mode_indices.get(mode, [])
        if len(indices) < W:
            print(f"  {mode:<12s}: 샘플 부족 ({len(indices)} < W={W})")
            continue

        n_windows = len(indices) - W + 1
        attack_detections = []
        for start in range(n_windows):
            win_idx = indices[start:start + W]
            win = all_features[win_idx]
            feat = extract_window_features(win).reshape(1, -1)
            pred = pipeline.predict(feat)[0]
            attack_detections.append(int(pred == 1))

        det_rate = np.mean(attack_detections)
        s2_results[mode] = {
            "det_rate": det_rate,
            "n_windows": n_windows,
            "detections": sum(attack_detections),
        }

        if mode == "Normal":
            print(f"  Normal: FA rate = {det_rate:.4f} ({sum(attack_detections)}/{n_windows} windows)")
        else:
            print(f"  {mode:<12s}: Det = {det_rate:.4f} ({sum(attack_detections)}/{n_windows} windows)")

    return s2_results


# ─────────────────────────────────────────────
# Combined Stage 1+2
# ─────────────────────────────────────────────
def eval_combined(s1_results, s2_results):
    """Stage 1+2 Combined detection rate"""
    print("\n" + "=" * 60)
    print("Combined Stage 1+2 Detection Rate")
    print("=" * 60)

    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]

    for mode in modes:
        s1 = s1_results.get(mode, {})
        s2 = s2_results.get(mode, {})

        if mode == "Normal":
            s1_fa = s1.get("fa_rate", 0)
            s2_fa = s2.get("det_rate", 0)
            # Combined FA: Stage 1 FA + Stage 2 FA on Stage1-Normal samples
            # Stage 2 only sees samples that Stage 1 marked Normal
            # FA = S1_FA + (1-S1_FA) * S2_FA
            combined_fa = s1_fa + (1 - s1_fa) * s2_fa
            print(f"  Normal FA: S1={s1_fa:.4f}, S2={s2_fa:.4f}, Combined≈{combined_fa:.4f}")
        else:
            s1_det = s1.get("det_rate", 0)
            s2_det = s2.get("det_rate", 0)
            # Combined: S1_det + (1-S1_det) * S2_det
            combined_det = s1_det + (1 - s1_det) * s2_det
            n = s1.get("n", "?")
            print(f"  {mode:<12s}: S1={s1_det:.4f}, S2={s2_det:.4f}, Combined≈{combined_det:.4f} (N={n})")


def eval_stage2_holdout(csv_path: str, window_size: int = WINDOW_SIZE, train_ratio: float = 0.8):
    """
    Stage 2 KSVM — 실측 데이터 hold-out 평가.
    각 모드별 시계열을 앞 80%로 학습, 뒤 20%로 테스트.
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SkPipeline

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    mode_indices = defaultdict(list)
    for i, r in enumerate(rows):
        mode_indices[r['label']].append(i)

    all_features = np.array([csv_row_to_features(r) for r in rows], dtype=np.float32)

    W = window_size
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_mode_map = []  # (mode, window_idx) for per-mode analysis

    for mode, indices in mode_indices.items():
        if len(indices) < W:
            continue
        is_attack = 0 if mode == "Normal" else 1

        # 전체 window 생성
        n_windows = len(indices) - W + 1
        all_windows_feat = []
        for start in range(n_windows):
            win_idx = indices[start:start + W]
            win = all_features[win_idx]
            all_windows_feat.append(extract_window_features(win))

        # 시간 기준 split (앞 80% train, 뒤 20% test)
        split_idx = int(n_windows * train_ratio)
        for i, feat in enumerate(all_windows_feat):
            if i < split_idx:
                X_train.append(feat)
                y_train.append(is_attack)
            else:
                X_test.append(feat)
                y_test.append(is_attack)
                test_mode_map.append(mode)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    print(f"\n{'='*60}")
    print(f"Stage 2 KSVM — Hold-Out 평가 (Train {train_ratio*100:.0f}% / Test {(1-train_ratio)*100:.0f}%)")
    print(f"{'='*60}")
    print(f"Train: {len(X_train)} windows (Normal={sum(y_train==0)}, Attack={sum(y_train==1)})")
    print(f"Test:  {len(X_test)} windows (Normal={sum(y_test==0)}, Attack={sum(y_test==1)})")

    # 학습
    pipeline = SkPipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale",
                     probability=True, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    # 테스트
    preds = pipeline.predict(X_test)
    from sklearn.metrics import classification_report as cr
    print(cr(y_test, preds, target_names=["Normal", "Attack"], digits=4))

    # Per-mode 분석
    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]
    s2_holdout = {}
    print("[Hold-Out Test] 모드별 결과:")
    for mode in modes:
        mask = np.array([m == mode for m in test_mode_map])
        if mask.sum() == 0:
            continue
        mode_preds = preds[mask]
        if mode == "Normal":
            fa = (mode_preds == 1).mean()
            s2_holdout[mode] = {"fa_rate": fa, "n": int(mask.sum())}
            print(f"  Normal: FA = {fa:.4f} ({(mode_preds==1).sum()}/{mask.sum()} windows)")
        else:
            det = (mode_preds == 1).mean()
            s2_holdout[mode] = {"det_rate": det, "n": int(mask.sum())}
            print(f"  {mode:<12s}: Det = {det:.4f} ({(mode_preds==1).sum()}/{mask.sum()} windows)")

    return pipeline, s2_holdout


def save_results(s1_results, s2_results, s2_holdout, total_samples):
    """모든 evaluation 결과를 JSON으로 저장"""
    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]

    results = {
        "total_samples": total_samples,
        "modes": {},
    }

    for mode in modes:
        s1 = s1_results.get(mode, {})
        s2 = s2_results.get(mode, {})
        s2h = s2_holdout.get(mode, {})

        entry = {"n_samples": s1.get("n", 0)}

        if mode == "Normal":
            s1_fa = s1.get("fa_rate", 0)
            s2_fa = s2.get("det_rate", 0)
            s2h_fa = s2h.get("fa_rate", 0)
            # Combined는 holdout 기준 (unbiased)
            combined_fa = s1_fa + (1 - s1_fa) * s2h_fa
            entry["s1_fa_rate"] = round(s1_fa * 100, 2)
            entry["s2_fa_rate"] = round(s2_fa * 100, 2)
            entry["s2_holdout_fa_rate"] = round(s2h_fa * 100, 2)
            entry["combined_fa_rate"] = round(combined_fa * 100, 2)
        else:
            s1_det = s1.get("det_rate", 0)
            s2_det = s2.get("det_rate", 0)
            s2h_det = s2h.get("det_rate", 0)
            # Combined는 holdout 기준 (unbiased)
            combined_det = s1_det + (1 - s1_det) * s2h_det
            entry["s1_det_rate"] = round(s1_det * 100, 2)
            entry["s2_det_rate"] = round(s2_det * 100, 2)
            entry["s2_holdout_det_rate"] = round(s2h_det * 100, 2)
            entry["combined_det_rate"] = round(combined_det * 100, 2)

        if mode in s2:
            entry["s2_n_windows"] = s2.get("n_windows", 0)

        results["modes"][mode] = entry

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[저장] eval_results.json → {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "kpm_fdd_alldata.csv")

    print(f"실측 데이터: {csv_path}")

    # Stage 1 평가 (현재 학습된 모델로 전체 데이터 평가)
    X, y_raw, s1_preds, s1_results = eval_stage1_real(csv_path)

    # Stage 2 평가 1: 현재 모델로 전체 데이터 평가
    from stage2_ksvm import load_model
    s2_pipeline = load_model()
    s2_results = eval_stage2_real(csv_path, s2_pipeline)

    # Stage 2 평가 2: Hold-out 평가 (80/20 split, 실측 학습+테스트)
    s2_holdout_pipeline, s2_holdout = eval_stage2_holdout(csv_path)

    # Combined 평가 (hold-out 기준)
    print("\n" + "=" * 60)
    print("Combined Stage 1+2 Detection Rate (Hold-Out 기준)")
    print("=" * 60)
    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]
    for mode in modes:
        s1 = s1_results.get(mode, {})
        s2 = s2_holdout.get(mode, {})
        if mode == "Normal":
            s1_fa = s1.get("fa_rate", 0)
            s2_fa = s2.get("fa_rate", 0)
            combined_fa = s1_fa + (1 - s1_fa) * s2_fa
            print(f"  Normal FA: S1={s1_fa:.4f}, S2(holdout)={s2_fa:.4f}, Combined≈{combined_fa:.4f}")
        else:
            s1_det = s1.get("det_rate", 0)
            s2_det = s2.get("det_rate", 0)
            combined_det = s1_det + (1 - s1_det) * s2_det
            n = s1.get("n", "?")
            print(f"  {mode:<12s}: S1={s1_det:.4f}, S2(holdout)={s2_det:.4f}, Combined≈{combined_det:.4f} (N={n})")

    # JSON으로 결과 저장
    save_results(s1_results, s2_results, s2_holdout, len(X))
