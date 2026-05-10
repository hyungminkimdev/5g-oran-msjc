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
# Stage 1: Blocked Temporal K-Fold CV
# ─────────────────────────────────────────────
def _build_temporal_blocks(mode_indices: dict, n_blocks: int):
    """
    각 모드의 시계열 인덱스를 n_blocks개 연속 블록으로 분할.
    Returns: list of (block_indices, raw_labels) per block
    """
    blocks = [[] for _ in range(n_blocks)]
    block_labels = [[] for _ in range(n_blocks)]

    for mode, indices in mode_indices.items():
        n = len(indices)
        if n == 0:
            continue
        block_size = max(1, n // n_blocks)
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size if b < n_blocks - 1 else n
            for idx in indices[start:end]:
                blocks[b].append(idx)
                block_labels[b].append(mode)

    return blocks, block_labels


def eval_stage1_blocked_cv(csv_path: str, n_folds: int = 5,
                           n_per_class: int = 500, epochs: int = 50):
    """
    Stage 1 MLP — Blocked Temporal K-Fold Cross-Validation.

    각 모드의 시계열을 k개 연속 블록으로 분할 후,
    매 fold마다 1블록을 테스트, 나머지 k-1블록의 실측 데이터 + 합성 데이터로 학습.
    Train/Test에 동일 샘플이 포함되지 않아 leakage 없음.
    """
    import torch
    from stage1_mlp import (LABELS, LABEL_IDX,
                            generate_synthetic_dataset_with_real_split,
                            train_model_from_data, classify)

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    LABEL_TO_S1 = {
        "Normal": "Normal", "Constant": "Constant",
        "Random": "Random", "Reactive": "Reactive",
        "Deceptive": "Deceptive", "PSS": "Deceptive",
        "PDCCH": "Deceptive", "DMRS": "Deceptive",
    }

    # 모드별 인덱스
    mode_indices = defaultdict(list)
    for i, r in enumerate(rows):
        mode_indices[r['label']].append(i)

    # 모든 feature 미리 변환
    all_features = np.array([csv_row_to_features(r) for r in rows], dtype=np.float32)
    all_raw_labels = [r['label'] for r in rows]

    # Temporal block 분할
    blocks, block_labels = _build_temporal_blocks(mode_indices, n_folds)

    print("=" * 60)
    print(f"Stage 1 MLP — Blocked Temporal {n_folds}-Fold CV")
    print("=" * 60)

    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]

    # per-fold 결과 수집
    all_fold_results = []  # list of dict per fold

    for fold in range(n_folds):
        test_indices = blocks[fold]
        train_indices = []
        for b in range(n_folds):
            if b != fold:
                train_indices.extend(blocks[b])

        # 학습: 합성 데이터 + train fold의 실측 데이터
        X_train, y_train = generate_synthetic_dataset_with_real_split(
            n_per_class=n_per_class,
            real_csv=csv_path,
            real_indices=train_indices,
        )

        # 모델 학습
        model, scaler, device = train_model_from_data(X_train, y_train, epochs=epochs)

        # 테스트: test fold 실측 데이터만
        fold_preds = []
        fold_true_raw = []
        for idx in test_indices:
            feat = all_features[idx]
            raw_label = all_raw_labels[idx]
            s1_label = LABEL_TO_S1.get(raw_label, raw_label)
            if s1_label not in LABEL_IDX:
                continue

            label, conf, probs = classify(feat, model, scaler, device)
            pred_idx = LABEL_IDX[label]
            fold_preds.append(pred_idx)
            fold_true_raw.append(raw_label)

        fold_preds = np.array(fold_preds)
        fold_true_raw = np.array(fold_true_raw)

        # Per-mode 결과
        fold_result = {}
        for mode in modes:
            mask = fold_true_raw == mode
            if mask.sum() == 0:
                continue
            mode_preds = fold_preds[mask]
            if mode == "Normal":
                fa_rate = (mode_preds != 0).mean()
                fold_result[mode] = {"fa_rate": fa_rate, "n": int(mask.sum())}
            else:
                det_rate = (mode_preds != 0).mean()
                fold_result[mode] = {"det_rate": det_rate, "n": int(mask.sum())}

        all_fold_results.append(fold_result)
        n_test = len(fold_preds)
        n_correct = sum(1 for p, r in zip(fold_preds, fold_true_raw)
                       if (r == "Normal" and p == 0) or (r != "Normal" and p != 0))
        print(f"  Fold {fold+1}: test={n_test}, binary_acc={n_correct/max(n_test,1):.4f}")

    # 전체 결과 집계 (weighted average)
    print(f"\n{'='*60}")
    print(f"Stage 1 MLP — Blocked {n_folds}-Fold CV 결과 (weighted avg)")
    print(f"{'='*60}")

    s1_results = {}
    for mode in modes:
        fold_vals = []
        fold_ns = []
        for fr in all_fold_results:
            if mode in fr:
                key = "fa_rate" if mode == "Normal" else "det_rate"
                fold_vals.append(fr[mode][key])
                fold_ns.append(fr[mode]["n"])

        if not fold_vals:
            continue

        # Weighted average by sample count
        total_n = sum(fold_ns)
        weighted_avg = sum(v * n for v, n in zip(fold_vals, fold_ns)) / total_n
        # Per-fold std
        fold_std = np.std(fold_vals)

        if mode == "Normal":
            s1_results[mode] = {"fa_rate": weighted_avg, "n": total_n,
                                "fa_std": fold_std, "per_fold": fold_vals}
            print(f"  Normal FA:     {weighted_avg:.4f} ± {fold_std:.4f} "
                  f"(N={total_n}, folds={[f'{v:.4f}' for v in fold_vals]})")
        else:
            s1_results[mode] = {"det_rate": weighted_avg, "n": total_n,
                                "det_std": fold_std, "per_fold": fold_vals}
            print(f"  {mode:<12s}:  {weighted_avg:.4f} ± {fold_std:.4f} "
                  f"(N={total_n}, folds={[f'{v:.4f}' for v in fold_vals]})")

    return s1_results


def eval_stage1_real(csv_path: str):
    """Stage 1 MLP를 실측 데이터로 평가 (학습된 모델 로드, 참고용)"""
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
    print("Stage 1 MLP — 실측 데이터 평가 (기존 모델, 참고용)")
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


def eval_stage2_blocked_cv(csv_path: str, n_folds: int = 5,
                           window_size: int = WINDOW_SIZE,
                           n_synth_per_class: int = 500):
    """
    Stage 2 KSVM — Blocked Temporal K-Fold Cross-Validation.

    각 모드의 시계열을 k개 연속 블록으로 분할 후,
    sliding window를 각 블록 **내부에서만** 생성하여
    train/test 간 샘플 공유를 완전히 차단한다.

    합성 window 혼합 이유:
      실측 Normal 데이터의 temporal non-stationarity로 인해
      특정 블록이 학습 분포에서 벗어남. 합성 데이터가 넓은 Normal 분포를
      커버하여 일반화 성능을 향상시킴 (Stage 1과 동일한 hybrid 전략).
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SkPipeline
    from stage2_ksvm import generate_window_dataset

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    mode_indices = defaultdict(list)
    for i, r in enumerate(rows):
        mode_indices[r['label']].append(i)

    all_features = np.array([csv_row_to_features(r) for r in rows], dtype=np.float32)

    W = window_size
    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]

    # 모드별로 블록 내 window 생성
    block_windows = {mode: [[] for _ in range(n_folds)] for mode in modes}

    for mode in modes:
        indices = mode_indices.get(mode, [])
        n = len(indices)
        if n < W:
            continue
        is_attack = 0 if mode == "Normal" else 1

        block_size = n // n_folds
        for b in range(n_folds):
            b_start = b * block_size
            b_end = b_start + block_size if b < n_folds - 1 else n

            for s in range(b_start, b_end - W + 1):
                win_idx = indices[s:s + W]
                feat = extract_window_features(all_features[win_idx])
                block_windows[mode][b].append((feat, is_attack))

    print(f"\n{'='*60}")
    print(f"Stage 2 KSVM — Blocked Temporal {n_folds}-Fold CV")
    print(f"  합성 window 혼합: {n_synth_per_class}개/class")
    print(f"  블록 경계 = natural gap (window가 블록을 넘지 않음)")
    print(f"{'='*60}")

    for mode in modes:
        counts = [len(block_windows[mode][b]) for b in range(n_folds)]
        print(f"  {mode:<12s}: blocks={counts}, total={sum(counts)}")

    all_fold_results = []

    for fold in range(n_folds):
        X_train_real, y_train_real = [], []
        X_test, y_test = [], []
        test_mode_map = []

        for mode in modes:
            for b in range(n_folds):
                for feat, label in block_windows[mode][b]:
                    if b == fold:
                        X_test.append(feat)
                        y_test.append(label)
                        test_mode_map.append(mode)
                    else:
                        X_train_real.append(feat)
                        y_train_real.append(label)

        # 합성 window 생성 + 실측 train 혼합
        X_synth, y_synth = generate_window_dataset(n_synth_per_class, W)

        X_train = np.concatenate([
            np.array(X_train_real, dtype=np.float32),
            X_synth,
        ])
        y_train = np.concatenate([
            np.array(y_train_real, dtype=np.int32),
            y_synth,
        ])

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  Fold {fold+1}: SKIP")
            continue

        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)

        pipeline = SkPipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=0.5, gamma="scale",
                        probability=True, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        fold_result = {}
        test_mode_arr = np.array(test_mode_map)
        for mode in modes:
            mask = test_mode_arr == mode
            if mask.sum() == 0:
                continue
            mode_preds = preds[mask]
            if mode == "Normal":
                fa = float((mode_preds == 1).mean())
                fold_result[mode] = {"fa_rate": fa, "n": int(mask.sum())}
            else:
                det = float((mode_preds == 1).mean())
                fold_result[mode] = {"det_rate": det, "n": int(mask.sum())}

        all_fold_results.append(fold_result)
        n_real = len(X_train_real)
        acc = float((preds == y_test).mean())
        print(f"  Fold {fold+1}: train={len(X_train)}(real={n_real}+synth={len(X_synth)}), "
              f"test={len(X_test)}, acc={acc:.4f}")

    # 전체 결과 집계
    print(f"\n{'='*60}")
    print(f"Stage 2 KSVM — Blocked {n_folds}-Fold CV 결과 (weighted avg)")
    print(f"{'='*60}")

    s2_results = {}
    for mode in modes:
        fold_vals, fold_ns = [], []
        for fr in all_fold_results:
            if mode in fr:
                key = "fa_rate" if mode == "Normal" else "det_rate"
                fold_vals.append(fr[mode][key])
                fold_ns.append(fr[mode]["n"])

        if not fold_vals:
            continue

        total_n = sum(fold_ns)
        weighted_avg = sum(v * n for v, n in zip(fold_vals, fold_ns)) / total_n
        fold_std = float(np.std(fold_vals))

        if mode == "Normal":
            s2_results[mode] = {"fa_rate": weighted_avg, "n": total_n,
                                "fa_std": fold_std, "per_fold": fold_vals}
            print(f"  Normal FA:     {weighted_avg:.4f} ± {fold_std:.4f} "
                  f"(N={total_n}, folds={[f'{v:.4f}' for v in fold_vals]})")
        else:
            s2_results[mode] = {"det_rate": weighted_avg, "n": total_n,
                                "det_std": fold_std, "per_fold": fold_vals}
            print(f"  {mode:<12s}:  {weighted_avg:.4f} ± {fold_std:.4f} "
                  f"(N={total_n}, folds={[f'{v:.4f}' for v in fold_vals]})")

    return s2_results


def save_results(s1_results, s2_results, total_samples):
    """모든 evaluation 결과를 JSON으로 저장"""
    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]

    results = {
        "total_samples": total_samples,
        "eval_method": "blocked_temporal_5fold_cv",
        "modes": {},
    }

    for mode in modes:
        s1 = s1_results.get(mode, {})
        s2 = s2_results.get(mode, {})

        entry = {"n_samples": s1.get("n", 0)}

        if mode == "Normal":
            s1_fa = s1.get("fa_rate", 0)
            s1_std = s1.get("fa_std", 0)
            s2_fa = s2.get("fa_rate", 0)
            s2_std = s2.get("fa_std", 0)
            combined_fa = s1_fa + (1 - s1_fa) * s2_fa
            entry["s1_fa_rate"] = round(s1_fa * 100, 2)
            entry["s1_fa_std"] = round(s1_std * 100, 2)
            entry["s2_fa_rate"] = round(s2_fa * 100, 2)
            entry["s2_fa_std"] = round(s2_std * 100, 2)
            entry["combined_fa_rate"] = round(combined_fa * 100, 2)
            if "per_fold" in s1:
                entry["s1_per_fold"] = [round(v * 100, 2) for v in s1["per_fold"]]
            if "per_fold" in s2:
                entry["s2_per_fold"] = [round(v * 100, 2) for v in s2["per_fold"]]
        else:
            s1_det = s1.get("det_rate", 0)
            s1_std = s1.get("det_std", 0)
            s2_det = s2.get("det_rate", 0)
            s2_std = s2.get("det_std", 0)
            combined_det = s1_det + (1 - s1_det) * s2_det
            entry["s1_det_rate"] = round(s1_det * 100, 2)
            entry["s1_det_std"] = round(s1_std * 100, 2)
            entry["s2_det_rate"] = round(s2_det * 100, 2)
            entry["s2_det_std"] = round(s2_std * 100, 2)
            entry["combined_det_rate"] = round(combined_det * 100, 2)
            if "per_fold" in s1:
                entry["s1_per_fold"] = [round(v * 100, 2) for v in s1["per_fold"]]
            if "per_fold" in s2:
                entry["s2_per_fold"] = [round(v * 100, 2) for v in s2["per_fold"]]

        entry["s2_n_windows"] = s2.get("n", 0)
        results["modes"][mode] = entry

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[저장] eval_results.json → {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "..", "kpm_fdd_alldata.csv")

    print(f"실측 데이터: {csv_path}")

    # ── Stage 1: Blocked Temporal 5-Fold CV ──
    s1_results = eval_stage1_blocked_cv(csv_path, n_folds=5,
                                         n_per_class=500, epochs=50)

    # ── Stage 1: 기존 모델 참고용 ──
    X, y_raw, s1_preds, s1_old = eval_stage1_real(csv_path)

    # ── Stage 2: Blocked Temporal 5-Fold CV ──
    s2_results = eval_stage2_blocked_cv(csv_path, n_folds=5)

    # ── Combined 평가 ──
    print("\n" + "=" * 60)
    print("Combined Stage 1+2 Detection Rate")
    print("  S1: Blocked 5-Fold CV | S2: Blocked 5-Fold CV")
    print("=" * 60)
    modes = ["Normal", "Constant", "Random", "Reactive",
             "Deceptive", "PSS", "PDCCH", "DMRS"]
    for mode in modes:
        s1 = s1_results.get(mode, {})
        s2 = s2_results.get(mode, {})
        if mode == "Normal":
            s1_fa = s1.get("fa_rate", 0)
            s1_std = s1.get("fa_std", 0)
            s2_fa = s2.get("fa_rate", 0)
            s2_std = s2.get("fa_std", 0)
            combined_fa = s1_fa + (1 - s1_fa) * s2_fa
            print(f"  Normal FA: S1={s1_fa:.4f}±{s1_std:.4f}, "
                  f"S2={s2_fa:.4f}±{s2_std:.4f}, Combined≈{combined_fa:.4f}")
        else:
            s1_det = s1.get("det_rate", 0)
            s1_std = s1.get("det_std", 0)
            s2_det = s2.get("det_rate", 0)
            s2_std = s2.get("det_std", 0)
            combined_det = s1_det + (1 - s1_det) * s2_det
            n = s1.get("n", "?")
            print(f"  {mode:<12s}: S1={s1_det:.4f}±{s1_std:.4f}, "
                  f"S2={s2_det:.4f}±{s2_std:.4f}, Combined≈{combined_det:.4f} (N={n})")

    # JSON으로 결과 저장
    with open(csv_path) as f:
        total_samples = sum(1 for _ in csv.DictReader(f))
    save_results(s1_results, s2_results, total_samples)
