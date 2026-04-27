"""
실측 Normal / Attack 데이터 수집 → Stage 1 MLP 재학습

사용법:
  # Step 1a: UE만 켠 상태에서 Normal 수집 (30초)
  python3 collect_and_retrain.py --collect --duration 30

  # Step 1b: Jammer를 켠 상태에서 Attack 수집 (모드별)
  python3 collect_and_retrain.py --collect-attack --label Constant --duration 30
  python3 collect_and_retrain.py --collect-attack --label Random --duration 30
  python3 collect_and_retrain.py --collect-attack --label Reactive --duration 30
  python3 collect_and_retrain.py --collect-attack --label Deceptive --duration 30

  # Step 2: 수집된 데이터로 재학습 (실측 Normal + 실측 Attack + 합성 보충)
  python3 collect_and_retrain.py --retrain

  # Step 1a+2 한 번에
  python3 collect_and_retrain.py --collect --retrain --duration 30
"""

import argparse
import os
import pickle
import time
import numpy as np
import uhd
import yaml
import torch

from stage1_mlp import (
    extract_features, generate_synthetic_dataset,
    JammingMLP, LABELS, LABEL_IDX, NUM_FEATURES, NUM_CLASSES,
    MODEL_PATH, SCALER_PATH, CHUNK, RATE
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
REAL_NORMAL_PATH = os.path.join(os.path.dirname(__file__), "real_normal_features.npy")
REAL_ATTACK_DIR  = os.path.join(os.path.dirname(__file__), "real_attack_features")

RATE_HW  = 20e6
FREQ_HW  = 3.5e9
GAIN_HW  = 45
ANT_HW   = "RX2"

ATTACK_LABELS = ["Constant", "Random", "Reactive", "Deceptive"]


# ─────────────────────────────────────────────
# 1. 실측 Normal I/Q 피처 수집
# ─────────────────────────────────────────────
def collect_normal(duration: int = 30):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    addr = cfg["network"]["nodes"]["instance2_classifier"]["usrp"]["ip"]

    print(f"[Collect] USRP 연결: {addr}")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    usrp.set_rx_rate(RATE_HW)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(FREQ_HW))
    usrp.set_rx_gain(GAIN_HW)
    usrp.set_rx_antenna(ANT_HW, 0)
    time.sleep(0.2)

    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)

    buf  = np.zeros((1, CHUNK), dtype=np.complex64)
    meta = uhd.types.RXMetadata()

    features = []
    t_end = time.time() + duration
    n_chunk = 0

    print(f"[Collect] {duration}초간 Normal (UE only) 신호 수집 시작...")
    print(f"[Collect] !! 이 시간 동안 Jammer는 반드시 꺼져 있어야 합니다 !!\n")

    while time.time() < t_end:
        n = streamer.recv(buf, meta)
        if meta.error_code != uhd.types.RXMetadataErrorCode.none or n == 0:
            continue
        feat = extract_features(buf[0])
        features.append(feat)
        n_chunk += 1
        if n_chunk % 500 == 0:
            elapsed = duration - (t_end - time.time())
            print(f"  [{elapsed:5.1f}s] 수집: {n_chunk}청크 | "
                  f"RSSI: {feat[0]:+.2f} dB | Flatness: {feat[2]:.4f}")

    cmd_stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(cmd_stop)

    X_real = np.array(features, dtype=np.float32)
    np.save(REAL_NORMAL_PATH, X_real)
    print(f"\n[Collect] 완료: {n_chunk}개 청크 저장 → {REAL_NORMAL_PATH}")
    print(f"[Collect] RSSI  — mean: {X_real[:,0].mean():+.2f} dB, "
          f"std: {X_real[:,0].std():.3f}")
    print(f"[Collect] Flatness — mean: {X_real[:,2].mean():.4f}, "
          f"std: {X_real[:,2].std():.4f}")
    return X_real


# ─────────────────────────────────────────────
# 1b. 실측 Attack I/Q 피처 수집
# ─────────────────────────────────────────────
def collect_attack(label: str, duration: int = 30):
    if label not in ATTACK_LABELS:
        print(f"[Error] 지원하지 않는 공격 유형: {label}")
        print(f"        지원 유형: {', '.join(ATTACK_LABELS)}")
        return None

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    addr = cfg["network"]["nodes"]["instance2_classifier"]["usrp"]["ip"]

    print(f"[Collect-Attack] USRP 연결: {addr}")
    usrp = uhd.usrp.MultiUSRP(f"addr={addr}")
    usrp.set_rx_rate(RATE_HW)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(FREQ_HW))
    usrp.set_rx_gain(GAIN_HW)
    usrp.set_rx_antenna(ANT_HW, 0)
    time.sleep(0.2)

    streamer = usrp.get_rx_stream(uhd.usrp.StreamArgs("fc32", "sc16"))
    cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    cmd.stream_now = True
    streamer.issue_stream_cmd(cmd)

    buf  = np.zeros((1, CHUNK), dtype=np.complex64)
    meta = uhd.types.RXMetadata()

    features = []
    t_end = time.time() + duration
    n_chunk = 0

    print(f"[Collect-Attack] {duration}초간 {label} 공격 신호 수집 시작...")
    print(f"[Collect-Attack] !! 이 시간 동안 Jammer가 '{label}' 모드로 켜져 있어야 합니다 !!\n")

    while time.time() < t_end:
        n = streamer.recv(buf, meta)
        if meta.error_code != uhd.types.RXMetadataErrorCode.none or n == 0:
            continue
        feat = extract_features(buf[0])
        features.append(feat)
        n_chunk += 1
        if n_chunk % 500 == 0:
            elapsed = duration - (t_end - time.time())
            print(f"  [{elapsed:5.1f}s] 수집: {n_chunk}청크 | "
                  f"RSSI: {feat[0]:+.2f} dB | Flatness: {feat[2]:.4f}")

    cmd_stop = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(cmd_stop)

    X_real = np.array(features, dtype=np.float32)

    os.makedirs(REAL_ATTACK_DIR, exist_ok=True)
    save_path = os.path.join(REAL_ATTACK_DIR, f"real_{label.lower()}_features.npy")
    np.save(save_path, X_real)
    print(f"\n[Collect-Attack] 완료: {n_chunk}개 청크 저장 → {save_path}")
    print(f"[Collect-Attack] RSSI  — mean: {X_real[:,0].mean():+.2f} dB, "
          f"std: {X_real[:,0].std():.3f}")
    print(f"[Collect-Attack] Flatness — mean: {X_real[:,2].mean():.4f}, "
          f"std: {X_real[:,2].std():.4f}")
    return X_real


# ─────────────────────────────────────────────
# 2. 실측 Normal + 실측/합성 Attack 혼합 재학습
# ─────────────────────────────────────────────
def retrain(n_synth_attack: int = 1000, epochs: int = 80, lr: float = 0.001):
    if not os.path.exists(REAL_NORMAL_PATH):
        print(f"[Retrain] 실측 Normal 데이터 없음: {REAL_NORMAL_PATH}")
        print("[Retrain] 먼저 --collect 를 실행하세요.")
        return

    # 실측 Normal 로드
    X_real_normal = np.load(REAL_NORMAL_PATH)
    n_real = len(X_real_normal)
    y_real = np.zeros(n_real, dtype=np.int64)  # label=0 (Normal)
    print(f"[Retrain] 실측 Normal: {n_real}개")

    # 실측 Attack 데이터 로드 (있으면 사용)
    X_real_attack_list, y_real_attack_list = [], []
    real_attack_counts = {}
    for label in ATTACK_LABELS:
        fpath = os.path.join(REAL_ATTACK_DIR, f"real_{label.lower()}_features.npy")
        if os.path.exists(fpath):
            X_atk = np.load(fpath)
            real_attack_counts[label] = len(X_atk)
            X_real_attack_list.append(X_atk)
            y_real_attack_list.extend([LABEL_IDX[label]] * len(X_atk))
            print(f"[Retrain] 실측 {label}: {len(X_atk)}개 로드")

    # 합성 Attack 데이터 생성 (실측 없는 클래스만 합성으로 보충)
    from stage1_mlp import simulate_chunk
    X_synth_list, y_synth_list = [], []
    for label in ATTACK_LABELS:
        if label in real_attack_counts:
            # 실측이 있으면 합성은 소량만 보충 (다양성 확보)
            n_synth = min(n_synth_attack // 4, 200)
            print(f"[Retrain] 합성 {label}: {n_synth}개 (실측 보충)")
        else:
            n_synth = n_synth_attack
            print(f"[Retrain] 합성 {label}: {n_synth}개 (실측 없음 — 합성만 사용)")
        for _ in range(n_synth):
            gain = np.random.uniform(0.5, 2.0)
            iq   = simulate_chunk(label, gain_factor=gain)
            feat = extract_features(iq)
            X_synth_list.append(feat)
            y_synth_list.append(LABEL_IDX[label])

    # 모든 데이터 합치기
    all_X = [X_real_normal]
    all_y = [y_real]
    if X_real_attack_list:
        all_X.append(np.vstack(X_real_attack_list))
        all_y.append(np.array(y_real_attack_list, dtype=np.int64))
    if X_synth_list:
        all_X.append(np.array(X_synth_list, dtype=np.float32))
        all_y.append(np.array(y_synth_list, dtype=np.int64))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    n_real_atk = sum(real_attack_counts.values()) if real_attack_counts else 0
    n_synth_total = len(X_synth_list)
    print(f"[Retrain] 전체 학습 데이터: {len(X)}개 "
          f"(실측Normal:{n_real} / 실측Attack:{n_real_atk} / 합성Attack:{n_synth_total})")

    # 클래스 분포 출력
    for i, lbl in enumerate(LABELS):
        cnt = (y == i).sum()
        print(f"  {lbl:<12s}: {cnt}개")

    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Retrain] Scaler 저장: {SCALER_PATH}")

    # 학습
    dataset = TensorDataset(
        torch.from_numpy(X_scaled), torch.from_numpy(y)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = JammingMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\n[Retrain] 학습 시작 (device={device}, epochs={epochs})\n")
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {total_loss/len(loader):.4f} | "
                  f"Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[Retrain] 모델 저장: {MODEL_PATH}")
    print(f"[Retrain] 최종 학습 정확도: {correct/total:.4f}")

    # 간단 검증 (실측 Normal)
    model.eval()
    X_test_scaled = scaler.transform(X_real_normal).astype(np.float32)
    inp = torch.from_numpy(X_test_scaled).to(device)
    with torch.no_grad():
        pred = model(inp).argmax(1).cpu().numpy()
    normal_acc = (pred == 0).mean()
    print(f"\n[Retrain] 실측 Normal 재확인 정확도: {normal_acc:.4f} "
          f"({(pred==0).sum()}/{len(pred)} 정상 판정)")
    if normal_acc < 0.9:
        print("[WARN] Normal 정확도가 낮습니다. --duration 을 늘려 더 많이 수집하거나 "
              "epoch 수를 늘려보세요.")
    else:
        print("[OK] 재학습 완료 — pipeline_runner.py 를 다시 실행하세요.")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="실측 Normal/Attack 수집 + Stage 1 MLP 재학습")
    parser.add_argument("--collect", action="store_true",
                        help="USRP에서 실측 Normal 신호 수집")
    parser.add_argument("--collect-attack", action="store_true",
                        help="USRP에서 실측 Attack 신호 수집 (--label 필수)")
    parser.add_argument("--label", type=str, default=None,
                        choices=["Constant", "Random", "Reactive", "Deceptive"],
                        help="수집할 공격 유형 (--collect-attack 시 필수)")
    parser.add_argument("--retrain", action="store_true",
                        help="수집된 실측 데이터로 Stage 1 재학습")
    parser.add_argument("--duration", type=int, default=30,
                        help="수집 시간 (초, 기본: 30)")
    parser.add_argument("--n-synth-attack", type=int, default=1000,
                        help="합성 Attack 샘플 수/클래스 (기본: 1000)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="학습 에포크 (기본: 80)")
    args = parser.parse_args()

    if not args.collect and not args.collect_attack and not args.retrain:
        parser.print_help()
    if args.collect:
        collect_normal(args.duration)
    if args.collect_attack:
        if not args.label:
            print("[Error] --collect-attack 사용 시 --label 을 지정하세요.")
            print(f"        예: --collect-attack --label Deceptive --duration 30")
        else:
            collect_attack(args.label, args.duration)
    if args.retrain:
        retrain(args.n_synth_attack, args.epochs)
