"""
Labeler rApp — MSJC Phase 6 (Rahman et al. 2025, Section III-A)
Non-RT RIC rApp: 비지도 GMM 기반 KPI 자동 레이블링

동작:
  1. InfluxDB에서 최근 KPI 배치를 가져옴 (sliding window)
  2. Moving Average 스무딩
  3. Standard Scaling 정규화
  4. ARC (Average Rate of Change) 계산 → 드리프트 감지
  5. GMM 클러스터링 → Normal(0) / Attack(1) 레이블
  6. 레이블된 데이터를 InfluxDB labeled_kpis measurement에 저장

MSJC 확장:
  - Rahman은 4 KPI (UL SNR, MCS, bitrate, BLER) → MSJC는 8 KPI
  - Rahman은 binary → MSJC는 binary 후 xApp Stage 1이 5-class 세분류
  - ARC 임계값 τ = 0.0004 (논문 기본값, 테스트베드 조건에 따라 튜닝)

실행:
  python3 labeler_rapp.py [--once] [--interval 30]
"""

import time
import yaml
import os
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# 논문 하이퍼파라미터
BATCH_SIZE = 30          # sliding window 크기
SMOOTHING_WINDOW = 5     # Moving Average 윈도우
ARC_THRESHOLD = 0.0004   # 드리프트 감지 임계값 τ
GMM_N_COMPONENTS = 2     # Normal(0) vs Attack(1)

# MSJC 8-dim KPI에서 레이블링에 사용할 필드
LABEL_FEATURES = ["sinr", "bler", "cqi_mean", "s1_confidence"]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# InfluxDB 연동
# ─────────────────────────────────────────────
class InfluxKPISource:
    """InfluxDB에서 KPI 읽기/쓰기"""

    def __init__(self, cfg: dict):
        from influxdb_client import InfluxDBClient
        self._client = InfluxDBClient(
            url=cfg["url"], token=cfg["token"], org=cfg["org"])
        self._query_api = self._client.query_api()
        self._bucket = cfg["bucket"]
        self._org = cfg["org"]

        # 쓰기용
        from influxdb_client.client.write_api import SYNCHRONOUS
        self._write_api = self._client.write_api(write_options=SYNCHRONOUS)

    def fetch_recent_kpis(self, minutes: int = 5) -> list[dict]:
        """최근 N분간 detection measurement에서 KPI 조회"""
        query = f'''
        from(bucket: "{self._bucket}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r._measurement == "detection")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
          |> limit(n: {BATCH_SIZE * 3})
        '''
        tables = self._query_api.query(query, org=self._org)
        rows = []
        for table in tables:
            for record in table.records:
                row = record.values
                rows.append({
                    "timestamp": row.get("_time", datetime.now(timezone.utc)),
                    "sinr": float(row.get("sinr", 0) or 0),
                    "bler": float(row.get("spectral_flatness", 0) or 0),  # bler 필드
                    "cqi_mean": float(row.get("rssi", 0) or 0),  # cqi 필드
                    "s1_confidence": float(row.get("s1_confidence", 0) or 0),
                    "s1_label": str(row.get("s1_label", "")),
                    "final_verdict": str(row.get("final_verdict", "")),
                })
        return rows

    def write_labels(self, labeled_rows: list[dict]):
        """레이블된 KPI를 labeled_kpis measurement에 저장"""
        from influxdb_client import Point, WritePrecision
        points = []
        for row in labeled_rows:
            p = (Point("labeled_kpis")
                 .tag("gmm_label", row["gmm_label"])
                 .field("sinr", row["sinr"])
                 .field("bler", row["bler"])
                 .field("cqi_mean", row["cqi_mean"])
                 .field("s1_confidence", row["s1_confidence"])
                 .field("gmm_prob_normal", row["gmm_prob_normal"])
                 .field("gmm_prob_attack", row["gmm_prob_attack"])
                 .time(row["timestamp"], WritePrecision.US))
            points.append(p)
        self._write_api.write(bucket=self._bucket, record=points)

    def close(self):
        self._write_api.close()
        self._client.close()


# ─────────────────────────────────────────────
# Labeler 핵심 로직 (Rahman Algorithm 1)
# ─────────────────────────────────────────────
class LabelerRApp:
    """GMM 기반 비지도 자동 레이블러"""

    def __init__(self, influx_cfg: dict):
        self._source = InfluxKPISource(influx_cfg)
        self._gmm = None
        self._scaler = StandardScaler()
        self._prev_means = None  # ARC 계산용 이전 GMM 평균
        self._fitted = False

    def run_once(self) -> dict:
        """1회 레이블링 사이클 실행. 결과 통계 반환."""
        ts = datetime.now().strftime("%H:%M:%S")

        # Step 1: InfluxDB에서 KPI 가져오기
        rows = self._source.fetch_recent_kpis(minutes=5)
        if len(rows) < BATCH_SIZE:
            print(f"[{ts}] [Labeler] 데이터 부족: {len(rows)}개 < {BATCH_SIZE}")
            return {"status": "insufficient_data", "count": len(rows)}

        # 최근 BATCH_SIZE개만 사용
        rows = rows[-BATCH_SIZE:]

        # Step 2: Feature matrix 구성
        X_raw = np.array([[r["sinr"], r["bler"], r["cqi_mean"], r["s1_confidence"]]
                          for r in rows], dtype=np.float32)

        # Step 3: Moving Average 스무딩
        X_smooth = self._moving_average(X_raw, SMOOTHING_WINDOW)

        # Step 4: Standard Scaling
        X_scaled = self._scaler.fit_transform(X_smooth)

        # Step 5: ARC 계산 → 드리프트 감지
        need_retrain = self._check_drift(X_scaled)

        # Step 6: GMM 학습 (첫 실행 또는 드리프트 감지 시)
        if not self._fitted or need_retrain:
            self._fit_gmm(X_scaled)

        # Step 7: GMM 클러스터링
        labels = self._gmm.predict(X_scaled)
        probs = self._gmm.predict_proba(X_scaled)

        # 클러스터 매핑: BLER 높은 쪽 = Attack
        # GMM 클러스터 0,1 중 어느 것이 Attack인지 자동 판별
        cluster_bler_means = [
            np.mean(X_raw[labels == c, 1]) for c in range(GMM_N_COMPONENTS)
        ]
        attack_cluster = int(np.argmax(cluster_bler_means))
        normal_cluster = 1 - attack_cluster

        # Step 8: 레이블 생성 + InfluxDB 저장
        labeled = []
        n_normal, n_attack = 0, 0
        for i, row in enumerate(rows[-len(labels):]):
            gmm_label = "Attack" if labels[i] == attack_cluster else "Normal"
            row["gmm_label"] = gmm_label
            row["gmm_prob_normal"] = float(probs[i][normal_cluster])
            row["gmm_prob_attack"] = float(probs[i][attack_cluster])
            labeled.append(row)
            if gmm_label == "Normal":
                n_normal += 1
            else:
                n_attack += 1

        self._source.write_labels(labeled)

        result = {
            "status": "ok",
            "count": len(labeled),
            "normal": n_normal,
            "attack": n_attack,
            "drift_detected": need_retrain,
        }
        print(f"[{ts}] [Labeler] {len(labeled)}개 레이블 완료: "
              f"Normal={n_normal}, Attack={n_attack}, drift={need_retrain}")
        return result

    def _moving_average(self, X: np.ndarray, window: int) -> np.ndarray:
        """Moving Average 스무딩 (논문 Step 2)"""
        if len(X) < window:
            return X
        smoothed = np.zeros_like(X)
        for i in range(len(X)):
            start = max(0, i - window + 1)
            smoothed[i] = np.mean(X[start:i + 1], axis=0)
        return smoothed

    def _check_drift(self, X_scaled: np.ndarray) -> bool:
        """ARC (Average Rate of Change) 기반 드리프트 감지 (논문 Step 4)"""
        if self._prev_means is None:
            return True  # 첫 실행 → 학습 필요

        # ARC = 현재 GMM 평균과 이전 평균의 변화율
        current_means = np.mean(X_scaled, axis=0)
        arc = np.mean(np.abs(current_means - self._prev_means))

        if arc > ARC_THRESHOLD:
            print(f"  [Labeler] ARC={arc:.6f} > τ={ARC_THRESHOLD} → 드리프트 감지, GMM 재학습")
            return True
        return False

    def _fit_gmm(self, X_scaled: np.ndarray):
        """GMM 학습 (논문 Step 5)"""
        self._gmm = GaussianMixture(
            n_components=GMM_N_COMPONENTS,
            covariance_type="full",
            random_state=42,
            max_iter=100,
        )
        self._gmm.fit(X_scaled)
        self._prev_means = np.mean(X_scaled, axis=0)
        self._fitted = True
        print(f"  [Labeler] GMM 학습 완료: {GMM_N_COMPONENTS} components")

    def close(self):
        self._source.close()


# ─────────────────────────────────────────────
# Standalone 모드 (InfluxDB 없이 CSV 기반 테스트)
# ─────────────────────────────────────────────
def run_csv_demo(csv_path: str):
    """실측 CSV로 GMM 레이블링 데모 (InfluxDB 불필요)"""
    from kpi_feature_extractor import load_real_csv

    X_real, labels_real = load_real_csv(csv_path)
    print(f"[Demo] CSV 로드: {len(X_real)}개, 모드: {set(labels_real)}")

    # 8-dim 중 sinr(2), bler(3), cqi(6), 사용
    X_feat = X_real[:, [2, 3, 6]]

    # Moving Average
    W = SMOOTHING_WINDOW
    X_smooth = np.zeros_like(X_feat)
    for i in range(len(X_feat)):
        start = max(0, i - W + 1)
        X_smooth[i] = np.mean(X_feat[start:i + 1], axis=0)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_smooth)

    # GMM
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(X_scaled)
    pred = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    # 클러스터 매핑 (BLER 높은 쪽 = Attack)
    c0_bler = np.mean(X_feat[pred == 0, 1])
    c1_bler = np.mean(X_feat[pred == 1, 1])
    attack_c = 0 if c0_bler > c1_bler else 1
    gmm_labels = ["Attack" if p == attack_c else "Normal" for p in pred]

    # 정확도 평가 (실제 레이블 vs GMM 레이블)
    from collections import Counter
    print(f"\n[Demo] GMM 클러스터 분포: {Counter(gmm_labels)}")
    print(f"[Demo] 공격 클러스터(c={attack_c}): BLER mean={max(c0_bler, c1_bler):.3f}")
    print(f"[Demo] 정상 클러스터: BLER mean={min(c0_bler, c1_bler):.3f}")

    # 실제 레이블과 비교
    correct = 0
    for i, (real, gmm_l) in enumerate(zip(labels_real, gmm_labels)):
        real_binary = "Normal" if real == "Normal" else "Attack"
        if real_binary == gmm_l:
            correct += 1

    acc = correct / len(labels_real) * 100
    print(f"\n[Demo] Binary 정확도: {acc:.1f}% ({correct}/{len(labels_real)})")

    # 모드별 상세
    print("\n[Demo] 모드별 GMM 레이블 분포:")
    from collections import defaultdict
    mode_gmm = defaultdict(lambda: Counter())
    for real, gmm_l in zip(labels_real, gmm_labels):
        mode_gmm[real][gmm_l] += 1

    for mode in ['Normal', 'Constant', 'Random', 'Reactive', 'Deceptive']:
        cnt = mode_gmm[mode]
        total = sum(cnt.values())
        if total == 0:
            continue
        dist = ', '.join(f'{k}:{v}({v/total*100:.0f}%)' for k, v in cnt.most_common())
        print(f"  {mode:<12s}: {dist}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MSJC Labeler rApp — GMM 자동 레이블링")
    parser.add_argument("--once", action="store_true", help="1회 실행 후 종료")
    parser.add_argument("--interval", type=int, default=30, help="반복 주기 (초, 기본: 30)")
    parser.add_argument("--csv-demo", type=str, default=None,
                        help="InfluxDB 대신 CSV 파일로 데모 실행")
    args = parser.parse_args()

    if args.csv_demo:
        run_csv_demo(args.csv_demo)
        exit(0)

    cfg = load_config()
    influx_cfg = cfg["data"]["influxdb"]

    labeler = LabelerRApp(influx_cfg)

    try:
        if args.once:
            labeler.run_once()
        else:
            print(f"[Labeler] 시작 — {args.interval}초 간격 반복")
            while True:
                labeler.run_once()
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[Labeler] 종료")
    finally:
        labeler.close()
