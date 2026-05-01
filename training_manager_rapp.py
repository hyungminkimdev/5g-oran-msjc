"""
Training Manager rApp — MSJC Phase 6 (Rahman et al. 2025, Section III-B)
Non-RT RIC rApp: ML 모델 라이프사이클 오케스트레이션

동작:
  1. xApp 정확도 모니터링 (InfluxDB labeled_kpis vs detection 비교)
  2. 정확도 < 30% 또는 주기적 스케줄 → 재학습 트리거
  3. ClearML: template task clone → labeled data 연결 → queue에 enqueue
  4. 학습 완료 대기 → model registry에서 최신 모델 URL 획득
  5. A1 인터페이스로 xApp에 MODEL_UPDATE 알림

MSJC 확장:
  - Rahman은 단일 모델 → MSJC는 Stage 1(MLP)/2(KSVM)/3(MobileNet) 3개
  - 재학습 대상: Stage 1 MLP (5-class, 핵심 분류기)
  - Stage 2/3은 주기적 재학습 (별도 트리거)

ClearML 미설치 시:
  - Standalone 모드: 직접 stage1_mlp.py를 호출하여 재학습
  - 모델 파일 갱신 → xApp hot-reload가 mtime 감지하여 자동 로드

실행:
  python3 training_manager_rapp.py [--once] [--interval 300]
"""

import time
import os
import yaml
import json
import requests
import numpy as np
from datetime import datetime, timezone, timedelta

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
PROJECT_DIR = os.path.dirname(__file__)

# 논문 하이퍼파라미터
ACCURACY_THRESHOLD = 0.30   # 30% 미만이면 재학습 (Rahman et al.)
CHECK_WINDOW_MIN = 10       # 최근 10분 정확도 확인
RETRAIN_COOLDOWN = 600      # 재학습 최소 간격 (10분)

# xApp A1 엔드포인트
XAPP_A1_URL = "http://127.0.0.1:5000/a1/model_update"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 정확도 모니터링
# ─────────────────────────────────────────────
class AccuracyMonitor:
    """InfluxDB에서 xApp 추론 결과와 GMM 레이블 비교하여 정확도 계산"""

    def __init__(self, influx_cfg: dict):
        self._cfg = influx_cfg
        self._client = None
        self._query_api = None
        self._init_client()

    def _init_client(self):
        try:
            from influxdb_client import InfluxDBClient
            self._client = InfluxDBClient(
                url=self._cfg["url"],
                token=self._cfg["token"],
                org=self._cfg["org"])
            self._query_api = self._client.query_api()
        except Exception as e:
            print(f"[TrainMgr] InfluxDB 연결 실패: {e}")

    def check_accuracy(self, window_min: int = CHECK_WINDOW_MIN) -> dict:
        """최근 N분 추론 정확도 계산"""
        if not self._query_api:
            return {"accuracy": 1.0, "count": 0, "status": "no_influx"}

        # detection measurement에서 최근 데이터
        query = f'''
        from(bucket: "{self._cfg['bucket']}")
          |> range(start: -{window_min}m)
          |> filter(fn: (r) => r._measurement == "detection")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"])
        '''
        try:
            tables = self._query_api.query(query, org=self._cfg["org"])
            rows = []
            for table in tables:
                for record in table.records:
                    rows.append(record.values)

            if len(rows) < 10:
                return {"accuracy": 1.0, "count": len(rows), "status": "insufficient"}

            # FN rate 기반 정확도 추정
            fn_count = sum(1 for r in rows if r.get("fn_caught", False))
            total = len(rows)
            fn_rate = fn_count / total
            accuracy = 1.0 - fn_rate

            return {
                "accuracy": accuracy,
                "fn_rate": fn_rate,
                "count": total,
                "status": "ok",
            }
        except Exception as e:
            print(f"[TrainMgr] 정확도 조회 실패: {e}")
            return {"accuracy": 1.0, "count": 0, "status": "error"}

    def close(self):
        if self._client:
            self._client.close()


# ─────────────────────────────────────────────
# 재학습 오케스트레이션
# ─────────────────────────────────────────────
class TrainingManagerRApp:
    """ML 모델 재학습 + 배포 오케스트레이션"""

    def __init__(self, influx_cfg: dict):
        self._monitor = AccuracyMonitor(influx_cfg)
        self._last_retrain_time = 0
        self._clearml_available = self._check_clearml()

    def _check_clearml(self) -> bool:
        try:
            from clearml import Task
            return True
        except ImportError:
            print("[TrainMgr] ClearML 미설치 — Standalone 재학습 모드")
            return False

    def run_once(self) -> dict:
        """1회 모니터링 + 필요시 재학습 사이클"""
        ts = datetime.now().strftime("%H:%M:%S")

        # Step 1: 정확도 확인
        acc_result = self._monitor.check_accuracy()
        accuracy = acc_result["accuracy"]
        count = acc_result["count"]

        print(f"[{ts}] [TrainMgr] 정확도={accuracy:.2%} ({count}개 샘플)")

        # Step 2: 재학습 필요 여부 판단
        need_retrain = False
        reason = ""

        if accuracy < ACCURACY_THRESHOLD and count >= 10:
            need_retrain = True
            reason = f"정확도 {accuracy:.2%} < 임계값 {ACCURACY_THRESHOLD:.0%}"

        # 쿨다운 확인
        if need_retrain:
            elapsed = time.time() - self._last_retrain_time
            if elapsed < RETRAIN_COOLDOWN:
                print(f"[{ts}] [TrainMgr] 쿨다운 중 ({RETRAIN_COOLDOWN - elapsed:.0f}초 남음)")
                return {"action": "cooldown", **acc_result}

        if not need_retrain:
            return {"action": "monitor", **acc_result}

        # Step 3: 재학습 실행
        print(f"[{ts}] [TrainMgr] 재학습 트리거: {reason}")
        retrain_result = self._execute_retrain()
        self._last_retrain_time = time.time()

        # Step 4: xApp에 모델 업데이트 알림
        if retrain_result.get("success"):
            self._notify_xapp(retrain_result)

        return {"action": "retrained", **acc_result, **retrain_result}

    def _execute_retrain(self) -> dict:
        """재학습 실행 (ClearML 또는 Standalone)"""
        if self._clearml_available:
            return self._retrain_clearml()
        else:
            return self._retrain_standalone()

    def _retrain_clearml(self) -> dict:
        """ClearML 기반 재학습: template task clone → enqueue"""
        try:
            from clearml import Task

            ts = int(time.time())
            task = Task.create(
                project_name="msjc-5g-oran",
                task_name=f"msjc-stage1-retrain-{ts}",
                task_type=Task.TaskTypes.training,
            )
            task.set_parameter("triggered_by", "training_manager_rapp")
            task.set_parameter("data_source", "kpm_fdd_alldata.csv")
            task.set_parameter("n_per_class", 1000)
            task.set_parameter("epochs", 100)

            Task.enqueue(task, queue_name="default")
            print(f"  [TrainMgr] ClearML task enqueued: {task.id}")

            return {"success": True, "method": "clearml", "task_id": task.id}
        except Exception as e:
            print(f"  [TrainMgr] ClearML 실패: {e} — Standalone fallback")
            return self._retrain_standalone()

    def _retrain_standalone(self) -> dict:
        """Standalone 재학습: stage1_mlp.py 직접 호출"""
        import subprocess

        csv_path = os.path.join(PROJECT_DIR, "kpm_fdd_alldata.csv")
        cmd = [
            "python3", os.path.join(PROJECT_DIR, "stage1_mlp.py"),
            "--retrain", "--n-per-class", "1000", "--epochs", "100",
        ]
        if os.path.exists(csv_path):
            cmd.extend(["--real-csv", csv_path])

        print(f"  [TrainMgr] Standalone 재학습 시작...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            success = result.returncode == 0
            if success:
                print(f"  [TrainMgr] 재학습 완료")
                # 마지막 accuracy 추출
                for line in result.stdout.split("\n"):
                    if "최종 학습 정확도" in line:
                        print(f"  {line.strip()}")
            else:
                print(f"  [TrainMgr] 재학습 실패: {result.stderr[-200:]}")

            return {
                "success": success,
                "method": "standalone",
                "model_path": os.path.join(PROJECT_DIR, "stage1_mlp.pth"),
            }
        except subprocess.TimeoutExpired:
            print(f"  [TrainMgr] 재학습 타임아웃 (300초)")
            return {"success": False, "method": "standalone", "error": "timeout"}

    def _notify_xapp(self, retrain_result: dict):
        """A1 인터페이스로 xApp에 모델 업데이트 알림"""
        payload = {
            "action": "MODEL_UPDATE",
            "stage": "stage1",
            "model_path": retrain_result.get("model_path", ""),
            "task_id": retrain_result.get("task_id", ""),
            "method": retrain_result.get("method", ""),
            "timestamp": int(time.time()),
        }

        try:
            resp = requests.post(XAPP_A1_URL, json=payload, timeout=5)
            if resp.status_code == 200:
                print(f"  [TrainMgr] A1 → xApp: MODEL_UPDATE 전송 성공")
            else:
                print(f"  [TrainMgr] A1 → xApp: HTTP {resp.status_code}")
        except requests.ConnectionError:
            # xApp A1 미실행 → hot-reload가 mtime으로 감지
            print(f"  [TrainMgr] A1 연결 불가 — xApp hot-reload가 mtime으로 감지 예정")
        except Exception as e:
            print(f"  [TrainMgr] A1 알림 실패: {e}")

    def close(self):
        self._monitor.close()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MSJC Training Manager rApp — ML 재학습 오케스트레이션")
    parser.add_argument("--once", action="store_true", help="1회 실행 후 종료")
    parser.add_argument("--interval", type=int, default=300,
                        help="모니터링 주기 (초, 기본: 300)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="정확도 무관하게 즉시 재학습")
    args = parser.parse_args()

    cfg = load_config()
    influx_cfg = cfg["data"]["influxdb"]
    manager = TrainingManagerRApp(influx_cfg)

    try:
        if args.force_retrain:
            print("[TrainMgr] 강제 재학습 실행")
            result = manager._execute_retrain()
            if result.get("success"):
                manager._notify_xapp(result)
            print(f"[TrainMgr] 결과: {result}")
        elif args.once:
            manager.run_once()
        else:
            print(f"[TrainMgr] 시작 — {args.interval}초 간격 모니터링")
            while True:
                manager.run_once()
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[TrainMgr] 종료")
    finally:
        manager.close()
