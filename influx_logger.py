"""
MSJC InfluxDB Logger — 비동기 배치 KPI 로깅

탐지 루프(stage1_runner.py)에서 매 청크마다 호출됩니다.
influxdb-client의 batching WriteApi를 사용하여 백그라운드 스레드로
비동기 쓰기 → 탐지 루프 레이턴시에 영향 없음.

InfluxDB 미설치/미연결 시 graceful degradation:
로깅만 비활성화되고 탐지 루프는 정상 동작합니다.

사용법:
    logger = InfluxLogger(cfg["data"]["influxdb"])
    logger.log_detection(rssi=..., s1_label=..., ...)
    logger.close()   # 종료 시 잔여 배치 flush
"""

import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class InfluxLogger:
    def __init__(self, cfg: dict):
        """
        cfg: config.yaml의 data.influxdb 섹션
        """
        self._enabled = cfg.get("enabled", False)
        if not self._enabled:
            print("[InfluxLogger] 비활성화 (config: enabled=false)")
            return

        # token이 설정되지 않은 경우
        token = cfg.get("token", "")
        if not token or token == "PASTE_YOUR_TOKEN_HERE":
            print("[InfluxLogger] 비활성화 — config.yaml에 InfluxDB 토큰을 설정하세요")
            self._enabled = False
            return

        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import WriteOptions

            self._client = InfluxDBClient(
                url=cfg["url"],
                token=token,
                org=cfg["org"],
            )

            write_opts = WriteOptions(
                batch_size=cfg.get("batch_size", 100),
                flush_interval=cfg.get("flush_interval_ms", 5000),
                jitter_interval=1000,
                retry_interval=5000,
                max_retries=3,
                max_retry_delay=30000,
            )
            self._write_api = self._client.write_api(write_options=write_opts)
            self._bucket = cfg["bucket"]
            self._org = cfg["org"]

            # 연결 확인
            health = self._client.health()
            if health.status == "pass":
                print(f"[InfluxLogger] 연결 성공: {cfg['url']} (bucket: {self._bucket})")
            else:
                print(f"[InfluxLogger] 연결 경고: {health.message}")

        except ImportError:
            print("[InfluxLogger] 비활성화 — influxdb-client 미설치 (pip install influxdb-client)")
            self._enabled = False
        except Exception as e:
            print(f"[InfluxLogger] 초기화 실패: {e} — 로깅 비활성화")
            self._enabled = False

    def log_detection(self, *,
                      rssi: float,
                      spectral_flatness: float,
                      s1_label: str,
                      s1_confidence: float,
                      s2_confidence: float = 0.0,
                      s3_label: str = "",
                      s3_confidence: float = 0.0,
                      final_verdict: str,
                      fn_caught: bool = False,
                      latency_ms: float):
        """
        탐지 결과 1건을 InfluxDB에 기록합니다.
        비동기 배치 쓰기이므로 호출 자체는 즉시 반환됩니다.
        """
        if not self._enabled:
            return

        try:
            from influxdb_client import Point, WritePrecision

            point = (
                Point("detection")
                .tag("s1_label", s1_label)
                .tag("final_verdict", final_verdict)
                .tag("s3_label", s3_label)
                .field("rssi", rssi)
                .field("spectral_flatness", spectral_flatness)
                .field("s1_confidence", s1_confidence)
                .field("s2_confidence", s2_confidence)
                .field("s3_confidence", s3_confidence)
                .field("latency_ms", latency_ms)
                .field("fn_caught", fn_caught)
                .time(datetime.now(timezone.utc), WritePrecision.US)
            )

            self._write_api.write(bucket=self._bucket, record=point)
        except Exception as e:
            log.debug("[InfluxLogger] 쓰기 오류: %s", e)

    def close(self):
        """잔여 배치 flush 후 연결 종료"""
        if self._enabled:
            try:
                self._write_api.close()
                self._client.close()
                print("[InfluxLogger] 종료 (잔여 데이터 flush 완료)")
            except Exception as e:
                log.debug("[InfluxLogger] 종료 오류: %s", e)
