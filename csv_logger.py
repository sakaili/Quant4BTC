# csv_logger.py
import os
import threading

import pandas as pd


class CsvLogger:
    """�ѽ���ִ�м�¼�־û��� CSV �ļ���"""

    def __init__(self, path: str, logger):
        """����/������־�ļ�·����д���ͷ��"""
        self.path = path
        self.logger = logger
        self._ensure_headers()
        self._lock = threading.Lock()

    def _ensure_headers(self):
        """���ļ������ڻ�Ϊ�գ���д���׼��ͷ��"""
        if (not os.path.exists(self.path)) or (os.path.getsize(self.path) == 0):
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "signal",
                    "close",
                    "position",
                    "action",
                    "exec_price",
                    "fee",
                    "order_id",
                    "stop_loss",
                    "take_profit",
                    "best_factor",
                    "equity",
                    "mode",
                ]
            ).to_csv(self.path, index=False)

    def append(self, row: dict):
        """׷��һ�н��׼�¼�� CSV��ʧ��ʱ��¼������־��"""
        try:
            header_needed = (not os.path.exists(self.path)) or (os.path.getsize(self.path) == 0)
            with self._lock:
                pd.DataFrame([row]).to_csv(self.path, mode="a", header=header_needed, index=False)
        except Exception as e:
            self.logger.error(f"д��CSVʧ��: {e}")
