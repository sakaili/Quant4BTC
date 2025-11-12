# csv_logger.py
import os
import pandas as pd


class CsvLogger:
    """把交易执行记录持久化到 CSV 文件。"""

    def __init__(self, path: str, logger):
        """创建/引用日志文件路径并写入表头。"""
        self.path = path
        self.logger = logger
        self._ensure_headers()

    def _ensure_headers(self):
        """若文件不存在或为空，则写入标准表头。"""
        if (not os.path.exists(self.path)) or (os.path.getsize(self.path) == 0):
            pd.DataFrame(columns=[
                'timestamp', 'symbol', 'signal', 'close', 'position', 'action', 'exec_price', 'fee',
                'order_id', 'stop_loss', 'take_profit', 'best_factor', 'equity', 'mode'
            ]).to_csv(self.path, index=False)

    def append(self, row: dict):
        """追加一行交易记录到 CSV，失败时记录错误日志。"""
        try:
            header_needed = (not os.path.exists(self.path)) or (os.path.getsize(self.path) == 0)
            pd.DataFrame([row]).to_csv(self.path, mode='a', header=header_needed, index=False)
        except Exception as e:
            self.logger.error(f"写入CSV失败: {e}")
