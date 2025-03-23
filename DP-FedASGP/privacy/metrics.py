from collections import defaultdict
import numpy as np
from typing import Dict
class Metrics:
    """Enhanced metrics tracking"""
    def __init__(self):
        self.history = defaultdict(list)

    def update(self, metrics: Dict):
        """Cập nhật dữ liệu đánh giá"""
        for key, value in metrics.items():
            if value is not None and not np.isnan(value):
                self.history[key].append(float(value))
                self.save_to_file(key, float(value))

    def save_to_file(self, metric_name: str, value: float):
        """Ghi từng tiêu chí vào file .txt riêng biệt"""
        filename = f"{metric_name}.txt"
        with open(filename, "a") as f:
            f.write(f"{value}\n")