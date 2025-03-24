import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import  wasserstein_distance
from sklearn.feature_selection import mutual_info_regression


class PrivacyMetrics:
    """Theo dõi rò rỉ thông tin và độ méo, ghi vào từng file riêng biệt."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
        # Định nghĩa file log cho từng tiêu chí
        self.files = {
            "info_leakage": "info_leakage.txt",
            "distortion": "distortion.txt",
            "accuracy": "accuracy.txt"
        }
        
        # Khởi tạo các file log (xóa nội dung cũ nếu có)
        for file in self.files.values():
            with open(file, 'w') as f:
                f.write("Round,Value\n")

    def compute_information_leakage(self, original: torch.Tensor, protected: torch.Tensor) -> float:
        """Tính độ rò rỉ thông tin (Mutual Information)"""
        try:
            orig = original.cpu().numpy().flatten().reshape(-1, 1)
            prot = protected.cpu().numpy().flatten()

            mi_score = mutual_info_regression(orig, prot)[0]
            return float(mi_score)

        except Exception as e:
            print(f"Error calculating information leakage: {e}")
            return 1.0

    def compute_distortion(self, original: torch.Tensor, protected: torch.Tensor) -> dict:
        """Tính độ méo (Wasserstein, MSE, L2)"""
        try:
            orig = original.cpu().numpy().flatten()
            prot = protected.cpu().numpy().flatten()

            return {
                'wasserstein': wasserstein_distance(orig, prot),
                'mse_distortion': np.mean((orig - prot) ** 2),
                'l2_distortion': np.linalg.norm(orig - prot) / (np.linalg.norm(orig) + 1e-10)
            }

        except Exception as e:
            print(f"Error calculating distortion: {e}")
            return {'wasserstein': 0.0, 'mse_distortion': 0.0, 'l2_distortion': 0.0}

    def update(self, original: torch.Tensor, protected: torch.Tensor, round_num: int):
        """Cập nhật và ghi kết quả vào từng file riêng"""
        try:
            info_leakage = self.compute_information_leakage(original, protected)
            distortion = self.compute_distortion(original, protected)
            privacy_score = 1.0 - info_leakage

            # Ghi độ rò rỉ thông tin
            with open(self.files["info_leakage"], 'a') as f:
                f.write(f"{round_num},{info_leakage:.6f}\n")

            # Ghi độ méo
            with open(self.files["distortion"], 'a') as f:
                f.write(f"{round_num},{distortion['wasserstein']:.6f},{distortion['mse_distortion']:.6f},{distortion['l2_distortion']:.6f}\n")

            # Ghi điểm bảo mật
            with open(self.files["accuracy"], 'a') as f:
                f.write(f"{round_num},{privacy_score:.6f}\n")

        except Exception as e:
            print(f"Error updating metrics: {e}")