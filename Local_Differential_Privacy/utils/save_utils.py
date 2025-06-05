import os

def write_to_file(filename, data):
    """Ghi dữ liệu vào file, mỗi dòng là một giá trị mới."""
    with open(filename, "a") as f:
        f.write(f"{data}\n")
import os

def save_metric_to_txt(metric_name: str, value: float, server_round: int, p_weight: float):
    """
    Ghi metric vào file trong thư mục 'results/privacy_{p_weight}/'.
    
    Args:
        metric_name: Tên của metric, ví dụ 'accuracy', 'loss', 'privacy_leakage', v.v.
        value: Giá trị float của metric.
        server_round: Số vòng hiện tại của FL.
        p_weight: Trọng số bảo mật privacy_weight đang dùng.
    """
    results_dir = "results"
    subdir = os.path.join(results_dir, f"privacy_{p_weight}")
    os.makedirs(subdir, exist_ok=True)

    file_path = os.path.join(subdir, f"{metric_name}.txt")
    with open(file_path, "a") as f:
        f.write(f"Round {server_round}: {value:.6f}\n")
