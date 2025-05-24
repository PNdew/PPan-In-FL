import os

def save_metric_to_txt(metric_name, metric_value, phase="train"):
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = os.path.join(RESULTS_DIR, f"{metric_name}_{phase}.txt")
    with open(filename, "a") as f:
        f.write( str(metric_value)+ "\n")