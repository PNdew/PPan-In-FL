import os

def write_to_file(filename, data):
    """Ghi dữ liệu vào file, mỗi dòng là một giá trị mới."""
    with open(filename, "a") as f:
        f.write(f"{data}\n")