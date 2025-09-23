from pathlib import Path
import os

def create_directories(paths):
    """创建多个目录"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def check_file_exists(path):
    """检查文件是否存在"""
    return os.path.exists(path) and os.path.isfile(path)

def check_dir_exists(path):
    """检查目录是否存在"""
    return os.path.exists(path) and os.path.isdir(path)
