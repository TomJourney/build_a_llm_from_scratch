import os


def get_root_dir():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    return os.path.dirname(current_dir)
