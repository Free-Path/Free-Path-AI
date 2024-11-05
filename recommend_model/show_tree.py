import os
from pathlib import Path
from typing import List


def print_tree(directory: str, exclude_dirs: List[str] = None, exclude_files: List[str] = None):
    """
    디렉토리 구조를 트리 형태로 출력

    Args:
        directory: 출력할 디렉토리 경로
        exclude_dirs: 제외할 디렉토리 목록
        exclude_files: 제외할 파일 목록
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', '.ipynb_checkpoints']
    if exclude_files is None:
        exclude_files = ['.gitignore', '.DS_Store']

    def should_exclude(path: str) -> bool:
        """제외 대상인지 확인"""
        name = os.path.basename(path)
        if os.path.isdir(path):
            return name in exclude_dirs
        return name in exclude_files

    def inner_print_tree(directory: str, prefix: str = ""):
        """재귀적으로 트리 출력"""
        if should_exclude(directory):
            return

        # 현재 디렉토리 출력
        print(f"{prefix}📁 {os.path.basename(directory)}/")

        # 내용물 목록 생성
        entries = os.listdir(directory)
        entries = [e for e in entries if not should_exclude(os.path.join(directory, e))]
        entries = sorted(entries, key=lambda x: (os.path.isfile(os.path.join(directory, x)), x))

        # 각 항목 처리
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            entry_prefix = prefix + ("└── " if is_last else "├── ")
            next_prefix = prefix + ("    " if is_last else "│   ")

            entry_path = os.path.join(directory, entry)
            if os.path.isdir(entry_path):
                inner_print_tree(entry_path, next_prefix)
            else:
                print(f"{entry_prefix}📄 {entry}")

    # 시작 디렉토리의 절대 경로 출력
    abs_path = os.path.abspath(directory)
    print(f"\n📂 Project Root: {abs_path}\n")

    # 트리 출력 시작
    inner_print_tree(directory)


if __name__ == "__main__":
    # 현재 디렉토리 구조 출력
    current_dir = "."

    # 제외할 디렉토리와 파일 설정
    exclude_dirs = ['.git', '__pycache__', '.ipynb_checkpoints', 'venv', 'env', 'aihub']
    exclude_files = ['.gitignore', '.DS_Store', '*.pyc']

    print_tree(current_dir, exclude_dirs, exclude_files)