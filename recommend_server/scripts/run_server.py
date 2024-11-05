import uvicorn
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
import logging
from app.core.logging import setup_logging

logger = setup_logging()


def check_required_files():
    """필요한 파일들이 존재하는지 확인"""
    required_files = [
        (settings.PREPROCESSED_PATH, "Preprocessed data"),
        (settings.USER_DATA_PATH, "User data")
    ]

    missing_files = []
    for file_path, file_desc in required_files:
        if not os.path.exists(file_path):
            missing_files.append(f"{file_desc}: {file_path}")

    if missing_files:
        raise FileNotFoundError(
            "Required files not found:\n" + "\n".join(missing_files)
        )


def create_required_directories():
    """필요한 디렉토리 생성"""
    directories = [
        settings.LOG_DIR,
        os.path.dirname(settings.PREPROCESSED_PATH),
        os.path.dirname(settings.MODEL_PATH),
        os.path.dirname(settings.SIMILARITIES_PATH)
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def start_server():
    """서버 시작"""
    try:
        logger.info("Starting recommendation server...")

        # 디렉토리 생성
        create_required_directories()

        # 필수 파일 확인
        check_required_files()

        # 서버 실행
        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=False,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True
        )

    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        raise


if __name__ == "__main__":
    start_server()