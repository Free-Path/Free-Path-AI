import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from .config import settings


def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    # 로거 생성
    logger = logging.getLogger("recommendation-api")
    logger.setLevel(settings.LOG_LEVEL)

    # 포맷터 생성
    formatter = logging.Formatter(settings.LOG_FORMAT)

    # 파일 핸들러
    log_file = os.path.join(
        settings.LOG_DIR,
        f'api_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    # 스트림 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger