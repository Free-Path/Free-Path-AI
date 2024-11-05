from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # 기본 설정
    PROJECT_NAME: str = "Travel Recommendation API"
    PROJECT_DESCRIPTION: str = "여행지 추천 API 서비스"
    VERSION: str = "1.0.0"

    # 경로 설정
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 데이터 파일 경로
    VISIT_DATA_PATH: str = os.path.join(DATA_DIR, "Training/label/csv/tn_visit_area_info_E.csv")
    USER_DATA_PATH: str = os.path.join(DATA_DIR, "Training/label/csv/tn_traveller_master_E.csv")
    PREPROCESSED_PATH: str = os.path.join(DATA_DIR, "preprocessed/dfE.csv")
    MODEL_PATH: str = os.path.join(BASE_DIR, "experiments/best_model/model.pkl")
    SIMILARITIES_PATH: str = os.path.join(DATA_DIR, "similarities/item_similarities.pkl")

    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")

    class Config:
        env_file = ".env"


settings = Settings()