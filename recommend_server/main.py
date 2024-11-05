from fastapi import FastAPI
from app.core.config import settings
from app.api.endpoints import router
from app.core.logging import setup_logging
from app.services.recommender import RecommendationService
from datetime import datetime

# 로깅 설정
logger = setup_logging()

# FastAPI 앱 생성
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION
)

# 기본 헬스 체크 엔드포인트
@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.now().isoformat()
    }

# API 라우터 등록 (프리픽스 없이)
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행될 이벤트"""
    logger.info("Starting recommendation server...")
    # 추천 서비스 초기화
    RecommendationService.get_instance()

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행될 이벤트"""
    logger.info("Shutting down recommendation server...")