from fastapi import APIRouter, HTTPException
from ..models.schemas import TravelRequest, RecommendationResponse
from ..services.recommender import RecommendationService
import logging
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/recommend")
async def get_recommendations(request: TravelRequest):
    """추천 생성 엔드포인트"""
    try:
        service = RecommendationService.get_instance()
        recommendations = service.get_recommendations(request)

        return {
            **recommendations,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))