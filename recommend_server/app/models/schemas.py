from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class TravelRequest(BaseModel):
    startAt: str = Field(..., description="여행 시작 날짜")
    endAt: str = Field(..., description="여행 종료 날짜")
    people: int = Field(..., description="총 인원 수")
    destination: str = Field(..., description="목적지(시도)")
    disabilities: Optional[List[str]] = Field(None, description="가지고 있는 장애 리스트")
    age: List[int] = Field(..., description="연령대")
    theme: List[int] = Field(..., description="여행 테마(MIS)")
    purpose: List[int] = Field(..., description="여행 목적(TMT)")
    visit: List[int] = Field(..., description="방문지 스타일(VIS)")
    environment: int = Field(..., description="선호 환경(TSY)")

class SimilarityScores(BaseModel):
    age: float
    people: float
    destination: float
    purpose: float
    style: float
    final: float

class RecommendationItem(BaseModel):
    item_id: str = Field(..., description="장소 ID")
    sido: str = Field(..., description="지역")
    predicted_rating: float = Field(..., description="예상 평점")
    confidence_score: float = Field(..., description="추천 신뢰도")
    similarity_scores: SimilarityScores = Field(..., description="유사도 상세 점수")

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]
    similar_users_count: int = Field(..., description="유사 사용자 수")
    timestamp: datetime = Field(default_factory=datetime.now)