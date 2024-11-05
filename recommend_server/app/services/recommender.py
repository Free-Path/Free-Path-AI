from typing import List, Dict, Tuple
import pandas as pd
import logging
from collections import defaultdict
from ..models.schemas import TravelRequest, SimilarityScores
from ..core.config import settings
from .similarity_calculator import UserSimilarityCalculator

logger = logging.getLogger(__name__)


class RecommendationService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.load_resources()

    def load_resources(self):
        """데이터 로드"""
        try:
            logger.info("Loading data...")

            # 방문 데이터 로드
            logger.info("Loading preprocessed visit data...")
            self.df = pd.read_csv(settings.PREPROCESSED_PATH)

            # 사용자 마스터 데이터 로드
            logger.info("Loading user data...")
            self.user_data = pd.read_csv(settings.USER_DATA_PATH)

            # 필요한 컬럼 확인
            required_columns = {
                'visit_data': ['userID', 'itemID', 'rating', 'SIDO'],
                'user_data': ['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STATUS_DESTINATION',
                              'TRAVEL_STATUS_ACCOMPANY', 'TRAVEL_COMPANIONS_NUM']
            }

            # 여행 동기 컬럼 추가
            required_columns['user_data'].extend([f'TRAVEL_MOTIVE_{i}' for i in range(1, 4)])
            # 여행 스타일 컬럼 추가
            required_columns['user_data'].extend([f'TRAVEL_STYL_{i}' for i in range(1, 9)])

            # 컬럼 존재 확인
            missing_visit_columns = [col for col in required_columns['visit_data'] if col not in self.df.columns]
            missing_user_columns = [col for col in required_columns['user_data'] if col not in self.user_data.columns]

            if missing_visit_columns:
                raise ValueError(f"Missing required columns in visit data: {missing_visit_columns}")
            if missing_user_columns:
                raise ValueError(f"Missing required columns in user data: {missing_user_columns}")

            logger.info(f"Loaded {len(self.df)} visit records and {len(self.user_data)} user records")

        except Exception as e:
            logger.error(f"Error loading resources: {str(e)}")
            raise

    def find_similar_users(
            self,
            request: TravelRequest,
            n_similar: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """유사한 사용자 찾기"""
        similarities = []

        request_dict = request.dict()
        for _, user in self.user_data.iterrows():
            similarity, detailed_scores = UserSimilarityCalculator.calculate_user_similarity(
                request_dict,
                user.to_dict()
            )
            similarities.append((
                user['TRAVELER_ID'],
                similarity,
                detailed_scores
            ))

        # 유사도 순으로 정렬
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n_similar]

    def get_place_recommendations(
            self,
            similar_users: List[Tuple[str, float, Dict]],
            destination: str,
            n_recommendations: int = 5
    ) -> List[Dict]:
        """장소 추천 생성"""
        place_scores = defaultdict(float)
        place_counts = defaultdict(int)
        place_max_similarity = defaultdict(float)
        place_similarity_scores = defaultdict(lambda: None)

        for user_id, similarity, detailed_scores in similar_users:
            user_visits = self.df[self.df['userID'] == user_id]

            for _, visit in user_visits.iterrows():
                place_id = visit['itemID']

                # 목적지가 일치하는 경우만 추천
                if visit['SIDO'] != destination:
                    continue

                rating = visit['rating']
                place_scores[place_id] += rating * similarity
                place_counts[place_id] += 1

                if similarity > place_max_similarity[place_id]:
                    place_max_similarity[place_id] = similarity
                    place_similarity_scores[place_id] = detailed_scores

        # 추천 목록 생성
        recommendations = []
        for place_id in place_scores:
            if place_counts[place_id] > 0:
                avg_score = place_scores[place_id] / place_counts[place_id]

                recommendations.append({
                    'item_id': place_id,
                    'sido': destination,
                    'predicted_rating': float(avg_score),
                    'confidence_score': float(avg_score * place_max_similarity[place_id]),
                    'similarity_scores': SimilarityScores(
                        **place_similarity_scores[place_id]
                    )
                })

        # 점수순 정렬
        recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
        return recommendations[:n_recommendations]

    def get_recommendations(
            self,
            request: TravelRequest,
            n_recommendations: int = 5
    ) -> Dict:
        """추천 생성 메인 함수"""
        try:
            # 유사 사용자 찾기
            similar_users = self.find_similar_users(request)

            # 장소 추천 생성
            recommendations = self.get_place_recommendations(
                similar_users,
                request.destination,
                n_recommendations
            )

            return {
                "recommendations": recommendations,
                "similar_users_count": len(similar_users)
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise
