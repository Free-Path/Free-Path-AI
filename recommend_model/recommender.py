import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(self, df: pd.DataFrame, model, similarities: dict):
        """
        추천 엔진 초기화

        Args:
            df: 전처리된 데이터프레임
            model: 학습된 추천 모델
            similarities: 아이템 간 유사도 사전
        """
        self.df = df
        self.model = model
        self.similarities = similarities
        self.setup_metrics()

    def setup_metrics(self):
        """기본 메트릭 설정"""
        logger.info("Setting up metrics...")

        # 1. 기본 메트릭 계산
        # 장소별 방문 횟수
        visit_counts = self.df['itemID'].value_counts()
        self.visit_counts = dict(zip(visit_counts.index, visit_counts.values))

        # 장소별 평균 평점
        rating_means = self.df.groupby('itemID')['rating'].mean()
        self.avg_ratings = dict(zip(rating_means.index, rating_means.values))

        # 장소별 SIDO 정보
        sido_info = self.df[['itemID', 'SIDO']].drop_duplicates()
        self.item_sidos = dict(zip(sido_info['itemID'], sido_info['SIDO']))

        # 2. SIDO별 인기 장소 계산
        self.sido_items = defaultdict(dict)
        for sido in self.df['SIDO'].unique():
            sido_df = self.df[self.df['SIDO'] == sido]

            for item_id in sido_df['itemID'].unique():
                item_data = sido_df[sido_df['itemID'] == item_id]

                self.sido_items[sido][item_id] = {
                    'visits': len(item_data),
                    'rating': item_data['rating'].mean(),
                    'score': None  # 점수는 나중에 계산
                }

            # 인기도 점수 계산
            if self.sido_items[sido]:
                max_visits = max(item['visits'] for item in self.sido_items[sido].values())
                for item_id in self.sido_items[sido]:
                    item = self.sido_items[sido][item_id]
                    normalized_visits = item['visits'] / max_visits if max_visits > 0 else 0
                    normalized_rating = item['rating'] / 5.0

                    # 방문수와 평점을 결합한 점수
                    item['score'] = 0.7 * normalized_visits + 0.3 * normalized_rating

        logger.info("Metrics setup completed")

    def get_recommendations_for_existing_user(
            self,
            user_id: str,
            n_recommendations: int = 5,
            diversity_weight: float = 0.3
    ) -> List[Dict]:
        """
        기존 사용자를 위한 추천 생성

        Args:
            user_id: 사용자 ID
            n_recommendations: 추천 개수
            diversity_weight: 다양성 가중치 (0~1)

        Returns:
            recommendations: 추천 목록
        """
        try:
            # 1. 방문한 장소 확인
            visited = set(self.df[self.df['userID'] == user_id]['itemID'])

            # 2. 후보 아이템 생성
            candidates = []
            all_items = set(self.df['itemID'].unique())
            unvisited = list(all_items - visited)

            # 3. 예측 점수 계산
            for item_id in unvisited:
                try:
                    pred = self.model.predict(user_id, item_id)
                    candidates.append({
                        'item_id': item_id,
                        'predicted_rating': pred.est,
                        'sido': self.item_sidos[item_id]
                    })
                except Exception as e:
                    logger.warning(f"Prediction failed for user {user_id}, item {item_id}: {e}")
                    continue

            # 4. 추천 생성
            recommendations = []
            selected_items = set()

            # 예측 점수로 정렬
            candidates.sort(key=lambda x: x['predicted_rating'], reverse=True)

            for candidate in candidates:
                if len(recommendations) >= n_recommendations:
                    break

                item_id = candidate['item_id']

                # 다양성 점수 계산
                diversity_score = 1.0
                if selected_items:
                    sim_scores = []
                    for selected_item in selected_items:
                        sim = self.similarities.get(item_id, {}).get(selected_item, 0)
                        sim_scores.append(max(0, sim))  # 음수 유사도는 0으로 처리
                    diversity_score = 1 - (sum(sim_scores) / len(sim_scores))

                # 인기도 점수 가져오기
                sido = candidate['sido']
                popularity_score = self.sido_items[sido].get(item_id, {}).get('score', 0.1)

                # 최종 점수 계산
                final_score = (1 - diversity_weight) * candidate['predicted_rating']
                final_score += diversity_weight * diversity_score
                final_score *= (1 + popularity_score)  # 인기도 보너스

                recommendations.append({
                    'item_id': item_id,
                    'sido': sido,
                    'predicted_rating': float(candidate['predicted_rating']),
                    'diversity_score': float(diversity_score),
                    'confidence_score': float(final_score)
                })

                selected_items.add(item_id)

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            raise

    def get_recommendations_for_new_user(
            self,
            preferred_sidos: Optional[List[str]] = None,
            n_recommendations: int = 5
    ) -> List[Dict]:
        """
        새로운 사용자를 위한 추천 생성

        Args:
            preferred_sidos: 선호 지역 목록
            n_recommendations: 추천 개수

        Returns:
            recommendations: 추천 목록
        """
        try:
            recommendations = []
            selected_sidos = set()

            # 1. 선호 지역 기반 추천
            if preferred_sidos:
                for sido in preferred_sidos:
                    if sido not in self.sido_items:
                        continue

                    # 해당 지역의 인기 장소 정렬
                    items = sorted(
                        self.sido_items[sido].items(),
                        key=lambda x: x[1]['score'] if x[1]['score'] is not None else 0,
                        reverse=True
                    )

                    for item_id, item_data in items:
                        if len(recommendations) >= n_recommendations:
                            break

                        recommendations.append({
                            'item_id': item_id,
                            'sido': sido,
                            'predicted_rating': float(item_data['rating']),
                            'popularity_score': float(item_data['score'] or 0),
                            'confidence_score': float(item_data['score'] or 0)
                        })
                        selected_sidos.add(sido)

            # 2. 남은 추천 슬롯을 전체 인기 장소로 채우기
            if len(recommendations) < n_recommendations:
                # 전체 인기 장소 목록 생성
                all_popular = []
                for sido, items in self.sido_items.items():
                    if sido in selected_sidos:
                        continue

                    for item_id, item_data in items.items():
                        all_popular.append({
                            'item_id': item_id,
                            'sido': sido,
                            'rating': item_data['rating'],
                            'score': item_data['score'] or 0
                        })

                # 점수순 정렬
                all_popular.sort(key=lambda x: x['score'], reverse=True)

                # 추천 목록 채우기
                for item in all_popular:
                    if len(recommendations) >= n_recommendations:
                        break

                    recommendations.append({
                        'item_id': item['item_id'],
                        'sido': item['sido'],
                        'predicted_rating': float(item['rating']),
                        'popularity_score': float(item['score']),
                        'confidence_score': float(item['score'])
                    })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations for new user: {e}")
            raise

    def get_recommendations(
            self,
            user_id: Optional[str] = None,
            preferred_sidos: Optional[List[str]] = None,
            n_recommendations: int = 5,
            diversity_weight: float = 0.3
    ) -> Dict:
        """
        통합 추천 함수

        Args:
            user_id: 사용자 ID (기존 사용자)
            preferred_sidos: 선호 지역 목록 (새로운 사용자)
            n_recommendations: 추천 개수
            diversity_weight: 다양성 가중치

        Returns:
            result: 추천 결과
        """
        try:
            if user_id and user_id in set(self.df['userID']):
                # 기존 사용자
                recommendations = self.get_recommendations_for_existing_user(
                    user_id,
                    n_recommendations,
                    diversity_weight
                )
                user_type = "existing"
            else:
                # 새로운 사용자
                recommendations = self.get_recommendations_for_new_user(
                    preferred_sidos,
                    n_recommendations
                )
                user_type = "new"

            return {
                "user_id": user_id,
                "user_type": user_type,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.error(f"Error in get_recommendations: {e}")
            raise