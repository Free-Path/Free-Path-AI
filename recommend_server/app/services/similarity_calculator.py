import numpy as np
from typing import List, Dict
import pandas as pd


class UserSimilarityCalculator:
    # 동반 유형별 인원수 매핑
    ACCOMPANY_TYPE_MAPPING = {
        "나홀로 여행": 1,
        "2인 가족 여행": 2,
        "2인 여행(가족 외)": 2,
        "3인 이상 여행(가족 외)": 3,
        "자녀 동반 여행": 2,
        "부모 동반 여행": 2,
        "3대 동반 여행(친척 포함)": 3
    }

    @staticmethod
    def calculate_age_similarity(request_ages: List[int], user_age: str) -> float:
        """
        연령대 유사도 계산
        예: request_ages=[20, 30], user_age="20" -> 1.0
            request_ages=[20, 30], user_age="40" -> 0.0
        """
        try:
            user_age_num = int(user_age.replace('대', ''))
            if user_age_num in request_ages:
                return 1.0
            # 인접 연령대는 부분 점수 부여
            elif any(abs(user_age_num - req_age) == 10 for req_age in request_ages):
                return 0.5
            return 0.0
        except:
            return 0.0

    @staticmethod
    def calculate_people_similarity(request_people: int, accompany_type: str) -> float:
        """
        인원수 유사도 계산
        동반 유형을 기반으로 실제 인원수와 비교
        """
        try:
            type_people = UserSimilarityCalculator.ACCOMPANY_TYPE_MAPPING.get(accompany_type, 1)
            diff = abs(request_people - type_people)
            return max(0, 1 - (diff * 0.25))  # 차이가 4명 이상이면 0
        except:
            return 0.0

    @staticmethod
    def calculate_destination_similarity(request_dest: str, user_dest: str) -> float:
        """
        목적지 유사도 계산
        완전 일치: 1.0
        """
        return 1.0 if request_dest == user_dest else 0.0

    @staticmethod
    def calculate_purpose_similarity(
            request_purposes: List[int],
            user_motives: List[str]
    ) -> float:
        """
        여행 목적 유사도 계산
        공통 목적 비율 계산
        """
        try:
            user_purposes = [int(m) for m in user_motives if pd.notna(m)]
            if not user_purposes:
                return 0.0

            common = len(set(request_purposes) & set(user_purposes))
            total = max(len(request_purposes), len(user_purposes))
            return common / total
        except:
            return 0.0

    @staticmethod
    def calculate_travel_style_similarity(
            request_environment: int,
            request_visit: List[int],
            user_styles: List[str]
    ) -> float:
        """
        여행 스타일 유사도 계산
        환경 선호도와 방문 스타일 모두 고려
        """
        try:
            # 유효한 스타일만 추출
            user_styles = [int(s) for s in user_styles if pd.notna(s)]
            if not user_styles:
                return 0.0

            # 환경 선호도 매칭
            env_score = 1.0 if str(request_environment) in map(str, user_styles) else 0.0

            # 방문 스타일 매칭
            common_visits = len(set(request_visit) & set(user_styles))
            visit_score = common_visits / len(request_visit) if request_visit else 0.0

            # 환경과 방문 스타일 점수 결합
            return 0.6 * env_score + 0.4 * visit_score
        except:
            return 0.0

    @staticmethod
    def calculate_user_similarity(request: Dict, user_data: Dict) -> float:
        """
        전체 유사도 계산
        각 요소별 가중치를 적용하여 최종 유사도 계산
        """
        similarities = {}

        # 1. 연령대 유사도
        similarities['age'] = UserSimilarityCalculator.calculate_age_similarity(
            request['age'],
            user_data['AGE_GRP']
        )

        # 2. 인원수/동반유형 유사도
        similarities['people'] = UserSimilarityCalculator.calculate_people_similarity(
            request['people'],
            user_data['TRAVEL_STATUS_ACCOMPANY']
        )

        # 3. 목적지 유사도
        similarities['destination'] = UserSimilarityCalculator.calculate_destination_similarity(
            request['destination'],
            user_data['TRAVEL_STATUS_DESTINATION']
        )

        # 4. 여행 목적 유사도
        user_motives = [
            user_data.get(f'TRAVEL_MOTIVE_{i}')
            for i in range(1, 4)
        ]
        similarities['purpose'] = UserSimilarityCalculator.calculate_purpose_similarity(
            request['purpose'],
            user_motives
        )

        # 5. 여행 스타일 유사도
        user_styles = [
            user_data.get(f'TRAVEL_STYL_{i}')
            for i in range(1, 9)
        ]
        similarities['style'] = UserSimilarityCalculator.calculate_travel_style_similarity(
            request['environment'],
            request['visit'],
            user_styles
        )

        # 가중치 적용
        weights = {
            'age': 0.15,  # 연령대
            'people': 0.15,  # 인원수/동반유형
            'destination': 0.2,  # 목적지
            'purpose': 0.25,  # 여행 목적
            'style': 0.25  # 여행 스타일
        }

        final_similarity = sum(
            similarities[key] * weights[key]
            for key in weights
        )

        # 디버깅을 위한 상세 점수 기록
        detailed_scores = {
            key: round(similarities[key], 3)
            for key in similarities
        }
        detailed_scores['final'] = round(final_similarity, 3)

        return final_similarity, detailed_scores