import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict
from datetime import datetime
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_user_data(user_data_path: str) -> pd.DataFrame:
    """사용자 마스터 데이터 로드"""
    try:
        user_df = pd.read_csv(user_data_path)
        logging.info(f"Loaded user data with {len(user_df)} records")
        return user_df
    except Exception as e:
        logging.error(f"Error loading user data: {str(e)}")
        raise


def process_travel_styles(df: pd.DataFrame) -> pd.DataFrame:
    """여행 스타일 데이터 처리"""
    # 여행 스타일 컬럼들
    style_columns = [f'TRAVEL_STYL_{i}' for i in range(1, 9)]

    # 각 스타일별 선호도를 이진값으로 변환
    for col in style_columns:
        if col in df.columns:
            df[f"{col}_encoded"] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[f"{col}_encoded"] = (df[f"{col}_encoded"] > 0).astype(int)

    return df


def clean_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 데이터 정제"""
    numeric_columns = [
        'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
        'INCOME', 'HOUSE_INCOME', 'EDU_NM'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def encode_categorical_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """범주형 데이터 인코딩"""
    categorical_columns = {
        'GENDER': {'여': 0, '남': 1},
        'AGE_GRP': {'20': 0, '30': 1, '40': 2, '50': 3, '60': 4},
        'MARR_STTS': {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4},
        'TRAVEL_TERM': {'1': 0, '2': 1, '3': 2, '4': 3}
    }

    encodings = {}
    for col, mapping in categorical_columns.items():
        if col in df.columns:
            if col == 'AGE_GRP':
                # 연령대 처리
                df[col] = df[col].astype(str).str.replace('대', '')
            df[f"{col}_encoded"] = df[col].map(mapping)
            encodings[col] = mapping

    return df, encodings


def extract_travel_regions(df: pd.DataFrame) -> pd.DataFrame:
    """선호 여행 지역 정보 추출"""
    for i in range(1, 4):
        sido_col = f'TRAVEL_LIKE_SIDO_{i}'
        sgg_col = f'TRAVEL_LIKE_SGG_{i}'

        if sido_col in df.columns and sgg_col in df.columns:
            # 지역 코드를 문자열로 변환하고 결합
            df[f'region_{i}'] = df[sido_col].astype(str) + '_' + df[sgg_col].astype(str)

    return df


def preprocess_pipeline(visit_path: str, user_path: str, output_path: str) -> Tuple[pd.DataFrame, Dict]:
    """데이터 전처리 파이프라인"""
    try:
        # 방문 데이터 로드
        logging.info("Loading visit data...")
        visit_data = pd.read_csv(visit_path)

        # 사용자 데이터 로드
        logging.info("Loading user data...")
        user_data = load_user_data(user_path)

        # 방문 정보 필터링
        logging.info("Filtering visit information...")
        visit_info = visit_data[
            (visit_data['VISIT_AREA_TYPE_CD'].isin(range(1, 9)))
        ]
        visit_info = visit_info.groupby('VISIT_AREA_NM').filter(lambda x: len(x) > 1)
        visit_info = visit_info.reset_index(drop=True)

        # 평점 계산
        logging.info("Calculating ratings...")
        visit_info['rating'] = visit_info[
            ['DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']
        ].mean(axis=1)

        # 여행객 ID 추출
        visit_info['TRAVELER_ID'] = visit_info['TRAVEL_ID'].str.split('_').str[1]

        # SIDO 정보 추출
        visit_info['SIDO'] = visit_info['LOTNO_ADDR'].str.split().str[0]

        # 사용자 데이터 전처리
        logging.info("Processing user data...")
        user_data = clean_numeric_values(user_data)
        user_data = process_travel_styles(user_data)
        user_data, encodings = encode_categorical_values(user_data)
        user_data = extract_travel_regions(user_data)

        # 데이터 결합
        logging.info("Merging data...")
        merged_data = visit_info.merge(
            user_data,
            on='TRAVELER_ID',
            how='left'
        )

        # 필요한 컬럼 선택
        columns_to_keep = [
                              'TRAVELER_ID', 'VISIT_AREA_NM', 'rating', 'SIDO',
                              'GENDER_encoded', 'AGE_GRP_encoded', 'MARR_STTS_encoded',
                              'INCOME', 'TRAVEL_NUM', 'TRAVEL_TERM_encoded',
                              'region_1', 'region_2', 'region_3'
                          ] + [f'TRAVEL_STYL_{i}_encoded' for i in range(1, 9)]

        df_final = merged_data[columns_to_keep].rename(columns={
            'TRAVELER_ID': 'userID',
            'VISIT_AREA_NM': 'itemID'
        })

        # 결측치 처리
        logging.info("Handling missing values...")
        numeric_cols = ['INCOME', 'TRAVEL_NUM', 'rating']
        for col in numeric_cols:
            df_final[col] = df_final[col].fillna(df_final[col].median())

        encoded_cols = [col for col in df_final.columns if col.endswith('_encoded')]
        for col in encoded_cols:
            df_final[col] = df_final[col].fillna(0)

        # 통계 수집
        stats = {
            'total_users': df_final['userID'].nunique(),
            'total_items': df_final['itemID'].nunique(),
            'total_ratings': len(df_final),
            'rating_stats': df_final['rating'].describe().to_dict(),
            'sparsity': 1 - (len(df_final) /
                             (df_final['userID'].nunique() * df_final['itemID'].nunique())),
            'encodings': encodings,
            'timestamp': datetime.now().isoformat()
        }

        # 전처리 통계 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        stats_path = os.path.join(os.path.dirname(output_path), 'preprocessing_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        # 전처리된 데이터 저장
        df_final.to_csv(output_path, index=False)
        logging.info(f"Preprocessed data saved to: {output_path}")

        return df_final, stats

    except Exception as e:
        logging.error(f"Error in preprocessing pipeline: {str(e)}")
        raise