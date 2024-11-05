import requests
import json
from datetime import datetime, timedelta
import logging
from typing import Dict
import pandas as pd
import os


class RecommendationTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results_dir = self._create_results_dir()
        self._setup_logging()

    def _create_results_dir(self) -> str:
        """테스트 결과 저장 디렉토리 생성"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join('test_results', timestamp)
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

    def _setup_logging(self):
        """로깅 설정"""
        log_file = os.path.join(self.results_dir, 'test_log.txt')

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # 파일 핸들러
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(console_handler)

    def save_results(self, results: Dict, test_name: str):
        """결과 저장"""
        # JSON 결과 저장
        json_path = os.path.join(self.results_dir, f'{test_name}_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # CSV 형식으로도 저장 (추천 목록)
        if 'recommendations' in results:
            csv_path = os.path.join(self.results_dir, f'{test_name}_recommendations.csv')
            df = pd.DataFrame(results['recommendations'])
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        self.logger.info(f"Results saved to {json_path}")

    def test_health(self) -> bool:
        """서버 헬스 체크"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            self.logger.info("Health check successful")
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def get_recommendations(self, request_data: Dict) -> Dict:
        """추천 요청"""
        try:
            response = requests.post(
                f"{self.base_url}/recommend",
                json=request_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            raise

    def run_test_cases(self):
        """테스트 케이스 실행"""
        start_date = datetime.now()
        end_date = start_date + timedelta(days=2)

        test_cases = [
            {
                "name": "가족여행객",
                "request": {
                    "startAt": start_date.strftime('%Y-%m-%d'),
                    "endAt": end_date.strftime('%Y-%m-%d'),
                    "people": 3,
                    "destination": "서울",
                    "disabilities": ["helpdog"],
                    "age": [30, 40],
                    "theme": [1, 2],
                    "purpose": [1, 2, 3],
                    "visit": [1, 2],
                    "environment": 3
                }
            },
            {
                "name": "혼자여행객",
                "request": {
                    "startAt": start_date.strftime('%Y-%m-%d'),
                    "endAt": end_date.strftime('%Y-%m-%d'),
                    "people": 1,
                    "destination": "부산",
                    "disabilities": None,
                    "age": [20],
                    "theme": [2, 3],
                    "purpose": [2, 4],
                    "visit": [2, 3, 4],
                    "environment": 2
                }
            },
            {
                "name": "친구여행객",
                "request": {
                    "startAt": start_date.strftime('%Y-%m-%d'),
                    "endAt": end_date.strftime('%Y-%m-%d'),
                    "people": 4,
                    "destination": "제주",
                    "disabilities": None,
                    "age": [20, 20],
                    "theme": [3, 4],
                    "purpose": [3, 5],
                    "visit": [3, 4, 5],
                    "environment": 4
                }
            }
        ]

        # 테스트 시작 시간 기록
        test_summary = {
            'start_time': datetime.now().isoformat(),
            'results': {}
        }

        # 헬스 체크
        if not self.test_health():
            self.logger.error("Server is not healthy, stopping tests")
            return

        # 테스트 케이스 실행
        for test_case in test_cases:
            self.logger.info(f"\nExecuting test case: {test_case['name']}")

            try:
                result = self.get_recommendations(test_case['request'])
                self.logger.info(f"Found {result['similar_users_count']} similar users")

                # 결과 분석
                analysis = self.analyze_results(result)

                # 결과 저장
                test_results = {
                    'request': test_case['request'],
                    'response': result,
                    'analysis': analysis
                }

                self.save_results(test_results, test_case['name'])
                test_summary['results'][test_case['name']] = {
                    'status': 'success',
                    'analysis': analysis
                }

            except Exception as e:
                error_msg = str(e)
                self.logger.error(f"Test case {test_case['name']} failed: {error_msg}")
                test_summary['results'][test_case['name']] = {
                    'status': 'failed',
                    'error': error_msg
                }

        # 테스트 종료 시간 기록
        test_summary['end_time'] = datetime.now().isoformat()

        # 전체 결과 저장
        self.save_results(test_summary, 'test_summary')

        # 결과 요약 출력
        self.print_summary(test_summary)

    def analyze_results(self, result: Dict) -> Dict:
        """결과 분석"""
        analysis = {
            'total_recommendations': len(result['recommendations']),
            'similar_users_count': result['similar_users_count']
        }

        if result['recommendations']:
            ratings = [r['predicted_rating'] for r in result['recommendations']]
            confidence_scores = [r['confidence_score'] for r in result['recommendations']]

            analysis.update({
                'ratings': {
                    'min': min(ratings),
                    'max': max(ratings),
                    'avg': sum(ratings) / len(ratings)
                },
                'confidence_scores': {
                    'min': min(confidence_scores),
                    'max': max(confidence_scores),
                    'avg': sum(confidence_scores) / len(confidence_scores)
                }
            })

            # 지역 분포
            sido_counts = {}
            for rec in result['recommendations']:
                sido = rec['sido']
                sido_counts[sido] = sido_counts.get(sido, 0) + 1
            analysis['region_distribution'] = sido_counts

        return analysis

    def print_summary(self, summary: Dict):
        """테스트 결과 요약 출력"""
        self.logger.info("\n=== Test Summary ===")
        self.logger.info(f"Test started at: {summary['start_time']}")
        self.logger.info(f"Test ended at: {summary['end_time']}")

        for test_name, result in summary['results'].items():
            self.logger.info(f"\n{test_name}:")
            if result['status'] == 'success':
                analysis = result['analysis']
                self.logger.info(f"- Similar users: {analysis['similar_users_count']}")
                self.logger.info(f"- Total recommendations: {analysis['total_recommendations']}")
                if 'ratings' in analysis:
                    self.logger.info(f"- Average rating: {analysis['ratings']['avg']:.2f}")
                if 'region_distribution' in analysis:
                    self.logger.info("- Region distribution:")
                    for sido, count in analysis['region_distribution'].items():
                        self.logger.info(f"  {sido}: {count}")
            else:
                self.logger.info(f"- Failed: {result['error']}")


def main():
    # 테스터 초기화
    tester = RecommendationTester()

    try:
        # 테스트 실행
        tester.run_test_cases()

    except Exception as e:
        logging.error(f"Test execution failed: {e}")


if __name__ == "__main__":
    main()