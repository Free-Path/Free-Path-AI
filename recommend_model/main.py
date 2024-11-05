from datetime import datetime
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from preprocess import preprocess_pipeline
from model import train_model, save_results, compare_metrics, save_best_model, AlgoType
from recommender import RecommendationEngine
import logging
from pathlib import Path
import pickle


def create_experiment_directory(base_dir: str) -> Tuple[str, str, Dict[str, str]]:
    """
    실험용 디렉토리 구조 생성

    Args:
        base_dir: 기본 디렉토리 경로

    Returns:
        experiment_dir: 실험 디렉토리 경로
        timestamp: 타임스탬프
        directories: 하위 디렉토리 경로들
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')

    # Create subdirectories
    directories = {
        'models': os.path.join(experiment_dir, 'models'),
        'results': os.path.join(experiment_dir, 'results'),
        'metrics': os.path.join(experiment_dir, 'metrics'),
        'recommendations': os.path.join(experiment_dir, 'recommendations'),
        'logs': os.path.join(experiment_dir, 'logs')
    }

    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

    return experiment_dir, timestamp, directories


def setup_logging(base_dir: str, experiment_time: str) -> str:
    """
    로깅 설정

    Args:
        base_dir: 기본 디렉토리 경로
        experiment_time: 실험 시작 시간

    Returns:
        log_file: 로그 파일 경로
    """
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f'experiment_{experiment_time}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def get_algorithm_choice() -> List[str]:
    """
    사용할 알고리즘 선택

    Returns:
        algorithms: 선택된 알고리즘 리스트
    """
    algorithms = [
        'svd', 'svdpp', 'nmf', 'knn_basic', 'knn_means',
        'knn_zscore', 'knn_baseline', 'coclustering',
        'baseline', 'slope_one'
    ]

    print("\nAvailable algorithms:")
    print("0. Find best model (runs all algorithms and selects the best)")
    for i, algo in enumerate(algorithms, 1):
        print(f"{i}. {algo}")

    while True:
        try:
            choice = input("\nEnter the number or name of the algorithm: ").strip()

            # Best model option
            if choice == '0':
                print("\nRunning all algorithms to find the best model...")
                return algorithms

            # Number choice
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(algorithms):
                    return [algorithms[index]]
            # Name choice
            elif choice.lower() in algorithms:
                return [choice.lower()]

            print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or algorithm name.")


def log_experiment_stats(stats: Dict, experiment_dir: str):
    """
    실험 통계 저장

    Args:
        stats: 실험 통계 데이터
        experiment_dir: 실험 디렉토리 경로
    """
    stats_path = os.path.join(experiment_dir, 'experiment_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)


def run_algorithm(algo_name: str, df: pd.DataFrame, directories: Dict[str, str],
                  diversity_weight: float = 0.3) -> Dict:
    """
    단일 알고리즘 실행

    Args:
        algo_name: 알고리즘 이름
        df: 데이터프레임
        directories: 디렉토리 경로들
        diversity_weight: 다양성 가중치

    Returns:
        result: 실행 결과
    """
    logging.info(f"\nRunning {algo_name}...")
    try:
        # Train and evaluate model
        model, metrics, predictions = train_model(df, algo_name)

        # Save results
        model_path = os.path.join(directories['models'], f'model_{algo_name}.pkl')
        save_results(model, metrics, algo_name, model_path)

        # Initialize recommendation engine
        engine = RecommendationEngine(df, model, {})

        # Generate recommendations
        output_path = os.path.join(directories['recommendations'], f'recommendations_{algo_name}.csv')

        # Get all unique users
        all_users = df['userID'].unique()
        all_recommendations = []

        # Generate recommendations for each user
        for user_id in all_users:
            try:
                user_recs = engine.get_recommendations(
                    user_id=user_id,
                    n_recommendations=5,
                    diversity_weight=diversity_weight
                )
                all_recommendations.extend([
                    {
                        'user_id': user_id,
                        'item_id': rec['item_id'],
                        'sido': rec['sido'],
                        'predicted_rating': rec['predicted_rating'],
                        'confidence_score': rec['confidence_score']
                    }
                    for rec in user_recs['recommendations']
                ])
            except Exception as e:
                logging.warning(f"Failed to generate recommendations for user {user_id}: {str(e)}")

        # Save recommendations
        recommendations_df = pd.DataFrame(all_recommendations)
        recommendations_df.to_csv(output_path, index=False)

        logging.info(f"Completed {algo_name} successfully!")
        return {
            'status': 'success',
            'metrics': metrics,
            'model': model,
            'recommendations': recommendations_df
        }

    except Exception as e:
        logging.error(f"Error running {algo_name}: {str(e)}")
        return {'status': 'error', 'error': str(e)}


def find_best_model(df: pd.DataFrame, algorithms: List[str], directories: Dict[str, str],
                    diversity_weight: float = 0.3) -> Dict:
    """
    최적 모델 찾기

    Args:
        df: 데이터프레임
        algorithms: 알고리즘 리스트
        directories: 디렉토리 경로들
        diversity_weight: 다양성 가중치

    Returns:
        results: 모든 알고리즘의 실행 결과
    """
    results = {}
    all_metrics = []
    all_models = []
    algo_names = []

    for algo_name in algorithms:
        result = run_algorithm(algo_name, df, directories, diversity_weight)
        results[algo_name] = result

        if result['status'] == 'success':
            all_metrics.append(result['metrics'])
            all_models.append(result['model'])
            algo_names.append(algo_name)

    # Find and save best model
    if all_metrics:
        best_result = compare_metrics(all_metrics)
        best_idx = best_result['index']
        best_model = all_models[best_idx]
        best_algo = algo_names[best_idx]

        logging.info("\nBest performing model:")
        logging.info(f"Algorithm: {best_algo}")
        logging.info("\nMetrics:")
        for metric, value in best_result['metrics'].items():
            logging.info(f"{metric}: {value:.4f}")

        # Save best model
        best_model_path = os.path.join(directories['models'], 'best_model')
        os.makedirs(best_model_path, exist_ok=True)
        save_best_model(best_model, best_result['metrics'], best_algo, best_model_path)

        # Generate recommendations using best model
        output_path = os.path.join(directories['recommendations'], 'recommendations_best.csv')
        engine = RecommendationEngine(df, best_model, {})

        # Get recommendations for all users
        all_users = df['userID'].unique()
        all_recommendations = []

        for user_id in all_users:
            try:
                user_recs = engine.get_recommendations(
                    user_id=user_id,
                    n_recommendations=5,
                    diversity_weight=diversity_weight
                )
                all_recommendations.extend([
                    {
                        'user_id': user_id,
                        'item_id': rec['item_id'],
                        'sido': rec['sido'],
                        'predicted_rating': rec['predicted_rating'],
                        'confidence_score': rec['confidence_score']
                    }
                    for rec in user_recs['recommendations']
                ])
            except Exception as e:
                logging.warning(f"Failed to generate recommendations for user {user_id}: {str(e)}")

        # Save recommendations
        recommendations_df = pd.DataFrame(all_recommendations)
        recommendations_df.to_csv(output_path, index=False)

    return results


def main():
    # Configuration
    config = {
        'base_dir': './experiments',
        'visit_data_path': './data/Training/label/csv/tn_visit_area_info_E.csv',
        'user_data_path': './data/Training/label/csv/tn_traveller_master_E.csv',
        'preprocessed_path': './preprocessed/dfE.csv',
        'diversity_weight': 0.3
    }

    # Start timing
    start_time = datetime.now()

    # Create experiment directory
    experiment_dir, timestamp, directories = create_experiment_directory(config['base_dir'])

    # Setup logging
    log_file = setup_logging(experiment_dir, timestamp)

    logging.info(f"Starting experiment at: {start_time}")

    try:
        # Ensure input files exist
        for path in [config['visit_data_path'], config['user_data_path']]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")

        # Preprocessing
        logging.info("\nPreprocessing data...")
        df, stats = preprocess_pipeline(
            config['visit_data_path'],
            config['user_data_path'],
            config['preprocessed_path']
        )

        # Get algorithm choice
        algorithms = get_algorithm_choice()

        # Run algorithm(s)
        if len(algorithms) > 1:  # Best model mode
            results = find_best_model(df, algorithms, directories, config['diversity_weight'])
        else:  # Single algorithm mode
            results = {algorithms[0]: run_algorithm(
                algorithms[0],
                df,
                directories,
                config['diversity_weight']
            )}

        # Calculate execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Save experiment summary
        summary = {
            'execution_time_seconds': execution_time,
            'configuration': config,
            'preprocessing_stats': stats,
            'results': {}
        }

        for algo_name, result in results.items():
            if result['status'] == 'success':
                summary['results'][algo_name] = {
                    'status': 'success',
                    'metrics': result['metrics']
                }
            else:
                summary['results'][algo_name] = {
                    'status': 'failed',
                    'error': str(result['error'])
                }

        # Save summary
        summary_path = os.path.join(experiment_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Print final summary
        logging.info("\nExecution Summary:")
        logging.info("=" * 50)
        for algo_name, result in results.items():
            if result['status'] == 'success':
                logging.info(f"\n{algo_name}:")
                for metric, value in result['metrics'].items():
                    logging.info(f"  {metric}: {value:.4f}")
            else:
                logging.info(f"\n{algo_name}: Failed - {result['error']}")

        logging.info(f"\nExperiment completed at: {end_time}")
        logging.info(f"Total execution time: {execution_time:.2f} seconds")
        logging.info(f"Results saved in: {experiment_dir}")

        return experiment_dir

    except Exception as e:
        logging.error(f"Error in experiment: {str(e)}")
        raise


if __name__ == "__main__":
    main()