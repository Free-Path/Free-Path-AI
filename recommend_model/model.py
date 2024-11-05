from surprise import (
    SVD, SVDpp, NMF, KNNBasic, KNNWithMeans,
    KNNWithZScore, KNNBaseline, CoClustering,
    BaselineOnly, SlopeOne, Dataset, Reader
)
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import pandas as pd
import pickle
import os
import numpy as np
from collections import defaultdict
from typing import Literal, Dict, Any, Tuple

AlgoType = Literal[
    'svd', 'svdpp', 'nmf', 'knn_basic', 'knn_means',
    'knn_zscore', 'knn_baseline', 'coclustering',
    'baseline', 'slope_one'
]


def get_algorithm(algo_name: AlgoType, params: Dict[str, Any] = None) -> Any:
    """Get algorithm instance based on name and parameters"""
    algorithms = {
        'svd': SVD,
        'svdpp': SVDpp,
        'nmf': NMF,
        'knn_basic': KNNBasic,
        'knn_means': KNNWithMeans,
        'knn_zscore': KNNWithZScore,
        'knn_baseline': KNNBaseline,
        'coclustering': CoClustering,
        'baseline': BaselineOnly,
        'slope_one': SlopeOne
    }

    default_params = {
        'svd': {'n_factors': 100, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02},
        'svdpp': {'n_factors': 20, 'n_epochs': 20, 'lr_all': 0.005},
        'nmf': {'n_factors': 15, 'n_epochs': 50},
        'knn_basic': {'k': 40, 'min_k': 1, 'sim_options': {'name': 'pearson_baseline', 'user_based': True}},
        'knn_means': {'k': 40, 'min_k': 1, 'sim_options': {'name': 'pearson_baseline', 'user_based': True}},
        'knn_zscore': {'k': 40, 'min_k': 1, 'sim_options': {'name': 'pearson_baseline', 'user_based': True}},
        'knn_baseline': {'k': 40, 'min_k': 1, 'sim_options': {'name': 'pearson_baseline', 'user_based': True}},
        'coclustering': {'n_cltr_u': 3, 'n_cltr_i': 3, 'n_epochs': 20},
        'baseline': {'bsl_options': {'method': 'als', 'n_epochs': 5}},
        'slope_one': {}
    }

    if algo_name not in algorithms:
        raise ValueError(f"Algorithm {algo_name} not supported. Choose from: {list(algorithms.keys())}")

    params = params if params is not None else default_params[algo_name]
    return algorithms[algo_name](**params)


def calculate_top_k_metrics(predictions, k=5):
    """Calculate top-k recall and precision"""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= 4.0) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= 4.0) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= 4.0) and (est >= 4.0))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    mean_precision = sum(prec for prec in precisions.values()) / len(precisions) if precisions else 0
    mean_recall = sum(rec for rec in recalls.values()) / len(recalls) if recalls else 0

    return {f'Precision@{k}': mean_precision, f'Recall@{k}': mean_recall}


def evaluate_model(model, testset):
    """Evaluate model performance"""
    predictions = model.test(testset)

    metrics = {
        'RMSE': accuracy.rmse(predictions),
        'MAE': accuracy.mae(predictions)
    }

    for k in [5, 10]:
        metrics.update(calculate_top_k_metrics(predictions, k))

    return metrics, predictions


def train_model(df, algo_name: AlgoType, params: Dict[str, Any] = None) -> Tuple[Any, Dict[str, float], list]:
    """Train the specified model and evaluate its performance"""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = get_algorithm(algo_name, params)
    model.fit(trainset)
    metrics, predictions = evaluate_model(model, testset)

    return model, metrics, predictions


def save_results(model, metrics, algo_name: str, base_dir: str):
    """Save model, metrics, and create result directory structure"""
    # Create directory structure
    algo_dir = os.path.join(base_dir, algo_name)
    model_dir = os.path.join(algo_dir, 'model')
    metrics_dir = os.path.join(algo_dir, 'metrics')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f'model_E_{algo_name}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")

    # Save metrics
    metrics_path = os.path.join(metrics_dir, f'metrics_{algo_name}.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Performance Metrics - {algo_name}\n")
        f.write("=" * 50 + "\n\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Metrics saved to: {metrics_path}")

    return model_path


def load_model(model_path: str) -> Any:
    """Load a saved model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


# ... (이전 코드는 동일) ...

def compare_metrics(metrics_list: list, metric_priority: list = None) -> dict:
    """
    Compare metrics from different models and find the best model

    Parameters:
    -----------
    metrics_list : list of dict
        List of dictionaries containing metrics for each model
    metric_priority : list, optional
        List of metrics to prioritize in order of importance
        Default: ['RMSE', 'MAE', 'Recall@10', 'Precision@10']

    Returns:
    --------
    dict
        Dictionary containing the best model's information
    """
    if not metric_priority:
        metric_priority = ['RMSE', 'MAE', 'Recall@10', 'Precision@10']

    best_metrics = None
    best_idx = None

    for idx, metrics in enumerate(metrics_list):
        if best_metrics is None:
            best_metrics = metrics
            best_idx = idx
            continue

        # Compare metrics in order of priority
        for metric in metric_priority:
            if metric not in metrics or metric not in best_metrics:
                continue

            # For RMSE and MAE, lower is better
            if metric in ['RMSE', 'MAE']:
                if metrics[metric] < best_metrics[metric]:
                    best_metrics = metrics
                    best_idx = idx
                    break
                elif metrics[metric] > best_metrics[metric]:
                    break
            # For all other metrics, higher is better
            else:
                if metrics[metric] > best_metrics[metric]:
                    best_metrics = metrics
                    best_idx = idx
                    break
                elif metrics[metric] < best_metrics[metric]:
                    break

    return {'index': best_idx, 'metrics': best_metrics}


def save_best_model(best_model, best_metrics, algo_name, base_dir: str):
    """Save the best performing model"""
    # Create best model directory
    best_model_dir = os.path.join(base_dir, 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(best_model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # Save metrics and model info
    info_path = os.path.join(best_model_dir, 'best_model_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("Best Model Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Algorithm: {algo_name}\n\n")
        f.write("Performance Metrics:\n")
        f.write("-" * 20 + "\n")
        for metric, value in best_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"\nBest model saved to: {model_path}")
    print(f"Best model info saved to: {info_path}")