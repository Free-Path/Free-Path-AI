import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os


def calculate_item_similarities(df_original, n_similar=20, save_path='./data/item_similarities.pkl'):
    """Calculate and save item similarities"""

    if os.path.exists(save_path):
        print(f"Loading pre-computed similarities from {save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print("Computing item similarities...")

    # Create item-item similarity matrix
    items = df_original['itemID'].unique()
    item_similarities = {}

    for item1 in tqdm(items):
        item_similarities[item1] = {}
        item1_users = set(df_original[df_original['itemID'] == item1]['userID'])

        for item2 in items:
            if item1 == item2:
                continue

            item2_users = set(df_original[df_original['itemID'] == item2]['userID'])
            common_users = item1_users & item2_users

            if len(common_users) < 2:
                continue

            ratings1 = df_original[
                (df_original['itemID'] == item1) &
                (df_original['userID'].isin(common_users))
                ]['rating']
            ratings2 = df_original[
                (df_original['itemID'] == item2) &
                (df_original['userID'].isin(common_users))
                ]['rating']

            if ratings1.std() == 0 or ratings2.std() == 0:
                continue

            correlation = ratings1.corr(ratings2)
            if not np.isnan(correlation):
                item_similarities[item1][item2] = correlation

        # Keep only top N similar items
        if item_similarities[item1]:
            top_similar = dict(sorted(
                item_similarities[item1].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:n_similar])
            item_similarities[item1] = top_similar

    # Save similarities
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(item_similarities, f)
    print(f"Similarities saved to {save_path}")

    return item_similarities


def load_item_similarities(path='./data/item_similarities.pkl'):
    """Load pre-computed item similarities"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Similarity file not found at {path}")

    with open(path, 'rb') as f:
        return pickle.load(f)


def update_item_similarities(df_original, new_items,
                             existing_similarities_path='./data/item_similarities.pkl',
                             n_similar=20):
    """Update similarities with new items"""

    # Load existing similarities
    if os.path.exists(existing_similarities_path):
        with open(existing_similarities_path, 'rb') as f:
            item_similarities = pickle.load(f)
    else:
        item_similarities = {}

    # Get all existing items
    existing_items = set(item_similarities.keys())

    # Calculate similarities for new items
    print("Computing similarities for new items...")
    for item1 in tqdm(new_items):
        if item1 in existing_items:
            continue

        item_similarities[item1] = {}
        item1_users = set(df_original[df_original['itemID'] == item1]['userID'])

        # Calculate similarity with all items (new and existing)
        all_items = set(df_original['itemID'].unique())
        for item2 in all_items:
            if item1 == item2:
                continue

            item2_users = set(df_original[df_original['itemID'] == item2]['userID'])
            common_users = item1_users & item2_users

            if len(common_users) < 2:
                continue

            ratings1 = df_original[
                (df_original['itemID'] == item1) &
                (df_original['userID'].isin(common_users))
                ]['rating']
            ratings2 = df_original[
                (df_original['itemID'] == item2) &
                (df_original['userID'].isin(common_users))
                ]['rating']

            if ratings1.std() == 0 or ratings2.std() == 0:
                continue

            correlation = ratings1.corr(ratings2)
            if not np.isnan(correlation):
                item_similarities[item1][item2] = correlation
                # Update reverse similarity
                if item2 in item_similarities:
                    item_similarities[item2][item1] = correlation

        # Keep only top N similar items
        if item_similarities[item1]:
            top_similar = dict(sorted(
                item_similarities[item1].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:n_similar])
            item_similarities[item1] = top_similar

    # Save updated similarities
    with open(existing_similarities_path, 'wb') as f:
        pickle.dump(item_similarities, f)
    print(f"Updated similarities saved to {existing_similarities_path}")

    return item_similarities