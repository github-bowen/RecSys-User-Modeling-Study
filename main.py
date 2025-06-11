#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for running the user modeling strategies comparison experiment.

This script:
1. Loads and preprocesses the MovieLens 1M dataset
2. Stratifies users based on interaction density
3. Creates different user representations (demographic, behavioral, hybrid)
4. Trains a recommendation model using the specified user representations
5. Evaluates the model on test data for each user stratum
6. Saves the results for further analysis
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import utility functions
from src.utils.data_loader import (
    load_movielens_1m, 
    preprocess_data, 
    stratify_users, 
    train_test_split_leave_one_out, 
    get_negative_samples
)
from src.utils.evaluation import evaluate_recommendations

# Import user modeling strategies
from src.user_modeling.demographic import DemographicUserModel
from src.user_modeling.behavioral import BehavioralUserModel
from src.user_modeling.hybrid import HybridUserModel

# Import recommendation algorithms
from src.algorithms.knn import KNNRecommender
# Additional algorithms could be imported here

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run user modeling strategies comparison experiment'
    )
    
    parser.add_argument(
        '--user_model',
        type=str,
        choices=['demographic', 'behavioral', 'hybrid'],
        required=True,
        help='User modeling strategy to use'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['knn'],  # Add more algorithms as they're implemented
        default='knn',
        help='Recommendation algorithm to use'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def initialize_user_model(model_name, config):
    """Initialize user model based on specified strategy."""
    user_models_config = config.get('user_models', {})
    
    if model_name == 'demographic':
        model_config = user_models_config.get('demographic', {})
        return DemographicUserModel(config=model_config)
    
    elif model_name == 'behavioral':
        model_config = user_models_config.get('behavioral', {})
        return BehavioralUserModel(config=model_config)
    
    elif model_name == 'hybrid':
        model_config = user_models_config.get('hybrid', {})
        return HybridUserModel(config=model_config)
    
    else:
        raise ValueError(f"Unknown user model: {model_name}")

def initialize_recommender(algorithm_name, user_representations, config):
    """Initialize recommender algorithm."""
    algorithms_config = config.get('algorithms', {})
    
    if algorithm_name == 'knn':
        algorithm_config = algorithms_config.get('knn', {})
        return KNNRecommender(user_representations, config=algorithm_config)
    
    # Add more algorithms as they're implemented
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

def save_results(results_df, config):
    """Save experiment results to CSV file."""
    results_dir = config.get('data', {}).get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'metrics.csv')
    
    # If file exists, append without header; if not, create with header
    if os.path.exists(results_path):
        results_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")

def main():
    """Main execution function."""
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # 1. Load and preprocess data
    print("Loading data...")
    data_dir = config.get('data', {}).get('base_dir')
    users_data, ratings_data, movies_data = load_movielens_1m(data_dir)
    
    print("Preprocessing data...")
    min_interactions = config.get('preprocessing', {}).get('min_interactions', 5)
    users_data, ratings_data = preprocess_data(users_data, ratings_data, min_interactions)
    
    # 2. Stratify users
    print("Stratifying users...")
    user_strata = stratify_users(ratings_data)
    
    # 3. Split data
    print("Splitting data into train and test sets...")
    train_data, test_data = train_test_split_leave_one_out(ratings_data)
    
    # 4. Create user representations
    print(f"Creating {args.user_model} user representations...")
    user_model = initialize_user_model(args.user_model, config)
    user_representations = user_model.create_user_representations(users_data, train_data)
    
    # 5. Initialize recommender
    print(f"Initializing {args.algorithm} recommender...")
    recommender = initialize_recommender(args.algorithm, user_representations, config)
    
    # 6. Train recommender
    print("Training recommender...")
    recommender.fit(train_data)
    
    # 7. Evaluate on each stratum
    print("Evaluating recommendations...")
    k = config.get('evaluation', {}).get('k', 10)
    all_items = ratings_data['item_id'].unique()
    
    results = []
    for stratum_name, user_ids in user_strata.items():
        print(f"Evaluating on {stratum_name} users...")
        
        # Filter test data for current stratum
        stratum_test_data = test_data[test_data['user_id'].isin(user_ids)]
        
        # Sample a subset for faster evaluation if needed
        max_eval_users = 200  # Max users to evaluate per stratum
        if len(stratum_test_data) > max_eval_users:
            sampled_users = np.random.choice(
                stratum_test_data['user_id'].unique(), 
                max_eval_users, 
                replace=False
            )
            stratum_test_data = stratum_test_data[stratum_test_data['user_id'].isin(sampled_users)]
        
        # Evaluate
        metrics = evaluate_recommendations(
            recommender, stratum_test_data, train_data, all_items, k
        )
        
        # Store results
        results.append({
            'user_model': args.user_model,
            'algorithm': args.algorithm,
            'user_stratum': stratum_name,
            'success_rate': metrics['success_rate'],
            'mrr': metrics['mrr'],
            'precision': metrics['precision']
        })
        
        print(f"{stratum_name} - Success Rate@{k}: {metrics['success_rate']:.4f}, MRR: {metrics['mrr']:.4f}")
    
    # 8. Save results
    results_df = pd.DataFrame(results)
    save_results(results_df, config)
    
    print("Experiment completed!")

if __name__ == '__main__':
    main() 