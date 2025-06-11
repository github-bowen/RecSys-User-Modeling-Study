import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_movielens_1m(data_dir):
    """Load MovieLens 1M dataset.
    
    Args:
        data_dir: Directory containing the dataset files
        
    Returns:
        users_data: DataFrame with user demographic information
        ratings_data: DataFrame with user-item interactions
        movies_data: DataFrame with movie information
    """
    # Define file paths
    ratings_file = os.path.join(data_dir, 'ratings.dat')
    users_file = os.path.join(data_dir, 'users.dat')
    movies_file = os.path.join(data_dir, 'movies.dat')
    
    # Load ratings data
    ratings_columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_data = pd.read_csv(ratings_file, sep='::', names=ratings_columns, engine='python', encoding='latin-1')
    
    # Load users data
    users_columns = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users_data = pd.read_csv(users_file, sep='::', names=users_columns, engine='python', encoding='latin-1')
    
    # Load movies data
    movies_columns = ['item_id', 'title', 'genres']
    movies_data = pd.read_csv(movies_file, sep='::', names=movies_columns, engine='python', encoding='latin-1')
    
    return users_data, ratings_data, movies_data

def preprocess_data(users_data, ratings_data, min_interactions=5):
    """Preprocess data for recommendation.
    
    Args:
        users_data: DataFrame with user demographic information
        ratings_data: DataFrame with user-item interactions
        min_interactions: Minimum number of interactions per user
        
    Returns:
        users_data: Filtered user data
        ratings_data: Filtered rating data
    """
    # Convert to implicit feedback (all ratings considered as positive interactions)
    ratings_data = ratings_data.sort_values('timestamp')
    
    # Filter users with too few interactions
    user_counts = ratings_data['user_id'].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index
    ratings_data = ratings_data[ratings_data['user_id'].isin(active_users)]
    
    # Keep only users with both demographic data and interactions
    common_users = set(ratings_data['user_id'].unique()) & set(users_data['user_id'].unique())
    ratings_data = ratings_data[ratings_data['user_id'].isin(common_users)]
    users_data = users_data[users_data['user_id'].isin(common_users)]
    
    return users_data, ratings_data

def stratify_users(ratings_data):
    """Stratify users based on interaction density.
    
    Args:
        ratings_data: DataFrame with user-item interactions
        
    Returns:
        Dictionary with user_id lists for each stratum: 'sparse', 'medium', 'dense'
    """
    # Count interactions per user
    user_counts = ratings_data['user_id'].value_counts().reset_index()
    user_counts.columns = ['user_id', 'interaction_count']
    
    # Calculate percentiles
    percentiles = [0, 33.33, 66.67, 100]
    thresholds = np.percentile(user_counts['interaction_count'], percentiles)
    
    # Assign users to strata
    sparse_users = user_counts[(user_counts['interaction_count'] >= thresholds[0]) & 
                               (user_counts['interaction_count'] < thresholds[1])]['user_id'].tolist()
    medium_users = user_counts[(user_counts['interaction_count'] >= thresholds[1]) & 
                               (user_counts['interaction_count'] < thresholds[2])]['user_id'].tolist()
    dense_users = user_counts[(user_counts['interaction_count'] >= thresholds[2])]['user_id'].tolist()
    
    return {
        'sparse': sparse_users,
        'medium': medium_users,
        'dense': dense_users
    }

def train_test_split_leave_one_out(ratings_data):
    """Split data using leave-one-out approach based on timestamp.
    
    For each user, the most recent interaction is used for testing,
    and all other interactions are used for training.
    
    Args:
        ratings_data: DataFrame with user-item interactions
        
    Returns:
        train_data: Training data
        test_data: Test data
    """
    # Sort by timestamp
    ratings_data = ratings_data.sort_values(['user_id', 'timestamp'])
    
    # Group by user and split
    train_data = ratings_data.groupby('user_id').apply(lambda x: x.iloc[:-1])
    test_data = ratings_data.groupby('user_id').apply(lambda x: x.iloc[-1])
    
    # Reset indices
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    
    return train_data, test_data

def get_negative_samples(user_id, test_item_id, ratings_data, n_samples=100):
    """Generate negative samples for a user's test item.
    
    Args:
        user_id: Target user ID
        test_item_id: The positive test item ID
        ratings_data: DataFrame with all user-item interactions
        n_samples: Number of negative samples to generate
        
    Returns:
        List with test_item_id and n_samples negative item IDs
    """
    # Get items the user has interacted with
    user_items = set(ratings_data[ratings_data['user_id'] == user_id]['item_id'])
    
    # Get all items
    all_items = set(ratings_data['item_id'].unique())
    
    # Find items the user has not interacted with
    negative_items = list(all_items - user_items)
    
    # If we don't have enough negative items, sample with replacement
    if len(negative_items) < n_samples:
        negative_samples = np.random.choice(negative_items, size=n_samples, replace=True)
    else:
        negative_samples = np.random.choice(negative_items, size=n_samples, replace=False)
    
    # Combine with the positive test item
    test_samples = [test_item_id] + negative_samples.tolist()
    
    return test_samples 