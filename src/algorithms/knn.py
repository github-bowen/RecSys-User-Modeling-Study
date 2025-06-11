import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from .base_recommender import BaseRecommender

class KNNRecommender(BaseRecommender):
    """K-Nearest Neighbors recommender using user representations.
    
    This recommender finds similar users based on their vector representations
    and recommends items that similar users have interacted with but the target
    user has not.
    """
    
    def __init__(self, user_representations, config=None):
        """Initialize the KNN recommender.
        
        Args:
            user_representations: Dictionary mapping user_id to user vector representation
            config: Dictionary with algorithm-specific configurations
                   - k: Number of neighbors to consider (default: 10)
        """
        super().__init__(user_representations, config)
        self.k = self.config.get('k', 10)
        self.user_item_matrix = None
        self.user_ids = None
    
    def fit(self, train_data):
        """Build the user-item interaction matrix from training data.
        
        Args:
            train_data: DataFrame with columns ['user_id', 'item_id', 'timestamp']
        """
        # Get unique users and items
        all_users = sorted(train_data['user_id'].unique())
        all_items = sorted(train_data['item_id'].unique())
        
        # Only keep users that have vector representations
        valid_users = set(all_users) & set(self.user_representations.keys())
        self.user_ids = np.array(list(valid_users))
        
        # Create mapping dictionaries for fast lookup
        self.user_to_idx = {user: idx for idx, user in enumerate(self.user_ids)}
        self.item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        # Create user-item interaction matrix (binary)
        n_users = len(self.user_ids)
        n_items = len(all_items)
        self.user_item_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        
        # Fill the user-item matrix
        for _, row in train_data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            if user_id in self.user_to_idx and item_id in self.item_to_idx:
                user_idx = self.user_to_idx[user_id]
                item_idx = self.item_to_idx[item_id]
                self.user_item_matrix[user_idx, item_idx] = 1.0
        
        # Pre-compute user similarity matrix
        ## 1. Convert all user vector representations to numpy array, each row is a user vector
        user_vectors = np.array([self.user_representations[user] for user in self.user_ids])
        ## 2. Calculate user similarity matrix using cosine similarity
        ## Result is an n_users x n_users matrix where each element represents similarity between corresponding user pairs
        self.similarity_matrix = cosine_similarity(user_vectors)
        
        self.is_fitted = True
        return self
    
    def _get_top_k_neighbors(self, user_id):
        """Find top K most similar users to the target user.
        
        Args:
            user_id: ID of the target user
            
        Returns:
            List of (user_id, similarity_score) tuples for top-K neighbors
        """
        if user_id not in self.user_to_idx:
            print(f"Error: User {user_id} not found in training data")
            exit(1)
        
        user_idx = self.user_to_idx[user_id]
        similarities = self.similarity_matrix[user_idx]
        
        # Sort by similarity (excluding self)
        similar_users = [(self.user_ids[i], similarities[i]) 
                         for i in range(len(self.user_ids)) 
                         if i != user_idx]
        similar_users.sort(key=lambda x: x[1], reverse=True)
        
        return similar_users[:self.k]
    
    def rank(self, user_id, item_ids):
        """Rank items for a given user based on neighbor preferences.
        
        Args:
            user_id: ID of the target user
            item_ids: List of item IDs to rank
            
        Returns:
            List of item_ids, sorted by predicted relevance (highest first)
        """
        if not self.is_fitted or user_id not in self.user_to_idx:
            # If user not in training data or model not fitted, exit with error
            print(f"Error: User {user_id} not found in training data or model not fitted")
            exit(1)
        
        # Get similar users
        neighbors = self._get_top_k_neighbors(user_id)
        if not neighbors:
            # If no neighbors found, exit with error
            print(f"Error: No neighbors found for user {user_id}")
            exit(1)
        
        # Map item_ids to indices
        valid_items = [item for item in item_ids if item in self.item_to_idx]
        invalid_items = [item for item in item_ids if item not in self.item_to_idx]
        
        scores = defaultdict(float)
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self.user_to_idx[neighbor_id]
            for item in valid_items:
                item_idx = self.item_to_idx[item]
                # Weighted sum of interactions
                scores[item] += similarity * self.user_item_matrix[neighbor_idx, item_idx]
        
        # Sort items by score (descending)
        ranked_items = sorted(valid_items, key=lambda x: scores[x], reverse=True)
        
        # Append items not in training data at the end
        return ranked_items + invalid_items 