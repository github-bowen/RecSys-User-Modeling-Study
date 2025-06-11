import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .base_model import BaseUserModel

class BehavioralUserModel(BaseUserModel):
    """User modeling strategy based on interaction behaviors.
    
    Creates user representations based on SVD decomposition of the user-item
    interaction matrix. This produces latent factors that capture user preferences.
    """
    
    def __init__(self, config=None):
        """Initialize the behavioral user model.
        
        Args:
            config: Dictionary with model-specific configurations
                   - n_factors: Number of latent factors (default: 50)
        """
        super().__init__(config)
        self.n_factors = self.config.get('n_factors', 50)
        self.svd = None
        
    def create_user_representations(self, users_data=None, interactions_data=None):
        """Create vector representations for users based on interaction data.
        
        This method ignores the users_data and only uses interaction patterns.
        
        Args:
            users_data: Not used in this model
            interactions_data: DataFrame with user-item interactions
                Expected columns: 'user_id', 'item_id', 'rating', 'timestamp'
            
        Returns:
            Dictionary mapping user_id to vector representation
        """
        if interactions_data is None or interactions_data.empty:
            raise ValueError("Interaction data is required for behavioral user modeling")
        
        # Get unique users and items
        users = np.sort(interactions_data['user_id'].unique())
        items = np.sort(interactions_data['item_id'].unique())
        
        # Create mappings for matrix indices
        user_to_idx = {user: i for i, user in enumerate(users)}
        item_to_idx = {item: i for i, item in enumerate(items)}
        
        # Create user-item interaction matrix (implicit feedback)
        rows = interactions_data['user_id'].map(user_to_idx).values
        cols = interactions_data['item_id'].map(item_to_idx).values
        values = np.ones_like(rows)  # Binary interaction indicator
        
        # Create sparse matrix
        n_users = len(users)
        n_items = len(items)
        user_item_matrix = csr_matrix((values, (rows, cols)), shape=(n_users, n_items))
        
        # Apply SVD to get user latent factors
        n_factors = min(self.n_factors, min(user_item_matrix.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        user_factors = self.svd.fit_transform(user_item_matrix)
        
        # Create dictionary mapping user_id to representation vector
        user_representations = {users[i]: user_factors[i] for i in range(n_users)}
        
        return user_representations 