from abc import ABC, abstractmethod
import numpy as np

class BaseRecommender(ABC):
    """Base class for recommendation algorithms.
    
    All recommendation algorithms should inherit from this class
    and implement the fit and rank methods.
    """
    
    def __init__(self, user_representations, config=None):
        """Initialize the recommender with user representations.
        
        Args:
            user_representations: Dictionary mapping user_id to user vector representation
            config: Dictionary with algorithm-specific configurations
        """
        self.user_representations = user_representations
        self.config = config or {}
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, train_data):
        """Train the recommendation model.
        
        Args:
            train_data: DataFrame with columns ['user_id', 'item_id', 'timestamp']
                        containing the training interactions
        """
        pass
    
    @abstractmethod
    def rank(self, user_id, item_ids):
        """Rank a list of items for a given user.
        
        Args:
            user_id: ID of the target user
            item_ids: List of item IDs to rank
            
        Returns:
            List of item_ids, sorted by predicted relevance (highest first)
        """
        pass 