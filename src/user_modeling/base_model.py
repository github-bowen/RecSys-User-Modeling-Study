from abc import ABC, abstractmethod

class BaseUserModel(ABC):
    """Base class for user modeling strategies.
    
    All user modeling strategies should inherit from this class
    and implement the create_user_representations method.
    """
    
    def __init__(self, config=None):
        """Initialize the user model.
        
        Args:
            config: Dictionary with model-specific configurations
        """
        self.config = config or {}
    
    @abstractmethod
    def create_user_representations(self, users_data, interactions_data):
        """Create vector representations for users.
        
        Args:
            users_data: DataFrame with users demographic data
            interactions_data: DataFrame with user-item interactions
            
        Returns:
            Dictionary mapping user_id to vector representation
        """
        pass 