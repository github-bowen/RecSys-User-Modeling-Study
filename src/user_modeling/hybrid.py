import numpy as np

from .base_model import BaseUserModel
from .demographic import DemographicUserModel
from .behavioral import BehavioralUserModel

class HybridUserModel(BaseUserModel):
    """Hybrid user modeling strategy combining demographic and behavioral information.
    
    Creates user representations by concatenating demographic features
    with behavioral latent factors.
    """
    
    def __init__(self, config=None):
        """Initialize the hybrid user model.
        
        Args:
            config: Dictionary with model-specific configurations
                   - demo_config: Configuration for demographic model
                   - behav_config: Configuration for behavioral model
        """
        super().__init__(config)
        # Extract configurations for sub-models
        demo_config = self.config.get('demo_config', {})
        behav_config = self.config.get('behav_config', {})
        
        # Initialize sub-models
        self.demographic_model = DemographicUserModel(config=demo_config)
        self.behavioral_model = BehavioralUserModel(config=behav_config)
    
    def create_user_representations(self, users_data, interactions_data):
        """Create hybrid vector representations for users.
        
        Args:
            users_data: DataFrame with users demographic data
            interactions_data: DataFrame with user-item interactions
            
        Returns:
            Dictionary mapping user_id to combined vector representation
        """
        # Get representations from both models
        demo_representations = self.demographic_model.create_user_representations(
            users_data, interactions_data
        )
        
        behav_representations = self.behavioral_model.create_user_representations(
            users_data, interactions_data
        )
        
        # Find common users
        common_users = set(demo_representations.keys()) & set(behav_representations.keys())
        
        # Combine representations by concatenation
        hybrid_representations = {}
        for user_id in common_users:
            demo_vector = demo_representations[user_id]
            behav_vector = behav_representations[user_id]
            hybrid_vector = np.concatenate([demo_vector, behav_vector])
            hybrid_representations[user_id] = hybrid_vector
            
        return hybrid_representations 