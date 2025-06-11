import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from .base_model import BaseUserModel

class DemographicUserModel(BaseUserModel):
    """User modeling strategy based on demographic attributes.
    
    Creates user representations based solely on demographic features
    such as age, gender, occupation, etc.
    """
    
    def __init__(self, config=None):
        """Initialize the demographic user model.
        
        Args:
            config: Dictionary with model-specific configurations
        """
        super().__init__(config)
        self.encoders = {}
    
    def _encode_categorical_features(self, data, column):
        """One-hot encode categorical features.
        
        Args:
            data: DataFrame with the categorical column
            column: Name of the column to encode
            
        Returns:
            Numpy array with one-hot encoded features
        """
        # Initialize encoder if not already done
        if column not in self.encoders:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoders[column] = encoder.fit(data[[column]])
            
        # Transform data
        encoded = self.encoders[column].transform(data[[column]])
        return encoded
    
    def create_user_representations(self, users_data, interactions_data=None):
        """Create vector representations for users based on demographic data.
        
        This method ignores the interactions_data and only uses demographic information.
        
        Args:
            users_data: DataFrame with users demographic data
                Expected columns: 'user_id', 'gender', 'age', 'occupation', 'zip_code'
            interactions_data: Not used in this model
            
        Returns:
            Dictionary mapping user_id to vector representation
        """
        if users_data is None or users_data.empty:
            raise ValueError("Users data is required for demographic user modeling")
        
        # Make a copy to avoid modifying the original
        users = users_data.copy()
        
        # Process age into age groups (binning)
        # MovieLens 1M age values: 1: "Under 18", 18: "18-24", 25: "25-34",
        # 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"
        users['age_group'] = users['age']  # MovieLens 1M already has age groups
        
        # Get one-hot encodings for categorical features
        gender_encoded = self._encode_categorical_features(users, 'gender')
        age_encoded = self._encode_categorical_features(users, 'age_group')
        occupation_encoded = self._encode_categorical_features(users, 'occupation')
        
        # Concatenate all features to create the final user representation
        user_vectors = np.hstack([gender_encoded, age_encoded, occupation_encoded])
        
        # Create dictionary mapping user_id to representation vector
        user_representations = {user_id: user_vectors[i] 
                               for i, user_id in enumerate(users['user_id'])}
        
        return user_representations 