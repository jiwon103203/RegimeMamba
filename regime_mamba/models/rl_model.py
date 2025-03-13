"""
Neural network models for RL-based investment strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_model import TimeSeriesMamba

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network for RL agent.
    
    The actor predicts actions, and the critic estimates state value.
    Uses Mamba model as the feature extractor for time series data.
    """
    
    def __init__(self, input_dim, d_model=128, d_state=128, n_layers=4, dropout=0.1):
        """
        Initialize the Actor-Critic network.
        
        Args:
            input_dim (int): Dimension of input features
            d_model (int): Model dimension
            d_state (int): State dimension for Mamba
            n_layers (int): Number of Mamba layers
            dropout (float): Dropout rate
        """
        super(ActorCritic, self).__init__()
        
        # Feature extractor (Mamba)
        self.feature_extractor = TimeSeriesMamba(
            input_dim=input_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, features, position=None):
        """
        Forward pass through the network.
        
        Args:
            features (torch.Tensor): Input features
            position (torch.Tensor, optional): Current position
            
        Returns:
            tuple: (action, value)
        """
        # Extract features with Mamba
        if position is not None:
            # Add position information to features
            position_embedded = F.one_hot(position, 3).float()
            position_embedded = position_embedded.unsqueeze(1).repeat(1, features.shape[1], 1)
            # Concatenate features and position
            combined_features = torch.cat([features, position_embedded], dim=-1)
            hidden = self.feature_extractor(combined_features)
        else:
            hidden = self.feature_extractor(features)
        
        # Average over sequence dimension
        hidden = torch.mean(hidden, dim=1)
        
        # Get action and value
        action = self.actor(hidden)
        value = self.critic(hidden)
        
        return action, value
