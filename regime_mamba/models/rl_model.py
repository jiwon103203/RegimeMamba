"""
Neural network models for RL-based investment strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_model import TimeSeriesMamba

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network for RL agent.
    
    The actor predicts actions, and the critic estimates state value.
    Uses Mamba model as the feature extractor for time series data.
    """
    
    def __init__(self, config, n_positions = 3):
        """
        Initialize the Actor-Critic network.
        
        Args:
            config (dict, optional): Configuration dictionary for additional parameters
        """
        super(ActorCritic, self).__init__()
        
        # Feature extractor (Mamba)
        self.feature_extractor = TimeSeriesMamba(
            input_dim=config.input_dim,
            d_model=config.d_model,
            d_state=config.d_state,
            n_layers=config.n_layers,
            dropout=config.dropout,
            config=config
        )

        self.vae = config.vae
        self.n_positions = n_positions
        self.position_embedding = nn.Embedding(n_positions, config.d_model//16)  if config.vae else nn.Embedding(n_positions, config.d_model) # 3 positions: -1, 0, 1


        if config is not None and config.model_path is not None:
            # Load pre-trained Mamba model
            self.feature_extractor.load_state_dict(torch.load(config.model_path, map_location=config.device))
        
        # Actor network (policy)
        if config.vae:
            # If using VAE, adjust input dimension
            self.actor = nn.Sequential(              # 256//16 = 16 2^8=256, 2^4=16
                nn.Linear(config.d_model//16, config.d_model//32),
                nn.ReLU(),
                nn.Linear(config.d_model//32, config.d_model//64),
                nn.ReLU(),
                nn.Linear(config.d_model//64, 1),
                nn.Tanh()  # Output between -1 and 1
            )

            self.critic = nn.Sequential(
                nn.Linear(config.d_model//16, config.d_model//32),
                nn.ReLU(),
                nn.Linear(config.d_model//32, config.d_model//64),
                nn.ReLU(),
                nn.Linear(config.d_model//64, 1)
            )

        else:
            self.actor = nn.Sequential(
                nn.Linear(config.d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.ReLU(),
                nn.Linear(4, 1),
                nn.Tanh()  # Output between -1 and 1
            )
            
            # Critic network (value function)
            self.critic = nn.Sequential(
                nn.Linear(config.d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.ReLU(),
                nn.Linear(4, 1)
            )
        
        # Store dimensions for debugging
        self.d_model = config.d_model
        self.input_dim = config.input_dim
        
    def forward(self, features, position=None):
        """
        Forward pass through the Actor-Critic network.

        Args:
            features (torch.Tensor): Input features from the environment.
            position (torch.Tensor, optional): Position information for the agent.
        
        Returns:
            action (torch.Tensor): Predicted action.
            value (torch.Tensor): Estimated state value.
        """
        
        # Mamba는 원래 입력 차원으로만 처리
        if self.vae:
            _, _, _, _, hidden, _ = self.feature_extractor(features, return_hidden=True)
        else:
            _, hidden = self.feature_extractor(features, return_hidden=True)
        
        if position is not None:
            hidden = hidden.view(hidden.size(0), -1)
            # Position 정보를 원-핫 인코딩으로 변환
            position_embedded = self.position_embedding(position)
            
            # Mamba의 hidden state와 position embedding을 결합
            combined_hidden = hidden + position_embedded # (1,16) + ()
            
            # Actor와 Critic 네트워크도 입력 차원 변경 필요
            action = self.actor(combined_hidden)
            value = self.critic(combined_hidden)
        else:
            action = self.actor(hidden)
            value = self.critic(hidden)
        
        return action, value