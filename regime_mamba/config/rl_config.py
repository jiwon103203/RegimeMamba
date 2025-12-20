"""
Reinforcement Learning Regime Mamba Configuration
Extended configuration class for RL-based end-to-end training.

Replaces Jump Model's Dynamic Programming with Actor-Critic RL.
"""

import torch
import os
from .config import RollingWindowTrainConfig


class RLRegimeMambaConfig(RollingWindowTrainConfig):
    """
    Configuration class for RL-based Regime Mamba training.
    Uses Actor-Critic architecture for regime detection.
    """
    
    def __init__(self):
        super().__init__()
        
        # ============================================
        # RL Algorithm Selection
        # ============================================
        self.rl_algorithm = 'ppo'  # 'ppo', 'a2c', 'sac', 'reinforce'
        
        # ============================================
        # Actor (Policy) Network Parameters
        # ============================================
        self.actor_hidden_dims = [64, 32]  # Hidden layer dimensions
        self.actor_activation = 'relu'  # 'relu', 'tanh', 'gelu'
        self.actor_output_activation = 'softmax'  # For discrete actions
        
        # ============================================
        # Critic (Value) Network Parameters
        # ============================================
        self.critic_hidden_dims = [64, 32]
        self.critic_activation = 'relu'
        self.use_separate_critic = True  # Separate network for value estimation
        
        # ============================================
        # PPO Specific Parameters
        # ============================================
        self.ppo_clip_epsilon = 0.2  # PPO clipping parameter
        self.ppo_epochs = 4  # PPO update epochs per batch
        self.ppo_mini_batch_size = 256
        self.target_kl = 0.01  # Target KL divergence for early stopping
        self.use_gae = True  # Generalized Advantage Estimation
        self.gae_lambda = 0.95  # GAE lambda parameter
        
        # ============================================
        # A2C Specific Parameters
        # ============================================
        self.a2c_entropy_coef = 0.01  # Entropy coefficient for exploration
        self.a2c_value_coef = 0.5  # Value loss coefficient
        
        # ============================================
        # Reward Design Parameters
        # ============================================
        self.reward_type = 'sharpe'  # 'return', 'sharpe', 'sortino', 'calmar'
        self.reward_scale = 100.0  # Reward scaling factor
        self.transaction_penalty = 0.001  # Transaction cost penalty
        self.holding_bonus = 0.0001  # Small bonus for holding position
        self.drawdown_penalty = 0.5  # Penalty for drawdown
        self.risk_free_rate = 0.02 / 252  # Daily risk-free rate
        
        # ============================================
        # Exploration Parameters
        # ============================================
        self.epsilon_start = 1.0  # Initial exploration rate
        self.epsilon_end = 0.05  # Final exploration rate
        self.epsilon_decay_steps = 10000  # Steps to decay epsilon
        self.entropy_coef = 0.01  # Entropy regularization coefficient
        self.entropy_decay = 0.995  # Entropy coefficient decay per epoch
        self.min_entropy_coef = 0.001  # Minimum entropy coefficient
        
        # ============================================
        # Discount and Return Parameters
        # ============================================
        self.gamma = 0.99  # Discount factor
        self.n_steps = 5  # N-step returns
        self.normalize_advantages = True
        self.normalize_rewards = True
        
        # ============================================
        # Experience Replay Parameters (for off-policy)
        # ============================================
        self.use_replay_buffer = False  # PPO/A2C don't use replay
        self.buffer_size = 100000
        self.batch_size_rl = 256
        self.min_buffer_size = 1000  # Minimum samples before training
        
        # ============================================
        # Training Parameters
        # ============================================
        self.max_epochs = 100
        self.patience = 30
        self.learning_rate = 3e-4
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.weight_decay = 0.01
        self.max_grad_norm = 0.5  # Gradient clipping
        self.rollout_length = 2048  # Steps per rollout
        
        # ============================================
        # Mamba Backbone Parameters (Inherited)
        # ============================================
        self.d_model = 8
        self.d_state = 32
        self.d_conv = 4
        self.expand = 2
        self.n_layers = 4
        self.dropout = 0.1
        self.input_dim = 4
        self.seq_len = 60
        
        # ============================================
        # Action Space Parameters
        # ============================================
        self.n_actions = 2  # 0: Bear/Cash, 1: Bull/Long
        self.action_type = 'discrete'  # 'discrete' or 'continuous'
        
        # ============================================
        # State Representation
        # ============================================
        self.use_portfolio_state = True  # Include current position in state
        self.use_return_history = True  # Include recent returns in state
        self.return_history_len = 5  # Number of recent returns to include
        
        # ============================================
        # Rolling Window Parameters (Inherited)
        # ============================================
        self.total_window_years = 20
        self.train_years = 16
        self.valid_years = 4
        self.clustering_years = 0  # Not used in RL
        self.forward_months = 24
        
        # ============================================
        # Results Directory
        # ============================================
        self.results_dir = './rl_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # ============================================
        # Flags
        # ============================================
        self.direct_train = False
        self.jump_model = False  # Replaced by RL
        self.use_onecycle = False  # RL uses its own scheduler
        self.rl_mode = True  # Flag for RL mode
        
    def __str__(self):
        """Return configuration information as a string."""
        config_str = "RL Regime Mamba Configuration:\n"
        config_str += "-" * 50 + "\n"
        
        sections = {
            'RL Algorithm': ['rl_algorithm'],
            'Actor Network': ['actor_hidden_dims', 'actor_activation', 'actor_output_activation'],
            'Critic Network': ['critic_hidden_dims', 'critic_activation', 'use_separate_critic'],
            'PPO Parameters': ['ppo_clip_epsilon', 'ppo_epochs', 'ppo_mini_batch_size', 'target_kl', 'use_gae', 'gae_lambda'],
            'Reward Design': ['reward_type', 'reward_scale', 'transaction_penalty', 'holding_bonus', 'drawdown_penalty'],
            'Exploration': ['epsilon_start', 'epsilon_end', 'entropy_coef', 'entropy_decay'],
            'Discount': ['gamma', 'n_steps', 'normalize_advantages'],
            'Training': ['max_epochs', 'patience', 'actor_lr', 'critic_lr', 'rollout_length'],
            'Mamba': ['d_model', 'd_state', 'n_layers', 'dropout', 'input_dim', 'seq_len'],
            'Rolling Window': ['total_window_years', 'train_years', 'valid_years', 'forward_months']
        }
        
        for section_name, keys in sections.items():
            config_str += f"\n[{section_name}]\n"
            for key in keys:
                if hasattr(self, key):
                    config_str += f"  {key}: {getattr(self, key)}\n"
        
        return config_str
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from a JSON file."""
        import json
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config
    
    def get_actor_config(self) -> dict:
        """Get actor network configuration."""
        return {
            'hidden_dims': self.actor_hidden_dims,
            'activation': self.actor_activation,
            'output_activation': self.actor_output_activation,
            'n_actions': self.n_actions
        }
    
    def get_critic_config(self) -> dict:
        """Get critic network configuration."""
        return {
            'hidden_dims': self.critic_hidden_dims,
            'activation': self.critic_activation,
            'use_separate': self.use_separate_critic
        }
    
    def get_reward_config(self) -> dict:
        """Get reward design configuration."""
        return {
            'reward_type': self.reward_type,
            'reward_scale': self.reward_scale,
            'transaction_penalty': self.transaction_penalty,
            'holding_bonus': self.holding_bonus,
            'drawdown_penalty': self.drawdown_penalty,
            'risk_free_rate': self.risk_free_rate
        }
    
    def get_ppo_config(self) -> dict:
        """Get PPO-specific configuration."""
        return {
            'clip_epsilon': self.ppo_clip_epsilon,
            'ppo_epochs': self.ppo_epochs,
            'mini_batch_size': self.ppo_mini_batch_size,
            'target_kl': self.target_kl,
            'use_gae': self.use_gae,
            'gae_lambda': self.gae_lambda,
            'gamma': self.gamma,
            'normalize_advantages': self.normalize_advantages
        }


class RLConfigPresets:
    """Preset configurations for different RL strategies."""
    
    @staticmethod
    def default() -> RLRegimeMambaConfig:
        """Default balanced PPO configuration."""
        config = RLRegimeMambaConfig()
        return config
    
    @staticmethod
    def aggressive_trading() -> RLRegimeMambaConfig:
        """Configuration for aggressive trading with more frequent switches."""
        config = RLRegimeMambaConfig()
        config.transaction_penalty = 0.0005  # Lower transaction penalty
        config.holding_bonus = 0.0  # No holding bonus
        config.gamma = 0.95  # Lower discount factor
        config.entropy_coef = 0.02  # Higher exploration
        config.reward_type = 'return'  # Focus on raw returns
        return config
    
    @staticmethod
    def conservative_trading() -> RLRegimeMambaConfig:
        """Configuration for conservative trading with fewer switches."""
        config = RLRegimeMambaConfig()
        config.transaction_penalty = 0.002  # Higher transaction penalty
        config.holding_bonus = 0.0005  # Bonus for holding
        config.gamma = 0.99  # Higher discount factor
        config.entropy_coef = 0.005  # Lower exploration
        config.reward_type = 'sharpe'  # Risk-adjusted returns
        return config
    
    @staticmethod
    def risk_adjusted() -> RLRegimeMambaConfig:
        """Configuration emphasizing risk-adjusted returns."""
        config = RLRegimeMambaConfig()
        config.reward_type = 'sortino'  # Downside risk focus
        config.drawdown_penalty = 1.0  # High drawdown penalty
        config.gamma = 0.99
        config.entropy_coef = 0.01
        return config
    
    @staticmethod
    def a2c_config() -> RLRegimeMambaConfig:
        """Configuration for A2C algorithm."""
        config = RLRegimeMambaConfig()
        config.rl_algorithm = 'a2c'
        config.a2c_entropy_coef = 0.01
        config.a2c_value_coef = 0.5
        config.n_steps = 5
        config.use_gae = True
        return config
    
    @staticmethod
    def high_capacity() -> RLRegimeMambaConfig:
        """Configuration with larger model capacity."""
        config = RLRegimeMambaConfig()
        config.d_model = 16
        config.d_state = 64
        config.n_layers = 6
        config.actor_hidden_dims = [128, 64, 32]
        config.critic_hidden_dims = [128, 64, 32]
        config.max_epochs = 150
        return config
    
    @staticmethod
    def fast_training() -> RLRegimeMambaConfig:
        """Configuration for faster training."""
        config = RLRegimeMambaConfig()
        config.d_model = 4
        config.d_state = 16
        config.n_layers = 2
        config.actor_hidden_dims = [32, 16]
        config.critic_hidden_dims = [32, 16]
        config.max_epochs = 50
        config.rollout_length = 1024
        config.ppo_epochs = 2
        return config
