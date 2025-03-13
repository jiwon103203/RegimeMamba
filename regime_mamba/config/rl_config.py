"""
Configuration classes and utilities for RL-based investment.
"""

import os
import torch

class RLInvestmentConfig:
    """Configuration for RL-based investment."""
    
    def __init__(self):
        """Initialize RL investment configuration with default values."""
        # Data related settings
        self.data_path = None
        self.total_window_years = 40
        self.train_years = 20
        self.valid_years = 10
        self.forward_months = 60
        self.start_date = '1990-01-01'
        self.end_date = '2023-12-31'
        
        # RL related settings
        self.reward_type = 'sharpe'  # 'sharpe' or 'diff_sharpe'
        self.position_penalty = 0.01
        self.window_size = 252  # Window size for Sharpe calculation
        self.gamma = 0.99  # Discount factor
        self.n_episodes = 100
        self.steps_per_episode = 2048
        self.n_epochs = 10
        self.batch_size = 64
        self.transaction_cost = 0.001
        
        # Model related settings
        self.d_model = 128
        self.d_state = 128
        self.n_layers = 4
        self.dropout = 0.1
        self.seq_len = 128
        self.learning_rate = 1e-4
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results directory
        self.results_dir = './rl_investment_results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def save_config(self, output_dir):
        """
        Save configuration to file.
        
        Args:
            output_dir: Directory to save configuration
        """
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, 'config.txt')
        
        with open(config_path, 'w') as f:
            f.write("=== RL Investment Configuration ===\n")
            f.write(f"Data path: {self.data_path}\n")
            f.write(f"Start date: {self.start_date}\n")
            f.write(f"End date: {self.end_date}\n")
            f.write(f"Total data period: {self.total_window_years} years\n")
            f.write(f"Training period: {self.train_years} years\n")
            f.write(f"Validation period: {self.valid_years} years\n")
            f.write(f"Forward testing period: {self.forward_months} months\n")
            f.write(f"Model dimension: {self.d_model}\n")
            f.write(f"State dimension: {self.d_state}\n")
            f.write(f"Number of layers: {self.n_layers}\n")
            f.write(f"Dropout rate: {self.dropout}\n")
            f.write(f"Sequence length: {self.seq_len}\n")
            f.write(f"Batch size: {self.batch_size}\n")
            f.write(f"Learning rate: {self.learning_rate}\n")
            f.write(f"Reward type: {self.reward_type}\n")
            f.write(f"Position penalty: {self.position_penalty}\n")
            f.write(f"Window size for Sharpe: {self.window_size}\n")
            f.write(f"Number of episodes: {self.n_episodes}\n")
            f.write(f"Steps per episode: {self.steps_per_episode}\n")
            f.write(f"Number of update epochs: {self.n_epochs}\n")
            f.write(f"Discount factor: {self.gamma}\n")
            f.write(f"Transaction cost: {self.transaction_cost}\n")
            f.write(f"Device: {self.device}\n")
        
        return config_path
    
    @classmethod
    def from_args(cls, args):
        """
        Create configuration from arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            RLInvestmentConfig: Configuration object
        """
        config = cls()
        
        # Data parameters
        if hasattr(args, 'data_path'):
            config.data_path = args.data_path
        if hasattr(args, 'results_dir'):
            config.results_dir = args.results_dir
        if hasattr(args, 'start_date'):
            config.start_date = args.start_date
        if hasattr(args, 'end_date'):
            config.end_date = args.end_date
        
        # Period settings
        if hasattr(args, 'total_window_years'):
            config.total_window_years = args.total_window_years
        if hasattr(args, 'train_years'):
            config.train_years = args.train_years
        if hasattr(args, 'valid_years'):
            config.valid_years = args.valid_years
        if hasattr(args, 'forward_months'):
            config.forward_months = args.forward_months
        
        # Model parameters
        if hasattr(args, 'd_model'):
            config.d_model = args.d_model
        if hasattr(args, 'd_state'):
            config.d_state = args.d_state
        if hasattr(args, 'n_layers'):
            config.n_layers = args.n_layers
        if hasattr(args, 'dropout'):
            config.dropout = args.dropout
        if hasattr(args, 'seq_len'):
            config.seq_len = args.seq_len
        if hasattr(args, 'batch_size'):
            config.batch_size = args.batch_size
        if hasattr(args, 'learning_rate'):
            config.learning_rate = args.learning_rate
        
        # RL parameters
        if hasattr(args, 'reward_type'):
            config.reward_type = args.reward_type
        if hasattr(args, 'position_penalty'):
            config.position_penalty = args.position_penalty
        if hasattr(args, 'window_size'):
            config.window_size = args.window_size
        if hasattr(args, 'n_episodes'):
            config.n_episodes = args.n_episodes
        if hasattr(args, 'steps_per_episode'):
            config.steps_per_episode = args.steps_per_episode
        if hasattr(args, 'n_epochs'):
            config.n_epochs = args.n_epochs
        if hasattr(args, 'gamma'):
            config.gamma = args.gamma
        
        # Other settings
        if hasattr(args, 'transaction_cost'):
            config.transaction_cost = args.transaction_cost
        
        return config
