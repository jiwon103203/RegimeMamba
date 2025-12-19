"""
End-to-End Regime Mamba Configuration
Extended configuration class with additional parameters for E2E training.

Updated: Two-Level Entropy parameters, Direction Loss removed
"""

import torch
import os
from .config import RollingWindowTrainConfig


class E2ERegimeMambaConfig(RollingWindowTrainConfig):
    """
    Configuration class for End-to-End Regime Mamba training.
    Extends RollingWindowTrainConfig with Gumbel Softmax and loss parameters.
    
    Updated: Two-Level Entropy Regularization, No Direction Loss
    """
    
    def __init__(self):
        super().__init__()
        
        # ============================================
        # Gumbel Softmax Parameters
        # ============================================
        self.temperature = 1.0          # Initial Gumbel Softmax temperature
        self.initial_temp = 1.0         # Starting temperature (lower = sharper)
        self.final_temp = 0.3           # Final temperature after annealing
        self.temp_schedule = 'exponential'  # Temperature schedule: 'linear', 'exponential', 'cosine'
        self.warmup_epochs = 5          # Warmup epochs before temperature annealing
        
        # ============================================
        # Regime Head Parameters
        # ============================================
        self.regime_hidden_dim = None   # Hidden dimension for regime projection (None = direct projection)
        self.n_clusters = 2             # Number of regimes (2 for Bull/Bear)
        
        # ============================================
        # Loss Weights (Updated - NO direction loss)
        # ============================================
        self.w_return = 0.5            # Return prediction loss weight (auxiliary)
        self.w_jump = 0.5              # Jump penalty loss weight
        self.w_separation = 2.0        # Regime separation loss weight
        self.w_entropy = 1.0           # Two-level entropy weight (INCREASED)
        
        # NOTE: w_direction removed - return sign assumption eliminated
        
        # ============================================
        # Jump Penalty Parameters
        # ============================================
        self.jump_penalty = 1.0        # Jump penalty coefficient (λ)
        self.jump_penalty_type = 'l2'  # Jump penalty type: 'l1', 'l2', 'kl'
        
        # ============================================
        # Separation Loss Parameters
        # ============================================
        self.separation_margin = 2.0   # Margin for centroid separation
        self.separation_loss_type = 'centroid'  # Loss type: 'centroid', 'contrastive', 'silhouette', 'return_weighted'
        self.lambda_inter = 1.5        # Weight for inter-cluster separation
        self.lambda_intra = 1.0        # Weight for intra-cluster compactness
        
        # ============================================
        # Two-Level Entropy Parameters (NEW)
        # ============================================
        self.lambda_entropy = 1.0      # Overall entropy regularization coefficient
        self.target_entropy = None     # Target entropy (None = use two-level approach)
        self.lambda_sample_entropy = 1.0   # Sample-level: minimize (confidence)
        self.lambda_batch_entropy = 0.5    # Batch-level: maximize (balance)
        
        # ============================================
        # Training Parameters (Override defaults)
        # ============================================
        self.max_epochs = 100
        self.patience = 30
        self.learning_rate = 5e-4
        self.weight_decay = 0.01
        self.batch_size = 4096
        
        # ============================================
        # Model Architecture (Inherited + Extended)
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
        # Rolling Window Parameters (Inherited)
        # ============================================
        self.total_window_years = 20
        self.train_years = 16
        self.valid_years = 4
        self.clustering_years = 0  # Not used in E2E
        self.forward_months = 24
        
        # ============================================
        # Results Directory
        # ============================================
        self.results_dir = './e2e_results'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # ============================================
        # Flags
        # ============================================
        self.direct_train = False  # Not used in E2E (regime is learned end-to-end)
        self.jump_model = False    # Using differentiable jump penalty instead
        self.use_onecycle = True
    
    def __str__(self):
        """Return configuration information as a string."""
        config_str = "E2E RegimeMamba Configuration:\n"
        config_str += "-" * 50 + "\n"
        
        sections = {
            'Gumbel Softmax': ['temperature', 'initial_temp', 'final_temp', 'temp_schedule', 'warmup_epochs'],
            'Regime Head': ['regime_hidden_dim', 'n_clusters'],
            'Loss Weights': ['w_return', 'w_jump', 'w_separation', 'w_entropy'],
            'Jump Penalty': ['jump_penalty', 'jump_penalty_type'],
            'Separation Loss': ['separation_margin', 'separation_loss_type', 'lambda_inter', 'lambda_intra'],
            'Two-Level Entropy': ['lambda_entropy', 'lambda_sample_entropy', 'lambda_batch_entropy', 'target_entropy'],
            'Training': ['max_epochs', 'patience', 'learning_rate', 'weight_decay', 'batch_size'],
            'Model': ['d_model', 'd_state', 'n_layers', 'dropout', 'input_dim', 'seq_len'],
            'Rolling Window': ['total_window_years', 'train_years', 'valid_years', 'forward_months']
        }
        
        for section_name, keys in sections.items():
            config_str += f"\n[{section_name}]\n"
            for key in keys:
                if hasattr(self, key):
                    config_str += f"  {key}: {getattr(self, key)}\n"
        
        return config_str
    
    @classmethod
    def from_base_config(cls, base_config):
        """
        Create E2E config from base RegimeMambaConfig.
        
        Args:
            base_config: Base configuration object
            
        Returns:
            E2ERegimeMambaConfig with copied base parameters
        """
        e2e_config = cls()
        
        # Copy common parameters
        common_params = [
            'data_path', 'd_model', 'd_state', 'd_conv', 'expand', 'n_layers',
            'dropout', 'input_dim', 'seq_len', 'batch_size', 'learning_rate',
            'max_epochs', 'patience', 'transaction_cost', 'device', 'seed',
            'n_clusters', 'start_date', 'end_date', 'total_window_years',
            'train_years', 'valid_years', 'forward_months', 'results_dir'
        ]
        
        for param in common_params:
            if hasattr(base_config, param):
                setattr(e2e_config, param, getattr(base_config, param))
        
        return e2e_config
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from a JSON file."""
        import json
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        # Restore device object
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config
    
    def get_loss_config(self) -> dict:
        """Get loss-related configuration as dictionary."""
        return {
            'w_return': self.w_return,
            'w_jump': self.w_jump,
            'w_separation': self.w_separation,
            'w_entropy': self.w_entropy,
            'jump_penalty': self.jump_penalty,
            'jump_penalty_type': self.jump_penalty_type,
            'separation_margin': self.separation_margin,
            'separation_loss_type': self.separation_loss_type,
            'lambda_entropy': self.lambda_entropy,
            'lambda_sample_entropy': self.lambda_sample_entropy,
            'lambda_batch_entropy': self.lambda_batch_entropy
        }
    
    def get_temperature_config(self) -> dict:
        """Get temperature-related configuration as dictionary."""
        return {
            'initial_temp': self.initial_temp,
            'final_temp': self.final_temp,
            'temp_schedule': self.temp_schedule,
            'warmup_epochs': self.warmup_epochs
        }
    
    def get_entropy_config(self) -> dict:
        """Get entropy-related configuration as dictionary."""
        return {
            'lambda_entropy': self.lambda_entropy,
            'lambda_sample_entropy': self.lambda_sample_entropy,
            'lambda_batch_entropy': self.lambda_batch_entropy,
            'target_entropy': self.target_entropy
        }


# Preset configurations for different scenarios
class E2EConfigPresets:
    """Preset configurations for different use cases."""
    
    @staticmethod
    def default() -> E2ERegimeMambaConfig:
        """
        Default balanced configuration with Two-Level Entropy.
        """
        config = E2ERegimeMambaConfig()
        return config
    
    @staticmethod
    def aggressive_trading() -> E2ERegimeMambaConfig:
        """
        Configuration for aggressive trading with more frequent regime switches.
        Lower jump penalty, higher sensitivity.
        """
        config = E2ERegimeMambaConfig()
        config.jump_penalty = 0.3
        config.w_jump = 0.3
        config.w_separation = 2.5
        config.separation_margin = 2.5
        config.initial_temp = 0.8
        config.final_temp = 0.2
        # Higher sample entropy weight for more confident predictions
        config.lambda_sample_entropy = 1.5
        config.lambda_batch_entropy = 0.5
        return config
    
    @staticmethod
    def conservative_trading() -> E2ERegimeMambaConfig:
        """
        Configuration for conservative trading with fewer regime switches.
        Higher jump penalty, longer holding periods.
        """
        config = E2ERegimeMambaConfig()
        config.jump_penalty = 2.0
        config.w_jump = 1.5
        config.w_separation = 1.5
        config.separation_margin = 1.5
        config.initial_temp = 1.0
        config.final_temp = 0.5
        # Lower sample entropy weight for more gradual learning
        config.lambda_sample_entropy = 0.8
        config.lambda_batch_entropy = 0.6
        config.w_entropy = 0.5
        return config
    
    @staticmethod
    def balanced() -> E2ERegimeMambaConfig:
        """
        Balanced configuration for moderate trading frequency.
        Uses default values which are already balanced.
        """
        config = E2ERegimeMambaConfig()
        return config
    
    @staticmethod
    def strong_separation() -> E2ERegimeMambaConfig:
        """
        Configuration emphasizing strong regime separation.
        Good for when regimes are collapsing to similar values.
        """
        config = E2ERegimeMambaConfig()
        config.w_separation = 3.0
        config.separation_margin = 3.0
        config.lambda_inter = 2.0
        config.lambda_intra = 1.0
        config.initial_temp = 0.7
        config.final_temp = 0.2
        config.w_jump = 0.3
        # Stronger confidence push
        config.lambda_sample_entropy = 1.5
        config.w_entropy = 1.5
        return config
    
    @staticmethod
    def high_confidence() -> E2ERegimeMambaConfig:
        """
        Configuration emphasizing confident regime predictions.
        Higher sample-level entropy weight to push away from [0.5, 0.5].
        """
        config = E2ERegimeMambaConfig()
        config.lambda_sample_entropy = 2.0   # Strong confidence push
        config.lambda_batch_entropy = 0.5    # Maintain balance
        config.w_entropy = 1.5               # Higher overall weight
        config.initial_temp = 0.8
        config.final_temp = 0.2
        return config
    
    @staticmethod
    def high_capacity() -> E2ERegimeMambaConfig:
        """
        Configuration with larger model capacity.
        """
        config = E2ERegimeMambaConfig()
        config.d_model = 16
        config.d_state = 64
        config.n_layers = 6
        config.regime_hidden_dim = 32
        config.max_epochs = 150
        return config
    
    @staticmethod
    def fast_training() -> E2ERegimeMambaConfig:
        """
        Configuration for faster training (smaller model, fewer epochs).
        """
        config = E2ERegimeMambaConfig()
        config.d_model = 4
        config.d_state = 16
        config.n_layers = 2
        config.max_epochs = 50
        config.patience = 15
        config.batch_size = 4096
        return config
    
    @staticmethod
    def debug_symmetry_breaking() -> E2ERegimeMambaConfig:
        """
        Configuration for debugging symmetry breaking issues.
        Very strong confidence push, good for diagnosing [0.5, 0.5] problems.
        """
        config = E2ERegimeMambaConfig()
        config.lambda_sample_entropy = 3.0   # Very strong confidence push
        config.lambda_batch_entropy = 0.3    # Slightly lower balance
        config.w_entropy = 2.0               # Very high entropy weight
        config.w_separation = 1.0            # Lower separation initially
        config.w_jump = 0.1                  # Very low jump penalty
        config.initial_temp = 0.5            # Start with lower temperature
        config.final_temp = 0.1
        return config
