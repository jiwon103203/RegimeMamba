import torch
import os

class RegimeMambaConfig:
    def __init__(self):
        """RegimeMamba model configuration class"""
        # Data-related settings
        self.data_path = None
        
        # Model structure settings
        self.d_model = 8
        self.d_state = 32
        self.d_conv = 4
        self.expand = 2
        self.n_layers = 4
        self.dropout = 0.1
        self.input_dim = 4
        self.seq_len = 60
        
        # Training settings
        self.batch_size = 1024
        self.learning_rate = 5e-4
        self.max_epochs = 300
        self.patience = 60
        self.transaction_cost = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.direct_train = False
        self.use_onecycle = True

        # Clustering settings
        self.n_clusters = 2  # Clustering into two regimes: Bull and Bear
        self.cluster_method = 'cosine_kmeans'

        # Extra settings
        self.jump_model = False
        self.jump_penalty = 0
        self.n_positions = 3
        self.lstm = False
        self.seed = 10
        self.scale = 10

        # Prediction settings
        self.predict = False

    def __str__(self):
        """Return configuration information as a string"""
        config_str = "RegimeMamba Configuration:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str
        
    def save(self, filepath):
        """Save configuration to a JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4, default=str)
        
    @classmethod
    def load(cls, filepath):
        """Load configuration from a JSON file"""
        import json
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        # Restore device object
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config

class RollingWindowConfig(RegimeMambaConfig):

    def __init__(self):
        """Rolling Window based RegimeMamba model configuration class"""
        super().__init__()
        self.lookback_years = 10      # Past data period (years) for clustering
        self.forward_months = 12      # Future application period (months)
        self.start_date = '2010-01-01'  # Backtest start date
        self.end_date = '2023-12-31'    # Backtest end date
        self.transaction_cost = 0.001 # Transaction cost (0.1%)
        self.model_path = None        # Path to pretrained model

        # Save path
        self.results_dir = './rolling_window_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def __str__(self):
        """Return configuration information as a string"""
        config_str = "RollingWindowConfig Configuration:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from a JSON file"""
        import json
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        # Restore device object
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config
    
class RollingWindowTrainConfig(RollingWindowConfig):
    def __init__(self):
        """Rolling Window based RegimeMamba training configuration class"""
        super().__init__()
        self.total_window_years = 40
        self.train_years = 20
        self.valid_years = 10
        self.clustering_years = 10
        self.forward_months = 60

        # Training settings
        self.max_epochs = 100

        self.apply_filtering = True
        self.filter_method = 'minimum_holding'
        self.min_holding_days = 20

        self.results_dir = './rolling_window_train_results'
        os.makedirs(self.results_dir, exist_ok=True)

    def __str__(self):
        """Return configuration information as a string"""
        config_str = "RollingWindowTrainconfig Configuration:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str
    
    @classmethod
    def load(cls, filepath):
        """Load configuration from a JSON file"""
        import json
        config = cls()
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
        
        # Restore device object
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config