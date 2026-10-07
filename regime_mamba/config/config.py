import torch


class RegimeMambaConfig:
    def __init__(self):
        """Base configuration for the rolling-window Mamba backtest (extended by E2ERegimeMambaConfig)"""
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

        # Input feature settings (see regime_mamba/features.py)
        self.feature_set = None             # None (CSV columns by input_dim 3/4) | paper | example | extra | none
        self.extra_feature_cols = []        # CSV columns appended to the feature set (e.g. ['dollar_index'])
        self.feature_return_col = 'returns' # Return column the feature set is computed from
        self.feature_warmup = 252           # Leading rows dropped after computing the features
        self.standardize_features = True    # Standardize inputs with each window's training-period stats
        self.returns_pct = None             # Whether returns are in % (None: input_dim == 4, or inferred)
        self.feature_cols = None            # Filled by prepare_feature_set()

        # Training settings
        self.batch_size = 1024
        self.learning_rate = 5e-4
        self.max_epochs = 300
        self.patience = 60
        self.transaction_cost = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_clusters = 2  # Number of regimes (Bull / Bear)
        self.seed = 10

        # Rolling window: [train][valid] -> forward, moved by forward_months
        self.start_date = '2010-01-01'
        self.end_date = '2023-12-31'
        self.total_window_years = 20
        self.train_years = 16
        self.valid_years = 4
        self.forward_months = 24

        # Runtime options (set from the command line by the scripts)
        self.gpu_id = 0
        self.enable_checkpointing = False
        self.checkpoint_interval = 1

        self.results_dir = './results'

    def __str__(self):
        """Return configuration information as a string"""
        config_str = f"{type(self).__name__} Configuration:\n"
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
            for key, value in json.load(f).items():
                setattr(config, key, value)

        # Restore device object
        config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return config
