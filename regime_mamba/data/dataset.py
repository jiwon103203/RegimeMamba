import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class RegimeMambaDataset(Dataset):
    def __init__(self, config, mode="train"):
        """
        Dataset class for RegimeMamba model

        Args:
            path: Data file path
            config: Configuration object
            mode: One of 'train', 'valid', 'test'
        """
        super().__init__()
        self.data = pd.read_csv(config.data_path)
        # epsilon = 1e-10
        # self.data['Close'] = np.log(self.data['Close'] + epsilon) - np.log(self.data['Close'].shift(1) + epsilon)
        # self.data['Open'] = np.log(self.data['Open'] + epsilon) - np.log(self.data['Close'].shift(1) + epsilon)
        # self.data['High'] = np.log(self.data['High'] + epsilon) - np.log(self.data['Close'].shift(1) + epsilon)
        # self.data['Low'] = np.log(self.data['Low'] + epsilon) - np.log(self.data['Close'].shift(1) + epsilon)

        # Handle null values
        self.data = self.data.fillna(0)
        self.seq_len = config.seq_len

        # Define feature columns
        if config.input_dim == 3:
            self.feature_cols = ["dd_10", "sortino_20", "sortino_60"]
        elif config.input_dim == 4:
            self.feature_cols = ["dd_10", "sortino_20", "sortino_60", "dollar_index"]


        # Split data based on date
        date_col = 'Date'

        if mode == "train":
            self.subset = self.data[(self.data[date_col] >= '1970-01-01') & (self.data[date_col] <= '1999-12-31')]
        elif mode == "valid":
            self.subset = self.data[(self.data[date_col] >= '2000-01-01') & (self.data[date_col] <= '2009-12-31')]
        elif mode == "test":
            self.subset = self.data[self.data[date_col] >= '2010-01-01']

        # Create sequences and targets
        self.sequences = []
        self.targets = []
        self.dates = []  # Store date information
        self.seq_len = config.seq_len

        features = np.array(self.subset[self.feature_cols])
        dates = np.array(self.subset[date_col])

        if config.direct_train:
            self.target_col = "target_returns_1_c"
            targets = np.array(self.subset[self.target_col])
            targets = torch.nn.functional.one_hot(torch.tensor(targets), num_classes=3).numpy()

            for i in range(len(features) - config.seq_len+1):
                self.sequences.append(features[i:i+config.seq_len])
                self.targets.append(targets[i+config.seq_len-1])
                self.dates.append(dates[i+config.seq_len-1])

        elif "target_returns_1" in self.data.columns:
            
            self.target_col="target_returns_1"
            targets = np.array(self.subset[self.target_col])

            for i in range(len(features) - config.seq_len+1):
                self.sequences.append(features[i:i+config.seq_len])
                self.targets.append(targets[i+config.seq_len-1])
                self.dates.append(dates[i+config.seq_len-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
            
        return(
                torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                self.dates[idx],
                torch.tensor(np.array(self.subset['returns'])[idx], dtype=torch.float32)
        )

class DateRangeRegimeMambaDataset(Dataset):
    def __init__(self, data=None, path=None, seq_len=128, start_date=None, end_date=None, 
                 config=None):
        """
        Dataset class that filters data based on date range

        Args:
            data: Full dataframe (load from path if None)
            path: Data file path (used if data is None)
            seq_len: Sequence length
            start_date: Start date (string, 'YYYY-MM-DD' format)
            end_date: End date (string, 'YYYY-MM-DD' format)
            config: Configuration object (required attributes: direct_train)
        """
        super().__init__()
        self.seq_len = seq_len
        self.direct_train = config.direct_train
        # Load data
        if data is None and path is not None:
            data = pd.read_csv(path)
            
            # epsilon = 1e-10
            # data['Close'] = np.log(data['Close'] + epsilon) - np.log(data['Close'].shift(1) + epsilon)
            # data['Open'] = np.log(data['Open'] + epsilon) - np.log(data['Close'].shift(1) + epsilon)
            # data['High'] = np.log(data['High'] + epsilon) - np.log(data['Close'].shift(1) + epsilon)
            # data['Low'] = np.log(data['Low'] + epsilon) - np.log(data['Close'].shift(1) + epsilon)

            # Handle null values
            data = data.fillna(0)

        # Error if no data provided
        if data is None:
            raise ValueError("Either data or path must be provided")
        if config is None:
            raise ValueError("Config must be provided")

        # Identify date column
        date_col = 'Date'

        # Filter by date
        if start_date and end_date:
            self.data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)].copy()
        elif start_date:
            self.data = data[data[date_col] >= start_date].copy()
        elif end_date:
            self.data = data[data[date_col] <= end_date].copy()
        else:
            self.data = data.copy()

        # Define feature columns
        if config.input_dim == 3:
            self.feature_cols = ["dd_10", "sortino_20", "sortino_60"]
        elif config.input_dim == 4:
            self.feature_cols = ["dd_10", "sortino_20", "sortino_60", "dollar_index"]

        # Create sequences and targets
        self.sequences = []
        self.targets = []
        self.dates = []

        features = np.array(self.data[self.feature_cols])
        dates = np.array(self.data[date_col])
        if self.direct_train:
            self.target_col = "target_returns_1_c"
            targets = np.array(self.data[self.target_col])
            targets = torch.nn.functional.one_hot(torch.tensor(targets), num_classes=3).numpy()

            for i in range(len(features) - config.seq_len+1):
                self.sequences.append(features[i:i+config.seq_len])
                self.targets.append(targets[i+config.seq_len-1])
                self.dates.append(dates[i+config.seq_len-1])

        elif "target_returns_1" in self.data.columns:
            
            self.target_col="target_returns_1"
            targets = np.array(self.data[self.target_col])

            for i in range(len(features) - seq_len+1):
                self.sequences.append(features[i:i+seq_len])
                self.targets.append(targets[i+seq_len-1])
                self.dates.append(dates[i+seq_len-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return(
                torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                self.dates[idx],
                torch.tensor(np.array(self.data['returns'])[idx], dtype=torch.float32)
        )

def create_dataloaders(config):
    """
    Function to create datasets and dataloaders
    
    Args:
        config: Configuration object (required attributes: data_path, seq_len, batch_size)
        
    Returns:
        train_loader, valid_loader, test_loader: Tuple of dataloaders
    """

    train_dataset = RegimeMambaDataset(
        config=config,
        mode="train"
    )
    
    valid_dataset = RegimeMambaDataset(
        config=config,
        mode="valid",
    )
    
    test_dataset = RegimeMambaDataset(
        config=config, 
        mode="test"
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, valid_loader, test_loader

def create_date_range_dataloader(data=None, path=None, seq_len=128, batch_size=64, 
                                start_date=None, end_date=None, shuffle=False, num_workers=2):
    """
    Utility function to create dataloader based on date range
    
    Args:
        data: Dataframe (load from path if None)
        path: Data file path (used if data is None)
        seq_len: Sequence length
        batch_size: Batch size
        start_date: Start date
        end_date: End date
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        
    Returns:
        dataloader: Created dataloader
    """
    dataset = DateRangeRegimeMambaDataset(
        data=data,
        path=path,
        seq_len=seq_len,
        start_date=start_date,
        end_date=end_date
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader