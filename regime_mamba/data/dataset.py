import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..features import get_feature_columns

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
            config: Configuration object (seq_len and the feature-column settings)
        """
        super().__init__()
        self.seq_len = seq_len
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
        self.feature_cols = get_feature_columns(config)

        # Create sequences and targets
        self.sequences = []
        self.targets = []
        self.dates = []

        features = np.array(self.data[self.feature_cols])
        dates = np.array(self.data[date_col])
        if "target_returns_1" in self.data.columns:
            self.target_col = "target_returns_1"
            targets = np.array(self.data[self.target_col])

            for i in range(len(features) - seq_len + 1):
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
