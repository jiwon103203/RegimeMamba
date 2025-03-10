import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class RegimeMambaDataset(Dataset):
    def __init__(self, path, seq_len=128, mode="train"):
        """
        RegimeMamba 모델을 위한 데이터셋 클래스

        Args:
            path: 데이터 파일 경로
            seq_len: 시퀀스 길이
            mode: 'train', 'valid', 'test' 중 하나
        """
        super().__init__()
        self.data = pd.read_csv(path)
        self.data = self.data.iloc[2:]
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data['returns'] = self.data['returns'] * 100
        self.data["dd_10"] = self.data["dd_10"] * 100
        self.data["sortino_20"] = self.data["sortino_20"] * 100
        self.data["sortino_60"] = self.data["sortino_60"] * 100
        self.seq_len = seq_len

        # 특성과 타겟 칼럼 지정
        self.feature_cols = ["returns", "dd_10", "sortino_20", "sortino_60"]
        self.target_col = "returns"  # 수익률을 타겟으로 사용

        # 일자 기준으로 데이터 분할
        date_col = 'Price' if 'Price' in self.data.columns else 'Date'

        if mode == "train":
            self.subset = self.data[(self.data[date_col] >= '1970-01-01') & (self.data[date_col] <= '1999-12-31')]
        elif mode == "valid":
            self.subset = self.data[(self.data[date_col] >= '2000-01-01') & (self.data[date_col] <= '2009-12-31')]
        elif mode == "test":
            self.subset = self.data[self.data[date_col] >= '2010-01-01']

        # 시퀀스 및 타겟 생성
        self.sequences = []
        self.targets = []
        self.dates = []  # 날짜 정보도 저장

        features = np.array(self.subset[self.feature_cols])
        dates = np.array(self.subset[date_col])

        for i in range(len(features) - seq_len):
            self.sequences.append(features[i:i+seq_len])
            self.targets.append([features[i+seq_len][0]])  # 바로 다음날 수익률
            self.dates.append(dates[i+seq_len])  # 타겟 날짜 저장

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            self.dates[idx]
        )

def create_dataloaders(config):
    """
    데이터셋 및 데이터로더 생성 함수
    
    Args:
        config: 설정 객체
        
    Returns:
        train_loader, valid_loader, test_loader: 데이터로더 튜플
    """
    train_dataset = RegimeMambaDataset(config.data_path, seq_len=config.seq_len, mode="train")
    valid_dataset = RegimeMambaDataset(config.data_path, seq_len=config.seq_len, mode="valid")
    test_dataset = RegimeMambaDataset(config.data_path, seq_len=config.seq_len, mode="test")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return train_loader, valid_loader, test_loader