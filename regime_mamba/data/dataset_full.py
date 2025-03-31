import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class RegimeMambaDataset(Dataset):
    def __init__(self, config, mode="train"):
        """
        RegimeMamba 모델을 위한 데이터셋 클래스

        Args:
            path: 데이터 파일 경로
            config: 설정 객체
            mode: 'train', 'valid', 'test' 중 하나
        """
        super().__init__()
        self.data = pd.read_csv(config.data_path)
        self.data['Close'] = self.data['Close'] / 100 # 스케일 조정
        self.data['Open'] = self.data['Open'] / 100
        self.data['High'] = self.data['High'] / 100
        self.data['Low'] = self.data['Low'] / 100

        self.seq_len = config.seq_len
        self.preprocessed = config.preprocessed

        # 타겟 칼럼 지정
        self.feature_cols = ["Open", "Close", "High", "Low", "treasury_rate"]
        

        # 일자 기준으로 데이터 분할
        date_col = 'Date'

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

        if config.direct_train:
            self.target_col = f"target_returns_{config.target_horizon}_c"
            targets = np.array(self.subset[self.target_col])
            targets = torch.nn.functional.one_hot(torch.tensor(targets), num_classes=3).numpy()

            for i in range(len(features) - config.seq_len+1):
                self.sequences.append(features[i:i+config.seq_len])
                self.targets.append(targets[i+config.seq_len-1])
                # 타겟 날짜 저장 (타겟 기간의 마지막 날짜)
                self.dates.append(dates[i+config.seq_len-1])


        elif config.target_type == "average" and (f"target_SMA_{config.target_horizon}" in self.data.columns or f"target_returns_{config.target_horizon}" in self.data.columns):
            
            print("미리 처리된 데이터 포착")
            
            if self.preprocessed:
                self.target_col=f"target_returns_{config.target_horizon}"
                targets = np.array(self.subset[self.target_col]/self.target_horizon)
            else:
                self.target_col=f"target_SMA_{config.target_horizon}"
                targets = np.array(self.subset[self.target_col] / 100)

            for i in range(len(features) - config.seq_len+1):
                self.sequences.append(features[i:i+config.seq_len])
                self.targets.append(targets[i+config.seq_len-1])
                # 타겟 날짜 저장 (타겟 기간의 마지막 날짜)
                self.dates.append(dates[i+config.seq_len-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
            
        return(
                torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                self.dates[idx],
                torch.tensor(np.array(self.subset['returns']), dtype=torch.float32)
        )

class DateRangeRegimeMambaDataset(Dataset):
    def __init__(self, data=None, path=None, seq_len=128, start_date=None, end_date=None, 
                 config=None):
        """
        날짜 범위를 기반으로 데이터를 필터링하는 데이터셋 클래스

        Args:
            data: 전체 데이터프레임 (None인 경우 path에서 로드)
            path: 데이터 파일 경로 (data가 None인 경우 사용)
            seq_len: 시퀀스 길이
            start_date: 시작 날짜 (문자열, 'YYYY-MM-DD' 형식)
            end_date: 종료 날짜 (문자열, 'YYYY-MM-DD' 형식)
            target_type: 타겟 유형
                - average: 지정된 기간 동안의 평균 수익률
            target_horizon: 타겟 계산을 위한 기간 (일)
            preprocessed: 수익률 전처리 여부
        """
        super().__init__()
        self.seq_len = seq_len
        self.target_type = config.target_type
        self.target_horizon = config.target_horizon
        self.preprocessed = config.preprocessed
        self.direct_train = config.direct_train
        # 데이터 로드
        if data is None and path is not None:
            data = pd.read_csv(path)
            data['returns'] = data['returns'] * 100
            data['Close'] = data['Close'] / 100

        # 데이터가 제공되지 않은 경우 에러
        if data is None:
            raise ValueError("Either data or path must be provided")
        if config is None:
            raise ValueError("Config must be provided")

        # 날짜 칼럼 식별
        date_col = 'Date'

        # 날짜 필터링
        if start_date and end_date:
            self.data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)].copy()
        elif start_date:
            self.data = data[data[date_col] >= start_date].copy()
        elif end_date:
            self.data = data[data[date_col] <= end_date].copy()
        else:
            self.data = data.copy()

        # 타겟 칼럼 지정
        self.feature_cols = ["Open", "Close", "High", "Low", "treasury_rate"]

        # 시퀀스 및 타겟 생성
        self.sequences = []
        self.targets = []
        self.dates = []

        features = np.array(self.data[self.feature_cols])
        dates = np.array(self.data[date_col])
        if self.direct_train:
            self.target_col = f"target_returns_{config.target_horizon}_c"
            targets = np.array(self.data[self.target_col])
            targets = torch.nn.functional.one_hot(torch.tensor(targets), num_classes=3).numpy()

            for i in range(len(features) - config.seq_len+1):
                self.sequences.append(features[i:i+config.seq_len])
                self.targets.append(targets[i+config.seq_len-1])
                # 타겟 날짜 저장 (타겟 기간의 마지막 날짜)
                self.dates.append(dates[i+config.seq_len-1])

        elif config.target_type == "average" and (f"target_SMA_{config.target_horizon}" in self.data.columns or f"target_returns_{config.target_horizon}" in self.data.columns):
            
            print("미리 처리된 데이터 포착")
            
            if self.preprocessed:
                self.target_col=f"target_returns_{config.target_horizon}"
                targets = np.array(self.data[self.target_col])/self.target_horizon
            else:
                self.target_col=f"target_SMA_{config.target_horizon}"
                targets = np.array(self.data[self.target_col])

            for i in range(len(features) - seq_len+1):
                self.sequences.append(features[i:i+seq_len])
                self.targets.append(targets[i+seq_len-1])
                # 타겟 날짜 저장 (타겟 기간의 마지막 날짜)
                self.dates.append(dates[i+seq_len-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return(
                torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                self.dates[idx],
                torch.tensor(np.array(self.subset['returns']), dtype=torch.float32)
        )

def create_dataloaders(config):
    """
    데이터셋 및 데이터로더 생성 함수
    
    Args:
        config: 설정 객체 (필요한 속성: data_path, seq_len, batch_size, target_type, target_horizon, preprocessed)
        
    Returns:
        train_loader, valid_loader, test_loader: 데이터로더 튜플
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
                                start_date=None, end_date=None, shuffle=False, target_type="next_day",
                                target_horizon=1, num_workers=2, preprocessed=True):
    """
    날짜 범위 기반 데이터로더 생성 유틸리티 함수
    
    Args:
        data: 데이터프레임 (None인 경우 path에서 로드)
        path: 데이터 파일 경로 (data가 None인 경우)
        seq_len: 시퀀스 길이
        batch_size: 배치 크기
        start_date: 시작 날짜
        end_date: 종료 날짜
        shuffle: 데이터 셔플 여부
        target_type: 타겟 계산 방식
        target_horizon: 타겟 계산 기간
        num_workers: 데이터 로딩 워커 수
        preprocessed: 수익률 전처리 여부
        
    Returns:
        dataloader: 생성된 데이터로더
    """
    dataset = DateRangeRegimeMambaDataset(
        data=data,
        path=path,
        seq_len=seq_len,
        start_date=start_date,
        end_date=end_date,
        target_type=target_type,
        target_horizon=target_horizon,
        preprocessed=preprocessed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader
