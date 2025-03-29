import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class RegimeMambaDataset(Dataset):
    def __init__(self, path, seq_len=128, mode="train", target_type="next_day", target_horizon=1, preprocessed = True):
        """
        RegimeMamba 모델을 위한 데이터셋 클래스

        Args:
            path: 데이터 파일 경로
            seq_len: 시퀀스 길이
            mode: 'train', 'valid', 'test' 중 하나
            target_type: 타겟 유형
                - next_day: 다음 날의 수익률
                - average: 지정된 기간 동안의 평균 수익률
                - cumulative: 지정된 기간 동안의 누적 수익률
                - trend_strength: 선형 회귀로 측정한 추세 강도
                - direction: 기간 동안의 방향성 (분류 문제로 변환)
                - volatility_adjusted: 변동성 조정 수익률 (샤프 비율과 유사)
                - up_ratio: 기간 중 상승한 날의 비율
                - log_return_sum: 로그 수익률의 합계
            target_horizon: 타겟 계산을 위한 기간 (일)
            preprocessed: 수익률 전처리 여부
        """
        super().__init__()
        self.data = pd.read_csv(path)
        self.preprocessed = preprocessed
        if self.preprocessed:
            self.data['returns'] = self.data['returns'] * 100
        else:
            self.data['Close'] = self.data['Close'] / 100 # 스케일 조정

        self.data["dd_10"] = self.data["dd_10"] * 100
        self.data["sortino_20"] = self.data["sortino_20"] * 100
        self.data["sortino_60"] = self.data["sortino_60"] * 100
        self.seq_len = seq_len
        self.target_type = target_type
        self.target_horizon = target_horizon

        # 특성과 타겟 칼럼 지정
        if self.preprocessed:
            self.feature_cols = ["returns", "dd_10", "sortino_20", "sortino_60"]
            self.target_col = "returns"  # 수익률을 타겟으로 사용
        else: 
            self.feature_cols = ["Close", "dd_10", "sortino_20", "sortino_60"]
            self.target_col = "Close" 
        

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
        self.returns = []  # 수익률 정보도 저장

        features = np.array(self.subset[self.feature_cols])
        dates = np.array(self.subset[date_col])

        if target_type == "average" and (f"target_SMA_{target_horizon}" in self.data.columns or f"target_returns_{target_horizon}" in self.data.columns):
            
            print("미리 처리된 데이터 포착")
            
            if self.preprocessed:
                self.target_col=f"target_returns_{target_horizon}"
                targets = np.array(self.subset[self.target_col]/self.target_horizon)
            else:
                self.target_col=f"target_SMA_{target_horizon}"
                targets = np.array(self.subset[self.target_col]/ 100)

            for i in range(len(features) - seq_len+1):
                self.sequences.append(features[i:i+seq_len])
                self.targets.append(targets[i+seq_len-1])
                # 타겟 날짜 저장 (타겟 기간의 마지막 날짜)
                self.dates.append(dates[i+seq_len-1])
                self.returns.append(self.subset['returns'][i+seq_len])

        else:
            # target_horizon을 고려한 인덱스 범위 조정
            for i in range(len(features) - seq_len - target_horizon + 1):
                self.sequences.append(features[i:i+seq_len])
                
                # 타겟 기간의 수익률 데이터 추출
                target_returns = [features[i+seq_len+j][0] for j in range(target_horizon)]
                if preprocessed:
                    decimal_returns = [r/100 for r in target_returns]  # 백분율 -> 소수
                else: # 수익률이 아니라 종가일 경우
                    raise ValueError("수익률 전처리가 필요합니다.")
    
                # 타겟 타입에 따른 계산 방식 선택
                if target_type == "next_day":
                    # 다음 날의 수익률
                    self.targets.append([features[i+seq_len][0]])
                    
                elif target_type == "average":
                    # 지정된 기간 동안의 평균 수익률
                    self.targets.append([np.mean(target_returns)])
                    
                elif target_type == "cumulative":
                    # 지정된 기간 동안의 누적 수익률
                    cumulative_return = np.prod([1 + r for r in decimal_returns]) - 1
                    self.targets.append([cumulative_return * 100])  # 소수 -> 백분율
                    
                elif target_type == "trend_strength":
                    # 선형 회귀로 측정한 추세 강도 (기울기)
                    if target_horizon >= 2:  # 최소 2일 이상 필요
                        x = np.arange(target_horizon)
                        slope, _ = np.polyfit(x, target_returns, 1)
                        self.targets.append([slope])
                    else:
                        self.targets.append([target_returns[0]])
                        
                elif target_type == "direction":
                    # 기간의 누적 수익률의 방향 (양수:1, 음수:-1, 제로:0)
                    cumulative_return = np.prod([1 + r for r in decimal_returns]) - 1
                    direction = np.sign(cumulative_return)
                    self.targets.append([float(direction)])
                    
                elif target_type == "volatility_adjusted":
                    # 변동성으로 조정된 수익률 (샤프 비율과 유사)
                    mean_return = np.mean(target_returns)
                    std_return = np.std(target_returns) if np.std(target_returns) > 0 else 1.0
                    sharpe_like = mean_return / std_return
                    self.targets.append([sharpe_like])
                    
                elif target_type == "up_ratio":
                    # 상승한 날의 비율
                    up_days = sum(1 for r in target_returns if r > 0)
                    up_ratio = up_days / len(target_returns) if target_returns else 0
                    self.targets.append([up_ratio * 100])  # 백분율로 변환
                    
                elif target_type == "log_return_sum":
                    # 로그 수익률의 합계 (복리 효과를 더 잘 반영)
                    log_returns = [np.log(1 + r) for r in decimal_returns]
                    log_return_sum = np.sum(log_returns) * 100  # 백분율로 변환
                    self.targets.append([log_return_sum])
                    
                # 타겟 날짜 저장 (타겟 기간의 마지막 날짜)
                self.dates.append(dates[i+seq_len+target_horizon-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.preprocessed:
            return (
                torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                self.dates[idx]
            )
        else:
            return(
                torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                self.dates[idx],
                torch.tensor(np.array(self.returns), dtype=torch.float32)
            )

class DateRangeRegimeMambaDataset(Dataset):
    def __init__(self, data=None, path=None, seq_len=128, start_date=None, end_date=None, 
                 target_type="next_day", target_horizon=1, preprocessed=True):
        """
        날짜 범위를 기반으로 데이터를 필터링하는 데이터셋 클래스

        Args:
            data: 전체 데이터프레임 (None인 경우 path에서 로드)
            path: 데이터 파일 경로 (data가 None인 경우 사용)
            seq_len: 시퀀스 길이
            start_date: 시작 날짜 (문자열, 'YYYY-MM-DD' 형식)
            end_date: 종료 날짜 (문자열, 'YYYY-MM-DD' 형식)
            target_type: 타겟 유형
                - next_day: 다음 날의 수익률
                - average: 지정된 기간 동안의 평균 수익률
                - cumulative: 지정된 기간 동안의 누적 수익률
                - trend_strength: 선형 회귀로 측정한 추세 강도
                - direction: 기간 동안의 방향성 (분류 문제로 변환)
                - volatility_adjusted: 변동성 조정 수익률 (샤프 비율과 유사)
                - max_drawdown: 기간 내 최대 낙폭
                - up_ratio: 기간 중 상승한 날의 비율
                - log_return_sum: 로그 수익률의 합계
            target_horizon: 타겟 계산을 위한 기간 (일)
            preprocessed: 수익률 전처리 여부
        """
        super().__init__()
        self.seq_len = seq_len
        self.target_type = target_type
        self.target_horizon = target_horizon
        self.preprocessed = preprocessed
        # 데이터 로드
        if data is None and path is not None:
            data = pd.read_csv(path)
            if self.preprocessed:
                data['returns'] = data['returns'] * 100
            else:
                data['Close'] = data['Close'] / 100

            data["dd_10"] = data["dd_10"] * 100
            data["sortino_20"] = data["sortino_20"] * 100
            data["sortino_60"] = data["sortino_60"] * 100

        # 데이터가 제공되지 않은 경우 에러
        if data is None:
            raise ValueError("Either data or path must be provided")

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

        # 특성과 타겟 칼럼 지정
        if self.preprocessed:
            self.feature_cols = ["returns", "dd_10", "sortino_20", "sortino_60"]
            self.target_col = "returns"
        else:
            self.feature_cols = ["Close", "dd_10", "sortino_20", "sortino_60"]
            self.target_col = "Close"

        # 시퀀스 및 타겟 생성
        self.sequences = []
        self.targets = []
        self.dates = []

        features = np.array(self.data[self.feature_cols])
        dates = np.array(self.data[date_col])

        if target_type == "average" and (f"target_SMA_{target_horizon}" in self.data.columns or f"target_returns_{target_horizon}" in self.data.columns):
            
            print("미리 처리된 데이터 포착")
            
            if self.preprocessed:
                self.target_col=f"target_returns_{target_horizon}"
                targets = np.array(self.data[self.target_col])/self.target_horizon
            else:
                self.target_col=f"target_SMA_{target_horizon}"
                targets = np.array(self.data[self.target_col])

            for i in range(len(features) - seq_len+1):
                self.sequences.append(features[i:i+seq_len])
                self.targets.append(targets[i+seq_len-1])
                # 타겟 날짜 저장 (타겟 기간의 마지막 날짜)
                self.dates.append(dates[i+seq_len-1])

        else:        
            # target_horizon을 고려한 인덱스 범위 조정
            for i in range(len(features) - seq_len - target_horizon + 1):
                self.sequences.append(features[i:i+seq_len])
                
                # 타겟 기간의 수익률 데이터 추출
                target_returns = [features[i+seq_len+j][0] for j in range(target_horizon)]
                decimal_returns = [r/100 for r in target_returns]  # 백분율 -> 소수
    
                # 타겟 타입에 따른 계산 방식 선택
                if target_type == "next_day":
                    # 다음 날의 수익률
                    self.targets.append([features[i+seq_len][0]])
                    
                elif target_type == "average":
                    # 지정된 기간 동안의 평균 수익률
                    self.targets.append([np.mean(target_returns)])
                    
                elif target_type == "cumulative":
                    # 지정된 기간 동안의 누적 수익률
                    cumulative_return = np.prod([1 + r for r in decimal_returns]) - 1
                    self.targets.append([cumulative_return * 100])  # 소수 -> 백분율
                    
                elif target_type == "trend_strength":
                    # 선형 회귀로 측정한 추세 강도 (기울기)
                    if target_horizon >= 2:  # 최소 2일 이상 필요
                        x = np.arange(target_horizon)
                        slope, _ = np.polyfit(x, target_returns, 1)
                        self.targets.append([slope])
                    else:
                        self.targets.append([target_returns[0]])
                        
                elif target_type == "direction":
                    # 기간의 누적 수익률의 방향 (양수:1, 음수:-1, 제로:0)
                    cumulative_return = np.prod([1 + r for r in decimal_returns]) - 1
                    direction = np.sign(cumulative_return)
                    self.targets.append([float(direction)])
                    
                elif target_type == "volatility_adjusted":
                    # 변동성으로 조정된 수익률 (샤프 비율과 유사)
                    mean_return = np.mean(target_returns)
                    std_return = np.std(target_returns) if np.std(target_returns) > 0 else 1.0
                    sharpe_like = mean_return / std_return
                    self.targets.append([sharpe_like])
                    
                elif target_type == "up_ratio":
                    # 상승한 날의 비율
                    up_days = sum(1 for r in target_returns if r > 0)
                    up_ratio = up_days / len(target_returns) if target_returns else 0
                    self.targets.append([up_ratio * 100])  # 백분율로 변환
                    
                elif target_type == "log_return_sum":
                    # 로그 수익률의 합계 (복리 효과를 더 잘 반영)
                    log_returns = [np.log(1 + r) for r in decimal_returns]
                    log_return_sum = np.sum(log_returns) * 100  # 백분율로 변환
                    self.targets.append([log_return_sum])
                
                # 타겟 날짜 저장 (타겟 기간의 마지막 날짜)
                self.dates.append(dates[i+seq_len+target_horizon-1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.preprocessed:
            return (
                torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                self.dates[idx]
            )
        else:
            return(
                torch.tensor(self.sequences[idx], dtype=torch.float32),
                torch.tensor(self.targets[idx], dtype=torch.float32),
                self.dates[idx],
                torch.tensor(np.array(self.data['returns']), dtype=torch.float32)
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
        path=config.data_path, 
        seq_len=config.seq_len, 
        mode="train",
        target_type=config.target_type,
        target_horizon=config.target_horizon,
        preprocessed=config.preprocessed
    )
    
    valid_dataset = RegimeMambaDataset(
        path=config.data_path, 
        seq_len=config.seq_len, 
        mode="valid",
        target_type=config.target_type,
        target_horizon=config.target_horizon,
        preprocessed=config.preprocessed
    )
    
    test_dataset = RegimeMambaDataset(
        path=config.data_path, 
        seq_len=config.seq_len, 
        mode="test",
        target_type=config.target_type,
        target_horizon=config.target_horizon,
        preprocessed=config.preprocessed
    )
    
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

def create_date_range_dataloader(data=None, path=None, seq_len=128, batch_size=64, 
                                start_date=None, end_date=None, shuffle=False, target_type="next_day",
                                target_horizon=1, num_workers=4, preprocessed=True):
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
