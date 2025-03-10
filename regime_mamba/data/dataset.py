import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class RegimeMambaDataset(Dataset):
    def __init__(self, path, seq_len=128, mode="train", target_type="next_day", target_horizon=1):
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
                - max_drawdown: 기간 내 최대 낙폭
                - up_ratio: 기간 중 상승한 날의 비율
                - log_return_sum: 로그 수익률의 합계
            target_horizon: 타겟 계산을 위한 기간 (일)
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
        self.target_type = target_type
        self.target_horizon = target_horizon

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
                
            elif target_type == "max_drawdown":
                # 기간 내 최대 낙폭
                # 누적 곱으로 가격 시뮬레이션
                price_curve = np.cumprod([1 + r for r in decimal_returns])
                max_dd = 0
                peak = price_curve[0]
                
                for price in price_curve:
                    if price > peak:
                        peak = price
                    drawdown = (peak - price) / peak
                    max_dd = max(max_dd, drawdown)
                
                self.targets.append([max_dd * 100])  # 백분율로 변환
                
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
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            self.dates[idx]
        )

def create_dataloaders(config):
    """
    데이터셋 및 데이터로더 생성 함수
    
    Args:
        config: 설정 객체 (필요한 속성: data_path, seq_len, batch_size, target_type, target_horizon)
        
    Returns:
        train_loader, valid_loader, test_loader: 데이터로더 튜플
    """
    train_dataset = RegimeMambaDataset(
        config.data_path, 
        seq_len=config.seq_len, 
        mode="train",
        target_type=config.target_type,
        target_horizon=config.target_horizon
    )
    
    valid_dataset = RegimeMambaDataset(
        config.data_path, 
        seq_len=config.seq_len, 
        mode="valid",
        target_type=config.target_type,
        target_horizon=config.target_horizon
    )
    
    test_dataset = RegimeMambaDataset(
        config.data_path, 
        seq_len=config.seq_len, 
        mode="test",
        target_type=config.target_type,
        target_horizon=config.target_horizon
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
