'''
2005-01-01 ~ 오늘까지의 데이터를 가져오고
2005-01-01~2020-01-01 데이터를 학습 데이터
2020-01-01~2025-01-01 데이터를 검증 데이터
2025-01-01~2025-05-01 데이터를 테스트 데이터로 사용
각 데이터셋 분리 후 개별적으로 전처리 실행
EWM 계산을 위해 60일 추가 데이터 사용
Regime Mamba 훈련 진행하고 예측 진행
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset

from regime_mamba.config import RollingWindowTrainConfig
from regime_mamba.models.mamba_model import TimeSeriesMamba
from regime_mamba.train.train import train_with_early_stopping
from regime_mamba.models.jump_model import ModifiedJumpModel
from regime_mamba.utils.utils import set_seed

def flatten_multiindex(df):
    """
    MultiIndex 컬럼을 가진 데이터프레임을 단일 레벨 컬럼으로 변환
    
    Args:
        df: MultiIndex 컬럼을 가진 DataFrame
        
    Returns:
        단일 레벨 컬럼을 가진 DataFrame
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    
    print("Flattening MultiIndex columns")
    
    # 첫 번째 레벨만 추출하여 단일 레벨로 변환
    new_cols = []
    for col in df.columns:
        # 두 번째 레벨이 비어있거나 None인 경우
        if len(col) > 1 and (col[1] == '' or col[1] is None):
            new_cols.append(col[0])
        # 두 번째 레벨에 값이 있는 경우 (티커 정보 등)
        elif len(col) > 1:
            new_cols.append(f"{col[0]}")
        else:
            # 단일 레벨이거나 예상치 못한 구조인 경우
            new_cols.append(str(col))
    
    # 중복 컬럼명 처리
    seen = set()
    for i, col in enumerate(new_cols):
        if col in seen:
            # 중복된 컬럼명에 _N 접미사 추가
            counter = 1
            while f"{col}_{counter}" in seen:
                counter += 1
            new_cols[i] = f"{col}_{counter}"
        seen.add(new_cols[i])
    
    result_df = df.copy()
    result_df.columns = new_cols
    
    print(f"Columns after flattening: {result_df.columns.tolist()}")
    return result_df

def get_args():
    parser = argparse.ArgumentParser(description='Regime Mamba Overfitting Check')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--jump_model', type=bool, default=True, help='Use jump model')
    args = parser.parse_args()
    if args.config is None:
        raise ValueError("Configuration file path is required.")
    
    return args

def load_config(args):
    config = RollingWindowTrainConfig()
    with open(args.config, 'r') as file:
        yaml_config = yaml.safe_load(file)
    
    config.seed = args.seed

    for key, value in yaml_config.items():
        if hasattr(config, key):
            setattr(config, key, value)

    if torch.cuda.is_available():
        config.device = 'cuda'
    
    return config

def get_data():
    # 시퀀스 길이(200일) + EWM 계산을 위한 60일 추가 = 최소 260일 이전부터 데이터 가져오기
    index_ticker = '^GSPC'
    treasury_ticker = '^IRX'  # 미국 3개월 국채
    
    # 2005-01-01보다 260일 이전부터 데이터 가져오기 (시퀀스 + EWM 계산)
    start_date = '2004-04-15'  # 약 260일 이전
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Yahoo Finance에서 데이터 다운로드
    data = yf.download(index_ticker, start=start_date, end=end_date, auto_adjust=True)
    treasury_data = yf.download(treasury_ticker, start=start_date, end=end_date, auto_adjust=True)
    
    # 종가만 추출
    data = data[['Close']]
    data['returns'] = data['Close'].pct_change() * 100  # 수익률 계산
    data['target_20'] = data['Close'].shift(-20)  # 20일 후 종가

    treasury_data = treasury_data[['Close']].rename(columns={'Close':'Close_Treasury'})
    
    # 데이터프레임 병합하기 인덱스 컬럼(Date형식)를 기준으로 병합
    data = pd.merge(data, treasury_data, how='left', left_index=True, right_index=True)
    data = flatten_multiindex(data)
    print(f"Total data fetched: {len(data)} rows")
    
    return data

def split_and_process_data(data):
    # 데이터프레임 확인
    print(f"DataFrame structure before splitting:")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Index type: {type(data.index)}")
    
    # MultiIndex 확인 및 처리 - 공통 함수 사용
    data = flatten_multiindex(data)
    
    # 데이터 분할 (테스트 기간 확장: 약 1년 4개월로 설정)
    train_chunk = data[(data.index >= '2005-01-01') & (data.index < '2020-01-01')].copy()
    valid_chunk = data[(data.index >= '2020-01-01') & (data.index < '2024-01-01')].copy()  # 1년 줄임
    test_chunk = data[(data.index >= '2024-01-01') & (data.index < '2025-05-01')].copy()   # 약 16개월로 확장
    
    print(f"Train data: {len(train_chunk)} rows from {train_chunk.index.min()} to {train_chunk.index.max()}")
    print(f"Valid data: {len(valid_chunk)} rows from {valid_chunk.index.min()} to {valid_chunk.index.max()}")
    print(f"Test data: {len(test_chunk)} rows from {test_chunk.index.min()} to {test_chunk.index.max()}")
    
    # 각 데이터 청크의 열 확인 (모든 데이터셋이 동일한 구조를 가지는지)
    print(f"Train chunk columns: {train_chunk.columns.tolist()}")
    print(f"Valid chunk columns: {valid_chunk.columns.tolist()}")
    print(f"Test chunk columns: {test_chunk.columns.tolist()}")
    
    # 각 데이터셋 개별 전처리
    train_processed = preprocess_data(train_chunk, is_training=True)
    valid_processed = preprocess_data(valid_chunk, is_training=True)
    valid_processed = valid_processed.iloc[1:,:]
    test_processed = preprocess_data(test_chunk, is_training=False)  # 테스트 데이터는 target 필요 없음

    
    # 필요한 특성 컬럼 지정
    feature_col = ['dd_10', 'dd_20', 'dd_60', 'sortino_10', 'sortino_20', 'sortino_60']
    target_col = ['target_returns_20']
    
    # 필수 특성 및 타겟 열 존재 확인
    print("Checking feature columns:")
    for dataset, name in zip([train_processed, valid_processed, test_processed], ['Train', 'Valid', 'Test']):
        missing_cols = [col for col in feature_col if col not in dataset.columns]
        if missing_cols:
            print(f"WARNING: {name} is missing columns: {missing_cols}")
        else:
            print(f"{name} has all required feature columns")
    
    print(f"Processed train data: {len(train_processed)} rows")
    print(f"Processed valid data: {len(valid_processed)} rows")
    print(f"Processed test data: {len(test_processed)} rows")
    
    return train_processed, valid_processed, test_processed, feature_col, target_col

def preprocess_data(data_chunk, is_training=True):
    """
    각 데이터셋(train/valid/test)에 대해 개별적으로 전처리 수행
    is_training: True인 경우 target_returns_20을 계산, False인 경우 계산하지 않음 (테스트 데이터용)
    """
    # MultiIndex 확인 및 처리 - 공통 함수 사용
    data_chunk = flatten_multiindex(data_chunk)
    
    # 일별 수익률 특성 계산
    data_chunk['returns_10'] = data_chunk['returns'].ewm(halflife=10, min_periods=1, adjust=False).mean().shift(1)
    data_chunk['returns_20'] = data_chunk['returns'].ewm(halflife=20, min_periods=1, adjust=False).mean().shift(1)
    data_chunk['returns_60'] = data_chunk['returns'].ewm(halflife=60, min_periods=1, adjust=False).mean().shift(1)
    data_chunk['returns_120'] = data_chunk['returns'].ewm(halflife=120, min_periods=1, adjust=False).mean().shift(1)
    data_chunk['returns_200'] = data_chunk['returns'].ewm(halflife=200, min_periods=1, adjust=False).mean().shift(1)
    
    # 일별 무위험 수익률 계산
    data_chunk['daily_rf'] = (1 + data_chunk['Close_Treasury'] / 100) ** (1/252) - 1
    data_chunk['daily_rf_10'] = data_chunk['daily_rf'].ewm(halflife=10, min_periods=1, adjust=False).mean().shift(1)
    data_chunk['daily_rf_20'] = data_chunk['daily_rf'].ewm(halflife=20, min_periods=1, adjust=False).mean().shift(1)
    data_chunk['daily_rf_60'] = data_chunk['daily_rf'].ewm(halflife=60, min_periods=1, adjust=False).mean().shift(1)
    data_chunk['daily_rf_120'] = data_chunk['daily_rf'].ewm(halflife=120, min_periods=1, adjust=False).mean().shift(1)
    data_chunk['daily_rf_200'] = data_chunk['daily_rf'].ewm(halflife=200, min_periods=1, adjust=False).mean().shift(1)

    # 하방 편차 계산
    data_chunk['dd_10'] = np.sqrt(data_chunk['returns'].apply(lambda x: x**2 if x < 0 else 0).ewm(halflife=10, min_periods=1, adjust=False).mean().shift(1))
    data_chunk['dd_20'] = np.sqrt(data_chunk['returns'].apply(lambda x: x**2 if x < 0 else 0).ewm(halflife=20, min_periods=1, adjust=False).mean().shift(1))
    data_chunk['dd_60'] = np.sqrt(data_chunk['returns'].apply(lambda x: x**2 if x < 0 else 0).ewm(halflife=60, min_periods=1, adjust=False).mean().shift(1))
    data_chunk['dd_120'] = np.sqrt(data_chunk['returns'].apply(lambda x: x**2 if x < 0 else 0).ewm(halflife=120, min_periods=1, adjust=False).mean().shift(1))
    data_chunk['dd_200'] = np.sqrt(data_chunk['returns'].apply(lambda x: x**2 if x < 0 else 0).ewm(halflife=200, min_periods=1, adjust=False).mean().shift(1))

    # 소티노 비율 계산
    data_chunk['sortino_10'] = (data_chunk['returns_10'] - data_chunk['daily_rf_10']) / data_chunk['dd_10']
    data_chunk['sortino_20'] = (data_chunk['returns_20'] - data_chunk['daily_rf_20']) / data_chunk['dd_20']
    data_chunk['sortino_60'] = (data_chunk['returns_60'] - data_chunk['daily_rf_60']) / data_chunk['dd_60']
    data_chunk['sortino_120'] = (data_chunk['returns_120'] - data_chunk['daily_rf_120']) / data_chunk['dd_120']
    data_chunk['sortino_200'] = (data_chunk['returns_200'] - data_chunk['daily_rf_200']) / data_chunk['dd_200']

    # 학습 또는 검증 데이터인 경우에만 target_returns_20
    try:
        if is_training:
            # Shape 문제를 해결하기 위해 데이터 변환 및 타입 확인
            print(f"Close column type: {type(data_chunk['Close'])}")
            print(f"Close shape: {data_chunk['Close'].shape if hasattr(data_chunk['Close'], 'shape') else 'N/A'}")
            
            # 안전하게 값 추출
            close_values = data_chunk['Close'].values
            # 필요한 경우 차원 축소 (pandas Series의 경우 필요 없음)
            if hasattr(close_values, 'flatten') and len(close_values.shape) > 1:
                close_values = close_values.flatten()
                
            target_values = data_chunk['target_20'].values
            
            # 20일 후 수익률 계산
            data_chunk['target_returns_20'] = (target_values - close_values) / close_values
            
            # 결측값 처리 (훈련 데이터는 결측치 제거)
            data_chunk.dropna(inplace=True)
        else:
            # 테스트 데이터의 경우 타겟 값은 더미로 설정하고 결측치 제거하지 않음
            data_chunk['target_returns_20'] = 0.0  # np.nan 대신 0.0으로 설정
            
            # 테스트 데이터는 필수 특성 컬럼만 결측치 제거
            cols_to_check = ['dd_10', 'dd_20', 'dd_60', 'sortino_10', 'sortino_20', 'sortino_60']
            
            # 컬럼 존재 여부 확인 및 안전하게 결측치 제거
            existing_cols = [col for col in cols_to_check if col in data_chunk.columns]
            if len(existing_cols) != len(cols_to_check):
                print(f"Warning: Some columns not found. Available columns: {data_chunk.columns.tolist()}")
                print(f"Checking for NaN in existing columns: {existing_cols}")
            
            if existing_cols:
                data_chunk = data_chunk.dropna(subset=existing_cols)
            else:
                print("Warning: None of the required columns found for NaN check")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        print(f"Available columns: {data_chunk.columns.tolist()}")
        raise
    
    # 데이터 전처리 후 컬럼 확인 로깅
    print(f"Columns after preprocessing: {data_chunk.columns.tolist()}")
    
    return data_chunk

def create_sequences(data, feature_columns, seq_length=200):
    """
    각 데이터 포인트에 대해 이전 seq_length일의 시퀀스를 생성
    
    Args:
        data: pandas DataFrame - 시계열 데이터
        feature_columns: list - 사용할 특성 컬럼명 리스트
        seq_length: int - 시퀀스 길이 (기본값: 200)
        
    Returns:
        sequences: 3D numpy array - (n_samples, seq_length, n_features) 형태의 시퀀스 데이터
    """
    feature_data = data[feature_columns].values
    n_samples = len(data) - seq_length + 1
    n_features = len(feature_columns)
    
    sequences = np.zeros((n_samples, seq_length, n_features), dtype=np.float32)
    
    for i in range(n_samples):
        sequences[i] = feature_data[i:i+seq_length]
        
    return sequences

def create_dataloaders(train_data, valid_data, test_data, feature_col, target_col, config, seq_length=200):
    """
    시퀀스 데이터로 DataLoader 생성
    각 입력 데이터 포인트는 이전 seq_length일의 특성 시퀀스를 포함
    
    Args:
        train_data, valid_data, test_data: pandas DataFrame - 훈련/검증/테스트 데이터
        feature_col: list - 사용할 특성 컬럼명 리스트
        target_col: list - 타겟 컬럼명 리스트
        config: 설정 객체
        seq_length: int - 시퀀스 길이 (기본값: 200)
        
    Returns:
        train_loader, valid_loader, test_loader: DataLoader 객체
    """
    # 시퀀스 데이터 생성 (처음 seq_length-1일은 충분한 이전 데이터가 없어 사용 불가)
    print(f"Creating sequence data with sequence length: {seq_length}")
    
    # 학습 데이터 시퀀스 생성 (seq_length일 이후부터)
    train_sequences = create_sequences(train_data, feature_col, seq_length)
    # 타겟 데이터는 시퀀스의 마지막 시점에 해당하는 값만 사용
    train_targets = train_data[target_col].values[seq_length-1:]
    train_returns = train_data['returns'].values[seq_length-1:]
    
    print(f"Train sequences shape: {train_sequences.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    
    # 검증 데이터 시퀀스 생성
    valid_sequences = create_sequences(valid_data, feature_col, seq_length)
    valid_targets = valid_data[target_col].values[seq_length-1:]
    valid_returns = valid_data['returns'].values[seq_length-1:]
    
    print(f"Validation sequences shape: {valid_sequences.shape}")
    
    # 테스트 데이터 시퀀스 생성
    if len(test_data) >= seq_length:
        test_sequences = create_sequences(test_data, feature_col, seq_length)
        # 테스트 데이터는 타겟 없이 더미 값 사용
        dummy_test_targets = np.zeros((len(test_sequences), 1), dtype=np.float32)
        test_returns = test_data['returns'].values[seq_length-1:]
        
        print(f"Test sequences shape: {test_sequences.shape}")
    else:
        print(f"Warning: Test data length ({len(test_data)}) is less than sequence length ({seq_length})")
        # 테스트 데이터가 부족한 경우 더미 데이터로 대체
        # 빈 배열 + 더미 타겟 생성
        test_sequences = np.zeros((1, seq_length, len(feature_col)), dtype=np.float32)
        dummy_test_targets = np.zeros((1, 1), dtype=np.float32)
        test_returns = np.zeros(1, dtype=np.float32)
    
    # 텐서 변환 (float32 타입으로 명시)
    X_train = torch.tensor(train_sequences, dtype=torch.float32)
    y_train = torch.tensor(train_targets, dtype=torch.float32)
    returns_train = torch.tensor(train_returns, dtype=torch.float32)
    
    X_valid = torch.tensor(valid_sequences, dtype=torch.float32)
    y_valid = torch.tensor(valid_targets, dtype=torch.float32)
    returns_valid = torch.tensor(valid_returns, dtype=torch.float32)
    
    X_test = torch.tensor(test_sequences, dtype=torch.float32)
    dummy_y_test = torch.tensor(dummy_test_targets, dtype=torch.float32)
    returns_test = torch.tensor(test_returns, dtype=torch.float32)
    
    # DataLoader 생성 - 각 배치는 (x, y, returns, returns) 형태로 반환
    train_loader = DataLoader(
        [(x, y, r, r) for x, y, r in zip(X_train, y_train, returns_train)],
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    valid_loader = DataLoader(
        [(x, y, r, r) for x, y, r in zip(X_valid, y_valid, returns_valid)],
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        [(x, y, r, r) for x, y, r in zip(X_test, dummy_y_test, returns_test)],
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    return train_loader, valid_loader, test_loader

def train_and_predict(config):
    set_seed(config.seed)
    
    # 1. 데이터 가져오기
    data = get_data()
    
    # 2. 데이터 분할 및 전처리
    train_data, valid_data, test_data, feature_col, target_col = split_and_process_data(data)
    
    # 시퀀스 길이 설정 (config에서 가져오거나 기본값 200 사용)
    seq_length = getattr(config, 'seq_len', 200)
    print(f"Using sequence length: {seq_length}")
    
    # 3. 시퀀스 데이터로 DataLoader 생성
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_data, valid_data, test_data, feature_col, target_col, config, seq_length=seq_length
    )
    
    # 4. 모델 초기화 (input_dim은 feature_col 차원과 일치해야 함)
    model = TimeSeriesMamba(
        input_dim=len(feature_col),  # 특성 차원 수
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        n_layers=config.n_layers,
        dropout=config.dropout,
        config=config
    ).to(config.device)
    
    print(f"Model initialized with input dimension: {len(feature_col)}")
    
    # 5. 모델 훈련
    print("Starting model training...")
    _, _, model = train_with_early_stopping(
        model, train_loader, valid_loader, config, use_onecycle=config.use_onecycle
    )
    
    # 6. Jump 모델 훈련 및 예측
    print("Initializing Jump model...")
    jump_model = ModifiedJumpModel(config=config)
    jump_model.feature_extractor = model
    
    print("Training Jump model on window 2005-01-01 to 2020-01-01...")
    jump_model.train_for_window('2005-01-01', '2020-01-01', train_data, 4)
    
    # 테스트 기간에 대한 예측 (수정된 테스트 기간으로 변경)
    print("Making predictions for test period 2024-01-01 to 2025-05-01...")
    jump_model.predict('2024-01-01', '2025-05-01', test_data, window_number=1)
    
    print("Training and prediction completed successfully.")
    return

if __name__ == "__main__":
    args = get_args()
    config = load_config(args)
    train_and_predict(config)