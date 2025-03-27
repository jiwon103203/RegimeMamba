import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset, DataLoader
import torch

from ..utils.utils import set_seed
from ..models.mamba_model import create_model_from_config
from .clustering import identify_bull_bear_regimes, predict_regimes, extract_hidden_states
from .strategy import evaluate_regime_strategy, visualize_all_periods_performance
from ..data.dataset import RegimeMambaDataset, create_dataloaders, create_date_range_dataloader

# class RollingWindowConfig:
#     def __init__(self):
#         """롤링 윈도우 백테스트 설정 클래스"""
#         self.lookback_years = 10      # 클러스터링에 사용할 과거 데이터 기간(년)
#         self.forward_months = 12      # 적용할 미래 기간(개월)
#         self.start_date = '2010-01-01'  # 백테스트 시작일
#         self.end_date = '2023-12-31'    # 백테스트 종료일
#         self.n_clusters = 2           # 클러스터 수 (Bull/Bear)
#         self.cluster_method = 'cosine_kmeans'  # 클러스터링 방법
#         self.transaction_cost = 0.001 # 거래 비용 (0.1%)
#         self.model_path = None        # 사전 훈련된 모델 경로
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.target_type = 'average'
#         self.target_horizon = 5
#         self.preprocessed = False

#         # 모델 파라미터 (기본값)
#         self.d_model = 128
#         self.d_state = 128
#         self.n_layers = 4
#         self.dropout = 0.1
#         self.d_conv = 4
#         self.expand = 2
#         self.input_dim = 4
#         self.seq_len = 128
#         self.batch_size = 64

#         # 저장 경로
#         self.results_dir = './rolling_window_results'
#         os.makedirs(self.results_dir, exist_ok=True)

def load_pretrained_model(config):
    """
    사전 학습된 RegimeMamba 모델 로드

    Args:
        config: 설정 객체

    Returns:
        로드된 모델
    """
    from ..models.mamba_model import TimeSeriesMamba
    
    # 모델 초기화
    model = TimeSeriesMamba(
        input_dim=config.input_dim,
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        n_layers=config.n_layers,
        dropout=config.dropout
    )

    # 체크포인트 로드
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 평가 모드로 설정
    model.eval()
    model.to(config.device)

    print(f"모델 로드 완료: {config.model_path}")
    return model

def identify_regimes_for_period(model, data, config, period_start, period_end):
    """
    특정 기간에 대한 레짐 식별

    Args:
        model: 사전 학습된 모델
        data: 전체 데이터프레임
        config: 설정 객체
        period_start: 기간 시작일 (문자열)
        period_end: 기간 종료일 (문자열)

    Returns:
        kmeans: 훈련된 KMeans 모델
        bull_regime: Bull 레짐 클러스터 ID
    """
    print(f"기간 {period_start} ~ {period_end}에 대한 레짐 식별 중...")

    # 데이터셋 및 데이터로더 생성
    dataset = create_date_range_dataloader(
        data=data,
        seq_len=config.seq_len,
        start_date=period_start,
        end_date=period_end,
        target_type = config.target_type,
        target_horizon = config.target_horizon
    )

    # 데이터가 충분한지 확인
    if len(dataset) < 100:
        print(f"경고: 기간 {period_start} ~ {period_end}에 데이터가 부족합니다 ({len(dataset)} 샘플).")
        return None, None

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Hidden states 추출
    hidden_states, returns, dates = extract_hidden_states(model, dataloader, config)

    kmeans, bull_regime = identify_bull_bear_regimes(hidden_states, returns, config)

    return kmeans, bull_regime

def apply_regimes_to_future_period(model, data, config, kmeans, bull_regime, future_start, future_end):
    """
    식별된 레짐을 미래 기간에 적용

    Args:
        model: 사전 학습된 모델
        data: 전체 데이터프레임
        config: 설정 객체
        kmeans: 훈련된 KMeans 모델
        bull_regime: Bull 레짐 클러스터 ID
        future_start: 미래 기간 시작일 (문자열)
        future_end: 미래 기간 종료일 (문자열)

    Returns:
        results_df: 결과 데이터프레임
        performance: 성과 지표 딕셔너리
    """
    print(f"기간 {future_start} ~ {future_end}에 레짐 적용 중...")

    # 데이터셋 및 데이터로더 생성
    dataset = create_date_range_dataloader(
        data=data,
        seq_len=config.seq_len,
        start_date=future_start,
        end_date=future_end,
        target_type=config.target_type,
        target_horizon=config.target_horizon
    )

    # 데이터가 충분한지 확인
    if len(dataset) < 10:
        print(f"경고: 기간 {future_start} ~ {future_end}에 데이터가 부족합니다 ({len(dataset)} 샘플).")
        return None, None

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 레짐 예측
    predictions, true_returns, dates = predict_regimes(model, dataloader, kmeans, bull_regime, config)

    # 거래 비용을 고려한 성과 평가
    results_df, performance = evaluate_regime_strategy(
        predictions,
        true_returns,
        dates,
        transaction_cost=config.transaction_cost
    )

    return results_df, performance

def visualize_period_performance(results_df, title, save_path):
    """
    단일 기간에 대한 성과 시각화

    Args:
        results_df: 결과 데이터프레임
        title: 차트 제목
        save_path: 저장 경로
    """
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(results_df['Cum_Market'] * 100, label='Market', color='gray')
    plt.plot(results_df['Cum_Strategy'] * 100, label='Regime Strategy', color='blue')
    plt.title(f'{title} - Cumulative Returns')
    plt.legend()
    plt.ylabel('Return (%)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(results_df['Regime'], label='Regime (1=Bull, 0=Bear)', color='red')
    plt.title('Regime Signal')
    plt.ylabel('Regime')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_rolling_window_backtest(config, data_path):
    """
    롤링 윈도우 방식의 백테스트 실행

    Args:
        config: 설정 객체
        data_path: 데이터 파일 경로
        
    Returns:
        combined_results: 결합된 결과 데이터프레임
        period_performances: 기간별 성과 지표 리스트
    """
    # 데이터 로드
    data = pd.read_csv(data_path)

    # 사전 학습된 모델 로드
    model = load_pretrained_model(config)

    # 백테스트 결과 저장 객체
    all_results = []
    period_performances = []
    all_predictions = []

    # 시작 및 종료 날짜 파싱
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')

    period_counter = 1

    # 롤링 윈도우 백테스팅 메인 루프
    while current_date < end_date:
        print(f"\n=== 기간 {period_counter} 처리 중 ===")

        # 룩백 기간 계산 (현재 날짜에서 lookback_years년 전)
        lookback_start = (current_date - relativedelta(years=config.lookback_years)).strftime('%Y-%m-%d')
        lookback_end = current_date.strftime('%Y-%m-%d')

        # 미래 기간 계산 (현재 날짜에서 forward_months월 후)
        future_start = current_date.strftime('%Y-%m-%d')
        future_end = (current_date + relativedelta(months=config.forward_months)).strftime('%Y-%m-%d')

        print(f"룩백 기간: {lookback_start} ~ {lookback_end}")
        print(f"미래 기간: {future_start} ~ {future_end}")

        # 레짐 식별 (과거 데이터 기반)
        kmeans, bull_regime = identify_regimes_for_period(model, data, config, lookback_start, lookback_end)

        if kmeans is not None and bull_regime is not None:
            # 식별된 레짐을 미래 기간에 적용
            results_df, performance = apply_regimes_to_future_period(
                model, data, config, kmeans, bull_regime, future_start, future_end
            )

            if results_df is not None:
                # 기간 정보 추가
                results_df['lookback_start'] = lookback_start
                results_df['lookback_end'] = lookback_end
                results_df['future_start'] = future_start
                results_df['future_end'] = future_end
                results_df['period'] = period_counter

                # 결과 저장
                all_results.append(results_df)

                # 성과 정보에 기간 정보 추가
                performance['period'] = period_counter
                performance['lookback_start'] = lookback_start
                performance['lookback_end'] = lookback_end
                performance['future_start'] = future_start
                performance['future_end'] = future_end
                period_performances.append(performance)

                # 결과 파일 저장
                results_df.to_csv(
                    f"{config.results_dir}/period_{period_counter}_results.csv",
                    index=False
                )

                # 성과 시각화
                visualize_period_performance(
                    results_df,
                    f"기간 {period_counter}: {future_start} ~ {future_end}",
                    f"{config.results_dir}/period_{period_counter}_performance.png"
                )

        # 다음 기간으로 이동
        current_date += relativedelta(months=config.forward_months)
        period_counter += 1

    # 결과 병합 및 저장
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{config.results_dir}/all_periods_results.csv", index=False)

        # 전체 성과 저장
        with open(f"{config.results_dir}/all_periods_performance.json", 'w') as f:
            json.dump(period_performances, f, indent=4)

        # 전체 성과 시각화
        visualize_all_periods_performance(period_performances, config.results_dir)

        print(f"\n백테스트 완료! 총 {period_counter-1}개 기간 처리됨.")
        return combined_results, period_performances
    else:
        print("백테스트 실패: 유효한 결과가 없습니다.")
        return None, None
