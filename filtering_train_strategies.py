import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import copy
import argparse
from collections import defaultdict

from regime_mamba.utils.utils import set_seed
from regime_mamba.evaluate.rolling_window_w_train import (
    RollingWindowTrainConfig, 
    DateRangeRegimeMambaDataset,
    train_model_for_window, 
    identify_regimes_for_window
)
from regime_mamba.evaluate.smoothing import (
    apply_regime_smoothing,
    apply_confirmation_rule,
    apply_minimum_holding_period
)
from regime_mamba.evaluate.strategy import evaluate_regime_strategy

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description='다양한 Smoothing 기법 비교 (롤링 윈도우 학습)')
    
    parser.add_argument('--data_path', type=str, required=True, help='데이터 파일 경로')
    parser.add_argument('--results_dir', type=str, default='./smoothing_train_results', help='결과 저장 디렉토리')
    parser.add_argument('--start_date', type=str, default='1990-01-01', help='시작 날짜')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='종료 날짜')
    
    # 기간 관련 설정
    parser.add_argument('--total_window_years', type=int, default=40, help='총 사용할 데이터 기간(년)')
    parser.add_argument('--train_years', type=int, default=20, help='학습에 사용할 기간(년)')
    parser.add_argument('--valid_years', type=int, default=10, help='검증에 사용할 기간(년)')
    parser.add_argument('--clustering_years', type=int, default=10, help='클러스터링에 사용할 기간(년)')
    parser.add_argument('--forward_months', type=int, default=60, help='다음 윈도우까지의 간격(개월)')
    
    # 모델 파라미터
    parser.add_argument('--d_model', type=int, default=128, help='모델 차원')
    parser.add_argument('--d_state', type=int, default=128, help='상태 차원')
    parser.add_argument('--n_layers', type=int, default=4, help='레이어 수')
    parser.add_argument('--dropout', type=float, default=0.1, help='드롭아웃 비율')
    parser.add_argument('--seq_len', type=int, default=128, help='시퀀스 길이')
    parser.add_argument('--batch_size', type=int, default=64, help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='학습률')
    
    # 학습 관련 설정
    parser.add_argument('--max_epochs', type=int, default=100, help='최대 학습 에폭')
    parser.add_argument('--patience', type=int, default=10, help='조기 종료 인내심')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='거래 비용 (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    return parser.parse_args()

def apply_and_evaluate_with_smoothing(model, data, kmeans, bull_regime, forward_start, forward_end, 
                                    smoothing_method, device, batch_size, seq_len, transaction_cost, 
                                    smoothing_params):
    """
    특정 smoothing 기법을 적용하여 레짐 전략 평가
    
    Args:
        model: 학습된 모델
        data: 전체 데이터프레임
        kmeans: K-Means 모델
        bull_regime: Bull 레짐 클러스터 ID
        forward_start: 미래 기간 시작일
        forward_end: 미래 기간 종료일
        smoothing_method: smoothing 기법 이름
        device: 연산 장치
        batch_size: 배치 크기
        seq_len: 시퀀스 길이
        transaction_cost: 거래 비용
        smoothing_params: smoothing 기법 파라미터
        
    Returns:
        results_df: 결과 데이터프레임
        performance: 성과 지표 딕셔너리
    """
    # 미래 데이터셋 생성
    forward_dataset = DateRangeRegimeMambaDataset(
        data=data, 
        seq_len=seq_len,
        start_date=forward_start,
        end_date=forward_end
    )
    
    # 데이터 로더 생성
    forward_loader = DataLoader(
        forward_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 데이터가 충분한지 확인
    if len(forward_dataset) < 10:
        print(f"경고: 평가를 위한 데이터가 부족합니다. ({len(forward_dataset)} 샘플)")
        return None, None
    
    # 원본 레짐 예측
    from regime_mamba.evaluate.clustering import predict_regimes
    raw_predictions, true_returns, dates = predict_regimes(model, forward_loader, kmeans, bull_regime, device)
    
    # Smoothing 적용
    if smoothing_method == 'none':
        smoothed_predictions = raw_predictions
    elif smoothing_method == 'ma':
        window = smoothing_params.get('window', 10)
        smoothed_predictions = apply_regime_smoothing(
            raw_predictions, method='ma', window=window
        ).reshape(-1, 1)
    elif smoothing_method == 'exp':
        window = smoothing_params.get('window', 10)
        smoothed_predictions = apply_regime_smoothing(
            raw_predictions, method='exp', window=window
        ).reshape(-1, 1)
    elif smoothing_method == 'gaussian':
        window = smoothing_params.get('window', 10)
        smoothed_predictions = apply_regime_smoothing(
            raw_predictions, method='gaussian', window=window
        ).reshape(-1, 1)
    elif smoothing_method == 'confirmation':
        days = smoothing_params.get('days', 3)
        smoothed_predictions = apply_confirmation_rule(
            raw_predictions, confirmation_days=days
        ).reshape(-1, 1)
    elif smoothing_method == 'min_holding':
        days = smoothing_params.get('days', 20)
        smoothed_predictions = apply_minimum_holding_period(
            raw_predictions, min_holding_days=days
        ).reshape(-1, 1)
    else:
        print(f"알 수 없는 smoothing 기법: {smoothing_method}, 원본 예측 사용")
        smoothed_predictions = raw_predictions
    
    # 거래 비용을 고려한 전략 평가
    results_df, performance = evaluate_regime_strategy(
        smoothed_predictions,
        true_returns,
        dates,
        transaction_cost=transaction_cost
    )
    
    # 결과에 smoothing 정보 추가
    if results_df is not None:
        results_df['smoothing_method'] = smoothing_method
        for param_name, param_value in smoothing_params.items():
            results_df[f'smoothing_{param_name}'] = param_value
        
        # 원본 레짐 정보 추가
        results_df['raw_regime'] = raw_predictions.flatten()
        
        # 원본과 smoothing 후 거래 횟수 계산
        raw_trades = (np.diff(raw_predictions.flatten()) != 0).sum() + (raw_predictions[0] == 1)
        smoothed_trades = (np.diff(smoothed_predictions.flatten()) != 0).sum() + (smoothed_predictions[0] == 1)
        
        # 성과 정보에 거래 감소 정보 추가
        performance['smoothing_method'] = smoothing_method
        for param_name, param_value in smoothing_params.items():
            performance[f'smoothing_{param_name}'] = param_value
        performance['raw_trades'] = int(raw_trades)
        performance['smoothed_trades'] = int(smoothed_trades)
        performance['trade_reduction'] = int(raw_trades - smoothed_trades)
        if raw_trades > 0:
            performance['trade_reduction_pct'] = ((raw_trades - smoothed_trades) / raw_trades) * 100
        else:
            performance['trade_reduction_pct'] = 0
        
    return results_df, performance

def visualize_comparison(all_methods_results, window_number, title, save_path):
    """
    다양한 smoothing 기법의 결과 비교 시각화
    
    Args:
        all_methods_results: 모든 기법의 결과 딕셔너리
        window_number: 윈도우 번호
        title: 차트 제목
        save_path: 저장 경로
    """
    plt.figure(figsize=(15, 10))
    
    # 누적 수익률 비교
    plt.subplot(2, 1, 1)
    
    # 시장 수익률은 한 번만 그림
    first_method = list(all_methods_results.keys())[0]
    plt.plot(all_methods_results[first_method]['df']['Cum_Market'] * 100, 
            label='시장', color='gray', linestyle='--')
    
    # 각 smoothing 기법의 수익률 표시
    for method_name, result in all_methods_results.items():
        plt.plot(result['df']['Cum_Strategy'] * 100, label=method_name)
    
    plt.title(f'{title}')
    plt.legend()
    plt.ylabel('수익률 (%)')
    plt.grid(True)
    
    # 거래 횟수 및 수익률 요약
    plt.subplot(2, 1, 2)
    
    methods = list(all_methods_results.keys())
    returns = [all_methods_results[method]['cum_return'] for method in methods]
    trades = [all_methods_results[method]['n_trades'] for method in methods]
    
    # 두 개의 y축 생성
    fig = plt.gca()
    ax1 = fig.axes
    ax2 = ax1.twinx()
    
    # 첫 번째 y축: 수익률
    bars1 = ax1.bar(np.arange(len(methods)) - 0.2, returns, width=0.4, color='blue', alpha=0.7, label='수익률 (%)')
    ax1.set_ylabel('수익률 (%)', color='blue')
    ax1.tick_params(axis='y', colors='blue')
    
    # 두 번째 y축: 거래 횟수
    bars2 = ax2.bar(np.arange(len(methods)) + 0.2, trades, width=0.4, color='red', alpha=0.7, label='거래 횟수')
    ax2.set_ylabel('거래 횟수', color='red')
    ax2.tick_params(axis='y', colors='red')
    
    plt.xticks(np.arange(len(methods)), methods, rotation=45)
    plt.title('Smoothing 기법별 수익률 및 거래 횟수')
    plt.grid(True, alpha=0.3)
    
    # 범례 추가
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=4, alpha=0.7),
        Line2D([0], [0], color='red', lw=4, alpha=0.7)
    ]
    plt.legend(custom_lines, ['수익률 (%)', '거래 횟수'])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_all_smoothing_methods(window_results_dir, model, data, kmeans, bull_regime, 
                                 forward_start, forward_end, window_number, config):
    """
    다양한 smoothing 기법을 평가하고 결과 비교
    
    Args:
        window_results_dir: 윈도우 결과 저장 디렉토리
        model: 학습된 모델
        data: 전체 데이터프레임
        kmeans: K-Means 모델
        bull_regime: Bull 레짐 클러스터 ID
        forward_start: 미래 기간 시작일
        forward_end: 미래 기간 종료일
        window_number: 윈도우 번호
        config: 설정 객체
        
    Returns:
        all_methods_results: 모든 기법의 결과 딕셔너리
        all_methods_performances: 모든 기법의 성과 리스트
    """
    # 테스트할 smoothing 기법 정의
    smoothing_methods = [
        ('none', {}),
        ('ma', {'window': 5}),
        ('ma', {'window': 10}),
        ('ma', {'window': 20}),
        ('exp', {'window': 10}),
        ('gaussian', {'window': 10}),
        ('confirmation', {'days': 3}),
        ('confirmation', {'days': 5}),
        ('min_holding', {'days': 10}),
        ('min_holding', {'days': 20}),
        ('min_holding', {'days': 30})
    ]
    
    # 결과 저장 객체
    all_methods_results = {}
    all_methods_performances = []
    
    # 각 smoothing 기법 평가
    for method_name, params in smoothing_methods:
        # 메서드 및 파라미터를 문자열로 변환하여 식별자 생성
        param_str = '_'.join([f"{k}={v}" for k, v in params.items()]) if params else "default"
        method_id = f"{method_name}_{param_str}" if params else method_name
        
        print(f"\n--- 평가 중: {method_id} ---")
        
        results_df, performance = apply_and_evaluate_with_smoothing(
            model, data, kmeans, bull_regime, forward_start, forward_end, 
            method_name, config.device, config.batch_size, config.seq_len, 
            config.transaction_cost, params
        )
        
        if results_df is not None and performance is not None:
            # 결과 저장
            method_result = {
                'df': results_df,
                'performance': performance,
                'cum_return': performance['cumulative_returns']['strategy'],
                'n_trades': performance['trading_metrics']['number_of_trades']
            }
            all_methods_results[method_id] = method_result
            all_methods_performances.append(performance)
            
            # 개별 결과 저장
            method_results_dir = os.path.join(window_results_dir, method_id)
            os.makedirs(method_results_dir, exist_ok=True)
            results_df.to_csv(os.path.join(method_results_dir, 'results.csv'), index=False)
            
            with open(os.path.join(method_results_dir, 'performance.json'), 'w') as f:
                json.dump(performance, f, indent=4)
    
    # 모든 smoothing 기법 비교 시각화
    if all_methods_results:
        visualize_comparison(
            all_methods_results,
            window_number,
            f"윈도우 {window_number}: {forward_start} ~ {forward_end} - Smoothing 기법 비교",
            os.path.join(window_results_dir, 'all_methods_comparison.png')
        )
    
    return all_methods_results, all_methods_performances

def visualize_final_comparison(combined_results, save_dir):
    """
    모든 윈도우에 걸친 smoothing 기법 비교 결과 시각화
    
    Args:
        combined_results: 결합된 결과 딕셔너리
        save_dir: 저장 디렉토리
    """
    # 모든 smoothing 기법과 윈도우 식별
    all_methods = sorted(list(combined_results.keys()))
    all_windows = sorted(list(set(combined_results[all_methods[0]]['window'])))
    
    # 1. 기법별 평균 성과 비교
    method_avg_returns = []
    method_avg_trades = []
    method_avg_sharpes = []
    
    for method in all_methods:
        method_returns = [combined_results[method]['returns'][window] for window in all_windows]
        method_trades = [combined_results[method]['trades'][window] for window in all_windows]
        method_sharpes = [combined_results[method]['sharpes'][window] for window in all_windows]
        
        method_avg_returns.append(np.mean(method_returns))
        method_avg_trades.append(np.mean(method_trades))
        method_avg_sharpes.append(np.mean(method_sharpes))
    
    # 성과별로 정렬
    sorted_indices = np.argsort(method_avg_returns)[::-1]  # 수익률 기준 내림차순
    sorted_methods = [all_methods[i] for i in sorted_indices]
    sorted_returns = [method_avg_returns[i] for i in sorted_indices]
    sorted_trades = [method_avg_trades[i] for i in sorted_indices]
    sorted_sharpes = [method_avg_sharpes[i] for i in sorted_indices]
    
    # 종합 결과 시각화
    plt.figure(figsize=(15, 12))
    
    # 평균 수익률 비교
    plt.subplot(2, 2, 1)
    plt.bar(range(len(sorted_methods)), sorted_returns, color='blue', alpha=0.7)
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45)
    plt.title('Smoothing 기법별 평균 수익률 (%)')
    plt.ylabel('평균 수익률 (%)')
    plt.grid(True, alpha=0.3)
    
    # 평균 거래 횟수 비교
    plt.subplot(2, 2, 2)
    plt.bar(range(len(sorted_methods)), sorted_trades, color='red', alpha=0.7)
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45)
    plt.title('Smoothing 기법별 평균 거래 횟수')
    plt.ylabel('평균 거래 횟수')
    plt.grid(True, alpha=0.3)
    
    # 평균 샤프 비율 비교
    plt.subplot(2, 2, 3)
    plt.bar(range(len(sorted_methods)), sorted_sharpes, color='green', alpha=0.7)
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45)
    plt.title('Smoothing 기법별 평균 샤프 비율')
    plt.ylabel('평균 샤프 비율')
    plt.grid(True, alpha=0.3)
    
    # 수익률 대 거래 횟수 산점도
    plt.subplot(2, 2, 4)
    plt.scatter(sorted_trades, sorted_returns, color='purple', alpha=0.7)
    
    # 각 점에 기법 이름 표시
    for i, method in enumerate(sorted_methods):
        plt.annotate(method, (sorted_trades[i], sorted_returns[i]), 
                    textcoords="offset points", xytext=(0,5), ha='center')
    
    plt.xlabel('평균 거래 횟수')
    plt.ylabel('평균 수익률 (%)')
    plt.title('수익률 vs. 거래 횟수')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_methods_comparison.png'))
    plt.close()
    
    # 2. 윈도우별 최적 기법 히스토그램
    best_methods = []
    for window in all_windows:
        window_returns = {method: combined_results[method]['returns'][window] for method in all_methods}
        best_method = max(window_returns.items(), key=lambda x: x[1])[0]
        best_methods.append(best_method)
    
    # 최적 기법 빈도 계산
    method_counts = {}
    for method in all_methods:
        method_counts[method] = best_methods.count(method)
    
    # 빈도 기준 정렬
    sorted_method_counts = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_count_methods = [item[0] for item in sorted_method_counts]
    sorted_counts = [item[1] for item in sorted_method_counts]
    
    # 최적 기법 빈도 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_count_methods)), sorted_counts, color='orange', alpha=0.7)
    plt.xticks(range(len(sorted_count_methods)), sorted_count_methods, rotation=45)
    plt.title('윈도우별 최적 Smoothing 기법 빈도')
    plt.ylabel('윈도우 수')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_methods_histogram.png'))
    plt.close()
    
    # 3. 각 기법의 누적 성과 비교
    plt.figure(figsize=(12, 8))
    
    for method in all_methods:
        method_cum_returns = [combined_results[method]['returns'][window] for window in all_windows]
        cum_performance = np.cumsum(method_cum_returns)
        plt.plot(all_windows, cum_performance, marker='o', label=method)
    
    plt.title('Smoothing 기법별 누적 성과')
    plt.xlabel('윈도우')
    plt.ylabel('누적 수익률 (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cumulative_method_performance.png'))
    plt.close()
    
    # 4. 요약 통계 저장
    summary = {
        'avg_returns': {method: float(avg) for method, avg in zip(sorted_methods, sorted_returns)},
        'avg_trades': {method: float(avg) for method, avg in zip(sorted_methods, sorted_trades)},
        'avg_sharpes': {method: float(avg) for method, avg in zip(sorted_methods, sorted_sharpes)},
        'best_method_counts': {method: count for method, count in sorted_method_counts}
    }
    
    # 최적 방법 결정
    best_by_return = sorted_methods[0]
    best_by_sharpe = sorted_methods[np.argmax(sorted_sharpes)]
    best_by_frequency = sorted_count_methods[0]
    
    summary['best_methods'] = {
        'by_return': best_by_return,
        'by_sharpe': best_by_sharpe,
        'by_frequency': best_by_frequency
    }
    
    with open(os.path.join(save_dir, 'smoothing_methods_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # 5. 요약 출력
    print("\n===== Smoothing 기법 성과 요약 =====")
    print("수익률 기준 Top 3:")
    for i in range(min(3, len(sorted_methods))):
        print(f"  {i+1}. {sorted_methods[i]}: {sorted_returns[i]:.2f}%")
    
    print("\n샤프 비율 기준 Top 3:")
    sharpe_indices = np.argsort(sorted_sharpes)[::-1][:3]
    for i, idx in enumerate(sharpe_indices):
        print(f"  {i+1}. {sorted_methods[idx]}: {sorted_sharpes[idx]:.2f}")
    
    print("\n가장 빈번하게 최적인 기법:")
    for i in range(min(3, len(sorted_count_methods))):
        if sorted_counts[i] > 0:
            print(f"  {i+1}. {sorted_count_methods[i]}: {sorted_counts[i]} 윈도우")
    
    return summary

def run_smoothing_comparison(config):
    """
    다양한 smoothing 기법을 비교하는 롤링 윈도우 학습 실행
    
    Args:
        config: 설정 객체
        
    Returns:
        combined_results: 결합된 결과 딕셔너리
        summary: 종합 요약 딕셔너리
    """
    # 데이터 로드
    print("데이터 로드 중...")
    data = pd.read_csv(config.data_path)
    data = data.iloc[2:]  # 첫 2행 제외
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data['returns'] = data['returns'] * 100
    data["dd_10"] = data["dd_10"] * 100
    data["sortino_20"] = data["sortino_20"] * 100
    data["sortino_60"] = data["sortino_60"] * 100
    
    # 결과 저장 객체 (2중 딕셔너리: 기법 -> 윈도우 -> 성과)
    combined_results = defaultdict(lambda: {
        'window': [],
        'returns': {},
        'trades': {},
        'sharpes': {},
        'performances': []
    })
    
    # 시작 및 종료 날짜 파싱
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    
    window_number = 1
    
    # 롤링 윈도우 메인 루프
    while current_date <= end_date:
        print(f"\n=== 윈도우 {window_number} 처리 중 ===")
        
        # 윈도우별 결과 디렉토리 생성
        window_dir = os.path.join(config.results_dir, f"window_{window_number}")
        os.makedirs(window_dir, exist_ok=True)
        
        # 학습 기간 계산
        train_start = (current_date - relativedelta(years=config.total_window_years)).strftime('%Y-%m-%d')
        train_end = (current_date - relativedelta(years=config.valid_years + config.clustering_years)).strftime('%Y-%m-%d')
        
        # 검증 기간 계산
        valid_start = (current_date - relativedelta(years=config.clustering_years)).strftime('%Y-%m-%d')
        valid_end = current_date.strftime('%Y-%m-%d')
        
        # 클러스터링 기간 계산
        clustering_start = (current_date - relativedelta(years=config.clustering_years)).strftime('%Y-%m-%d')
        clustering_end = current_date.strftime('%Y-%m-%d')
        
        # 미래 적용 기간 계산
        forward_start = current_date.strftime('%Y-%m-%d')
        forward_end = (current_date + relativedelta(months=config.forward_months)).strftime('%Y-%m-%d')
        
        print(f"학습 기간: {train_start} ~ {train_end} ({config.train_years}년)")
        print(f"검증 기간: {train_end} ~ {valid_end} ({config.valid_years}년)")
        print(f"클러스터링 기간: {clustering_start} ~ {clustering_end} ({config.clustering_years}년)")
        print(f"미래 적용 기간: {forward_start} ~ {forward_end} ({config.forward_months/12:.1f}년)")
        
        # 1. 모델 학습
        model, val_loss = train_model_for_window(
            config, train_start, train_end, valid_start, valid_end, data
        )
        
        # 학습 실패 시 다음 윈도우로
        if model is None:
            print("모델 학습 실패, 다음 윈도우로 넘어갑니다.")
            current_date += relativedelta(months=config.forward_months)
            window_number += 1
            continue
        
        # 2. 레짐 식별
        kmeans, bull_regime = identify_regimes_for_window(
            config, model, data, clustering_start, clustering_end
        )
        
        # 레짐 식별 실패 시 다음 윈도우로
        if kmeans is None or bull_regime is None:
            print("레짐 식별 실패, 다음 윈도우로 넘어갑니다.")
            current_date += relativedelta(months=config.forward_months)
            window_number += 1
            continue
        
        # 3. 다양한 smoothing 기법 평가
        all_methods_results, all_methods_performances = evaluate_all_smoothing_methods(
            window_dir, model, data, kmeans, bull_regime, 
            forward_start, forward_end, window_number, config
        )
        
        # 4. 결과 저장
        if all_methods_results:
            # 각 기법별로 윈도우 결과 저장
            for method_name, result in all_methods_results.items():
                combined_results[method_name]['window'].append(window_number)
                combined_results[method_name]['returns'][window_number] = result['cum_return']
                combined_results[method_name]['trades'][window_number] = result['n_trades']
                combined_results[method_name]['sharpes'][window_number] = result['performance']['sharpe_ratio']['strategy']
                combined_results[method_name]['performances'].append(result['performance'])
        
        # 다음 윈도우로 이동
        current_date += relativedelta(months=config.forward_months)
        window_number += 1
    
    # 5. 종합 비교 시각화 및 요약
    if combined_results:
        summary = visualize_final_comparison(combined_results, config.results_dir)
        
        # 결과 저장
        with open(os.path.join(config.results_dir, 'all_results.json'), 'w') as f:
            # defaultdict는 직접 JSON으로 변환할 수 없으므로 일반 dict로 변환
            json_results = {}
            for method, result in combined_results.items():
                json_results[method] = {
                    'window': result['window'],
                    'returns': {str(k): v for k, v in result['returns'].items()},
                    'trades': {str(k): v for k, v in result['trades'].items()},
                    'sharpes': {str(k): v for k, v in result['sharpes'].items()}
                }
            json.dump(json_results, f, indent=4)
        
        print(f"\n종합 비교 완료! 총 {window_number-1}개 윈도우에서 {len(combined_results)}개 기법 비교됨.")
        return combined_results, summary
    else:
        print("비교 실패: 유효한 결과가 없습니다.")
        return None, None

def main():
    """메인 실행 함수"""
    # 명령줄 인수 파싱
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 출력 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, f"smoothing_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 설정 객체 생성
    config = RollingWindowTrainConfig()
    config.data_path = args.data_path
    config.results_dir = output_dir
    config.start_date = args.start_date
    config.end_date = args.end_date
    
    # 기간 관련 설정
    config.total_window_years = args.total_window_years
    config.train_years = args.train_years
    config.valid_years = args.valid_years
    config.clustering_years = args.clustering_years
    config.forward_months = args.forward_months
    
    # 모델 파라미터
    config.d_model = args.d_model
    config.d_state = args.d_state
    config.n_layers = args.n_layers
    config.dropout = args.dropout
    config.seq_len = args.seq_len
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    # 학습 관련 설정
    config.max_epochs = args.max_epochs
    config.patience = args.patience
    config.transaction_cost = args.transaction_cost
    
    # 설정 정보 저장
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write("=== Smoothing 기법 비교 설정 ===\n")
        f.write(f"데이터 경로: {config.data_path}\n")
        f.write(f"시작 날짜: {config.start_date}\n")
        f.write(f"종료 날짜: {config.end_date}\n")
        f.write(f"총 데이터 기간: {config.total_window_years}년\n")
        f.write(f"학습 기간: {config.train_years}년\n")
        f.write(f"검증 기간: {config.valid_years}년\n")
        f.write(f"클러스터링 기간: {config.clustering_years}년\n")
        f.write(f"다음 윈도우 간격: {config.forward_months}개월\n")
        f.write(f"모델 차원: {config.d_model}\n")
        f.write(f"상태 차원: {config.d_state}\n")
        f.write(f"레이어 수: {config.n_layers}\n")
        f.write(f"드롭아웃 비율: {config.dropout}\n")
        f.write(f"시퀀스 길이: {config.seq_len}\n")
        f.write(f"배치 크기: {config.batch_size}\n")
        f.write(f"학습률: {config.learning_rate}\n")
        f.write(f"최대 에폭: {config.max_epochs}\n")
        f.write(f"조기 종료 인내심: {config.patience}\n")
        f.write(f"거래 비용: {config.transaction_cost}\n")
    
    # 실행
    print("=== Smoothing 기법 비교 시작 ===")
    combined_results, summary = run_smoothing_comparison(config)
    
    if combined_results is not None:
        print(f"Smoothing 기법 비교 완료! 결과가 {output_dir}에 저장되었습니다.")
    else:
        print("Smoothing 기법 비교 실패!")

if __name__ == "__main__":
    main()
