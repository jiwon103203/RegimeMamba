import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import copy

from ..utils.utils import set_seed
from ..data.dataset import RegimeMambaDataset, create_dataloaders, DateRangeRegimeMambaDataset
from ..models.mamba_model import TimeSeriesMamba, create_model_from_config
from ..models.lstm import StackedLSTM
from ..models import ActorCritic
from ..models.jump_model import ModifiedJumpModel
from ..train.train import train_with_early_stopping
from ..train import train_rl_agent_for_window
from .clustering import identify_bull_bear_regimes, predict_regimes, extract_hidden_states
from .strategy import evaluate_regime_strategy, visualize_all_periods_performance
from .smoothing import apply_regime_smoothing, apply_minimum_holding_period


def train_model_for_window(config, train_start, train_end, valid_start, valid_end, data, window_number=1):
    """
    특정 기간에 대해 모델 학습
    
    Args:
        config: 설정 객체
        train_start: 학습 시작일
        train_end: 학습 종료일
        valid_start: 검증 시작일
        valid_end: 검증 종료일
        data: 전체 데이터프레임
        
    Returns:
        trained_model: 학습된 모델
        best_val_loss: 최적 검증 손실
    """
    print(f"\n학습 기간: {train_start} ~ {train_end}")
    print(f"검증 기간: {valid_start} ~ {valid_end}")
    
    # 데이터셋 생성
    train_dataset = DateRangeRegimeMambaDataset(
        data=data, 
        seq_len=config.seq_len,
        start_date=train_start,
        end_date=train_end,
        config=config
    )
    
    valid_dataset = DateRangeRegimeMambaDataset(
        data=data, 
        seq_len=config.seq_len,
        start_date=valid_start,
        end_date=valid_end,
        config=config
    )
    
    # 데이터 로더 생성
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
    
    # 데이터가 충분한지 확인
    if len(train_dataset) < 100 or len(valid_dataset) < 50:
        print(f"경고: 데이터가 부족합니다. 학습: {len(train_dataset)}, 검증: {len(valid_dataset)}")
        return None, float('inf')
    
    # 모델 생성
    if config.lstm:
        model = StackedLSTM(
            input_dim = config.input_dim
        )
        model.train_for_window(train_start, train_end, data, valid_window=config.valid_years, outputdir=config.results_dir)
        # 저장된 모델 불러오기
        model.load_state_dict(torch.load(f"./{config.results_dir}/best_model.pth"))
    else:
        model = TimeSeriesMamba(
                input_dim=config.input_dim,
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                n_layers=config.n_layers,
                dropout=config.dropout,
                config=config
        )
    
        if config.progressive_train:
            for i in range(1,3):
                best_val_loss, best_epoch, model = train_with_early_stopping(
                    model, 
                    train_loader, 
                    valid_loader, 
                    config, 
                    use_onecycle=config.use_onecycle,
                    progressive_train=i
                )
        else:
            # 조기 종료를 적용한 모델 학습
            best_val_loss, best_epoch, model = train_with_early_stopping(
                model, 
                train_loader, 
                valid_loader, 
                config, 
                use_onecycle=config.use_onecycle
            )
    
        print(f"학습 완료. 최적 검증 손실: {best_val_loss:.6f} (에폭 {best_epoch+1})")
    
    if config.rl_model:
        rl_model=ActorCritic(config, config.n_positions)
        rl_model.feature_extractor = model
        rl_model.to(config.device)
        
        # valid_start~valid end를 8:2 비율로 test, valid로 나누는 기준 만들기 (ex. valid_start = 2020-01-01, valid_end = 2020-12-31 -> test_start = 2020-10-01, test_end = 2020-12-31)
        total_valid_days = (datetime.strptime(valid_end, "%Y-%m-%d") - datetime.strptime(valid_start, "%Y-%m-%d")).days
        test_days = int(total_valid_days * 0.2)
        rl_train_start = valid_start
        rl_train_end = (datetime.strptime(valid_end, "%Y-%m-%d") - relativedelta(days=test_days + 1)).strftime("%Y-%m-%d")
        rl_test_start = (datetime.strptime(valid_end, "%Y-%m-%d") - relativedelta(days=test_days)).strftime("%Y-%m-%d")
        rl_test_end = valid_end
        agent, model, history = train_rl_agent_for_window(config, rl_model, rl_train_start, rl_train_end, rl_test_start, rl_test_end, data)
        

        return agent, model, history

    elif config.jump_model:
        jump_model = ModifiedJumpModel(config=config)
        jump_model.feature_extractor = model
        jump_model.train_for_window(train_start, train_end, data, config.valid_years, sort='cumret', window=window_number)

        return jump_model
    
    return model, best_val_loss

def identify_regimes_for_window(config, model, data, clustering_start, clustering_end):
    """
    특정 윈도우에서 레짐 식별
    
    Args:
        config: 설정 객체
        model: 학습된 모델
        data: 전체 데이터프레임
        clustering_start: 클러스터링 시작일
        clustering_end: 클러스터링 종료일
        
    Returns:
        kmeans: K-Means 모델
        bull_regime: Bull 레짐 클러스터 ID
    """
    print(f"\n레짐 식별 기간: {clustering_start} ~ {clustering_end}")
    
    # 클러스터링용 데이터셋 생성
    clustering_dataset = DateRangeRegimeMambaDataset(
        data=data, 
        seq_len=config.seq_len,
        start_date=clustering_start,
        end_date=clustering_end,
        config=config
    )
    
    # 데이터 로더 생성
    clustering_loader = DataLoader(
        clustering_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 데이터가 충분한지 확인
    if len(clustering_dataset) < 100:
        print(f"경고: 클러스터링을 위한 데이터가 부족합니다. ({len(clustering_dataset)} 샘플)")
        return None, None
    
    # Hidden states 추출
    hidden_states, returns, _ = extract_hidden_states(model, clustering_loader, config)
    
    # 클러스터링
    kmeans, bull_regime = identify_bull_bear_regimes(hidden_states, returns, config)
    
    return kmeans, bull_regime

def apply_and_evaluate_regimes(config, model, data, kmeans, bull_regime, forward_start, forward_end, window_number):
    """
    미래 기간에 레짐 적용 및 평가
    
    Args:
        config: 설정 객체
        model: 학습된 모델
        data: 전체 데이터프레임
        kmeans: K-Means 모델
        bull_regime: Bull 레짐 클러스터 ID
        forward_start: 미래 기간 시작일
        forward_end: 미래 기간 종료일
        window_number: 윈도우 번호
        
    Returns:
        results_df: 결과 데이터프레임
        performance: 성과 지표 딕셔너리
    """
    print(f"\n적용 기간: {forward_start} ~ {forward_end}")
    
    # 미래 데이터셋 생성
    forward_dataset = DateRangeRegimeMambaDataset(
        data=data, 
        seq_len=config.seq_len,
        start_date=forward_start,
        end_date=forward_end,
        config=config
    )
    
    # 데이터 로더 생성
    forward_loader = DataLoader(
        forward_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 데이터가 충분한지 확인
    if len(forward_dataset) < 10:
        print(f"경고: 평가를 위한 데이터가 부족합니다. ({len(forward_dataset)} 샘플)")
        return None, None
    
    # 레짐 예측
    predictions, true_returns, dates = predict_regimes(model, forward_loader, kmeans, bull_regime, config)
    
    # 원본 예측 저장
    raw_predictions = copy.deepcopy(predictions)
    
    # 필터링 적용 (옵션)
    if config.apply_filtering:
        if config.filter_method == 'minimum_holding':
            predictions = apply_minimum_holding_period(predictions, min_holding_days=config.min_holding_days).reshape(-1, 1)
        elif config.filter_method == 'smoothing':
            predictions = apply_regime_smoothing(predictions, method='ma', window=10).reshape(-1, 1)
    
    # 거래 비용을 고려한 전략 평가
    results_df, performance = evaluate_regime_strategy(
        predictions,
        true_returns,
        dates,
        transaction_cost=config.transaction_cost,
        config=config
    )
    
    # 결과에 기간 정보 추가
    if results_df is not None:
        results_df['window'] = window_number
        forward_start = datetime.strptime(forward_start, "%Y-%m-%d")
        results_df['train_valid_period'] = f"{forward_start - relativedelta(years=config.total_window_years)} ~ {forward_start}"
        results_df['forward_start'] = forward_start
        results_df['forward_end'] = forward_end
        
        # 원본 레짐 정보 추가
        results_df['raw_regime'] = raw_predictions.flatten()
        
        # 필터링 정보 추가
        if config.apply_filtering:
            results_df['filter_method'] = config.filter_method
            
            # 원본과 필터링 후 거래 횟수 계산
            raw_trades = (np.diff(raw_predictions.flatten()) != 0).sum() + (raw_predictions[0] == 1)
            filtered_trades = (np.diff(predictions.flatten()) != 0).sum() + (predictions[0][0] == 1)
            
            # 성과 정보에 거래 감소 정보 추가
            performance['raw_trades'] = int(raw_trades)
            performance['filtered_trades'] = int(filtered_trades)
            performance['trade_reduction'] = int(raw_trades - filtered_trades)
            if raw_trades > 0:
                performance['trade_reduction_pct'] = ((raw_trades - filtered_trades) / raw_trades) * 100
            else:
                performance['trade_reduction_pct'] = 0
            
        # 성과 정보에 윈도우 정보 추가
        performance['window'] = window_number
        performance['forward_start'] = forward_start
        performance['forward_end'] = forward_end
        
    return results_df, performance

def visualize_window_performance(results_df, model_loss, window_number, title, save_path):
    """
    윈도우 성과 시각화
    
    Args:
        results_df: 결과 데이터프레임
        model_loss: 모델 검증 손실
        window_number: 윈도우 번호
        title: 차트 제목
        save_path: 저장 경로
    """
    plt.figure(figsize=(15, 10))
    
    # 누적 수익률 및 레짐 표시
    plt.subplot(2, 1, 1)
    plt.plot(results_df['Cum_Market'] * 100, label='시장', color='gray')
    plt.plot(results_df['Cum_Strategy'] * 100, label='레짐 전략', color='blue')
    plt.title(f'{title} (검증 손실: {model_loss:.6f})')
    plt.legend()
    plt.ylabel('수익률 (%)')
    plt.grid(True)
    
    # 레짐 신호 표시
    plt.subplot(2, 1, 2)
    plt.plot(results_df['Regime'], label='레짐 (1=Bull, 0=Bear)', color='red')
    plt.title('레짐 신호')
    plt.ylabel('레짐')
    plt.grid(True)
    
    # 필터링 사용 시 원본 레짐도 표시
    if 'raw_regime' in results_df.columns:
        plt.plot(results_df['raw_regime'], label='원본 레짐', color='green', alpha=0.5, linestyle='--')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_rolling_window_train(config):
    """
    롤링 윈도우 재학습 실행 함수
    
    Args:
        config: 설정 객체
        
    Returns:
        combined_results: 전체 결과 데이터프레임
        all_performances: 전체 성과 리스트
        model_histories: 모델 학습 이력 리스트
    """
    # 데이터 로드
    print("데이터 로드 중...")
    data = pd.read_csv(config.data_path)
    
    
    # 결과 저장 객체
    all_results = []
    all_performances = []
    model_histories = []
    
    # 시작 및 종료 날짜 파싱
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    
    window_number = 1
    
    # 롤링 윈도우 메인 루프
    while current_date <= end_date:
        print(f"\n=== 윈도우 {window_number} 처리 중 ===")
        
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
        
        # 3. 미래 기간에 레짐 적용 및 평가
        results_df, performance = apply_and_evaluate_regimes(
            config, model, data, kmeans, bull_regime, forward_start, forward_end, window_number
        )
        
        # 결과 저장
        if results_df is not None and performance is not None:
            all_results.append(results_df)
            all_performances.append(performance)
            
            # 모델 정보 저장
            model_history = {
                'window': window_number,
                'train_start': train_start,
                'train_end': train_end,
                'valid_start': valid_start,
                'valid_end': valid_end,
                'val_loss': val_loss,
                'bull_regime': bull_regime
            }
            model_histories.append(model_history)
            
            # 결과 파일 저장
            results_df.to_csv(
                f"{config.results_dir}/window_{window_number}_results.csv",
                index=False
            )
            
            # 결과 시각화
            visualize_window_performance(
                results_df,
                val_loss,
                window_number,
                f"윈도우 {window_number}: {forward_start} ~ {forward_end}",
                f"{config.results_dir}/window_{window_number}_performance.png"
            )
            
            # 모델 저장 (선택적)
            model_save_path = f"{config.results_dir}/window_{window_number}_model.pth"
            torch.save({
                'window': window_number,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'bull_regime': bull_regime
            }, model_save_path)
            
        # 다음 윈도우로 이동
        current_date += relativedelta(months=config.forward_months)
        window_number += 1
    
    # 전체 결과 병합 및 저장
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{config.results_dir}/all_windows_results.csv", index=False)
        
        # 모델 학습 이력 저장
        with open(f"{config.results_dir}/model_histories.json", 'w') as f:
            json.dump(model_histories, f, indent=4)
        
        # 전체 성과 저장
        with open(f"{config.results_dir}/all_performances.json", 'w') as f:
            json.dump(all_performances, f, default=default_converter, indent=4)
        
        # 전체 성과 시각화
        visualize_all_windows_performance(all_performances, config.results_dir)
        
        print(f"\n롤링 윈도우 재학습 완료! 총 {len(all_performances)}개 윈도우 처리됨.")
        return combined_results, all_performances, model_histories
    else:
        print("롤링 윈도우 재학습 실패: 유효한 결과가 없습니다.")
        return None, None, None

def default_converter(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Type {type(o).__name__} is not serializable")

def visualize_all_windows_performance(all_performances, save_dir):
    """
    모든 윈도우의 성과 시각화
    
    Args:
        all_performances: 모든 윈도우의 성과 리스트
        save_dir: 저장 디렉토리
    """
    # 성과 데이터 추출
    windows = [p['window'] for p in all_performances]
    market_returns = [p['cumulative_returns']['market'] for p in all_performances]
    strategy_returns = [p['cumulative_returns']['strategy'] for p in all_performances]
    market_sharpes = [p['sharpe_ratio']['market'] for p in all_performances]
    strategy_sharpes = [p['sharpe_ratio']['strategy'] for p in all_performances]
    
    # 필터링 통계가 있는 경우
    if 'raw_trades' in all_performances[0]:
        raw_trades = [p['raw_trades'] for p in all_performances]
        filtered_trades = [p['filtered_trades'] for p in all_performances]
        trade_reductions_pct = [p['trade_reduction_pct'] for p in all_performances]
        
    # 시각화 (여러 서브플롯)
    plt.figure(figsize=(15, 12))
    
    # 1. 윈도우별 수익률 비교
    plt.subplot(2, 2, 1)
    width = 0.35
    x = np.arange(len(windows))
    plt.bar(x - width/2, market_returns, width, label='시장', color='gray')
    plt.bar(x + width/2, strategy_returns, width, label='레짐 전략', color='blue')
    plt.xlabel('윈도우')
    plt.ylabel('누적 수익률 (%)')
    plt.title('윈도우별 누적 수익률 비교')
    plt.xticks(x, windows)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 윈도우별 샤프 비율 비교
    plt.subplot(2, 2, 2)
    plt.bar(x - width/2, market_sharpes, width, label='시장', color='gray')
    plt.bar(x + width/2, strategy_sharpes, width, label='레짐 전략', color='blue')
    plt.xlabel('윈도우')
    plt.ylabel('샤프 비율')
    plt.title('윈도우별 샤프 비율 비교')
    plt.xticks(x, windows)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 필터링 통계가 있는 경우 거래 횟수 비교
    if 'raw_trades' in all_performances[0]:
        plt.subplot(2, 2, 3)
        plt.bar(x - width/2, raw_trades, width, label='원본 거래', color='red', alpha=0.7)
        plt.bar(x + width/2, filtered_trades, width, label='필터링 후 거래', color='blue')
        plt.xlabel('윈도우')
        plt.ylabel('거래 횟수')
        plt.title('윈도우별 거래 횟수 비교')
        plt.xticks(x, windows)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 거래 감소율
        plt.subplot(2, 2, 4)
        plt.bar(x, trade_reductions_pct, color='green')
        plt.axhline(y=np.mean(trade_reductions_pct), color='black', linestyle='--',
                   label=f'평균: {np.mean(trade_reductions_pct):.2f}%')
        plt.xlabel('윈도우')
        plt.ylabel('거래 감소율 (%)')
        plt.title('윈도우별 거래 감소율')
        plt.xticks(x, windows)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # 대체 그래프 (누적 성과)
        plt.subplot(2, 1, 2)
        plt.plot(windows, np.cumsum(market_returns), 'o-', label='시장 누적', color='gray')
        plt.plot(windows, np.cumsum(strategy_returns), 'o-', label='전략 누적', color='blue')
        plt.xlabel('윈도우')
        plt.ylabel('누적 수익률 (%)')
        plt.title('윈도우 누적 성과')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_windows_comparison.png")
    plt.close()
    
    # 전체 성과 요약 계산
    total_market_return = sum(market_returns)
    total_strategy_return = sum(strategy_returns)
    avg_market_sharpe = np.mean(market_sharpes)
    avg_strategy_sharpe = np.mean(strategy_sharpes)
    win_rate = sum(np.array(strategy_returns) > np.array(market_returns)) / len(windows) * 100
    
    # 평균 복리 수익률 계산
    annualized_market_return = ((1 + total_market_return/100) ** (1/len(windows)) - 1) * 100
    annualized_strategy_return = ((1 + total_strategy_return/100) ** (1/len(windows)) - 1) * 100
    
    summary = {
        'total_windows': len(windows),
        'total_returns': {
            'market': total_market_return,
            'strategy': total_strategy_return,
            'difference': total_strategy_return - total_market_return
        },
        'annualized_returns': {
            'market': annualized_market_return,
            'strategy': annualized_strategy_return,
            'difference': annualized_strategy_return - annualized_market_return
        },
        'sharpe_ratios': {
            'market_avg': avg_market_sharpe,
            'strategy_avg': avg_strategy_sharpe,
            'difference': avg_strategy_sharpe - avg_market_sharpe
        },
        'win_rate': win_rate
    }
    
    # 필터링 통계가 있는 경우 추가
    if 'raw_trades' in all_performances[0]:
        total_raw_trades = sum(raw_trades)
        total_filtered_trades = sum(filtered_trades)
        total_reduction = total_raw_trades - total_filtered_trades
        total_reduction_pct = (total_reduction / total_raw_trades) * 100 if total_raw_trades > 0 else 0
        
        summary['trading_statistics'] = {
            'total_raw_trades': total_raw_trades,
            'total_filtered_trades': total_filtered_trades,
            'total_reduction': total_reduction,
            'total_reduction_pct': total_reduction_pct,
            'avg_reduction_pct': np.mean(trade_reductions_pct)
        }
    
    # 요약 저장
    with open(f"{save_dir}/performance_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    # 요약 출력
    print("\n===== 전체 성과 요약 =====")
    print(f"총 윈도우 수: {summary['total_windows']}")
    print(f"총 시장 수익률: {summary['total_returns']['market']:.2f}%")
    print(f"총 전략 수익률: {summary['total_returns']['strategy']:.2f}%")
    print(f"수익률 차이: {summary['total_returns']['difference']:.2f}%")
    print(f"연율화 시장 수익률: {summary['annualized_returns']['market']:.2f}%")
    print(f"연율화 전략 수익률: {summary['annualized_returns']['strategy']:.2f}%")
    print(f"평균 시장 샤프 비율: {summary['sharpe_ratios']['market_avg']:.2f}")
    print(f"평균 전략 샤프 비율: {summary['sharpe_ratios']['strategy_avg']:.2f}")
    print(f"시장 대비 승률: {summary['win_rate']:.2f}%")
    
    if 'trading_statistics' in summary:
        print("\n거래 통계:")
        print(f"  총 원본 거래 횟수: {summary['trading_statistics']['total_raw_trades']}")
        print(f"  총 필터링 후 거래 횟수: {summary['trading_statistics']['total_filtered_trades']}")
        print(f"  총 거래 감소: {summary['trading_statistics']['total_reduction']} ({summary['trading_statistics']['total_reduction_pct']:.2f}%)")
        print(f"  평균 거래 감소율: {summary['trading_statistics']['avg_reduction_pct']:.2f}%")
