import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from tqdm import tqdm
import os

def apply_regime_smoothing(regime_predictions, method="ma", window=5, threshold=0.5):
    """
    레짐 예측 결과에 다양한 평활화 기법 적용

    Args:
        regime_predictions: 원본 레짐 예측값 (1=Bull, 0=Bear)
        method: 평활화 방법 ('ma'=이동평균, 'exp'=지수평활화)
        window: 평활화 윈도우 크기
        threshold: 레짐 결정 임계값

    Returns:
        smoothed_regimes: 평활화된 레짐 (1=Bull, 0=Bear)
    """
    regime_series = pd.Series(regime_predictions.flatten())

    # 방법에 따른 평활화 적용
    if method == "ma":
        # 이동 평균 적용
        smoothed_probs = regime_series.rolling(window=window, center=False).mean()
        # NaN 값을 첫 번째 유효 값으로 채움
        smoothed_probs.fillna(regime_series.iloc[0], inplace=True)

    elif method == "exp":
        # 지수 이동 평균 적용
        smoothed_probs = regime_series.ewm(span=window, adjust=False).mean()
    else:
        raise ValueError(f"지원하지 않는 평활화 방법: {method}")

    # 임계값을 적용하여 최종 레짐 결정
    smoothed_regimes = (smoothed_probs > threshold).astype(int)

    return smoothed_regimes.values

def apply_confirmation_rule(regime_predictions, confirmation_days=3):
    """
    레짐 변경에 확인 규칙 적용 - N일 연속으로 같은 신호가 나타날 때만 레짐 변경

    Args:
        regime_predictions: 원본 레짐 예측값 (1=Bull, 0=Bear)
        confirmation_days: 레짐 변경 확인에 필요한 연속 일수

    Returns:
        confirmed_regimes: 확인 규칙이 적용된 레짐 (1=Bull, 0=Bear)
    """
    regimes = regime_predictions.flatten()
    confirmed_regimes = np.copy(regimes)

    # 첫 번째 레짐 설정
    current_regime = regimes[0]
    confirmation_count = 1

    # 각 날짜별로 확인 규칙 적용
    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            # 현재 레짐과 동일하면 확인 카운트 유지
            confirmed_regimes[i] = current_regime
            confirmation_count = min(confirmation_count + 1, confirmation_days)
        else:
            # 다른 레짐 신호 발생
            if confirmation_count >= confirmation_days:
                # 이전 레짐이 충분히 확인됨
                confirmation_count = 1
                current_regime = regimes[i]
                confirmed_regimes[i] = current_regime
            else:
                # 아직 확인되지 않음 - 이전 레짐 유지
                confirmed_regimes[i] = current_regime
                confirmation_count += 1

    return confirmed_regimes

def apply_minimum_holding_period(regime_predictions, returns=None, min_holding_days=20):
    """
    최소 보유 기간 규칙 적용 - 레짐 변경 후 최소 N일은 유지

    Args:
        regime_predictions: 원본 레짐 예측값 (1=Bull, 0=Bear)
        returns: 수익률 데이터 (선택적으로 수익률 기반 탈출 규칙에 사용)
        min_holding_days: 최소 보유 기간 (일)

    Returns:
        filtered_regimes: 최소 보유 기간이 적용된 레짐 (1=Bull, 0=Bear)
    """
    regimes = regime_predictions.flatten()
    filtered_regimes = np.copy(regimes)

    # 첫 날 레짐 설정
    current_regime = regimes[0]
    days_since_change = 0

    # 각 날짜별로 최소 보유 기간 규칙 적용
    for i in range(1, len(regimes)):
        if days_since_change < min_holding_days:
            # 최소 보유 기간이 지나지 않았으면 기존 레짐 유지
            filtered_regimes[i] = current_regime
            days_since_change += 1
        else:
            # 최소 보유 기간이 지났으면 레짐 변경 가능
            if regimes[i] != current_regime:
                # 레짐 변경
                current_regime = regimes[i]
                days_since_change = 0
            filtered_regimes[i] = current_regime

    return filtered_regimes

def apply_probability_threshold(hidden_states, kmeans, bull_regime, threshold=0.6):
    """
    클러스터 확률 임계값 기반 레짐 판별

    Args:
        hidden_states: 모델의 hidden states
        kmeans: 훈련된 KMeans 모델
        bull_regime: Bull 레짐 클러스터 ID
        threshold: 확률 임계값 (해당 클러스터에 속할 확률이 이 값을 넘어야 함)

    Returns:
        regime_predictions: 확률 임계값이 적용된 레짐 (1=Bull, 0=Bear)
    """
    # KMeans의 군집 중심과 각 데이터 포인트 간의 거리 계산
    distances = kmeans.transform(hidden_states)

    # 거리를 확률로 변환 (음수 거리를 사용하여 더 가까울수록 확률이 높아지도록)
    neg_distances = -distances
    probabilities = np.exp(neg_distances) / np.sum(np.exp(neg_distances), axis=1, keepdims=True)

    # Bull 레짐 클러스터에 속할 확률
    bull_probabilities = probabilities[:, bull_regime]

    # 임계값 적용: Bull 확률이 threshold보다 크면 Bull(1), 작으면 Bear(0)
    regime_predictions = (bull_probabilities > threshold).astype(int)

    return regime_predictions

def apply_filtering(predictions, method='minimum_holding', window=10, confirmation_days=3, min_holding_days=20, threshold=0.6):
    """
    레짐 예측에 필터링 적용

    Args:
        predictions: 원본 레짐 예측값 (1=Bull, 0=Bear)
        method: 필터링 방식 ('smoothing', 'confirmation', 'minimum_holding', 'none')
        window: 이동평균 윈도우 크기
        confirmation_days: 레짐 변경 확인에 필요한 연속 일수
        min_holding_days: 최소 보유 기간(일)
        threshold: 임계값

    Returns:
        filtered_predictions: 필터링된 레짐 예측
    """
    # 예측을 1차원 배열로 변환
    preds = predictions.flatten()

    if method == 'none':
        return predictions

    elif method == 'smoothing':
        # 이동평균 적용
        return apply_regime_smoothing(predictions, method="ma", window=window, threshold=threshold)

    elif method == 'exp':
        # 지수 이동평균 적용
        return apply_regime_smoothing(predictions, method="exp", window=window, threshold=threshold)
    elif method == 'confirmation':
        # 레짐 변경 확인 규칙 적용
        return apply_confirmation_rule(predictions, confirmation_days=confirmation_days).reshape(-1, 1)

    elif method == 'minimum_holding':
        # 최소 보유 기간 적용
        return apply_minimum_holding_period(predictions, min_holding_days=min_holding_days).reshape(-1, 1)

    else:
        print(f"Unknown filtering method: {method}, no filtering applied")
        return predictions

def predict_regimes_with_filtering(model, dataloader, kmeans, bull_regime, device,
                                  filter_method='minimum_holding',
                                  window=10,
                                  confirmation_days=3,
                                  min_holding_days=20,
                                  probability_threshold=0.6):
    """
    필터링이 적용된 레짐 예측 함수

    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        kmeans: 훈련된 KMeans 모델
        bull_regime: Bull 레짐 클러스터 ID
        device: 연산 장치
        filter_method: 필터링 방식 ('smoothing', 'confirmation', 'minimum_holding', 'probability', 'none')
        기타 필터링 관련 파라미터들

    Returns:
        predictions: 필터링이 적용된 레짐 예측 (1=Bull, 0=Bear)
        true_returns: 실제 수익률
        dates: 날짜 정보
        raw_predictions: 필터링 전 원본 레짐 예측
        hidden_states: 모델의 hidden states
    """
    model.eval()
    hidden_states_list = []
    predictions = []
    true_returns = []
    dates = []

    with torch.no_grad():
        for x, y, date in dataloader:
            x = x.to(device)
            _, hidden = model(x, return_hidden=True)
            hidden_cpu = hidden.cpu().numpy()

            # 군집 할당
            cluster = kmeans.predict(hidden_cpu)

            # 확률 기반 레짐 결정 적용 시
            if filter_method == 'probability':
                # 클러스터까지의 거리 계산
                distances = kmeans.transform(hidden_cpu)
                # 확률로 변환
                neg_distances = -distances
                probabilities = np.exp(neg_distances) / np.sum(np.exp(neg_distances), axis=1, keepdims=True)
                # Bull 레짐 확률
                bull_probs = probabilities[:, bull_regime]
                # 임계값 적용
                regime_pred = (bull_probs > probability_threshold).astype(int)
            else:
                # 기본 레짐 예측 (Bull 레짐이면 1, 아니면 0)
                regime_pred = np.where(cluster == bull_regime, 1, 0)

            hidden_states_list.append(hidden_cpu)
            predictions.extend(regime_pred)
            true_returns.extend(y.numpy())
            dates.extend(date)

    # NumPy 배열로 변환
    predictions = np.array(predictions)
    true_returns = np.array(true_returns)
    hidden_states = np.vstack(hidden_states_list)

    # 원본 예측 저장
    raw_predictions = predictions.copy()

    # 선택한 필터링 방법 적용
    if filter_method == 'smoothing':
        predictions = apply_regime_smoothing(predictions, method="ma", window=window)
    elif filter_method == 'exp':
        predictions = apply_regime_smoothing(predictions, method="exp", window=window)
    elif filter_method == 'confirmation':
        predictions = apply_confirmation_rule(predictions, confirmation_days=confirmation_days)
    elif filter_method == 'minimum_holding':
        predictions = apply_minimum_holding_period(predictions, min_holding_days=min_holding_days)
    elif filter_method == 'probability':
        # 이미 위에서 적용됨
        pass
    elif filter_method != 'none':
        print(f"알 수 없는 필터링 방법: {filter_method}, 필터링 적용 안함")

    return predictions, true_returns, dates, raw_predictions, hidden_states

def compare_filtering_strategies(original_regimes, returns, dates, transaction_cost=0.001, save_path=None):
    """
    다양한 레짐 필터링 전략의 성과 비교

    Args:
        original_regimes: 원본 레짐 예측값 (1=Bull, 0=Bear)
        returns: 실제 수익률
        dates: 날짜 정보
        transaction_cost: 거래 비용
        save_path: 결과 그래프 저장 경로 (기본값: None)

    Returns:
        results: 각 전략의 성과 비교 결과
        summary_df: 요약 테이블
    """
    # 필터링 전략 정의
    strategies = {
        "Original": original_regimes,
        "MA(5)": apply_regime_smoothing(original_regimes, method="ma", window=5),
        "MA(10)": apply_regime_smoothing(original_regimes, method="ma", window=10),
        "EMA(10)": apply_regime_smoothing(original_regimes, method="exp", window=10),
        "Confirm(3)": apply_confirmation_rule(original_regimes, confirmation_days=3),
        "Confirm(5)": apply_confirmation_rule(original_regimes, confirmation_days=5),
        "MinHold(10)": apply_minimum_holding_period(original_regimes, min_holding_days=10),
        "MinHold(20)": apply_minimum_holding_period(original_regimes, min_holding_days=20),
    }

    # 결과 저장 객체
    results = {}

    # 각 전략 평가
    for name, regimes in strategies.items():
        # 결과 생성
        df = pd.DataFrame({
            'Date': dates,
            'Regime': regimes.flatten(),
            'Return': returns.flatten()
        })

        # 레짐 변화 감지 (거래 발생)
        df['Regime_Change'] = df['Regime'].diff().fillna(0) != 0

        # 첫 번째 진입도 거래로 간주
        df.loc[0, 'Regime_Change'] = df.loc[0, 'Regime'] == 1

        # 거래 비용 계산 (레짐이 변할 때마다 적용)
        df['Transaction_Cost'] = np.where(df['Regime_Change'], transaction_cost * 100, 0)

        # 거래 비용을 고려한 전략 수익률 계산
        df['Strategy_Return'] = df['Regime'] * df['Return'] - df['Transaction_Cost']

        # 누적 수익률 계산
        df['Cum_Market'] = (1 + df['Return']/100).cumprod() - 1
        df['Cum_Strategy'] = (1 + df['Strategy_Return']/100).cumprod() - 1

        # 주요 지표 계산
        market_return = df['Cum_Market'].iloc[-1] * 100
        strategy_return = df['Cum_Strategy'].iloc[-1] * 100
        n_trades = df['Regime_Change'].sum()
        total_cost = df['Transaction_Cost'].sum()

        # 최대 낙폭 계산
        df['Strategy_Peak'] = df['Cum_Strategy'].cummax()
        df['Strategy_Drawdown'] = (df['Cum_Strategy'] - df['Strategy_Peak']) / (1 + df['Strategy_Peak']) * 100
        max_drawdown = df['Strategy_Drawdown'].min()

        # 결과 저장
        results[name] = {
            'cum_return': strategy_return,
            'n_trades': n_trades,
            'total_cost': total_cost,
            'max_drawdown': max_drawdown,
            'df': df
        }

    # 성과 비교 차트 생성
    plt.figure(figsize=(15, 12))

    # 누적 수익률 비교
    plt.subplot(3, 1, 1)
    for name, result in results.items():
        plt.plot(result['df']['Cum_Strategy'] * 100, label=name)
    plt.plot(results['Original']['df']['Cum_Market'] * 100, label='Market', color='black', linestyle='--')
    plt.title('Cumulative Returns of Different Filtering Strategies')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)

    # 거래 횟수 비교
    plt.subplot(3, 1, 2)
    names = list(results.keys())
    trade_counts = [results[name]['n_trades'] for name in names]
    plt.bar(names, trade_counts)
    plt.title('Number of Trades by Strategy')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    # 누적 수익률 및 최대 낙폭 비교
    plt.subplot(3, 1, 3)
    returns = [results[name]['cum_return'] for name in names]
    drawdowns = [results[name]['max_drawdown'] for name in names]

    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width/2, returns, width, label='Cumulative Return (%)')
    plt.bar(x + width/2, drawdowns, width, label='Maximum Drawdown (%)')
    plt.title('Returns vs. Maximum Drawdowns')
    plt.xticks(x, names, rotation=45)
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    # 요약 테이블 반환
    summary_data = {
        'Strategy': names,
        'Cumulative Return (%)': [results[name]['cum_return'] for name in names],
        'Number of Trades': [results[name]['n_trades'] for name in names],
        'Total Cost (%)': [results[name]['total_cost'] for name in names],
        'Maximum Drawdown (%)': [results[name]['max_drawdown'] for name in names]
    }

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.sort_values('Cumulative Return (%)', ascending=False).to_string(index=False))

    return results, summary_df

def visualize_filtered_vs_original(results_df, raw_predictions, filtered_predictions, title, save_path=None):
    """
    원본 레짐과 필터링된 레짐 비교 시각화

    Args:
        results_df: 결과 데이터프레임
        raw_predictions: 원본 레짐 예측
        filtered_predictions: 필터링된 레짐 예측
        title: 차트 제목
        save_path: 저장 경로 (기본값: None)
    """
    # 원본 레짐 거래 발생 지점 계산
    raw_regime = raw_predictions.flatten()
    raw_changes = np.diff(raw_regime, prepend=raw_regime[0])
    raw_trade_points = np.where(raw_changes != 0)[0]

    # 필터링된 레짐 거래 발생 지점 계산
    filtered_regime = filtered_predictions.flatten()
    filtered_changes = np.diff(filtered_regime, prepend=filtered_regime[0])
    filtered_trade_points = np.where(filtered_changes != 0)[0]

    # 누적 수익률 계산
    raw_cum_return = (1 + results_df['Return']/100 * raw_regime).cumprod() - 1
    filtered_cum_return = results_df['Cum_Strategy']
    market_cum_return = results_df['Cum_Market']

    # 시각화 (3개의 서브플롯)
    plt.figure(figsize=(15, 12))

    # 1. 누적 수익률 비교
    plt.subplot(3, 1, 1)
    plt.plot(market_cum_return * 100, label='Market', color='gray')
    plt.plot(raw_cum_return * 100, label='Original Strategy', color='red', linestyle='--')
    plt.plot(filtered_cum_return * 100, label='Filtered Strategy', color='blue')
    plt.title(f'{title} - Cumulative Returns Comparison')
    plt.legend()
    plt.ylabel('Return (%)')
    plt.grid(True)

    # 2. 레짐 신호 비교
    plt.subplot(3, 1, 2)
    plt.plot(raw_regime, label='Original Regime', color='red', alpha=0.6)
    plt.plot(filtered_regime, label='Filtered Regime', color='blue')

    # 거래 발생 지점 표시
    for point in raw_trade_points:
        plt.axvline(x=point, color='red', alpha=0.3, linestyle='--')
    for point in filtered_trade_points:
        plt.axvline(x=point, color='blue', alpha=0.3, linestyle='-')

    plt.title('Regime Signals Comparison')
    plt.ylabel('Regime (1=Bull, 0=Bear)')
    plt.legend()
    plt.grid(True)

    # 3. 누적 거래 비용 비교
    plt.subplot(3, 1, 3)

    # 원본 거래 비용 계산
    raw_costs = np.zeros_like(raw_regime)
    raw_costs[raw_trade_points] = results_df['Transaction_Cost'].iloc[0]  # 첫 번째 거래 비용 값 사용
    if raw_regime[0] == 1:  # 첫 날이 Bull이면 거래 발생
        raw_costs[0] = results_df['Transaction_Cost'].iloc[0]

    # 필터링된 거래 비용 (이미 results_df에 있음)
    filtered_costs = results_df['Transaction_Cost'].values

    # 누적 비용 계산
    raw_cum_costs = np.cumsum(raw_costs)
    filtered_cum_costs = np.cumsum(filtered_costs)

    plt.plot(raw_cum_costs, label=f'Original Cumulative Cost ({len(raw_trade_points)} trades)', color='red', alpha=0.6)
    plt.plot(filtered_cum_costs, label=f'Filtered Cumulative Cost ({len(filtered_trade_points)} trades)', color='blue')
    plt.title('Cumulative Transaction Costs')
    plt.ylabel('Cost (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    # 개선 통계 출력
    trade_reduction = len(raw_trade_points) - len(filtered_trade_points)
    trade_reduction_pct = trade_reduction / len(raw_trade_points) * 100 if len(raw_trade_points) > 0 else 0
    
    raw_final_return = raw_cum_return.iloc[-1] * 100
    filtered_final_return = filtered_cum_return.iloc[-1] * 100
    return_improvement = filtered_final_return - raw_final_return
    
    print(f"\n필터링 개선 통계:")
    print(f"  원본 거래 횟수: {len(raw_trade_points)}")
    print(f"  필터링 후 거래 횟수: {len(filtered_trade_points)}")
    print(f"  거래 감소: {trade_reduction} ({trade_reduction_pct:.2f}%)")
    print(f"  원본 최종 수익률: {raw_final_return:.2f}%")
    print(f"  필터링 후 최종 수익률: {filtered_final_return:.2f}%")
    print(f"  수익률 개선: {return_improvement:.2f}%")
    
    return {
        'trade_reduction': trade_reduction,
        'trade_reduction_pct': trade_reduction_pct,
        'raw_final_return': raw_final_return,
        'filtered_final_return': filtered_final_return,
        'return_improvement': return_improvement
    }

def find_optimal_filtering(config, data_path, save_path=None):
    """
    다양한 필터링 방법을 시도하여 최적의 레짐 필터링 파라미터 탐색

    Args:
        config: 설정 객체
        data_path: 데이터 파일 경로
        save_path: 결과 저장 경로 (기본값: None)

    Returns:
        optimal_params: 최적의 필터링 파라미터
        results: 각 전략의 성과 비교 결과
    """
    from torch.utils.data import DataLoader
    from sklearn.cluster import KMeans
    
    from ..data.dataset import RegimeMambaDataset
    from ..models.mamba_model import create_model_from_config
    from .clustering import extract_hidden_states, predict_regimes
    
    # 데이터 로드
    data = pd.read_csv(data_path)
    data = data.iloc[2:]
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data['returns'] = data['returns'] * 100
    data["dd_10"] = data["dd_10"] * 100
    data["sortino_20"] = data["sortino_20"] * 100
    data["sortino_60"] = data["sortino_60"] * 100

    # 날짜 칼럼 식별
    date_col = 'Price' if 'Price' in data.columns else 'Date'

    print("모델 로드 중...")
    # 모델 로드
    model = create_model_from_config(config)
    if hasattr(config, 'model_path') and config.model_path:
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"모델 체크포인트 로드: {config.model_path}")
    
    model.to(config.device)
    model.eval()

    # 데이터 기간 분할
    print("데이터 분할 중...")
    train_end = '2009-12-31'
    test_start = '2010-01-01'
    test_end = '2015-12-31'
    
    train_data = data[(data[date_col] >= '2000-01-01') & (data[date_col] <= train_end)]
    test_data = data[(data[date_col] >= test_start) & (data[date_col] <= test_end)]

    # 데이터셋 및 데이터로더 생성
    print("데이터셋 생성 중...")
    train_dataset = RegimeMambaDataset(data_path, seq_len=config.seq_len, mode="valid", target_type=config.target_type, target_horizon=config.target_horizon)  # valid 모드는 2000-2009 데이터 사용
    test_dataset = RegimeMambaDataset(data_path, seq_len=config.seq_len, mode="test", target_type=config.target_type, target_horizon=config.target_horizon)    # test 모드는 2010 이후 데이터 사용

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 훈련 데이터로 레짐 식별
    print("훈련 데이터로 레짐 식별 중...")
    train_hidden, train_returns, _ = extract_hidden_states(model, train_loader, config.device)
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
    clusters = kmeans.fit_predict(train_hidden)

    # Bull 레짐 식별 (평균 수익률이 가장 높은 클러스터)
    cluster_returns = {}
    for i in range(config.n_clusters):
        cluster_mask = (clusters == i)
        avg_return = train_returns[cluster_mask].mean()
        cluster_returns[i] = avg_return

    bull_regime = max(cluster_returns, key=cluster_returns.get)
    print(f"Bull 레짐 클러스터: {bull_regime}")

    # 테스트 데이터에 적용
    print("테스트 데이터에 레짐 예측 적용 중...")
    test_predictions, test_returns, test_dates = predict_regimes(
        model, test_loader, kmeans, bull_regime, config.device
    )

    # 다양한 필터링 전략 비교
    print("다양한 필터링 전략 비교 중...")
    results, summary = compare_filtering_strategies(
        test_predictions, test_returns, test_dates, 
        transaction_cost=config.transaction_cost,
        save_path=save_path
    )

    # 최적 전략 찾기
    optimal_strategy = summary.sort_values('Cumulative Return (%)', ascending=False).iloc[0]['Strategy']
    print(f"\n최적 필터링 전략: {optimal_strategy}")

    # 최적 전략의 파라미터 결정
    if optimal_strategy == 'Original':
        optimal_params = {'filter_method': 'none'}
    elif optimal_strategy.startswith('MA'):
        window = int(optimal_strategy.split('(')[1].split(')')[0])
        optimal_params = {'filter_method': 'smoothing', 'window': window}
    elif optimal_strategy.startswith('EMA'):
        window = int(optimal_strategy.split('(')[1].split(')')[0])
        optimal_params = {'filter_method': 'exp', 'window': window}
    elif optimal_strategy.startswith('Confirm'):
        days = int(optimal_strategy.split('(')[1].split(')')[0])
        optimal_params = {'filter_method': 'confirmation', 'confirmation_days': days}
    elif optimal_strategy.startswith('MinHold'):
        days = int(optimal_strategy.split('(')[1].split(')')[0])
        optimal_params = {'filter_method': 'minimum_holding', 'min_holding_days': days}
    else:
        optimal_params = {'filter_method': 'none'}

    print("최적 필터링 파라미터:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value}")

    # 최적 파라미터 적용하여 시각화
    if optimal_params['filter_method'] != 'none':
        if optimal_params['filter_method'] == 'smoothing':
            filtered_predictions = apply_regime_smoothing(
                test_predictions, 
                method="ma", 
                window=optimal_params['window']
            )
        elif optimal_params['filter_method'] == 'exp':
            filtered_predictions = apply_regime_smoothing(
                test_predictions, 
                method="exp", 
                window=optimal_params['window']
            )
        elif optimal_params['filter_method'] == 'confirmation':
            filtered_predictions = apply_confirmation_rule(
                test_predictions, 
                confirmation_days=optimal_params['confirmation_days']
            )
        elif optimal_params['filter_method'] == 'minimum_holding':
            filtered_predictions = apply_minimum_holding_period(
                test_predictions, 
                min_holding_days=optimal_params['min_holding_days']
            )
            
        # 필터링된 레짐을 사용하여 성과 평가
        from .strategy import evaluate_regime_strategy
        filtered_results_df, _ = evaluate_regime_strategy(
            filtered_predictions.reshape(-1, 1), 
            test_returns, 
            test_dates, 
            transaction_cost=config.transaction_cost
        )
        
        # 원본과 필터링된 레짐 비교
        if save_path:
            save_dir = os.path.dirname(save_path)
            optimal_viz_path = os.path.join(save_dir, 'optimal_filtering_comparison.png')
        else:
            optimal_viz_path = None
            
        visualize_filtered_vs_original(
            filtered_results_df,
            test_predictions,
            filtered_predictions.reshape(-1, 1),
            f"Optimal Filtering Strategy: {optimal_strategy}",
            save_path=optimal_viz_path
        )

    return optimal_params, results
