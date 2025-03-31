import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json

def evaluate_regime_strategy(predictions, returns, dates=None, transaction_cost=0.001, save_path=None, config=None):
    """
    레짐 기반 전략의 성과를 평가하며 거래 비용을 고려

    Args:
        predictions: 예측된 레짐 (1=Bull, 0=Bear)
        returns: 실제 수익률
        dates: 날짜 정보
        transaction_cost: 거래 비용 (백분율, 예: 0.001 = 0.1%)
        save_path: 결과 그래프 저장 경로 (None이면 저장하지 않음)

    Returns:
        df: 상세 결과 데이터프레임
        performance: 성과 지표 딕셔너리
    """
    df = pd.DataFrame({
        'Date': dates,
        'Regime': predictions.flatten(),
        'Return': returns.flatten()
    })

    # Date 순서로 정렬
    df.sort_values('Date').reset_index(drop=True, inplace=True)

    if config.direct_train: # j 0 (bear), 1(No Move) ,2 (bull)
        current = 0  # 0: 매도 상태, 1: 매수 상태
        for i, j in enumerate(df['Regime']):
            if j == 0:
                df.loc[i, 'Regime_Change'] = 1 if current == 1 else 0
                current = 0
            elif j == 1:
                df.loc[i, 'Regime_Change'] = 0
            elif j == 2:
                if current == 0:
                    df.loc[i, 'Regime_Change'] = 1
                    current = 1
                else:
                    df.loc[i, 'Regime_Change'] = 0

    elif config is None or config.num_clusters == 2:
        # 레짐 변화 감지 (거래 발생)
        df['Regime_Change'] = df['Regime'].diff().fillna(0) != 0
        # 첫 번째 진입도 거래로 간주
        df.loc[0, 'Regime_Change'] = df.loc[0, 'Regime'] == 1
    elif config.num_clusters == 3:
        # 3개의 레짐인 경우, Bull, Neutral이면 매수 포지션 유지 Bear는 매도
        df['Regime_Change'] = (df['Regime'] == 0) & (df['Regime'].shift(1) == 1) | (df['Regime'] == 1) & (df['Regime'].shift(1) == 0) | (df['Regime'] == 2) & (df['Regime'].shift(1) == 0) | (df['Regime'] == 0) & (df['Regime'].shift(1) == 2)
        # 첫 번째 진입도 거래로 간주
        df.loc[0, 'Regime_Change'] = df.loc[0, 'Regime'] == 1 or df.loc[0, 'Regime'] == 2


    # 거래 비용 계산 (레짐이 변할 때마다 적용)
    df['Transaction_Cost'] = np.where(df['Regime_Change'], transaction_cost * 100, 0)

    # 다음날 반영 방식으로 수정
    df['Strategy_Regime'] = df['Regime'].shift(1).fillna(0)  # 첫날은 포지션 없음
    df['Strategy_Return'] = df['Strategy_Regime'] * df['Return'] - df['Transaction_Cost']

    # 누적 수익률 계산
    df['Cum_Market'] = (1 + df['Return']/100).cumprod() - 1
    df['Cum_Strategy'] = (1 + df['Strategy_Return']/100).cumprod() - 1

    # 기본 통계량
    market_return = df['Cum_Market'].iloc[-1] * 100
    strategy_return = df['Cum_Strategy'].iloc[-1] * 100

    # 매수 비율
    long_ratio = df['Regime'].mean() * 100

    # 거래 횟수
    n_trades = df['Regime_Change'].sum()

    # 총 거래 비용
    total_cost = df['Transaction_Cost'].sum()

    print(f"시장 누적 수익률: {market_return:.2f}%")
    print(f"전략 누적 수익률 (거래 비용 포함): {strategy_return:.2f}%")
    print(f"롱 포지션 비율: {long_ratio:.2f}%")
    print(f"총 거래 횟수: {n_trades}")
    print(f"총 거래 비용: {total_cost:.2f}%")

    # 차트 그리기
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(df['Cum_Market'] * 100, label='Market', color='gray')
    plt.plot(df['Cum_Strategy'] * 100, label='Regime Strategy (incl. costs)', color='blue')
    plt.legend()
    plt.title('Cumulative Return Comparison')
    plt.ylabel('Return (%)')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    if config == None or config.num_clusters == 2:
        plt.plot(df['Regime'], label='Regime (1=Bull, 0=Bear)', color='red')
    elif config.num_clusters == 3:
        plt.plot(df['Regime'], label='Regime (0=Bear, 1=Bull, 2=Neutral)', color='red')
    plt.title('Regime Signal')
    plt.ylabel('Regime')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.bar(range(len(df['Transaction_Cost'])), df['Transaction_Cost'], color='orange', alpha=0.7)
    plt.title('Transaction Costs')
    plt.ylabel('Cost (%)')
    plt.xlabel('Trading Days')
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    # 추가 성과 지표 계산
    # 연율화된 수익률 (연간 거래일 252일 가정)
    days = len(df)
    years = days / 252

    market_annual_return = ((1 + market_return/100) ** (1/years) - 1) * 100
    strategy_annual_return = ((1 + strategy_return/100) ** (1/years) - 1) * 100

    # 최대 낙폭 (Maximum Drawdown)
    df['Market_Peak'] = df['Cum_Market'].cummax()
    df['Strategy_Peak'] = df['Cum_Strategy'].cummax()

    df['Market_Drawdown'] = (df['Cum_Market'] - df['Market_Peak']) / (1 + df['Market_Peak']) * 100
    df['Strategy_Drawdown'] = (df['Cum_Strategy'] - df['Strategy_Peak']) / (1 + df['Strategy_Peak']) * 100

    market_max_drawdown = df['Market_Drawdown'].min()
    strategy_max_drawdown = df['Strategy_Drawdown'].min()

    # 샤프 비율 계산 (무위험 수익률 2% 가정)
    risk_free_rate = 0.02
    market_daily_returns = df['Return'] / 100
    strategy_daily_returns = df['Strategy_Return'] / 100

    market_volatility = market_daily_returns.std() * np.sqrt(252)
    strategy_volatility = strategy_daily_returns.std() * np.sqrt(252)

    market_sharpe = (market_annual_return/100 - risk_free_rate) / market_volatility
    strategy_sharpe = (strategy_annual_return/100 - risk_free_rate) / strategy_volatility

    print("\n추가 성과 지표:")
    print(f"연율화 시장 수익률: {market_annual_return:.2f}%")
    print(f"연율화 전략 수익률: {strategy_annual_return:.2f}%")
    print(f"시장 최대 낙폭: {market_max_drawdown:.2f}%")
    print(f"전략 최대 낙폭: {strategy_max_drawdown:.2f}%")
    print(f"시장 연율화 변동성: {market_volatility*100:.2f}%")
    print(f"전략 연율화 변동성: {strategy_volatility*100:.2f}%")
    print(f"시장 샤프 비율: {market_sharpe:.2f}")
    print(f"전략 샤프 비율: {strategy_sharpe:.2f}")

    # 성과 평가 결과 저장
    performance = {
        'cumulative_returns': {
            'market': market_return,
            'strategy': strategy_return,
        },
        'annual_returns': {
            'market': market_annual_return,
            'strategy': strategy_annual_return,
        },
        'max_drawdown': {
            'market': market_max_drawdown,
            'strategy': strategy_max_drawdown,
        },
        'volatility': {
            'market': market_volatility * 100,
            'strategy': strategy_volatility * 100,
        },
        'sharpe_ratio': {
            'market': market_sharpe,
            'strategy': strategy_sharpe,
        },
        'trading_metrics': {
            'long_ratio': long_ratio,
            'number_of_trades': int(n_trades),
            'total_transaction_cost': total_cost,
        }
    }

    return df, performance

def analyze_transaction_cost_impact(model, valid_loader, test_loader, config, kmeans, bull_regime, bear_regime=None, save_path=None):
    """
    다양한 거래 비용 수준에서 전략 성과를 비교 분석

    Args:
        model: 평가할 모델
        valid_loader: 검증 데이터 로더
        test_loader: 테스트 데이터 로더
        config: 설정 객체
        kmeans: 훈련된 KMeans 모델
        bull_regime: Bull 레짐 클러스터 ID
        save_path: 결과 그래프 저장 경로 (None이면 저장하지 않음)

    Returns:
        cost_df: 비용 분석 결과 데이터프레임
    """
    from .clustering import predict_regimes
    
    # 테스트 데이터에 대한 레짐 예측
    test_predictions, test_returns, test_dates = predict_regimes(
        model, test_loader, kmeans, bull_regime, config, bear_regime=bear_regime
    )

    # 다양한 거래 비용 수준
    cost_levels = [0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]
    results = []

    # 비용 레벨별 성과 평가
    for cost in cost_levels:
        print(f"\n거래 비용 {cost*100:.2f}% 분석 중...")
        _, performance = evaluate_regime_strategy(
            test_predictions, test_returns, test_dates, transaction_cost=cost
        )

        results.append({
            'cost': cost * 100,  # 백분율로 변환
            'return': performance['cumulative_returns']['strategy'],
            'annual_return': performance['annual_returns']['strategy'],
            'max_drawdown': performance['max_drawdown']['strategy'],
            'sharpe': performance['sharpe_ratio']['strategy'],
            'trades': performance['trading_metrics']['number_of_trades'],
            'total_cost': performance['trading_metrics']['total_transaction_cost']
        })

    # 결과를 데이터프레임으로 변환
    cost_df = pd.DataFrame(results)

    # 차트 그리기
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.plot(cost_df['cost'], cost_df['return'], 'o-', linewidth=2)
    plt.title('Cumulative Return vs Transaction Cost')
    plt.xlabel('Transaction Cost (%)')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(cost_df['cost'], cost_df['sharpe'], 'o-', color='green', linewidth=2)
    plt.title('Sharpe Ratio vs Transaction Cost')
    plt.xlabel('Transaction Cost (%)')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(cost_df['cost'], cost_df['max_drawdown'], 'o-', color='red', linewidth=2)
    plt.title('Maximum Drawdown vs Transaction Cost')
    plt.xlabel('Transaction Cost (%)')
    plt.ylabel('Maximum Drawdown (%)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(cost_df['cost'], cost_df['total_cost'], 'o-', color='orange', linewidth=2)
    plt.title('Total Cost vs Transaction Cost Rate')
    plt.xlabel('Transaction Cost Rate (%)')
    plt.ylabel('Total Cost (%)')
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    # 손익분기점 거래 비용 분석 (시장 대비 초과 수익이 0이 되는 지점)
    market_return = test_returns.mean() * len(test_returns)

    # 손익분기점 계산을 위한 함수
    # 수익률과 비용의 관계를 보간
    if len(cost_df) > 2:  # 보간을 위해 최소 3개 이상의 데이터 포인트 필요
        f = interp1d(cost_df['return'], cost_df['cost'], kind='linear', fill_value='extrapolate')
        breakeven_cost = float(f(market_return))
        print(f"\n손익분기점 거래 비용 (시장 대비): {breakeven_cost:.4f}%")

    return cost_df

def visualize_all_periods_performance(period_performances, save_dir):
    """
    모든 기간에 대한 성과 시각화

    Args:
        period_performances: 모든 기간의 성과 정보 리스트
        save_dir: 저장 디렉토리
    """
    # 성과 데이터 추출
    periods = [p['period'] for p in period_performances]
    market_returns = [p['cumulative_returns']['market'] for p in period_performances]
    strategy_returns = [p['cumulative_returns']['strategy'] for p in period_performances]
    market_sharpes = [p['sharpe_ratio']['market'] for p in period_performances]
    strategy_sharpes = [p['sharpe_ratio']['strategy'] for p in period_performances]

    # 기간별 수익률 비교
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    width = 0.35
    x = np.arange(len(periods))
    plt.bar(x - width/2, market_returns, width, label='Market')
    plt.bar(x + width/2, strategy_returns, width, label='Regime Strategy')
    plt.xlabel('Period')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Comparison of Cumulative Returns by Period')
    plt.xticks(x, periods)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, market_sharpes, width, label='Market')
    plt.bar(x + width/2, strategy_sharpes, width, label='Regime Strategy')
    plt.xlabel('Period')
    plt.ylabel('Sharpe Ratio')
    plt.title('Comparison of Sharpe Ratios by Period')
    plt.xticks(x, periods)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_periods_comparison.png")

    # 누적 성과 계산
    cumulative_df = pd.DataFrame({
        'period': periods,
        'market_return': market_returns,
        'strategy_return': strategy_returns
    })

    cumulative_df['cum_market'] = (1 + cumulative_df['market_return']/100).cumprod() - 1
    cumulative_df['cum_strategy'] = (1 + cumulative_df['strategy_return']/100).cumprod() - 1

    # 누적 성과 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_df['period'], cumulative_df['cum_market'] * 100, 'o-', label='Market', color='gray')
    plt.plot(cumulative_df['period'], cumulative_df['cum_strategy'] * 100, 'o-', label='Regime Strategy', color='blue')
    plt.xlabel('Period')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Total Cumulative Performance Across All Periods')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/total_cumulative_performance.png")

    # 전체 성과 요약 저장
    total_market_return = (1 + np.array(market_returns)/100).prod() - 1
    total_strategy_return = (1 + np.array(strategy_returns)/100).prod() - 1

    summary = {
        'total_periods': len(periods),
        'total_market_return': total_market_return * 100,
        'total_strategy_return': total_strategy_return * 100,
        'avg_market_sharpe': np.mean(market_sharpes),
        'avg_strategy_sharpe': np.mean(strategy_sharpes),
        'win_rate': sum(np.array(strategy_returns) > np.array(market_returns)) / len(periods) * 100
    }

    with open(f"{save_dir}/total_performance_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)

    print("\n===== Overall Backtest Performance Summary =====")
    print(f"Total Periods: {summary['total_periods']} periods")
    print(f"Total Market Return: {summary['total_market_return']:.2f}%")
    print(f"Total Strategy Return: {summary['total_strategy_return']:.2f}%")
    print(f"Average Market Sharpe Ratio: {summary['avg_market_sharpe']:.2f}")
    print(f"Average Strategy Sharpe Ratio: {summary['avg_strategy_sharpe']:.2f}")
    print(f"Win Rate (vs Market): {summary['win_rate']:.2f}%")
