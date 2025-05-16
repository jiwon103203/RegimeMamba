import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json

def evaluate_regime_strategy(predictions, returns, dates=None, transaction_cost=0.001, save_path=None, config=None):
    """
    Evaluate performance of regime-based strategy considering transaction costs

    Args:
        predictions: Predicted regimes (1=Bull, 0=Bear)
        returns: Actual returns
        dates: Date information
        transaction_cost: Transaction cost (percentage, e.g., 0.001 = 0.1%)
        save_path: Path to save results graph (None if not saving)

    Returns:
        df: Detailed results dataframe
        performance: Performance metrics dictionary
    """
    if len(predictions) == 0 or len(returns)==0:
      raise ValueError("Empty predictions or returns array was passed.")
    
    df = pd.DataFrame({
        'Date': dates,
        'Regime': predictions.flatten() if isinstance(predictions, np.ndarray) else predictions,
        'Return': returns.flatten() if isinstance(predictions, np.ndarray) else returns,
    })

    # Sort by Date
    df.sort_values('Date').reset_index(drop=True, inplace=True)

    if config is not None and config.direct_train: # j 0 (bear), 1(No Move) ,2 (bull)
        current = 0  # 0: Sell state, 1: Buy state
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

    elif config.n_clusters == 3:
        # For 3 regimes, maintain buy position for Bull and Neutral, sell for Bear
        df['Regime_Change'] = (df['Regime'] == 0) & (df['Regime'].shift(1) == 1) | (df['Regime'] == 1) & (df['Regime'].shift(1) == 0) | (df['Regime'] == 2) & (df['Regime'].shift(1) == 0) | (df['Regime'] == 0) & (df['Regime'].shift(1) == 2)
        # First entry also counts as a trade
        df.loc[0, 'Regime_Change'] = df.loc[0, 'Regime'] == 1 or df.loc[0, 'Regime'] == 2
    
    else:
        # Detect regime changes (when trades occur)
        df['Regime_Change'] = df['Regime'].diff().fillna(0) != 0
        # First entry also counts as a trade
        df.loc[0, 'Regime_Change'] = df.loc[0, 'Regime'] == 1


    # Calculate transaction costs (applied whenever regime changes)
    if config is not None and config.input_dim == 4:
        df['Transaction_Cost'] = np.where(df['Regime_Change'], transaction_cost * 100, 0)
    else:
        df['Transaction_Cost'] = np.where(df['Regime_Change'], transaction_cost, 0)

    # Modified to apply next day
    df['Strategy_Regime'] = df['Regime'].shift(1).fillna(0)  # No position on first day
    df['Strategy_Return'] = df['Strategy_Regime'] * df['Return'] - df['Transaction_Cost']

    # Calculate cumulative returns
    if config is not None and config.input_dim == 4:
        df['Cum_Market'] = (1 + df['Return']/100).cumprod() - 1
        df['Cum_Strategy'] = (1 + df['Strategy_Return']/100).cumprod() - 1
    else:
        df['Cum_Market'] = (1 + df['Return']).cumprod() - 1
        df['Cum_Strategy'] = (1 + df['Strategy_Return']).cumprod() - 1

    # Basic statistics
    market_return = df['Cum_Market'].iloc[-1] * 100
    strategy_return = df['Cum_Strategy'].iloc[-1] * 100

    # Long ratio
    long_ratio = df['Regime'].mean() * 100

    # Number of trades
    n_trades = df['Regime_Change'].sum()

    # Total transaction cost
    total_cost = df['Transaction_Cost'].sum()

    print(f"Market cumulative return: {market_return:.2f}%")
    print(f"Strategy cumulative return (including transaction costs): {strategy_return:.2f}%")
    print(f"Long position ratio: {long_ratio:.2f}%")
    print(f"Total number of trades: {n_trades}")
    print(f"Total transaction cost: {total_cost:.2f}%")

    # Create chart
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.plot(df['Cum_Market'] * 100, label='Market', color='gray')
    plt.plot(df['Cum_Strategy'] * 100, label='Regime Strategy (incl. costs)', color='blue')
    plt.legend()
    plt.title('Cumulative Return Comparison')
    plt.ylabel('Return (%)')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    if config == None or config.n_clusters == 2:
        plt.plot(df['Regime'], label='Regime (1=Bull, 0=Bear)', color='red')
    elif config.n_clusters == 3:
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

    # Calculate additional performance metrics
    # Annualized returns (assuming 252 trading days per year)
    days = len(df)
    years = days / 252

    market_annual_return = ((1 + market_return/100) ** (1/years) - 1) * 100
    strategy_annual_return = ((1 + strategy_return/100) ** (1/years) - 1) * 100

    # Maximum Drawdown
    df['Market_Peak'] = df['Cum_Market'].cummax()
    df['Strategy_Peak'] = df['Cum_Strategy'].cummax()

    df['Market_Drawdown'] = (df['Cum_Market'] - df['Market_Peak']) / (1 + df['Market_Peak']) * 100
    df['Strategy_Drawdown'] = (df['Cum_Strategy'] - df['Strategy_Peak']) / (1 + df['Strategy_Peak']) * 100

    market_max_drawdown = df['Market_Drawdown'].min()
    strategy_max_drawdown = df['Strategy_Drawdown'].min()

    # Sharpe ratio calculation (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    if config is not None and config.input_dim == 4:
        market_daily_returns = df['Return'] / 100
        strategy_daily_returns = df['Strategy_Return'] /100
    else:
        market_daily_returns = df['Return']
        strategy_daily_returns = df['Strategy_Return']

    market_volatility = market_daily_returns.std() * np.sqrt(252)
    strategy_volatility = strategy_daily_returns.std() * np.sqrt(252)

    market_sharpe = (market_annual_return/100 - risk_free_rate) / market_volatility
    strategy_sharpe = (strategy_annual_return/100 - risk_free_rate) / strategy_volatility

    print("\nAdditional performance metrics:")
    print(f"Annualized market return: {market_annual_return:.2f}%")
    print(f"Annualized strategy return: {strategy_annual_return:.2f}%")
    print(f"Market maximum drawdown: {market_max_drawdown:.2f}%")
    print(f"Strategy maximum drawdown: {strategy_max_drawdown:.2f}%")
    print(f"Market annualized volatility: {market_volatility*100:.2f}%")
    print(f"Strategy annualized volatility: {strategy_volatility*100:.2f}%")
    print(f"Market Sharpe ratio: {market_sharpe:.2f}")
    print(f"Strategy Sharpe ratio: {strategy_sharpe:.2f}")

    # Save performance evaluation results
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
    Analyze strategy performance under various transaction cost levels

    Args:
        model: Model to evaluate
        valid_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration object
        kmeans: Trained KMeans model
        bull_regime: Bull regime cluster ID
        save_path: Path to save results (None if not saving)

    Returns:
        cost_df: Transaction cost analysis results dataframe
    """
    from .clustering import predict_regimes
    
    # Predict regimes for test data
    test_predictions, test_returns, test_dates = predict_regimes(
        model, test_loader, kmeans, bull_regime, config, bear_regime=bear_regime
    )

    # Various transaction cost levels
    cost_levels = [0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]
    results = []

    # Evaluate performance at each cost level
    for cost in cost_levels:
        print(f"\nAnalyzing transaction cost {cost*100:.2f}%...")
        _, performance = evaluate_regime_strategy(
            test_predictions, test_returns, test_dates, transaction_cost=cost
        )

        results.append({
            'cost': cost * 100,  # Convert to percentage
            'return': performance['cumulative_returns']['strategy'],
            'annual_return': performance['annual_returns']['strategy'],
            'max_drawdown': performance['max_drawdown']['strategy'],
            'sharpe': performance['sharpe_ratio']['strategy'],
            'trades': performance['trading_metrics']['number_of_trades'],
            'total_cost': performance['trading_metrics']['total_transaction_cost']
        })

    # Convert results to dataframe
    cost_df = pd.DataFrame(results)

    # Create chart
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

    # Breakeven transaction cost analysis (point where excess return over market becomes zero)
    market_return = test_returns.mean() * len(test_returns)

    # Function for calculating breakeven point
    # Interpolate relationship between return and cost
    if len(cost_df) > 2:  # Need at least 3 data points for interpolation
        f = interp1d(cost_df['return'], cost_df['cost'], kind='linear', fill_value='extrapolate')
        breakeven_cost = float(f(market_return))
        print(f"\nBreakeven transaction cost (vs market): {breakeven_cost:.4f}%")

    return cost_df

def visualize_all_periods_performance(period_performances, save_dir):
    """
    Visualize performance across all periods

    Args:
        period_performances: List of performance metrics for all periods
        save_dir: Directory to save visualizations
    """
    # Extract performance data
    periods = [p['period'] for p in period_performances]
    market_returns = [p['cumulative_returns']['market'] for p in period_performances]
    strategy_returns = [p['cumulative_returns']['strategy'] for p in period_performances]
    market_sharpes = [p['sharpe_ratio']['market'] for p in period_performances]
    strategy_sharpes = [p['sharpe_ratio']['strategy'] for p in period_performances]

    # Compare returns by period
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

    # Calculate cumulative performance
    cumulative_df = pd.DataFrame({
        'period': periods,
        'market_return': market_returns,
        'strategy_return': strategy_returns
    })

    cumulative_df['cum_market'] = (1 + cumulative_df['market_return']/100).cumprod() - 1
    cumulative_df['cum_strategy'] = (1 + cumulative_df['strategy_return']/100).cumprod() - 1

    # Visualize cumulative performance
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

    # Save performance summary
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