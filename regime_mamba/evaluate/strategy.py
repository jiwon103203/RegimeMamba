import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..features import returns_in_percent

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

    if config is not None and config.n_clusters == 3:
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
    if returns_in_percent(config):
        df['Transaction_Cost'] = np.where(df['Regime_Change'], transaction_cost * 100, 0)
    else:
        df['Transaction_Cost'] = np.where(df['Regime_Change'], transaction_cost, 0)

    # Modified to apply next day
    df['Strategy_Regime'] = df['Regime'].shift(1).fillna(0)  # No position on first day
    df['Strategy_Return'] = df['Strategy_Regime'] * df['Return'] - df['Transaction_Cost']

    # Calculate cumulative returns
    if returns_in_percent(config):
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
    if returns_in_percent(config):
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
