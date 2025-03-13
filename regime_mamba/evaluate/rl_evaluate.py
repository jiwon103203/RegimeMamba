"""
Evaluation functions for RL-based investment strategies.
"""

import pandas as pd
import numpy as np
from ..utils.rl_environments import FinancialTradingEnv

def evaluate_rl_agent(config, agent, data, forward_start, forward_end, window_number):
    """
    Evaluate RL agent on future data.
    
    Args:
        config: Configuration object
        agent: Trained RL agent
        data: DataFrame with price and feature data
        forward_start: Forward testing start date
        forward_end: Forward testing end date
        window_number: Window number for tracking
        
    Returns:
        tuple: (results_df, performance_metrics)
    """
    print(f"\nEvaluation period: {forward_start} ~ {forward_end}")
    
    # Filter data for forward testing period
    forward_data = data[(data['Date'] >= forward_start) & (data['Date'] <= forward_end)].copy()
    
    # Ensure we have enough data
    if len(forward_data) < config.seq_len:
        print(f"Warning: Not enough data for evaluation. Got {len(forward_data)} samples.")
        return None, None
    
    # Prepare features and returns
    features = forward_data[['returns', 'dd_10', 'sortino_20', 'sortino_60']].values
    returns = forward_data['returns'].values / 100
    dates = forward_data['Date'].values
    
    # Create environment
    env = FinancialTradingEnv(
        returns=returns,
        features=features,
        dates=dates,
        seq_len=config.seq_len,
        transaction_cost=config.transaction_cost,
        reward_type=config.reward_type,
        position_penalty=config.position_penalty,
        window_size=config.window_size
    )
    
    # Evaluate agent
    results = agent.evaluate(env)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': dates[config.seq_len:],
        'Market_Return': returns[config.seq_len:],
        'Strategy_Return': results['strategy_returns'],
        'Position': results['positions'],
        'NAV': results['navs'],
        'Market_NAV': results['market_nav']
    })
    
    # Add cumulative returns
    results_df['Cum_Market'] = (1 + results_df['Market_Return']).cumprod() - 1
    results_df['Cum_Strategy'] = (1 + results_df['Strategy_Return']).cumprod() - 1
    
    # Calculate performance metrics
    performance = {
        'window': window_number,
        'forward_start': forward_start,
        'forward_end': forward_end,
        'total_returns': {
            'market': results_df['Cum_Market'].iloc[-1],
            'strategy': results_df['Cum_Strategy'].iloc[-1]
        },
        'sharpe_ratio': {
            'market': results['market_sharpe'],
            'strategy': results['sharpe_ratio']
        },
        'max_drawdown': results['max_drawdown'],
        'position_changes': results['position_changes'],
        'avg_position': np.mean(results['positions'])
    }
    
    return results_df, performance

def calculate_overall_performance(all_performances):
    """
    Calculate overall performance metrics across all windows.
    
    Args:
        all_performances: List of performance metrics from all windows
        
    Returns:
        dict: Overall performance summary
    """
    # Extract metrics
    windows = [p['window'] for p in all_performances]
    market_returns = [p['total_returns']['market'] * 100 for p in all_performances]
    strategy_returns = [p['total_returns']['strategy'] * 100 for p in all_performances]
    market_sharpes = [p['sharpe_ratio']['market'] for p in all_performances]
    strategy_sharpes = [p['sharpe_ratio']['strategy'] for p in all_performances]
    position_changes = [p['position_changes'] for p in all_performances]
    max_drawdowns = [p['max_drawdown'] * 100 for p in all_performances]
    
    # Calculate summary statistics
    total_market_return = sum(market_returns)
    total_strategy_return = sum(strategy_returns)
    avg_market_sharpe = np.mean(market_sharpes)
    avg_strategy_sharpe = np.mean(strategy_sharpes)
    avg_position_changes = np.mean(position_changes)
    avg_max_drawdown = np.mean(max_drawdowns)
    win_rate = sum(np.array(strategy_returns) > np.array(market_returns)) / len(windows) * 100
    
    # Create summary
    summary = {
        'total_windows': len(windows),
        'total_returns': {
            'market': total_market_return,
            'strategy': total_strategy_return,
            'difference': total_strategy_return - total_market_return
        },
        'sharpe_ratios': {
            'market_avg': avg_market_sharpe,
            'strategy_avg': avg_strategy_sharpe,
            'difference': avg_strategy_sharpe - avg_market_sharpe
        },
        'position_changes_avg': avg_position_changes,
        'max_drawdown_avg': avg_max_drawdown,
        'win_rate': win_rate
    }
    
    return summary
