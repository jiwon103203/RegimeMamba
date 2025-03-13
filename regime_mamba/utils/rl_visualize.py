"""
Visualization functions for RL-based investment strategies.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def visualize_rl_results(results_df, performance, window_number, title, save_path):
    """
    Visualize RL investment results.
    
    Args:
        results_df: Results DataFrame
        performance: Performance metrics
        window_number: Window number
        title: Chart title
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 12))
    
    # Plot cumulative returns
    plt.subplot(3, 1, 1)
    plt.plot(results_df['Date'], results_df['Cum_Market'] * 100, label='Buy and Hold', color='grey')
    plt.plot(results_df['Date'], results_df['Cum_Strategy'] * 100, label='RL Strategy', color='blue')
    plt.title(f"{title}\nStrategy: {performance['total_returns']['strategy']*100:.2f}% vs Market: {performance['total_returns']['market']*100:.2f}%")
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True)
    plt.legend()
    
    # Plot positions
    plt.subplot(3, 1, 2)
    plt.plot(results_df['Date'], results_df['Position'], color='red', label='Position (-1=Short, 0=Neutral, 1=Long)')
    plt.title(f"Trading Positions (Changes: {performance['position_changes']})")
    plt.ylabel('Position')
    plt.ylim([-1.1, 1.1])
    plt.grid(True)
    plt.legend()
    
    # Plot market returns with buy/sell points
    plt.subplot(3, 1, 3)
    
    # Plot market returns
    plt.plot(results_df['Date'], results_df['Market_Return'] * 100, color='grey', alpha=0.5, label='Daily Returns')
    
    # Find where positions change
    position_changes = np.diff(results_df['Position'], prepend=0)
    
    # Buy signals (from 0 or -1 to 1)
    buy_points = results_df[position_changes == 1]
    plt.scatter(buy_points['Date'], buy_points['Market_Return'] * 100, color='green', marker='^', s=100, label='Buy')
    
    # Sell signals (from 1 to 0 or -1)
    sell_points = results_df[position_changes == -1]
    plt.scatter(sell_points['Date'], sell_points['Market_Return'] * 100, color='red', marker='v', s=100, label='Sell')
    
    # Short signals (from 0 to -1)
    short_points = results_df[(position_changes == -1) & (results_df['Position'] == -1)]
    plt.scatter(short_points['Date'], short_points['Market_Return'] * 100, color='purple', marker='v', s=100, label='Short')
    
    # Cover signals (from -1 to 0 or 1)
    cover_points = results_df[(position_changes == 1) & (results_df['Position'].shift(1) == -1)]
    plt.scatter(cover_points['Date'], cover_points['Market_Return'] * 100, color='orange', marker='^', s=100, label='Cover')
    
    plt.title('Buy/Sell Signals')
    plt.ylabel('Daily Return (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_all_results(all_performances, save_dir):
    """
    Visualize results across all windows.
    
    Args:
        all_performances: List of performance metrics from all windows
        save_dir: Directory to save visualizations
    """
    # Extract metrics
    windows = [p['window'] for p in all_performances]
    market_returns = [p['total_returns']['market'] * 100 for p in all_performances]
    strategy_returns = [p['total_returns']['strategy'] * 100 for p in all_performances]
    market_sharpes = [p['sharpe_ratio']['market'] for p in all_performances]
    strategy_sharpes = [p['sharpe_ratio']['strategy'] for p in all_performances]
    position_changes = [p['position_changes'] for p in all_performances]
    max_drawdowns = [p['max_drawdown'] * 100 for p in all_performances]
    
    # Create visualization
    plt.figure(figsize=(15, 12))
    
    # 1. Returns comparison
    plt.subplot(2, 2, 1)
    width = 0.35
    x = np.arange(len(windows))
    plt.bar(x - width/2, market_returns, width, label='Buy and Hold', color='gray')
    plt.bar(x + width/2, strategy_returns, width, label='RL Strategy', color='blue')
    plt.xlabel('Window')
    plt.ylabel('Total Return (%)')
    plt.title('Window Returns Comparison')
    plt.xticks(x, windows)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Sharpe ratio comparison
    plt.subplot(2, 2, 2)
    plt.bar(x - width/2, market_sharpes, width, label='Buy and Hold', color='gray')
    plt.bar(x + width/2, strategy_sharpes, width, label='RL Strategy', color='blue')
    plt.xlabel('Window')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison')
    plt.xticks(x, windows)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Position changes and drawdowns
    plt.subplot(2, 2, 3)
    plt.bar(x, position_changes, color='red')
    plt.xlabel('Window')
    plt.ylabel('Position Changes')
    plt.title('Trading Activity')
    plt.xticks(x, windows)
    plt.grid(True, alpha=0.3)
    
    # 4. Maximum drawdowns
    plt.subplot(2, 2, 4)
    plt.bar(x, max_drawdowns, color='purple')
    plt.xlabel('Window')
    plt.ylabel('Maximum Drawdown (%)')
    plt.title('Risk Assessment')
    plt.xticks(x, windows)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_windows_comparison.png")
    plt.close()
    
    # Create summary visualization
    plt.figure(figsize=(10, 6))
    labels = ['Total Return (%)', 'Avg Sharpe', 'Avg Max Drawdown (%)']
    
    # Calculate summary metrics
    total_market_return = sum(market_returns)
    total_strategy_return = sum(strategy_returns)
    avg_market_sharpe = np.mean(market_sharpes)
    avg_strategy_sharpe = np.mean(strategy_sharpes)
    avg_max_drawdown = np.mean(max_drawdowns)
    
    market_metrics = [total_market_return, avg_market_sharpe, avg_max_drawdown]
    strategy_metrics = [total_strategy_return, avg_strategy_sharpe, avg_max_drawdown]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, market_metrics, width, label='Buy and Hold', color='gray')
    plt.bar(x + width/2, strategy_metrics, width, label='RL Strategy', color='blue')
    
    plt.ylabel('Value')
    plt.title('Strategy Performance Summary')
    plt.xticks(x, labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_summary.png")
    plt.close()

def visualize_training_history(history, save_path):
    """
    Visualize training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save visualization
    """
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['value_loss'], label='Value Loss')
    plt.plot(history['policy_loss'], label='Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.grid(True)
    plt.legend()
    
    # Plot entropy
    plt.subplot(2, 2, 2)
    plt.plot(history['entropy'], color='green')
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy')
    plt.grid(True)
    
    # Plot rewards
    plt.subplot(2, 2, 3)
    plt.plot(history['rewards'], label='Average Reward')
    plt.plot(history['returns'], label='Average Return')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Rewards and Returns')
    plt.grid(True)
    plt.legend()
    
    # Plot NAV
    plt.subplot(2, 2, 4)
    plt.plot(history['nav'], color='blue')
    plt.xlabel('Episode')
    plt.ylabel('NAV')
    plt.title('Net Asset Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_rolling_sharpe_comparison(results_df, window_size=252, save_path=None):
    """
    Plot rolling Sharpe ratio comparison.
    
    Args:
        results_df: Results DataFrame with strategy and market returns
        window_size: Window size for rolling calculation
        save_path: Path to save visualization
    """
    # Calculate rolling metrics
    rolling_market_mean = results_df['Market_Return'].rolling(window=window_size).mean()
    rolling_market_std = results_df['Market_Return'].rolling(window=window_size).std()
    rolling_strategy_mean = results_df['Strategy_Return'].rolling(window=window_size).mean()
    rolling_strategy_std = results_df['Strategy_Return'].rolling(window=window_size).std()
    
    # Calculate rolling Sharpe ratios
    rolling_market_sharpe = rolling_market_mean / (rolling_market_std + 1e-6)
    rolling_strategy_sharpe = rolling_strategy_mean / (rolling_strategy_std + 1e-6)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Date'], rolling_market_sharpe, label='Buy and Hold', color='gray')
    plt.plot(results_df['Date'], rolling_strategy_sharpe, label='RL Strategy', color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Rolling Sharpe Ratio (Window: {window_size} days)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
