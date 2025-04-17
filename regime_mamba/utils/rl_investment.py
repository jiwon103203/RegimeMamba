"""
Main module for RL-based investment strategies.
Orchestrates the training, evaluation, and visualization of RL agents
for financial trading with a rolling window approach.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import torch

from ..train.rl_train import train_rl_agent_for_window, save_training_results
from ..evaluate.rl_evaluate import evaluate_rl_agent, calculate_overall_performance
from .rl_visualize import visualize_rl_results, visualize_all_results, visualize_training_history

def run_rl_investment(config):
    """
    Run RL-based investment with rolling windows.
    
    Args:
        config: Configuration object
        
    Returns:
        tuple: (combined_results, all_performances, training_histories)
    """
    # Load data
    print("Loading data...")
    data = pd.read_csv(config.data_path)
    data = data.iloc[2:]  # Skip first 2 rows
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    
    # Ensure 'Date' column exists
    if 'Date' not in data.columns:
        if 'Price' in data.columns:
            data = data.rename(columns={'Price': 'Date'})
        else:
            raise ValueError("Data must contain a 'Date' or 'Price' column")
    
    # Convert percentage values
    data['returns'] = data['returns'] * 100
    data["dd_10"] = data["dd_10"] * 100
    data["sortino_20"] = data["sortino_20"] * 100
    data["sortino_60"] = data["sortino_60"] * 100
    
    # Storage for results
    all_results = []
    all_performances = []
    training_histories = []
    
    # Parse dates
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    
    window_number = 1
    
    # Main loop for rolling windows
    while current_date <= end_date:
        print(f"\n=== Processing Window {window_number} ===")
        
        # Calculate periods
        train_start = (current_date - relativedelta(years=config.total_window_years)).strftime('%Y-%m-%d')
        train_end = (current_date - relativedelta(years=config.valid_years)).strftime('%Y-%m-%d')
        
        valid_start = train_end
        valid_end = current_date.strftime('%Y-%m-%d')
        
        forward_start = current_date.strftime('%Y-%m-%d')
        forward_end = (current_date + relativedelta(months=config.forward_months)).strftime('%Y-%m-%d')
        
        print(f"Training period: {train_start} ~ {train_end} ({config.train_years} years)")
        print(f"Validation period: {valid_start} ~ {valid_end} ({config.valid_years} years)")
        print(f"Forward testing period: {forward_start} ~ {forward_end} ({config.forward_months/12:.1f} years)")
        
        # 1. Train RL agent
        agent, model, history = train_rl_agent_for_window(
            config, train_start, train_end, valid_start, valid_end, data
        )
        
        # If training failed, move to next window
        if agent is None:
            print("Agent training failed, moving to next window.")
            current_date += relativedelta(months=config.forward_months)
            window_number += 1
            continue
        
        # Save training results
        save_training_results(agent, model, history, config.results_dir, window_number)
        
        # Create visualization of training history
        history_path = os.path.join(config.results_dir, f"window_{window_number}_training_history.png")
        visualize_training_history(history, history_path)
        
        # 2. Evaluate on forward testing period
        results_df, performance = evaluate_rl_agent(
            config, agent, data, forward_start, forward_end, window_number
        )
        
        # If evaluation succeeded, store results
        if results_df is not None and performance is not None:
            all_results.append(results_df)
            all_performances.append(performance)
            training_histories.append(history)
            
            # Save results to file
            results_df.to_csv(
                f"{config.results_dir}/window_{window_number}_results.csv",
                index=False
            )
            
            # Visualize results
            visualize_rl_results(
                results_df,
                performance,
                window_number,
                f"Window {window_number}: {forward_start} ~ {forward_end}",
                f"{config.results_dir}/window_{window_number}_performance.png"
            )
        
        # Move to next window
        current_date += relativedelta(months=config.forward_months)
        window_number += 1
    
    # Combine and save all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{config.results_dir}/all_windows_results.csv", index=False)
        
        # Save training histories
        with open(f"{config.results_dir}/training_histories.json", 'w') as f:
            json.dump(training_histories, f, default=lambda o: str(o) if isinstance(o, np.ndarray) else o, indent=4)
        
        # Save performance metrics
        with open(f"{config.results_dir}/all_performances.json", 'w') as f:
            json.dump(all_performances, f, default=lambda o: o.isoformat() if isinstance(o, datetime) else o, indent=4)
        
        # Calculate and save overall performance summary
        overall_performance = calculate_overall_performance(all_performances)
        with open(f"{config.results_dir}/performance_summary.json", 'w') as f:
            json.dump(overall_performance, f, indent=4)
        
        # Print summary
        print("\n===== Overall Performance Summary =====")
        print(f"Total Windows: {overall_performance['total_windows']}")
        print(f"Total Market Return: {overall_performance['total_returns']['market']:.2f}%")
        print(f"Total Strategy Return: {overall_performance['total_returns']['strategy']:.2f}%")
        print(f"Return Difference: {overall_performance['total_returns']['difference']:.2f}%")
        print(f"Average Market Sharpe: {overall_performance['sharpe_ratios']['market_avg']:.4f}")
        print(f"Average Strategy Sharpe: {overall_performance['sharpe_ratios']['strategy_avg']:.4f}")
        print(f"Sharpe Ratio Difference: {overall_performance['sharpe_ratios']['difference']:.4f}")
        print(f"Average Position Changes: {overall_performance['position_changes_avg']:.2f}")
        print(f"Average Maximum Drawdown: {overall_performance['max_drawdown_avg']:.2f}%")
        print(f"Win Rate: {overall_performance['win_rate']:.2f}%")
        
        # Visualize overall results
        visualize_all_results(all_performances, config.results_dir)
        
        print(f"\nRL investment analysis complete! Processed {len(all_performances)} windows.")
        return combined_results, all_performances, training_histories
    else:
        print("RL investment analysis failed: No valid results.")
        return None, None, None
