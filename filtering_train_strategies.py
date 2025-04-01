import os
import json
import yaml
import logging
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
from typing import Dict, List, Tuple, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

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


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Comparing Smoothing Techniques (Rolling Window Training)')
    
    # Main configuration arguments
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--data_path', type=str, required=True, help='Data file path')
    parser.add_argument('--results_dir', type=str, default='./smoothing_train_results', help='Results directory')
    parser.add_argument('--start_date', type=str, default='2000-01-01', help='Start date')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date')
    parser.add_argument('--preprocessed', type=bool, default=False, help='Whether data is preprocessed')
    
    # Period-related settings
    parser.add_argument('--total_window_years', type=int, default=40, help='Total data period (years)')
    parser.add_argument('--train_years', type=int, default=20, help='Training period (years)')
    parser.add_argument('--valid_years', type=int, default=10, help='Validation period (years)')
    parser.add_argument('--clustering_years', type=int, default=10, help='Clustering period (years)')
    parser.add_argument('--forward_months', type=int, default=60, help='Interval to next window (months)')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=4, help='Number of input variables')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=128, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--target_type', type=str, help='Target type')
    parser.add_argument('--target_horizon', type=int, help='Target horizon')
    parser.add_argument('--cluster_method', type=str, default='cosine_kmeans', help='Clustering method')
    parser.add_argument('--direct_train', action='store_true', help='Train model directly for clasification')
    parser.add_argument('--vae', action='store_true', help='Train model with VAE')
    
    # Training-related settings
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_workers', type=int, default=None, help='Maximum number of worker processes')
    
    return parser.parse_args()


def load_config(args) -> RollingWindowTrainConfig:
    """
    Load configuration from file or command-line arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        RollingWindowTrainConfig: Configuration object
    """
    config = RollingWindowTrainConfig()

    # command-line arguments if provided
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    # If config file is provided, load it
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        # Update config from YAML
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config


def default_converter(o):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(o, np.ndarray):
        return o.item() if o.ndim == 0 else o.tolist()
    if isinstance(o, np.float32):
        return o.item() if o.ndim == 0 else o.tolist()
    raise TypeError(f"Type {type(o).__name__} is not serializable")


def apply_smoothing_method(
    raw_predictions: np.ndarray, 
    smoothing_method: str, 
    smoothing_params: Dict[str, Any]
) -> np.ndarray:
    """
    Apply a specific smoothing method to raw predictions
    
    Args:
        raw_predictions: Raw regime predictions
        smoothing_method: Smoothing method name
        smoothing_params: Dictionary of smoothing parameters
        
    Returns:
        np.ndarray: Smoothed predictions
    """
    if smoothing_method == 'none':
        return raw_predictions
    elif smoothing_method == 'ma':
        window = smoothing_params.get('window', 10)
        return apply_regime_smoothing(
            raw_predictions, method='ma', window=window
        ).reshape(-1, 1)
    elif smoothing_method == 'exp':
        window = smoothing_params.get('window', 10)
        return apply_regime_smoothing(
            raw_predictions, method='exp', window=window
        ).reshape(-1, 1)
    elif smoothing_method == 'gaussian':
        window = smoothing_params.get('window', 10)
        return apply_regime_smoothing(
            raw_predictions, method='gaussian', window=window
        ).reshape(-1, 1)
    elif smoothing_method == 'confirmation':
        days = smoothing_params.get('days', 3)
        return apply_confirmation_rule(
            raw_predictions, confirmation_days=days
        ).reshape(-1, 1)
    elif smoothing_method == 'min_holding':
        days = smoothing_params.get('days', 20)
        return apply_minimum_holding_period(
            raw_predictions, min_holding_days=days
        ).reshape(-1, 1)
    else:
        logging.warning(f"Unknown smoothing method: {smoothing_method}, using raw predictions")
        return raw_predictions


def apply_and_evaluate_with_smoothing(
    model, 
    data: pd.DataFrame, 
    kmeans, 
    bull_regime: int, 
    forward_start: str, 
    forward_end: str, 
    smoothing_method: str, 
    config: RollingWindowTrainConfig, 
    smoothing_params: Dict[str, Any]
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Apply a specific smoothing technique and evaluate the regime strategy
    
    Args:
        model: Trained model
        data: Full dataframe
        kmeans: K-Means model
        bull_regime: Bull regime cluster ID
        forward_start: Future period start date
        forward_end: Future period end date
        smoothing_method: Smoothing method name
        config: Configuration object
        smoothing_params: Dictionary of smoothing parameters
        
    Returns:
        Tuple: (results_df, performance) or (None, None) on error
    """
    try:
        # Create future dataset
        forward_dataset = DateRangeRegimeMambaDataset(
            data=data, 
            seq_len=config.seq_len,
            start_date=forward_start,
            end_date=forward_end,
            target_type=config.target_type,
            target_horizon=config.target_horizon,
            preprocessed=config.preprocessed
        )
        
        # Create data loader
        forward_loader = DataLoader(
            forward_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1)
        )
        
        # Check if enough data
        if len(forward_dataset) < 10:
            logging.warning(f"Warning: Not enough data for evaluation ({len(forward_dataset)} samples)")
            return None, None
        
        # Get original regime predictions
        from regime_mamba.evaluate.clustering import predict_regimes
        raw_predictions, true_returns, dates = predict_regimes(model, forward_loader, kmeans, bull_regime, config)
        
        # Apply smoothing
        smoothed_predictions = apply_smoothing_method(raw_predictions, smoothing_method, smoothing_params)
        
        # Evaluate strategy with transaction costs
        results_df, performance = evaluate_regime_strategy(
            smoothed_predictions,
            true_returns,
            dates,
            transaction_cost=config.transaction_cost
        )
        
        # Add smoothing information to results
        if results_df is not None:
            results_df['smoothing_method'] = smoothing_method
            for param_name, param_value in smoothing_params.items():
                results_df[f'smoothing_{param_name}'] = param_value
            
            # Add original regime information
            results_df['raw_regime'] = raw_predictions.flatten()
            
            # Calculate original and smoothed trade counts
            raw_trades = (np.diff(raw_predictions.flatten()) != 0).sum() + (raw_predictions[0] == 1)
            smoothed_trades = (np.diff(smoothed_predictions.flatten()) != 0).sum() + (smoothed_predictions[0] == 1)
            
            # Add trade reduction information to performance
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
        
    except Exception as e:
        logging.error(f"Error in apply_and_evaluate_with_smoothing: {str(e)}")
        return None, None


def visualize_comparison(
    all_methods_results: Dict[str, Dict[str, Any]], 
    window_number: int, 
    title: str, 
    save_path: str
):
    """
    Visualize comparison of various smoothing methods' results
    
    Args:
        all_methods_results: Dictionary of all methods' results
        window_number: Window number
        title: Chart title
        save_path: Save path
    """
    plt.figure(figsize=(15, 10))
    
    # Compare cumulative returns
    plt.subplot(2, 1, 1)
    
    # Draw market returns only once
    first_method = list(all_methods_results.keys())[0]
    plt.plot(all_methods_results[first_method]['df']['Cum_Market'] * 100, 
            label='Market', color='gray', linestyle='--')
    
    # Draw returns for each smoothing method
    for method_name, result in all_methods_results.items():
        plt.plot(result['df']['Cum_Strategy'] * 100, label=method_name)
    
    plt.title(f'{title}')
    plt.legend()
    plt.ylabel('Returns (%)')
    plt.grid(True)
    
    # Summarize trade counts and returns
    plt.subplot(2, 1, 2)
    
    methods = list(all_methods_results.keys())
    returns = [all_methods_results[method]['cum_return'] for method in methods]
    trades = [all_methods_results[method]['n_trades'] for method in methods]
    
    # Create two y-axes
    fig = plt.gca()
    ax1 = fig.axes
    ax2 = ax1.twinx()
    
    # First y-axis: Returns
    bars1 = ax1.bar(np.arange(len(methods)) - 0.2, returns, width=0.4, color='blue', alpha=0.7, label='Returns (%)')
    ax1.set_ylabel('Returns (%)', color='blue')
    ax1.tick_params(axis='y', colors='blue')
    
    # Second y-axis: Trade counts
    bars2 = ax2.bar(np.arange(len(methods)) + 0.2, trades, width=0.4, color='red', alpha=0.7, label='Trade Count')
    ax2.set_ylabel('Trade Count', color='red')
    ax2.tick_params(axis='y', colors='red')
    
    plt.xticks(np.arange(len(methods)), methods, rotation=45)
    plt.title('Smoothing Method Returns and Trade Counts')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=4, alpha=0.7),
        Line2D([0], [0], color='red', lw=4, alpha=0.7)
    ]
    plt.legend(custom_lines, ['Returns (%)', 'Trade Count'])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_smoothing_method(
    method_info: Tuple[str, Dict[str, Any]],
    model, 
    data: pd.DataFrame, 
    kmeans, 
    bull_regime: int, 
    forward_start: str, 
    forward_end: str, 
    window_results_dir: str,
    config: RollingWindowTrainConfig
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Evaluate a single smoothing method and save its results
    
    Args:
        method_info: Tuple of (method_name, parameters)
        model: Trained model
        data: Full dataframe
        kmeans: K-Means model
        bull_regime: Bull regime cluster ID
        forward_start: Future period start date
        forward_end: Future period end date
        window_results_dir: Window results directory
        config: Configuration object
        
    Returns:
        Tuple: (method_id, method_result) or (None, None) on error
    """
    method_name, params = method_info
    
    # Create method identifier
    param_str = '_'.join([f"{k}={v}" for k, v in params.items()]) if params else "default"
    method_id = f"{method_name}_{param_str}" if params else method_name
    
    logging.info(f"Evaluating: {method_id}")
    
    # Apply smoothing and evaluate
    results_df, performance = apply_and_evaluate_with_smoothing(
        model, data, kmeans, bull_regime, forward_start, forward_end, 
        method_name, config, params
    )
    
    if results_df is not None and performance is not None:
        # Create result dictionary
        method_result = {
            'df': results_df,
            'performance': performance,
            'cum_return': performance['cumulative_returns']['strategy'],
            'n_trades': performance['trading_metrics']['number_of_trades']
        }
        
        # Save individual results
        method_results_dir = os.path.join(window_results_dir, method_id)
        os.makedirs(method_results_dir, exist_ok=True)
        results_df.to_csv(os.path.join(method_results_dir, 'results.csv'), index=False)
        
        with open(os.path.join(method_results_dir, 'performance.json'), 'w') as f:
            json.dump(performance, f, default=default_converter, indent=4)
        
        return method_id, method_result
    
    return None, None


def evaluate_all_smoothing_methods(
    window_results_dir: str, 
    model, 
    data: pd.DataFrame, 
    kmeans, 
    bull_regime: int, 
    forward_start: str, 
    forward_end: str, 
    window_number: int, 
    config: RollingWindowTrainConfig,
    max_workers: Optional[int] = None
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Evaluate multiple smoothing methods and compare results
    
    Args:
        window_results_dir: Window results directory
        model: Trained model
        data: Full dataframe
        kmeans: K-Means model
        bull_regime: Bull regime cluster ID
        forward_start: Future period start date
        forward_end: Future period end date
        window_number: Window number
        config: Configuration object
        max_workers: Maximum number of worker processes
        
    Returns:
        Tuple: (all_methods_results, all_methods_performances)
    """
    # Define smoothing methods to test
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
    
    # Result storage objects
    all_methods_results = {}
    all_methods_performances = []
    
    # Use ProcessPoolExecutor for parallel evaluation
    if torch.cuda.is_available():
        # Cannot use multiprocessing with CUDA due to limitations
        # Use sequential evaluation instead
        for method_info in tqdm(smoothing_methods, desc="Evaluating methods"):
            method_id, method_result = evaluate_smoothing_method(
                method_info, model, data, kmeans, bull_regime, 
                forward_start, forward_end, window_results_dir, config
            )
            if method_id is not None and method_result is not None:
                all_methods_results[method_id] = method_result
                all_methods_performances.append(method_result['performance'])
    else:
        # Use multiprocessing for CPU evaluation
        max_workers = max_workers or min(len(smoothing_methods), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for method_info in smoothing_methods:
                futures.append(
                    executor.submit(
                        evaluate_smoothing_method,
                        method_info, model, data, kmeans, bull_regime,
                        forward_start, forward_end, window_results_dir, config
                    )
                )
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating methods"):
                method_id, method_result = future.result()
                if method_id is not None and method_result is not None:
                    all_methods_results[method_id] = method_result
                    all_methods_performances.append(method_result['performance'])
    
    # Visualize comparison of all smoothing methods
    if all_methods_results:
        visualize_comparison(
            all_methods_results,
            window_number,
            f"Window {window_number}: {forward_start} ~ {forward_end} - Smoothing Method Comparison",
            os.path.join(window_results_dir, 'all_methods_comparison.png')
        )
    
    return all_methods_results, all_methods_performances


def visualize_final_comparison(
    combined_results: Dict[str, Dict[str, Any]], 
    save_dir: str
) -> Dict[str, Any]:
    """
    Visualize comparison of smoothing methods across all windows
    
    Args:
        combined_results: Combined results dictionary
        save_dir: Save directory
        
    Returns:
        Dict: Summary dictionary
    """
    # Identify all smoothing methods and windows
    all_methods = sorted(list(combined_results.keys()))
    all_windows = sorted(list(set(combined_results[all_methods[0]]['window'])))
    
    # 1. Compare average performance by method
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
    
    # Sort by performance
    sorted_indices = np.argsort(method_avg_returns)[::-1]  # Descending by return
    sorted_methods = [all_methods[i] for i in sorted_indices]
    sorted_returns = [method_avg_returns[i] for i in sorted_indices]
    sorted_trades = [method_avg_trades[i] for i in sorted_indices]
    sorted_sharpes = [method_avg_sharpes[i] for i in sorted_indices]
    
    # Create consolidated visualization
    plt.figure(figsize=(15, 12))
    
    # Compare average returns
    plt.subplot(2, 2, 1)
    plt.bar(range(len(sorted_methods)), sorted_returns, color='blue', alpha=0.7)
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45)
    plt.title('Average Returns by Smoothing Method (%)')
    plt.ylabel('Average Returns (%)')
    plt.grid(True, alpha=0.3)
    
    # Compare average trade counts
    plt.subplot(2, 2, 2)
    plt.bar(range(len(sorted_methods)), sorted_trades, color='red', alpha=0.7)
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45)
    plt.title('Average Trade Count by Smoothing Method')
    plt.ylabel('Average Trade Count')
    plt.grid(True, alpha=0.3)
    
    # Compare average Sharpe ratios
    plt.subplot(2, 2, 3)
    plt.bar(range(len(sorted_methods)), sorted_sharpes, color='green', alpha=0.7)
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45)
    plt.title('Average Sharpe Ratio by Smoothing Method')
    plt.ylabel('Average Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Scatter plot of returns vs. trade counts
    plt.subplot(2, 2, 4)
    plt.scatter(sorted_trades, sorted_returns, color='purple', alpha=0.7)
    
    # Label each point with method name
    for i, method in enumerate(sorted_methods):
        plt.annotate(method, (sorted_trades[i], sorted_returns[i]), 
                    textcoords="offset points", xytext=(0,5), ha='center')
    
    plt.xlabel('Average Trade Count')
    plt.ylabel('Average Returns (%)')
    plt.title('Returns vs. Trade Count')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_methods_comparison.png'))
    plt.close()
    
    # 2. Histogram of best methods by window
    best_methods = []
    for window in all_windows:
        window_returns = {method: combined_results[method]['returns'][window] for method in all_methods}
        best_method = max(window_returns.items(), key=lambda x: x[1])[0]
        best_methods.append(best_method)
    
    # Calculate frequency of each best method
    method_counts = {}
    for method in all_methods:
        method_counts[method] = best_methods.count(method)
    
    # Sort by frequency
    sorted_method_counts = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_count_methods = [item[0] for item in sorted_method_counts]
    sorted_counts = [item[1] for item in sorted_method_counts]
    
    # Visualize best method frequency
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(sorted_count_methods)), sorted_counts, color='orange', alpha=0.7)
    plt.xticks(range(len(sorted_count_methods)), sorted_count_methods, rotation=45)
    plt.title('Frequency of Best Smoothing Method by Window')
    plt.ylabel('Number of Windows')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_methods_histogram.png'))
    plt.close()
    
    # 3. Compare cumulative performance across methods
    plt.figure(figsize=(12, 8))
    
    for method in all_methods:
        method_cum_returns = [combined_results[method]['returns'][window] for window in all_windows]
        cum_performance = np.cumsum(method_cum_returns)
        plt.plot(all_windows, cum_performance, marker='o', label=method)
    
    plt.title('Cumulative Performance by Smoothing Method')
    plt.xlabel('Window')
    plt.ylabel('Cumulative Returns (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cumulative_method_performance.png'))
    plt.close()
    
    # 4. Save summary statistics
    summary = {
        'avg_returns': {method: float(avg) for method, avg in zip(sorted_methods, sorted_returns)},
        'avg_trades': {method: float(avg) for method, avg in zip(sorted_methods, sorted_trades)},
        'avg_sharpes': {method: float(avg) for method, avg in zip(sorted_methods, sorted_sharpes)},
        'best_method_counts': {method: count for method, count in sorted_method_counts}
    }
    
    # Determine best methods
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
    
    # 5. Print summary
    logging.info("\n===== Smoothing Method Performance Summary =====")
    logging.info("Top 3 by Returns:")
    for i in range(min(3, len(sorted_methods))):
        logging.info(f"  {i+1}. {sorted_methods[i]}: {sorted_returns[i]:.2f}%")
    
    logging.info("\nTop 3 by Sharpe Ratio:")
    sharpe_indices = np.argsort(sorted_sharpes)[::-1][:3]
    for i, idx in enumerate(sharpe_indices):
        logging.info(f"  {i+1}. {sorted_methods[idx]}: {sorted_sharpes[idx]:.2f}")
    
    logging.info("\nMost Frequently Optimal Methods:")
    for i in range(min(3, len(sorted_count_methods))):
        if sorted_counts[i] > 0:
            logging.info(f"  {i+1}. {sorted_count_methods[i]}: {sorted_counts[i]} windows")
    
    return summary


def run_smoothing_comparison(
    config: RollingWindowTrainConfig,
    max_workers: Optional[int] = None
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """
    Run rolling window training to compare smoothing methods
    
    Args:
        config: Configuration object
        max_workers: Maximum number of worker processes
        
    Returns:
        Tuple: (combined_results, summary) or (None, None) on error
    """
    # Load data
    logging.info("Loading data...")
    try:
        data = pd.read_csv(config.data_path)
        data['returns'] = data['returns'] * 100
        data["dd_10"] = data["dd_10"] * 100
        data["sortino_20"] = data["sortino_20"] * 100
        data["sortino_60"] = data["sortino_60"] * 100
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None
    
    # Result storage object (nested dictionary: method -> window -> performance)
    combined_results = defaultdict(lambda: {
        'window': [],
        'returns': {},
        'trades': {},
        'sharpes': {},
        'performances': []
    })
    
    # Parse start and end dates
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    
    window_number = 1
    window_data = []
    
    # Precompute all window dates for better progress tracking
    while current_date <= end_date:
        # Calculate training period
        train_start = (current_date - relativedelta(years=config.total_window_years)).strftime('%Y-%m-%d')
        train_end = (current_date - relativedelta(years=config.valid_years + config.clustering_years)).strftime('%Y-%m-%d')
        
        # Calculate validation period
        valid_start = (current_date - relativedelta(years=config.clustering_years)).strftime('%Y-%m-%d')
        valid_end = current_date.strftime('%Y-%m-%d')
        
        # Calculate clustering period
        clustering_start = (current_date - relativedelta(years=config.clustering_years)).strftime('%Y-%m-%d')
        clustering_end = current_date.strftime('%Y-%m-%d')
        
        # Calculate future application period
        forward_start = current_date.strftime('%Y-%m-%d')
        forward_end = (current_date + relativedelta(months=config.forward_months)).strftime('%Y-%m-%d')
        
        window_data.append({
            'window_number': window_number,
            'train_start': train_start,
            'train_end': train_end,
            'valid_start': valid_start,
            'valid_end': valid_end,
            'clustering_start': clustering_start,
            'clustering_end': clustering_end,
            'forward_start': forward_start,
            'forward_end': forward_end,
            'current_date': current_date
        })
        
        current_date += relativedelta(months=config.forward_months)
        window_number += 1
    
    # Run rolling window main loop
    for window_info in tqdm(window_data, desc="Processing windows"):
        window_number = window_info['window_number']
        logging.info(f"\n=== Processing Window {window_number} ===")
        
        # Create window-specific results directory
        window_dir = os.path.join(config.results_dir, f"window_{window_number}")
        os.makedirs(window_dir, exist_ok=True)
        
        # Log period information
        logging.info(f"Training: {window_info['train_start']} ~ {window_info['train_end']} ({config.train_years} years)")
        logging.info(f"Validation: {window_info['valid_start']} ~ {window_info['valid_end']} ({config.valid_years} years)")
        logging.info(f"Clustering: {window_info['clustering_start']} ~ {window_info['clustering_end']} ({config.clustering_years} years)")
        logging.info(f"Future: {window_info['forward_start']} ~ {window_info['forward_end']} ({config.forward_months/12:.1f} years)")
        
        # 1. Train model
        model, val_loss = train_model_for_window(
            config, 
            window_info['train_start'], 
            window_info['train_end'], 
            window_info['valid_start'], 
            window_info['valid_end'], 
            data
        )
        
        # Skip to next window if training fails
        if model is None:
            logging.warning("Model training failed, skipping to next window.")
            continue
        
        # 2. Identify regimes
        kmeans, bull_regime = identify_regimes_for_window(
            config, model, data, window_info['clustering_start'], window_info['clustering_end']
        )
        
        # Skip to next window if regime identification fails
        if kmeans is None or bull_regime is None:
            logging.warning("Regime identification failed, skipping to next window.")
            continue
        
        # 3. Evaluate various smoothing methods
        all_methods_results, all_methods_performances = evaluate_all_smoothing_methods(
            window_dir, model, data, kmeans, bull_regime, 
            window_info['forward_start'], window_info['forward_end'], 
            window_number, config, max_workers
        )
        
        # 4. Save results
        if all_methods_results:
            # Save results by method and window
            for method_name, result in all_methods_results.items():
                combined_results[method_name]['window'].append(window_number)
                combined_results[method_name]['returns'][window_number] = result['cum_return']
                combined_results[method_name]['trades'][window_number] = result['n_trades']
                combined_results[method_name]['sharpes'][window_number] = result['performance']['sharpe_ratio']['strategy']
                combined_results[method_name]['performances'].append(result['performance'])
    
    # 5. Create final comparison visualization and summary
    if combined_results:
        summary = visualize_final_comparison(combined_results, config.results_dir)
        
        # Save all results
        with open(os.path.join(config.results_dir, 'all_results.json'), 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            json_results = {}
            for method, result in combined_results.items():
                json_results[method] = {
                    'window': result['window'],
                    'returns': {str(k): v for k, v in result['returns'].items()},
                    'trades': {str(k): v for k, v in result['trades'].items()},
                    'sharpes': {str(k): v for k, v in result['sharpes'].items()}
                }
            json.dump(json_results, f, indent=4)
        
        logging.info(f"\nComparison complete! {len(window_data)} windows, {len(combined_results)} methods compared.")
        return combined_results, summary
    else:
        logging.warning("Comparison failed: No valid results.")
        return None, None


def main():
    """Main execution function"""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, f"smoothing_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args)
    config.results_dir = output_dir
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write("=== Smoothing Method Comparison Settings ===\n")
        f.write(f"Data path: {config.data_path}\n")
        f.write(f"Start date: {config.start_date}\n")
        f.write(f"End date: {config.end_date}\n")
        f.write(f"Total data period: {config.total_window_years} years\n")
        f.write(f"Training period: {config.train_years} years\n")
        f.write(f"Validation period: {config.valid_years} years\n")
        f.write(f"Clustering period: {config.clustering_years} years\n")
        f.write(f"Next window interval: {config.forward_months} months\n")
        f.write(f"Model dimension: {config.d_model}\n")
        f.write(f"State dimension: {config.d_state}\n")
        f.write(f"Number of layers: {config.n_layers}\n")
        f.write(f"Dropout rate: {config.dropout}\n")
        f.write(f"Sequence length: {config.seq_len}\n")
        f.write(f"Batch size: {config.batch_size}\n")
        f.write(f"Learning rate: {config.learning_rate}\n")
        f.write(f"Maximum epochs: {config.max_epochs}\n")
        f.write(f"Early stopping patience: {config.patience}\n")
        f.write(f"Transaction cost: {config.transaction_cost}\n")
    
    # Run comparison
    logger.info("=== Starting Smoothing Method Comparison ===")
    combined_results, summary = run_smoothing_comparison(config, args.max_workers)
    
    if combined_results is not None:
        logger.info(f"Smoothing method comparison complete! Results saved to {output_dir}")
    else:
        logger.error("Smoothing method comparison failed!")


if __name__ == "__main__":
    main()
