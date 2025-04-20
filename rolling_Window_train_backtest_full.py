#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rolling Window Training Backtest with Smoothing Technique Comparison
This script trains models in a rolling window fashion and compares various smoothing techniques.
"""

import os
import sys
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
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import copy

from regime_mamba.utils.utils import set_seed
from regime_mamba.config.config import RollingWindowTrainConfig
from regime_mamba.data.dataset_full import DateRangeRegimeMambaDataset
from regime_mamba.evaluate.rolling_window_w_train_full import (
    train_model_for_window, 
    identify_regimes_for_window
)
from regime_mamba.evaluate.smoothing import (
    apply_regime_smoothing,
    apply_confirmation_rule,
    apply_minimum_holding_period
)
from regime_mamba.evaluate.strategy import evaluate_regime_strategy
from regime_mamba.evaluate.clustering import predict_regimes


def json_serializer(obj):
    """JSON 직렬화를 위한 변환 함수 - 순환 참조 처리 및 다양한 타입 지원
    
    Args:
        obj: 직렬화할 객체
        
    Returns:
        직렬화 가능한 객체
    """
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # 다른 유형의 경우 문자열로 변환 시도
        try:
            return str(obj)
        except:
            return None


def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
        
    Returns:
        logging.Logger: Logger instance
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Rolling Window Training Backtest with Smoothing Technique Comparison')
    
    # Configuration sources
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Required parameters
    parser.add_argument('--data_path', type=str, help='Data file path')
    
    # Optional parameters with defaults
    parser.add_argument('--results_dir', type=str, help='Results directory')
    parser.add_argument('--start_date', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--preprocessed', action='store_true', help='Whether data is preprocessed')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    
    # Period-related settings
    parser.add_argument('--total_window_years', type=int, help='Total data period (years)')
    parser.add_argument('--train_years', type=int, help='Training period (years)')
    parser.add_argument('--valid_years', type=int, help='Validation period (years)')
    parser.add_argument('--clustering_years', type=int, help='Clustering period (years)')
    parser.add_argument('--forward_months', type=int, help='Interval to next window (months)')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=128, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--target_type', type=str, default='average', help='Target type')
    parser.add_argument('--target_horizon', type=int, default=5, help='Target horizon')
    parser.add_argument('--cluster_method', type=str, default='cosine_kmeans', help='Clustering method')
    parser.add_argument('--direct_train', action='store_true', help='Train model directly for clasification')
    parser.add_argument('--vae', action='store_true', help='Train model with VAE')

    # Training-related settings
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_onecycle', type=bool, default=True, help='Use one-cycle learning rate policy')
    parser.add_argument('--progressive_train', type=bool, default=False, help='Progressive training flag')

    # Extra Settings (jump model and rl_model)
    parser.add_argument('--jump_model', type=bool, default=False, help='Jump model flag')
    parser.add_argument('--rl_model', type=bool, default=False, help='Reinforcement learning model flag')
    parser.add_argument('--rl_learning_rate', type=float, default=1e-4, help='Reinforcement learning learning rate')
    parser.add_argument('--rl_gamma', type=float, default=0.99, help='Reinforcement learning discount factor')
    parser.add_argument('--freeze_feature_extractor', type=bool, default=True, help='Freeze feature extractor during training')
    parser.add_argument('--position_penalty', type=float, default=0.01, help='Reinforcement learning position penalty')
    parser.add_argument('--reward_type', type=str, default='sharpe', help='Reinforcement learning reward type')
    parser.add_argument('--window_size', type=int, default=252, help='Window size for Sharpe calculation')
    parser.add_argument('--n_episodes', type=int, default=50, help='Number of episodes for reinforcement learning')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs for reinforcement learning')
    parser.add_argument('--rl_batch_size', type=int, default=512, help='Batch size for reinforcement learning')
    parser.add_argument('--steps_per_episode', type=int, default=2048, help='Steps per episode for reinforcement learning')
    parser.add_argument('--n_positions', type=int, default=3, help='Number of positions for reinforcement learning')
    parser.add_argument('--optimize_thresholds', action='store_true', help='Optimize action thresholds for reinforcement learning')

    # Performance-related settings
    parser.add_argument('--max_workers', type=int, help='Maximum number of worker processes')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--enable_checkpointing', action='store_true', help='Enable checkpointing')
    parser.add_argument('--checkpoint_interval', type=int, help='Checkpoint interval (windows)')

    # Optional parameters
    parser.add_argument('--predict', default=False, help='Predict price to predict regime')
    
    return parser.parse_args()


def load_config(args) -> RollingWindowTrainConfig:
    """Load configuration from file and command-line arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        RollingWindowTrainConfig: Configuration object
    """
    config = RollingWindowTrainConfig()
    
    # Set default values / 1982-04-20
    defaults = {
        'results_dir': '/content/drive/MyDrive/train_backtest_results',
        'start_date': '2012-04-20',
        'end_date': '2023-12-31',
        'preprocessed': True,
        'total_window_years': 30,
        'train_years': 10,
        'valid_years': 10,
        'clustering_years': 10,
        'forward_months': 24,
        'd_model': 128,
        'd_state': 128,
        'n_layers': 4,
        'dropout': 0.1,
        'seq_len': 128,
        'batch_size': 64,
        'learning_rate': 1e-6,
        'max_epochs': 100,
        'patience': 10,
        'transaction_cost': 0.001,
        'seed': 42,
        'max_workers': None,
        'gpu_id': 0,
        'enable_checkpointing': False,
        'checkpoint_interval': 1
    }

    # Update with command-line arguments if provided (overrides YAML)
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Load from YAML file if provided
    yaml_config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    
    # Update with values from YAML
    for key, value in yaml_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Check for required parameters
    required_params = ['data_path']
    missing_params = [param for param in required_params if getattr(config, param, None) is None]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
    
    # Fill in defaults for missing parameters
    for key, value in defaults.items():
        if getattr(config, key, None) is None:
            setattr(config, key, value)
    
    # Set device
    if config.gpu_id >= 0 and torch.cuda.is_available():
        config.device = f'cuda:{config.gpu_id}'
    else:
        config.device = 'cpu'
    
    return config


def prepare_output_directory(output_dir: str) -> tuple:
    """Create output directory structure with timestamped subdirectory
    
    Args:
        output_dir: Base output directory
        
    Returns:
        tuple: (result_dir, log_file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"train_backtest_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    log_file = os.path.join(result_dir, "train_backtest.log")
    
    return result_dir, log_file


def save_config(config: RollingWindowTrainConfig, output_dir: str):
    """Save configuration to file
    
    Args:
        config: Configuration object
        output_dir: Output directory
    """
    config_dict = {key: getattr(config, key) for key in dir(config) 
                   if not key.startswith('__') and not callable(getattr(config, key))}
    
    # Save as YAML
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Also save as text for easier reading
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write("=== Rolling Window Train Backtest Configuration ===\n\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")


def get_smoothing_methods() -> List[Tuple[str, Dict[str, Any]]]:
    """Get list of smoothing methods to evaluate
    
    Returns:
        List[Tuple[str, Dict[str, Any]]]: List of (method_name, parameters) tuples
    """
    return [
        ('none', {}),
        ('ma', {'window': 3}),
        ('ma', {'window': 5}),
        ('exp', {'window': 5}),
        ('confirmation', {'days': 1}),
        ('confirmation', {'days': 2}),
        ('confirmation', {'days': 3}),
        ('min_holding', {'days': 10}),
        ('min_holding', {'days': 20}),
        ('min_holding', {'days': 30}),
        ('min_holding', {'days': 60}),

    ]


def apply_smoothing_method(
    raw_predictions: np.ndarray, 
    method_name: str, 
    params: Dict[str, Any]
) -> np.ndarray:
    """Apply a specific smoothing method to raw predictions
    
    Args:
        raw_predictions: Raw regime predictions
        method_name: Smoothing method name
        params: Smoothing parameters
        
    Returns:
        np.ndarray: Smoothed predictions
    """
    if method_name == 'none':
        return raw_predictions
    elif method_name == 'ma':
        window = params.get('window', 10)
        return apply_regime_smoothing(
            raw_predictions, method='ma', window=window
        ).reshape(-1, 1)
    elif method_name == 'exp':
        window = params.get('window', 10)
        return apply_regime_smoothing(
            raw_predictions, method='exp', window=window
        ).reshape(-1, 1)
    elif method_name == 'confirmation':
        days = params.get('days', 3)
        return apply_confirmation_rule(
            raw_predictions, confirmation_days=days
        ).reshape(-1, 1)
    elif method_name == 'min_holding':
        days = params.get('days', 20)
        return apply_minimum_holding_period(
            raw_predictions, min_holding_days=days
        ).reshape(-1, 1)
    else:
        logging.warning(f"Unknown smoothing method: {method_name}, using raw predictions")
        return raw_predictions


def evaluate_method(
    model,
    data: pd.DataFrame, 
    kmeans,
    bull_regime: int,
    method_info: Tuple[str, Dict[str, Any]],
    config: RollingWindowTrainConfig,
    forward_period: Dict[str, str]
) -> Dict[str, Any]:
    """Evaluate a single smoothing method
    
    Args:
        model: Trained model
        data: Full dataframe
        kmeans: K-Means model
        bull_regime: Bull regime cluster ID
        method_info: Tuple of (method_name, parameters)
        config: Configuration object
        forward_period: Dictionary with forward start and end dates
        
    Returns:
        Dict[str, Any]: Result dictionary
    """
    method_name, params = method_info
    forward_start, forward_end = forward_period['start'], forward_period['end']
    
    try:
        # Create forward dataset
        forward_dataset = DateRangeRegimeMambaDataset(
            data=data, 
            seq_len=config.seq_len,
            start_date=forward_start,
            end_date=forward_end,
            config=config
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
            return None
        
        # Get original regime predictions
        raw_predictions, true_returns, dates = predict_regimes(
            model, forward_loader, kmeans, bull_regime, config
        )
        
        # Apply smoothing
        smoothed_predictions = apply_smoothing_method(raw_predictions, method_name, params)
        
        # Evaluate strategy with transaction costs
        results_df, performance = evaluate_regime_strategy(
            smoothed_predictions,
            true_returns,
            dates,
            transaction_cost=config.transaction_cost,
            config=config
        )
        
        if results_df is not None and performance is not None:
            # Add smoothing information
            results_df['smoothing_method'] = method_name
            for param_name, param_value in params.items():
                results_df[f'smoothing_{param_name}'] = param_value
            
            # Add original regime information
            results_df['raw_regime'] = raw_predictions.flatten()
            
            # Calculate trade metrics
            raw_trades = (np.diff(raw_predictions.flatten()) != 0).sum() + (raw_predictions[0] == 1)
            smoothed_trades = (np.diff(smoothed_predictions.flatten()) != 0).sum() + (smoothed_predictions[0] == 1)
            
            # Add trade metrics to performance
            performance['smoothing_method'] = method_name
            for param_name, param_value in params.items():
                performance[f'smoothing_{param_name}'] = param_value
            performance['raw_trades'] = int(raw_trades)
            performance['smoothed_trades'] = int(smoothed_trades)
            performance['trade_reduction'] = int(raw_trades - smoothed_trades)
            performance['trade_reduction_pct'] = ((raw_trades - smoothed_trades) / raw_trades * 100) if raw_trades > 0 else 0
            
            # Create method ID
            param_str = '_'.join([f"{k}={v}" for k, v in params.items()]) if params else "default"
            method_id = f"{method_name}_{param_str}" if params else method_name
            
            # Return result
            return {
                'method_id': method_id,
                'method_name': method_name,
                'params': params,
                'df': results_df,
                'performance': performance,
                'cum_return': performance['cumulative_returns']['strategy'],
                'n_trades': performance['trading_metrics']['number_of_trades'],
                'sharpe': performance['sharpe_ratio']['strategy']
            }
        
        return None
    
    except Exception as e:
        logging.error(f"Error evaluating method {method_name}: {str(e)}")
        traceback.print_exc()
        return None


def evaluate_smoothing_methods(
    model,
    data: pd.DataFrame,
    kmeans,
    bull_regime: int,
    config: RollingWindowTrainConfig,
    forward_period: Dict[str, str],
    window_results_dir: str,
    max_workers: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """Evaluate multiple smoothing methods
    
    Args:
        model: Trained model
        data: Full dataframe
        kmeans: K-Means model
        bull_regime: Bull regime cluster ID
        config: Configuration object
        forward_period: Dictionary with forward start and end dates
        window_results_dir: Window results directory
        max_workers: Maximum number of worker processes
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of method results
    """
    smoothing_methods = get_smoothing_methods()
    all_methods_results = {}
    
    # Use sequential evaluation if using GPU (multiprocessing issues with CUDA)
    if 'cuda' in config.device:
        for method_info in tqdm(smoothing_methods, desc="Evaluating methods"):
            result = evaluate_method(model, data, kmeans, bull_regime, method_info, config, forward_period)
            if result:
                all_methods_results[result['method_id']] = result
                
                # Save individual result
                method_dir = os.path.join(window_results_dir, result['method_id'])
                os.makedirs(method_dir, exist_ok=True)
                result['df'].to_csv(os.path.join(method_dir, 'results.csv'), index=False)
                
                # 수정된 부분: 성능 지표 저장 시 커스텀 직렬화 함수 사용
                with open(os.path.join(method_dir, 'performance.json'), 'w') as f:
                    json.dump(result['performance'], f, default=json_serializer, indent=4)
    else:
        # Use parallel evaluation for CPU
        max_workers = max_workers or min(len(smoothing_methods), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for method_info in smoothing_methods:
                futures.append(
                    executor.submit(
                        evaluate_method,
                        model, data, kmeans, bull_regime, method_info, config, forward_period
                    )
                )
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating methods"):
                try:
                    result = future.result()
                    if result:
                        all_methods_results[result['method_id']] = result
                        
                        # Save individual result
                        method_dir = os.path.join(window_results_dir, result['method_id'])
                        os.makedirs(method_dir, exist_ok=True)
                        result['df'].to_csv(os.path.join(method_dir, 'results.csv'), index=False)
                        
                        # 수정된 부분: 성능 지표 저장 시 커스텀 직렬화 함수 사용
                        with open(os.path.join(method_dir, 'performance.json'), 'w') as f:
                            json.dump(result['performance'], f, default=json_serializer, indent=4)
                except Exception as e:
                    logging.error(f"Error processing future: {str(e)}")
    
    # Create comparison visualization
    if all_methods_results:
        visualize_methods_comparison(
            all_methods_results, 
            os.path.join(window_results_dir, 'methods_comparison.png'), 
            forward_period
        )
    
    return all_methods_results


def visualize_methods_comparison(
    methods_results: Dict[str, Dict[str, Any]],
    save_path: str,
    forward_period: Dict[str, str]
):
    """Visualize comparison of smoothing methods
    
    Args:
        methods_results: Dictionary of method results
        save_path: Path to save visualization
        forward_period: Dictionary with forward start and end dates
    """
    plt.figure(figsize=(15, 10))
    
    # Plot cumulative returns
    plt.subplot(2, 1, 1)
    
    # Plot market returns once
    first_method = list(methods_results.keys())[0]
    plt.plot(
        methods_results[first_method]['df']['Cum_Market'] * 100, 
        label='Market', 
        color='gray', 
        linestyle='--'
    )
    
    # Plot strategy returns for each method
    for method_id, result in methods_results.items():
        plt.plot(result['df']['Cum_Strategy'] * 100, label=method_id)
    
    plt.title(f"Comparison of Smoothing Methods ({forward_period['start']} to {forward_period['end']})")
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot returns vs trades
    plt.subplot(2, 1, 2)
    
    method_ids = list(methods_results.keys())
    returns = [methods_results[method_id]['cum_return'] for method_id in method_ids]
    trades = [methods_results[method_id]['n_trades'] for method_id in method_ids]
    
    # Create two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot returns
    bars1 = ax1.bar(np.arange(len(method_ids)) - 0.2, returns, width=0.4, color='blue', alpha=0.7)
    ax1.set_ylabel('Returns (%)', color='blue')
    ax1.tick_params(axis='y', colors='blue')
    
    # Plot trades
    bars2 = ax2.bar(np.arange(len(method_ids)) + 0.2, trades, width=0.4, color='red', alpha=0.7)
    ax2.set_ylabel('Number of Trades', color='red')
    ax2.tick_params(axis='y', colors='red')
    
    plt.xticks(np.arange(len(method_ids)), method_ids, rotation=45, ha='right')
    plt.title('Returns vs. Number of Trades by Method')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=4, alpha=0.7),
        Line2D([0], [0], color='red', lw=4, alpha=0.7)
    ]
    plt.legend(custom_lines, ['Returns (%)', 'Number of Trades'])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_final_comparison(
    combined_results: Dict[str, Dict[str, Any]],
    save_dir: str
) -> Dict[str, Any]:
    """Visualize final comparison of smoothing methods across all windows
    
    Args:
        combined_results: Combined results dictionary
        save_dir: Directory to save visualizations
        
    Returns:
        Dict[str, Any]: Summary dictionary
    """
    # Get all methods and windows
    all_methods = sorted(list(combined_results.keys()))
    all_windows = sorted(list(set(combined_results[all_methods[0]]['window'])))
    
    # Calculate average metrics
    method_metrics = {}
    for method in all_methods:
        returns = [combined_results[method]['returns'][window] for window in all_windows]
        trades = [combined_results[method]['trades'][window] for window in all_windows]
        sharpes = [combined_results[method]['sharpes'][window] for window in all_windows]
        
        method_metrics[method] = {
            'avg_return': np.mean(returns),
            'avg_trades': np.mean(trades),
            'avg_sharpe': np.mean(sharpes),
            'returns': returns,
            'cum_returns': np.cumsum(returns),
            'windows': all_windows
        }
    
    # Sort methods by average return
    sorted_methods = sorted(all_methods, key=lambda x: method_metrics[x]['avg_return'], reverse=True)
    
    # 1. Average metrics comparison
    plt.figure(figsize=(15, 12))
    
    # Average returns
    plt.subplot(2, 2, 1)
    plt.bar(
        range(len(sorted_methods)), 
        [method_metrics[method]['avg_return'] for method in sorted_methods], 
        color='blue', 
        alpha=0.7
    )
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45, ha='right')
    plt.title('Average Returns by Method (%)')
    plt.ylabel('Average Return (%)')
    plt.grid(True, alpha=0.3)
    
    # Average trades
    plt.subplot(2, 2, 2)
    plt.bar(
        range(len(sorted_methods)), 
        [method_metrics[method]['avg_trades'] for method in sorted_methods], 
        color='red', 
        alpha=0.7
    )
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45, ha='right')
    plt.title('Average Trades by Method')
    plt.ylabel('Average Trades')
    plt.grid(True, alpha=0.3)
    
    # Average Sharpe ratios
    plt.subplot(2, 2, 3)
    plt.bar(
        range(len(sorted_methods)), 
        [method_metrics[method]['avg_sharpe'] for method in sorted_methods], 
        color='green', 
        alpha=0.7
    )
    plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45, ha='right')
    plt.title('Average Sharpe Ratio by Method')
    plt.ylabel('Average Sharpe Ratio')
    plt.grid(True, alpha=0.3)
    
    # Returns vs. trades scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(
        [method_metrics[method]['avg_trades'] for method in all_methods],
        [method_metrics[method]['avg_return'] for method in all_methods],
        color='purple', 
        alpha=0.7
    )
    
    # Label each point
    for method in all_methods:
        plt.annotate(
            method, 
            (method_metrics[method]['avg_trades'], method_metrics[method]['avg_return']), 
            textcoords="offset points", 
            xytext=(0, 5), 
            ha='center'
        )
    
    plt.xlabel('Average Trades')
    plt.ylabel('Average Return (%)')
    plt.title('Returns vs. Trades')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_metrics_comparison.png'))
    plt.close()
    
    # 2. Cumulative returns comparison
    plt.figure(figsize=(12, 8))
    
    for method in sorted_methods:
        plt.plot(
            all_windows, 
            method_metrics[method]['cum_returns'], 
            marker='o', 
            markersize=4, 
            label=method
        )
    
    plt.title('Cumulative Returns by Method')
    plt.xlabel('Window')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cumulative_returns.png'))
    plt.close()
    
    # 3. Best method by window
    best_methods = []
    for window in all_windows:
        window_returns = {method: combined_results[method]['returns'][window] for method in all_methods}
        best_method = max(window_returns.items(), key=lambda x: x[1])[0]
        best_methods.append(best_method)
    
    # Count occurrences of each best method
    from collections import Counter
    best_method_counts = Counter(best_methods)
    
    # Sort by count
    sorted_best_methods = sorted(best_method_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Plot best method counts
    plt.figure(figsize=(12, 6))
    plt.bar(
        range(len(sorted_best_methods)), 
        [count for _, count in sorted_best_methods], 
        color='orange', 
        alpha=0.7
    )
    plt.xticks(
        range(len(sorted_best_methods)), 
        [method for method, _ in sorted_best_methods], 
        rotation=45, 
        ha='right'
    )
    plt.title('Best Method by Window')
    plt.ylabel('Number of Windows')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_method_histogram.png'))
    plt.close()
    
    # 4. Create summary dictionary
    summary = {
        'methods': {},
        'best_methods': {
            'by_return': sorted_methods[0],
            'by_sharpe': sorted(all_methods, key=lambda x: method_metrics[x]['avg_sharpe'], reverse=True)[0],
            'by_frequency': sorted_best_methods[0][0] if sorted_best_methods else None
        },
        'windows': len(all_windows),
        'total_methods': len(all_methods)
    }
    
    # Add method metrics
    for method in all_methods:
        summary['methods'][method] = {
            'avg_return': float(method_metrics[method]['avg_return']),
            'avg_trades': float(method_metrics[method]['avg_trades']),
            'avg_sharpe': float(method_metrics[method]['avg_sharpe']),
            'best_window_count': best_method_counts.get(method, 0)
        }
    
    # 수정된 부분: 커스텀 직렬화 함수로 요약 저장
    with open(os.path.join(save_dir, 'methods_summary.json'), 'w') as f:
        json.dump(summary, f, default=json_serializer, indent=4)
    
    # Log summary
    logging.info("\n===== Smoothing Method Performance Summary =====")
    logging.info("Top methods by average return:")
    for i, method in enumerate(sorted_methods[:3]):
        logging.info(f"  {i+1}. {method}: {method_metrics[method]['avg_return']:.2f}%")
    
    logging.info("\nTop methods by average Sharpe ratio:")
    sharpe_sorted = sorted(all_methods, key=lambda x: method_metrics[x]['avg_sharpe'], reverse=True)
    for i, method in enumerate(sharpe_sorted[:3]):
        logging.info(f"  {i+1}. {method}: {method_metrics[method]['avg_sharpe']:.2f}")
    
    logging.info("\nMost frequently best methods:")
    for i, (method, count) in enumerate(sorted_best_methods[:3]):
        logging.info(f"  {i+1}. {method}: {count} windows")
    
    return summary


def create_window_schedule(
    config: RollingWindowTrainConfig, 
    start_from_window: int = 1
) -> List[Dict[str, Any]]:
    """Create window schedule for rolling window backtest
    
    Args:
        config: Configuration object
        start_from_window: Window number to start from (for resuming)
        
    Returns:
        List[Dict[str, Any]]: List of window dictionaries
    """
    window_schedule = []
    
    # Parse start and end dates
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    
    window_number = 1
    
    # Create window schedule
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
        
        # Calculate forward application period
        forward_start = current_date.strftime('%Y-%m-%d')
        forward_end = (current_date + relativedelta(months=config.forward_months)).strftime('%Y-%m-%d')
        
        # Add to schedule if window number >= start_from_window
        if window_number >= start_from_window:
            window_schedule.append({
                'window_number': window_number,
                'train_period': {
                    'start': train_start,
                    'end': train_end
                },
                'valid_period': {
                    'start': valid_start,
                    'end': valid_end
                },
                'clustering_period': {
                    'start': clustering_start,
                    'end': clustering_end
                },
                'forward_period': {
                    'start': forward_start,
                    'end': forward_end
                },
                'current_date': current_date.strftime('%Y-%m-%d')
            })
        
        # Move to next window
        current_date += relativedelta(months=config.forward_months)
        window_number += 1
    
    return window_schedule


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dict[str, Any]: Checkpoint data
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        return checkpoint
    except Exception as e:
        raise ValueError(f"Error loading checkpoint: {str(e)}")


def save_checkpoint(
    checkpoint_data: Dict[str, Any], 
    checkpoint_path: str
):
    """Save checkpoint
    
    Args:
        checkpoint_data: Checkpoint data
        checkpoint_path: Path to save checkpoint
    """
    try:
        # 수정된 부분: 체크포인트 저장 시 커스텀 직렬화 함수 사용
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, default=json_serializer, indent=4)
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")


def run_rolling_window_backtest(
    config: RollingWindowTrainConfig,
    data: pd.DataFrame,
    logger: logging.Logger,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run rolling window backtest with smoothing technique comparison
    
    Args:
        config: Configuration object
        data: Preprocessed data
        logger: Logger instance
        checkpoint_path: Path to checkpoint file (optional)
        
    Returns:
        Dict[str, Any]: Results dictionary
    """
    # Create results storage
    combined_results = defaultdict(lambda: {
        'window': [],
        'returns': {},
        'trades': {},
        'sharpes': {},
        'performances': []
    })
    
    # Determine starting point
    start_from_window = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = load_checkpoint(checkpoint_path)
            combined_results = defaultdict(lambda: {
                'window': [],
                'returns': {},
                'trades': {},
                'sharpes': {},
                'performances': []
            })
            
            # Restore from checkpoint
            for method, result in checkpoint['results'].items():
                combined_results[method]['window'] = result['window']
                combined_results[method]['returns'] = {int(k): v for k, v in result['returns'].items()}
                combined_results[method]['trades'] = {int(k): v for k, v in result['trades'].items()}
                combined_results[method]['sharpes'] = {int(k): v for k, v in result['sharpes'].items()}
                combined_results[method]['performances'] = result['performances']
            
            start_from_window = checkpoint['next_window']
            logger.info(f"Resuming from checkpoint at window {start_from_window}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.info("Starting from beginning")
    
    # Create window schedule
    window_schedule = create_window_schedule(config, start_from_window)
    
    # Exit if no windows to process
    if not window_schedule:
        logger.info("No windows to process")
        return {'combined_results': combined_results, 'summary': None}
    
    # Process windows
    total_windows = len(window_schedule)
    logger.info(f"Processing {total_windows} windows from {window_schedule[0]['window_number']} to {window_schedule[-1]['window_number']}")
    
    for i, window_info in enumerate(window_schedule):
        window_number = window_info['window_number']
        logger.info(f"\n=== Processing Window {window_number} ({i+1}/{total_windows}) ===")
        
        # Create window directory
        window_dir = os.path.join(config.results_dir, f"window_{window_number}")
        os.makedirs(window_dir, exist_ok=True)
        
        # Log period information
        logger.info(f"Training period: {window_info['train_period']['start']} to {window_info['train_period']['end']}")
        logger.info(f"Validation period: {window_info['valid_period']['start']} to {window_info['valid_period']['end']}")
        logger.info(f"Clustering period: {window_info['clustering_period']['start']} to {window_info['clustering_period']['end']}")
        logger.info(f"Forward period: {window_info['forward_period']['start']} to {window_info['forward_period']['end']}")
        
        try:
            # 1. Train model
            logger.info("Training model...")
            if config.rl_model:
                # RL 모델 학습 - ActorCritic + PPO 에이전트 학습
                from regime_mamba.train.rl_train import train_rl_agent_for_window
                from regime_mamba.evaluate.rl_evaluate import evaluate_rl_agent
                
                logger.info("Training RL agent...")
                agent, model, history = train_model_for_window(
                    config,
                    window_info['train_period']['start'],
                    window_info['train_period']['end'],
                    window_info['valid_period']['start'],
                    window_info['valid_period']['end'],
                    data,
                    window_number=window_number
                )
                
                # Agent training 실패 시 다음 윈도우로 넘어감
                if agent is None or model is None:
                    logger.warning("RL agent training failed, skipping window")
                    continue
                
                # 학습 이력 저장
                history_path = os.path.join(window_dir, 'training_history.json')
                with open(history_path, 'w') as f:
                    json.dump(history, f, default=json_serializer, indent=4)
                
                # 학습 이력 시각화
                from regime_mamba.utils.rl_visualize import visualize_training_history
                visualize_training_history(
                    history, 
                    os.path.join(window_dir, 'training_history.png')
                )
            elif config.jump_model:

                logger.info("Training jump model...")
                model = train_model_for_window(
                    config,
                    window_info['train_period']['start'],
                    window_info['train_period']['end'],
                    window_info['valid_period']['start'],
                    window_info['valid_period']['end'],
                    data
                )

                if model is None or model is None:
                    logger.warning("Jump model training failed, skipping window")
                    continue

            else:
                model = train_model_for_window(
                    config,
                    window_info['train_period']['start'],
                    window_info['train_period']['end'],
                    window_info['valid_period']['start'],
                    window_info['valid_period']['end'],
                    data
                )
                
                # 학습 실패 시 다음 윈도우로 넘어감
                if model is None:
                    logger.warning("Model training failed, skipping window")
                    continue
            
            # 2. Identify regimes
            if config.rl_model:
                # RL 모델은 클러스터링 대신 에이전트가 직접 결정
                logger.info("Skipping regime identification for RL model (agent makes decisions)")
                # RL에서는 kmeans와 bull_regime이 필요 없음
                kmeans, bull_regime = None, None
            elif config.jump_model:
                # Jump model은 클러스터링 대신 에이전트가 직접 결정
                logger.info("Skipping regime identification for jump model (agent makes decisions)")
                # Jump model에서는 kmeans와 bull_regime이 필요 없음
                kmeans, bull_regime = None, None
            else:
                logger.info("Identifying regimes...")
                kmeans, bull_regime = identify_regimes_for_window(
                    config,
                    model,
                    data,
                    window_info['clustering_period']['start'],
                    window_info['clustering_period']['end']
                )
                
                # 레짐 식별 실패 시 다음 윈도우로 넘어감
                if kmeans is None or bull_regime is None:
                    logger.warning("Regime identification failed, skipping window")
                    continue
            
            # 3. Evaluate smoothing methods
            if config.rl_model:
                # RL 모델 평가 단계
                logger.info("Evaluating RL agent...")
                
                # 미래 기간에 대한 에이전트 평가
                results_df, performance = evaluate_rl_agent(
                    config, 
                    agent, 
                    data, 
                    window_info['forward_period']['start'], 
                    window_info['forward_period']['end'], 
                    window_number
                )
                
                if results_df is None or performance is None:
                    logger.warning("RL evaluation failed, skipping window")
                    continue
                
                # RL 결과를 smoothing methods와 동일한 형식으로 변환
                methods_results = {
                    'rl_agent': {
                        'method_id': 'rl_agent',
                        'method_name': 'rl_agent',
                        'params': {'type': 'ppo'},
                        'df': results_df,
                        'performance': performance,
                        'cum_return': performance['total_returns']['strategy'],
                        'n_trades': performance['position_changes'],
                        'sharpe': performance['sharpe_ratio']['strategy']
                    }
                }
                
                # 결과 저장
                results_df.to_csv(os.path.join(window_dir, 'rl_results.csv'), index=False)
                with open(os.path.join(window_dir, 'rl_performance.json'), 'w') as f:
                    json.dump(performance, f, default=json_serializer, indent=4)
                    
                # RL 결과 시각화
                from regime_mamba.utils.rl_visualize import visualize_rl_results
                visualize_rl_results(
                    results_df,
                    performance,
                    window_number,
                    f"Window {window_number}: {window_info['forward_period']['start']} ~ {window_info['forward_period']['end']}",
                    os.path.join(window_dir, 'rl_performance.png')
                )
            elif config.jump_model:
                # 미래 기간에 대한 에이전트 평가
                model.predict(
                    window_info['forward_period']['start'], 
                    window_info['forward_period']['end'], 
                    data, 
                    window_number,
                    sort="cumret"
                )
            else:
                logger.info("Evaluating smoothing methods...")
                methods_results = evaluate_smoothing_methods(
                    model,
                    data,
                    kmeans,
                    bull_regime,
                    config,
                    window_info['forward_period'],
                    window_dir,
                    config.max_workers
                )
            
            # 4. Save results
            if config.jump_model == True:
                print("Jump model results saved")
            elif methods_results:
                logger.info(f"Found {len(methods_results)} valid results")
                for method_id, result in methods_results.items():
                    combined_results[method_id]['window'].append(window_number)
                    combined_results[method_id]['returns'][window_number] = result['cum_return']
                    combined_results[method_id]['trades'][window_number] = result['n_trades']
                    combined_results[method_id]['sharpes'][window_number] = result['sharpe']
                    
                    # 성능 지표 저장 (깊은 복사)
                    combined_results[method_id]['performances'].append(copy.deepcopy(result['performance']))
            
                # 5. Save checkpoint
                if config.enable_checkpointing and (i + 1) % config.checkpoint_interval == 0:
                    checkpoint_data = {
                        'next_window': window_number + 1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'results': {}
                    }
                    
                    # Convert defaultdict to regular dict for JSON serialization
                    for method, result in combined_results.items():
                        checkpoint_data['results'][method] = {
                            'window': result['window'],
                            'returns': {str(k): v for k, v in result['returns'].items()},
                            'trades': {str(k): v for k, v in result['trades'].items()},
                            'sharpes': {str(k): v for k, v in result['sharpes'].items()},
                            'performances': result['performances']
                        }
                    
                    checkpoint_file = os.path.join(config.results_dir, 'checkpoint.json')
                    save_checkpoint(checkpoint_data, checkpoint_file)
                    logger.info(f"Checkpoint saved after window {window_number}")
            
        except Exception as e:
            logger.error(f"Error processing window {window_number}: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning(f"Skipping window {window_number}")
    
    # 6. Create final comparison
    logger.info("Creating final comparison...")
    if combined_results:
        # RL 모델과 일반 모델을 모두 포함한 비교 시각화
        summary = visualize_final_comparison(combined_results, config.results_dir)
        
        # Save combined results
        logger.info("Saving combined results...")
        result_data = {}
        for method, result in combined_results.items():
            result_data[method] = {
                'window': result['window'],
                'returns': {str(k): v for k, v in result['returns'].items()},
                'trades': {str(k): v for k, v in result['trades'].items()},
                'sharpes': {str(k): v for k, v in result['sharpes'].items()}
            }
        
        with open(os.path.join(config.results_dir, 'combined_results.json'), 'w') as f:
            json.dump(result_data, f, default=json_serializer, indent=4)
        
        # RL 모델 결과만 별도로 저장
        if config.rl_model:
            rl_summary = {k: v for k, v in result_data.items() if k.startswith('rl_')}
            with open(os.path.join(config.results_dir, 'rl_results.json'), 'w') as f:
                json.dump(rl_summary, f, default=json_serializer, indent=4)
        
        logger.info(f"Backtest complete with {len(result_data)} methods across {len(window_schedule)} windows")
        return {'combined_results': combined_results, 'summary': summary}
    else:
        logger.warning("No results to compare")
        return {'combined_results': combined_results, 'summary': None}

def main():
    """Main execution function"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Prepare output directory
        result_dir, log_file = prepare_output_directory(args.results_dir or './train_backtest_results')
        
        # Set up logging
        logger = setup_logging(log_file=log_file)
        logger.info("Starting rolling window train backtest")
        
        # Load configuration
        try:
            config = load_config(args)
            config.results_dir = result_dir
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            sys.exit(1)
        
        # Save configuration
        save_config(config, result_dir)
        logger.info(f"Configuration saved to {result_dir}")
        
        # Set random seed
        set_seed(config.seed)
        logger.info(f"Random seed set to {config.seed}")
        
        # Load and preprocess data
        try:
            logger.info(f"Loading data from {config.data_path}")
            data = pd.read_csv(config.data_path)
            logger.info(f"Loaded data with {len(data)} rows")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            sys.exit(1)
        
        # Run backtest
        checkpoint_path = args.checkpoint if args.checkpoint else None
        results = run_rolling_window_backtest(config, data, logger, checkpoint_path)
        
        # Log completion
        logger.info(f"Train backtest complete! Results saved to {result_dir}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()