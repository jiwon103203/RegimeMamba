#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rolling Window Training Backtest with Smoothing Technique Comparison
Supports both original 2-stage approach and End-to-End Regime Mamba.

Usage:
    # Original 2-stage approach
    python rolling_window_train_backtest_e2e.py --config config.yaml --data_path data.csv
    
    # End-to-End Regime Mamba
    python rolling_window_train_backtest_e2e.py --config config.yaml --data_path data.csv --e2e
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
from regime_mamba.data.dataset import DateRangeRegimeMambaDataset
from regime_mamba.evaluate.rolling_window_w_train import (
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

# E2E imports
from regime_mamba.config.e2e_config import E2ERegimeMambaConfig, E2EConfigPresets
from regime_mamba.models.e2e_regime_mamba import (
    EndToEndRegimeMamba, 
    create_e2e_model_from_config,
    align_regime_labels_numpy
)
from regime_mamba.train.e2e_train import (
    train_e2e_regime_mamba,
    evaluate_e2e_regime_strategy,
    TemperatureScheduler,
    HardnessScheduler
)


def json_serializer(obj):
    """JSON 직렬화를 위한 변환 함수 - 순환 참조 처리 및 다양한 타입 지원"""
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
        try:
            return str(obj)
        except:
            return None


def setup_logging(log_file=None, log_level=logging.INFO):
    """Set up logging configuration"""
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
    parser = argparse.ArgumentParser(
        description='Rolling Window Training Backtest with Smoothing Technique Comparison'
    )
    
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
    parser.add_argument('--d_model', type=int, default=8, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=32, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--cluster_method', type=str, default='cosine_kmeans', help='Clustering method')
    parser.add_argument('--direct_train', action='store_true', help='Train model directly for classification')

    # Training-related settings
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_onecycle', type=bool, default=True, help='Use one-cycle learning rate policy')

    # Extra Settings
    parser.add_argument('--jump_model', type=bool, default=False, help='Jump model flag')
    parser.add_argument('--jump_penalty', type=int, default=0, help='Jump penalty')
    parser.add_argument('--freeze_feature_extractor', type=bool, default=True, help='Freeze feature extractor')
    parser.add_argument('--window_size', type=int, default=252, help='Window size for Sharpe calculation')
    parser.add_argument('--lstm', action='store_true', help='Use LSTM model')
    parser.add_argument('--scale', type=int, default=1, help='Scaling Dollar Index')

    # Performance-related settings
    parser.add_argument('--max_workers', type=int, help='Maximum number of worker processes')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--enable_checkpointing', action='store_true', help='Enable checkpointing')
    parser.add_argument('--checkpoint_interval', type=int, help='Checkpoint interval (windows)')

    # E2E Regime Mamba specific arguments
    parser.add_argument('--e2e', action='store_true', help='Use End-to-End Regime Mamba')
    parser.add_argument('--e2e_preset', type=str, default='balanced',
                        choices=['aggressive', 'conservative', 'balanced', 'high_capacity', 'fast'],
                        help='E2E configuration preset')
    
    # E2E Gumbel Softmax parameters
    parser.add_argument('--initial_temp', type=float, default=2.0, help='Initial Gumbel temperature')
    parser.add_argument('--final_temp', type=float, default=0.5, help='Final Gumbel temperature')
    parser.add_argument('--temp_schedule', type=str, default='exponential',
                        choices=['linear', 'exponential', 'cosine'],
                        help='Temperature annealing schedule')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs before annealing')
    
    # E2E Loss weights
    parser.add_argument('--w_return', type=float, default=1.0, help='Return prediction loss weight')
    parser.add_argument('--w_direction', type=float, default=1.0, help='Direction prediction loss weight')
    parser.add_argument('--w_jump', type=float, default=1.0, help='Jump penalty loss weight')
    parser.add_argument('--w_separation', type=float, default=1.0, help='Regime separation loss weight')
    parser.add_argument('--w_entropy', type=float, default=0.1, help='Entropy regularization weight')
    
    # E2E Separation loss parameters
    parser.add_argument('--separation_loss_type', type=str, default='centroid',
                        choices=['centroid', 'contrastive', 'silhouette', 'return_weighted'],
                        help='Separation loss type')
    parser.add_argument('--separation_margin', type=float, default=1.0, help='Centroid separation margin')
    parser.add_argument('--lambda_inter', type=float, default=1.0, help='Inter-cluster separation weight')
    parser.add_argument('--lambda_intra', type=float, default=1.0, help='Intra-cluster compactness weight')

    # Optional parameters
    parser.add_argument('--predict', default=False, help='Predict price to predict regime')
    
    return parser.parse_args()


def load_config(args) -> RollingWindowTrainConfig:
    """Load configuration from file and command-line arguments"""
    # Use E2E config if --e2e flag is set
    if args.e2e:
        config = load_e2e_config(args)
    else:
        config = load_original_config(args)
    
    return config


def load_original_config(args) -> RollingWindowTrainConfig:
    """Load original 2-stage configuration"""
    config = RollingWindowTrainConfig()
    
    # Set default values
    defaults = {
        'results_dir': './train_backtest_results',
        'start_date': '1990-04-20',
        'end_date': '2023-12-31',
        'total_window_years': 54,
        'train_years': 50,
        'valid_years': 4,
        'clustering_years': 4,
        'forward_months': 24,
        'd_model': 8,
        'd_state': 32,
        'n_layers': 4,
        'dropout': 0.1,
        'seq_len': 60,
        'batch_size': 1024,
        'learning_rate': 5e-4,
        'max_epochs': 300,
        'patience': 50,
        'transaction_cost': 0.001,
        'max_workers': None,
        'gpu_id': 0,
        'enable_checkpointing': False,
        'checkpoint_interval': 1
    }

    # Update with command-line arguments
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Load from YAML file if provided
    yaml_config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    
    for key, value in yaml_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Check for required parameters
    required_params = ['data_path']
    missing_params = [param for param in required_params if getattr(config, param, None) is None]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
    
    # Fill in defaults
    for key, value in defaults.items():
        if getattr(config, key, None) is None:
            setattr(config, key, value)
    
    # Set device
    if config.gpu_id >= 0 and torch.cuda.is_available():
        config.device = torch.device(f'cuda:{config.gpu_id}')
    else:
        config.device = torch.device('cpu')
    
    return config


def load_e2e_config(args) -> E2ERegimeMambaConfig:
    """Load E2E Regime Mamba configuration"""
    # Get preset config
    preset_map = {
        'aggressive': E2EConfigPresets.aggressive_trading,
        'conservative': E2EConfigPresets.conservative_trading,
        'balanced': E2EConfigPresets.balanced,
        'high_capacity': E2EConfigPresets.high_capacity,
        'fast': E2EConfigPresets.fast_training
    }
    
    config = preset_map[args.e2e_preset]()
    
    # Set default values
    defaults = {
        'results_dir': './e2e_backtest_results',
        'start_date': '1990-04-20',
        'end_date': '2023-12-31',
        'total_window_years': 20,
        'train_years': 16,
        'valid_years': 4,
        'clustering_years': 0,  # Not used in E2E
        'forward_months': 24,
        'max_workers': None,
        'gpu_id': 0,
        'enable_checkpointing': False,
        'checkpoint_interval': 1
    }
    
    # Load from YAML file if provided
    yaml_config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    
    for key, value in yaml_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Update with command-line arguments (overrides YAML and preset)
    arg_dict = vars(args)
    e2e_params = [
        'data_path', 'results_dir', 'start_date', 'end_date',
        'total_window_years', 'train_years', 'valid_years', 'forward_months',
        'input_dim', 'd_model', 'd_state', 'n_layers', 'dropout', 'seq_len',
        'batch_size', 'learning_rate', 'max_epochs', 'patience', 'transaction_cost',
        'seed', 'max_workers', 'gpu_id', 'enable_checkpointing', 'checkpoint_interval',
        # E2E specific
        'initial_temp', 'final_temp', 'temp_schedule', 'warmup_epochs',
        'w_return', 'w_direction', 'w_jump', 'w_separation', 'w_entropy',
        'separation_loss_type', 'separation_margin', 'lambda_inter', 'lambda_intra',
        'jump_penalty'
    ]
    
    for key in e2e_params:
        value = arg_dict.get(key)
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Check for required parameters
    if getattr(config, 'data_path', None) is None:
        raise ValueError("Missing required parameter: data_path")
    
    # Fill in defaults
    for key, value in defaults.items():
        if getattr(config, key, None) is None:
            setattr(config, key, value)
    
    # Set device
    gpu_id = getattr(config, 'gpu_id', 0)
    if gpu_id >= 0 and torch.cuda.is_available():
        config.device = torch.device(f'cuda:{gpu_id}')
    else:
        config.device = torch.device('cpu')
    
    # E2E doesn't use clustering
    config.clustering_years = 0
    
    # Mark as E2E mode
    config.e2e_mode = True
    
    return config


def prepare_output_directory(output_dir: str, is_e2e: bool = False) -> tuple:
    """Create output directory structure with timestamped subdirectory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "e2e_backtest" if is_e2e else "train_backtest"
    result_dir = os.path.join(output_dir, f"{prefix}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    log_file = os.path.join(result_dir, f"{prefix}.log")
    
    return result_dir, log_file


def save_config(config, output_dir: str):
    """Save configuration to file"""
    config_dict = {key: getattr(config, key) for key in dir(config) 
                   if not key.startswith('__') and not callable(getattr(config, key))}
    
    # Convert device to string
    if 'device' in config_dict:
        config_dict['device'] = str(config_dict['device'])
    
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        is_e2e = getattr(config, 'e2e_mode', False)
        title = "E2E Regime Mamba" if is_e2e else "Rolling Window Train Backtest"
        f.write(f"=== {title} Configuration ===\n\n")
        for key, value in sorted(config_dict.items()):
            f.write(f"{key}: {value}\n")


def get_smoothing_methods() -> List[Tuple[str, Dict[str, Any]]]:
    """Get list of smoothing methods to evaluate"""
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
    """Apply a specific smoothing method to raw predictions"""
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


# ============================================================================
# E2E Regime Mamba specific functions
# ============================================================================

def train_e2e_model_for_window(
    config: E2ERegimeMambaConfig,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    data: pd.DataFrame,
    window_number: int = 1
) -> Optional[EndToEndRegimeMamba]:
    """
    Train E2E Regime Mamba model for a specific window.
    
    Args:
        config: E2E configuration object
        train_start: Training start date
        train_end: Training end date
        valid_start: Validation start date
        valid_end: Validation end date
        data: Full dataframe
        window_number: Window number
        
    Returns:
        Trained E2E model or None if training failed
    """
    print(f"\n[E2E] Training period: {train_start} ~ {train_end}")
    print(f"[E2E] Validation period: {valid_start} ~ {valid_end}")
    
    # Create datasets
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
    
    # Check data availability
    if len(train_dataset) < 100 or len(valid_dataset) < 50:
        print(f"[E2E] Warning: Insufficient data. Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
        return None
    
    print(f"[E2E] Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    
    # Create data loaders
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
    
    # Create window save directory
    window_save_dir = os.path.join(config.results_dir, f'window_{window_number}', 'model')
    os.makedirs(window_save_dir, exist_ok=True)
    
    # Create model
    model = create_e2e_model_from_config(config)
    
    # Train model
    model, history = train_e2e_regime_mamba(
        model,
        train_loader,
        valid_loader,
        config,
        save_dir=window_save_dir
    )
    
    print(f"[E2E] Training complete. Best validation loss: {min(history['valid_loss']):.6f}")
    
    return model


def predict_e2e_regimes(
    model: EndToEndRegimeMamba,
    dataloader: DataLoader,
    config: E2ERegimeMambaConfig
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Predict regimes using E2E model.
    
    Args:
        model: Trained E2E model
        dataloader: Data loader
        config: Configuration object
        
    Returns:
        Tuple of (predictions, returns, dates)
    """
    device = config.device
    model.eval()
    
    all_predictions = []
    all_returns = []
    all_dates = []
    
    with torch.no_grad():
        for x, y, dates, returns in dataloader:
            x = x.to(device)
            
            # Get regime predictions
            outputs = model.forward(x, hard=True)
            regime_probs = outputs['regime_probs'].cpu().numpy()
            
            # Get raw regime assignments
            predictions = regime_probs.argmax(axis=-1)
            
            all_predictions.extend(predictions.flatten())
            all_returns.extend(returns.numpy().flatten())
            all_dates.extend(dates)
    
    predictions = np.array(all_predictions)
    returns = np.array(all_returns)
    
    # Align regime labels: Regime 1 = Bull (higher returns), Regime 0 = Bear
    predictions = align_regime_labels_numpy(predictions, returns)
    
    return predictions, returns, all_dates


def evaluate_e2e_method(
    model: EndToEndRegimeMamba,
    data: pd.DataFrame,
    method_info: Tuple[str, Dict[str, Any]],
    config: E2ERegimeMambaConfig,
    forward_period: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single smoothing method for E2E model.
    
    Args:
        model: Trained E2E model
        data: Full dataframe
        method_info: Tuple of (method_name, parameters)
        config: Configuration object
        forward_period: Dictionary with forward start and end dates
        
    Returns:
        Result dictionary or None
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
        
        if len(forward_dataset) < 10:
            return None
        
        forward_loader = DataLoader(
            forward_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Get E2E regime predictions
        raw_predictions, true_returns, dates = predict_e2e_regimes(
            model, forward_loader, config
        )
        
        # Apply smoothing
        smoothed_predictions = apply_smoothing_method(raw_predictions, method_name, params)
        
        # Evaluate strategy
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
            
            results_df['raw_regime'] = raw_predictions.flatten()
            
            # Calculate trade metrics
            raw_trades = (np.diff(raw_predictions.flatten()) != 0).sum() + (raw_predictions[0] == 1)
            smoothed_trades = (np.diff(smoothed_predictions.flatten()) != 0).sum() + (smoothed_predictions[0] == 1)
            
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
        logging.error(f"[E2E] Error evaluating method {method_name}: {str(e)}")
        traceback.print_exc()
        return None


def evaluate_e2e_smoothing_methods(
    model: EndToEndRegimeMamba,
    data: pd.DataFrame,
    config: E2ERegimeMambaConfig,
    forward_period: Dict[str, str],
    window_results_dir: str
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple smoothing methods for E2E model.
    
    Args:
        model: Trained E2E model
        data: Full dataframe
        config: Configuration object
        forward_period: Forward period dict
        window_results_dir: Window results directory
        
    Returns:
        Dictionary of method results
    """
    smoothing_methods = get_smoothing_methods()
    all_methods_results = {}
    
    for method_info in tqdm(smoothing_methods, desc="[E2E] Evaluating methods"):
        result = evaluate_e2e_method(model, data, method_info, config, forward_period)
        
        if result:
            all_methods_results[result['method_id']] = result
            
            # Save individual result
            method_dir = os.path.join(window_results_dir, result['method_id'])
            os.makedirs(method_dir, exist_ok=True)
            result['df'].to_csv(os.path.join(method_dir, 'results.csv'), index=False)
            
            with open(os.path.join(method_dir, 'performance.json'), 'w') as f:
                json.dump(result['performance'], f, default=json_serializer, indent=4)
    
    # Create comparison visualization
    if all_methods_results:
        visualize_methods_comparison(
            all_methods_results,
            os.path.join(window_results_dir, 'methods_comparison.png'),
            forward_period
        )
    
    return all_methods_results


# ============================================================================
# Original 2-stage functions (unchanged from original file)
# ============================================================================

def evaluate_method(
    model,
    data: pd.DataFrame, 
    kmeans,
    bull_regime: int,
    method_info: Tuple[str, Dict[str, Any]],
    config: RollingWindowTrainConfig,
    forward_period: Dict[str, str]
) -> Dict[str, Any]:
    """Evaluate a single smoothing method (original 2-stage)"""
    method_name, params = method_info
    forward_start, forward_end = forward_period['start'], forward_period['end']
    
    try:
        forward_dataset = DateRangeRegimeMambaDataset(
            data=data, 
            seq_len=config.seq_len,
            start_date=forward_start,
            end_date=forward_end,
            config=config
        )
        
        forward_loader = DataLoader(
            forward_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1)
        )
        
        if len(forward_dataset) < 10:
            return None
        
        raw_predictions, true_returns, dates = predict_regimes(
            model, forward_loader, kmeans, bull_regime, config
        )
        
        smoothed_predictions = apply_smoothing_method(raw_predictions, method_name, params)
        
        results_df, performance = evaluate_regime_strategy(
            smoothed_predictions,
            true_returns,
            dates,
            transaction_cost=config.transaction_cost,
            config=config
        )
        
        if results_df is not None and performance is not None:
            results_df['smoothing_method'] = method_name
            for param_name, param_value in params.items():
                results_df[f'smoothing_{param_name}'] = param_value
            
            results_df['raw_regime'] = raw_predictions.flatten()
            
            raw_trades = (np.diff(raw_predictions.flatten()) != 0).sum() + (raw_predictions[0] == 1)
            smoothed_trades = (np.diff(smoothed_predictions.flatten()) != 0).sum() + (smoothed_predictions[0] == 1)
            
            performance['smoothing_method'] = method_name
            for param_name, param_value in params.items():
                performance[f'smoothing_{param_name}'] = param_value
            performance['raw_trades'] = int(raw_trades)
            performance['smoothed_trades'] = int(smoothed_trades)
            performance['trade_reduction'] = int(raw_trades - smoothed_trades)
            performance['trade_reduction_pct'] = ((raw_trades - smoothed_trades) / raw_trades * 100) if raw_trades > 0 else 0
            
            param_str = '_'.join([f"{k}={v}" for k, v in params.items()]) if params else "default"
            method_id = f"{method_name}_{param_str}" if params else method_name
            
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
    """Evaluate multiple smoothing methods (original 2-stage)"""
    smoothing_methods = get_smoothing_methods()
    all_methods_results = {}
    
    for method_info in tqdm(smoothing_methods, desc="Evaluating methods"):
        result = evaluate_method(model, data, kmeans, bull_regime, method_info, config, forward_period)
        if result:
            all_methods_results[result['method_id']] = result
            
            method_dir = os.path.join(window_results_dir, result['method_id'])
            os.makedirs(method_dir, exist_ok=True)
            result['df'].to_csv(os.path.join(method_dir, 'results.csv'), index=False)
            
            with open(os.path.join(method_dir, 'performance.json'), 'w') as f:
                json.dump(result['performance'], f, default=json_serializer, indent=4)
    
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
    """Visualize comparison of smoothing methods"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    
    first_method = list(methods_results.keys())[0]
    plt.plot(
        methods_results[first_method]['df']['Cum_Market'] * 100, 
        label='Market', 
        color='gray', 
        linestyle='--'
    )
    
    for method_id, result in methods_results.items():
        plt.plot(result['df']['Cum_Strategy'] * 100, label=method_id)
    
    plt.title(f"Comparison of Smoothing Methods ({forward_period['start']} to {forward_period['end']})")
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    
    method_ids = list(methods_results.keys())
    returns = [methods_results[method_id]['cum_return'] for method_id in method_ids]
    trades = [methods_results[method_id]['n_trades'] for method_id in method_ids]
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(np.arange(len(method_ids)) - 0.2, returns, width=0.4, color='blue', alpha=0.7)
    ax1.set_ylabel('Returns (%)', color='blue')
    ax1.tick_params(axis='y', colors='blue')
    
    bars2 = ax2.bar(np.arange(len(method_ids)) + 0.2, trades, width=0.4, color='red', alpha=0.7)
    ax2.set_ylabel('Number of Trades', color='red')
    ax2.tick_params(axis='y', colors='red')
    
    plt.xticks(np.arange(len(method_ids)), method_ids, rotation=45, ha='right')
    plt.title('Returns vs. Number of Trades by Method')
    plt.grid(True, alpha=0.3)
    
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
    """Visualize final comparison of smoothing methods across all windows"""
    all_methods = sorted(list(combined_results.keys()))
    all_windows = sorted(list(set(combined_results[all_methods[0]]['window'])))
    
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
    
    sorted_methods = sorted(all_methods, key=lambda x: method_metrics[x]['avg_return'], reverse=True)
    
    # 1. Average metrics comparison
    plt.figure(figsize=(15, 12))
    
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
    
    plt.subplot(2, 2, 4)
    plt.scatter(
        [method_metrics[method]['avg_trades'] for method in all_methods],
        [method_metrics[method]['avg_return'] for method in all_methods],
        color='purple', 
        alpha=0.7
    )
    
    for method in all_methods:
        plt.annotate(
            method, 
            (method_metrics[method]['avg_trades'], method_metrics[method]['avg_return']), 
            textcoords="offset points", 
            xytext=(0, 5), 
            ha='center',
            fontsize=8
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
    
    from collections import Counter
    best_method_counts = Counter(best_methods)
    sorted_best_methods = sorted(best_method_counts.items(), key=lambda x: x[1], reverse=True)
    
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
    
    # 4. Create summary
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
    
    for method in all_methods:
        summary['methods'][method] = {
            'avg_return': float(method_metrics[method]['avg_return']),
            'avg_trades': float(method_metrics[method]['avg_trades']),
            'avg_sharpe': float(method_metrics[method]['avg_sharpe']),
            'best_window_count': best_method_counts.get(method, 0)
        }
    
    with open(os.path.join(save_dir, 'methods_summary.json'), 'w') as f:
        json.dump(summary, f, default=json_serializer, indent=4)
    
    logging.info("\n===== Smoothing Method Performance Summary =====")
    logging.info("Top methods by average return:")
    for i, method in enumerate(sorted_methods[:3]):
        logging.info(f"  {i+1}. {method}: {method_metrics[method]['avg_return']:.2f}%")
    
    logging.info("\nTop methods by average Sharpe ratio:")
    sharpe_sorted = sorted(all_methods, key=lambda x: method_metrics[x]['avg_sharpe'], reverse=True)
    for i, method in enumerate(sharpe_sorted[:3]):
        logging.info(f"  {i+1}. {method}: {method_metrics[method]['avg_sharpe']:.2f}")
    
    return summary


def create_window_schedule(
    config, 
    start_from_window: int = 1
) -> List[Dict[str, Any]]:
    """Create window schedule for rolling window backtest"""
    window_schedule = []
    
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    
    window_number = 1
    
    while current_date <= end_date:
        train_start = (current_date - relativedelta(years=config.total_window_years)).strftime('%Y-%m-%d')
        train_end = (current_date - relativedelta(years=config.valid_years + config.clustering_years)).strftime('%Y-%m-%d')
        
        valid_start = (current_date - relativedelta(years=config.valid_years)).strftime('%Y-%m-%d')
        valid_end = current_date.strftime('%Y-%m-%d')
        
        clustering_start = (current_date - relativedelta(years=config.clustering_years)).strftime('%Y-%m-%d')
        clustering_end = current_date.strftime('%Y-%m-%d')
        
        forward_start = current_date.strftime('%Y-%m-%d')
        forward_end = (current_date + relativedelta(months=config.forward_months)).strftime('%Y-%m-%d')
        
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
        
        current_date += relativedelta(months=config.forward_months)
        window_number += 1
    
    return window_schedule


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    return checkpoint


def save_checkpoint(checkpoint_data: Dict[str, Any], checkpoint_path: str):
    """Save checkpoint"""
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, default=json_serializer, indent=4)
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")


def run_rolling_window_backtest(
    config,
    data: pd.DataFrame,
    logger: logging.Logger,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run rolling window backtest (supports both original and E2E modes)"""
    
    is_e2e = getattr(config, 'e2e_mode', False)
    
    combined_results = defaultdict(lambda: {
        'window': [],
        'returns': {},
        'trades': {},
        'sharpes': {},
        'performances': []
    })
    
    start_from_window = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = load_checkpoint(checkpoint_path)
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
    
    window_schedule = create_window_schedule(config, start_from_window)
    
    if not window_schedule:
        logger.info("No windows to process")
        return {'combined_results': combined_results, 'summary': None}
    
    total_windows = len(window_schedule)
    mode_str = "[E2E]" if is_e2e else "[2-Stage]"
    logger.info(f"{mode_str} Processing {total_windows} windows")
    
    for i, window_info in enumerate(window_schedule):
        window_number = window_info['window_number']
        logger.info(f"\n=== {mode_str} Window {window_number} ({i+1}/{total_windows}) ===")
        
        window_dir = os.path.join(config.results_dir, f"window_{window_number}")
        os.makedirs(window_dir, exist_ok=True)
        
        logger.info(f"Training: {window_info['train_period']['start']} ~ {window_info['train_period']['end']}")
        logger.info(f"Validation: {window_info['valid_period']['start']} ~ {window_info['valid_period']['end']}")
        logger.info(f"Forward: {window_info['forward_period']['start']} ~ {window_info['forward_period']['end']}")
        
        try:
            if is_e2e:
                # =====================
                # E2E Regime Mamba Flow
                # =====================
                logger.info(f"{mode_str} Training E2E model...")
                model = train_e2e_model_for_window(
                    config,
                    window_info['train_period']['start'],
                    window_info['train_period']['end'],
                    window_info['valid_period']['start'],
                    window_info['valid_period']['end'],
                    data,
                    window_number=window_number
                )
                
                if model is None:
                    logger.warning(f"{mode_str} Model training failed, skipping window")
                    continue
                
                # E2E doesn't need separate regime identification
                logger.info(f"{mode_str} Evaluating smoothing methods...")
                methods_results = evaluate_e2e_smoothing_methods(
                    model,
                    data,
                    config,
                    window_info['forward_period'],
                    window_dir
                )
                
            else:
                # =====================
                # Original 2-Stage Flow
                # =====================
                if config.jump_model:
                    logger.info("Training jump model...")
                    model = train_model_for_window(
                        config,
                        window_info['train_period']['start'],
                        window_info['train_period']['end'],
                        window_info['valid_period']['start'],
                        window_info['valid_period']['end'],
                        data,
                        window_number=window_number
                    )
                    
                    if model is None:
                        logger.warning("Jump model training failed, skipping window")
                        continue
                    
                    model.predict(
                        window_info['forward_period']['start'], 
                        window_info['forward_period']['end'], 
                        data, 
                        window_number,
                        sort="cumret"
                    )
                    continue
                    
                else:
                    logger.info("Training model...")
                    model, _ = train_model_for_window(
                        config,
                        window_info['train_period']['start'],
                        window_info['train_period']['end'],
                        window_info['valid_period']['start'],
                        window_info['valid_period']['end'],
                        data,
                        window_number=window_number
                    )
                    
                    if model is None:
                        logger.warning("Model training failed, skipping window")
                        continue
                    
                    logger.info("Identifying regimes...")
                    kmeans, bull_regime = identify_regimes_for_window(
                        config,
                        model,
                        data,
                        window_info['clustering_period']['start'],
                        window_info['clustering_period']['end']
                    )
                    
                    if kmeans is None or bull_regime is None:
                        logger.warning("Regime identification failed, skipping window")
                        continue
                    
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
            
            # Save results (common for both modes)
            if methods_results:
                logger.info(f"Found {len(methods_results)} valid results")
                for method_id, result in methods_results.items():
                    combined_results[method_id]['window'].append(window_number)
                    combined_results[method_id]['returns'][window_number] = result['cum_return']
                    combined_results[method_id]['trades'][window_number] = result['n_trades']
                    combined_results[method_id]['sharpes'][window_number] = result['sharpe']
                    combined_results[method_id]['performances'].append(copy.deepcopy(result['performance']))
            
                # Save checkpoint
                if config.enable_checkpointing and (i + 1) % config.checkpoint_interval == 0:
                    checkpoint_data = {
                        'next_window': window_number + 1,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'mode': 'e2e' if is_e2e else '2-stage',
                        'results': {}
                    }
                    
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
    
    # Create final comparison
    logger.info("Creating final comparison...")
    if combined_results:
        summary = visualize_final_comparison(combined_results, config.results_dir)
        
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
        
        logger.info(f"Backtest complete with {len(result_data)} methods across {len(window_schedule)} windows")
        return {'combined_results': combined_results, 'summary': summary}
    else:
        logger.warning("No results to compare")
        return {'combined_results': combined_results, 'summary': None}


def main():
    """Main execution function"""
    try:
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'serif'
        
        args = parse_args()
        print(f"\nMode: {'E2E Regime Mamba' if args.e2e else 'Original 2-Stage'}")
        print(f"Arguments: {args}\n")
        
        # Prepare output directory
        default_dir = './e2e_backtest_results' if args.e2e else './train_backtest_results'
        result_dir, log_file = prepare_output_directory(
            args.results_dir or default_dir,
            is_e2e=args.e2e
        )
        
        # Set up logging
        logger = setup_logging(log_file=log_file)
        logger.info(f"Starting {'E2E' if args.e2e else '2-Stage'} rolling window backtest")
        
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
        
        # Load data
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
        
        logger.info(f"Backtest complete! Results saved to {result_dir}")
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
