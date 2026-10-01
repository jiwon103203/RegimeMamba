#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rolling Window Training Backtest with Smoothing Technique Comparison
Supports both original 2-stage approach and End-to-End Regime Mamba.

Updated: Two-Level Entropy Regularization, Direction Loss removed

Usage:
    # Original 2-stage approach
    python scripts/rolling_window_train_backtest_e2e.py --config config.yaml --data_path data.csv

    # End-to-End Regime Mamba
    python scripts/rolling_window_train_backtest_e2e.py --config config.yaml --data_path data.csv --e2e

    # E2E with high confidence preset (strong symmetry breaking)
    python scripts/rolling_window_train_backtest_e2e.py --data_path data.csv --e2e --e2e_preset high_confidence
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from regime_mamba.config.config import RollingWindowTrainConfig
from regime_mamba.config.e2e_config import E2ERegimeMambaConfig, E2EConfigPresets
from regime_mamba.data.dataset import DateRangeRegimeMambaDataset
from regime_mamba.evaluate.backtest_runner import run_two_stage_window, run_windowed_backtest
from regime_mamba.features import FEATURE_SETS, prepare_feature_set, standardize_for_window
from regime_mamba.evaluate.smoothing_eval import evaluate_smoothing_methods
from regime_mamba.models.e2e_regime_mamba import (
    EndToEndRegimeMamba,
    create_e2e_model_from_config,
    align_regime_labels_numpy
)
from regime_mamba.train.e2e_train import train_e2e_regime_mamba
from regime_mamba.utils.io import (
    apply_overrides,
    load_yaml_config,
    prepare_output_directory,
    save_config_files,
    setup_logging,
)
from regime_mamba.utils.utils import set_seed

E2E_ENTROPY_PARAMS = ['lambda_entropy', 'lambda_sample_entropy', 'lambda_batch_entropy', 'w_entropy']
E2E_LOSS_PARAMS = ['w_return', 'w_jump', 'w_separation']


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

    # Input feature settings (regime_mamba/features.py)
    parser.add_argument('--feature_set', type=str, choices=FEATURE_SETS,
                        help='Mamba input features (default: CSV columns by --input_dim; sets input_dim automatically)')
    parser.add_argument('--extra_feature_cols', nargs='+', help='CSV columns appended to --feature_set (e.g. dollar_index)')
    parser.add_argument('--feature_return_col', type=str, help='Return column the feature set is computed from')

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
                        choices=['aggressive', 'conservative', 'balanced', 'high_capacity', 'fast',
                                 'high_confidence', 'strong_separation', 'debug_symmetry'],
                        help='E2E configuration preset')
    
    # E2E Gumbel Softmax parameters
    parser.add_argument('--initial_temp', type=float, default=1.0, help='Initial Gumbel temperature')
    parser.add_argument('--final_temp', type=float, default=0.3, help='Final Gumbel temperature')
    parser.add_argument('--temp_schedule', type=str, default='exponential',
                        choices=['linear', 'exponential', 'cosine'],
                        help='Temperature annealing schedule')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs before annealing')
    
    # E2E Loss weights (Updated: NO w_direction)
    parser.add_argument('--w_return', type=float, default=0.5, help='Return prediction loss weight')
    parser.add_argument('--w_jump', type=float, default=0.5, help='Jump penalty loss weight')
    parser.add_argument('--w_separation', type=float, default=2.0, help='Regime separation loss weight')
    parser.add_argument('--w_entropy', type=float, default=1.0, help='Two-level entropy weight (INCREASED)')
    
    # E2E Two-Level Entropy parameters (NEW)
    parser.add_argument('--lambda_entropy', type=float, default=1.0, 
                        help='Overall entropy regularization coefficient')
    parser.add_argument('--lambda_sample_entropy', type=float, default=1.0,
                        help='Sample-level entropy weight (minimize for confidence)')
    parser.add_argument('--lambda_batch_entropy', type=float, default=0.5,
                        help='Batch-level entropy weight (maximize for balance)')
    
    # E2E Separation loss parameters
    parser.add_argument('--separation_loss_type', type=str, default='centroid',
                        choices=['centroid', 'contrastive', 'silhouette', 'return_weighted'],
                        help='Separation loss type')
    parser.add_argument('--separation_margin', type=float, default=2.0, help='Centroid separation margin')
    parser.add_argument('--lambda_inter', type=float, default=1.5, help='Inter-cluster separation weight')
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

    # Command-line arguments first, then the YAML file (YAML wins)
    apply_overrides(config, vars(args))
    apply_overrides(config, load_yaml_config(args.config), skip_none=False)
    
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
    """Load E2E Regime Mamba configuration with Two-Level Entropy support"""
    # Get preset config (Updated preset map)
    preset_map = {
        'aggressive': E2EConfigPresets.aggressive_trading,
        'conservative': E2EConfigPresets.conservative_trading,
        'balanced': E2EConfigPresets.balanced,
        'high_capacity': E2EConfigPresets.high_capacity,
        'fast': E2EConfigPresets.fast_training,
        'high_confidence': E2EConfigPresets.high_confidence,
        'strong_separation': E2EConfigPresets.strong_separation,
        'debug_symmetry': E2EConfigPresets.debug_symmetry_breaking
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
    apply_overrides(config, load_yaml_config(args.config), skip_none=False)
    
    # Update with command-line arguments (overrides YAML and preset)
    # Updated: removed w_direction, added Two-Level Entropy params
    arg_dict = vars(args)
    e2e_params = [
        'data_path', 'results_dir', 'start_date', 'end_date',
        'total_window_years', 'train_years', 'valid_years', 'forward_months',
        'input_dim', 'd_model', 'd_state', 'n_layers', 'dropout', 'seq_len',
        'batch_size', 'learning_rate', 'max_epochs', 'patience', 'transaction_cost',
        'seed', 'max_workers', 'gpu_id', 'enable_checkpointing', 'checkpoint_interval',
        # E2E Gumbel Softmax
        'initial_temp', 'final_temp', 'temp_schedule', 'warmup_epochs',
        # E2E Loss weights (NO w_direction)
        'w_return', 'w_jump', 'w_separation', 'w_entropy',
        # Two-Level Entropy (NEW)
        'lambda_entropy', 'lambda_sample_entropy', 'lambda_batch_entropy',
        # Separation loss
        'separation_loss_type', 'separation_margin', 'lambda_inter', 'lambda_intra',
        'jump_penalty',
        # Input features
        'feature_set', 'extra_feature_cols', 'feature_return_col'
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


def save_config(config, output_dir: str):
    """Save configuration to file (E2E runs list entropy / loss parameters first)"""
    if getattr(config, 'e2e_mode', False):
        save_config_files(
            config, output_dir,
            title="E2E Regime Mamba (Two-Level Entropy)",
            sections=[
                ("Two-Level Entropy Parameters", E2E_ENTROPY_PARAMS),
                ("Loss Weights (No Direction Loss)", E2E_LOSS_PARAMS),
            ],
            sort_keys=True
        )
    else:
        save_config_files(config, output_dir, title="Rolling Window Train Backtest", sort_keys=True)


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
    print(f"[E2E] Two-Level Entropy: λ_sample={config.lambda_sample_entropy}, λ_batch={config.lambda_batch_entropy}")
    
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
        shuffle=False,
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
    
    # Log training results with Two-Level Entropy info
    best_val_loss = min(history['valid_loss']) if history['valid_loss'] else float('inf')
    final_sample_entropy = history['sample_entropy'][-1] if history.get('sample_entropy') else None
    final_batch_entropy = history['batch_entropy'][-1] if history.get('batch_entropy') else None
    
    print(f"[E2E] Training complete. Best validation loss: {best_val_loss:.6f}")
    if final_sample_entropy is not None:
        print(f"[E2E] Final sample entropy: {final_sample_entropy:.4f} (want LOW)")
        print(f"[E2E] Final batch entropy: {final_batch_entropy:.4f} (want HIGH ≈ 0.693)")
        
        # Warning if model might be stuck
        if final_sample_entropy > 0.6:
            print(f"[E2E] ⚠️ Warning: High sample entropy suggests model may be stuck at [0.5, 0.5]")
    
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


def run_rolling_window_backtest(
    config,
    data: pd.DataFrame,
    logger: logging.Logger,
    checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """Run rolling window backtest (supports both original and E2E modes)"""
    is_e2e = getattr(config, 'e2e_mode', False)
    mode_str = "[E2E]" if is_e2e else "[2-Stage]"

    if is_e2e:
        logger.info(f"{mode_str} Two-Level Entropy: λ_sample={config.lambda_sample_entropy}, λ_batch={config.lambda_batch_entropy}")
        logger.info(f"{mode_str} Loss weights: return={config.w_return}, jump={config.w_jump}, sep={config.w_separation}, ent={config.w_entropy}")

    def process_window(window_info, window_dir):
        if not is_e2e:
            return run_two_stage_window(config, data, window_info, window_dir, logger)

        # feature_set 모드: 이 윈도우의 학습 구간 통계로 입력 피처를 표준화 (기존 모드는 그대로)
        window_data = standardize_for_window(
            data, config, window_info['train_period']['start'], window_info['train_period']['end']
        )

        logger.info(f"{mode_str} Training E2E model with Two-Level Entropy...")
        model = train_e2e_model_for_window(
            config,
            window_info['train_period']['start'],
            window_info['train_period']['end'],
            window_info['valid_period']['start'],
            window_info['valid_period']['end'],
            window_data,
            window_number=window_info['window_number']
        )
        if model is None:
            logger.warning(f"{mode_str} Model training failed, skipping window")
            return None

        # E2E doesn't need separate regime identification
        logger.info(f"{mode_str} Evaluating smoothing methods...")
        return evaluate_smoothing_methods(
            partial(predict_e2e_regimes, model, config=config),
            window_data,
            config,
            window_info['forward_period'],
            window_dir,
            log_prefix="[E2E] "
        )

    return run_windowed_backtest(
        config, logger, process_window,
        checkpoint_path=checkpoint_path,
        use_clustering=not is_e2e,
        log_prefix=f"{mode_str} ",
        checkpoint_extra={'mode': 'e2e' if is_e2e else '2-stage'}
    )


def main():
    """Main execution function"""
    try:
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'serif'

        args = parse_args()

        mode_name = 'E2E Regime Mamba (Two-Level Entropy)' if args.e2e else 'Original 2-Stage'
        print(f"\n{'='*60}")
        print(f"Mode: {mode_name}")
        if args.e2e:
            print(f"Preset: {args.e2e_preset}")
            print(f"Two-Level Entropy: λ_sample={args.lambda_sample_entropy}, λ_batch={args.lambda_batch_entropy}")
        print(f"{'='*60}\n")

        default_dir = './e2e_backtest_results' if args.e2e else './train_backtest_results'
        result_dir, log_file = prepare_output_directory(
            args.results_dir or default_dir,
            prefix="e2e_backtest" if args.e2e else "train_backtest"
        )

        logger = setup_logging(log_file=log_file)
        logger.info(f"Starting {mode_name} rolling window backtest")

        try:
            config = load_config(args)
            config.results_dir = result_dir
            logger.info("Configuration loaded successfully")
            if args.e2e:
                logger.info(f"Two-Level Entropy Config:")
                logger.info(f"  lambda_sample_entropy: {config.lambda_sample_entropy}")
                logger.info(f"  lambda_batch_entropy: {config.lambda_batch_entropy}")
                logger.info(f"  w_entropy: {config.w_entropy}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            sys.exit(1)

        try:
            logger.info(f"Loading data from {config.data_path}")
            data = pd.read_csv(config.data_path)
            logger.info(f"Loaded data with {len(data)} rows")
            # feature_set 이 있으면 피처를 계산해 추가하고 input_dim / feature_cols 를 설정
            data = prepare_feature_set(data, config, logger)
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            sys.exit(1)

        save_config(config, result_dir)
        logger.info(f"Configuration saved to {result_dir}")

        set_seed(config.seed)
        logger.info(f"Random seed set to {config.seed}")

        run_rolling_window_backtest(config, data, logger, args.checkpoint)
        logger.info(f"Backtest complete! Results saved to {result_dir}")

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
