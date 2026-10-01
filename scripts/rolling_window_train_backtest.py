#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rolling Window Training Backtest with Smoothing Technique Comparison
This script trains models in a rolling window fashion and compares various smoothing techniques.

Usage:
    python scripts/rolling_window_train_backtest.py --config regime_mamba/config/paper_config.yaml
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from typing import Any, Dict, Optional

import pandas as pd
import torch

from regime_mamba.config.config import RollingWindowTrainConfig
from regime_mamba.evaluate.backtest_runner import run_two_stage_window, run_windowed_backtest
from regime_mamba.features import FEATURE_SETS, prepare_feature_set
from regime_mamba.utils.io import (
    apply_overrides,
    load_yaml_config,
    prepare_output_directory,
    save_config_files,
    setup_logging,
)
from regime_mamba.utils.utils import set_seed


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
    parser.add_argument('--d_model', type=int, default=8, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=32, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--cluster_method', type=str, default='cosine_kmeans', help='Clustering method')
    parser.add_argument('--direct_train', action='store_true', help='Train model directly for clasification')

    # Input feature settings (regime_mamba/features.py)
    parser.add_argument('--feature_set', type=str, choices=FEATURE_SETS,
                        help='Mamba input features (default: CSV columns by --input_dim; sets input_dim automatically)')
    parser.add_argument('--extra_feature_cols', nargs='+', help='CSV columns appended to --feature_set (e.g. dollar_index)')
    parser.add_argument('--feature_return_col', type=str, help='Return column the feature set is computed from')

    # Training-related settings
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost (0.001 = 0.1%%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_onecycle', type=bool, default=True, help='Use one-cycle learning rate policy')

    # Extra Settings (jump model,and lstm)
    parser.add_argument('--jump_model', type=bool, default=False, help='Jump model flag')
    parser.add_argument('--jump_penalty', type=int, default=0, help='Jump penalty')
    parser.add_argument('--freeze_feature_extractor', type=bool, default=True, help='Freeze feature extractor during training')
    parser.add_argument('--window_size', type=int, default=252, help='Window size for Sharpe calculation')
    parser.add_argument('--lstm', action='store_true', help='Use LSTM model')
    parser.add_argument('--scale', type=int, default=1, help='Scaling Dollar Index')

    # Performance-related settings
    parser.add_argument('--max_workers', type=int, help='Maximum number of worker processes')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--enable_checkpointing', action='store_true', help='Enable checkpointing')
    parser.add_argument('--checkpoint_interval', type=int, help='Checkpoint interval (windows)')

    # Optional parameters
    parser.add_argument('--predict', default=False, help='Predict price to predict regime')

    return parser.parse_args()


def load_config(args) -> RollingWindowTrainConfig:
    """Load configuration from command-line arguments and a YAML file (YAML wins)

    Args:
        args: Command-line arguments

    Returns:
        RollingWindowTrainConfig: Configuration object
    """
    config = RollingWindowTrainConfig()

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

    apply_overrides(config, vars(args))
    apply_overrides(config, load_yaml_config(args.config), skip_none=False)

    if getattr(config, 'data_path', None) is None:
        raise ValueError("Missing required parameters: data_path")

    for key, value in defaults.items():
        if getattr(config, key, None) is None:
            setattr(config, key, value)

    # Device is kept as a string: the parallel smoothing evaluation checks `'cuda' in config.device`
    if config.gpu_id >= 0 and torch.cuda.is_available():
        config.device = f'cuda:{config.gpu_id}'
    else:
        config.device = 'cpu'

    return config


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
    # Methods are evaluated in a process pool on CPU (CUDA does not mix with multiprocessing)
    parallel = 'cuda' not in str(config.device)

    def process_window(window_info, window_dir):
        return run_two_stage_window(
            config, data, window_info, window_dir, logger,
            parallel=parallel, loader_workers=min(4, os.cpu_count() or 1)
        )

    return run_windowed_backtest(config, logger, process_window, checkpoint_path=checkpoint_path)


def main():
    """Main execution function"""
    logger = logging.getLogger(__name__)
    try:
        import matplotlib as mpl
        # usetex를 사용할 때 올바른 font.family 설정
        mpl.rcParams['font.family'] = 'serif'

        args = parse_args()
        print(args)
        result_dir, log_file = prepare_output_directory(args.results_dir or './train_backtest_results')

        logger = setup_logging(log_file=log_file)
        logger.info("Starting rolling window train backtest")

        try:
            config = load_config(args)
            config.results_dir = result_dir
            logger.info("Configuration loaded successfully")
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

        save_config_files(config, result_dir, title="Rolling Window Train Backtest")
        logger.info(f"Configuration saved to {result_dir}")

        set_seed(config.seed)
        logger.info(f"Random seed set to {config.seed}")

        run_rolling_window_backtest(config, data, logger, args.checkpoint)
        logger.info(f"Train backtest complete! Results saved to {result_dir}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
