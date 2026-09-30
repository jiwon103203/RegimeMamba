#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rolling Window Training Backtest with RL Regime Mamba

Uses Reinforcement Learning (Actor-Critic) for end-to-end regime detection.
Replaces Jump Model's Dynamic Programming with deep RL.

NOT RUNNABLE YET: the RL training loop (``regime_mamba/train/rl_train.py`` with
``train_rl_regime_mamba``) has not been implemented. The model itself lives in
``regime_mamba/models/rl_regime_mamba.py``; ``main()`` stops with a clear error
until the training module is added.

Usage:
    # Default PPO configuration
    python scripts/rolling_window_train_backtest_rl.py --data_path data.csv

    # Conservative trading preset
    python scripts/rolling_window_train_backtest_rl.py --data_path data.csv --rl_preset conservative

    # A2C algorithm
    python scripts/rolling_window_train_backtest_rl.py --data_path data.csv --rl_algorithm a2c

    # Custom reward design
    python scripts/rolling_window_train_backtest_rl.py --data_path data.csv --reward_type sharpe --transaction_penalty 0.002
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

from regime_mamba.config.rl_config import RLRegimeMambaConfig, RLConfigPresets
from regime_mamba.data.dataset import DateRangeRegimeMambaDataset
from regime_mamba.evaluate.backtest_runner import run_windowed_backtest
from regime_mamba.evaluate.smoothing_eval import evaluate_smoothing_methods
from regime_mamba.models.rl_regime_mamba import (
    RLRegimeMamba,
    create_rl_model_from_config
)
from regime_mamba.utils.io import (
    apply_overrides,
    load_yaml_config,
    prepare_output_directory,
    save_config_files,
    setup_logging,
)
from regime_mamba.utils.utils import set_seed

try:
    from regime_mamba.train.rl_train import train_rl_regime_mamba
except ImportError:  # 미구현: RL 학습 루프가 아직 없음
    train_rl_regime_mamba = None

RL_REWARD_PARAMS = ['reward_type', 'reward_scale', 'transaction_penalty', 'holding_bonus', 'drawdown_penalty']


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Rolling Window Training Backtest with RL Regime Mamba'
    )
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--data_path', type=str, required=True, help='Data file path')
    parser.add_argument('--results_dir', type=str, default='./rl_backtest_results',
                        help='Results directory')
    
    # Date parameters
    parser.add_argument('--start_date', type=str, default='1990-04-20',
                        help='Backtest start date')
    parser.add_argument('--end_date', type=str, default='2023-12-31',
                        help='Backtest end date')
    
    # Period parameters
    parser.add_argument('--total_window_years', type=int, default=20,
                        help='Total data period (years)')
    parser.add_argument('--train_years', type=int, default=16,
                        help='Training period (years)')
    parser.add_argument('--valid_years', type=int, default=4,
                        help='Validation period (years)')
    parser.add_argument('--forward_months', type=int, default=24,
                        help='Forward testing period (months)')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, default=4, help='Input dimension')
    parser.add_argument('--d_model', type=int, default=8, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=32, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of Mamba layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=60, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    
    # RL algorithm selection
    parser.add_argument('--rl_algorithm', type=str, default='ppo',
                        choices=['ppo', 'a2c'],
                        help='RL algorithm to use')
    parser.add_argument('--rl_preset', type=str, default='default',
                        choices=['default', 'aggressive', 'conservative', 'risk_adjusted',
                                 'a2c', 'high_capacity', 'fast'],
                        help='RL configuration preset')
    
    # Actor-Critic parameters
    parser.add_argument('--actor_hidden_dims', type=int, nargs='+', default=[64, 32],
                        help='Actor hidden dimensions')
    parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[64, 32],
                        help='Critic hidden dimensions')
    parser.add_argument('--actor_lr', type=float, default=3e-4, help='Actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Critic learning rate')
    
    # PPO parameters
    parser.add_argument('--ppo_clip_epsilon', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='PPO update epochs')
    parser.add_argument('--target_kl', type=float, default=0.01,
                        help='Target KL divergence')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter')
    
    # Reward design
    parser.add_argument('--reward_type', type=str, default='return',
                        choices=['return', 'sharpe', 'sortino', 'calmar'],
                        help='Reward calculation type')
    parser.add_argument('--reward_scale', type=float, default=100.0,
                        help='Reward scaling factor')
    parser.add_argument('--transaction_penalty', type=float, default=0.001,
                        help='Transaction cost penalty')
    parser.add_argument('--holding_bonus', type=float, default=0.0001,
                        help='Holding position bonus')
    parser.add_argument('--drawdown_penalty', type=float, default=0.5,
                        help='Drawdown penalty')
    
    # Exploration parameters
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    
    # Training parameters
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--rollout_length', type=int, default=2048,
                        help='Rollout length for PPO')
    parser.add_argument('--transaction_cost', type=float, default=0.001,
                        help='Transaction cost for evaluation')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--direct_train', action='store_true',
                        help='Direct training flag')
    parser.add_argument('--scale', type=int, default=1, help='Dollar index scaling')
    
    return parser.parse_args()


def load_config(args) -> RLRegimeMambaConfig:
    """Load RL configuration from preset, YAML file and command-line arguments (CLI wins)."""
    preset_map = {
        'default': RLConfigPresets.default,
        'aggressive': RLConfigPresets.aggressive_trading,
        'conservative': RLConfigPresets.conservative_trading,
        'risk_adjusted': RLConfigPresets.risk_adjusted,
        'a2c': RLConfigPresets.a2c_config,
        'high_capacity': RLConfigPresets.high_capacity,
        'fast': RLConfigPresets.fast_training
    }

    config = preset_map[args.rl_preset]()
    apply_overrides(config, load_yaml_config(args.config), skip_none=False)
    apply_overrides(config, vars(args))

    if args.gpu_id >= 0 and torch.cuda.is_available():
        config.device = torch.device(f'cuda:{args.gpu_id}')
    else:
        config.device = torch.device('cpu')

    return config


def save_config(config, output_dir: str):
    """Save configuration to file."""
    save_config_files(
        config, output_dir,
        title="RL Regime Mamba",
        sections=[
            ("RL Algorithm", ['rl_algorithm']),
            ("Reward Design", RL_REWARD_PARAMS),
        ],
        sort_keys=True
    )


def predict_rl_regimes(
    model: RLRegimeMamba,
    dataloader: DataLoader,
    config
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Predict regimes using RL model."""
    from collections import deque
    
    device = config.device
    model.eval()
    
    all_predictions = []
    all_returns = []
    all_dates = []
    
    prev_action = None
    return_history = deque(maxlen=getattr(config, 'return_history_len', 5))
    
    with torch.no_grad():
        for x, y, dates, returns in dataloader:
            x = x.to(device)
            batch_size = x.size(0)
            
            if prev_action is None:
                current_position = torch.zeros(batch_size, device=device)
            else:
                current_position = torch.full((batch_size,), prev_action, device=device, dtype=torch.float)
            
            if len(return_history) > 0:
                return_hist_tensor = torch.tensor(
                    list(return_history), device=device
                ).unsqueeze(0).repeat(batch_size, 1)
                if return_hist_tensor.size(1) < getattr(config, 'return_history_len', 5):
                    padding = torch.zeros(
                        batch_size,
                        getattr(config, 'return_history_len', 5) - return_hist_tensor.size(1),
                        device=device
                    )
                    return_hist_tensor = torch.cat([padding, return_hist_tensor], dim=1)
            else:
                return_hist_tensor = torch.zeros(
                    batch_size, getattr(config, 'return_history_len', 5), device=device
                )
            
            predictions = model.predict_regimes(
                x, current_position, return_hist_tensor, deterministic=True
            )
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_returns.extend(returns.numpy().flatten())
            all_dates.extend(dates)
            
            for r in returns.numpy():
                return_history.append(r)
            
            prev_action = int(np.bincount(predictions.cpu().numpy().flatten()).argmax())
    
    return np.array(all_predictions), np.array(all_returns), all_dates


def train_rl_model_for_window(
    config,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    data: pd.DataFrame,
    window_number: int = 1
) -> Optional[RLRegimeMamba]:
    """Train RL model for a specific window."""
    if train_rl_regime_mamba is None:
        raise NotImplementedError(
            "regime_mamba.train.rl_train.train_rl_regime_mamba is not implemented yet"
        )
    print(f"\n[RL] Training period: {train_start} ~ {train_end}")
    print(f"[RL] Validation period: {valid_start} ~ {valid_end}")
    print(f"[RL] Algorithm: {config.rl_algorithm}, Reward: {config.reward_type}")
    
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
    
    if len(train_dataset) < 100 or len(valid_dataset) < 50:
        print(f"[RL] Warning: Insufficient data. Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")
        return None
    
    print(f"[RL] Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")
    
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
    
    window_save_dir = os.path.join(config.results_dir, f'window_{window_number}', 'model')
    os.makedirs(window_save_dir, exist_ok=True)
    
    model = create_rl_model_from_config(config)
    
    model, history = train_rl_regime_mamba(
        model,
        train_loader,
        valid_loader,
        config,
        save_dir=window_save_dir
    )
    
    best_valid_reward = max(history['valid_rewards']) if history['valid_rewards'] else float('-inf')
    print(f"[RL] Training complete. Best validation reward: {best_valid_reward:.4f}")
    
    return model


def run_rl_rolling_window_backtest(
    config,
    data: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run rolling window backtest with RL model."""
    logger.info(f"[RL] Algorithm: {config.rl_algorithm}, Reward: {config.reward_type}")

    def process_window(window_info, window_dir):
        logger.info("[RL] Training model...")
        model = train_rl_model_for_window(
            config,
            window_info['train_period']['start'],
            window_info['train_period']['end'],
            window_info['valid_period']['start'],
            window_info['valid_period']['end'],
            data,
            window_number=window_info['window_number']
        )
        if model is None:
            logger.warning("[RL] Model training failed, skipping window")
            return None

        logger.info("[RL] Evaluating smoothing methods...")
        return evaluate_smoothing_methods(
            partial(predict_rl_regimes, model, config=config),
            data,
            config,
            window_info['forward_period'],
            window_dir,
            log_prefix="[RL] ",
            title_prefix="[RL] "
        )

    return run_windowed_backtest(
        config, logger, process_window,
        use_clustering=False,
        log_prefix="[RL] ",
        title_prefix="[RL] "
    )


def main():
    """Main execution function."""
    try:
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'serif'
        
        args = parse_args()
        if train_rl_regime_mamba is None:
            raise NotImplementedError(
                "RL training module is missing: implement train_rl_regime_mamba in "
                "regime_mamba/train/rl_train.py before running this script."
            )
        
        print(f"\n{'='*60}")
        print(f"Mode: RL Regime Mamba ({args.rl_algorithm.upper()})")
        print(f"Preset: {args.rl_preset}")
        print(f"Reward Type: {args.reward_type}")
        print(f"{'='*60}\n")
        
        # Prepare output directory
        result_dir, log_file = prepare_output_directory(args.results_dir, prefix="rl_backtest")
        
        # Set up logging
        logger = setup_logging(log_file=log_file)
        logger.info(f"[RL] Starting rolling window backtest")
        
        # Load configuration
        try:
            config = load_config(args)
            config.results_dir = result_dir
            logger.info("[RL] Configuration loaded successfully")
            logger.info(f"[RL] Algorithm: {config.rl_algorithm}")
            logger.info(f"[RL] Reward: {config.reward_type}, Scale: {config.reward_scale}")
            logger.info(f"[RL] Transaction Penalty: {config.transaction_penalty}")
        except Exception as e:
            logger.error(f"[RL] Error loading configuration: {str(e)}")
            sys.exit(1)
        
        # Save configuration
        save_config(config, result_dir)
        logger.info(f"[RL] Configuration saved to {result_dir}")
        
        # Set random seed
        set_seed(config.seed)
        logger.info(f"[RL] Random seed set to {config.seed}")
        
        # Load data
        try:
            logger.info(f"[RL] Loading data from {config.data_path}")
            data = pd.read_csv(config.data_path)
            logger.info(f"[RL] Loaded data with {len(data)} rows")
        except Exception as e:
            logger.error(f"[RL] Error loading data: {str(e)}")
            sys.exit(1)
        
        # Run backtest
        results = run_rl_rolling_window_backtest(config, data, logger)
        
        logger.info(f"[RL] Backtest complete! Results saved to {result_dir}")
        
    except Exception as e:
        logging.error(f"[RL] Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
