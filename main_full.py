#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RegimeMamba: Market Regime Identification System based on Mamba Architecture

This is the main entry point script for training, evaluating, and running
market regime identification models.
"""

import os
import sys
import time
import json
import yaml
import logging
import argparse
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import traceback

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regime_mamba.config.config import RegimeMambaConfig
from regime_mamba.data.dataset_full import create_dataloaders
from regime_mamba.models.mamba_model import create_model_from_config
from regime_mamba.train.train import train_regime_mamba
from regime_mamba.train.optimize_full import optimize_regime_mamba_bayesian
from regime_mamba.evaluate.clustering import extract_hidden_states, identify_bull_bear_regimes, predict_regimes
from regime_mamba.evaluate.strategy import evaluate_regime_strategy, analyze_transaction_cost_impact
from regime_mamba.utils.utils import set_seed


def setup_logging(log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_dir: Directory to save log file (optional)
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger('regime_mamba')
    logger.setLevel(level)
    logger.addHandler(console_handler)
    
    # Create file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
    
    return logger


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='RegimeMamba: Market Regime Identification System based on Mamba Architecture',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration sources
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Data and model paths
    parser.add_argument('--data_path', type=str, help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory path')
    parser.add_argument('--load_model', type=str, help='Path to load model checkpoint')
    
    # Runtime behavior
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--skip_training', action='store_true', help='Skip training step')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--direct_train', action='store_true', help='Train model directly for clasification')

    # Model parameters
    parser.add_argument('--input_dim', type=int, default=5, help='Input dimension')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=64, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost (0.001 = 0.1%)')
    parser.add_argument('--target_type', type=str, default="next_day", help='Target type')
    parser.add_argument('--target_horizon', type=int, default=1, help='Target horizon')
    parser.add_argument('--preprocessed', type=bool, default=False, help='Data is preprocessed')
    parser.add_argument('--cluster_method', type=str, default='cosine_kmeans', help='Clustering method')
    parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters')
    
    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--opt_iterations', type=int, default=30, help='Number of optimization iterations')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Performance parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> RegimeMambaConfig:
    """
    Load configuration from file and command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        RegimeMambaConfig: Configuration object
    """
    # Create default config
    config = RegimeMambaConfig()
    
    # Command line arguments
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    # Load from YAML if provided
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update config from YAML
            for key, value in yaml_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logging.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logging.warning(f"Error loading config from {args.config}: {str(e)}")
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        config.device = f'cuda:{args.gpu}'
    else:
        config.device = 'cpu'

    return config


def setup_output_directory(output_dir: str) -> Tuple[str, Dict[str, str]]:
    """
    Set up output directory structure
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Tuple[str, Dict[str, str]]: Run directory and paths dictionary
    """
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    dirs = {
        'model': os.path.join(run_dir, 'model'),
        'results': os.path.join(run_dir, 'results'),
        'plots': os.path.join(run_dir, 'plots'),
        'logs': os.path.join(run_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Create paths dictionary
    paths = {
        'run_dir': run_dir,
        'model_checkpoint': os.path.join(dirs['model'], 'best_regime_mamba.pth'),
        'config_save': os.path.join(run_dir, 'config.yaml'),
        'optimization_results': os.path.join(dirs['results'], 'bayesian_optimization_results.json'),
        'strategy_performance_plot': os.path.join(dirs['plots'], 'regime_strategy_performance.png'),
        'cost_analysis_plot': os.path.join(dirs['plots'], 'transaction_cost_analysis.png'),
        'strategy_results': os.path.join(dirs['results'], 'regime_strategy_detailed_results.csv'),
        'strategy_performance': os.path.join(dirs['results'], 'regime_strategy_results_with_costs.json'),
        'cost_analysis': os.path.join(dirs['results'], 'transaction_cost_analysis.csv'),
        'logs':dirs['logs']
    }

    return run_dir, paths


def save_config(config: RegimeMambaConfig, save_path: str):
    """
    Save configuration to file
    
    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    # Convert config to dictionary
    config_dict = {key: getattr(config, key) for key in dir(config) 
                  if not key.startswith('__') and not callable(getattr(config, key))}
    
    # Save as YAML
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def optimize_hyperparameters(
    config: RegimeMambaConfig, 
    paths: Dict[str, str], 
    iterations: int = 30, 
    logger: logging.Logger = None
) -> RegimeMambaConfig:
    """
    Run Bayesian optimization to find optimal hyperparameters
    
    Args:
        config: Base configuration
        paths: Paths dictionary
        iterations: Number of optimization iterations
        logger: Logger instance
        
    Returns:
        RegimeMambaConfig: Optimized configuration
    """
    logger = logger or logging.getLogger('regime_mamba')
    logger.info("Starting Bayesian hyperparameter optimization")
    
    try:
        # Run optimization
        start_time = time.time()
        optimized_config = optimize_regime_mamba_bayesian(
            config.data_path, config, n_iterations=iterations, 
            save_path=paths['optimization_results']
        )
        
        # Save optimized config
        save_config(optimized_config, paths['config_save'])
        
        elapsed_time = time.time() - start_time
        logger.info(f"Optimization completed in {elapsed_time/60:.1f} minutes")
        
        return optimized_config
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        logger.error("Falling back to default configuration")
        return config


def train_model(
    config: RegimeMambaConfig, 
    train_loader: torch.utils.data.DataLoader, 
    valid_loader: torch.utils.data.DataLoader, 
    save_path: str, 
    logger: logging.Logger = None
) -> torch.nn.Module:
    """
    Train the model
    
    Args:
        config: Configuration object
        train_loader: Training data loader
        valid_loader: Validation data loader
        save_path: Path to save model checkpoint
        logger: Logger instance
        
    Returns:
        torch.nn.Module: Trained model
    """
    logger = logger or logging.getLogger('regime_mamba')
    logger.info("Creating model")
    model = create_model_from_config(config)
    
    logger.info(f"Training model on {config.device}")
    start_time = time.time()
    
    try:
        model = train_regime_mamba(model, train_loader, valid_loader, config, save_path=save_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time/60:.1f} minutes")
        
        return model
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def load_model(
    config: RegimeMambaConfig, 
    model_path: str, 
    logger: logging.Logger = None
) -> torch.nn.Module:
    """
    Load model from checkpoint
    
    Args:
        config: Configuration object
        model_path: Path to model checkpoint
        logger: Logger instance
        
    Returns:
        torch.nn.Module: Loaded model
    """
    logger = logger or logging.getLogger('regime_mamba')
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = create_model_from_config(config)
        checkpoint = torch.load(model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)
        model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def evaluate_strategy(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    config: RegimeMambaConfig,
    paths: Dict[str, str],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate regime-based trading strategy
    
    Args:
        model: Trained model
        test_loader: Test data loader
        kmeans: K-means clustering model
        bull_regime: Bull regime cluster ID
        config: Configuration object
        paths: Paths dictionary
        logger: Logger instance
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Results dataframe and performance metrics
    """
    logger = logger or logging.getLogger('regime_mamba')
    logger.info("Predicting regimes on test data")
    
    try:
        test_predictions = [0] # 첫 포지션은 0으로 시작
        test_returns = []
        test_dates = []
        model.to(config.device)
        model.eval()
        for batch in test_loader:
            features, _, dates, returns = batch
            features, targets = features.to(config.device), targets.to(config.device)
            with torch.no_grad():
                predictions = model(features)
            
            if config.direct_train:
                predictions = torch.argmax(predictions, dim=1) # (batch_size ,3) -> (batch_size, 1)
                test_predictions+=predictions.cpu().numpy().flatten()
                test_returns+=returns.cpu().numpy().flatten()
                test_dates+=dates.flatten()
            else:
                binary = features[:,1] > predictions # (batch_size, 5) -> (batch_size, 1)
                test_predictions+binary.astype(int).cpu().numpy().flatten()
                test_returns+returns.cpu().numpy().flatten()
                test_dates+dates.flatten()
        # # Predict regimes
        # test_predictions, test_returns, test_dates = predict_regimes(
        #     model, test_loader, config
        # )
        
        # Evaluate strategy
        logger.info(f"Evaluating trading strategy with transaction cost {config.transaction_cost*100:.2f}%")
        results_df, performance = evaluate_regime_strategy(
            test_predictions[:-1], test_returns, test_dates,
            transaction_cost=config.transaction_cost,
            save_path=paths['strategy_performance_plot']
        )
        
        # Save results
        results_df.to_csv(paths['strategy_results'], index=False)
        with open(paths['strategy_performance'], 'w') as f:
            json.dump(performance, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else x)
        
        # Log key performance metrics
        logger.info(f"Strategy Performance:")
        logger.info(f"Market Return: {performance['cumulative_returns']['market']:.2f}%")
        logger.info(f"Strategy Return: {performance['cumulative_returns']['strategy']:.2f}%")
        logger.info(f"Strategy Sharpe Ratio: {performance['sharpe_ratio']['strategy']:.2f}")
        logger.info(f"Number of Trades: {performance['trading_metrics']['number_of_trades']}")
        
        return results_df, performance
    except Exception as e:
        logger.error(f"Error evaluating strategy: {str(e)}")
        raise


def main():
    """
    Main execution function
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set up output directory
    run_dir, paths = setup_output_directory(args.output_dir)
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(paths['logs'], log_level)
    
    # Log system information
    logger.info(f"RegimeMamba: Market Regime Identification System")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    try:
        # Load configuration
        config = load_config(args)
        # Save initial configuration
        save_config(config, paths['config_save'])
        logger.info(f"Configuration saved to {paths['config_save']}")
        
        # Record start time
        start_time = time.time()
        
        # 1. Hyperparameter optimization (optional)
        if args.optimize:
            logger.info("Step 1: Hyperparameter Optimization")
            config = optimize_hyperparameters(config, paths, args.opt_iterations, logger)
            config.input_dim = args.input_dim
            config.target_type = args.target_type
            config.target_horizon = args.target_horizon
            config.preprocessed = args.preprocessed

        else:
            logger.info("Step 1: Skipping Hyperparameter Optimization")
        
        # 2. Create dataloaders
        logger.info("Step 2: Creating Data Loaders")
        train_loader, valid_loader, test_loader = create_dataloaders(config)
        logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, " 
                   f"Validation: {len(valid_loader)} batches, Test: {len(test_loader)} batches")
        
        # 3. Load or train model
        logger.info("Step 3: Model Preparation")
        if args.load_model:
            model = load_model(config, args.load_model, logger)
        elif not args.skip_training:
            model = train_model(config, train_loader, valid_loader, paths['model_checkpoint'], logger)
        else:
            logger.error("Either --load_model or training is required. Set --load_model or remove --skip_training")
            sys.exit(1)
        
        
        # 5. Evaluate strategy
        logger.info("Step 4: Strategy Evaluation")
        results_df, performance = evaluate_strategy(
            model, test_loader, config, paths, logger
        )
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Complete! Total execution time: {total_time/60:.1f} minutes")
        
        # Return results
        return {
            'model': model,
            'config': config,
            'performance': performance,
            'paths': paths
        }
    
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
