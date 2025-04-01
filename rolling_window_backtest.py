#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rolling Window Backtest
This script runs a rolling window backtest using a pre-trained model.
"""

import os
import sys
import argparse
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from regime_mamba.config.config import RollingWindowConfig
from regime_mamba.evaluate.rolling_window import run_rolling_window_backtest
from regime_mamba.utils.utils import set_seed


def setup_logging(log_level=logging.INFO, log_file=None):
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
    parser = argparse.ArgumentParser(description='Run Rolling Window Backtest')
    
    # Configuration sources
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    
    # Required parameters
    parser.add_argument('--data_path', type=str, help='Data file path')
    parser.add_argument('--model_path', type=str, help='Pre-trained model path')
    
    # Optional parameters with defaults
    parser.add_argument('--results_dir', type=str, help='Results directory')
    parser.add_argument('--start_date', type=str, help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--lookback_years', type=int, help='Lookback period for clustering (years)')
    parser.add_argument('--forward_months', type=int, help='Forward application period (months)')
    parser.add_argument('--transaction_cost', type=float, help='Transaction cost (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--preprocessed', action='store_true', help='Whether data is preprocessed')
    
    # Model parameters
    parser.add_argument('--input_dim', type=int, help='Input dimension')
    parser.add_argument('--d_model', type=int, help='Model dimension')
    parser.add_argument('--d_state', type=int, help='State dimension')
    parser.add_argument('--n_layers', type=int, help='Number of layers')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--cluster_method', type=str, default='cosine_kmeans', help='Clustering method')
    parser.add_argument('--direct_train', action='store_true', help='Train model directly for clasification')
    parser.add_argument('--vae', action='store_true', help='Train model with VAE')
    
    return parser.parse_args()


def load_config(args) -> RollingWindowConfig:
    """Load configuration from file and command-line arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        RollingWindowConfig: Configuration object
    """
    config = RollingWindowConfig()
    
    # Set default values
    defaults = {
        'results_dir': './rolling_window_results',
        'start_date': '2010-01-01',
        'end_date': '2023-12-31',
        'lookback_years': 10,
        'forward_months': 12,
        'transaction_cost': 0.001,
        'seed': 42,
        'preprocessed': True,
        'd_model': 128,
        'd_state': 128,
        'n_layers': 4,
        'dropout': 0.1
    }
    
    # Load from YAML file if provided
    yaml_config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
    
    # Update with values from YAML
    for key, value in yaml_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Update with command-line arguments if provided (overrides YAML)
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    # Check for required parameters
    required_params = ['data_path', 'model_path']
    missing_params = [param for param in required_params if getattr(config, param, None) is None]
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
    
    # Fill in defaults for missing parameters
    for key, value in defaults.items():
        if getattr(config, key, None) is None:
            setattr(config, key, value)
    
    return config


def prepare_output_directory(output_dir: str) -> tuple:
    """Create output directory structure with timestamped subdirectory
    
    Args:
        output_dir: Base output directory
        
    Returns:
        tuple: (result_dir, log_file)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"backtest_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    
    log_file = os.path.join(result_dir, "backtest.log")
    
    return result_dir, log_file


def save_config(config: RollingWindowConfig, output_dir: str):
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
        f.write("=== Rolling Window Backtest Configuration ===\n\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")


def check_data_file(data_path: str, preprocessed: bool) -> pd.DataFrame:
    """Check if data file exists and is valid
    
    Args:
        data_path: Path to data file
        preprocessed: Whether data is preprocessed
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
        
        # Basic validation
        required_columns = ['date']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Data file missing required columns: {', '.join(missing_columns)}")
        
        # Check date format
        try:
            pd.to_datetime(data['date'])
        except:
            raise ValueError("Invalid date format in data file")
        
        # If not preprocessed, do basic preprocessing
        if not preprocessed:
            data = preprocess_data(data)
        
        return data
    
    except Exception as e:
        raise ValueError(f"Error loading data file: {str(e)}")


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Perform basic preprocessing on data
    
    Args:
        data: Raw data DataFrame
        
    Returns:
        pd.DataFrame: Preprocessed data
    """
    # Convert date to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Forward fill missing values
    data = data.fillna(method='ffill')
    
    # Backward fill any remaining missing values
    data = data.fillna(method='bfill')
    
    # Scale returns to percentages if needed
    if 'returns' in data.columns and data['returns'].max() < 1.0:
        data['returns'] = data['returns'] * 100
    
    return data


def run_backtest(config: RollingWindowConfig, logger: logging.Logger) -> Dict[str, Any]:
    """Run rolling window backtest with progress tracking
    
    Args:
        config: Configuration object
        logger: Logger instance
        
    Returns:
        Dict[str, Any]: Results dictionary
    """
    # Check and load data
    logger.info(f"Loading data from {config.data_path}")
    data = check_data_file(config.data_path, config.preprocessed)
    logger.info(f"Loaded data with {len(data)} rows")
    
    # Create a progress callback for run_rolling_window_backtest
    progress_bar = tqdm(total=100, desc="Running backtest")
    
    def update_progress(progress: float):
        progress_bar.update(int(progress * 100) - progress_bar.n)
    
    # Run backtest
    logger.info("Starting rolling window backtest")
    try:
        results = run_rolling_window_backtest(config, data, progress_callback=update_progress)
        progress_bar.close()
        logger.info("Backtest completed successfully")
        return results
    except Exception as e:
        progress_bar.close()
        logger.error(f"Error running backtest: {str(e)}")
        raise


def main():
    """Main execution function"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Prepare output directory
        result_dir, log_file = prepare_output_directory(args.results_dir or './rolling_window_results')
        
        # Set up logging
        logger = setup_logging(log_file=log_file)
        logger.info("Starting rolling window backtest")
        
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
        
        # Run backtest
        results = run_backtest(config, logger)
        
        # Log completion
        logger.info(f"Backtest complete! Results saved to {result_dir}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
