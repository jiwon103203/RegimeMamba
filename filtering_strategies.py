import argparse
import os
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
from tqdm import tqdm

from regime_mamba.config.config import RegimeMambaConfig
from regime_mamba.models.mamba_model import create_model_from_config
from regime_mamba.utils.utils import set_seed
from regime_mamba.evaluate.smoothing import find_optimal_filtering


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
    parser = argparse.ArgumentParser(description='Regime Filtering Strategy Optimization')
    
    # Main configuration arguments
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--data_path', type=str, help='Data file path')
    parser.add_argument('--model_path', type=str, help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./filtering_results', help='Result directory')
    parser.add_argument('--transaction_cost', type=float, help='Transaction cost (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=128, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_clusters', type=int, default=2, help='Number of clusters')
    parser.add_argument('--cluster_method', type=str, default='cosine_kmeans',help='Clustering method')
    
    # Target parameters
    parser.add_argument('--target_type', type=str, help='Target type')
    parser.add_argument('--target_horizon', type=int, help='Target horizon')
    
    return parser.parse_args()


def load_config(args) -> RegimeMambaConfig:
    """
    Load configuration from file or command-line arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        RegimeMambaConfig: Configuration object
    """
    config = RegimeMambaConfig()
    
    # If config file is provided, load it
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        # Update config from YAML
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Override with command-line arguments if provided
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    
    return config


def prepare_output_directory(base_dir: str) -> Tuple[str, str, str]:
    """
    Create output directory structure
    
    Args:
        base_dir: Base output directory
        
    Returns:
        tuple: (output_dir, param_file_path, chart_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"filtering_test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    param_file_path = os.path.join(output_dir, 'parameters.txt')
    chart_path = os.path.join(output_dir, 'filtering_strategies_comparison.png')
    
    return output_dir, param_file_path, chart_path


def save_parameters(config: RegimeMambaConfig, file_path: str, data_path: str):
    """
    Save configuration parameters to file
    
    Args:
        config: Configuration object
        file_path: File path for saving parameters
        data_path: Path to the data file
    """
    with open(file_path, 'w') as f:
        f.write(f"데이터 경로: {data_path}\n")
        f.write(f"모델 경로: {config.model_path}\n")
        f.write(f"거래 비용: {config.transaction_cost}\n")
        f.write(f"d_model: {config.d_model}\n")
        f.write(f"d_state: {config.d_state}\n")
        f.write(f"n_layers: {config.n_layers}\n")
        f.write(f"dropout: {config.dropout}\n")
        f.write(f"seq_len: {config.seq_len}\n")
        f.write(f"batch_size: {config.batch_size}\n")
        f.write(f"n_clusters: {config.n_clusters}\n")
        f.write(f"target_type: {config.target_type}\n")
        f.write(f"target_horizon: {config.target_horizon}\n")


def save_optimal_parameters(params: Dict[str, Any], output_dir: str):
    """
    Save optimal parameters to file
    
    Args:
        params: Dictionary of optimal parameters
        output_dir: Output directory
    """
    with open(os.path.join(output_dir, 'optimal_parameters.txt'), 'w') as f:
        f.write("최적 필터링 파라미터:\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")


def main():
    """Main function"""
    logger = setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Prepare output directory
    output_dir, param_file_path, chart_path = prepare_output_directory(args.output_dir)
    
    # Load configuration
    config = load_config(args)
    
    # Save parameters
    save_parameters(config, param_file_path, args.data_path)
    
    # Run optimization
    try:
        logger.info("다양한 레짐 필터링 전략 비교 분석 시작...")
        with tqdm(total=100, desc="Finding optimal filtering") as pbar:
            def update_progress(progress):
                pbar.update(int(progress * 100) - pbar.n)
            
            optimal_params, results = find_optimal_filtering(
                config, 
                args.data_path,
                save_path=chart_path,
                progress_callback=update_progress
            )
        
        # Save optimal parameters
        save_optimal_parameters(optimal_params, output_dir)
        
        logger.info(f"분석 완료! 결과가 {output_dir}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise


if __name__ == "__main__":
    main()
