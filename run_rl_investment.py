import os
import sys
import argparse
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from regime_mamba.utils.utils import set_seed
from regime_mamba.config.rl_config import RLInvestmentConfig
from regime_mamba.utils.rl_investment import run_rl_investment

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL-based Investment Backtesting')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Data file path')
    parser.add_argument('--results_dir', type=str, default='./rl_investment_results', help='Results directory')
    parser.add_argument('--start_date', type=str, default='1990-01-01', help='Start date')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date')
    
    # Period settings
    parser.add_argument('--total_window_years', type=int, default=40, help='Total data period to use (years)')
    parser.add_argument('--train_years', type=int, default=20, help='Training period (years)')
    parser.add_argument('--valid_years', type=int, default=10, help='Validation period (years)')
    parser.add_argument('--forward_months', type=int, default=60, help='Forward testing period (months)')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--d_state', type=int, default=128, help='State dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    # RL parameters
    parser.add_argument('--reward_type', type=str, default='sharpe', choices=['sharpe', 'diff_sharpe'], 
                       help='Reward type (sharpe or diff_sharpe)')
    parser.add_argument('--position_penalty', type=float, default=0.01, help='Penalty for position changes')
    parser.add_argument('--window_size', type=int, default=252, help='Window size for Sharpe calculation')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--steps_per_episode', type=int, default=2048, help='Steps per episode')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of update epochs')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    # Other settings
    parser.add_argument('--transaction_cost', type=float, default=0.001, help='Transaction cost (0.001 = 0.1%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, f"rl_investment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create configuration from arguments
    config = RLInvestmentConfig.from_args(args)
    config.results_dir = output_dir
    
    # Save configuration
    config.save_config(output_dir)
    
    # Run RL investment analysis
    print("=== Starting RL Investment Analysis ===")
    combined_results, all_performances, training_histories = run_rl_investment(config)
    
    if combined_results is not None:
        print(f"RL investment analysis complete! Results saved to {output_dir}")
    else:
        print("RL investment analysis failed!")

if __name__ == "__main__":
    main()
