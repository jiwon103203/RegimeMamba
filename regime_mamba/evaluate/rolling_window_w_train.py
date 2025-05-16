import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import copy

from ..utils.utils import set_seed
from ..data.dataset import RegimeMambaDataset, create_dataloaders, DateRangeRegimeMambaDataset
from ..models.mamba_model import TimeSeriesMamba, create_model_from_config
from ..models.lstm import StackedLSTM
from ..models.jump_model import ModifiedJumpModel
from ..train.train import train_with_early_stopping
from .clustering import identify_bull_bear_regimes, predict_regimes, extract_hidden_states
from .strategy import evaluate_regime_strategy, visualize_all_periods_performance
from .smoothing import apply_regime_smoothing, apply_minimum_holding_period


def train_model_for_window(config, train_start, train_end, valid_start, valid_end, data, window_number=1):
    """
    Train model for a specific time window
    
    Args:
        config: Configuration object
        train_start: Training start date
        train_end: Training end date
        valid_start: Validation start date
        valid_end: Validation end date
        data: Full dataframe
        
    Returns:
        trained_model: Trained model
        best_val_loss: Best validation loss
    """
    print(f"\nTraining period: {train_start} ~ {train_end}")
    print(f"Validation period: {valid_start} ~ {valid_end}")
    
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
    
    # Check if there's enough data
    if len(train_dataset) < 100 or len(valid_dataset) < 50:
        print(f"Warning: Insufficient data. Training: {len(train_dataset)}, Validation: {len(valid_dataset)}")
        return None, float('inf')
    
    # Create model
    if config.lstm:
        model = StackedLSTM(
            input_dim = config.input_dim
        )
        model.train_for_window(train_start, train_end, data, valid_window=config.valid_years, outputdir=config.results_dir)
        # Load saved model
        model.load_state_dict(torch.load(f"./{config.results_dir}/best_model.pth"))
    else:
        model = TimeSeriesMamba(
                input_dim=config.input_dim,
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                n_layers=config.n_layers,
                dropout=config.dropout,
                config=config
        )
    
        # Train model with early stopping
        best_val_loss, best_epoch, model = train_with_early_stopping(
                model, 
                train_loader, 
                valid_loader, 
                config, 
                use_onecycle=config.use_onecycle
            )
    
        print(f"Training complete. Best validation loss: {best_val_loss:.6f} (Epoch {best_epoch+1})")

    if config.jump_model:
        jump_model = ModifiedJumpModel(config=config, jump_penalty=config.jump_penalty)
        jump_model.feature_extractor = model
        jump_model.train_for_window(train_start, train_end, data, config.valid_years, sort='cumret', window=window_number, dynamic=False)

        return jump_model
    
    return model, best_val_loss

def identify_regimes_for_window(config, model, data, clustering_start, clustering_end):
    """
    Identify regimes for a specific window
    
    Args:
        config: Configuration object
        model: Trained model
        data: Full dataframe
        clustering_start: Clustering start date
        clustering_end: Clustering end date
        
    Returns:
        kmeans: K-Means model
        bull_regime: Bull regime cluster ID
    """
    print(f"\nRegime identification period: {clustering_start} ~ {clustering_end}")
    
    # Create clustering dataset
    clustering_dataset = DateRangeRegimeMambaDataset(
        data=data, 
        seq_len=config.seq_len,
        start_date=clustering_start,
        end_date=clustering_end,
        config=config
    )
    
    # Create data loader
    clustering_loader = DataLoader(
        clustering_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Check if there's enough data
    if len(clustering_dataset) < 100:
        print(f"Warning: Insufficient data for clustering ({len(clustering_dataset)} samples)")
        return None, None
    
    # Extract hidden states
    hidden_states, returns, _ = extract_hidden_states(model, clustering_loader, config)
    
    # Clustering
    kmeans, bull_regime = identify_bull_bear_regimes(hidden_states, returns, config)
    
    return kmeans, bull_regime

def apply_and_evaluate_regimes(config, model, data, kmeans, bull_regime, forward_start, forward_end, window_number):
    """
    Apply and evaluate regimes in future period
    
    Args:
        config: Configuration object
        model: Trained model
        data: Full dataframe
        kmeans: K-Means model
        bull_regime: Bull regime cluster ID
        forward_start: Future period start date
        forward_end: Future period end date
        window_number: Window number
        
    Returns:
        results_df: Results dataframe
        performance: Performance metrics dictionary
    """
    print(f"\nApplication period: {forward_start} ~ {forward_end}")
    
    # Create future dataset
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
        num_workers=4
    )
    
    # Check if there's enough data
    if len(forward_dataset) < 10:
        print(f"Warning: Insufficient data for evaluation ({len(forward_dataset)} samples)")
        return None, None
    
    # Predict regimes
    predictions, true_returns, dates = predict_regimes(model, forward_loader, kmeans, bull_regime, config)
    
    # Save original predictions
    raw_predictions = copy.deepcopy(predictions)
    
    # Apply filtering (optional)
    if config.apply_filtering:
        if config.filter_method == 'minimum_holding':
            predictions = apply_minimum_holding_period(predictions, min_holding_days=config.min_holding_days).reshape(-1, 1)
        elif config.filter_method == 'smoothing':
            predictions = apply_regime_smoothing(predictions, method='ma', window=10).reshape(-1, 1)
    
    # Evaluate strategy considering transaction costs
    results_df, performance = evaluate_regime_strategy(
        predictions,
        true_returns,
        dates,
        transaction_cost=config.transaction_cost,
        config=config
    )
    
    # Add period information to results
    if results_df is not None:
        results_df['window'] = window_number
        forward_start = datetime.strptime(forward_start, "%Y-%m-%d")
        results_df['train_valid_period'] = f"{forward_start - relativedelta(years=config.total_window_years)} ~ {forward_start}"
        results_df['forward_start'] = forward_start
        results_df['forward_end'] = forward_end
        
        # Add original regime information
        results_df['raw_regime'] = raw_predictions.flatten()
        
        # Add filtering information
        if config.apply_filtering:
            results_df['filter_method'] = config.filter_method
            
            # Calculate original and filtered trade counts
            raw_trades = (np.diff(raw_predictions.flatten()) != 0).sum() + (raw_predictions[0] == 1)
            filtered_trades = (np.diff(predictions.flatten()) != 0).sum() + (predictions[0][0] == 1)
            
            # Add trade reduction information to performance
            performance['raw_trades'] = int(raw_trades)
            performance['filtered_trades'] = int(filtered_trades)
            performance['trade_reduction'] = int(raw_trades - filtered_trades)
            if raw_trades > 0:
                performance['trade_reduction_pct'] = ((raw_trades - filtered_trades) / raw_trades) * 100
            else:
                performance['trade_reduction_pct'] = 0
            
        # Add window information to performance
        performance['window'] = window_number
        performance['forward_start'] = forward_start
        performance['forward_end'] = forward_end
        
    return results_df, performance

def visualize_window_performance(results_df, model_loss, window_number, title, save_path):
    """
    Visualize window performance
    
    Args:
        results_df: Results dataframe
        model_loss: Model validation loss
        window_number: Window number
        title: Chart title
        save_path: Save path
    """
    plt.figure(figsize=(15, 10))
    
    # Plot cumulative returns and regime
    plt.subplot(2, 1, 1)
    plt.plot(results_df['Cum_Market'] * 100, label='Market', color='gray')
    plt.plot(results_df['Cum_Strategy'] * 100, label='Regime Strategy', color='blue')
    plt.title(f'{title} (Validation Loss: {model_loss:.6f})')
    plt.legend()
    plt.ylabel('Return (%)')
    plt.grid(True)
    
    # Plot regime signal
    plt.subplot(2, 1, 2)
    plt.plot(results_df['Regime'], label='Regime (1=Bull, 0=Bear)', color='red')
    plt.title('Regime Signal')
    plt.ylabel('Regime')
    plt.grid(True)
    
    # Show original regime if filtering is used
    if 'raw_regime' in results_df.columns:
        plt.plot(results_df['raw_regime'], label='Original Regime', color='green', alpha=0.5, linestyle='--')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_rolling_window_train(config):
    """
    Execute rolling window retraining
    
    Args:
        config: Configuration object
        
    Returns:
        combined_results: Combined results dataframe
        all_performances: List of all performance metrics
        model_histories: List of model training histories
    """
    # Load data
    print("Loading data...")
    data = pd.read_csv(config.data_path)
    
    
    # Storage for results
    all_results = []
    all_performances = []
    model_histories = []
    
    # Parse start and end dates
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    
    window_number = 1
    
    # Rolling window main loop
    while current_date <= end_date:
        print(f"\n=== Processing Window {window_number} ===")
        
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
        
        print(f"Training period: {train_start} ~ {train_end} ({config.train_years} years)")
        print(f"Validation period: {train_end} ~ {valid_end} ({config.valid_years} years)")
        print(f"Clustering period: {clustering_start} ~ {clustering_end} ({config.clustering_years} years)")
        print(f"Future application period: {forward_start} ~ {forward_end} ({config.forward_months/12:.1f} years)")
        
        # 1. Train model
        model, val_loss = train_model_for_window(
            config, train_start, train_end, valid_start, valid_end, data
        )
        
        # Skip to next window if training failed
        if model is None:
            print("Model training failed, moving to next window")
            current_date += relativedelta(months=config.forward_months)
            window_number += 1
            continue
        
        # 2. Identify regimes
        kmeans, bull_regime = identify_regimes_for_window(
            config, model, data, clustering_start, clustering_end
        )
        
        # Skip to next window if regime identification failed
        if kmeans is None or bull_regime is None:
            print("Regime identification failed, moving to next window")
            current_date += relativedelta(months=config.forward_months)
            window_number += 1
            continue
        
        # 3. Apply regimes to future period and evaluate
        results_df, performance = apply_and_evaluate_regimes(
            config, model, data, kmeans, bull_regime, forward_start, forward_end, window_number
        )
        
        # Save results
        if results_df is not None and performance is not None:
            all_results.append(results_df)
            all_performances.append(performance)
            
            # Save model information
            model_history = {
                'window': window_number,
                'train_start': train_start,
                'train_end': train_end,
                'valid_start': valid_start,
                'valid_end': valid_end,
                'val_loss': val_loss,
                'bull_regime': bull_regime
            }
            model_histories.append(model_history)
            
            # Save results file
            results_df.to_csv(
                f"{config.results_dir}/window_{window_number}_results.csv",
                index=False
            )
            
            # Visualize results
            visualize_window_performance(
                results_df,
                val_loss,
                window_number,
                f"Window {window_number}: {forward_start} ~ {forward_end}",
                f"{config.results_dir}/window_{window_number}_performance.png"
            )
            
            # Save model (optional)
            model_save_path = f"{config.results_dir}/window_{window_number}_model.pth"
            torch.save({
                'window': window_number,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'bull_regime': bull_regime
            }, model_save_path)
            
        # Move to next window
        current_date += relativedelta(months=config.forward_months)
        window_number += 1
    
    # Merge and save all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{config.results_dir}/all_windows_results.csv", index=False)
        
        # Save model training history
        with open(f"{config.results_dir}/model_histories.json", 'w') as f:
            json.dump(model_histories, f, indent=4)
        
        # Save overall performance
        with open(f"{config.results_dir}/all_performances.json", 'w') as f:
            json.dump(all_performances, f, default=default_converter, indent=4)
        
        # Visualize overall performance
        visualize_all_windows_performance(all_performances, config.results_dir)
        
        print(f"\nRolling window retraining complete! Processed {len(all_performances)} windows.")
        return combined_results, all_performances, model_histories
    else:
        print("Rolling window retraining failed: No valid results.")
        return None, None, None

def default_converter(o):
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Type {type(o).__name__} is not serializable")

def visualize_all_windows_performance(all_performances, save_dir):
    """
    Visualize performance across all windows
    
    Args:
        all_performances: List of performance metrics for all windows
        save_dir: Directory to save visualizations
    """
    # Extract performance data
    windows = [p['window'] for p in all_performances]
    market_returns = [p['cumulative_returns']['market'] for p in all_performances]
    strategy_returns = [p['cumulative_returns']['strategy'] for p in all_performances]
    market_sharpes = [p['sharpe_ratio']['market'] for p in all_performances]
    strategy_sharpes = [p['sharpe_ratio']['strategy'] for p in all_performances]
    
    # For filtering statistics if available
    if 'raw_trades' in all_performances[0]:
        raw_trades = [p['raw_trades'] for p in all_performances]
        filtered_trades = [p['filtered_trades'] for p in all_performances]
        trade_reductions_pct = [p['trade_reduction_pct'] for p in all_performances]
        
    # Visualization (multiple subplots)
    plt.figure(figsize=(15, 12))
    
    # 1. Compare returns by window
    plt.subplot(2, 2, 1)
    width = 0.35
    x = np.arange(len(windows))
    plt.bar(x - width/2, market_returns, width, label='Market', color='gray')
    plt.bar(x + width/2, strategy_returns, width, label='Regime Strategy', color='blue')
    plt.xlabel('Window')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Comparison of Cumulative Returns by Window')
    plt.xticks(x, windows)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Compare Sharpe ratios by window
    plt.subplot(2, 2, 2)
    plt.bar(x - width/2, market_sharpes, width, label='Market', color='gray')
    plt.bar(x + width/2, strategy_sharpes, width, label='Regime Strategy', color='blue')
    plt.xlabel('Window')
    plt.ylabel('Sharpe Ratio')
    plt.title('Comparison of Sharpe Ratios by Window')
    plt.xticks(x, windows)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Compare trade counts if filtering statistics are available
    if 'raw_trades' in all_performances[0]:
        plt.subplot(2, 2, 3)
        plt.bar(x - width/2, raw_trades, width, label='Original Trades', color='red', alpha=0.7)
        plt.bar(x + width/2, filtered_trades, width, label='Filtered Trades', color='blue')
        plt.xlabel('Window')
        plt.ylabel('Number of Trades')
        plt.title('Comparison of Trade Counts by Window')
        plt.xticks(x, windows)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Trade reduction percentage
        plt.subplot(2, 2, 4)
        plt.bar(x, trade_reductions_pct, color='green')
        plt.axhline(y=np.mean(trade_reductions_pct), color='black', linestyle='--',
                   label=f'Average: {np.mean(trade_reductions_pct):.2f}%')
        plt.xlabel('Window')
        plt.ylabel('Trade Reduction (%)')
        plt.title('Trade Reduction Percentage by Window')
        plt.xticks(x, windows)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Alternative graph (cumulative performance)
        plt.subplot(2, 1, 2)
        plt.plot(windows, np.cumsum(market_returns), 'o-', label='Market Cumulative', color='gray')
        plt.plot(windows, np.cumsum(strategy_returns), 'o-', label='Strategy Cumulative', color='blue')
        plt.xlabel('Window')
        plt.ylabel('Cumulative Return (%)')
        plt.title('Cumulative Window Performance')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_windows_comparison.png")
    plt.close()
    
    # Calculate overall performance summary
    total_market_return = sum(market_returns)
    total_strategy_return = sum(strategy_returns)
    avg_market_sharpe = np.mean(market_sharpes)
    avg_strategy_sharpe = np.mean(strategy_sharpes)
    win_rate = sum(np.array(strategy_returns) > np.array(market_returns)) / len(windows) * 100
    
    # Calculate average compound returns
    annualized_market_return = ((1 + total_market_return/100) ** (1/len(windows)) - 1) * 100
    annualized_strategy_return = ((1 + total_strategy_return/100) ** (1/len(windows)) - 1) * 100
    
    summary = {
        'total_windows': len(windows),
        'total_returns': {
            'market': total_market_return,
            'strategy': total_strategy_return,
            'difference': total_strategy_return - total_market_return
        },
        'annualized_returns': {
            'market': annualized_market_return,
            'strategy': annualized_strategy_return,
            'difference': annualized_strategy_return - annualized_market_return
        },
        'sharpe_ratios': {
            'market_avg': avg_market_sharpe,
            'strategy_avg': avg_strategy_sharpe,
            'difference': avg_strategy_sharpe - avg_market_sharpe
        },
        'win_rate': win_rate
    }
    
    # Add filtering statistics if available
    if 'raw_trades' in all_performances[0]:
        total_raw_trades = sum(raw_trades)
        total_filtered_trades = sum(filtered_trades)
        total_reduction = total_raw_trades - total_filtered_trades
        total_reduction_pct = (total_reduction / total_raw_trades) * 100 if total_raw_trades > 0 else 0
        
        summary['trading_statistics'] = {
            'total_raw_trades': total_raw_trades,
            'total_filtered_trades': total_filtered_trades,
            'total_reduction': total_reduction,
            'total_reduction_pct': total_reduction_pct,
            'avg_reduction_pct': np.mean(trade_reductions_pct)
        }
    
    # Save summary
    with open(f"{save_dir}/performance_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    print("\n===== Overall Performance Summary =====")
    print(f"Total number of windows: {summary['total_windows']}")
    print(f"Total market return: {summary['total_returns']['market']:.2f}%")
    print(f"Total strategy return: {summary['total_returns']['strategy']:.2f}%")
    print(f"Return difference: {summary['total_returns']['difference']:.2f}%")
    print(f"Annualized market return: {summary['annualized_returns']['market']:.2f}%")
    print(f"Annualized strategy return: {summary['annualized_returns']['strategy']:.2f}%")
    print(f"Average market Sharpe ratio: {summary['sharpe_ratios']['market_avg']:.2f}")
    print(f"Average strategy Sharpe ratio: {summary['sharpe_ratios']['strategy_avg']:.2f}")
    print(f"Win rate vs market: {summary['win_rate']:.2f}%")
    
    if 'trading_statistics' in summary:
        print("\nTrading statistics:")
        print(f"  Total original trade count: {summary['trading_statistics']['total_raw_trades']}")
        print(f"  Total filtered trade count: {summary['trading_statistics']['total_filtered_trades']}")
        print(f"  Total trade reduction: {summary['trading_statistics']['total_reduction']} ({summary['trading_statistics']['total_reduction_pct']:.2f}%)")
        print(f"  Average trade reduction percentage: {summary['trading_statistics']['avg_reduction_pct']:.2f}%")