import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from torch.utils.data import Dataset, DataLoader
import torch

from ..utils.utils import set_seed
from ..models.mamba_model import create_model_from_config
from .clustering import identify_bull_bear_regimes, predict_regimes, extract_hidden_states
from .strategy import evaluate_regime_strategy, visualize_all_periods_performance
from ..data.dataset import RegimeMambaDataset, create_dataloaders, create_date_range_dataloader

def load_pretrained_model(config):
    """
    Load a pretrained RegimeMamba model

    Args:
        config: Configuration object

    Returns:
        Loaded model
    """
    from ..models.mamba_model import TimeSeriesMamba
    
    # Initialize model
    model = TimeSeriesMamba(
        input_dim=config.input_dim,
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        n_layers=config.n_layers,
        dropout=config.dropout
    )

    # Load checkpoint
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()
    model.to(config.device)

    print(f"Model loaded: {config.model_path}")
    return model

def identify_regimes_for_period(model, data, config, period_start, period_end):
    """
    Identify regimes for a specific period

    Args:
        model: Pretrained model
        data: Full dataframe
        config: Configuration object
        period_start: Period start date (string)
        period_end: Period end date (string)

    Returns:
        kmeans: Trained KMeans model
        bull_regime: Bull regime cluster ID
    """
    print(f"Identifying regimes for period {period_start} ~ {period_end}...")

    # Create dataset and dataloader
    dataset = create_date_range_dataloader(
        data=data,
        seq_len=config.seq_len,
        start_date=period_start,
        end_date=period_end,
        target_type = config.target_type,
        target_horizon = config.target_horizon
    )

    # Check if enough data
    if len(dataset) < 100:
        print(f"Warning: Insufficient data for period {period_start} ~ {period_end} ({len(dataset)} samples).")
        return None, None

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Extract hidden states
    hidden_states, returns, dates = extract_hidden_states(model, dataloader, config)

    kmeans, bull_regime = identify_bull_bear_regimes(hidden_states, returns, config)

    return kmeans, bull_regime

def apply_regimes_to_future_period(model, data, config, kmeans, bull_regime, future_start, future_end):
    """
    Apply identified regimes to future period

    Args:
        model: Pretrained model
        data: Full dataframe
        config: Configuration object
        kmeans: Trained KMeans model
        bull_regime: Bull regime cluster ID
        future_start: Future period start date (string)
        future_end: Future period end date (string)

    Returns:
        results_df: Results dataframe
        performance: Performance metrics dictionary
    """
    print(f"Applying regimes to period {future_start} ~ {future_end}...")

    # Create dataset and dataloader
    dataset = create_date_range_dataloader(
        data=data,
        seq_len=config.seq_len,
        start_date=future_start,
        end_date=future_end,
        target_type=config.target_type,
        target_horizon=config.target_horizon
    )

    # Check if enough data
    if len(dataset) < 10:
        print(f"Warning: Insufficient data for period {future_start} ~ {future_end} ({len(dataset)} samples).")
        return None, None

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Predict regimes
    predictions, true_returns, dates = predict_regimes(model, dataloader, kmeans, bull_regime, config)

    # Evaluate strategy with transaction costs
    results_df, performance = evaluate_regime_strategy(
        predictions,
        true_returns,
        dates,
        transaction_cost=config.transaction_cost
    )

    return results_df, performance

def visualize_period_performance(results_df, title, save_path):
    """
    Visualize performance for a single period

    Args:
        results_df: Results dataframe
        title: Chart title
        save_path: Save path
    """
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(results_df['Cum_Market'] * 100, label='Market', color='gray')
    plt.plot(results_df['Cum_Strategy'] * 100, label='Regime Strategy', color='blue')
    plt.title(f'{title} - Cumulative Returns')
    plt.legend()
    plt.ylabel('Return (%)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(results_df['Regime'], label='Regime (1=Bull, 0=Bear)', color='red')
    plt.title('Regime Signal')
    plt.ylabel('Regime')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_rolling_window_backtest(config, data_path):
    """
    Run rolling window backtest

    Args:
        config: Configuration object
        data_path: Data file path
        
    Returns:
        combined_results: Combined results dataframe
        period_performances: List of period performance metrics
    """
    # Load data
    data = pd.read_csv(data_path)

    # Load pretrained model
    model = load_pretrained_model(config)

    # Storage for backtest results
    all_results = []
    period_performances = []
    all_predictions = []

    # Parse start and end dates
    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')

    period_counter = 1

    # Main loop for rolling window backtesting
    while current_date < end_date:
        print(f"\n=== Processing Period {period_counter} ===")

        # Calculate lookback period (from current date to lookback_years years ago)
        lookback_start = (current_date - relativedelta(years=config.lookback_years)).strftime('%Y-%m-%d')
        lookback_end = current_date.strftime('%Y-%m-%d')

        # Calculate future period (from current date to forward_months months later)
        future_start = current_date.strftime('%Y-%m-%d')
        future_end = (current_date + relativedelta(months=config.forward_months)).strftime('%Y-%m-%d')

        print(f"Lookback period: {lookback_start} ~ {lookback_end}")
        print(f"Future period: {future_start} ~ {future_end}")

        # Identify regimes (based on past data)
        kmeans, bull_regime = identify_regimes_for_period(model, data, config, lookback_start, lookback_end)

        if kmeans is not None and bull_regime is not None:
            # Apply identified regimes to future period
            results_df, performance = apply_regimes_to_future_period(
                model, data, config, kmeans, bull_regime, future_start, future_end
            )

            if results_df is not None:
                # Add period information
                results_df['lookback_start'] = lookback_start
                results_df['lookback_end'] = lookback_end
                results_df['future_start'] = future_start
                results_df['future_end'] = future_end
                results_df['period'] = period_counter

                # Save results
                all_results.append(results_df)

                # Add period information to performance metrics
                performance['period'] = period_counter
                performance['lookback_start'] = lookback_start
                performance['lookback_end'] = lookback_end
                performance['future_start'] = future_start
                performance['future_end'] = future_end
                period_performances.append(performance)

                # Save results file
                results_df.to_csv(
                    f"{config.results_dir}/period_{period_counter}_results.csv",
                    index=False
                )

                # Visualize performance
                visualize_period_performance(
                    results_df,
                    f"Period {period_counter}: {future_start} ~ {future_end}",
                    f"{config.results_dir}/period_{period_counter}_performance.png"
                )

        # Move to next period
        current_date += relativedelta(months=config.forward_months)
        period_counter += 1

    # Merge and save results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{config.results_dir}/all_periods_results.csv", index=False)

        # Save overall performance
        with open(f"{config.results_dir}/all_periods_performance.json", 'w') as f:
            json.dump(period_performances, f, indent=4)

        # Visualize overall performance
        visualize_all_periods_performance(period_performances, config.results_dir)

        print(f"\nBacktest completed! Processed {period_counter-1} periods.")
        return combined_results, period_performances
    else:
        print("Backtest failed: No valid results.")
        return None, None