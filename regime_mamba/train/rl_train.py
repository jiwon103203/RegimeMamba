"""
Training functions for RL-based investment strategies.
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

#from ..models.rl_model import ActorCritic
from ..utils.rl_environments import FinancialTradingEnv, optimize_action_thresholds
from ..utils.rl_agents import PPOAgent

def train_rl_agent_for_window(config, model, train_start, train_end, valid_start, valid_end, data):
    """
    Train RL agent for a specific window.
    
    Args:
        config: Configuration object
        train_start: Training start date
        train_end: Training end date
        valid_start: Validation start date
        valid_end: Validation end date
        data: DataFrame with price and feature data
        
    Returns:
        tuple: (trained_agent, trained_model, training_history)
    """
    print(f"\nTraining period: {train_start} ~ {train_end}")
    print(f"Validation period: {valid_start} ~ {valid_end}")
    
    # Filter data for training period
    train_data = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)].copy()
    train_data['Open'] = train_data['Open'] / 100
    train_data['Close'] = train_data['Close'] / 100
    train_data['High'] = train_data['High'] / 100
    train_data['Low'] = train_data['Low'] / 100
    
    # Ensure we have enough data
    if len(train_data) < config.seq_len * 5:
        print(f"Warning: Not enough data for training. Got {len(train_data)} samples.")
        return None, None, None

    # Determine feature columns
    if config.input_dim == 5:
        feature_cols = ["Open", "Close", "High", "Low", "treasury_rate"]
    elif config.input_dim == 7:
        feature_cols = ["Open", "Close", "High", "Low", "treasury_rate", "treasury_rate_5y", "dollar_index"]

    # Check if all columns exist in the dataset
    for col in feature_cols:
        if col not in train_data.columns:
            print(f"Warning: Column '{col}' not found in dataset. Available columns: {train_data.columns.tolist()}")
            return None, None, None
    
    # Prepare features and returns
    features = train_data[feature_cols].values
    if config.input_dim == 4:
        returns = train_data['returns'].values / 100  # Convert to decimal
    else:
        returns = train_data['returns'].values

    # Print feature information for debugging
    print(f"Features shape: {features.shape}")
    print(f"Feature stats - Mean: {np.mean(features, axis=0)}, Std: {np.std(features, axis=0)}")
    print(f"Returns shape: {returns.shape}")
    print(f"Returns stats - Mean: {np.mean(returns)}, Std: {np.std(returns)}")
    
    # Check for NaN or infinity values
    if np.isnan(features).any() or np.isinf(features).any():
        print("Warning: Features contain NaN or infinity values")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.isnan(returns).any() or np.isinf(returns).any():
        print("Warning: Returns contain NaN or infinity values")
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    if config.n_positions == 3:
        thresholds_to_try = [
            (0.2, 0.2), (0.33, 0.33), (0.5, 0.5),
            (0.2, 0.4), (0.4, 0.2), (0.25, 0.25)
        ]
    else:
        thresholds_to_try = [
            (0.3, 0), (0.4, 0), (0.5, 0), (0.6, 0), (0.7, 0)
        ]

    if config.optimize_thresholds:
        print("Optimizing action thresholds...")

        def create_agent(env):
            new_agent = PPOAgent(
                env=env,
                model=model,
                device=config.device,
                lr=config.rl_learning_rate,
                gamma=config.rl_gamma,
                freeze_feature_extractor=config.freeze_feature_extractor,
            )
            return new_agent
        
        validation_period = {
            'start': valid_start,
            'end': valid_end
        }

        best_thresholds, all_results = optimize_action_thresholds(
            config=config,
            agent_factory=create_agent,
            data=data,
            eval_period=validation_period,
            thresholds_to_try=thresholds_to_try
        )

        # save all_results
        all_results_path = os.path.join(config.output_dir, f"window_{config.window_number}_all_results.npy")
        np.save(all_results_path, all_results)
        print(f"All results saved to {all_results_path}")
        print(f"Best thresholds: Long - {best_thresholds[0]}, Short - {best_thresholds[1]}")

        if config.n_positions == 3:
            long_thresh, short_thresh = best_thresholds['long_threshold'], best_thresholds['short_threshold']
        else:
            long_thresh = best_thresholds['long_threshold']
            short_thresh = None
    else:
        long_thresh, short_thresh = 0.33, 0.33
            

    # Create environment
    env = FinancialTradingEnv(
        returns=returns,
        features=features,
        seq_len=config.seq_len,
        transaction_cost=config.transaction_cost,
        reward_type=config.reward_type,
        position_penalty=config.position_penalty,
        window_size=config.window_size,
        n_positions=config.n_positions,
        long_threshold=long_thresh,
        short_threshold=short_thresh
    )
    
    print(f"Creating model with input_dim={config.input_dim}, d_model={config.d_model}")
    
    # Create model
    # model = ActorCritic(
    #     input_dim=input_dim,
    #     d_model=config.d_model,
    #     d_state=config.d_state,
    #     n_layers=config.n_layers,
    #     dropout=config.dropout
    # ).to(config.device)
    
    # Verify model architecture
#    print(f"Model structure: {model}")
    
    # Create agent
    agent = PPOAgent(
        env=env,
        model=model,
        device=config.device,
        lr=config.rl_learning_rate,
        gamma=config.rl_gamma,
        freeze_feature_extractor=config.freeze_feature_extractor,
    )
    
    # Train agent
    try:
        history = agent.train(
            n_episodes=config.n_episodes,
            n_steps_per_episode=config.steps_per_episode,
            n_epochs=config.n_epochs,
            batch_size=config.rl_batch_size
        )
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
    # Validate on validation data
    valid_data = data[(data['Date'] >= valid_start) & (data['Date'] <= valid_end)].copy()
    valid_data['Open'] = valid_data['Open'] / 100
    valid_data['Close'] = valid_data['Close'] / 100
    valid_data['High'] = valid_data['High'] / 100
    valid_data['Low'] = valid_data['Low'] / 100
    
    if len(valid_data) < config.seq_len:
        print(f"Warning: Not enough data for validation. Got {len(valid_data)} samples.")
    else:
        valid_features = valid_data[feature_cols].values
        valid_returns = valid_data['returns'].values / 100
        
        # Check for NaN or infinity values
        if np.isnan(valid_features).any() or np.isinf(valid_features).any():
            print("Warning: Validation features contain NaN or infinity values")
            valid_features = np.nan_to_num(valid_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(valid_returns).any() or np.isinf(valid_returns).any():
            print("Warning: Validation returns contain NaN or infinity values")
            valid_returns = np.nan_to_num(valid_returns, nan=0.0, posinf=0.0, neginf=0.0)
        valid_env = FinancialTradingEnv(
            returns=valid_returns,
            features=valid_features,
            seq_len=config.seq_len,
            transaction_cost=config.transaction_cost,
            reward_type=config.reward_type,
            position_penalty=config.position_penalty,
            window_size=config.window_size
        )
        
        valid_results = agent.evaluate(valid_env)
        print("\nValidation Results:")
        print(f"  Total Return: {valid_results['total_return']:.4f}")
        print(f"  Sharpe Ratio: {valid_results['sharpe_ratio']:.4f}")
        print(f"  Market Sharpe: {valid_results['market_sharpe']:.4f}")
        print(f"  Max Drawdown: {valid_results['max_drawdown']:.4f}")
        print(f"  Position Changes: {valid_results['position_changes']}")
    
    return agent, model, history

def save_training_results(agent, model, history, output_dir, window_number):
    """
    Save training results to disk.
    
    Args:
        agent: Trained PPO agent
        model: Trained model
        history: Training history
        output_dir: Output directory
        window_number: Window number
    
    Returns:
        str: Path to saved model
    """
    # Create directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"window_{window_number}_model.pth")
    torch.save({
        'window': window_number,
        'model_state_dict': model.state_dict(),
    }, model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f"window_{window_number}_training_history.npy")
    np.save(history_path, history)
    
    return model_path
