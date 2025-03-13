"""
Training functions for RL-based investment strategies.
"""

import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

from ..models.rl_model import ActorCritic
from ..utils.rl_environments import FinancialTradingEnv
from ..utils.rl_agents import PPOAgent

def train_rl_agent_for_window(config, train_start, train_end, valid_start, valid_end, data):
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
    
    # Ensure we have enough data
    if len(train_data) < config.seq_len * 10:
        print(f"Warning: Not enough data for training. Got {len(train_data)} samples.")
        return None, None, None
    
    # Prepare features and returns
    features = train_data[['returns', 'dd_10', 'sortino_20', 'sortino_60']].values
    returns = train_data['returns'].values / 100  # Convert to decimal
    
    # Create environment
    env = FinancialTradingEnv(
        returns=returns,
        features=features,
        seq_len=config.seq_len,
        transaction_cost=config.transaction_cost,
        reward_type=config.reward_type,
        position_penalty=config.position_penalty,
        window_size=config.window_size
    )
    
    # Create model
    model = ActorCritic(
        input_dim=features.shape[1] + 3,  # +3 for one-hot position
        d_model=config.d_model,
        d_state=config.d_state,
        n_layers=config.n_layers,
        dropout=config.dropout
    ).to(config.device)
    
    # Create agent
    agent = PPOAgent(
        env=env,
        model=model,
        device=config.device,
        lr=config.learning_rate,
        gamma=config.gamma
    )
    
    # Train agent
    history = agent.train(
        n_episodes=config.n_episodes,
        n_steps_per_episode=config.steps_per_episode,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size
    )
    
    # Validate on validation data
    valid_data = data[(data['Date'] >= valid_start) & (data['Date'] <= valid_end)].copy()
    
    if len(valid_data) < config.seq_len:
        print(f"Warning: Not enough data for validation. Got {len(valid_data)} samples.")
    else:
        valid_features = valid_data[['returns', 'dd_10', 'sortino_20', 'sortino_60']].values
        valid_returns = valid_data['returns'].values / 100
        
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
