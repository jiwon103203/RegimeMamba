"""
Environments for RL-based investment strategies.
"""

import numpy as np
import gym
from gym import spaces

class FinancialTradingEnv(gym.Env):
    """
    A reinforcement learning environment for financial trading.
    
    This environment simulates a trading scenario where an agent can take long (1),
    neutral (0), or short (-1) positions on a financial asset.
    """
    
    def __init__(self, returns, features, dates=None, seq_len=128, transaction_cost=0.001, reward_type='sharpe', 
                 position_penalty=0.01, window_size=252):
        """
        Initialize the environment.
        
        Args:
            returns (numpy.array): Array of asset returns
            features (numpy.array): Array of features for prediction
            dates (numpy.array, optional): Array of dates corresponding to returns
            seq_len (int): Sequence length for observation
            transaction_cost (float): Cost of transactions as a fraction
            reward_type (str): Type of reward ('sharpe' or 'diff_sharpe')
            position_penalty (float): Penalty for changing position
            window_size (int): Window size for calculating Sharpe ratio
        """
        super(FinancialTradingEnv, self).__init__()
        
        self.returns = returns
        self.features = features
        self.dates = dates
        self.seq_len = seq_len
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type
        self.position_penalty = position_penalty
        self.window_size = window_size
        
        # State space: Sequence of feature vectors + current position
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(seq_len, features.shape[1])),
            'position': spaces.Discrete(3)  # -1 (short), 0 (neutral), 1 (long)
        })
        
        # Action space: Continuous value between -1 and 1
        # -1: fully short, 0: neutral, 1: fully long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Initialize state variables
        self.current_step = seq_len
        self.current_position = 0
        self.strategy_returns = []
        self.positions = []
        self.nav = 1.0  # Net Asset Value
        self.history = []
        
    def reset(self):
        """Reset the environment to the initial state."""
        self.current_step = self.seq_len
        self.current_position = 0
        self.strategy_returns = []
        self.positions = []
        self.nav = 1.0
        self.history = []
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (float): Action value between -1 and 1
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Convert continuous action to position (-1, 0, 1)
        new_position = self._action_to_position(action)
        
        # Calculate transaction costs
        cost = abs(new_position - self.current_position) * self.transaction_cost
        
        # Get current return and apply position
        current_return = self.returns[self.current_step]
        strategy_return = new_position * current_return - cost
        
        # Update NAV
        self.nav *= (1 + strategy_return)
        
        # Store history
        self.strategy_returns.append(strategy_return)
        self.positions.append(new_position)
        
        # Position change penalty
        position_change = abs(new_position - self.current_position)
        position_penalty = position_change * self.position_penalty
        
        # Calculate reward
        if len(self.strategy_returns) >= self.window_size:
            if self.reward_type == 'sharpe':
                reward = self._calculate_sharpe_ratio() - position_penalty
            elif self.reward_type == 'diff_sharpe':
                reward = self._calculate_differential_sharpe_ratio() - position_penalty
            else:
                reward = strategy_return - position_penalty
        else:
            reward = strategy_return - position_penalty
        
        # Update current position
        self.current_position = new_position
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'return': strategy_return,
            'nav': self.nav,
            'position': new_position,
            'position_change': position_change,
            'cost': cost,
            'date': self.dates[self.current_step - 1] if self.dates is not None else None
        }
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """Get the current observation."""
        # Get feature sequence
        start_idx = self.current_step - self.seq_len
        end_idx = self.current_step
        feature_seq = self.features[start_idx:end_idx]
        
        return {
            'features': feature_seq,
            'position': self.current_position + 1  # Map [-1, 0, 1] to [0, 1, 2]
        }
    
    def _action_to_position(self, action):
        """Convert continuous action to discrete position."""
        action_value = action[0]
        
        # Discretize the action space
        if action_value < -0.33:
            return -1  # Short
        elif action_value > 0.33:
            return 1   # Long
        else:
            return 0   # Neutral
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio based on recent returns."""
        recent_returns = self.strategy_returns[-self.window_size:]
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-6  # Add small constant to avoid division by zero
        
        return mean_return / std_return
    
    def _calculate_differential_sharpe_ratio(self):
        """Calculate Differential Sharpe Ratio (DSR) based on recent returns."""
        if len(self.strategy_returns) <= 1:
            return 0
            
        recent_returns = np.array(self.strategy_returns[-self.window_size:])
        
        # Previous Sharpe components
        if len(self.strategy_returns) > self.window_size:
            prev_returns = np.array(self.strategy_returns[-(self.window_size+1):-1])
            prev_mean = np.mean(prev_returns)
            prev_std = np.std(prev_returns) + 1e-6
            prev_sharpe = prev_mean / prev_std
        else:
            prev_sharpe = 0
            
        # Current Sharpe
        curr_mean = np.mean(recent_returns)
        curr_std = np.std(recent_returns) + 1e-6
        curr_sharpe = curr_mean / curr_std
        
        # Differential Sharpe
        return curr_sharpe - prev_sharpe
