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
    Or it can take long(1) or neutral(0) positions on a financial asset.
    The agent receives a reward based on the Sharpe ratio of its strategy returns.
    """
    
    def __init__(self, returns, features, dates=None, seq_len=128, transaction_cost=0.001, reward_type='sharpe', 
                 position_penalty=0.01, window_size=252, n_positions=3, long_threshold=0.33, short_threshold=0.33):
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
        self.n_positions = n_positions
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        
        # State space: Sequence of feature vectors + current position
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(seq_len, features.shape[1])),
            'position': spaces.Discrete(n_positions)  # -1 (short), 0 (neutral), 1 (long) or 0 (neutral), 1 (long)
        })
        
        # Action space: Continuous value between -1 and 1
        # -1: fully short, 0: neutral, 1: fully long
        if n_positions == 3:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
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
            'position': self.current_position + 1 if self.n_positions == 3 else self.current_position # Map [-1, 0, 1] to [0, 1, 2] or [0, 1]
        }
    
    def _action_to_position(self, action):
        """Convert continuous action to discrete position."""
        action_value = action[0]
        
        if self.n_positions == 3:
            # Discretize the action space
            if action_value < -self.short_threshold:
                return -1  # Short
            elif action_value > self.long_threshold:
                return 1   # Long
            else:
                return 0   # Neutral
        else:
            # Map action to [0, 1] for long and neutral
            return int(action_value > self.long_threshold)
            
    def _calculate_sharpe_ratio(self, annualize=True, risk_free_rate=0.0):
        """Calculate Sharpe ratio based on recent returns."""
        recent_returns = self.strategy_returns[-self.window_size:]

        daily_rf_rate = 0.0
        if risk_free_rate > 0:
            daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1

        excess_returns = np.array(recent_returns) - daily_rf_rate
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return < 1e-8:
            return 100.0 if mean_return >=0 else -100.0

        sharpe = mean_return / std_return

        if annualize:
            sharpe = sharpe * np.sqrt(252)  # Assuming daily returns
        
        return sharpe
    
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
    
def optimize_action_thresholds(config, agent_factory, data, eval_period, thresholds_to_try):
    """
    """
    results = []

    for long_thresh, short_thresh in thresholds_to_try:
        print(f"Testing thresholds: Long - {long_thresh}, Short - {short_thresh}")

        env = FinancialTradingEnv(
            returns=data['returns'].values / 100 if config.input_dim == 4 else data['returns'].values,
            features=data[['Open', 'Close', 'High', 'Low', 'treasury_rate']].values,
            dates=data['Date'].values,
            seq_len=config.seq_len,
            transaction_cost=config.transaction_cost,
            reward_type=config.reward_type,
            position_penalty=config.position_penalty,
            window_size=config.window_size,
            n_positions=config.n_positions,  # Assuming 3 positions: long, neutral, short
            long_threshold=long_thresh,
            short_threshold=short_thresh
        )

        agent = agent_factory(env)
        result = agent.evaluate(env)

        sharpe = result['sharpe_ratio']
        returns = result['total_return']

        results.append({
            'long_threshold': long_thresh,
            'short_threshold': short_thresh,
            'sharpe_ratio': sharpe,
            'total_return': returns,
            'trades': result['position_changes']
        })

    best_result = max(results, key=lambda x: x['sharpe_ratio'])

    return best_result, results
