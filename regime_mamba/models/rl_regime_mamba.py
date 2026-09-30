"""
Reinforcement Learning Regime Mamba Model Implementation

Uses Actor-Critic architecture for end-to-end regime detection.
Replaces Jump Model's Dynamic Programming with deep RL.

Architecture:
    Mamba Backbone → Hidden States → Actor (Policy) → Action (Regime)
                                  → Critic (Value) → Expected Return
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, NamedTuple
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP
from collections import deque
import random


class RolloutBuffer(NamedTuple):
    """Storage for rollout data."""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'elu': nn.ELU(),
        'silu': nn.SiLU()
    }
    return activations.get(name.lower(), nn.ReLU())


class ActorNetwork(nn.Module):
    """
    Policy network that outputs action probabilities.
    Maps hidden states to regime selection probabilities.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        n_actions: int = 2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.n_actions = n_actions
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                get_activation(activation),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.action_head = nn.Linear(prev_dim, n_actions)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Smaller initialization for output layer
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch, d_model) hidden states from Mamba
            
        Returns:
            action_probs: (batch, n_actions) action probabilities
            action_logits: (batch, n_actions) raw logits
        """
        features = self.mlp(hidden_states)
        action_logits = self.action_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, action_logits
    
    def get_action(
        self,
        hidden_states: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            hidden_states: (batch, d_model)
            deterministic: If True, take argmax action
            
        Returns:
            action: (batch,) sampled actions
            log_prob: (batch,) log probabilities
            entropy: (batch,) policy entropy
        """
        action_probs, _ = self.forward(hidden_states)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.
        
        Args:
            hidden_states: (batch, d_model)
            actions: (batch,) actions to evaluate
            
        Returns:
            log_prob: (batch,) log probabilities
            entropy: (batch,) policy entropy
        """
        action_probs, _ = self.forward(hidden_states)
        dist = torch.distributions.Categorical(action_probs)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """
    Value network that estimates state values.
    Predicts expected cumulative reward from a state.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                get_activation(activation),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        
        # Smaller initialization for output layer
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch, d_model)
            
        Returns:
            values: (batch, 1) state values
        """
        features = self.mlp(hidden_states)
        values = self.value_head(features)
        
        return values


class RewardCalculator:
    """
    Calculates rewards based on trading actions and market returns.
    Implements various reward shaping strategies.
    """
    
    def __init__(self, config):
        self.reward_type = getattr(config, 'reward_type', 'return')
        self.reward_scale = getattr(config, 'reward_scale', 100.0)
        self.transaction_penalty = getattr(config, 'transaction_penalty', 0.001)
        self.holding_bonus = getattr(config, 'holding_bonus', 0.0001)
        self.drawdown_penalty = getattr(config, 'drawdown_penalty', 0.5)
        self.risk_free_rate = getattr(config, 'risk_free_rate', 0.02 / 252)
        
        # Running statistics for normalization
        self.running_mean = 0.0
        self.running_std = 1.0
        self.count = 0
        
        # For Sharpe/Sortino calculation
        self.return_history = []
        self.window_size = 20
    
    def calculate_reward(
        self,
        action: int,
        prev_action: Optional[int],
        market_return: float,
        portfolio_value: float = 1.0,
        peak_value: float = 1.0
    ) -> float:
        """
        Calculate reward for a single step.
        
        Args:
            action: Current action (0=cash, 1=long)
            prev_action: Previous action
            market_return: Market return for this period
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value (for drawdown)
            
        Returns:
            Reward value
        """
        # Base return reward
        position_return = action * market_return
        
        # Transaction cost
        transaction_cost = 0.0
        if prev_action is not None and action != prev_action:
            transaction_cost = self.transaction_penalty
        
        # Holding bonus
        holding = self.holding_bonus if (prev_action is not None and action == prev_action) else 0.0
        
        # Calculate base reward
        reward = position_return - transaction_cost + holding
        
        # Apply reward type modifications
        if self.reward_type == 'sharpe':
            # Risk-adjusted reward
            self.return_history.append(position_return)
            if len(self.return_history) > self.window_size:
                self.return_history.pop(0)
            
            if len(self.return_history) >= 5:
                std = np.std(self.return_history) + 1e-8
                excess_return = position_return - self.risk_free_rate
                reward = excess_return / std
        
        elif self.reward_type == 'sortino':
            # Downside risk-adjusted reward
            self.return_history.append(position_return)
            if len(self.return_history) > self.window_size:
                self.return_history.pop(0)
            
            if len(self.return_history) >= 5:
                negative_returns = [r for r in self.return_history if r < 0]
                if negative_returns:
                    downside_std = np.std(negative_returns) + 1e-8
                else:
                    downside_std = 1e-8
                excess_return = position_return - self.risk_free_rate
                reward = excess_return / downside_std
        
        elif self.reward_type == 'calmar':
            # Drawdown-adjusted reward
            if portfolio_value < peak_value:
                drawdown = (peak_value - portfolio_value) / peak_value
                reward = reward - self.drawdown_penalty * drawdown
        
        # Scale reward
        reward = reward * self.reward_scale
        
        return reward
    
    def calculate_batch_rewards(
        self,
        actions: torch.Tensor,
        returns: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate rewards for a batch of actions.
        
        Args:
            actions: (batch,) actions
            returns: (batch,) market returns
            prev_actions: (batch,) previous actions
            
        Returns:
            rewards: (batch,) rewards
        """
        batch_size = actions.size(0)
        rewards = torch.zeros(batch_size, device=actions.device)
        
        for i in range(batch_size):
            action = actions[i].item()
            market_return = returns[i].item()
            prev_action = prev_actions[i].item() if prev_actions is not None else None
            
            rewards[i] = self.calculate_reward(action, prev_action, market_return)
        
        return rewards
    
    def reset(self):
        """Reset internal state."""
        self.return_history = []


class MambaFeatureExtractor(nn.Module):
    """
    Mamba-based feature extractor backbone.
    Shared between Actor and Critic networks.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config.input_dim
        self.d_model = config.d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(config.input_dim, config.d_model)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=config.d_model,
                mixer_cls=lambda dim: Mamba(
                    d_model=dim,
                    d_state=config.d_state,
                    d_conv=getattr(config, 'd_conv', 4),
                    expand=getattr(config, 'expand', 2)
                ),
                mlp_cls=lambda dim: GatedMLP(dim),
                fused_add_norm=True,
                residual_in_fp32=True
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input sequence.
        
        Args:
            x: (batch, seq_len, input_dim)
            
        Returns:
            hidden: (batch, d_model) hidden states
        """
        # Embedding
        x = self.input_embedding(x)
        x = self.dropout(x)
        
        # Mamba processing
        residual = None
        for i, block in enumerate(self.blocks):
            x, residual = block(x, residual)
            if i < len(self.blocks) - 1:
                x = self.dropout(x)
        
        # Extract last position hidden state
        hidden = x[:, -1, :]  # (batch, d_model)
        
        return hidden


class RLRegimeMamba(nn.Module):
    """
    Reinforcement Learning Regime Mamba Model.
    
    Uses Actor-Critic architecture for end-to-end regime detection.
    The Actor (policy) network outputs action probabilities,
    while the Critic (value) network estimates state values.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.n_actions = getattr(config, 'n_actions', 2)
        
        # Mamba feature extractor (shared backbone)
        self.feature_extractor = MambaFeatureExtractor(config)
        
        # State dimension (may include additional info)
        state_dim = config.d_model
        if getattr(config, 'use_portfolio_state', True):
            state_dim += 1  # Current position
        if getattr(config, 'use_return_history', True):
            state_dim += getattr(config, 'return_history_len', 5)
        
        self.state_dim = state_dim
        
        # Actor network (policy)
        self.actor = ActorNetwork(
            input_dim=state_dim,
            hidden_dims=getattr(config, 'actor_hidden_dims', [64, 32]),
            n_actions=self.n_actions,
            activation=getattr(config, 'actor_activation', 'relu')
        )
        
        # Critic network (value function)
        if getattr(config, 'use_separate_critic', True):
            self.critic = CriticNetwork(
                input_dim=state_dim,
                hidden_dims=getattr(config, 'critic_hidden_dims', [64, 32]),
                activation=getattr(config, 'critic_activation', 'relu')
            )
        else:
            # Shared critic head from actor features
            self.critic = nn.Linear(
                getattr(config, 'actor_hidden_dims', [64, 32])[-1], 
                1
            )
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(config)
        
        # Running statistics for state normalization
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.state_count = 0
    
    def get_state(
        self,
        x: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        return_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Construct full state representation.
        
        Args:
            x: (batch, seq_len, input_dim) input sequence
            current_position: (batch, 1) current position
            return_history: (batch, history_len) recent returns
            
        Returns:
            state: (batch, state_dim) full state representation
        """
        # Extract Mamba features
        hidden = self.feature_extractor(x)  # (batch, d_model)
        
        state_components = [hidden]
        
        # Add current position if enabled
        if getattr(self.config, 'use_portfolio_state', True) and current_position is not None:
            if current_position.dim() == 1:
                current_position = current_position.unsqueeze(-1)
            state_components.append(current_position.float())
        
        # Add return history if enabled
        if getattr(self.config, 'use_return_history', True) and return_history is not None:
            state_components.append(return_history.float())
        
        # Concatenate all components
        state = torch.cat(state_components, dim=-1)
        
        return state
    
    def forward(
        self,
        x: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        return_history: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for action selection.
        
        Args:
            x: (batch, seq_len, input_dim)
            current_position: (batch,) or (batch, 1)
            return_history: (batch, history_len)
            deterministic: If True, take greedy action
            
        Returns:
            Dictionary with action, log_prob, value, entropy
        """
        # Get state representation
        state = self.get_state(x, current_position, return_history)
        
        # Get action from actor
        action, log_prob, entropy = self.actor.get_action(state, deterministic)
        
        # Get value from critic
        if hasattr(self, 'critic') and isinstance(self.critic, CriticNetwork):
            value = self.critic(state)
        else:
            # Shared architecture (not recommended)
            actor_features = self.actor.mlp(state)
            value = self.critic(actor_features)
        
        return {
            'action': action,
            'log_prob': log_prob,
            'value': value.squeeze(-1),
            'entropy': entropy,
            'state': state
        }
    
    def get_value(
        self,
        x: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        return_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get value estimate for states.
        
        Args:
            x: (batch, seq_len, input_dim)
            current_position: (batch,) or (batch, 1)
            return_history: (batch, history_len)
            
        Returns:
            values: (batch,) state values
        """
        state = self.get_state(x, current_position, return_history)
        
        if isinstance(self.critic, CriticNetwork):
            value = self.critic(state)
        else:
            actor_features = self.actor.mlp(state)
            value = self.critic(actor_features)
        
        return value.squeeze(-1)
    
    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        return_history: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate log probabilities and values for given actions.
        
        Args:
            x: (batch, seq_len, input_dim)
            actions: (batch,) actions
            current_position: (batch,) or (batch, 1)
            return_history: (batch, history_len)
            
        Returns:
            Dictionary with log_prob, value, entropy
        """
        state = self.get_state(x, current_position, return_history)
        
        # Evaluate actions
        log_prob, entropy = self.actor.evaluate_actions(state, actions)
        
        # Get value
        if isinstance(self.critic, CriticNetwork):
            value = self.critic(state)
        else:
            actor_features = self.actor.mlp(state)
            value = self.critic(actor_features)
        
        return {
            'log_prob': log_prob,
            'value': value.squeeze(-1),
            'entropy': entropy
        }
    
    def predict_regimes(
        self,
        x: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        return_history: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Predict regime labels for inference.
        
        Args:
            x: (batch, seq_len, input_dim)
            current_position: (batch,)
            return_history: (batch, history_len)
            deterministic: If True, use greedy policy
            
        Returns:
            regimes: (batch,) regime labels (0=Bear, 1=Bull)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                x, current_position, return_history, 
                deterministic=deterministic
            )
        return outputs['action']
    
    def get_action_probs(
        self,
        x: torch.Tensor,
        current_position: Optional[torch.Tensor] = None,
        return_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get action probabilities for inference.
        
        Args:
            x: (batch, seq_len, input_dim)
            current_position: (batch,)
            return_history: (batch, history_len)
            
        Returns:
            probs: (batch, n_actions) action probabilities
        """
        self.eval()
        with torch.no_grad():
            state = self.get_state(x, current_position, return_history)
            action_probs, _ = self.actor(state)
        return action_probs


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: (T,) rewards
        values: (T,) value estimates
        dones: (T,) done flags
        next_value: scalar, value of next state
        gamma: discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: (T,) advantage estimates
        returns: (T,) discounted returns
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    returns = advantages + values
    
    return advantages, returns


def create_rl_model_from_config(config) -> RLRegimeMamba:
    """
    Factory function to create RLRegimeMamba from config.
    
    Args:
        config: Configuration object
        
    Returns:
        RLRegimeMamba model
    """
    # Set default values for config parameters
    default_params = {
        'n_actions': 2,
        'actor_hidden_dims': [64, 32],
        'critic_hidden_dims': [64, 32],
        'actor_activation': 'relu',
        'critic_activation': 'relu',
        'use_separate_critic': True,
        'use_portfolio_state': True,
        'use_return_history': True,
        'return_history_len': 5,
        'reward_type': 'return',
        'reward_scale': 100.0,
        'transaction_penalty': 0.001,
        'holding_bonus': 0.0001,
        'drawdown_penalty': 0.5,
        'risk_free_rate': 0.02 / 252
    }
    
    for key, value in default_params.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    return RLRegimeMamba(config)
