"""
RL agents for investment strategies.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for RL training.
    
    Implements the PPO algorithm for training an actor-critic model
    in financial trading environments.
    """
    
    def __init__(self, env, model, device, lr=3e-4, gamma=0.99, epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01):
        """
        Initialize the PPO agent.
        
        Args:
            env: Training environment
            model: Actor-Critic model
            device: Device for training (cpu or cuda)
            lr (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): PPO clipping parameter
            value_coef (float): Value loss coefficient
            entropy_coef (float): Entropy coefficient
        """
        self.env = env
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def collect_trajectories(self, n_steps=2048):
        """
        Collect trajectories by running the environment.
        
        Args:
            n_steps (int): Maximum number of steps to collect
            
        Returns:
            dict: Collected trajectory data
        """
        # Storage
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        positions = []
        
        # Reset environment
        state = self.env.reset()
        done = False
        
        for _ in range(min(n_steps, len(self.env.returns) - self.env.current_step)):
            # Convert observation to tensor
            features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
            position = torch.LongTensor([state['position']]).to(self.device)
            
            # Get action and value
            with torch.no_grad():
                action_mean, value = self.model(features, position)
                action_dist = Normal(action_mean, 0.1)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action.cpu().numpy())
            
            # Store data
            states.append(state)
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            values.append(value.cpu().numpy())
            rewards.append(reward)
            dones.append(done)
            positions.append(info['position'])
            
            # Update state
            state = next_state
            
            if done:
                break
        
        # Calculate returns
        returns = self._compute_returns(rewards, dones)
        
        # Return trajectory data
        trajectory = {
            'states': states,
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'rewards': np.array(rewards),
            'returns': np.array(returns),
            'positions': np.array(positions)
        }
        
        return trajectory
    
    def _compute_returns(self, rewards, dones):
        """
        Compute discounted returns.
        
        Args:
            rewards (list): List of rewards
            dones (list): List of done flags
            
        Returns:
            list: Discounted returns
        """
        returns = []
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            
            running_return = rewards[t] + self.gamma * running_return
            returns.insert(0, running_return)
            
        return returns
    
    def update(self, trajectory, n_epochs=10, batch_size=64):
        """
        Update the model using PPO.
        
        Args:
            trajectory (dict): Collected trajectory data
            n_epochs (int): Number of update epochs
            batch_size (int): Batch size for updates
            
        Returns:
            dict: Training metrics
        """
        # Convert to tensors
        states = trajectory['states']
        actions = torch.FloatTensor(trajectory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(trajectory['log_probs']).to(self.device)
        returns = torch.FloatTensor(trajectory['returns']).to(self.device).unsqueeze(1)
        
        # Calculate advantage
        values = torch.FloatTensor(trajectory['values']).to(self.device)
        advantages = returns - values
        
        # PPO update
        metrics = {
            'value_loss': [],
            'policy_loss': [],
            'entropy': []
        }
        
        for _ in range(n_epochs):
            # Generate random indices
            indices = np.random.permutation(len(states))
            
            # Iterate over mini-batches
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get mini-batch data
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                features_batch = torch.FloatTensor(np.stack([s['features'] for s in batch_states])).to(self.device)
                position_batch = torch.LongTensor([s['position'] for s in batch_states]).to(self.device)
                
                action_mean, value = self.model(features_batch, position_batch)
                action_dist = Normal(action_mean, 0.1)
                log_prob = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                # Calculate losses
                ratio = torch.exp(log_prob - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(value, batch_returns)
                
                # Combined loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Store metrics
                metrics['value_loss'].append(value_loss.item())
                metrics['policy_loss'].append(policy_loss.item())
                metrics['entropy'].append(entropy.item())
        
        # Calculate average metrics
        for k, v in metrics.items():
            metrics[k] = np.mean(v)
            
        return metrics
    
    def train(self, n_episodes, n_steps_per_episode=2048, n_epochs=10, batch_size=64):
        """
        Train the agent.
        
        Args:
            n_episodes (int): Number of episodes
            n_steps_per_episode (int): Steps per episode
            n_epochs (int): Number of update epochs
            batch_size (int): Batch size for updates
            
        Returns:
            dict: Training history
        """
        history = {
            'value_loss': [],
            'policy_loss': [],
            'entropy': [],
            'rewards': [],
            'returns': [],
            'nav': []
        }
        
        for episode in range(n_episodes):
            # Collect trajectories
            trajectory = self.collect_trajectories(n_steps=n_steps_per_episode)
            
            # Update model
            metrics = self.update(trajectory, n_epochs=n_epochs, batch_size=batch_size)
            
            # Store metrics
            for k, v in metrics.items():
                history[k].append(v)
            
            history['rewards'].append(np.mean(trajectory['rewards']))
            history['returns'].append(np.mean(trajectory['returns']))
            history['nav'].append(self.env.nav)
            
            # Print progress
            print(f"Episode {episode+1}/{n_episodes}")
            print(f"  Reward: {history['rewards'][-1]:.4f}")
            print(f"  Return: {history['returns'][-1]:.4f}")
            print(f"  NAV: {history['nav'][-1]:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            
        return history
    
    def evaluate(self, env=None):
        """
        Evaluate the agent.
        
        Args:
            env (optional): Evaluation environment
            
        Returns:
            dict: Evaluation results
        """
        if env is None:
            env = self.env
            
        # Reset environment
        state = env.reset()
        done = False
        
        # Storage
        positions = []
        strategy_returns = []
        market_returns = []
        navs = []
        dates = []
        
        # Run episode
        while not done:
            # Convert observation to tensor
            features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
            position = torch.LongTensor([state['position']]).to(self.device)
            
            # Get action
            with torch.no_grad():
                action_mean, _ = self.model(features, position)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action_mean.cpu().numpy())
            
            # Store data
            positions.append(info['position'])
            strategy_returns.append(info['return'])
            market_returns.append(env.returns[env.current_step-1])
            navs.append(info['nav'])
            if info['date'] is not None:
                dates.append(info['date'])
            
            # Update state
            state = next_state
        
        # Calculate metrics
        total_return = np.sum(strategy_returns)
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-6)
        market_sharpe = np.mean(market_returns) / (np.std(market_returns) + 1e-6)
        
        # Calculate drawdown
        max_nav = np.maximum.accumulate(navs)
        drawdown = (max_nav - navs) / max_nav
        max_drawdown = np.max(drawdown)
        
        # Calculate market NAV
        market_nav = np.cumprod(1 + np.array(market_returns))
        
        # Calculate position changes
        position_changes = np.sum(np.abs(np.diff(positions)))
        
        results = {
            'positions': positions,
            'strategy_returns': strategy_returns,
            'market_returns': market_returns,
            'navs': navs,
            'market_nav': market_nav,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'market_sharpe': market_sharpe,
            'max_drawdown': max_drawdown,
            'position_changes': position_changes,
            'dates': dates if dates else None
        }
        
        return results
