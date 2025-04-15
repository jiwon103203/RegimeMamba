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
                 value_coef=0.5, entropy_coef=0.01, freeze_feature_extractor=True):
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

        # Freeze feature_extractor
        if freeze_feature_extractor:
            model.feature_extractor.eval()
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
        
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
        
        max_steps = min(n_steps, len(self.env.returns) - self.env.current_step)
        print(f"Collecting trajectories for {max_steps} steps")
        
        for step in range(max_steps):
            try:
                # Convert observation to tensor
                features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
                position = torch.LongTensor([state['position']]).to(self.device)
                
                # Debug info
                if step == 0:
                    print(f"Features shape: {features.shape}, Position: {position.item()}")
                
                # Get action and value
                with torch.no_grad():
                    _, _, _, _, hidden , _=self.model.feature_extractor(features, return_hidden=True)
                    action_mean, value = self.model(features, position)
                    # Add small noise for exploration
                    action_std = torch.ones_like(action_mean) * 0.1
                    action_dist = Normal(action_mean, action_std)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                
                if step == 0:
                    print(f"Action mean: {action_mean}, Action: {action}, Value: {value}")
                
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
                    print(f"Environment done after {step+1} steps")
                    break
                    
            except Exception as e:
                print(f"Error in step {step}: {e}")
                import traceback
                traceback.print_exc()
                if step == 0:
                    # If we fail on the first step, we can't continue
                    raise e
                break
        
        if len(rewards) == 0:
            raise ValueError("No trajectories collected")
            
        print(f"Collected {len(rewards)} steps")
        
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
        Update the model using PPO, with explicit tensor dimension management.
        
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
        values = torch.FloatTensor(trajectory['values']).to(self.device).unsqueeze(1)
        advantages = returns - values
        
        # PPO update
        metrics = {
            'value_loss': [],
            'policy_loss': [],
            'entropy': []
        }
        
        # Adjust batch size if needed
        actual_batch_size = min(batch_size, len(states))
        if actual_batch_size != batch_size:
            print(f"Adjusting batch size from {batch_size} to {actual_batch_size}")
        
        for epoch in range(n_epochs):
            # Generate random indices
            indices = np.random.permutation(len(states))
            
            # Iterate over mini-batches
            for start_idx in range(0, len(states), actual_batch_size):
                # Get mini-batch indices
                batch_indices = indices[start_idx:min(start_idx + actual_batch_size, len(states))]
                
                # Get mini-batch data
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]  # Correctly indexed now
                
                try:
                    # Forward pass
                    features_batch = torch.FloatTensor(np.stack([s['features'] for s in batch_states])).to(self.device)
                    position_batch = torch.LongTensor([s['position'] for s in batch_states]).to(self.device)
                    
                    # Debug shapes before model forward pass
                    if epoch == 0 and start_idx == 0:
                        print(f"Batch shapes - features: {features_batch.shape}, position: {position_batch.shape}")
                        print(f"Batch shapes - actions: {batch_actions.shape}, advantages: {batch_advantages.shape}")
                    
                    action_mean, value = self.model(features_batch, position_batch)
                    action_std = torch.ones_like(action_mean) * 0.1
                    action_dist = Normal(action_mean, action_std)
                    log_prob = action_dist.log_prob(batch_actions)
                    entropy = action_dist.entropy().mean()
                    
                    # Ensure log_prob and batch_old_log_probs have matching dimensions
                    if log_prob.dim() != batch_old_log_probs.dim() or log_prob.shape != batch_old_log_probs.shape:
                        if epoch == 0 and start_idx == 0:
                            print(f"Dimension mismatch: log_prob {log_prob.shape}, old_log_probs {batch_old_log_probs.shape}")
                        
                        if log_prob.dim() > batch_old_log_probs.dim():
                            batch_old_log_probs = batch_old_log_probs.unsqueeze(-1).expand_as(log_prob)
                        elif log_prob.dim() < batch_old_log_probs.dim():
                            log_prob = log_prob.unsqueeze(-1).expand_as(batch_old_log_probs)
                        else:
                            # Same number of dimensions but different shape
                            if log_prob.shape[0] == batch_old_log_probs.shape[0]:
                                # Try to broadcast the tensors
                                log_prob = log_prob.view(log_prob.shape[0], -1)
                                batch_old_log_probs = batch_old_log_probs.view(batch_old_log_probs.shape[0], -1)
                    
                    # Calculate ratio
                    ratio = torch.exp(log_prob - batch_old_log_probs)
                    
                    # Ensure ratio and batch_advantages have matching dimensions
                    if ratio.dim() != batch_advantages.dim() or ratio.shape != batch_advantages.shape:
                        if epoch == 0 and start_idx == 0:
                            print(f"Advantage shape mismatch: ratio {ratio.shape}, advantages {batch_advantages.shape}")
                        
                        # 배치 크기(첫 번째 차원)만 유지하고 나머지는 flatten
                        batch_size = ratio.shape[0]
                        
                        # 두 텐서를 [batch_size, -1] 형태로 변환
                        ratio = ratio.view(batch_size, -1)
                        batch_advantages = batch_advantages.view(batch_size, -1)
                        
                        # 더 작은 차원에 맞춤 (안전하게 첫 번째 요소만 사용)
                        ratio = ratio[:, :1]
                        batch_advantages = batch_advantages[:, :1]
                    
                    # Now we can safely compute the surrogate objectives
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
                    
                except Exception as e:
                    print(f"Error in update: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Calculate average metrics
        for k, v in metrics.items():
            if v:  # Only compute mean if list is not empty
                metrics[k] = np.mean(v)
            else:
                metrics[k] = 0.0
            
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
            print(f"\nStarting episode {episode+1}/{n_episodes}")
            
            # Reset environment for new episode
            self.env.reset()
            
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
            print(f"Episode {episode+1}/{n_episodes} completed")
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
        step = 0
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
            
            # Safety check
            step += 1
            if step > len(env.returns) * 2:  # Should never happen
                print("Warning: Evaluation loop appears to be stuck, breaking")
                break
        
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
