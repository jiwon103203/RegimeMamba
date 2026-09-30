"""
End-to-End Regime Mamba Model Implementation
Uses Gumbel Softmax for differentiable regime detection, enabling joint optimization
of feature extraction (Mamba) and regime classification in a single training loop.

Updated: Two-Level Entropy Regularization (sample-level minimize + batch-level maximize)
         Direction Loss removed (return sign assumption eliminated)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP


class DifferentiableRegimeHead(nn.Module):
    """
    Differentiable regime classification head using Gumbel Softmax.
    Enables gradient flow through discrete regime assignments.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_regimes: int = 2, 
        temperature: float = 1.0,
        hidden_dim: Optional[int] = None
    ):
        """
        Args:
            d_model: Input dimension from Mamba hidden states
            n_regimes: Number of market regimes (default: 2 for Bull/Bear)
            temperature: Gumbel Softmax temperature (lower = harder assignments)
            hidden_dim: Optional hidden layer dimension for regime projection
        """
        super().__init__()
        self.n_regimes = n_regimes
        self.temperature = temperature
        
        if hidden_dim is not None:
            self.regime_proj = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, n_regimes)
            )
        else:
            self.regime_proj = nn.Linear(d_model, n_regimes)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Gumbel Softmax for differentiable regime sampling.
        
        Args:
            hidden_states: (batch, d_model) or (batch, seq_len, d_model)
            hard: If True, use Straight-Through Estimator (discrete forward, continuous backward)
            
        Returns:
            regime_probs: Soft/hard regime assignments
            logits: Raw logits before Gumbel Softmax
        """
        logits = self.regime_proj(hidden_states)
        
        if self.training:
            # Gumbel Softmax for differentiable sampling
            regime_probs = F.gumbel_softmax(
                logits, 
                tau=self.temperature, 
                hard=hard,
                dim=-1
            )
        else:
            # Hard assignment at inference
            indices = logits.argmax(dim=-1)
            regime_probs = F.one_hot(indices, num_classes=self.n_regimes).float()
        
        return regime_probs, logits
    
    def get_regime_labels(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get hard regime labels for inference."""
        logits = self.regime_proj(hidden_states)
        return logits.argmax(dim=-1)


class JumpPenaltyModule(nn.Module):
    """
    Differentiable jump penalty that penalizes frequent regime switches.
    Replaces the discrete jump penalty from the original JumpModel.
    
    Supports date-aware computation to only penalize actual consecutive days.
    """
    
    def __init__(self, lambda_jump: float = 1.0, penalty_type: str = 'l2'):
        """
        Args:
            lambda_jump: Jump penalty coefficient
            penalty_type: Type of penalty ('l1', 'l2', 'kl')
        """
        super().__init__()
        self.lambda_jump = lambda_jump
        self.penalty_type = penalty_type
    
    def forward(
        self, 
        regime_probs: torch.Tensor,
        dates: Optional[List] = None,
        max_gap_days: int = 5
    ) -> torch.Tensor:
        """
        Compute jump penalty loss.
        
        Args:
            regime_probs: (batch, seq_len, n_regimes) or (batch, n_regimes)
                         If 2D, assumes sequential samples in batch dimension
            dates: Optional list of dates for each sample. If provided, only
                   penalizes transitions between samples within max_gap_days.
            max_gap_days: Maximum number of days between samples to consider
                          them as consecutive (default: 5 for weekends/holidays)
                         
        Returns:
            Jump penalty loss value
        """
        if regime_probs.dim() == 2:
            batch_size = regime_probs.size(0)
            
            if batch_size < 2:
                return torch.tensor(0.0, device=regime_probs.device)
            
            # If dates provided, create mask for consecutive samples
            if dates is not None and len(dates) == batch_size:
                consecutive_mask = self._compute_consecutive_mask(dates, max_gap_days)
                consecutive_mask = consecutive_mask.to(regime_probs.device)
            else:
                # Assume all samples are consecutive (original behavior)
                consecutive_mask = torch.ones(batch_size - 1, device=regime_probs.device)
            
            # Only compute if there are consecutive pairs
            if consecutive_mask.sum() == 0:
                return torch.tensor(0.0, device=regime_probs.device)
            
            # Compute regime differences
            regime_diff = regime_probs[1:] - regime_probs[:-1]
            
            if self.penalty_type == 'l1':
                jump_costs = torch.abs(regime_diff).sum(dim=-1)
            elif self.penalty_type == 'l2':
                jump_costs = torch.norm(regime_diff, p=2, dim=-1)
            elif self.penalty_type == 'kl':
                eps = 1e-8
                p = regime_probs[:-1] + eps
                q = regime_probs[1:] + eps
                jump_costs = (p * (p.log() - q.log())).sum(dim=-1)
            else:
                raise ValueError(f"Unknown penalty type: {self.penalty_type}")
            
            # Apply mask and compute mean
            masked_costs = jump_costs * consecutive_mask
            n_consecutive = consecutive_mask.sum()
            
            if n_consecutive > 0:
                jump_cost = masked_costs.sum() / n_consecutive
            else:
                jump_cost = torch.tensor(0.0, device=regime_probs.device)
                
        else:
            # (batch, seq_len, n_regimes) - within-sequence jumps
            regime_diff = regime_probs[:, 1:, :] - regime_probs[:, :-1, :]
            
            if self.penalty_type == 'l1':
                jump_cost = torch.abs(regime_diff).sum(dim=-1).mean()
            elif self.penalty_type == 'l2':
                jump_cost = torch.norm(regime_diff, p=2, dim=-1).mean()
            elif self.penalty_type == 'kl':
                eps = 1e-8
                p = regime_probs[:, :-1, :] + eps
                q = regime_probs[:, 1:, :] + eps
                jump_cost = (p * (p.log() - q.log())).sum(dim=-1).mean()
            else:
                raise ValueError(f"Unknown penalty type: {self.penalty_type}")
        
        return self.lambda_jump * jump_cost
    
    def _compute_consecutive_mask(
        self, 
        dates: List, 
        max_gap_days: int
    ) -> torch.Tensor:
        """
        Compute mask indicating which adjacent samples are actually consecutive in time.
        
        Args:
            dates: List of dates (strings or datetime objects)
            max_gap_days: Maximum gap in days to consider consecutive
            
        Returns:
            Boolean mask tensor of shape (batch_size - 1,)
        """
        import pandas as pd
        
        # Convert to datetime if needed
        if len(dates) == 0:
            return torch.tensor([], dtype=torch.float32)
        
        parsed_dates = []
        for d in dates:
            if isinstance(d, str):
                parsed_dates.append(pd.to_datetime(d))
            elif hasattr(d, 'timestamp'):  # datetime-like
                parsed_dates.append(d)
            else:
                # Try to convert
                try:
                    parsed_dates.append(pd.to_datetime(d))
                except:
                    # If conversion fails, assume consecutive
                    return torch.ones(len(dates) - 1, dtype=torch.float32)
        
        # Compute gaps between consecutive samples
        mask = []
        for i in range(len(parsed_dates) - 1):
            gap = abs((parsed_dates[i+1] - parsed_dates[i]).days)
            # Consider consecutive if gap is within max_gap_days
            mask.append(1.0 if gap <= max_gap_days else 0.0)
        
        return torch.tensor(mask, dtype=torch.float32)


class SortedBatchJumpPenalty(nn.Module):
    """
    Jump penalty that sorts batch by date before computing penalties.
    More efficient when batch contains samples from different time periods.
    """
    
    def __init__(self, lambda_jump: float = 1.0, penalty_type: str = 'l2', max_gap_days: int = 5):
        super().__init__()
        self.lambda_jump = lambda_jump
        self.penalty_type = penalty_type
        self.max_gap_days = max_gap_days
        self.base_penalty = JumpPenaltyModule(lambda_jump=1.0, penalty_type=penalty_type)
    
    def forward(
        self, 
        regime_probs: torch.Tensor,
        dates: Optional[List] = None
    ) -> torch.Tensor:
        """
        Sort batch by date, then compute jump penalty on sorted sequence.
        """
        if dates is None or len(dates) == 0:
            return self.base_penalty(regime_probs)
        
        import pandas as pd
        
        # Parse and sort dates
        try:
            parsed_dates = [pd.to_datetime(d) for d in dates]
            sorted_indices = sorted(range(len(parsed_dates)), key=lambda i: parsed_dates[i])
            sorted_dates = [parsed_dates[i] for i in sorted_indices]
        except:
            return self.base_penalty(regime_probs)
        
        # Reorder regime_probs by date
        sorted_probs = regime_probs[sorted_indices]
        
        # Compute penalty with date-aware mask
        return self.base_penalty(sorted_probs, sorted_dates, self.max_gap_days) * self.lambda_jump


class RegimeSeparationLoss(nn.Module):
    """
    Encourages different regimes to have distinct hidden state representations.
    Uses centroid-based separation and intra-cluster compactness in hidden space.
    """
    
    def __init__(
        self, 
        margin: float = 1.0, 
        loss_type: str = 'centroid',
        lambda_inter: float = 1.0,
        lambda_intra: float = 1.0
    ):
        """
        Args:
            margin: Minimum margin between regime centroids
            loss_type: Type of separation loss ('centroid', 'contrastive', 'silhouette')
            lambda_inter: Weight for inter-cluster separation (push centroids apart)
            lambda_intra: Weight for intra-cluster compactness (pull samples to centroid)
        """
        super().__init__()
        self.margin = margin
        self.loss_type = loss_type
        self.lambda_inter = lambda_inter
        self.lambda_intra = lambda_intra
    
    def forward(
        self, 
        regime_probs: torch.Tensor, 
        hidden_states: torch.Tensor,
        returns: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute regime separation loss based on hidden states.
        
        Args:
            regime_probs: (batch, n_regimes) soft regime assignments
            hidden_states: (batch, d_model) hidden representations from Mamba
            returns: (batch,) optional returns for return-aware separation
            
        Returns:
            Separation loss value
        """
        if self.loss_type == 'centroid':
            return self._centroid_loss(regime_probs, hidden_states)
        elif self.loss_type == 'contrastive':
            return self._contrastive_loss(regime_probs, hidden_states)
        elif self.loss_type == 'silhouette':
            return self._silhouette_loss(regime_probs, hidden_states)
        elif self.loss_type == 'return_weighted':
            return self._return_weighted_loss(regime_probs, hidden_states, returns)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _centroid_loss(
        self, 
        regime_probs: torch.Tensor, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Centroid-based loss: maximize inter-centroid distance, minimize intra-cluster variance.
        """
        n_regimes = regime_probs.size(-1)
        
        # Compute soft centroids for each regime
        weighted_sum = torch.einsum('bn,bd->nd', regime_probs, hidden_states)
        regime_weights = regime_probs.sum(dim=0, keepdim=True).T + 1e-8
        centroids = weighted_sum / regime_weights
        
        # Inter-cluster loss: maximize pairwise centroid distances
        inter_loss = torch.tensor(0.0, device=hidden_states.device)
        n_pairs = 0
        for i in range(n_regimes):
            for j in range(i + 1, n_regimes):
                dist = F.pairwise_distance(
                    centroids[i:i+1], 
                    centroids[j:j+1]
                ).squeeze()
                inter_loss = inter_loss + F.relu(self.margin - dist)
                n_pairs += 1
        
        if n_pairs > 0:
            inter_loss = inter_loss / n_pairs
        
        # Intra-cluster loss: minimize weighted distance to assigned centroid
        hidden_expanded = hidden_states.unsqueeze(1)
        centroids_expanded = centroids.unsqueeze(0)
        sq_distances = ((hidden_expanded - centroids_expanded) ** 2).sum(dim=-1)
        intra_loss = (regime_probs * sq_distances).sum(dim=-1).mean()
        
        total_loss = self.lambda_intra * intra_loss + self.lambda_inter * inter_loss
        
        return total_loss
    
    def _contrastive_loss(
        self, 
        regime_probs: torch.Tensor, 
        hidden_states: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        Contrastive loss: samples in same regime should be closer than samples in different regimes.
        """
        batch_size = hidden_states.size(0)
        
        hidden_norm = F.normalize(hidden_states, p=2, dim=-1)
        sim_matrix = torch.mm(hidden_norm, hidden_norm.t())
        same_regime_prob = torch.mm(regime_probs, regime_probs.t())
        
        mask = torch.eye(batch_size, device=hidden_states.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        same_regime_prob = same_regime_prob.masked_fill(mask, 0)
        
        exp_sim = torch.exp(sim_matrix / temperature)
        pos_sim = (exp_sim * same_regime_prob).sum(dim=-1)
        all_sim = exp_sim.sum(dim=-1)
        
        loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8)).mean()
        
        return loss
    
    def _silhouette_loss(
        self, 
        regime_probs: torch.Tensor, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable silhouette-like score.
        """
        n_regimes = regime_probs.shape[-1]
        
        dist_matrix = torch.cdist(hidden_states, hidden_states, p=2)
        
        weighted_sum = torch.einsum('bn,bd->nd', regime_probs, hidden_states)
        regime_weights = regime_probs.sum(dim=0, keepdim=True).T + 1e-8
        centroids = weighted_sum / regime_weights
        
        dist_to_centroids = torch.cdist(hidden_states, centroids, p=2)
        
        a = (regime_probs * dist_to_centroids).sum(dim=-1)
        
        hard_assignment = regime_probs.argmax(dim=-1)
        own_cluster_mask = F.one_hot(hard_assignment, num_classes=n_regimes).bool()
        masked_dist = dist_to_centroids.masked_fill(own_cluster_mask, float('inf'))
        b = masked_dist.min(dim=-1).values
        
        max_ab = torch.max(a, b)
        silhouette = (b - a) / (max_ab + 1e-8)
        
        loss = -silhouette.mean()
        
        return loss
    
    def _return_weighted_loss(
        self, 
        regime_probs: torch.Tensor, 
        hidden_states: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Hybrid loss: hidden space separation with return-based guidance.
        """
        if returns is None:
            return self._centroid_loss(regime_probs, hidden_states)
        
        returns = returns.view(-1)
        n_regimes = regime_probs.size(-1)
        
        centroid_loss = self._centroid_loss(regime_probs, hidden_states)
        
        if n_regimes == 2:
            return_direction = torch.sigmoid(returns * 10)
            soft_labels = torch.stack([1 - return_direction, return_direction], dim=-1)
            log_probs = F.log_softmax(regime_probs, dim=-1)
            guidance_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
        else:
            guidance_loss = torch.tensor(0.0, device=hidden_states.device)
        
        total_loss = centroid_loss + 0.1 * guidance_loss
        
        return total_loss


class EntropyRegularization(nn.Module):
    """
    Two-Level Entropy Regularization:
    - Minimize sample-level entropy: each sample should be confident (close to [0,1] or [1,0])
    - Maximize batch-level entropy: overall distribution should be balanced
    
    This breaks the symmetric equilibrium where all samples output [0.5, 0.5]
    
    L = lambda_sample * (1/N) * Σ H(p_i) - lambda_batch * H(p̄)
    """
    
    def __init__(
        self, 
        lambda_entropy: float = 0.1, 
        target_entropy: Optional[float] = None,
        lambda_sample: float = 1.0,
        lambda_batch: float = 0.5
    ):
        """
        Args:
            lambda_entropy: Overall entropy regularization coefficient (legacy, for compatibility)
            target_entropy: Target entropy value (None = use two-level approach)
            lambda_sample: Weight for sample-level entropy minimization (confidence)
            lambda_batch: Weight for batch-level entropy maximization (balance)
        """
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.target_entropy = target_entropy
        self.lambda_sample = lambda_sample
        self.lambda_batch = lambda_batch
    
    def forward(self, regime_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute two-level entropy loss.
        
        Args:
            regime_probs: (batch, n_regimes) soft regime assignments
            
        Returns:
            Total entropy loss and component breakdown
        """
        eps = 1e-8
        n_regimes = regime_probs.size(-1)
        max_entropy = np.log(n_regimes)
        
        # ==========================================
        # Sample-level entropy: MINIMIZE (confidence)
        # ==========================================
        # H(p_i) for each sample
        sample_entropy = -(regime_probs * (regime_probs + eps).log()).sum(dim=-1)  # (batch,)
        avg_sample_entropy = sample_entropy.mean()
        
        # Normalize by max entropy for stable scaling
        sample_loss = avg_sample_entropy / max_entropy  # want to minimize
        
        # ==========================================
        # Batch-level entropy: MAXIMIZE (balance)
        # ==========================================
        # H(p̄) where p̄ = mean(p_i)
        avg_probs = regime_probs.mean(dim=0)  # (n_regimes,)
        batch_entropy = -(avg_probs * (avg_probs + eps).log()).sum()
        
        # Normalize
        batch_loss = -batch_entropy / max_entropy  # negative because we want to maximize
        
        # ==========================================
        # Total loss
        # ==========================================
        # Legacy mode: if target_entropy is set, use old behavior
        if self.target_entropy is not None:
            total_loss = self.lambda_entropy * (batch_entropy - self.target_entropy) ** 2
            components = {
                'sample_entropy': avg_sample_entropy.item(),
                'batch_entropy': batch_entropy.item(),
                'sample_loss': 0.0,
                'batch_loss': total_loss.item()
            }
            return total_loss, components
        
        # Two-level mode
        total_loss = self.lambda_sample * sample_loss + self.lambda_batch * batch_loss
        
        # Apply overall scaling
        total_loss = self.lambda_entropy * total_loss
        
        # Return components for logging
        components = {
            'sample_entropy': avg_sample_entropy.item(),
            'batch_entropy': batch_entropy.item(),
            'sample_loss': (self.lambda_sample * sample_loss).item(),
            'batch_loss': (self.lambda_batch * batch_loss).item()
        }
        
        return total_loss, components


class EndToEndRegimeMamba(nn.Module):
    """
    End-to-End Regime Mamba model that jointly learns feature extraction
    and regime detection using Gumbel Softmax for differentiable training.
    
    Updated: Two-Level Entropy, No Direction Loss
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config
        
        # Feature extraction (Mamba backbone)
        self.input_embedding = nn.Linear(config.input_dim, config.d_model)
        
        self.blocks = nn.ModuleList([
            Block(
                dim=config.d_model,
                mixer_cls=lambda dim: Mamba(
                    d_model=dim,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand
                ),
                mlp_cls=lambda dim: GatedMLP(dim),
                fused_add_norm=True,
                residual_in_fp32=True
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Regime detection head
        self.regime_head = DifferentiableRegimeHead(
            d_model=config.d_model,
            n_regimes=config.n_clusters,
            temperature=getattr(config, 'temperature', 1.0),
            hidden_dim=getattr(config, 'regime_hidden_dim', None)
        )
        
        # Loss modules
        self.jump_penalty = JumpPenaltyModule(
            lambda_jump=getattr(config, 'jump_penalty', 1.0),
            penalty_type=getattr(config, 'jump_penalty_type', 'l2')
        )
        
        self.separation_loss = RegimeSeparationLoss(
            margin=getattr(config, 'separation_margin', 1.0),
            loss_type=getattr(config, 'separation_loss_type', 'centroid'),
            lambda_inter=getattr(config, 'lambda_inter', 1.0),
            lambda_intra=getattr(config, 'lambda_intra', 1.0)
        )
        
        # Two-Level Entropy Regularization
        self.entropy_reg = EntropyRegularization(
            lambda_entropy=getattr(config, 'lambda_entropy', 1.0),
            target_entropy=getattr(config, 'target_entropy', None),
            lambda_sample=getattr(config, 'lambda_sample_entropy', 1.0),
            lambda_batch=getattr(config, 'lambda_batch_entropy', 0.5)
        )
        
        # Optional: Return prediction head for auxiliary loss
        self.return_pred_head = nn.Linear(config.d_model, 1)
        
        # Loss weights (NO w_direction)
        self.w_return = getattr(config, 'w_return', 0.5)
        self.w_jump = getattr(config, 'w_jump', 0.5)
        self.w_separation = getattr(config, 'w_separation', 2.0)
        self.w_entropy = getattr(config, 'w_entropy', 1.0)  # Increased importance
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract hidden features using Mamba backbone.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Hidden states (batch, d_model)
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
    
    def forward(
        self, 
        x: torch.Tensor, 
        hard: bool = False,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            hard: Whether to use hard regime assignments
            return_all: Whether to return all intermediate outputs
            
        Returns:
            Dictionary containing regime_probs, logits, hidden, and optionally return_pred
        """
        # Extract features
        hidden = self.extract_features(x)
        
        # Regime classification
        regime_probs, logits = self.regime_head(hidden, hard=hard)
        
        outputs = {
            'regime_probs': regime_probs,
            'logits': logits,
            'hidden': hidden
        }
        
        if return_all:
            # Return prediction
            return_pred = self.return_pred_head(hidden)
            outputs['return_pred'] = return_pred
        
        return outputs
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        returns: torch.Tensor,
        dates: Optional[List] = None,
        hard: bool = False,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute total loss for end-to-end training.
        
        Loss = w_return * L_return 
             + w_jump * L_jump 
             + w_separation * L_separation 
             + w_entropy * L_entropy_two_level
        
        NOTE: Direction loss has been REMOVED (no return sign assumption)
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            returns: Target returns (batch,) or (batch, 1)
            dates: Optional list of dates for jump penalty calculation
            hard: Whether to use hard regime assignments
            return_components: Whether to return individual loss components
            
        Returns:
            Total loss (and optionally loss components dict)
        """
        # Always get BOTH soft and hard outputs
        outputs_soft = self.forward(x, hard=False, return_all=True)
        soft_regime_probs = outputs_soft['regime_probs']
        hidden = outputs_soft['hidden']
        return_pred = outputs_soft['return_pred']
        
        if hard:
            outputs_hard = self.forward(x, hard=True, return_all=False)
            regime_probs = outputs_hard['regime_probs']
        else:
            regime_probs = soft_regime_probs
        
        returns = returns.view(-1)
        
        # 1. Return prediction loss (auxiliary)
        return_loss = F.mse_loss(return_pred.squeeze(), returns)
        
        # 2. Jump penalty - ALWAYS use soft probs for gradient flow
        jump_loss = self.jump_penalty(soft_regime_probs, dates=dates)
        
        # 3. Regime separation loss (based on hidden states)
        separation_loss = self.separation_loss(soft_regime_probs, hidden, returns)
        
        # 4. Two-Level Entropy loss
        entropy_loss, entropy_components = self.entropy_reg(soft_regime_probs)
        
        # Total loss (NO direction loss)
        total_loss = (
            self.w_return * return_loss +
            self.w_jump * jump_loss +
            self.w_separation * separation_loss +
            self.w_entropy * entropy_loss
        )
        
        if return_components:
            components = {
                'return_loss': return_loss.item(),
                'jump_loss': jump_loss.item(),
                'separation_loss': separation_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'sample_entropy': entropy_components['sample_entropy'],
                'batch_entropy': entropy_components['batch_entropy'],
                'total_loss': total_loss.item()
            }
            return total_loss, components
        
        return total_loss
    
    def predict_regimes(
        self,
        x: torch.Tensor,
        returns: torch.Tensor = None,
        hard: bool = True,
        align_by_returns: bool = True
    ) -> torch.Tensor:
        """
        Predict regime labels with optional alignment by returns.
        
        Ensures that:
        - Regime 1 = Bull (higher mean return)
        - Regime 0 = Bear (lower mean return)
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            returns: Return values for each sample (batch,) - required if align_by_returns=True
            hard: Whether to use hard assignments
            align_by_returns: Whether to align regime labels by mean returns
            
        Returns:
            Regime labels (batch,) with 1=Bull, 0=Bear
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, hard=hard)
            regime_probs = outputs['regime_probs']
            
            # Get hard regime assignments (argmax)
            regimes = regime_probs.argmax(dim=-1)  # (batch,)
            
            if align_by_returns and returns is not None:
                regimes = self._align_regime_labels(regimes, returns)
            
            return regimes
    
    def _align_regime_labels(
        self,
        regimes: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Align regime labels so that Regime 1 = Bull (higher returns), Regime 0 = Bear.
        
        Args:
            regimes: Regime labels (batch,) with values 0 or 1
            returns: Return values (batch,)
            
        Returns:
            Aligned regime labels (batch,)
        """
        # Ensure returns is 1D
        if returns.dim() > 1:
            returns = returns.squeeze(-1)
        
        # Calculate mean return for each regime
        regime_0_mask = (regimes == 0)
        regime_1_mask = (regimes == 1)
        
        # Handle edge cases where one regime has no samples
        if regime_0_mask.sum() == 0 or regime_1_mask.sum() == 0:
            return regimes
        
        mean_return_0 = returns[regime_0_mask].mean()
        mean_return_1 = returns[regime_1_mask].mean()
        
        # If regime 0 has higher mean return, swap labels
        if mean_return_0 > mean_return_1:
            regimes = 1 - regimes
        
        return regimes
    
    def get_regime_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get soft regime probabilities for inference.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Regime probabilities (batch, n_regimes)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, hard=False)
        return outputs['regime_probs']


def align_regime_labels_numpy(
    regimes: np.ndarray,
    returns: np.ndarray
) -> np.ndarray:
    """
    Align regime labels so that Regime 1 = Bull (higher returns), Regime 0 = Bear.
    
    Standalone numpy function for use outside of model.
    
    Args:
        regimes: Regime labels array with values 0 or 1
        returns: Return values array
        
    Returns:
        Aligned regime labels array
    """
    regimes = np.asarray(regimes).copy()
    returns = np.asarray(returns).flatten()
    
    # Calculate mean return for each regime
    regime_0_mask = (regimes == 0)
    regime_1_mask = (regimes == 1)
    
    # Handle edge cases
    if regime_0_mask.sum() == 0 or regime_1_mask.sum() == 0:
        return regimes
    
    mean_return_0 = returns[regime_0_mask].mean()
    mean_return_1 = returns[regime_1_mask].mean()
    
    # If regime 0 has higher mean return, swap labels
    if mean_return_0 > mean_return_1:
        regimes = 1 - regimes
    
    return regimes


class SequentialJumpPenalty(nn.Module):
    """
    Enhanced jump penalty that operates on sequential data within a batch.
    Useful when data loader preserves temporal order.
    """
    
    def __init__(
        self, 
        lambda_jump: float = 1.0, 
        lookback: int = 5,
        decay: float = 0.9
    ):
        super().__init__()
        self.lambda_jump = lambda_jump
        self.lookback = lookback
        self.decay = decay
        self.register_buffer('prev_regime_probs', None)
    
    def forward(
        self, 
        regime_probs: torch.Tensor, 
        reset: bool = False
    ) -> torch.Tensor:
        if reset or self.prev_regime_probs is None:
            self.prev_regime_probs = regime_probs.detach().clone()
            return torch.tensor(0.0, device=regime_probs.device)
        
        diff = regime_probs - self.prev_regime_probs
        jump_cost = torch.norm(diff, p=2, dim=-1).mean()
        
        self.prev_regime_probs = regime_probs.detach().clone()
        
        return self.lambda_jump * jump_cost
    
    def reset_memory(self):
        self.prev_regime_probs = None


def create_e2e_model_from_config(config) -> EndToEndRegimeMamba:
    """
    Factory function to create EndToEndRegimeMamba from config.
    
    Args:
        config: Configuration object
        
    Returns:
        EndToEndRegimeMamba model
    """
    # Set default values for new config parameters
    default_params = {
        'temperature': 1.0,
        'regime_hidden_dim': None,
        'jump_penalty': 1.0,
        'jump_penalty_type': 'l2',
        'separation_margin': 2.0,
        'separation_loss_type': 'centroid',
        'lambda_inter': 1.5,
        'lambda_intra': 1.0,
        'lambda_entropy': 1.0,  # Increased
        'target_entropy': None,
        'lambda_sample_entropy': 1.0,  # NEW
        'lambda_batch_entropy': 0.5,   # NEW
        'w_return': 0.5,
        'w_jump': 0.5,
        'w_separation': 2.0,
        'w_entropy': 1.0  # Increased
    }
    
    for key, value in default_params.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    return EndToEndRegimeMamba(config)
