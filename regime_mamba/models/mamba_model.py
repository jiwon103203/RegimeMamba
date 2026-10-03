import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP

class TimeSeriesMamba(nn.Module):
    def __init__(
        self,
        input_dim=4,        # Number of time series variables
        pred_len=1,         # Number of future points to predict
        d_model=128,        # Model dimension
        d_state=128,        # State dimension
        d_conv=4,           # Convolution kernel size
        expand=2,           # Expansion coefficient
        n_layers=4,         # Number of Mamba layers
        dropout=0.1,         # Dropout rate
        output_dim=1         # Output dimension
    ):
        """
        Mamba-based time series model implementation

        Args:
            input_dim: Number of input variables
            pred_len: Number of future points to predict
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion coefficient
            n_layers: Number of Mamba layers
            dropout: Dropout rate
            output_dim: Output dimension
        """
        super().__init__()

        self.input_dim = input_dim # (batch_size, seq_len, input_dim)
        self.d_model = d_model
        self.output_dim = output_dim

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Mamba blocks
        # Improved implementation
        self.blocks = nn.ModuleList([
            Block(
                dim=d_model,
                mixer_cls=lambda dim: Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                ),
                mlp_cls=lambda dim: GatedMLP(dim),
                fused_add_norm=True,  # Enhanced performance with fused Add and LayerNorm
                residual_in_fp32=True  # Maintain precision with fp32 format residuals
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Prediction head (predicts single value)
        self.pred_head = nn.Linear(d_model, self.output_dim)

    def forward(self, x, return_hidden=False):
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            return_hidden: Whether to return hidden state

        Returns:
            Predictions and optionally hidden state
        """
        # Embedding
        x = self.input_embedding(x)
        x = self.dropout(x)

        # Mamba processing
        residual = None
        for i,block in enumerate(self.blocks):
            x, residual = block(x, residual)
            if i < len(self.blocks) - 1:
                x = self.dropout(x)

        # Extract hidden state from last sequence position
        hidden = x[:, -1, :]  # [batch_size, d_model]

        # Prediction head (predicts return only)
        prediction = self.pred_head(hidden)  # [batch_size, 1]

        if return_hidden:
            return prediction, hidden
        return prediction
