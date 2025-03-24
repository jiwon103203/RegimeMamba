import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mlp import GatedMLP

class TimeSeriesMamba(nn.Module):
    def __init__(
        self,
        input_dim=4,        # 시계열 변수의 수
        pred_len=1,         # 예측할 미래 시점 수
        d_model=128,        # 모델 차원
        d_state=128,        # 상태 차원
        d_conv=4,           # 컨볼루션 커널 크기
        expand=2,           # 확장 계수
        n_layers=4,         # Mamba 레이어 수
        dropout=0.1         # 드롭아웃 비율
    ):
        """
        Mamba 기반 시계열 모델 구현

        Args:
            input_dim: 입력 변수의 수
            pred_len: 예측할 미래 시점 수
            d_model: 모델 차원
            d_state: 상태 차원
            d_conv: 컨볼루션 커널 크기
            expand: 확장 계수
            n_layers: Mamba 레이어 수
            dropout: 드롭아웃 비율
        """
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # 입력 임베딩
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Mamba 블록
        self.blocks = nn.ModuleList([
            Block(
                dim=d_model,
                mixer_cls=lambda dim: Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                ),
                mlp_cls=lambda dim: GatedMLP(dim)
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 예측 헤드 (1개 값만 예측)
        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, x, return_hidden=False):
        """
        순방향 전파

        Args:
            x: 입력 텐서 [batch_size, seq_len, input_dim]
            return_hidden: hidden state를 반환할지 여부

        Returns:
            예측값 및 선택적으로 hidden state
        """
        # 임베딩
        x = self.input_embedding(x)

        # Mamba 처리
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual)
            x = self.dropout(x)

        x = self.norm(x)

        # 마지막 시퀀스 포지션의 hidden state 추출
        hidden = x[:, -1, :]  # [batch_size, d_model]

        # 예측 헤드 (수익률만 예측)
        prediction = self.pred_head(hidden)  # [batch_size, 1]

        if return_hidden:
            return prediction, hidden
        return prediction

def create_model_from_config(config):
    """
    설정으로부터 모델 생성
    
    Args:
        config: 설정 객체
        
    Returns:
        model: 생성된 모델
    """
    model = TimeSeriesMamba(
        input_dim=config.input_dim,  # 기본적인 입력 차원은 4차원 (returns, dd_10, sortino_20, sortino_60)
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        n_layers=config.n_layers,
        dropout=config.dropout
    )
    return model