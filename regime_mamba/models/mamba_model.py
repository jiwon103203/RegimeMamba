import torch
import torch.nn as nn
import numpy as np
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
        dropout=0.1,         # 드롭아웃 비율
        output_dim=1,        # 출력 차원
        config=None         # 설정 객체
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
            output_dim: 출력 차원
            config: 설정 객체
        """
        super().__init__()

        self.input_dim = input_dim # (batch_size, seq_len, input_dim)
        self.d_model = d_model
        self.output_dim = output_dim
        self.config = config

        if self.config is not None:
            self.input_dim = config.input_dim
            self.d_model = config.d_model
            self.output_dim = 3 if config.direct_train else 1
            
            if self.config.direct_train:
                self.softmax = nn.Softmax(dim=1)
            
            if self.config.vae:
                self.start_point = int(np.log2(config.d_model)) - 1 # d_state가 128이면 6, 64이면 5
                self.latent_dim = 2^(self.start_point-2)
                self.fc_mu = nn.Linear(config.d_model, self.latent_dim)
                self.fc_var = nn.Linear(config.d_model, self.latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(2^(self.start_point-2), 2^(self.start_point-1)),
                    nn.BatchNorm1d(2^(self.start_point-1)),
                    nn.ReLU(),
                    nn.Linear(2^(self.start_point-1), 2^self.start_point),
                    nn.BatchNorm1d(2^self.start_point),
                    nn.ReLU(),
                    nn.Linear(2^self.start_point, config.d_model),
                    nn.BatchNorm1d(config.d_model),
                    nn.ReLU(),
                    )
                

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
        self.pred_head = nn.Linear(d_model, self.output_dim)

    def encode(self, x):
        """
        VAE 인코더: hidden state에서 평균과 로그 분산을 계산

        Args:
            x: hidden state 텐서 [batch_size, d_model]

        Returns:
            mu: 평균 [batch_size, latent_dim]
            log_var: 로그 분산 [batch_size, latent_dim]
        """
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        log_var = torch.clamp(log_var, -10, 10)  # 로그 분산의 최대/최소 값 제한

        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: 평균과 로그 분산에서 잠재 벡터 샘플링

        Args:
            mu: 평균 [batch_size, latent_dim]
            log_var: 로그 분산 [batch_size, latent_dim]

        Returns:
            z: 샘플링된 잠재 벡터 [batch_size, latent_dim]
        """

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        VAE 디코더

        Args:
            z: 잠재 벡터 [batch_size, latent_dim]

        Returns:
            reonstructed: 재구성된 텐서 [batch_size, d_model]
        """
        return self.decoder(z)


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
        if self.output_dim > 1:
            prediction = self.softmax(prediction)

        if self.config and self.config.vae:
            mu, log_var = self.encode(hidden)
            z = self.reparameterize(mu, log_var)
            reconstructed = self.decode(z)
            return prediction, hidden, mu, log_var, z ,reconstructed

        if return_hidden:
            return prediction, hidden
        return prediction

    def vae_loss_function(self, recon_hidden, hidden, mu, log_var, pred, target, beta=0.01):
        """
        VAE 손실 함수: 재구성 손실 + KL 발산 + 예측 손실
        
        Args:
            recon_hidden: 재구성된 hidden state
            hidden: 원본 hidden state
            mu: 잠재 공간의 평균
            log_var: 잠재 공간의 로그 분산
            pred: 모델의 예측값
            target: 실제 타겟값
            beta: KL 발산 가중치
            
        Returns:
            total_loss: 전체 손실
            recon_loss: 재구성 손실
            kl_loss: KL 발산 손실
            pred_loss: 예측 손실
        """
        # MSE 손실 계산기
        mse_loss = nn.MSELoss(reduction='sum')
        
        # 재구성 손실 (hidden state 복원에 대한 MSE)
        recon_loss = mse_loss(recon_hidden, hidden)
        
        # KL 발산: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = torch.sum(0.5 * (mu.pow(2) + log_var.exp() - log_var - 1))
        
        # 예측 손실 (주요 태스크에 대한 손실)
        if pred.shape[1] > 1:  # 다중 클래스 분류인 경우
            pred_loss = nn.CrossEntropyLoss()(pred, target)
        else:  # 회귀인 경우
            pred_loss = mse_loss(pred.squeeze(), target)
        
        # 전체 손실
        total_loss = pred_loss + 0.1*recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss, pred_loss

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
        dropout=config.dropout,
        output_dim=3 if config.direct_train else 1,
        config=config
    )
    return model