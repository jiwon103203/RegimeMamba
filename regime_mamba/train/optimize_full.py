import json
import numpy as np
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization
import torch

from ..data.dataset_full import RegimeMambaDataset
from ..models.mamba_model import TimeSeriesMamba
from ..config.config import RegimeMambaConfig
from .train import train_with_early_stopping

def optimize_regime_mamba_bayesian(data_path, base_config, n_iterations=30, save_path=None):
    """
    베이지안 최적화를 사용하여 RegimeMamba 모델의 하이퍼파라미터를 최적화
    배치 사이즈는 2048로 고정

    Args:
        data_path: 데이터 파일 경로
        base_config: 기본 설정 객체
        n_iterations: 최적화 반복 횟수
        save_path: 최적화 결과 저장 경로 (None이면 저장하지 않음)

    Returns:
        optimized_config: 최적화된 설정 객체
    """
    # 최적화 대상 함수 정의
    def evaluate_model(d_model_exp, d_state_exp, learning_rate_exp):
        """하이퍼파라미터 조합을 평가하고 검증 성능을 반환 (배치 사이즈 고정)"""
        # 연속 파라미터 공간에서 실제 값으로 변환
        current_config = RegimeMambaConfig()
        for key, value in base_config.__dict__.items():
            if value is not None and hasattr(current_config, key):
                # base_config의 속성을 current_config에 복사
                setattr(current_config, key, value)
        current_config.d_model = int(2 ** d_model_exp)  # 64 ~ 256 범위
        current_config.d_state = int(2 ** d_state_exp)  # 64 ~ 256 범위
        current_config.n_layers = base_config.n_layers  # config에서 고정
        current_config.dropout = 0.1                    # 0.1로 고정
        current_config.learning_rate = 10 ** learning_rate_exp  # 1e-6 ~ 1e-4 범위
        current_config.batch_size = 2048                 # 배치 사이즈 2048로 고정
        current_config.seq_len = base_config.target_horizon  # config에서 고정
        current_config.device = base_config.device
        print(f"\n평가 중인 하이퍼파라미터:")
        print(f"  d_model: {current_config.d_model}")
        print(f"  d_state: {current_config.d_state}")
        print(f"  n_layers: {current_config.n_layers} (고정)")
        print(f"  dropout: {current_config.dropout:.3f} (고정)")
        print(f"  learning_rate: {current_config.learning_rate:.2e}")
        print(f"  batch_size: {current_config.batch_size} (고정)")
        print(f"  seq_len: {current_config.seq_len}")

        # 데이터셋 및 데이터로더 생성
        try:
            train_dataset = RegimeMambaDataset(config = current_config, mode="train")
            valid_dataset = RegimeMambaDataset(config = current_config, mode="valid")
            train_loader = DataLoader(
                train_dataset,
                batch_size=current_config.batch_size,
                shuffle=True,
                num_workers=2
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=current_config.batch_size,
                shuffle=False,
                num_workers=2
            )

            # 모델 초기화
            model = TimeSeriesMamba(
                input_dim=current_config.input_dim,
                d_model=current_config.d_model,
                d_state=current_config.d_state,
                d_conv=current_config.d_conv,
                expand=current_config.expand,
                n_layers=current_config.n_layers,
                dropout=current_config.dropout,
                output_dim = 3 if current_config.direct_train else 1,
                config = current_config
            )

            # 조기 종료를 적용한 모델 훈련
            best_val_loss, early_stop_epoch, _ = train_with_early_stopping(
                model, train_loader, valid_loader, current_config, use_onecycle=True
            )

            # 베이지안 최적화는 최대화 문제를 해결하므로 손실의 음수를 반환
            # 또한 NaN 값 처리
            if np.isnan(best_val_loss):
                return -float('inf')  # 최악의 점수 반환

            print(f"  검증 손실: {best_val_loss:.6f} (에폭 {early_stop_epoch})")

            del model, train_loader, valid_loader, train_dataset, valid_dataset
            torch.cuda.empty_cache()

            return -best_val_loss  # 음수로 변환하여 최대화 문제로 변경

        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return -float('inf')  # 오류 발생 시 최악의 점수 반환

    # 최적화할 하이퍼파라미터 범위 정의
    pbounds = {
        'd_model_exp': (6, 9),         # 2^6=64 ~ 2^9=512
        'd_state_exp': (5, 9),         # 2^5=32 ~ 2^9=512
        'learning_rate_exp': (-6, -2) # 10^-6 ~ 10^-3
    }

    # 베이지안 최적화 객체 생성
    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # 탐색 시작
    print("베이지안 최적화 시작 (배치 사이즈 2048로 고정)...")
    optimizer.maximize(
        init_points=10,    # 초기 무작위 탐색 횟수
        n_iter=n_iterations,    # 베이지안 최적화 반복 횟수
    )

    # 최적 파라미터 변환 및 반환
    best_params = optimizer.max['params']

    # 최적 파라미터를 실제 값으로 변환
    optimized_config = RegimeMambaConfig()
    optimized_config.data_path = data_path
    optimized_config.d_model = int(2 ** best_params['d_model_exp'])
    optimized_config.d_state = int(2 ** best_params['d_state_exp'])
    optimized_config.n_layers = base_config.n_layers      # config에서 고정
    optimized_config.dropout = base_config.dropout         # config에서 고정
    optimized_config.learning_rate = 10 ** best_params['learning_rate_exp']
    optimized_config.batch_size = 2048  # 항상 2048로 고정
    optimized_config.seq_len = base_config.target_horizon  # config에서 고정

    print("\n최적화 완료!")
    print("최적 하이퍼파라미터:")
    print(f"  d_model: {optimized_config.d_model}")
    print(f"  d_state: {optimized_config.d_state}")
    print(f"  n_layers: {optimized_config.n_layers} (고정)")
    print(f"  dropout: {optimized_config.dropout:.3f} (고정)")
    print(f"  learning_rate: {optimized_config.learning_rate:.2e}")
    print(f"  batch_size: {optimized_config.batch_size} (고정)")
    print(f"  seq_len: {optimized_config.seq_len} (고정)")

    # 최적화 결과 저장
    if save_path is not None:
        optimization_results = {
            'best_hyperparameters': {
                'd_model': optimized_config.d_model,
                'd_state': optimized_config.d_state,
                'n_layers': optimized_config.n_layers,
                'dropout': optimized_config.dropout,
                'learning_rate': optimized_config.learning_rate,
                'batch_size': optimized_config.batch_size,
                'seq_len': optimized_config.seq_len
            },
            'best_val_loss': -optimizer.max['target'],
            'all_results': [
                {
                    'params': {
                        'd_model': int(2 ** res['params']['d_model_exp']),
                        'd_state': int(2 ** res['params']['d_state_exp']),
                        'n_layers': 2,      # 항상 2으로 고정
                        'dropout': 0.1,
                        'learning_rate': 10 ** res['params']['learning_rate_exp'],
                        'batch_size': 2048,  # 항상 2048로 고정
                        'seq_len': base_config.target_horizon  # config에서 고정
                    },
                    'val_loss': -res['target']
                }
                for res in optimizer.res
            ]
        }

        with open(save_path, 'w') as f:
            json.dump(optimization_results, f, indent=4)

    return optimized_config
