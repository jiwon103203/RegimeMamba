"""Jump Model 기반 시장 국면 파이프라인 (진입점: 저장소 루트의 ``run_pipeline.py``).

모듈 구성
    data_io          입력 로드·정제 → 초과수익률 / 벤치마크 상대수익률
    features         EWM downside deviation · Sortino 등 피처 세트와 사용자 변수 변환
    rolling          6개월 재추정 + 온라인 추론 (핵심)
    mamba_encoder    --model mamba: 재추정마다 Mamba 학습 → hidden 벡터 (GPU, torch·mamba-ssm 필요)
    sparse_pin       고정 피처를 지원하는 Sparse Jump Model
    backtest         0/1 전략 백테스트 · 성과표 · 거래 지연 로버스트니스
    hmm_benchmark    롤링 Gaussian HMM 벤치마크
    weights          sjm 피처 가중 비중
    regime_episodes  에피소드 · 유사 국면 · 종료 시나리오
    plotting         그림
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
