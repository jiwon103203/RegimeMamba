"""파이프라인 설정. ``run_pipeline.main()`` 이 CLI 인자를 이 dataclass 로 옮긴다."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PipelineConfig:
    # 입력
    input: str = ""
    out_dir: str = "out"
    date_col: Optional[str] = None
    close_col: Optional[str] = None
    rf_col: Optional[str] = None
    rf_unit: str = "annual_pct"            # annual_pct | annual | daily_pct | daily
    rf_const: Optional[float] = None       # 파일 대신 쓸 상수 무위험금리 (rf_unit 단위)
    relative_benchmark: Optional[str] = None
    bench_date_col: Optional[str] = None
    bench_close_col: Optional[str] = None
    start: Optional[str] = None            # 데이터 사용 시작일
    end: Optional[str] = None              # 데이터 사용 종료일
    signal_ret: str = "auto"               # auto | absolute | relative (모델이 학습할 수익률)
    backtest_ret: str = "absolute"         # absolute | relative (백테스트 평가 기준)

    # 피처
    feature_set: str = "paper"             # paper | example | extra | none
    extra_features: List[str] = field(default_factory=list)   # 파일:열:변환
    remove_series: List[str] = field(default_factory=list)
    warmup: int = 252

    # 모델
    model: str = "jm"                      # jm | sjm | mamba (Mamba hidden 벡터 → Jump Model, GPU 필요)
    cont: bool = True                      # True: 연속형 (CJM / 연속형 SJM)
    n_states: int = 2
    jump_penalty: float = 50.0
    max_feats: Optional[float] = None      # sjm 유효 피처 수 (기본: 피처 수의 1/3, 최소 2)
    pin_features: List[str] = field(default_factory=list)     # sjm 에서 꼭 남길 피처/계열
    grid_size: float = 0.05
    n_init: int = 10
    clip_mul: float = 3.0
    seed: int = 0

    # Mamba (--model mamba)
    device: str = "auto"                   # auto | cuda | cuda:N (mamba-ssm 은 CUDA 전용)
    seq_len: int = 60
    d_model: int = 8                       # hidden 벡터 차원 = Jump Model 입력 차원
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 4
    dropout: float = 0.1
    epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    lr: float = 5e-4
    valid_frac: float = 0.2

    # 롤링 재추정
    train_window: int = 3000
    min_train: int = 500
    refit_months: Tuple[int, ...] = (1, 7)
    oos_start: Optional[str] = None        # 이 날짜 이후 재추정부터 백테스트

    # 백테스트
    delay: int = 1
    delays: Tuple[int, ...] = (1, 2, 3, 5, 10)
    min_cash: float = 0.0
    max_cash: float = 1.0
    cost_bps: float = 10.0
    cost_buy_bps: Optional[float] = None
    cost_sell_bps: Optional[float] = None

    # HMM 벤치마크
    hmm: bool = False
    hmm_states: int = 2
    hmm_refit: int = 21
    hmm_median: int = 5
    hmm_n_init: int = 3

    # 해석
    top_k: int = 5

    # 실행 모드
    inference: bool = False
    plots: bool = True

    def mamba_settings(self):
        from .mamba_encoder import MambaSettings
        return MambaSettings(seq_len=self.seq_len, d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv,
                             expand=self.expand, n_layers=self.n_layers, dropout=self.dropout, epochs=self.epochs,
                             patience=self.patience, batch_size=self.batch_size, lr=self.lr,
                             valid_frac=self.valid_frac, clip_mul=self.clip_mul, seed=self.seed)

    @property
    def cost_buy(self) -> float:
        return (self.cost_bps if self.cost_buy_bps is None else self.cost_buy_bps) / 1e4

    @property
    def cost_sell(self) -> float:
        return (self.cost_bps if self.cost_sell_bps is None else self.cost_sell_bps) / 1e4

    def validate(self) -> "PipelineConfig":
        if not self.input:
            raise ValueError("입력 파일(input)이 필요합니다")
        if self.backtest_ret not in ("absolute", "relative"):
            raise ValueError("backtest_ret must be absolute or relative")
        if self.backtest_ret == "relative" and not self.relative_benchmark:
            raise ValueError("--backtest-ret relative 에는 --relative-benchmark 가 필요합니다")
        if self.model not in ("jm", "sjm", "mamba"):
            raise ValueError("model must be jm, sjm or mamba")
        if self.pin_features and self.model != "sjm":
            raise ValueError("--pin-features 는 --model sjm 에서만 쓸 수 있습니다")
        if not 0.0 <= self.min_cash <= self.max_cash <= 1.0:
            raise ValueError("0 <= min_cash <= max_cash <= 1 이어야 합니다")
        if self.n_states < 2:
            raise ValueError("n_states >= 2")
        if self.min_train < 2 or self.train_window < self.min_train:
            raise ValueError("train_window >= min_train >= 2 이어야 합니다")
        if self.model == "mamba":
            from .mamba_encoder import DEVICE_RE
            if not DEVICE_RE.match(str(self.device).strip().lower()):
                raise ValueError(f"--device 는 auto, cuda, cuda:N 중 하나여야 합니다 (got {self.device!r})")
            if min(self.seq_len, self.d_model, self.d_state, self.n_layers, self.epochs, self.patience,
                   self.batch_size) < 1:
                raise ValueError("seq_len, d_model, d_state, n_layers, epochs, patience, batch_size 는 1 이상이어야 합니다")
            if not 0.0 < self.valid_frac < 1.0:
                raise ValueError("valid_frac 은 0 과 1 사이여야 합니다")
        if self.delay < 0 or any(d < 0 for d in self.delays):
            raise ValueError("delay 는 0 이상이어야 합니다")
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost_buy"], d["cost_sell"] = self.cost_buy, self.cost_sell
        return d
