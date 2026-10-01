# Regime Mamba

Mamba(선택적 상태공간 모델)로 시계열 피처를 압축하고 Jump Model로 bull/bear 국면을 나누는 금융 국면 탐지 모델.

![Regime Mamba Architecture](./architecture.png)

## 구조

```
run_pipeline.py          진입점 (CLI · run_pipeline() · run_inference())
regime_jm/
  data_io.py             1) 입력 로드·정제 → 초과수익률 (+ 벤치마크 상대수익률 rel_ret)
  features.py            2) 피처 세트 (paper / example / extra) + 사용자 변수 변환
  mamba_encoder.py          (--encoder mamba) 피처 시퀀스 → Mamba hidden 벡터
  rolling.py             3) 1·7월 재추정 + 사이 구간 온라인 추론 (Jump Model)
  sparse_pin.py             Sparse Jump Model (피처 고정 지원)
  backtest.py            4) 0/1 전략 백테스트 · 성과 · 거래 지연 로버스트니스
  hmm_benchmark.py       5) Gaussian HMM 벤치마크 (--hmm)
  weights.py             6) sjm 피처 가중 비중
  regime_episodes.py        bear 에피소드 · 유사 국면 · 종료 시나리오
  plotting.py            7) 그림
regime_mamba/
  models/mamba_model.py  Mamba 백본 (TimeSeriesMamba, --encoder mamba 가 사용)
  models/e2e_regime_mamba.py, train/e2e_train.py, config/e2e_config.py
                         End-to-End Regime Mamba (Mamba + 미분 가능한 jump penalty 로 국면을 직접 학습)
  data/, evaluate/       E2E 용 데이터셋 · 롤링 윈도우 · 스무딩 비교
scripts/e2e_backtest.py  E2E 롤링 백테스트 진입점
tests/                   인과성·파이프라인 테스트
```

## 실행

```bash
python run_pipeline.py data.csv                                        # 피처 → Jump Model
python run_pipeline.py data.csv --encoder mamba --device cuda:0        # 피처 → Mamba → Jump Model
python run_pipeline.py data.csv --encoder mamba --model sjm --feature-set extra --hmm
python run_pipeline.py sector.csv --relative-benchmark kospi.csv --backtest-ret relative
python run_pipeline.py data.csv --extra-features macro.csv:VIX:log macro.csv:USDKRW:logdiff
python run_pipeline.py data.csv --inference                            # 현재 반기만 추론
```

`--encoder`는 Jump Model 앞에 붙는 단계일 뿐이라, 아래 데이터·피처·모델·재추정·백테스트 옵션은 `--encoder none`과 `mamba`에서 똑같이 동작합니다.

## 옵션

| 단계 | 옵션 (기본값) |
|---|---|
| 데이터 | `--date-col/--close-col/--rf-col` (자동 인식), `--rf-unit annual_pct`, `--rf-const`, `--relative-benchmark`, `--signal-ret auto`, `--start/--end` |
| 피처 | `--feature-set paper\|example\|extra\|none`, `--extra-features 파일:열:변환`, `--remove-series`, `--warmup 252` |
| 인코더 | `--encoder none\|mamba`, `--device auto\|cuda\|cuda:N`, `--mamba-seq-len 60`, `--mamba-d-model 8`, `--mamba-d-state 32`, `--mamba-d-conv 4`, `--mamba-expand 2`, `--mamba-layers 4`, `--mamba-dropout 0.1`, `--mamba-epochs 100`, `--mamba-patience 10`, `--mamba-batch-size 1024`, `--mamba-lr 5e-4`, `--mamba-valid-frac 0.2` |
| 모델 | `--model jm\|sjm`, `--discrete`, `--n-states 2`, `--jump-penalty 50`, `--max-feats`, `--pin-features`, `--n-init 10`, `--clip-mul 3` |
| 재추정 | `--train-window 3000`, `--min-train 500`, `--refit-months 1,7`, `--oos-start` |
| 백테스트 | `--delay 1`, `--delays 1,2,3,5,10`, `--min-cash 0`, `--max-cash 1`, `--cost-bps 10`, `--backtest-ret absolute\|relative` |
| HMM | `--hmm`, `--hmm-states 2`, `--hmm-refit 21`, `--hmm-median 5` |

- 피처 세트: `paper` = DD-log_10, sortino_20, sortino_60 / `example` = ret·DD-log·sortino × 5·20·60일 (9개) / `extra` = 9개 계열 × 5·20·60일 (27개).
- 사용자 변수 변환: `none, log, diff_n, pct_n, logdiff_n, zscore_w, ewm_hl, lag_n` (`+`로 연결).
- 상태 0 = bull, 마지막 상태 = bear. 0/1 전략은 bear일 때 현금(`relative`면 벤치마크)을 보유합니다.
- `--encoder mamba`: 재추정마다 학습창으로 Mamba를 새로 학습하고(시퀀스 → 다음 날 수익률, MSE, 뒤쪽 `--mamba-valid-frac`로 early stopping), 각 날짜의 마지막 hidden 벡터 `h_0..`를 Jump Model 입력으로 씁니다. mamba-ssm은 CUDA 전용이라 GPU가 없으면 시작 단계에서 에러로 끝납니다. `--pin-features`는 쓸 수 없습니다.

## 출력 (`--out`, 기본 `out/`)

| 파일 | 내용 |
|---|---|
| `regimes.csv` | 날짜별 국면·상태 확률·재추정일 |
| `refit_params.csv` | 재추정 × 상태별 중심점, 연율 수익률·변동성, stay_prob |
| `strategy.csv`, `performance.csv`, `delay_robustness.csv` | 전략 일별 내역, 성과, 지연별 성과 |
| `feat_weights.csv`, `weight_groups*.csv` | (sjm) 피처 가중치와 그룹별 비중 |
| `regime_episodes.csv`, `similar_episodes.csv`, `length_scenarios.csv` | 에피소드, 유사 국면, 종료 시나리오 |
| `hmm_regimes.csv`, `model_comparison.csv` | (--hmm) HMM 비교 |
| `mamba_train_log.csv` | (--encoder mamba) 재추정별 epoch, train/valid loss |
| `current_state.json`, `run_config.json`, `*.png` | 현재 국면, 실행 설정, 그림 |

`--inference`는 최근 반기 시작점 직전 학습창으로 한 번만 학습하고 현재 반기만 추론합니다 (`inference_*.csv`).

## End-to-End Regime Mamba (`scripts/e2e_backtest.py`)

Jump Model 없이 Mamba가 국면 확률을 직접 출력하도록 학습하는 실험용 모델입니다 (Gumbel-Softmax, jump penalty · 분리 · 2단계 엔트로피 손실). `[학습 train_years][검증 valid_years]` 창을 `forward_months`씩 옮기며 학습하고, 다음 구간에서 스무딩 방법별 전략 성과를 비교합니다.

```bash
python scripts/e2e_backtest.py --data_path data.csv --feature_set paper --e2e_preset balanced   # CSV: Date, returns, target_returns_1
```

## 인과성 규칙 (`tests/`에서 검증)

- 클리핑·표준화·Mamba 학습은 재추정일 이전 학습창으로만 합니다.
- t일의 국면(과 hidden 벡터)은 t일까지의 데이터만 씁니다.
- 거래는 신호보다 `--delay`일 늦게 체결됩니다.

## License

CC BY-SA 4.0. 데이터: S&P 500·T-bill·Dollar Index·개별 종목(Yahoo Finance), KOSPI·CD-91(한국은행 ECOS, KOGL Type 1).
