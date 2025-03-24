# RegimeMamba

[English](#english) | [한국어](#korean)

<a id="english"></a>

## 🌐 English

RegimeMamba is a market regime identification system based on the Mamba State Space Model (SSM). It effectively classifies financial market regimes (Bull/Bear) and evaluates trading strategies based on these classifications.

### Key Features

- Time series modeling using Mamba SSM architecture
- Automatic Bull/Bear market regime identification
- Clustering-based regime classification
- Trading strategy evaluation with transaction costs
- Hyperparameter tuning using Bayesian optimization
- Rolling window backtesting with model retraining
- Various regime smoothing techniques comparison

### Installation

```bash
git clone https://github.com/jiwon103203/RegimeMamba.git
cd RegimeMamba
pip install -e .
```

### Required Libraries

- PyTorch >= 1.8.0
- Mamba SSM >= 1.0.1
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- Matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- Bayesian Optimization >= 1.2.0
- SciPy >= 1.5.0
- tqdm >= 4.50.0
- umap-learn >= 0.5.2
- openTSNE >= 0.6.2

### Usage

#### Basic Usage

```bash
python main.py --data_path path/to/your/data.csv --output_dir ./outputs
```

#### Hyperparameter Optimization

```bash
python main.py --data_path path/to/your/data.csv --output_dir ./outputs --optimize
```

#### Rolling Window Backtesting

```bash
python rolling_window_backtest.py --data_path path/to/data.csv --model_path path/to/model.pth --results_dir ./results
```

#### Rolling Window with Model Retraining

```bash
python rolling_window_train_backtest.py --data_path path/to/data.csv --results_dir ./rolling_results
```

#### Comparing Smoothing Techniques

```bash
python filtering_train_strategies.py --data_path path/to/data.csv --results_dir ./smoothing_results
```

### Project Structure

```
Project directory/
 ├── RegimeMamba/
 │   └── regime_mamba/
 │       ├── config/                # Configuration related modules
 │       │   ├── __init__.py
 │       │   ├── config.py
 │       │   └── rl_config.py
 │       ├── data/                  # Dataset related modules
 │       │   ├── __init__.py
 │       │   ├── data_average_20_p.csv  # target : 20 days average returns(preprocessed)
 │       │   ├── data_average_20.csv    # target : 20 days average close
 │       │   ├── data_average_60_p.csv  # target : 60 days average returns(preprocessed)
 │       │   ├── data_average_60.csv    # target : 60 days average close
 │       │   ├── data_average_120_p.csv # target : 120 days average returns(preprocessed)
 │       │   ├── data_average_120.csv   # target : 120 days average close
 │       │   ├── data_average_200_p.csv # target : 200 days average returns(preprocessed)
 │       │   ├── data_average_200.csv   # target : 200 days average close
 │       │   ├── data.csv
 │       │   └── dataset.py
 │       ├── evaluate/              # Model evaluation modules
 │       │   ├── __init__.py
 │       │   ├── clustering.py      # Clustering functions
 │       │   ├── rl_evaluate.py
 │       │   ├── rolling_window.py  # Rolling window backtest
 │       │   ├── rolling_window_w_train.py  # Rolling window with retraining
 │       │   ├── smoothing.py       # Regime smoothing techniques
 │       │   └── strategy.py        # Strategy evaluation functions
 │       ├── models/                # Model definition modules
 │       │   ├── __init__.py
 │       │   ├── best_regime_mamba_6_average.pth
 │       │   ├── best_regime_mamba_6_cumulative.pth
 │       │   ├── best_regime_mamba_6.pth
 │       │   ├── mamba_model.py
 │       │   └── rl_model.py
 │       ├── train/                 # Model training modules
 │       │   ├── __init__.py
 │       │   ├── optimize.py        # Hyperparameter optimization
 │       │   ├── rl_train.py
 │       │   └── train.py           # Training functions
 │       ├── utils/                 # Utility functions
 │       │   ├── __init__.py
 │       │   ├── rl_agents.py
 │       │   ├── rl_environments.py
 │       │   ├── rl_investment.py
 │       │   ├── rl_visualize.py
 │       │   └── utils.py
 │       └── __init__.py
 ├── filtering_strategies.py
 ├── filtering_train_strategies.py
 ├── hidden_state_visualize.py
 ├── main.py
 ├── README.md
 ├── rolling_window_backtest.py
 ├── rolling_window_train_backtest.py
 ├── run_rl_investment.py
 └── setup.py
```

### Data Format

Input data should be a CSV file with the following columns:
- Date column: 'Date'
- Feature columns: 'returns', 'dd_10', 'sortino_20', 'sortino_60'

There are additional options for dependent variable(returns) (target_type & target_horizon)
- next_day: The return for the next day
- average: The average return over a specified period
- cumulative: The cumulative return over a specified period
- trend_strength: Trend strength measured using linear regression
- direction: Direction over the period (converted into a classification problem)
- volatility_adjusted: Volatility-adjusted return (similar to the Sharpe ratio)
- up_ratio: The ratio of days with positive returns during the period
- log_return_sum: The sum of log returns

### Supported Evaluation Methods

#### Rolling Window Backtesting
Tests regime identification on consecutive time windows using a pre-trained model.

#### Rolling Window with Retraining
Retrains the model periodically based on a rolling window of historical data:
- Uses 40 years of data for each window
- 20 years for training
- 10 years for validation
- 10 years for regime clustering
- Applies identified regimes to the next 5 years

#### Smoothing Techniques
Various filtering methods to reduce noise in regime signals:
- Moving Average (MA)
- Exponential Smoothing (EMA)
- Gaussian Filter
- Confirmation Rule
- Minimum Holding Period

### License

MIT

---

<a id="korean"></a>

## 🇰🇷 한국어

RegimeMamba는 Mamba 상태 공간 모델(SSM)을 기반으로 한 시장 레짐 식별 시스템입니다. 이 프로젝트는 금융 시장 레짐(강세장/약세장)을 효과적으로 분류하고 이를 기반으로 트레이딩 전략을 평가합니다.

### 주요 기능

- Mamba SSM 아키텍처를 활용한 시계열 모델링
- Bull/Bear 시장 레짐 자동 식별
- 클러스터링 기반 레짐 분류
- 거래 비용을 고려한 전략 성과 평가
- 베이지안 최적화를 이용한 하이퍼파라미터 튜닝
- 모델 재학습을 포함한 롤링 윈도우 백테스팅
- 다양한 레짐 평활화 기법 비교

### 설치 방법

```bash
git clone https://github.com/jiwon103203/RegimeMamba.git
cd RegimeMamba
pip install -e .
```

### 필요 라이브러리

- PyTorch >= 1.8.0
- Mamba SSM >= 1.0.1
- NumPy >= 1.19.0
- Pandas >= 1.2.0
- Matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- Bayesian Optimization >= 1.2.0
- SciPy >= 1.5.0
- tqdm >= 4.50.0
- umap-learn >= 0.5.2
- openTSNE >= 0.6.2

### 사용 방법

#### 기본 사용법

```bash
python main.py --data_path path/to/your/data.csv --output_dir ./outputs
```

#### 하이퍼파라미터 최적화

```bash
python main.py --data_path path/to/your/data.csv --output_dir ./outputs --optimize
```

#### 롤링 윈도우 백테스팅

```bash
python rolling_window_backtest.py --data_path path/to/data.csv --model_path path/to/model.pth --results_dir ./results
```

#### 모델 재학습이 포함된 롤링 윈도우

```bash
python rolling_window_train_backtest.py --data_path path/to/data.csv --results_dir ./rolling_results
```

#### 평활화 기법 비교

```bash
python filtering_train_strategies.py --data_path path/to/data.csv --results_dir ./smoothing_results
```

### 프로젝트 구조

```
Project directory/
 ├── RegimeMamba/
 │   └── regime_mamba/
 │       ├── config/                # Configuration related modules
 │       │   ├── __init__.py
 │       │   ├── config.py
 │       │   └── rl_config.py
 │       ├── data/                  # Dataset related modules
 │       │   ├── __init__.py
 │       │   ├── data_average_20_p.csv  # target : 20 days average returns(preprocessed)
 │       │   ├── data_average_20.csv    # target : 20 days average close
 │       │   ├── data_average_60_p.csv  # target : 60 days average returns(preprocessed)
 │       │   ├── data_average_60.csv    # target : 60 days average close
 │       │   ├── data_average_120_p.csv # target : 120 days average returns(preprocessed)
 │       │   ├── data_average_120.csv   # target : 120 days average close
 │       │   ├── data_average_200_p.csv # target : 200 days average returns(preprocessed)
 │       │   ├── data_average_200.csv   # target : 200 days average close
 │       │   ├── data.csv
 │       │   └── dataset.py
 │       ├── evaluate/              # Model evaluation modules
 │       │   ├── __init__.py
 │       │   ├── clustering.py      # Clustering functions
 │       │   ├── rl_evaluate.py
 │       │   ├── rolling_window.py  # Rolling window backtest
 │       │   ├── rolling_window_w_train.py  # Rolling window with retraining
 │       │   ├── smoothing.py       # Regime smoothing techniques
 │       │   └── strategy.py        # Strategy evaluation functions
 │       ├── models/                # Model definition modules
 │       │   ├── __init__.py
 │       │   ├── best_regime_mamba_6_average.pth
 │       │   ├── best_regime_mamba_6_cumulative.pth
 │       │   ├── best_regime_mamba_6.pth
 │       │   ├── mamba_model.py
 │       │   └── rl_model.py
 │       ├── train/                 # Model training modules
 │       │   ├── __init__.py
 │       │   ├── optimize.py        # Hyperparameter optimization
 │       │   ├── rl_train.py
 │       │   └── train.py           # Training functions
 │       ├── utils/                 # Utility functions
 │       │   ├── __init__.py
 │       │   ├── rl_agents.py
 │       │   ├── rl_environments.py
 │       │   ├── rl_investment.py
 │       │   ├── rl_visualize.py
 │       │   └── utils.py
 │       └── __init__.py
 ├── filtering_strategies.py
 ├── filtering_train_strategies.py
 ├── hidden_state_visualize.py
 ├── main.py
 ├── README.md
 ├── rolling_window_backtest.py
 ├── rolling_window_train_backtest.py
 ├── run_rl_investment.py
 └── setup.py
```

### 데이터 형식

입력 데이터는 다음 형식의 CSV 파일이어야 합니다:
- 날짜 열: 'Date'
- 특성 열: 'returns', 'dd_10', 'sortino_20', 'sortino_60'

종속 변수(수익률)에 대한 추가 옵션이 있습니다 (target_type 및 target_horizon)
- next_day: 다음 날의 수익률
- average: 지정된 기간 동안의 평균 수익률
- cumulative: 지정된 기간 동안의 누적 수익률
- trend_strength: 선형 회귀로 측정한 추세 강도
- direction: 기간 동안의 방향성 (분류 문제로 변환)
- volatility_adjusted: 변동성 조정 수익률 (샤프 비율과 유사)
- up_ratio: 기간 중 상승한 날의 비율
- log_return_sum: 로그 수익률의 합계

### 지원하는 평가 방법

#### 롤링 윈도우 백테스팅
사전 학습된 모델을 사용하여 연속적인 시간 윈도우에서 레짐 식별을 테스트합니다.

#### 모델 재학습이 포함된 롤링 윈도우
과거 데이터의 롤링 윈도우를 기반으로 주기적으로 모델을 재학습합니다:
- 각 윈도우마다 40년 데이터 사용
- 20년은 학습에 사용
- 10년은 검증에 사용
- 10년은 레짐 클러스터링에 사용
- 식별된 레짐을 다음 5년에 적용

#### 평활화 기법
레짐 신호의 노이즈를 줄이기 위한 다양한 필터링 방법:
- 이동 평균(MA)
- 지수 평활화(EMA)
- 가우시안 필터
- 확인 규칙
- 최소 보유 기간

### 라이센스

MIT
