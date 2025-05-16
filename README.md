# Regime Mamba: Regime Switch Detection in Financial Time Series

## 📝 Introduction

Regime Mamba is a novel hybrid deep learning architecture that combines the selective state space model (Mamba) with traditional Jump Models to identify financial market regimes. This project implements the approach described in the paper "Regime Mamba: Regime Switch Detection in Financial Time Series via Mamba–Jump Hybrid Deep Model."

### Key Features

- **Hybrid Architecture**: Combines Mamba SSM for complex temporal pattern recognition with traditional Jump Models for regime persistence
- **Superior Performance**: Achieves 5.5% annualized return with 11.6% volatility (compared to 5.0% return for state-of-the-art models and 18.9% volatility for buy-and-hold)
- **Enhanced Risk Management**: Demonstrates maximum drawdown of only -31.8% versus -65.2% for buy-and-hold
- **Cross-Market Applicability**: Works effectively across both developed (S&P 500) and emerging markets (KOSPI)
- **Strategy Versatility**: Supports Long/Cash Only and Long/Short trading strategies
- **Global Macro Integration**: Incorporates the Dollar Index as a global macro indicator, improving Sharpe ratios by 98.1% and reducing false signals by 6%

## 🧠 Theoretical Framework

Regime Mamba bridges the gap between neural architectures and economic regime theory through:

1. **Representation Learning vs. Statistical Clustering**: Mamba leverages deep representation learning for complex non-linear patterns while Jump Models employ statistical clustering with explicit penalty terms

2. **Adaptive Feature Extraction vs. Fixed Feature Analysis**: Mamba dynamically extracts hierarchical features while Jump Models operate on predefined features (Downside Deviation and Sortino Ratio)

3. **Parameter Update Mechanisms**: Mamba's Δt parameters are learned through backpropagation while Jump Models use explicit hyperparameter λ requiring manual calibration

The architecture allows for effectively capturing regime transitions in financial markets which traditional statistical approaches struggle to identify.

## 🔧 Installation

```bash
git clone https://github.com/yourusername/RegimeMamba.git
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
- causal-conv1d >= 1.4.0

## 📊 Dataset Format

The model expects input data as a CSV file with the following columns:
- Date column: 'Date'
- Feature columns: 
  - 'returns': Daily market returns
  - 'dd_10': Downside Deviation with 10-day lookback
  - 'sortino_20': Sortino Ratio with 20-day lookback 
  - 'sortino_60': Sortino Ratio with 60-day lookback
  - 'dollar_index' (optional but recommended): Dollar Index as a global macro indicator

## 💻 Usage

### Basic Usage

```bash
python main.py --data_path path/to/your/data.csv --output_dir ./outputs
```

### Hyperparameter Optimization

```bash
python main.py --data_path path/to/your/data.csv --output_dir ./outputs --optimize
```

### Rolling Window Backtesting

```bash
python rolling_window_backtest.py --data_path path/to/data.csv --model_path path/to/model.pth --results_dir ./results
```

### Rolling Window with Model Retraining

```bash
python rolling_window_train_backtest.py --data_path path/to/data.csv --results_dir ./rolling_results
```

### Comparing Smoothing Techniques

```bash
python filtering_train_strategies.py --data_path path/to/data.csv --results_dir ./smoothing_results
```

## 🧪 Evaluation Methods

### Rolling Window Backtesting
Tests regime identification on consecutive time windows using a pre-trained model.

### Rolling Window with Retraining
Retrains the model periodically based on a rolling window of historical data:
- Uses 40 years of data for each window
- 20 years for training
- 10 years for validation
- 10 years for regime clustering
- Applies identified regimes to the next 5 years

### Smoothing Techniques
Various filtering methods to reduce noise in regime signals:
- Moving Average (MA)
- Exponential Smoothing (EMA)
- Gaussian Filter
- Confirmation Rule
- Minimum Holding Period

## 📈 Results

The Regime Mamba model demonstrates superior performance across multiple markets:

| Market  | Strategy      | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown |
|---------|---------------|-------------|-----------------|--------------|--------------|
| S&P 500 | Regime Mamba  | 5.5%        | 11.6%           | 0.323        | -31.8%       |
| S&P 500 | Buy-and-Hold  | 6.1%        | 18.9%           | 0.287        | -65.2%       |
| S&P 500 | State-of-Art  | 5.0%        | 11.1%           | 0.293        | -25.5%       |
| KOSPI   | Regime Mamba  | 2.5%        | 13.9%           | 0.157        | -45.4%       |
| KOSPI   | Buy-and-Hold  | 3.1%        | 19.8%           | 0.180        | -63.6%       |

The model particularly excels in risk management and volatility reduction, offering a more stable investment approach with significantly improved drawdown protection.

## 🚀 Advanced Features

### Jump Model Integration
Integrate with traditional Jump Models for enhanced regime persistence:

```bash
python rolling_window_train_backtest.py --data_path path/to/data.csv --jump_model True --jump_penalty 32
```

### Dollar Index Ablation
The Dollar Index serves as a critical component, without which:
- Sharpe ratio falls by 49.5% (from 0.323 to 0.163)
- Annualized return drops from 5.5% to 3.4%

### LSTM Variant
Try the LSTM-based model variant instead of Mamba:

```bash
python rolling_window_train_backtest.py --data_path path/to/data.csv --lstm
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{anonymous2025regimemamba,
  title={Regime Mamba: Regime Switch Detection in Financial Time Series via Mamba–Jump Hybrid Deep Model},
  author={Anonymous},
  booktitle={Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025}
}
```

## 📄 License

CC-BY-SA (Creative Commons Attribution-ShareAlike)
- You are free to share and adapt the material
- You must give appropriate credit and indicate if changes were made
- You must distribute your contributions under the same license as the original

## 🙏 Acknowledgements

We acknowledge all data sources according to their respective licensing terms:
- S&P 500 index, Treasury bills, Dollar Index, and individual stock data from Yahoo Finance
- KOSPI and CD-91 data from the Bank of Korea's Economic Statistics System (ECOS) under the Korea Open Government License (KOGL Type 1)