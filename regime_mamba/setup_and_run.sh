#!/bin/bash
pip install causal-conv1d>=1.4.0 --no-build-isolation --quiet
pip install mamba-ssm --quiet
pip install bayesian-optimization --quiet
pip install jumpmodels --quiet
apt-get update
apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super
python rolling_Window_train_backtest_full.py --config /content/RegimeMamba/regime_mamba/config/config_full_10.yaml --jump_model True