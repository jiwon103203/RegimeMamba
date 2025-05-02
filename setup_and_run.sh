#!/bin/bash
pip install causal-conv1d>=1.4.0 --no-build-isolation --quiet
pip install mamba-ssm --no-build-isolation --no-cache-dir --quiet
pip install bayesian-optimization --quiet
pip install jumpmodels --quiet
pip install opencv-python==4.5.5.64
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super
# python rolling_Window_train_backtest_full.py --config regime_mamba/config/config_full_60.yaml --jump_model True