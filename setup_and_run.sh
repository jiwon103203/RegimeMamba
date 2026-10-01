#!/bin/bash
pip install "causal-conv1d==1.7.0" --no-build-isolation
pip install mamba-ssm --no-build-isolation --no-cache-dir
pip install bayesian-optimization --quiet
pip install jumpmodels --quiet
pip install opencv-python==4.5.5.64
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super
