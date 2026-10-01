"""--model mamba 테스트 (run_pipeline.py).

mamba-ssm 은 CUDA 전용이므로 학습·인과성 테스트는 같은 인터페이스의 작은 GRU 백본으로 CPU 에서 돌린다.

    pytest tests/test_mamba_pipeline.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_pipeline  # noqa: E402
from regime_jm import mamba_encoder  # noqa: E402
from regime_jm.config import PipelineConfig  # noqa: E402
from regime_jm.features import build_features  # noqa: E402
from regime_jm.mamba_encoder import MambaEncoder, MambaSettings, _windows, resolve_device  # noqa: E402

torch = pytest.importorskip("torch")

SMALL = MambaSettings(seq_len=10, d_model=4, epochs=3, patience=2, batch_size=128, seed=0)


class GRUBackbone(torch.nn.Module):
    """TimeSeriesMamba 와 같은 forward(x, return_hidden) 인터페이스."""

    def __init__(self, n_features, s):
        super().__init__()
        self.rnn = torch.nn.GRU(n_features, s.d_model, batch_first=True)
        self.head = torch.nn.Linear(s.d_model, 1)

    def forward(self, x, return_hidden=False):
        out, _ = self.rnn(x)
        h = out[:, -1, :]
        pred = self.head(h)
        return (pred, h) if return_hidden else pred


def simulate(n=1400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.995 if state[t - 1] == 0 else 0.98
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    ret = rng.normal(np.where(state == 0, 0.0006, -0.0015), np.where(state == 0, 0.008, 0.022))
    return pd.Series(ret, index=idx)


@pytest.fixture
def cpu_mamba(monkeypatch):
    """GPU 검사를 건너뛰고 GRU 백본을 쓰도록 바꾼다."""
    monkeypatch.setattr(mamba_encoder, "resolve_device", lambda spec="auto": "cpu")
    monkeypatch.setattr(mamba_encoder, "build_mamba_backbone", GRUBackbone)


def test_windows_end_at_row_and_zero_pad():
    Z = np.arange(10, dtype=np.float32).reshape(5, 2)
    w = _windows(Z, np.array([0, 4]), 3)
    assert w.shape == (2, 3, 2)
    np.testing.assert_array_equal(w[0], [[0, 0], [0, 0], [0, 1]])
    np.testing.assert_array_equal(w[1], Z[2:5])


def test_resolve_device_requires_gpu(monkeypatch):
    with pytest.raises(ValueError, match="CUDA 전용"):
        resolve_device("cpu")
    with pytest.raises(ValueError, match="auto, cuda, cuda:N"):
        resolve_device("gpu0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="GPU"):
        resolve_device("auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="cuda:3"):
        resolve_device("cuda:3")


def test_encoder_is_causal():
    """t 행의 hidden 은 t 이후 피처가 바뀌어도 그대로여야 한다 (학습은 pos 이전만)."""
    X = build_features(simulate(), "paper")
    y = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(X)), index=X.index)
    lo, pos, seg_end = 100, 500, 620
    H = MambaEncoder(SMALL, "cpu", GRUBackbone)(X, y, lo, pos, seg_end)
    assert H.shape == (seg_end - lo, SMALL.d_model) and list(H.columns) == [f"h_{k}" for k in range(4)]
    assert H.index.equals(X.index[lo:seg_end])

    cut = 560
    X2, y2 = X.copy(), y.copy()
    X2.iloc[cut:] = X2.iloc[cut:] * 5 + 3
    y2.iloc[pos:] = 0.5
    enc2 = MambaEncoder(SMALL, "cpu", GRUBackbone)
    H2 = enc2(X2, y2, lo, pos, seg_end)
    np.testing.assert_allclose(H2.iloc[:cut - lo], H.iloc[:cut - lo], atol=1e-6)
    assert not np.allclose(H2.iloc[cut - lo:], H.iloc[cut - lo:])
    log = enc2.history_frame()
    assert list(log["refit_date"]) == [X.index[pos]] and log["epochs"].iloc[0] >= 1


def test_config_and_cli_options():
    args = run_pipeline.build_parser().parse_args(
        ["d.csv", "--model", "mamba", "--device", "cuda:1", "--seq-len", "30", "--d-model", "16", "--epochs", "5"])
    cfg = run_pipeline.config_from_args(args).validate()
    assert (cfg.model, cfg.device, cfg.seq_len, cfg.d_model, cfg.epochs) == ("mamba", "cuda:1", 30, 16, 5)
    s = cfg.mamba_settings()
    assert (s.seq_len, s.d_model, s.clip_mul, s.seed) == (30, 16, cfg.clip_mul, cfg.seed)
    with pytest.raises(ValueError, match="device"):
        PipelineConfig(input="d.csv", model="mamba", device="tpu").validate()
    with pytest.raises(ValueError, match="valid_frac"):
        PipelineConfig(input="d.csv", model="mamba", valid_frac=1.0).validate()
    with pytest.raises(ValueError, match="sjm"):
        PipelineConfig(input="d.csv", model="mamba", pin_features=["sortino_20"]).validate()


@pytest.fixture
def asset_csv(tmp_path):
    ret = simulate()
    close = 1000 * (1 + ret).cumprod()
    path = tmp_path / "asset.csv"
    pd.DataFrame({"date": ret.index, "close": close.to_numpy(), "rf": 3.0}).to_csv(path, index=False)
    return str(path)


MAMBA_ARGS = ["--model", "mamba", "--discrete", "--train-window", "400", "--min-train", "250", "--n-init", "2",
              "--seq-len", "10", "--d-model", "4", "--epochs", "2", "--patience", "1", "--no-plots", "-q"]


def test_cli_mamba_pipeline(cpu_mamba, asset_csv, tmp_path):
    out = tmp_path / "out"
    assert run_pipeline.main([asset_csv, "--out", str(out), *MAMBA_ARGS]) == 0
    reg = pd.read_csv(out / "regimes.csv", index_col=0)
    params = pd.read_csv(out / "refit_params.csv")
    log = pd.read_csv(out / "mamba_train_log.csv")
    assert set(reg["regime"]) <= {0, 1}
    assert {"center_h_0", "center_h_3"} <= set(params.columns)
    assert len(log) == params["refit_date"].nunique()
    assert (out / "performance.csv").exists() and (out / "current_state.json").exists()


def test_cli_mamba_inference(cpu_mamba, asset_csv, tmp_path):
    out = tmp_path / "inf"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--inference", *MAMBA_ARGS]) == 0
    assert len(pd.read_csv(out / "inference_mamba_train_log.csv")) == 1
    summary = pd.read_csv(out / "inference_summary.csv")
    assert summary["model"].iloc[0] == "mamba (discrete)"


def test_cli_mamba_without_gpu_fails_cleanly(monkeypatch, asset_csv, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert run_pipeline.main([asset_csv, "--out", str(tmp_path), "--model", "mamba", "-q"]) == 1
