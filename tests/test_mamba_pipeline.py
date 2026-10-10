"""--encoder mamba 테스트 (run_pipeline.py).

mamba-ssm 은 CUDA 전용이므로 학습·인과성 테스트는 같은 인터페이스의 작은 GRU 백본으로 CPU 에서 돌린다.

    pytest tests/test_mamba_pipeline.py
"""

import dataclasses
import json
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
from regime_jm.mamba_encoder import MambaEncoder, _windows, forward_targets, resolve_device  # noqa: E402

torch = pytest.importorskip("torch")

SMALL = PipelineConfig(input="d.csv", encoder="mamba", mamba_seq_len=10, mamba_d_model=4, mamba_epochs=3,
                       mamba_patience=2, mamba_batch_size=128)


class GRUBackbone(torch.nn.Module):
    """TimeSeriesMamba 와 같은 forward(x, return_hidden) 인터페이스."""

    def __init__(self, n_features, cfg):
        super().__init__()
        self.rnn = torch.nn.GRU(n_features, cfg.mamba_d_model, batch_first=True)
        self.head = torch.nn.Linear(cfg.mamba_d_model, mamba_encoder.n_outputs(cfg))

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
    assert H.shape == (seg_end - lo, SMALL.mamba_d_model) and list(H.columns) == [f"h_{k}" for k in range(4)]
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


def test_build_features_warmup_context():
    """context 는 warmup 으로 버린 행 중 X 첫 행 직전까지 결측 없이 이어지는 부분이다."""
    ret = simulate()
    X = build_features(ret, "extra")
    X2, ctx = build_features(ret, "extra", return_context=True)
    pd.testing.assert_frame_equal(X, X2)
    assert 0 < len(ctx) <= 252 and list(ctx.columns) == list(X.columns)
    assert ctx.notna().all().all() and ctx.index[-1] < X.index[0]
    assert ret.index.get_loc(ctx.index[-1]) + 1 == ret.index.get_loc(X.index[0])
    assert len(build_features(ret, "paper", warmup=0, return_context=True)[1]) == 0


def test_encoder_warmup_context_replaces_zero_pad():
    """context 를 주면 X 시작 전 시퀀스 앞부분이 0 대신 그 행들로 채워진다 (X 앞에 붙여 넣은 것과 같음)."""
    X, ctx = build_features(simulate(), "paper", return_context=True)
    y = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(X)), index=X.index)
    L, pos, seg_end = SMALL.mamba_seq_len, 400, 450

    plain = MambaEncoder(SMALL, "cpu", GRUBackbone)
    H0 = plain(X, y, 0, pos, seg_end)
    with_ctx = MambaEncoder(SMALL, "cpu", GRUBackbone, context=ctx)
    H1 = with_ctx(X, y, 0, pos, seg_end)
    assert plain.history[0]["n_zero_pad"] == L - 1 and plain.history[0]["n_context_rows"] == 0
    assert with_ctx.history[0]["n_zero_pad"] == 0 and with_ctx.history[0]["n_context_rows"] == L - 1
    assert H1.index.equals(H0.index) and not np.allclose(H0.iloc[:L - 1], H1.iloc[:L - 1])

    k = L - 1  # context 를 X 앞에 직접 붙이고 lo 를 k 로 옮기면 같은 입력 · 같은 정규화 · 같은 시드
    full = pd.concat([ctx.iloc[-k:], X])
    y_full = pd.Series(0.0, index=full.index)
    y_full.iloc[k:] = y.to_numpy()
    H2 = MambaEncoder(SMALL, "cpu", GRUBackbone)(full, y_full, k, pos + k, seg_end + k)
    np.testing.assert_allclose(H1.to_numpy(), H2.to_numpy(), atol=1e-6)

    # 학습창이 X 시작에서 seq_len 이상 떨어져 있으면 context 는 쓰이지 않는다
    far = MambaEncoder(SMALL, "cpu", GRUBackbone, context=ctx)
    far(X, y, 50, pos, seg_end)
    assert far.history[0]["n_context_rows"] == 0 and far.history[0]["n_zero_pad"] == 0


def test_warmup_context_requires_mamba():
    with pytest.raises(ValueError, match="mamba-warmup-context"):
        PipelineConfig(input="d.csv", mamba_warmup_context=True).validate()


def test_config_and_cli_options():
    args = run_pipeline.build_parser().parse_args(
        ["d.csv", "--encoder", "mamba", "--model", "sjm", "--min-train", "300", "--device", "cuda:1",
         "--mamba-seq-len", "30", "--mamba-d-model", "16", "--mamba-epochs", "5"])
    cfg = run_pipeline.config_from_args(args).validate()
    assert (cfg.encoder, cfg.model, cfg.min_train, cfg.device) == ("mamba", "sjm", 300, "cuda:1")
    assert (cfg.mamba_seq_len, cfg.mamba_d_model, cfg.mamba_epochs) == (30, 16, 5)
    with pytest.raises(ValueError, match="device"):
        PipelineConfig(input="d.csv", encoder="mamba", device="tpu").validate()
    with pytest.raises(ValueError, match="valid-frac"):
        PipelineConfig(input="d.csv", encoder="mamba", mamba_valid_frac=1.0).validate()
    with pytest.raises(ValueError, match="pin-features"):
        PipelineConfig(input="d.csv", encoder="mamba", model="sjm", pin_features=["sortino_20"]).validate()
    args = run_pipeline.build_parser().parse_args(["d.csv", "--encoder", "mamba", "--mamba-horizons", "1,5,20"])
    assert run_pipeline.config_from_args(args).validate().mamba_horizons == (1, 5, 20)
    assert PipelineConfig().mamba_horizons == (1,)
    for bad in [(), (0, 5), (1, 1), (300,)]:
        with pytest.raises(ValueError, match="mamba-horizons"):
            PipelineConfig(input="d.csv", encoder="mamba", mamba_horizons=bad).validate()
    args = run_pipeline.build_parser().parse_args(["d.csv", "--encoder", "mamba", "--mamba-targets", "Vol, mdd"])
    assert run_pipeline.config_from_args(args).validate().mamba_targets == ("vol", "mdd")
    assert PipelineConfig().mamba_targets == ("return",)
    for bad in [(), ("sharpe",), ("vol", "vol")]:
        with pytest.raises(ValueError, match="mamba-targets"):
            PipelineConfig(input="d.csv", encoder="mamba", mamba_targets=bad).validate()


class CaptureEncoder(MambaEncoder):
    """_train 에 넘어가는 학습 / 검증 샘플을 기록한다."""

    def _train(self, model, X_tr, Y_tr, X_va, Y_va, horizons=(1,)):
        self.captured = (X_tr, Y_tr, X_va, Y_va)
        return super()._train(model, X_tr, Y_tr, X_va, Y_va, horizons)


def test_multi_horizon_targets():
    """h 마다 t+1..t+h 수익률 합 / 학습창 h일 수익률 표준편차, 타깃은 모두 pos 이전, 학습 · 검증 사이 max(h)-1 간격."""
    X = build_features(simulate(), "paper")
    y = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(X)), index=X.index)
    lo, pos, seg_end = 100, 500, 620
    yv = y.to_numpy()

    one = CaptureEncoder(SMALL, "cpu", GRUBackbone)
    one(X, y, lo, pos, seg_end)
    X_tr, Y_tr, X_va, Y_va = one.captured
    assert Y_tr.shape[1] == 1 and len(X_tr) + len(X_va) == pos - 1 - lo  # 기본값: 기존 다음 날 타깃 그대로
    np.testing.assert_allclose(np.concatenate([Y_tr, Y_va])[:, 0], yv[lo + 1:pos] / yv[lo:pos].std(), rtol=1e-5)

    hs = (1, 5, 20)
    cfg = dataclasses.replace(SMALL, mamba_horizons=hs)
    enc = CaptureEncoder(cfg, "cpu", GRUBackbone)
    H = enc(X, y, lo, pos, seg_end)
    assert H.shape == (seg_end - lo, cfg.mamba_d_model)
    X_tr, Y_tr, X_va, Y_va = enc.captured
    assert Y_tr.shape[1] == Y_va.shape[1] == 3
    t_tr = lo + np.arange(len(X_tr))
    t_va = pos - 20 - len(X_va) + np.arange(len(X_va))
    assert t_va[-1] + 20 == pos - 1 and t_va[0] - t_tr[-1] == 20  # 마지막 타깃은 pos-1, 사이 19개 버림
    for k, h in enumerate(hs):
        sums = np.convolve(yv[lo:pos], np.ones(h), "valid")  # 학습창 안의 h일 수익률
        for t, row in [(t_tr, Y_tr), (t_va, Y_va)]:
            want = np.array([yv[s + 1:s + 1 + h].sum() for s in t]) / sums.std()
            np.testing.assert_allclose(row[:, k], want, rtol=1e-4, atol=1e-6)
    log = enc.history_frame()
    assert {"valid_loss_h1", "valid_loss_h5", "valid_loss_h20"} <= set(log.columns)
    np.testing.assert_allclose(log[["valid_loss_h1", "valid_loss_h5", "valid_loss_h20"]].mean(axis=1),
                               log["valid_loss"], rtol=1e-6)
    assert "valid_loss_h1" not in one.history_frame().columns

    # pos 이후 수익률은 학습에 쓰이지 않는다
    y2 = y.copy()
    y2.iloc[pos:] = 0.5
    H2 = MambaEncoder(cfg, "cpu", GRUBackbone)(X, y2, lo, pos, seg_end)
    np.testing.assert_allclose(H2, H, atol=1e-6)

@pytest.fixture
def asset_csv(tmp_path):
    ret = simulate()
    close = 1000 * (1 + ret).cumprod()
    path = tmp_path / "asset.csv"
    pd.DataFrame({"date": ret.index, "close": close.to_numpy(), "rf": 3.0}).to_csv(path, index=False)
    return str(path)


MAMBA_ARGS = ["--encoder", "mamba", "--discrete", "--train-window", "400", "--min-train", "250", "--n-init", "2",
              "--mamba-seq-len", "10", "--mamba-d-model", "4", "--mamba-epochs", "2", "--mamba-patience", "1",
              "--no-plots", "-q"]


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


def test_cli_mamba_with_sjm(cpu_mamba, asset_csv, tmp_path):
    """Jump Model 옵션은 인코더와 무관하게 그대로 적용된다 (sjm · feature-set)."""
    out = tmp_path / "sjm"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--model", "sjm", "--feature-set", "example",
                              *MAMBA_ARGS]) == 0
    fw = pd.read_csv(out / "feat_weights.csv", index_col=0)
    assert list(fw.columns) == [f"h_{k}" for k in range(4)]


def test_cli_mamba_inference(cpu_mamba, asset_csv, tmp_path):
    out = tmp_path / "inf"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--inference", *MAMBA_ARGS]) == 0
    assert len(pd.read_csv(out / "inference_mamba_train_log.csv")) == 1
    summary = pd.read_csv(out / "inference_summary.csv")
    assert summary["model"].iloc[0] == "jm (discrete) + mamba"


def test_cli_mamba_without_gpu_fails_cleanly(monkeypatch, asset_csv, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert run_pipeline.main([asset_csv, "--out", str(tmp_path), "--encoder", "mamba", "-q"]) == 1


def test_cli_mamba_multi_seed(cpu_mamba, asset_csv, tmp_path):
    """시드마다 Mamba 가중치가 달라지고, 앙상블 모드는 시드별 학습 로그를 모아 저장한다."""
    out = tmp_path / "seeds"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--n-seeds", "2", "--seed-mode", "ensemble",
                              *MAMBA_ARGS]) == 0
    log = pd.read_csv(out / "ensemble" / "mamba_train_log.csv")
    assert sorted(log["seed"].unique()) == [0, 1]
    reg = pd.read_csv(out / "ensemble" / "regimes.csv", index_col=0)
    assert {"regime_seed0", "regime_seed1", "agreement"} <= set(reg.columns)
    assert (out / "ensemble" / "performance.csv").exists() and not (out / "seed_0").exists()

    both = tmp_path / "both"
    assert run_pipeline.main([asset_csv, "--out", str(both), "--seeds", "0,1", *MAMBA_ARGS]) == 0
    p0 = pd.read_csv(both / "seed_0" / "refit_params.csv")
    p1 = pd.read_csv(both / "seed_1" / "refit_params.csv")
    assert not np.allclose(p0["center_h_0"], p1["center_h_0"])
    pd.testing.assert_frame_equal(pd.read_csv(both / "ensemble" / "regimes.csv", index_col=0), reg)


def test_cli_mamba_multi_horizon(cpu_mamba, asset_csv, tmp_path):
    out = tmp_path / "mh"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--mamba-horizons", "1,5,20", *MAMBA_ARGS]) == 0
    log = pd.read_csv(out / "mamba_train_log.csv")
    assert {"valid_loss_h1", "valid_loss_h5", "valid_loss_h20"} <= set(log.columns)
    with open(out / "run_config.json", encoding="utf-8") as f:
        assert json.load(f)["mamba_horizons"] == [1, 5, 20]
    assert (out / "performance.csv").exists()


def test_forward_targets_match_brute_force():
    y = np.random.default_rng(3).normal(0, 0.02, 80)
    t = np.array([-1, 0, 7, 40, 58])
    for h in (1, 5, 20):
        ret, vol, mdd = (forward_targets(y, t, h, k) for k in ("return", "vol", "mdd"))
        for i, s in enumerate(t):
            w = y[s + 1:s + 1 + h]
            wealth = np.concatenate([[1.0], np.cumprod(1 + w)])
            assert ret[i] == pytest.approx(w.sum())
            assert vol[i] == pytest.approx(np.sqrt(np.mean(w ** 2)))
            assert mdd[i] == pytest.approx(np.max(1 - wealth / np.maximum.accumulate(wealth)))
    assert forward_targets(np.array([0.01, -0.05, 0.02]), np.array([-1]), 1, "mdd")[0] == 0.0
    assert forward_targets(np.array([0.01, -0.05, 0.02]), np.array([0]), 1, "mdd")[0] == pytest.approx(0.05)
    with pytest.raises(ValueError, match="sharpe"):
        forward_targets(y, t, 5, "sharpe")


def test_vol_mdd_targets():
    """타깃 × horizon 출력, vol · mdd 는 학습창 안의 h일 값으로 표준화, pos 이후 수익률은 쓰지 않는다."""
    X = build_features(simulate(), "paper")
    y = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(X)), index=X.index)
    lo, pos, seg_end = 100, 500, 620
    yv = y.to_numpy()
    targets, hs = ("vol", "mdd"), (5, 20)
    cfg = dataclasses.replace(SMALL, mamba_targets=targets, mamba_horizons=hs)
    enc = CaptureEncoder(cfg, "cpu", GRUBackbone)
    H = enc(X, y, lo, pos, seg_end)
    X_tr, Y_tr, X_va, Y_va = enc.captured
    assert Y_tr.shape[1] == 4
    t_all = np.concatenate([lo + np.arange(len(X_tr)), pos - 20 - len(X_va) + np.arange(len(X_va))])
    Y_all = np.concatenate([Y_tr, Y_va])
    for j, (k, h) in enumerate((k, h) for k in targets for h in hs):
        ref = forward_targets(yv, np.arange(lo - 1, pos - h), h, k)
        assert ref.min() >= 0
        want = (forward_targets(yv, t_all, h, k) - ref.mean()) / ref.std()
        np.testing.assert_allclose(Y_all[:, j], want, rtol=1e-4, atol=1e-5)
    log = enc.history_frame()
    cols = ["valid_loss_vol_h5", "valid_loss_vol_h20", "valid_loss_mdd_h5", "valid_loss_mdd_h20"]
    assert set(cols) <= set(log.columns)
    np.testing.assert_allclose(log[cols].mean(axis=1), log["valid_loss"], rtol=1e-6)

    y2 = y.copy()
    y2.iloc[pos:] = -0.5
    H2 = MambaEncoder(cfg, "cpu", GRUBackbone)(X, y2, lo, pos, seg_end)
    np.testing.assert_allclose(H2, H, atol=1e-6)


def test_cli_mamba_vol_mdd(cpu_mamba, asset_csv, tmp_path):
    out = tmp_path / "vm"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--mamba-targets", "return,vol,mdd",
                              "--mamba-horizons", "1,5", *MAMBA_ARGS]) == 0
    log = pd.read_csv(out / "mamba_train_log.csv")
    assert {f"valid_loss_{k}_h{h}" for k in ("return", "vol", "mdd") for h in (1, 5)} <= set(log.columns)
    with open(out / "run_config.json", encoding="utf-8") as f:
        assert json.load(f)["mamba_targets"] == ["return", "vol", "mdd"]


def test_cli_mamba_warmup_context(cpu_mamba, asset_csv, tmp_path):
    out = tmp_path / "ctx"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--mamba-warmup-context", *MAMBA_ARGS]) == 0
    log = pd.read_csv(out / "mamba_train_log.csv")
    assert (log["n_zero_pad"] == 0).all() and log["n_context_rows"].iloc[0] == 9
    assert json.load(open(out / "run_config.json"))["mamba_warmup_context"] is True
