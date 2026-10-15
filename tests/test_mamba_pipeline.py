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


def test_loss_weights_option():
    args = run_pipeline.build_parser().parse_args(["d.csv", "--encoder", "mamba", "--mamba-horizons", "1,5,20",
                                                   "--mamba-loss-weights", "0.5, 0.25,0.25"])
    cfg = run_pipeline.config_from_args(args).validate()
    assert cfg.mamba_loss_weights == (0.5, 0.25, 0.25)
    np.testing.assert_allclose(mamba_encoder.loss_weights(cfg), [0.5, 0.25, 0.25])
    assert PipelineConfig().mamba_loss_weights == () and mamba_encoder.loss_weights(SMALL) is None
    # 합으로 정규화, horizon 수만큼이면 모든 타깃에 같은 horizon 가중치, 타깃 × horizon 수면 출력별
    two = dataclasses.replace(SMALL, mamba_targets=("return", "vol"), mamba_horizons=(5, 20))
    np.testing.assert_allclose(mamba_encoder.loss_weights(dataclasses.replace(two, mamba_loss_weights=(3, 1))),
                               [0.375, 0.125, 0.375, 0.125])
    np.testing.assert_allclose(mamba_encoder.loss_weights(dataclasses.replace(two, mamba_loss_weights=(1, 2, 3, 4))),
                               [0.1, 0.2, 0.3, 0.4])
    base = PipelineConfig(input="d.csv", encoder="mamba", mamba_horizons=(1, 5, 20))
    for bad in [(1, 1), (1, 1, 1, 1), (1, -1, 1), (0, 0, 0), (1, float("nan"), 1)]:
        with pytest.raises(ValueError, match="mamba-loss-weights"):
            dataclasses.replace(base, mamba_loss_weights=bad).validate()
    with pytest.raises(ValueError, match="mamba-loss-weights"):
        PipelineConfig(input="d.csv", mamba_loss_weights=(1.0,)).validate()


def test_weighted_loss_training():
    """검증 손실(early stopping 기준)은 출력별 검증 MSE 의 가중합이고, 가중치를 바꾸면 학습 결과가 달라진다."""
    X = build_features(simulate(), "paper")
    y = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(X)), index=X.index)
    cfg = dataclasses.replace(SMALL, mamba_horizons=(1, 5, 20), mamba_epochs=4, mamba_patience=4)
    w = (0.5, 0.25, 0.25)
    enc = MambaEncoder(dataclasses.replace(cfg, mamba_loss_weights=w), "cpu", GRUBackbone)
    H = enc(X, y, 100, 500, 620)
    log = enc.history_frame()
    cols = ["valid_loss_h1", "valid_loss_h5", "valid_loss_h20"]
    np.testing.assert_allclose(log[cols].to_numpy() @ np.array(w), log["valid_loss"], rtol=1e-6)
    H_mean = MambaEncoder(cfg, "cpu", GRUBackbone)(X, y, 100, 500, 620)
    assert not np.allclose(H, H_mean)


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


def test_cli_mamba_loss_weights(cpu_mamba, asset_csv, tmp_path):
    out = tmp_path / "lw"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--mamba-horizons", "1,5,20",
                              "--mamba-loss-weights", "2,1,1", *MAMBA_ARGS]) == 0
    log = pd.read_csv(out / "mamba_train_log.csv")
    np.testing.assert_allclose(log[["valid_loss_h1", "valid_loss_h5", "valid_loss_h20"]].to_numpy()
                               @ np.array([0.5, 0.25, 0.25]), log["valid_loss"], rtol=1e-6)
    with open(out / "run_config.json", encoding="utf-8") as f:
        assert json.load(f)["mamba_loss_weights"] == [2.0, 1.0, 1.0]
    assert run_pipeline.main([asset_csv, "--out", str(out), "--mamba-horizons", "1,5",
                              "--mamba-loss-weights", "1,1,1", *MAMBA_ARGS]) == 1


def test_cli_mamba_warmup_context(cpu_mamba, asset_csv, tmp_path):
    out = tmp_path / "ctx"
    assert run_pipeline.main([asset_csv, "--out", str(out), "--mamba-warmup-context", *MAMBA_ARGS]) == 0
    log = pd.read_csv(out / "mamba_train_log.csv")
    assert (log["n_zero_pad"] == 0).all() and log["n_context_rows"].iloc[0] == 9
    assert json.load(open(out / "run_config.json"))["mamba_warmup_context"] is True


def test_separate_config():
    args = run_pipeline.build_parser().parse_args(["d.csv", "--encoder", "mamba", "--mamba-targets", "return,mdd",
                                                   "--separate", "--separate-weights", "0.3", "1,1", "2,0", "0.5"])
    cfg = run_pipeline.config_from_args(args).validate()
    assert cfg.separate and cfg.separate_weights == ((0.3,), (1.0, 1.0), (2.0, 0.0), (0.5,))
    np.testing.assert_allclose(cfg.separate_blends, [(0.3, 0.7), (0.5, 0.5), (1.0, 0.0)])  # 정규화 · 중복 제거
    sep = PipelineConfig(input="d.csv", encoder="mamba", mamba_targets=("return", "vol", "mdd"), separate=True)
    np.testing.assert_allclose(sep.validate().separate_blends, [(1 / 3,) * 3])  # 기본: 균등
    for bad in [((0.5,),), ((1, 2),), ((1, -1, 1),), ((0, 0, 0),), ((1, float("nan"), 1),)]:
        with pytest.raises(ValueError, match="separate-weights"):
            dataclasses.replace(sep, separate_weights=bad).validate()
    with pytest.raises(ValueError, match="separate"):
        PipelineConfig(input="d.csv", encoder="mamba", separate=True).validate()  # 타깃 1개
    with pytest.raises(ValueError, match="separate"):
        PipelineConfig(input="d.csv", mamba_targets=("return", "mdd"), separate=True).validate()  # encoder none
    with pytest.raises(ValueError, match="separate-weights"):
        PipelineConfig(input="d.csv", encoder="mamba", separate_weights=((0.5,),)).validate()
    # 시드 1개여도 --separate 면 (타깃 사이) bear 표 기준을 쓸 수 있다
    dataclasses.replace(sep, ensemble_bear_vote=0.5).validate()

    # 타깃별 설정: 그 타깃만, 출력별 손실 가중치는 그 타깃 몫, horizon 별 가중치는 그대로
    two = dataclasses.replace(sep, mamba_targets=("return", "mdd"), mamba_horizons=(5, 20),
                              mamba_loss_weights=(1, 2, 3, 4), ensemble_bear_vote=0.5)
    t1 = run_pipeline._target_cfg(two, 1, "x")
    assert (t1.mamba_targets, t1.mamba_loss_weights, t1.separate, t1.ensemble_bear_vote) == (("mdd",), (3, 4), False,
                                                                                             None)
    t1.validate()
    hw = run_pipeline._target_cfg(dataclasses.replace(two, mamba_loss_weights=(1, 3), n_seeds=2), 0, "x")
    assert hw.mamba_loss_weights == (1, 3) and hw.ensemble_bear_vote == 0.5
    assert run_pipeline.blend_name(["return", "mdd"], (0.7, 0.3)) == "return0.7_mdd0.3"


def test_separate_features_config():
    args = run_pipeline.build_parser().parse_args(["d.csv", "--encoder", "mamba", "--mamba-targets", "return,vol,mdd",
                                                   "--separate", "--separate-features", "MDD=DD-log_10, sortino_20",
                                                   "return:sortino,VIX[log]"])
    cfg = run_pipeline.config_from_args(args).validate()
    assert cfg.separate_features == {"mdd": ("DD-log_10", "sortino_20"), "return": ("sortino", "VIX[log]")}
    assert run_pipeline._target_cfg(cfg, 2, "x").use_features == ("DD-log_10", "sortino_20")
    vol = run_pipeline._target_cfg(cfg, 1, "x")  # 안 준 타깃은 전부
    assert vol.use_features == () and vol.separate_features == {}
    vol.validate()
    with pytest.raises(SystemExit):
        run_pipeline.build_parser().parse_args(["d.csv", "--separate-features", "mdd"])
    dup = run_pipeline.build_parser().parse_args(["d.csv", "--separate-features", "mdd=a", "mdd=b"])
    with pytest.raises(ValueError, match="두 번"):
        run_pipeline.config_from_args(dup)
    for bad, match in [({"sharpe": ("a",)}, "타깃"), ({"mdd": ()}, "비어")]:
        with pytest.raises(ValueError, match=match):
            dataclasses.replace(cfg, separate_features=bad).validate()
    with pytest.raises(ValueError, match="--separate 에서만"):
        PipelineConfig(input="d.csv", encoder="mamba", separate_features={"return": ("a",)}).validate()


def test_build_features_keep():
    ret = simulate()
    extra = pd.DataFrame({"late": np.where(np.arange(len(ret)) < 400, np.nan, 1.0)}, index=ret.index)
    X = build_features(ret, "paper", extra=extra, warmup=10, keep=["sortino", "DD-log_10"])
    assert list(X.columns) == ["DD-log_10", "sortino_20", "sortino_60"]  # 원래 열 순서
    assert len(X) == len(ret) - 10  # 안 쓰는 늦게 시작하는 변수 때문에 행을 버리지 않는다
    assert len(build_features(ret, "paper", extra=extra, warmup=10, keep=["late"])) == len(ret) - 400
    with pytest.raises(KeyError, match="사용 가능한 피처"):
        build_features(ret, "paper", keep=["nope"])
    assert run_pipeline.blend_name(["return", "mdd"], (0.0, 1.0)) == "mdd"


def _reg(probs_bear, start="2020-01-01"):
    p = np.asarray(probs_bear, dtype=float)
    idx = pd.bdate_range(start, periods=len(p))
    return pd.DataFrame({"regime": (p > 0.5).astype(int), "prob_0": 1 - p, "prob_1": p,
                         "refit_date": idx[0]}, index=idx)


def test_weighted_ensemble_regimes():
    a, b = _reg([0.9, 0.6, 0.2, 0.1]), _reg([0.1, 0.1, 0.7, 0.4])
    ens = run_pipeline.ensemble_regimes({"a": a, "b": b}, 2, weights={"a": 3, "b": 1}, column="regime_{}")
    np.testing.assert_allclose(ens["prob_1"], [0.7, 0.475, 0.325, 0.175])
    assert list(ens["regime"]) == [1, 0, 0, 0] and list(ens["regime_b"]) == [0, 0, 1, 0]
    np.testing.assert_allclose(ens["bear_vote"], [0.75, 0.75, 0.25, 0])
    np.testing.assert_allclose(ens["agreement"], [0.75, 0.25, 0.75, 1])
    voted = run_pipeline.ensemble_regimes({"a": a, "b": b}, 2, bear_vote=0.5, weights={"a": 3, "b": 1})
    assert list(voted["regime"]) == [1, 1, 0, 0]
    # 비중이 없으면 기존 시드 평균과 같다
    plain = run_pipeline.ensemble_regimes({0: a, 1: b}, 2)
    np.testing.assert_allclose(plain["prob_1"], [0.5, 0.35, 0.45, 0.25])
    assert {"regime_seed0", "regime_seed1"} <= set(plain.columns)


SEP_ARGS = ["--mamba-targets", "return,mdd", "--mamba-horizons", "1,5", "--separate", *MAMBA_ARGS]


def test_cli_mamba_separate(cpu_mamba, asset_csv, tmp_path):
    """타깃마다 따로 학습한 결과(target_<t>/)의 상태 확률을 비중대로 섞어 비중마다 백테스트한다."""
    out = tmp_path / "sep"
    assert run_pipeline.main([asset_csv, "--out", str(out), *SEP_ARGS, "--separate-weights", "0.7", "0.5", "1"]) == 0
    for t in ("return", "mdd"):
        log = pd.read_csv(out / f"target_{t}" / "mamba_train_log.csv")
        labels = mamba_encoder.target_labels([t], (1, 5))  # 타깃 하나만 학습
        assert {f"valid_loss_{k}" for k in labels} <= set(log.columns) and len(labels) == 2
        with open(out / f"target_{t}" / "run_config.json", encoding="utf-8") as f:
            assert json.load(f)["mamba_targets"] == [t]
    ret = pd.read_csv(out / "target_return" / "regimes.csv", index_col=0, parse_dates=True)
    mdd = pd.read_csv(out / "target_mdd" / "regimes.csv", index_col=0, parse_dates=True)

    table = pd.read_csv(out / "separate_performance.csv", index_col=0)
    assert list(table.index[:4]) == ["return", "mdd", "return0.7_mdd0.3", "return0.5_mdd0.5"]  # 1 = return 단독
    assert "Buy & Hold" in table.index and {"w_return", "w_mdd"} <= set(table.columns)
    np.testing.assert_allclose(table.loc["return0.7_mdd0.3", ["w_return", "w_mdd"]].astype(float), [0.7, 0.3])
    # 타깃 단독 행은 그 타깃 run 의 성과와 같다
    solo = pd.read_csv(out / "target_mdd" / "performance.csv", index_col=0).loc["JM strategy"]
    assert float(table.loc["mdd", "sharpe"]) == pytest.approx(float(solo["sharpe"]))

    blend = pd.read_csv(out / "blend_return0.7_mdd0.3" / "regimes.csv", index_col=0, parse_dates=True)
    idx = blend.index
    np.testing.assert_allclose(blend["prob_1"], 0.7 * ret.loc[idx, "prob_1"] + 0.3 * mdd.loc[idx, "prob_1"])
    np.testing.assert_allclose(blend["prob_1_mdd"], mdd.loc[idx, "prob_1"])
    assert (blend["regime"] == blend[["prob_0", "prob_1"]].to_numpy().argmax(axis=1)).all()
    assert {"regime_return", "regime_mdd", "agreement", "bear_vote"} <= set(blend.columns)
    assert (out / "blend_return0.5_mdd0.5" / "performance.csv").exists()
    assert not (out / "blend_return").exists()
    states = pd.read_csv(out / "separate_current_state.csv")
    assert list(states["name"]) == list(table.index[:4])
    with open(out / "blend_return0.5_mdd0.5" / "current_state.json", encoding="utf-8") as f:
        assert json.load(f)["weights"] == {"return": 0.5, "mdd": 0.5}


def test_cli_mamba_separate_features(cpu_mamba, asset_csv, tmp_path):
    """--separate-features: 타깃마다 고른 피처만 그 타깃 Mamba 의 입력으로 쓴다."""
    out = tmp_path / "sepfeat"
    assert run_pipeline.main([asset_csv, "--out", str(out), *SEP_ARGS,
                              "--separate-features", "mdd=DD-log_10,sortino_20", "return=sortino"]) == 0
    feats = {t: pd.read_csv(out / f"target_{t}" / "features.csv", index_col=0).columns.tolist()
             for t in ("return", "mdd")}
    assert feats == {"return": ["sortino_20", "sortino_60"], "mdd": ["DD-log_10", "sortino_20"]}
    with open(out / "target_mdd" / "run_config.json", encoding="utf-8") as f:
        assert json.load(f)["use_features"] == ["DD-log_10", "sortino_20"]
    with open(out / "run_config.json", encoding="utf-8") as f:
        assert json.load(f)["separate_features"] == {"mdd": ["DD-log_10", "sortino_20"], "return": ["sortino"]}
    assert (out / "separate_performance.csv").exists()
    # 두 번째 타깃의 피처 이름이 틀려도 첫 타깃을 학습하기 전에 실패한다
    bad = tmp_path / "sepbad"
    assert run_pipeline.main([asset_csv, "--out", str(bad), *SEP_ARGS, "--separate-features", "mdd=nope"]) == 1
    assert not (bad / "target_return").exists()


def test_cli_mamba_separate_multi_seed_inference(cpu_mamba, asset_csv, tmp_path):
    """여러 시드면 타깃마다 시드 평균 확률을 섞는다 (--inference: 현재 반기만)."""
    out = tmp_path / "sepinf"
    assert run_pipeline.main([asset_csv, "--out", str(out), *SEP_ARGS, "--n-seeds", "2", "--seed-mode", "ensemble",
                              "--inference"]) == 0
    ens = {t: pd.read_csv(out / f"target_{t}" / "ensemble" / "inference_regimes.csv", index_col=0, parse_dates=True)
           for t in ("return", "mdd")}
    blend = pd.read_csv(out / "blend_return0.5_mdd0.5" / "inference_regimes.csv", index_col=0, parse_dates=True)
    np.testing.assert_allclose(blend["prob_1"], 0.5 * ens["return"].loc[blend.index, "prob_1"]
                               + 0.5 * ens["mdd"].loc[blend.index, "prob_1"])
    assert {"regime_return_seed0", "regime_mdd_seed1"} <= set(blend.columns)
    assert not (out / "separate_performance.csv").exists()
    assert len(pd.read_csv(out / "separate_current_state.csv")) == 3
