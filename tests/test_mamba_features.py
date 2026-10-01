"""regime_mamba.features 테스트 — Mamba 입력 피처 세트 (paper / example / extra / none).

torch 없이 실행된다.

    pytest tests/test_mamba_features.py
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from regime_mamba.features import (FEATURE_PREFIX, get_feature_columns, prepare_feature_set,  # noqa: E402
                                   returns_in_percent, standardize_for_window)


def make_config(**kw):
    base = dict(input_dim=4, feature_set=None, extra_feature_cols=[], feature_return_col="returns",
                feature_warmup=252, standardize_features=True, returns_pct=None, feature_cols=None)
    base.update(kw)
    return SimpleNamespace(**base)


def make_data(n=1500, seed=0, pct=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n).strftime("%Y-%m-%d")
    ret = rng.normal(0.0004, 0.01, n)
    return pd.DataFrame({
        "Date": dates,
        "returns": ret * (100 if pct else 1),
        "dd_10": rng.normal(size=n),
        "sortino_20": rng.normal(size=n),
        "sortino_60": rng.normal(size=n),
        "dollar_index": 100 + rng.normal(size=n).cumsum(),
        "target_returns_1": np.roll(ret, -1),
    })


def test_legacy_mode_is_unchanged():
    data = make_data()
    cfg = make_config(input_dim=4)
    out = prepare_feature_set(data, cfg)
    assert out is data
    assert cfg.input_dim == 4 and cfg.feature_cols is None
    assert get_feature_columns(cfg) == ["dd_10", "sortino_20", "sortino_60", "dollar_index"]
    assert get_feature_columns(make_config(input_dim=3)) == ["dd_10", "sortino_20", "sortino_60"]
    assert standardize_for_window(data, cfg, "2000-01-01", "2002-12-31") is data
    assert returns_in_percent(cfg) and not returns_in_percent(make_config(input_dim=3))
    with pytest.raises(ValueError):
        get_feature_columns(make_config(input_dim=9))


@pytest.mark.parametrize("feature_set, n_feats", [("paper", 3), ("example", 9), ("extra", 27)])
def test_feature_sets_set_columns_and_input_dim(feature_set, n_feats):
    data = make_data()
    cfg = make_config(feature_set=feature_set)
    out = prepare_feature_set(data, cfg)
    assert cfg.input_dim == n_feats == len(cfg.feature_cols)
    assert all(c.startswith(FEATURE_PREFIX) for c in cfg.feature_cols)
    assert get_feature_columns(cfg) == cfg.feature_cols
    # 기존 CSV 열은 덮어쓰지 않는다
    pd.testing.assert_series_equal(out["sortino_20"], data["sortino_20"].iloc[252:].reset_index(drop=True))
    # warmup 이후에는 결측이 없다
    assert len(out) == len(data) - 252
    assert not out[cfg.feature_cols].isna().any().any()


def test_paper_matches_regime_jm_definition():
    from regime_jm.features import paper_features
    data = make_data()
    cfg = make_config(feature_set="paper", feature_warmup=0)
    out = prepare_feature_set(data, cfg)
    expected = paper_features(data["returns"])
    assert cfg.feature_cols == [f"{FEATURE_PREFIX}{c}" for c in expected.columns]
    np.testing.assert_allclose(out[cfg.feature_cols].values, expected.values)


def test_extra_feature_cols_are_appended():
    cfg = make_config(feature_set="paper", extra_feature_cols=["dollar_index"])
    out = prepare_feature_set(make_data(), cfg)
    assert cfg.feature_cols[-1] == "dollar_index" and cfg.input_dim == 4
    assert "dollar_index" in out

    cfg = make_config(feature_set="none", extra_feature_cols=["dollar_index", "dd_10"])
    prepare_feature_set(make_data(), cfg)
    assert cfg.feature_cols == ["dollar_index", "dd_10"] and cfg.input_dim == 2


def test_invalid_settings_raise():
    with pytest.raises(ValueError):
        prepare_feature_set(make_data(), make_config(feature_set="mamba"))
    with pytest.raises(ValueError):
        prepare_feature_set(make_data(), make_config(feature_set="none"))
    with pytest.raises(ValueError):
        prepare_feature_set(make_data(), make_config(feature_set="paper", extra_feature_cols=["VIX"]))
    with pytest.raises(ValueError):
        get_feature_columns(make_config(feature_set="paper"))


def test_features_are_causal():
    """미래 수익률을 바꿔도 과거 피처는 변하지 않는다."""
    data = make_data()
    changed = data.copy()
    changed.loc[1000:, "returns"] *= -3
    cfg_a, cfg_b = make_config(feature_set="extra"), make_config(feature_set="extra")
    a, b = prepare_feature_set(data, cfg_a), prepare_feature_set(changed, cfg_b)
    cut = 1000 - 252
    pd.testing.assert_frame_equal(a[cfg_a.feature_cols].iloc[:cut], b[cfg_b.feature_cols].iloc[:cut])


def test_returns_unit_inference():
    cfg = make_config(feature_set="paper", input_dim=3)
    prepare_feature_set(make_data(pct=True), cfg)
    assert cfg.returns_pct is True and returns_in_percent(cfg)

    cfg = make_config(feature_set="paper", input_dim=4)
    prepare_feature_set(make_data(pct=False), cfg)
    assert cfg.returns_pct is False and not returns_in_percent(cfg)

    cfg = make_config(feature_set="extra", returns_pct=True)
    prepare_feature_set(make_data(pct=False), cfg)
    assert cfg.returns_pct is True


def test_standardize_uses_training_period_only():
    cfg = make_config(feature_set="example", extra_feature_cols=["dollar_index"])
    data = prepare_feature_set(make_data(), cfg)
    start, end = "2001-06-01", "2003-06-01"
    out = standardize_for_window(data, cfg, start, end)
    train = out[(out["Date"] >= start) & (out["Date"] <= end)][cfg.feature_cols]
    np.testing.assert_allclose(train.mean().values, 0.0, atol=1e-10)
    np.testing.assert_allclose(train.std().values, 1.0, atol=1e-10)
    # 원본은 그대로, 피처 외 열도 그대로
    assert not np.allclose(data[cfg.feature_cols].values, out[cfg.feature_cols].values)
    pd.testing.assert_series_equal(out["returns"], data["returns"])

    # 학습 구간 이후 데이터를 바꿔도 학습 구간 표준화 값은 같다
    changed = data.copy()
    later = changed["Date"] > end
    changed.loc[later, cfg.feature_cols] *= 10
    out2 = standardize_for_window(changed, cfg, start, end)
    pd.testing.assert_frame_equal(out.loc[~later, cfg.feature_cols], out2.loc[~later, cfg.feature_cols])

    no_std = make_config(**{**vars(cfg), "standardize_features": False})
    assert standardize_for_window(data, no_std, start, end) is data
