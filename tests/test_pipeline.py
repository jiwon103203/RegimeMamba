"""regime_jm 파이프라인 테스트 — 특히 인과성(미래 정보 미사용) 규칙을 검증한다.

    pytest tests/test_pipeline.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_pipeline  # noqa: E402
from regime_jm.backtest import (delay_robustness_table, max_drawdown, performance_metrics,  # noqa: E402
                                performance_table, regime_to_weight, run_0_1_strategy, trade_legs)
from regime_jm.data_io import (align_to_index, load_market_data, parse_extra_spec, prepare_inputs,  # noqa: E402
                               resolve_signal_return, rf_to_daily)
from regime_jm.features import apply_transform, build_features, expand_feature_names, feature_meta  # noqa: E402
from regime_jm.hmm_benchmark import forward_filter, run_rolling_hmm, trailing_median  # noqa: E402
from regime_jm.regime_episodes import (compare_episode_paths, episode_metrics, extract_episodes,  # noqa: E402
                                       length_scenarios, rank_similar_episodes)
from regime_jm.rolling import WindowPreprocessor, refit_schedule, run_rolling_jm  # noqa: E402
from regime_jm.sparse_pin import PinnedSparseJumpModel, solve_lasso_pinned  # noqa: E402
from regime_jm.weights import group_weights, weight_shares  # noqa: E402

FAST = dict(model="jm", cont=False, train_window=400, min_train=250, n_init=3)


def simulate(n=1600, seed=0, start="2010-01-04"):
    """2-상태 마르코프 국면 수익률 (bull: +, 저변동 / bear: -, 고변동)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.995 if state[t - 1] == 0 else 0.98
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    mu = np.where(state == 0, 0.0006, -0.0015)
    sd = np.where(state == 0, 0.008, 0.022)
    return idx, pd.Series(rng.normal(mu, sd), index=idx), state


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    idx, ret, _ = simulate()
    rng = np.random.default_rng(1)
    close = 1000 * (1 + ret).cumprod()
    bench = 2000 * (1 + 0.6 * ret + rng.normal(0.0001, 0.006, len(ret))).cumprod()
    rf = 3.0 + np.cumsum(rng.normal(0, 0.01, len(ret)))
    asset = pd.DataFrame({
        "일자": idx.strftime("%Y.%m.%d"),
        "종가": [f"{c:,.2f}" for c in close],
        "CD91(%)": [f"{x:.2f}%" for x in rf],
        "거래량": rng.integers(1000, 5000, len(ret)),
    })
    asset.to_csv(d / "asset.csv", index=False, encoding="cp949")
    pd.DataFrame({"date": idx, "close": bench}).to_csv(d / "kospi.csv", index=False)
    pd.DataFrame({"Date": idx[::5], "VIX": 20 + np.abs(np.cumsum(rng.normal(0, 0.5, len(idx[::5]))))}).to_csv(
        d / "macro.csv", index=False)
    return {"asset": str(d / "asset.csv"), "bench": str(d / "kospi.csv"), "macro": str(d / "macro.csv"),
            "close": pd.Series(close.to_numpy(), index=idx), "rf": pd.Series(rf, index=idx)}


# ---------------------------------------------------------------------------
# 1) 데이터
# ---------------------------------------------------------------------------

def test_load_market_data_detects_korean_columns_cp949_commas_percent(files):
    df = load_market_data(files["asset"])
    assert df.attrs["source_columns"] == {"date": "일자", "close": "종가", "rf": "CD91(%)"}
    np.testing.assert_allclose(df["close"], files["close"].round(2), rtol=0, atol=1e-9)
    np.testing.assert_allclose(df["rf"], files["rf"].round(2), atol=1e-9)
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing


def test_rf_to_daily_units():
    rf = pd.Series([3.65])
    assert rf_to_daily(rf).iloc[0] == pytest.approx(3.65 / 100 / 252)
    assert rf_to_daily(rf / 100, "annual").iloc[0] == pytest.approx(0.0365 / 252)
    assert rf_to_daily(rf, "daily_pct").iloc[0] == pytest.approx(0.0365)
    with pytest.raises(ValueError):
        rf_to_daily(rf, "monthly")


def test_prepare_inputs_excess_and_relative_returns(files):
    data = prepare_inputs(files["asset"], benchmark=files["bench"])
    expected_rf = rf_to_daily(data["rf"]).shift(1)
    np.testing.assert_allclose(data["rf_ret"].iloc[1:], expected_rf.iloc[1:])
    np.testing.assert_allclose(data["ret"], data["asset_ret"] - data["rf_ret"])
    np.testing.assert_allclose(data["rel_ret"], data["asset_ret"] - data["bench_ret"])
    assert resolve_signal_return(data, "auto") == "rel_ret"
    assert resolve_signal_return(data, "absolute") == "ret"
    plain = prepare_inputs(files["asset"])
    assert resolve_signal_return(plain, "auto") == "ret"
    with pytest.raises(ValueError):
        resolve_signal_return(plain, "relative")


def test_parse_extra_spec_variants(tmp_path):
    f = tmp_path / "m.csv"
    f.write_text("date,VIX\n2020-01-01,1\n")
    s = parse_extra_spec(f"{f}:VIX:log")
    assert (s.file, s.column, s.transform, s.name) == (str(f), "VIX", "log", "VIX[log]")
    s = parse_extra_spec(f"{f}:VIX")
    assert (s.file, s.column, s.transform) == (str(f), "VIX", "none")
    s = parse_extra_spec("거래량:zscore_60")
    assert (s.file, s.column, s.transform) == (None, "거래량", "zscore_60")
    s = parse_extra_spec(r"C:\data\macro.csv:USDKRW:logdiff")
    assert (s.file, s.column, s.transform) == (r"C:\data\macro.csv", "USDKRW", "logdiff")


def test_align_to_index_uses_only_past_values():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-01", "2020-01-08"]))
    idx = pd.bdate_range("2020-01-01", "2020-01-10")
    out = align_to_index(s, idx)
    assert (out.loc[:"2020-01-07"] == 1.0).all() and (out.loc["2020-01-08":] == 2.0).all()


# ---------------------------------------------------------------------------
# 2) 피처
# ---------------------------------------------------------------------------

def test_feature_sets_and_removal():
    _, ret, _ = simulate(800)
    paper = build_features(ret, "paper")
    assert list(paper.columns) == ["DD-log_10", "sortino_20", "sortino_60"]
    assert len(paper) == len(ret) - 252
    example = build_features(ret, "example")
    assert example.shape[1] == 9 and {"ret_5", "DD-log_60", "sortino_20"} <= set(example.columns)
    extra = build_features(ret, "extra")
    assert extra.shape[1] == 27
    assert not any(c.startswith("DD-log") for c in extra.columns)
    assert {"DD_5", "std_60", "var_20", "mad_5", "rms_60", "vol-log_20", "vol-chg_60"} <= set(extra.columns)
    reduced = build_features(ret, "extra", remove=["var", "vol-chg_5"])
    assert reduced.shape[1] == 27 - 3 - 1 and not any(c.startswith("var_") for c in reduced.columns)
    meta = feature_meta(["DD-log_10", "VIX[logdiff]"])
    assert meta.loc["DD-log_10", "family"] == "DD-log" and meta.loc["DD-log_10", "horizon"] == 10
    assert meta.loc["VIX[logdiff]", "category"] == "custom"
    assert expand_feature_names(extra.columns, ["sortino"]) == ["sortino_5", "sortino_20", "sortino_60"]


@pytest.mark.parametrize("feature_set", ["paper", "extra"])
def test_features_are_causal(feature_set):
    _, ret, _ = simulate(900)
    full = build_features(ret, feature_set)
    cut = ret.index[700]
    trunc = build_features(ret.loc[:cut], feature_set)
    pd.testing.assert_frame_equal(full.loc[:cut], trunc)


def test_apply_transform_is_causal():
    s = pd.Series(np.exp(np.random.default_rng(0).normal(size=300).cumsum() * 0.01) * 100)
    for tf in ["log", "diff_5", "pct", "logdiff+ewm_10", "zscore_60", "lag_2"]:
        full = apply_transform(s, tf)
        part = apply_transform(s.iloc[:200], tf)
        pd.testing.assert_series_equal(full.iloc[:200], part)
    assert apply_transform(s, "lag_2").iloc[5] == s.iloc[3]
    with pytest.raises(ValueError):
        apply_transform(s, "cube")


# ---------------------------------------------------------------------------
# 3) 롤링 재추정
# ---------------------------------------------------------------------------

def test_refit_schedule_first_business_day_of_jan_jul():
    idx = pd.bdate_range("2010-01-04", "2016-12-30")
    dates = refit_schedule(idx, (1, 7), min_train=300)
    assert dates and all(d.month in (1, 7) for d in dates)
    for d in dates:
        assert d == idx[(idx.year == d.year) & (idx.month == d.month)][0]
        assert idx.get_loc(d) >= 300
    assert refit_schedule(idx, (1, 7), 300, start="2014-03-01")[0] == pd.Timestamp("2014-07-01")


def test_preprocessor_is_fit_on_training_window_only():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 2)), columns=["a", "b"])
    prep = WindowPreprocessor(3.0).fit(X.iloc[:300])
    X2 = X.copy()
    X2.iloc[300:] *= 100.0  # 학습창 밖의 값이 바뀌어도
    prep2 = WindowPreprocessor(3.0).fit(X2.iloc[:300])
    pd.testing.assert_frame_equal(prep.transform(X.iloc[:300]), prep2.transform(X2.iloc[:300]))
    z = prep.transform(X.iloc[:300])
    np.testing.assert_allclose(z.mean(), 0, atol=1e-10)


def test_rolling_jm_is_causal_within_segment():
    """재추정 구간 중간 이후의 데이터를 바꿔도 그 이전 날짜의 국면·확률은 변하지 않아야 한다.

    (스케일러를 구간 데이터로 fit 하거나 온라인이 아닌 추론을 쓰면 실패한다.)
    """
    _, ret, _ = simulate(1600)
    X = build_features(ret, "paper")
    base = run_rolling_jm(X, ret, **FAST)
    seg = base.regimes[base.regimes["refit_date"] == base.refit_dates[2]]
    cut = seg.index[len(seg) // 2]

    ret2 = ret.copy()
    rng = np.random.default_rng(99)
    ret2.loc[cut:] = rng.normal(-0.01, 0.05, int((ret.index >= cut).sum()))
    X2 = build_features(ret2, "paper")
    pert = run_rolling_jm(X2, ret2, **FAST)

    before = base.regimes.index < cut
    pd.testing.assert_frame_equal(base.regimes[before], pert.regimes[before])
    assert not base.regimes[~before].equals(pert.regimes[~before])  # 이후는 실제로 달라짐


def test_rolling_jm_windows_and_state_order():
    _, ret, true_state = simulate(1600)
    X = build_features(ret, "paper")
    res = run_rolling_jm(X, ret, **FAST)
    p = res.params
    assert (p["n_train"] <= 400).all() and (p["n_train"] >= 250).all()
    assert (pd.to_datetime(p["train_end"]) < pd.to_datetime(p["refit_date"])).all()
    # sort_by="cumret": 상태 0 의 누적 수익 기여 >= 마지막 상태
    for _, g in p.groupby("refit_date"):
        contrib = (g["ann_ret"].fillna(-np.inf) * g["freq"]).to_numpy()
        assert contrib[0] >= contrib[-1]
    assert set(res.regimes["regime"].unique()) <= {0, 1}
    assert res.regimes.index.is_unique and res.regimes.index[0] == res.refit_dates[0]
    acc = (res.regimes["regime"].to_numpy() == pd.Series(true_state, index=ret.index)
           .reindex(res.regimes.index).to_numpy()).mean()
    assert acc > 0.6


def test_continuous_and_sparse_models_run():
    _, ret, _ = simulate(1100)
    X = build_features(ret, "example")
    cjm = run_rolling_jm(X, ret, model="jm", cont=True, train_window=400, min_train=250, n_init=2)
    probs = cjm.regimes[["prob_0", "prob_1"]]
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    sjm = run_rolling_jm(X, ret, model="sjm", cont=False, train_window=400, min_train=250, n_init=2,
                         max_feats=2.0, pinned=["ret_60"])
    assert sjm.feat_weights is not None and (sjm.feat_weights["ret_60"] > 0).all()
    assert list(sjm.feat_weights.index) == sjm.refit_dates


# ---------------------------------------------------------------------------
# sparse_pin
# ---------------------------------------------------------------------------

def test_solve_lasso_pinned_keeps_pinned_and_respects_norms():
    a = np.array([1.0, 0.8, 0.05, 0.0, 0.0])
    pinned = np.array([False, False, False, False, True])
    w = solve_lasso_pinned(a, np.sqrt(2.0), pinned)
    assert w[4] > 0 and w[3] == 0
    assert np.linalg.norm(w) == pytest.approx(1.0)
    assert w.sum() <= np.sqrt(2.0) + 1e-6


def test_pinned_sparse_jump_model_selects_pinned_noise_feature():
    idx, ret, state = simulate(900)
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"sig1": state + rng.normal(0, 0.3, len(state)),
                      "sig2": state + rng.normal(0, 0.3, len(state)),
                      "noise": rng.normal(size=len(state))}, index=idx)
    X = (X - X.mean()) / X.std()
    plain = PinnedSparseJumpModel(n_components=2, max_feats=1.5, jump_penalty=10.0, random_state=0).fit(X, ret)
    pinned = PinnedSparseJumpModel(n_components=2, max_feats=1.5, jump_penalty=10.0, random_state=0,
                                   pinned=["noise"]).fit(X, ret)
    assert plain.feat_weights["noise"] == 0
    assert pinned.feat_weights["noise"] > 0


# ---------------------------------------------------------------------------
# 4) 백테스트
# ---------------------------------------------------------------------------

def _toy_data(n=10):
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"asset_ret": rng.normal(0, 0.01, n), "rf_ret": 0.0001, "bench_ret": rng.normal(0, 0.01, n)},
                        index=idx)
    regimes = pd.Series([0, 0, 1, 1, 1, 0, 0, 1, 0, 0], index=idx)
    return data, regimes


@pytest.mark.parametrize("delay", [1, 2])
def test_strategy_trades_after_delay(delay):
    data, regimes = _toy_data()
    s = run_0_1_strategy(regimes, data, delay=delay)
    expected_w = regime_to_weight(regimes).shift(delay).dropna()
    pd.testing.assert_series_equal(s["weight"], expected_w, check_names=False)
    assert s.index[0] == regimes.index[delay]
    w = s["weight"]
    np.testing.assert_allclose(s["strat_ret"], w * s["asset_ret"] + (1 - w) * s["rf_ret"])


def test_strategy_signal_uses_only_past_regimes():
    data, regimes = _toy_data()
    s1 = run_0_1_strategy(regimes, data, delay=1)
    changed = regimes.copy()
    changed.iloc[6:] = 1 - changed.iloc[6:]
    s2 = run_0_1_strategy(changed, data, delay=1)
    # t일 비중은 t-1일까지의 국면에만 의존
    pd.testing.assert_series_equal(s1["weight"].iloc[:6], s2["weight"].iloc[:6])


def test_trade_legs_and_separate_costs():
    w = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0])
    legs = trade_legs(w)
    assert legs["buy"].tolist() == [0, 0, 0, 0, 1] and legs["sell"].tolist() == [0, 0, 1, 0, 0]
    both = trade_legs(w, both_legs=True)
    assert both["buy"].tolist() == [0, 0, 1, 0, 1] and both["sell"].tolist() == [0, 0, 1, 0, 1]

    data, regimes = _toy_data()
    s = run_0_1_strategy(regimes, data, delay=1, cost_buy=0.001, cost_sell=0.003)
    np.testing.assert_allclose(s["cost"], s["buy"] * 0.001 + s["sell"] * 0.003)
    free = run_0_1_strategy(regimes, data, delay=1)
    np.testing.assert_allclose(free["strat_ret"] - s["strat_ret"], s["cost"])


def test_min_max_cash_bounds():
    data, regimes = _toy_data()
    s = run_0_1_strategy(regimes, data, delay=1, min_cash=0.1, max_cash=0.7)
    assert set(np.round(s["weight"], 10)) == {0.9, 0.3}
    with pytest.raises(ValueError):
        run_0_1_strategy(regimes, data, min_cash=0.8, max_cash=0.2)


def test_relative_mode_holds_benchmark_in_bear():
    data, regimes = _toy_data()
    s = run_0_1_strategy(regimes, data, delay=1, mode="relative")
    bear = s["weight"] == 0
    np.testing.assert_allclose(s.loc[bear, "active_ret"], 0.0, atol=1e-15)
    np.testing.assert_allclose(s.loc[~bear, "active_ret"], (s["asset_ret"] - s["bench_ret"])[~bear])
    table = performance_table(s)
    assert {"information_ratio", "tracking_error", "active_return"} <= set(table.columns)
    with pytest.raises(ValueError):
        run_0_1_strategy(regimes, data.drop(columns="bench_ret"), mode="relative")


def test_performance_metrics_known_values():
    idx = pd.bdate_range("2021-01-04", periods=3)
    r = pd.Series([0.1, -0.5, 0.2], index=idx)
    assert max_drawdown(r) == pytest.approx(-0.5)
    m = performance_metrics(r)
    assert m["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert m["max_drawdown"] == pytest.approx(-0.5)


def test_delay_robustness_uses_common_window():
    data, regimes = _toy_data(10)
    table = delay_robustness_table(regimes, data, delays=[1, 3])
    assert list(table.index) == [1, 3]
    assert table["n_days"].nunique() == 1 and table["start"].nunique() == 1


# ---------------------------------------------------------------------------
# 5) HMM
# ---------------------------------------------------------------------------

def test_hmm_forward_filter_and_median_are_causal():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0.05, 0.8, 200), rng.normal(-0.2, 2.5, 60), rng.normal(0.05, 0.8, 100)])
    args = (np.array([0.5, 0.5]), np.array([[0.98, 0.02], [0.05, 0.95]]), np.array([0.05, -0.2]),
            np.array([0.64, 6.25]))
    f1 = forward_filter(x, *args)
    x2 = x.copy()
    x2[300:] += 10
    f2 = forward_filter(x2, *args)
    np.testing.assert_allclose(f1[:300], f2[:300])
    np.testing.assert_allclose(f1.sum(axis=1), 1.0)
    reg = pd.Series(f1.argmax(axis=1))
    reg2 = pd.Series(f2.argmax(axis=1))
    pd.testing.assert_series_equal(trailing_median(reg, 5).iloc[:300], trailing_median(reg2, 5).iloc[:300])


def test_rolling_hmm_bull_is_higher_mean_state():
    _, ret, _ = simulate(900)
    hmm = run_rolling_hmm(ret, train_window=400, min_train=300, refit_every=63, n_init=1)
    assert hmm.index[0] == ret.index[300]
    grouped = ret.reindex(hmm.index).groupby(hmm["regime_raw"]).mean()
    assert grouped.loc[0] > grouped.loc[1]


# ---------------------------------------------------------------------------
# 6) 해석
# ---------------------------------------------------------------------------

def test_weight_shares_and_groups():
    fw = pd.DataFrame({"ret_5": [1.0, 0.0], "ret_20": [1.0, 1.0], "std_20": [0.0, 1.0], "VIX[log]": [1.0, 1.0]},
                      index=pd.to_datetime(["2020-01-02", "2020-07-01"]))
    shares = weight_shares(fw)
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    cat = group_weights(fw, "category")
    assert cat.loc["2020-01-02", "return"] == pytest.approx(2 / 3)
    hz = group_weights(fw, "horizon")
    assert hz.loc["2020-07-01", "20d"] == pytest.approx(2 / 3) and hz.loc["2020-07-01", "custom"] == pytest.approx(1 / 3)


def test_episodes_similarity_and_scenarios():
    idx = pd.bdate_range("2020-01-01", periods=30)
    reg = pd.Series([0] * 5 + [1] * 6 + [0] * 4 + [1] * 8 + [0] * 3 + [1] * 4, index=idx)
    ret = pd.Series(0.0, index=idx)
    ret.iloc[5:11] = -0.01      # bear 에피소드 1: 하락 → 정상 신호
    ret.iloc[15:23] = 0.01      # bear 에피소드 2: 상승 → 오신호
    ret.iloc[26:30] = -0.01     # 현재 bear: 하락
    eps = extract_episodes(reg)
    assert eps["n_days"].tolist() == [5, 6, 4, 8, 3, 4]
    assert eps["ongoing"].tolist() == [False] * 5 + [True]
    eps = episode_metrics(eps, ret, bear_state=1)
    assert eps.loc[1, "cum_ret"] == pytest.approx(0.99 ** 6 - 1)
    assert eps.loc[1, "mdd"] == pytest.approx(0.99 ** 6 - 1)
    assert bool(eps.loc[1, "false_signal"]) is False and bool(eps.loc[3, "false_signal"]) is True
    assert pd.isna(eps.loc[5, "false_signal"])

    sim = rank_similar_episodes(eps, ret, top_k=None)
    assert sim["episode_id"].tolist()[0] == 1  # 같은 하락 경로가 가장 비슷
    paths = compare_episode_paths(ret, eps)
    assert paths.iloc[0]["episode_id"] == 1 and paths.iloc[0]["rmse"] == pytest.approx(0.0)

    sc = length_scenarios(eps, similar_ids=[1]).set_index("scenario")
    assert sc.loc["all", "n_ref"] == 2                     # 길이 4 이상인 과거 bear: 6, 8
    assert sc.loc["all", "q50_remaining"] == pytest.approx(3.0)   # 남은 기간 2, 4
    assert sc.loc["similar", "mean_remaining"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 전체 파이프라인 / CLI
# ---------------------------------------------------------------------------

PIPE_FAST = dict(cont=False, train_window=400, min_train=250, n_init=2, delays=(1, 2))


def test_run_pipeline_end_to_end(files, tmp_path):
    out = tmp_path / "out"
    res = run_pipeline.run_pipeline(input=files["asset"], out_dir=str(out), hmm=True, hmm_refit=126,
                                    hmm_n_init=1, extra_features=[f"{files['macro']}:VIX:logdiff", "거래량:zscore_60"],
                                    **PIPE_FAST)
    for name in ["regimes.csv", "refit_params.csv", "strategy.csv", "performance.csv", "delay_robustness.csv",
                 "regime_episodes.csv", "similar_episodes.csv", "length_scenarios.csv", "model_comparison.csv",
                 "hmm_regimes.csv", "regimes_cumret.png", "similar_paths.png", "current_state.json"]:
        assert (out / name).exists(), name
    assert {"VIX[logdiff]", "거래량[zscore_60]"} <= set(res["features"].columns)
    strat, reg = res["strategy"], res["regimes"]
    # 거래는 신호보다 delay(1)일 늦게 체결
    expected = regime_to_weight(reg["regime"], bear_state=1).shift(1).reindex(strat.index)
    np.testing.assert_allclose(strat["weight"], expected)
    comp = res["model_comparison"]
    assert comp.loc["JM strategy", "start"] == comp.loc["HMM strategy", "start"]
    state = json.loads((out / "current_state.json").read_text(encoding="utf-8"))
    assert state["label"] in ("bull", "bear")


def test_run_pipeline_sjm_relative(files, tmp_path):
    res = run_pipeline.run_pipeline(input=files["asset"], out_dir=str(tmp_path), model="sjm",
                                    feature_set="example", pin_features=["DD-log_5"], max_feats=2.0,
                                    relative_benchmark=files["bench"], backtest_ret="relative", plots=False,
                                    **PIPE_FAST)
    assert res["signal_col"] == "rel_ret"
    assert "information_ratio" in res["performance"].columns
    assert (tmp_path / "feat_weights.csv").exists() and (tmp_path / "weight_groups.csv").exists()
    fw = pd.read_csv(tmp_path / "feat_weights.csv", index_col=0)
    assert (fw["DD-log_5"] > 0).all()


def test_inference_mode_fits_once_on_latest_half(files, tmp_path):
    res = run_pipeline.run_pipeline(input=files["asset"], out_dir=str(tmp_path), inference=True, **PIPE_FAST)
    X = res["features"]
    last_refit = refit_schedule(X.index, (1, 7), 250)[-1]
    assert res["result"].refit_dates == [last_refit]
    assert res["regimes"].index[0] == last_refit and res["regimes"].index[-1] == X.index[-1]
    assert res["params"]["n_train"].iloc[0] == 400
    assert (tmp_path / "inference_summary.csv").exists()
    assert not (tmp_path / "strategy.csv").exists()


def test_cli_main(files, tmp_path):
    rc = run_pipeline.main([files["asset"], "--out", str(tmp_path), "--discrete", "--train-window", "400",
                            "--min-train", "250", "--n-init", "2", "--delays", "1,2", "--no-plots", "-q",
                            "--cost-buy-bps", "5", "--cost-sell-bps", "25"])
    assert rc == 0
    cfg = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["cost_buy"] == pytest.approx(0.0005) and cfg["cost_sell"] == pytest.approx(0.0025)
    assert cfg["cont"] is False
    assert run_pipeline.main([str(tmp_path / "missing.csv"), "--out", str(tmp_path), "-q"]) == 1
