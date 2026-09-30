"""4) 0/1 전략 백테스트 · 성과표 · 거래 지연 로버스트니스.

t일 국면 신호는 t일 종가까지의 데이터로 계산된다. ``delay`` 일 뒤의 수익률부터 그 신호의 비중을 적용한다::

    weight[t] = target(regime[t - delay])          delay=1: 신호 다음 날부터 보유

absolute  bull → 위험자산 1-min_cash, bear → 위험자산 1-max_cash, 나머지는 무위험자산(rf_ret)
relative  bull → 자산, bear → 벤치마크 (현금 대신 벤치마크를 보유), 성과는 벤치마크 대비 초과성과(IR)

거래비용은 비중 변화량에 매수/매도 비용을 따로 곱해 그날 수익률에서 뺀다 (``trade_legs``).
relative 에서는 자산을 사고팔 때 반대편 벤치마크도 거래되므로 양쪽 모두 비용이 든다.
첫날의 초기 포지션 구축 비용은 계산하지 않는다 (Buy & Hold 와 같은 조건).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252
MODES = ("absolute", "relative")


def regime_to_weight(regimes: pd.Series, bear_state: int = 1, min_cash: float = 0.0,
                     max_cash: float = 1.0) -> pd.Series:
    """국면 → 목표 위험자산 비중. bear_state 만 bear, 나머지 상태는 bull 로 본다."""
    if not 0.0 <= min_cash <= max_cash <= 1.0:
        raise ValueError("0 <= min_cash <= max_cash <= 1 이어야 합니다")
    bull = regimes.to_numpy() != bear_state
    return pd.Series(np.where(bull, 1.0 - min_cash, 1.0 - max_cash), index=regimes.index, name="target_weight")


def trade_legs(weight: pd.Series, both_legs: bool = False) -> pd.DataFrame:
    """비중 변화 → 매수(buy)·매도(sell) 거래량. 첫날은 거래 없음으로 본다."""
    dw = weight.diff().fillna(0.0)
    buy = dw.clip(lower=0.0)
    sell = (-dw).clip(lower=0.0)
    if both_legs:
        buy, sell = buy + sell, sell + buy
    return pd.DataFrame({"buy": buy, "sell": sell}, index=weight.index)


def run_0_1_strategy(regimes: pd.Series, data: pd.DataFrame, *, delay: int = 1, bear_state: int = 1,
                     min_cash: float = 0.0, max_cash: float = 1.0, cost_buy: float = 0.0, cost_sell: float = 0.0,
                     mode: str = "absolute") -> pd.DataFrame:
    """국면 신호로 0/1 전략을 백테스트한다.

    Args:
        regimes: 날짜별 국면 (t일 종가까지의 정보로 계산된 값)
        data: prepare_inputs 결과 (asset_ret, rf_ret, [bench_ret])
        delay: 신호 → 체결 지연 (거래일). 0 은 같은 날 수익률에 적용하므로 look-ahead 이다.
        cost_buy, cost_sell: 비중 1 만큼 거래할 때의 비용 (소수, 10bp = 0.001)
        mode: absolute / relative

    Returns:
        DataFrame: regime, weight, asset_ret, alt_ret, buy, sell, cost, strat_ret, bh_ret, cum_* (+ relative 열)
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if delay < 0:
        raise ValueError("delay 는 0 이상이어야 합니다")
    if delay == 0:
        logger.warning("delay=0 은 신호를 같은 날 수익률에 적용하므로 look-ahead 입니다 (참고용)")
    if mode == "relative" and "bench_ret" not in data.columns:
        raise ValueError("relative 백테스트에는 --relative-benchmark 가 필요합니다")

    regimes = regimes.astype(int)
    target = regime_to_weight(regimes, bear_state, min_cash, max_cash)
    df = data.reindex(regimes.index)
    out = pd.DataFrame(index=regimes.index)
    out["regime"] = regimes
    out["signal_regime"] = regimes.shift(delay)
    out["weight"] = target.shift(delay)
    out["asset_ret"] = df["asset_ret"]
    out["rf_ret"] = df["rf_ret"]
    if "bench_ret" in df.columns:
        out["bench_ret"] = df["bench_ret"]
    out["alt_ret"] = df["rf_ret"] if mode == "absolute" else df["bench_ret"]
    out = out.dropna(subset=["weight"])

    legs = trade_legs(out["weight"], both_legs=(mode == "relative"))
    out["buy"], out["sell"] = legs["buy"], legs["sell"]
    out["cost"] = out["buy"] * cost_buy + out["sell"] * cost_sell
    w = out["weight"]
    out["strat_ret"] = w * out["asset_ret"] + (1.0 - w) * out["alt_ret"] - out["cost"]
    out["bh_ret"] = out["asset_ret"]
    out["cum_strategy"] = (1.0 + out["strat_ret"]).cumprod() - 1.0
    out["cum_bh"] = (1.0 + out["bh_ret"]).cumprod() - 1.0
    if "bench_ret" in out.columns:
        out["cum_bench"] = (1.0 + out["bench_ret"]).cumprod() - 1.0
    if mode == "relative":
        out["active_ret"] = out["strat_ret"] - out["bench_ret"]
        out["bh_active_ret"] = out["bh_ret"] - out["bench_ret"]
    out.attrs.update(mode=mode, delay=delay)
    return out


# ---------------------------------------------------------------------------
# 성과
# ---------------------------------------------------------------------------

def max_drawdown(ret: pd.Series) -> float:
    wealth = np.concatenate([[1.0], (1.0 + ret.to_numpy()).cumprod()])
    return float((wealth / np.maximum.accumulate(wealth) - 1.0).min())


def performance_metrics(ret: pd.Series, rf: Optional[pd.Series] = None, bench: Optional[pd.Series] = None,
                        weight: Optional[pd.Series] = None, periods: int = PERIODS_PER_YEAR) -> Dict[str, float]:
    """수익률, 변동성, Sharpe, Sortino, MDD, Calmar (+ 비중이 있으면 노출·거래 횟수, 벤치마크가 있으면 IR)."""
    ret = ret.dropna()
    n = len(ret)
    if n == 0:
        return {}
    years = n / periods
    total = float((1.0 + ret).prod() - 1.0)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    vol = float(ret.std() * np.sqrt(periods))
    ex = ret - (rf.reindex(ret.index).fillna(0.0) if rf is not None else 0.0)
    ex_std = ex.std()
    downside = np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0.0) ** 2))
    mdd = max_drawdown(ret)
    m = {
        "start": ret.index[0].date().isoformat(),
        "end": ret.index[-1].date().isoformat(),
        "n_days": n,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": float(ex.mean() / ex_std * np.sqrt(periods)) if ex_std > 0 else np.nan,
        "sortino": float(ex.mean() / downside * np.sqrt(periods)) if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }
    if weight is not None:
        w = weight.reindex(ret.index)
        dw = w.diff().abs().fillna(0.0)
        m["avg_weight"] = float(w.mean())
        m["n_trades"] = int((dw > 1e-12).sum())
        m["turnover"] = float(dw.sum() / years)
    if bench is not None:
        b = bench.reindex(ret.index)
        active = ret - b
        te = active.std() * np.sqrt(periods)
        rel_wealth = (1.0 + ret).cumprod() / (1.0 + b).cumprod()
        m["active_return"] = float(active.mean() * periods)
        m["tracking_error"] = float(te)
        m["information_ratio"] = float(active.mean() * periods / te) if te > 0 else np.nan
        m["active_cagr"] = float(rel_wealth.iloc[-1] ** (1.0 / years) - 1.0)
        m["active_mdd"] = float((rel_wealth / np.maximum(rel_wealth.cummax(), 1.0) - 1.0).min())
    return m


def performance_table(strat: pd.DataFrame, extra: Optional[Dict[str, pd.DataFrame]] = None,
                      name: str = "JM strategy") -> pd.DataFrame:
    """전략 · Buy & Hold (· 벤치마크 · 다른 전략) 성과표. relative 모드면 벤치마크 대비 지표가 붙는다."""
    mode = strat.attrs.get("mode", "absolute")
    rf = strat["rf_ret"]
    bench = strat["bench_ret"] if mode == "relative" else None
    rows = {name: performance_metrics(strat["strat_ret"], rf, bench, strat["weight"])}
    for other_name, other in (extra or {}).items():
        rows[other_name] = performance_metrics(other["strat_ret"], rf, bench, other["weight"])
    rows["Buy & Hold"] = performance_metrics(strat["bh_ret"], rf, bench)
    if "bench_ret" in strat.columns:
        rows["Benchmark"] = performance_metrics(strat["bench_ret"], rf, bench)
    table = pd.DataFrame(rows).T
    table.index.name = "portfolio"
    return table


def regime_summary(regimes: pd.Series, ret: pd.Series, n_states: int = 2,
                   periods: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    """국면별 일수·비중·연율 수익률/변동성·에피소드 수·평균 지속기간 (같은 날 수익률 기준, 설명용)."""
    from .rolling import state_label

    ret = ret.reindex(regimes.index)
    runs = (regimes != regimes.shift()).cumsum()
    rows = []
    for k in range(n_states):
        mask = regimes == k
        r = ret[mask]
        n_ep = int(runs[mask].nunique())
        std = r.std()
        rows.append({
            "state": k,
            "label": state_label(k, n_states),
            "n_days": int(mask.sum()),
            "pct_days": float(mask.mean()),
            "ann_ret": float(r.mean() * periods) if len(r) else np.nan,
            "ann_vol": float(std * np.sqrt(periods)) if len(r) > 1 else np.nan,
            "sharpe": float(r.mean() / std * np.sqrt(periods)) if len(r) > 1 and std > 0 else np.nan,
            "n_episodes": n_ep,
            "avg_duration": float(mask.sum() / n_ep) if n_ep else np.nan,
        })
    return pd.DataFrame(rows)


def delay_robustness_table(regimes: pd.Series, data: pd.DataFrame, delays: Iterable[int] = (1, 2, 3, 5, 10),
                           **strategy_kwargs) -> pd.DataFrame:
    """여러 거래 지연에서의 성과 (논문 Table 5). 가장 긴 지연 기준의 공통 구간에서 비교한다."""
    delays = sorted(set(int(d) for d in delays))
    runs = {d: run_0_1_strategy(regimes, data, delay=d, **strategy_kwargs) for d in delays}
    common = runs[delays[0]].index
    for s in runs.values():
        common = common.intersection(s.index)
    rows = {}
    mode = strategy_kwargs.get("mode", "absolute")
    for d, s in runs.items():
        s = s.loc[common]
        bench = s["bench_ret"] if mode == "relative" else None
        rows[d] = performance_metrics(s["strat_ret"], s["rf_ret"], bench, s["weight"])
    table = pd.DataFrame(rows).T
    table.index.name = "delay"
    return table
