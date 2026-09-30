"""7) 결과 그림 (png). 각 그림의 숫자는 같은 이름의 csv 에도 저장된다.

색은 검증된 categorical 순서(slot 1 blue → 2 orange → 3 aqua ...)를 고정 순서로 쓰고,
비교 기준(Buy & Hold 등)은 회색으로 두어 전략을 강조한다. 이축(dual-axis) 그림은 쓰지 않는다.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib import font_manager  # noqa: E402

logger = logging.getLogger(__name__)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BEAR_WASH = (0.890, 0.286, 0.282, 0.12)  # slot-8 red, 12%
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LW = 1.4

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
})

# 한글 열 이름(사용자 변수 등)이 그림에 나오므로 설치된 한글 글꼴을 fallback 으로 붙인다
KOREAN_FONTS = ("Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "NanumBarunGothic",
                "Noto Sans CJK KR", "Noto Sans KR", "UnDotum")
_installed = {f.name for f in font_manager.fontManager.ttflist}
_korean = [f for f in KOREAN_FONTS if f in _installed]
plt.rcParams["font.sans-serif"] = _korean + list(plt.rcParams["font.sans-serif"])
if not _korean:
    warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font")


def _has_hangul(fig) -> bool:
    return any(any("\uac00" <= ch <= "\ud7a3" for ch in t.get_text()) for t in fig.findobj(matplotlib.text.Text))


def _save(fig, path: str) -> str:
    if not _korean and _has_hangul(fig):
        logger.warning("한글 글꼴이 없어 %s 의 한글이 네모로 표시됩니다 (NanumGothic 등 설치 권장)",
                       os.path.basename(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _pct_axis(ax, decimals: int = 0):
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=decimals))


def _shade_state(ax, regimes: pd.Series, state: int):
    """``state`` 인 연속 구간에 배경 wash 를 칠한다."""
    mask = (regimes == state).to_numpy()
    if not mask.any():
        return
    idx = regimes.index
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        right = idx[e] if e < len(idx) else idx[-1]
        ax.axvspan(idx[s], right, color=BEAR_WASH, lw=0)


def plot_regimes_cumret(strategy: pd.DataFrame, bear_state: int, path: str,
                        others: Optional[Dict[str, pd.DataFrame]] = None, title: str = "") -> str:
    """누적수익률 (전략 · 다른 전략 · Buy & Hold · 벤치마크) + bear 신호 구간 음영."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    _shade_state(ax, strategy["regime"], bear_state)
    ax.plot(strategy.index, strategy["cum_bh"], color=MUTED, lw=LW, label="Buy & Hold")
    if "cum_bench" in strategy.columns:
        ax.plot(strategy.index, strategy["cum_bench"], color=INK_2, lw=LW, label="Benchmark")
    for i, (name, other) in enumerate((others or {}).items(), start=1):
        cum = (1.0 + other["strat_ret"]).cumprod() - 1.0
        ax.plot(cum.index, cum, color=SERIES[i], lw=LW, label=name)
    ax.plot(strategy.index, strategy["cum_strategy"], color=SERIES[0], lw=LW + 0.4, label="JM strategy")
    last = strategy.index[-1]
    ax.annotate(f"{strategy['cum_strategy'].iloc[-1]:+.0%}", (last, strategy["cum_strategy"].iloc[-1]),
                xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    _pct_axis(ax)
    ax.set_ylabel("Cumulative return")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color=BEAR_WASH, label="Bear regime (signal)"))
    ax.legend(handles=handles, loc="upper left", ncol=3)
    ax.set_title(title or "Out-of-sample regimes and cumulative return")
    return _save(fig, path)


def plot_refit_params(params: pd.DataFrame, path: str) -> str:
    """재추정 시점별 상태 파라미터 (연율 수익률 · 변동성 · stay_prob) small multiples."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for (label, grp), color in zip(params.groupby("label", sort=False), SERIES):
        grp = grp.sort_values("refit_date")
        for ax, col in zip(axes, ["ann_ret", "ann_vol", "stay_prob"]):
            ax.step(grp["refit_date"], grp[col], where="post", color=color, lw=LW, label=label)
            ax.plot(grp["refit_date"], grp[col], "o", color=color, ms=3.5, mec=SURFACE, mew=1)
    for ax, title, dec in zip(axes, ["Annualized return by state", "Annualized volatility by state",
                                     "Stay probability by state"], [0, 0, 1]):
        ax.set_title(title)
        _pct_axis(ax, dec)
    axes[0].axhline(0, color=AXIS, lw=0.8)
    axes[0].legend(loc="upper left", ncol=4)
    return _save(fig, path)


def plot_weights(strategy: pd.DataFrame, path: str) -> str:
    """보유 위험자산 비중."""
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.fill_between(strategy.index, 0, strategy["weight"], step="post", color=SERIES[0], alpha=0.25, lw=0)
    ax.step(strategy.index, strategy["weight"], where="post", color=SERIES[0], lw=LW)
    ax.set_ylim(-0.02, 1.05)
    _pct_axis(ax)
    alt = "benchmark" if strategy.attrs.get("mode") == "relative" else "risk-free"
    ax.set_title(f"Risky-asset weight (remainder in {alt})")
    return _save(fig, path)


def plot_feat_weights(feat_weights: pd.DataFrame, category_shares: pd.DataFrame,
                      horizon_shares: pd.DataFrame, path: str) -> str:
    """sjm 피처 가중치: 피처별 heatmap + 변수 유형·기간별 가중 비중 누적 영역."""
    shares = (feat_weights.astype(float) ** 2)
    shares = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0)
    fig = plt.figure(figsize=(11, 9.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.6, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    im = ax0.imshow(shares.T.to_numpy(), aspect="auto", cmap=cmap, vmin=0, interpolation="nearest")
    ax0.set_yticks(range(shares.shape[1]), shares.columns, fontsize=7)
    ticks = np.linspace(0, len(shares) - 1, min(len(shares), 8)).astype(int)
    ax0.set_xticks(ticks, [shares.index[i].strftime("%Y-%m") for i in ticks])
    ax0.grid(False)
    ax0.set_title("Feature weight share per refit (sjm)")
    fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.01).ax.tick_params(labelsize=7, colors=MUTED)
    for pos, (df, title) in enumerate([(category_shares, "Weight share by variable type"),
                                       (horizon_shares, "Weight share by horizon")], start=1):
        ax = fig.add_subplot(gs[pos])
        df = df.fillna(0.0)
        # 마지막 재추정의 가중치가 보이도록 다음 반기까지 한 칸 연장
        df = pd.concat([df, df.iloc[[-1]].set_axis([df.index[-1] + pd.DateOffset(months=6)])])
        ax.stackplot(df.index, df.T.to_numpy(), labels=df.columns, colors=SERIES[:df.shape[1]],
                     edgecolor=SURFACE, linewidth=1.0, step="post")
        ax.set_ylim(0, 1)
        _pct_axis(ax)
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path)


def plot_episode_lengths(episodes: pd.DataFrame, path: str, current: Optional[pd.Series] = None,
                         state_label: str = "bear") -> str:
    """완료된 에피소드 길이 분포와 현재 에피소드 길이."""
    done = episodes[(episodes["label"] == state_label) & (~episodes["ongoing"].astype(bool))]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    if len(done):
        bins = np.unique(np.geomspace(1, max(2, done["n_days"].max() + 1), 25).astype(int))
        counts, _, _ = ax.hist(done["n_days"], bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0)
        ax.set_xscale("log")
        ax.set_ylim(0, counts.max() * 1.2)
    if current is not None and current["label"] == state_label:
        ax.axvline(current["n_days"], color=INK, lw=LW)
        ax.annotate(f"current: {int(current['n_days'])}d", (current["n_days"], ax.get_ylim()[1] * 0.95),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Episode length (trading days)")
    ax.set_ylabel("Episodes")
    ax.set_title(f"{state_label.capitalize()} episode lengths (n={len(done)})")
    return _save(fig, path)


def plot_similar_paths(paths: pd.DataFrame, current_id: int, path: str, label: str = "") -> str:
    """현재 에피소드와 유사한 과거 에피소드의 누적수익 경로 (현재 강조, 과거는 회색)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col in paths.columns:
        if col == current_id:
            continue
        s = paths[col].dropna()
        ax.plot(s.index, s, color=MUTED, lw=1.0, alpha=0.8)
        ax.annotate(f"#{col}", (s.index[-1], s.iloc[-1]), xytext=(3, 0), textcoords="offset points",
                    fontsize=7, color=INK_2, va="center")
    cur = paths[current_id].dropna()
    ax.plot(cur.index, cur, color=SERIES[0], lw=LW + 0.6)
    ax.plot(cur.index[-1], cur.iloc[-1], "o", color=SERIES[0], ms=6, mec=SURFACE, mew=2)
    ax.axhline(0, color=AXIS, lw=0.8)
    _pct_axis(ax)
    ax.set_xlabel("Days since episode start")
    ax.set_ylabel("Cumulative return")
    ax.legend(handles=[plt.Line2D([], [], color=SERIES[0], lw=LW + 0.6, label=f"Current episode #{current_id}"),
                       plt.Line2D([], [], color=MUTED, lw=1.0, label="Similar past episodes")], loc="lower left")
    ax.set_title(f"Similar {label} episodes: price paths".replace("  ", " "))
    return _save(fig, path)


def plot_delay_robustness(table: pd.DataFrame, path: str) -> str:
    """거래 지연별 Sharpe · MDD (두 개의 패널)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(table))
    for ax, col, title in [(axes[0], "sharpe", "Sharpe ratio by trade delay"),
                           (axes[1], "max_drawdown", "Max drawdown by trade delay")]:
        vals = table[col].astype(float)
        ax.bar(x, vals, color=SERIES[0], width=0.6)
        ax.set_xticks(x, [str(d) for d in table.index])
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.2f}" if col == "sharpe" else f"{v:.0%}", (xi, v), ha="center", fontsize=7,
                        color=INK_2, xytext=(0, 3 if v >= 0 else -10), textcoords="offset points")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.margins(y=0.12)
        ax.set_xlabel("Delay (trading days)")
        ax.set_title(title)
        if col == "max_drawdown":
            _pct_axis(ax)
    return _save(fig, path)


def plot_inference(regimes: pd.DataFrame, close: pd.Series, bear_state: int, path: str) -> str:
    """추론 모드: 현재 반기 가격과 bear 확률 (두 개의 패널)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, height_ratios=[2, 1])
    px = close.reindex(regimes.index)
    _shade_state(axes[0], regimes["regime"], bear_state)
    axes[0].plot(px.index, px, color=SERIES[0], lw=LW)
    axes[0].set_title("Current half-year: price and online regime")
    axes[0].legend(handles=[Patch(color=BEAR_WASH, label="Bear regime")], loc="upper left")
    prob = regimes[f"prob_{bear_state}"]
    axes[1].fill_between(prob.index, 0, prob, step="post", color=SERIES[1], alpha=0.25, lw=0)
    axes[1].step(prob.index, prob, where="post", color=SERIES[1], lw=LW)
    axes[1].set_ylim(0, 1.02)
    _pct_axis(axes[1])
    axes[1].set_title("Bear probability (online)")
    return _save(fig, path)
