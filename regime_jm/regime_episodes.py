"""6-b) 국면 에피소드 분석: bear 구간 · 유사 국면 · 종료 시나리오.

1. ``extract_episodes``       연속된 같은 국면을 에피소드로 자른다 (마지막 에피소드는 진행 중)
2. ``episode_metrics``        길이, 누적수익, MDD, 변동성, 오신호 여부
                              (bear 인데 누적수익 > 0, bull 인데 누적수익 < 0 → 오신호. 진행 중이면 판단 보류)
3. ``rank_similar_episodes``  현재(진행 중) 에피소드와 같은 국면의 과거 에피소드를 처음 L일
                              (L = 현재 길이) 의 누적수익·MDD·변동성 거리로 순위화
4. ``compare_episode_paths``  처음 L일 누적수익 경로의 RMSE 로 정렬
5. ``length_scenarios``       과거 에피소드 중 L일 이상 지속된 것들의 남은 기간 분포로 종료 시점 시나리오
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .rolling import state_label

PERIODS_PER_YEAR = 252
PATH_FEATURES = ("cum_ret", "mdd", "ann_vol")


def extract_episodes(regimes: pd.Series, state: Optional[int] = None, n_states: int = 2) -> pd.DataFrame:
    """국면 시계열 → 에피소드 표 (episode_id, state, label, start, end, n_days, ongoing)."""
    r = regimes.dropna().astype(int)
    if r.empty:
        return pd.DataFrame(columns=["episode_id", "state", "label", "start", "end", "n_days", "ongoing"])
    run = r.ne(r.shift()).cumsum().to_numpy()
    dates = pd.Series(r.index, index=r.index)
    g = dates.groupby(run)
    eps = pd.DataFrame({
        "state": r.groupby(run).first().to_numpy(),
        "start": g.first().to_numpy(),
        "end": g.last().to_numpy(),
        "n_days": g.size().to_numpy(),
    })
    eps.insert(0, "episode_id", np.arange(len(eps)))
    eps.insert(2, "label", [state_label(s, n_states) for s in eps["state"]])
    eps["ongoing"] = eps["end"] == r.index[-1]
    if state is not None:
        eps = eps[eps["state"] == state]
    return eps.reset_index(drop=True)


def _path_stats(r: pd.Series) -> dict:
    wealth = (1.0 + r).cumprod()
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth.to_numpy()]))[1:]
    return {
        "cum_ret": float(wealth.iloc[-1] - 1.0) if len(r) else np.nan,
        "mdd": float((wealth.to_numpy() / peak - 1.0).min()) if len(r) else np.nan,
        "ann_vol": float(r.std() * np.sqrt(PERIODS_PER_YEAR)) if len(r) > 1 else 0.0,
    }


def episode_returns(path_ret: pd.Series, ep) -> pd.Series:
    return path_ret.loc[ep["start"]:ep["end"]]


def episode_metrics(episodes: pd.DataFrame, path_ret: pd.Series, bear_state: int = 1) -> pd.DataFrame:
    """각 에피소드의 누적수익, MDD, 연율 변동성, 오신호 여부를 붙인다."""
    out = episodes.copy()
    stats = [_path_stats(episode_returns(path_ret, ep)) for _, ep in out.iterrows()]
    for key in PATH_FEATURES:
        out[key] = [s[key] for s in stats]
    false_sig = np.where(out["state"] == bear_state, out["cum_ret"] > 0, out["cum_ret"] < 0)
    out["false_signal"] = pd.array(false_sig, dtype="boolean")
    out.loc[out["ongoing"], "false_signal"] = pd.NA
    return out


def _current(episodes: pd.DataFrame, current_id: Optional[int]) -> pd.Series:
    if current_id is None:
        return episodes.iloc[-1]
    match = episodes[episodes["episode_id"] == current_id]
    if match.empty:
        raise KeyError(f"episode {current_id} not found")
    return match.iloc[0]


def _candidates(episodes: pd.DataFrame, current: pd.Series) -> pd.DataFrame:
    return episodes[(episodes["state"] == current["state"]) & (episodes["episode_id"] != current["episode_id"])
                    & (~episodes["ongoing"].astype(bool))]


def rank_similar_episodes(episodes: pd.DataFrame, path_ret: pd.Series, current_id: Optional[int] = None,
                          top_k: Optional[int] = 5, features: Sequence[str] = PATH_FEATURES,
                          min_overlap_frac: float = 0.5) -> pd.DataFrame:
    """현재 에피소드와 비슷한 과거 에피소드 (처음 L일 통계의 표준화 유클리드 거리 오름차순).

    길이가 L 의 ``min_overlap_frac`` 배보다 짧은 과거 에피소드는 비교 기간이 달라 제외한다.
    """
    cur = _current(episodes, current_id)
    L = int(cur["n_days"])
    cands = _candidates(episodes, cur)
    cands = cands[cands["n_days"] >= max(1, int(np.ceil(min_overlap_frac * L)))]
    if cands.empty:
        return pd.DataFrame()
    cur_stats = _path_stats(episode_returns(path_ret, cur))
    rows = []
    for _, ep in cands.iterrows():
        r = episode_returns(path_ret, ep).iloc[:L]
        s = _path_stats(r)
        rows.append({"episode_id": ep["episode_id"], "state": ep["state"], "label": ep["label"],
                     "start": ep["start"], "end": ep["end"], "n_days": ep["n_days"], "overlap": len(r),
                     **{f"{k}_first_L": s[k] for k in features},
                     "final_cum_ret": _path_stats(episode_returns(path_ret, ep))["cum_ret"],
                     "remaining_days": max(0, int(ep["n_days"]) - L)})
    df = pd.DataFrame(rows)
    M = df[[f"{k}_first_L" for k in features]].to_numpy(dtype=float)
    c = np.array([cur_stats[k] for k in features], dtype=float)
    scale = np.nanstd(np.vstack([M, c]), axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    df["distance"] = np.sqrt(np.nansum(((M - c) / scale) ** 2, axis=1))
    df = df.sort_values("distance").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.insert(1, "current_episode_id", cur["episode_id"])
    df.insert(2, "current_length", L)
    for k in features:
        df[f"current_{k}"] = cur_stats[k]
    return df.head(top_k) if top_k else df


def episode_paths(path_ret: pd.Series, episodes: pd.DataFrame, ids: Iterable[int],
                  max_len: Optional[int] = None) -> pd.DataFrame:
    """에피소드 시작 기준 누적수익 경로 (행 = 경과일 0..n, 열 = episode_id)."""
    eps = episodes.set_index("episode_id")
    paths = {}
    for i in ids:
        r = path_ret.loc[eps.loc[i, "start"]:eps.loc[i, "end"]]
        if max_len:
            r = r.iloc[:max_len]
        paths[i] = np.concatenate([[0.0], ((1.0 + r).cumprod() - 1.0).to_numpy()])
    out = pd.DataFrame({k: pd.Series(v) for k, v in paths.items()})
    out.index.name = "day"
    return out


def compare_episode_paths(path_ret: pd.Series, episodes: pd.DataFrame, current_id: Optional[int] = None,
                          candidate_ids: Optional[Iterable[int]] = None,
                          min_overlap_frac: float = 0.5) -> pd.DataFrame:
    """처음 L일 누적수익 경로의 RMSE 로 과거 에피소드를 정렬한다 (겹치는 기간이 L 의 절반 미만이면 제외)."""
    cur = _current(episodes, current_id)
    L = int(cur["n_days"])
    cands = _candidates(episodes, cur)
    if candidate_ids is not None:
        cands = cands[cands["episode_id"].isin(list(candidate_ids))]
    cur_path = episode_paths(path_ret, episodes, [cur["episode_id"]]).iloc[1:, 0].to_numpy()
    need = max(1, int(np.ceil(min_overlap_frac * L)))
    rows = []
    for _, ep in cands.iterrows():
        path = episode_paths(path_ret, episodes, [ep["episode_id"]], max_len=L).iloc[1:, 0].dropna().to_numpy()
        o = min(len(path), len(cur_path))
        if o < need:
            continue
        rmse = float(np.sqrt(np.mean((path[:o] - cur_path[:o]) ** 2)))
        rows.append({"episode_id": ep["episode_id"], "start": ep["start"], "end": ep["end"],
                     "n_days": ep["n_days"], "overlap": o, "rmse": rmse,
                     "cum_ret_at_overlap": path[o - 1], "current_cum_ret_at_overlap": cur_path[o - 1]})
    if not rows:
        return pd.DataFrame(columns=["episode_id", "start", "end", "n_days", "overlap", "rmse"])
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def length_scenarios(episodes: pd.DataFrame, current_id: Optional[int] = None,
                     similar_ids: Optional[Iterable[int]] = None,
                     quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
                     horizons: Sequence[int] = (5, 20, 60), asof=None) -> pd.DataFrame:
    """현재 에피소드의 종료 시나리오.

    과거 같은 국면 에피소드 중 현재 길이 L 이상 지속된 것들의 남은 기간(n_days - L) 분포를 쓴다.
    scenario='all' 은 전체, 'similar' 는 rank_similar_episodes 의 상위 에피소드만 사용한다.
    """
    cur = _current(episodes, current_id)
    L = int(cur["n_days"])
    asof = pd.Timestamp(asof if asof is not None else cur["end"])
    completed = _candidates(episodes, cur)
    scenarios = {"all": completed}
    if similar_ids is not None:
        scenarios["similar"] = completed[completed["episode_id"].isin(list(similar_ids))]
    rows = []
    for name, eps in scenarios.items():
        ref = eps[eps["n_days"] >= L]
        rem = (ref["n_days"] - L).to_numpy(dtype=float)
        row = {"scenario": name, "state": cur["state"], "label": cur["label"],
               "current_episode_id": cur["episode_id"], "current_start": cur["start"], "current_length": L,
               "asof": asof, "n_episodes": len(eps), "n_ref": len(ref)}
        if len(ref):
            row["mean_remaining"] = float(rem.mean())
            for q in quantiles:
                row[f"q{int(round(q * 100))}_remaining"] = float(np.quantile(rem, q))
            for h in horizons:
                row[f"p_end_within_{h}d"] = float(np.mean(rem <= h))
            row["median_end_date"] = asof + pd.offsets.BDay(int(round(np.median(rem))))
        rows.append(row)
    return pd.DataFrame(rows)
