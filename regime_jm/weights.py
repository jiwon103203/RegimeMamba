"""6-a) sjm 피처 가중치를 변수 유형·계열·기간별로 묶은 비중 — "지금 무엇이 국면을 가르는가".

jumpmodels 의 ``feat_weights`` 는 sqrt(w) 이고 가중 거리에서 피처 j 의 기여는 w_j = feat_weights_j^2 이다.
비중은 share_j = w_j / Σ w 로 정의한다 (행 합 = 1).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .features import feature_meta

GROUP_BY = ("category", "family", "horizon")


def weight_shares(feat_weights: pd.DataFrame) -> pd.DataFrame:
    """재추정 × 피처 가중 비중 (행 합 1)."""
    w = feat_weights.astype(float) ** 2
    total = w.sum(axis=1).replace(0.0, np.nan)
    return w.div(total, axis=0)


def _group_labels(columns: Iterable[str], by: str) -> pd.Series:
    meta = feature_meta(columns)
    if by == "horizon":
        return meta["horizon"].map(lambda h: "custom" if pd.isna(h) else f"{int(h)}d")
    if by in ("family", "category"):
        return meta[by]
    raise ValueError(f"by must be one of {GROUP_BY}")


def group_weights(feat_weights: pd.DataFrame, by: str = "category") -> pd.DataFrame:
    """재추정 × 그룹 가중 비중."""
    shares = weight_shares(feat_weights)
    labels = _group_labels(shares.columns, by)
    grouped = shares.T.groupby(labels.reindex(shares.columns).to_numpy(), sort=False).sum().T
    grouped.index.name = feat_weights.index.name
    return grouped


def weight_groups_long(feat_weights: pd.DataFrame, by: Iterable[str] = GROUP_BY) -> pd.DataFrame:
    """weight_groups.csv 용 long 포맷: refit_date, group_by, group, share."""
    parts = []
    for b in by:
        g = group_weights(feat_weights, b)
        long = g.stack().rename("share").reset_index()
        long.columns = ["refit_date", "group", "share"]
        long.insert(1, "group_by", b)
        parts.append(long)
    return pd.concat(parts, ignore_index=True)


def current_weight_summary(feat_weights: pd.DataFrame, by: Iterable[str] = GROUP_BY) -> pd.DataFrame:
    """가장 최근 재추정의 그룹별 비중 (비중 내림차순)."""
    last = feat_weights.iloc[[-1]]
    rows = []
    for b in by:
        g = group_weights(last, b).iloc[0].sort_values(ascending=False)
        rows.extend({"refit_date": last.index[0], "group_by": b, "group": k, "share": v} for k, v in g.items())
    return pd.DataFrame(rows)
