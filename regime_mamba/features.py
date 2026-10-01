"""Mamba 입력 피처 세트 (paper / example / extra / none).

``regime_jm.features`` 의 피처 빌더를 그대로 써서 Jump Model 파이프라인(``run_pipeline.py``)과
같은 정의의 피처를 Mamba 에 넣는다. Mamba 가 압축한 hidden 벡터는 이후 ``ModifiedJumpModel``
(``jump_model: True``) 이 Jump Model 로 국면을 나눈다.

    feature_set: None     기존 동작. input_dim 3/4 → CSV 의 dd_10, sortino_20, sortino_60[, dollar_index]
    feature_set: paper    DD-log_10, sortino_20, sortino_60                                    (3개)
    feature_set: example  ret · DD-log · sortino × 5·20·60                                    (9개)
    feature_set: extra    ret · sortino · DD · std · var · mad · rms · vol-log · vol-chg × 5·20·60 (27개)
    feature_set: none     내장 피처 없음 (extra_feature_cols 만 사용)

피처는 ``feature_return_col`` (기본 ``returns``) 열로 계산하고, CSV 의 기존 열과 겹치지 않도록
``fs_`` 접두어를 붙여 데이터에 추가한다 (예: ``fs_sortino_20``). ``extra_feature_cols`` 로 지정한
CSV 열(예: ``dollar_index``)은 그 뒤에 그대로 붙는다. ``config.feature_cols`` 와 ``config.input_dim``
은 여기서 채워진다.

이 모듈은 torch 없이 import 할 수 있다.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from regime_jm.features import FEATURE_SETS, example_features, extra_features, paper_features

logger = logging.getLogger(__name__)

FEATURE_PREFIX = "fs_"
LEGACY_FEATURE_COLS = {
    3: ["dd_10", "sortino_20", "sortino_60"],
    4: ["dd_10", "sortino_20", "sortino_60", "dollar_index"],
}
_BUILDERS = {"paper": paper_features, "example": example_features, "extra": extra_features}


def uses_feature_set(config) -> bool:
    """``feature_set`` 이 지정되어 ``prepare_feature_set`` 으로 입력 열을 만드는지 여부."""
    return bool(getattr(config, "feature_set", None))


def get_feature_columns(config) -> List[str]:
    """Mamba 입력 열. ``prepare_feature_set`` 이 채운 ``config.feature_cols`` 가 있으면 그것을 쓴다."""
    cols = getattr(config, "feature_cols", None)
    if cols:
        return list(cols)
    if uses_feature_set(config):
        raise ValueError("feature_set 이 지정되었지만 prepare_feature_set() 이 아직 호출되지 않았습니다")
    if config.input_dim not in LEGACY_FEATURE_COLS:
        raise ValueError(f"feature_set 을 지정하지 않으면 input_dim 은 3 또는 4 여야 합니다 (got {config.input_dim})")
    return list(LEGACY_FEATURE_COLS[config.input_dim])


def returns_in_percent(config) -> bool:
    """수익률 열이 % 단위인지. ``returns_pct`` 가 없으면 기존 규칙(input_dim == 4)을 따른다."""
    if config is None:
        return False
    flag = getattr(config, "returns_pct", None)
    if flag is None:
        return config.input_dim == 4
    return bool(flag)


def prepare_feature_set(data: pd.DataFrame, config, log: Optional[logging.Logger] = None) -> pd.DataFrame:
    """``config.feature_set`` 피처를 계산해 데이터에 추가하고 ``feature_cols`` / ``input_dim`` 을 설정한다.

    ``feature_set`` 이 없으면 데이터와 config 를 그대로 둔다 (기존 동작).
    모든 피처는 t일까지의 수익률만 쓰는 EWM / trailing rolling 통계라서 인과적이다.
    앞의 ``feature_warmup`` 행(통계 안정화 구간)은 버린다.
    """
    log = log or logger
    feature_set = getattr(config, "feature_set", None)
    if not feature_set:
        return data
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
    if getattr(config, "lstm", False):
        raise ValueError("feature_set 은 Mamba 전용입니다 (LSTM 은 입력 열이 고정되어 있습니다)")

    ret_col = getattr(config, "feature_return_col", None) or "returns"
    extra_cols = list(getattr(config, "extra_feature_cols", None) or [])
    missing = [c for c in [ret_col, "Date", *extra_cols] if c not in data.columns]
    if missing:
        raise ValueError(f"데이터에 열이 없습니다: {missing}")

    data = data.sort_values("Date", kind="stable").reset_index(drop=True)
    ret = pd.to_numeric(data[ret_col], errors="coerce")

    cols: List[str] = []
    out = data.copy()
    if feature_set != "none":
        # 수익률 결측일은 직전 값을 이어 쓴다 (인과적)
        feats = _BUILDERS[feature_set](ret.dropna()).reindex(data.index).ffill()
        feats.columns = [f"{FEATURE_PREFIX}{c}" for c in feats.columns]
        dup = set(feats.columns) & set(data.columns)
        if dup:
            raise ValueError(f"피처 이름이 데이터의 기존 열과 겹칩니다: {sorted(dup)}")
        out = pd.concat([out, feats], axis=1)
        cols.extend(feats.columns)
    cols.extend(extra_cols)
    if not cols:
        raise ValueError("피처가 없습니다: feature_set none 이면 extra_feature_cols 가 필요합니다")

    warmup = int(getattr(config, "feature_warmup", 0) or 0)
    if warmup:
        first_valid = ret.first_valid_index()
        start = (first_valid if first_valid is not None else 0) + warmup
        out = out.iloc[start:].reset_index(drop=True)

    config.feature_cols = cols
    config.input_dim = len(cols)
    if getattr(config, "returns_pct", None) is None:
        # 일간 수익률 표준편차: 소수 단위 ~0.01, % 단위 ~1
        config.returns_pct = bool(ret.std() > 0.2)
        log.info("returns_pct 추정: %s (%s 표준편차 %.4f)", config.returns_pct, ret_col, ret.std())

    log.info("feature_set=%s, input_dim=%d, warmup=%d, rows=%d (%s ~ %s)", feature_set, config.input_dim,
             warmup, len(out), out["Date"].iloc[0] if len(out) else "-", out["Date"].iloc[-1] if len(out) else "-")
    log.info("Mamba 입력 열: %s", cols)
    return out


def standardize_for_window(data: pd.DataFrame, config, train_start: str, train_end: str) -> pd.DataFrame:
    """윈도우 학습 구간의 평균·표준편차로 Mamba 입력 열을 표준화한 사본을 돌려준다.

    ``feature_set`` 모드에서만 적용된다 (기존 input_dim 3/4 모드는 그대로).
    통계는 학습 구간에서만 구하므로 검증·예측 구간의 정보가 섞이지 않는다.
    """
    if not uses_feature_set(config) or not getattr(config, "standardize_features", True):
        return data
    cols = get_feature_columns(config)
    mask = (data["Date"] >= train_start) & (data["Date"] <= train_end)
    if not mask.any():
        raise ValueError(f"표준화할 학습 구간 데이터가 없습니다: {train_start} ~ {train_end}")
    train = data.loc[mask, cols]
    mean = train.mean()
    std = train.std().where(lambda s: s > 0, 1.0).fillna(1.0)
    out = data.copy()
    out[cols] = (data[cols] - mean.fillna(0.0)) / std
    return out
