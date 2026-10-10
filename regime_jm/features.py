"""2) 피처 생성: EWM downside deviation · Sortino (+ extra / 커스텀 변수).

모든 피처는 t일까지의 수익률만 쓰는 EWM / trailing rolling 통계라서 인과적이다.

피처 세트 (hl = EWM halflife, w = rolling window, 기본 5·20·60일)
    paper    논문 Table 2: DD-log_10, sortino_20, sortino_60
    example  ret_hl(EWM 평균), DD-log_hl(EWM downside deviation 의 로그), sortino_hl          (9개)
    extra    ret_hl, sortino_hl, DD_hl, std_w, var_w, mad_w, rms_w, vol-log_hl, vol-chg_hl   (27개)
    none     내장 피처 없음 (--extra-features 만 사용)

피처 이름은 ``계열_기간`` (예: ``sortino_20``) 형식이고, 사용자 변수는 ``열[변환]`` 형식이다.
``--remove-series`` 는 계열 이름(예: ``var``, ``DD-log``) 또는 피처 이름을 받는다.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-8
HALFLIFES = (5, 20, 60)
FEATURE_SETS = ("paper", "example", "extra", "none")

# 계열 → 변수 유형 (weights.py 에서 가중 비중을 묶을 때 사용)
FAMILY_CATEGORY = {
    "ret": "return",
    "sortino": "risk-adjusted",
    "DD": "downside",
    "DD-log": "downside",
    "std": "volatility",
    "var": "volatility",
    "mad": "volatility",
    "rms": "volatility",
    "vol-log": "volatility",
    "vol-chg": "volatility-change",
}
_BUILTIN_RE = re.compile(r"^(?P<family>[A-Za-z][A-Za-z\-]*)_(?P<horizon>\d+)$")


# ---------------------------------------------------------------------------
# 기본 통계
# ---------------------------------------------------------------------------

def ewm_mean(ret: pd.Series, hl: float) -> pd.Series:
    return ret.ewm(halflife=hl).mean()


def ewm_downside_dev(ret: pd.Series, hl: float) -> pd.Series:
    """EWM downside deviation: sqrt(EWM[min(r, 0)^2])."""
    neg = ret.clip(upper=0.0)
    return np.sqrt((neg ** 2).ewm(halflife=hl).mean()).clip(lower=EPS)


def ewm_sortino(ret: pd.Series, hl: float) -> pd.Series:
    return ewm_mean(ret, hl) / ewm_downside_dev(ret, hl)


def ewm_vol(ret: pd.Series, hl: float) -> pd.Series:
    """EWMA 변동성 (RiskMetrics 방식, 평균 차감 없음): sqrt(EWM[r^2])."""
    return np.sqrt((ret ** 2).ewm(halflife=hl).mean()).clip(lower=EPS)


# ---------------------------------------------------------------------------
# 피처 세트
# ---------------------------------------------------------------------------

def paper_features(ret: pd.Series) -> pd.DataFrame:
    """논문 Table 2 의 3개 피처."""
    return pd.DataFrame({
        "DD-log_10": np.log(ewm_downside_dev(ret, 10)),
        "sortino_20": ewm_sortino(ret, 20),
        "sortino_60": ewm_sortino(ret, 60),
    }, index=ret.index)


def example_features(ret: pd.Series, halflifes: Sequence[int] = HALFLIFES) -> pd.DataFrame:
    feats = {}
    for hl in halflifes:
        feats[f"ret_{hl}"] = ewm_mean(ret, hl)
    for hl in halflifes:
        feats[f"DD-log_{hl}"] = np.log(ewm_downside_dev(ret, hl))
    for hl in halflifes:
        feats[f"sortino_{hl}"] = ewm_sortino(ret, hl)
    return pd.DataFrame(feats, index=ret.index)


def extra_features(ret: pd.Series, halflifes: Sequence[int] = HALFLIFES) -> pd.DataFrame:
    """example 의 ret·sortino + DD(로그 없이) + 롤링 분산/절대값 계열 + EWMA 변동성 계열."""
    feats = {}
    for hl in halflifes:
        feats[f"ret_{hl}"] = ewm_mean(ret, hl)
    for hl in halflifes:
        feats[f"sortino_{hl}"] = ewm_sortino(ret, hl)
    for hl in halflifes:
        feats[f"DD_{hl}"] = ewm_downside_dev(ret, hl)
    for w in halflifes:
        feats[f"std_{w}"] = ret.rolling(w).std()
    for w in halflifes:
        feats[f"var_{w}"] = ret.rolling(w).var()
    for w in halflifes:
        feats[f"mad_{w}"] = ret.abs().rolling(w).mean()
    for w in halflifes:
        feats[f"rms_{w}"] = np.sqrt((ret ** 2).rolling(w).mean())
    vol_log = {hl: np.log(ewm_vol(ret, hl)) for hl in halflifes}
    for hl in halflifes:
        feats[f"vol-log_{hl}"] = vol_log[hl]
    for hl in halflifes:
        feats[f"vol-chg_{hl}"] = vol_log[hl] - vol_log[hl].shift(hl)
    return pd.DataFrame(feats, index=ret.index)


_BUILDERS = {"paper": paper_features, "example": example_features, "extra": extra_features}


# ---------------------------------------------------------------------------
# 사용자 변수 변환
# ---------------------------------------------------------------------------

_TRANSFORM_RE = re.compile(r"^(?P<op>none|level|log|diff|pct|logdiff|zscore|ewm|lag)(?:_(?P<n>\d+))?$")
_TRANSFORM_DEFAULT_N = {"diff": 1, "pct": 1, "logdiff": 1, "zscore": 252, "ewm": 20, "lag": 1}


def is_valid_transform(transform: str) -> bool:
    return all(_TRANSFORM_RE.match(step) for step in str(transform).split("+"))


def apply_transform(s: pd.Series, transform: str = "none") -> pd.Series:
    """사용자 변수 변환. ``+`` 로 이어 붙여 순서대로 적용할 수 있다 (예: ``logdiff+ewm_20``).

    none/level  그대로          log        log(x)
    diff[_n]    x - x(-n)       pct[_n]    x / x(-n) - 1
    logdiff[_n] log 차분        zscore[_w] trailing w 기간 z-score (기본 252)
    ewm[_hl]    EWM 평균 (기본 20)          lag[_n]    n 기간 지연 (발표 시차 반영용, 기본 1)

    모두 과거 값만 사용한다. 변환은 원래 관측 주기에서 적용된다.
    """
    out = s.astype(float)
    for step in str(transform).split("+"):
        m = _TRANSFORM_RE.match(step)
        if not m:
            raise ValueError(f"unknown transform {step!r} (none, log, diff_n, pct_n, logdiff_n, zscore_w, ewm_hl, lag_n)")
        op = m.group("op")
        n = int(m.group("n")) if m.group("n") else _TRANSFORM_DEFAULT_N.get(op)
        if op in ("none", "level"):
            continue
        if op in ("log", "logdiff") and (out <= 0).any():
            raise ValueError(f"'{s.name}' 에 0 이하 값이 있어 {op} 변환을 할 수 없습니다")
        if op == "log":
            out = np.log(out)
        elif op == "diff":
            out = out.diff(n)
        elif op == "pct":
            out = out.pct_change(n, fill_method=None)
        elif op == "logdiff":
            out = np.log(out).diff(n)
        elif op == "zscore":
            roll = out.rolling(n, min_periods=max(2, n // 2))
            out = (out - roll.mean()) / roll.std().replace(0.0, np.nan)
        elif op == "ewm":
            out = out.ewm(halflife=n).mean()
        elif op == "lag":
            out = out.shift(n)
    return out


# ---------------------------------------------------------------------------
# 피처 메타 / 계열 제거
# ---------------------------------------------------------------------------

def feature_family(name: str) -> str:
    m = _BUILTIN_RE.match(name)
    if m and m.group("family") in FAMILY_CATEGORY:
        return m.group("family")
    return name.split("[", 1)[0]


def feature_meta(names: Iterable[str]) -> pd.DataFrame:
    """피처별 계열(family), 기간(horizon), 변수 유형(category). 사용자 변수는 category='custom'."""
    rows = []
    for name in names:
        m = _BUILTIN_RE.match(name)
        if m and m.group("family") in FAMILY_CATEGORY:
            fam = m.group("family")
            rows.append((name, fam, int(m.group("horizon")), FAMILY_CATEGORY[fam]))
        else:
            rows.append((name, feature_family(name), np.nan, "custom"))
    return pd.DataFrame(rows, columns=["feature", "family", "horizon", "category"]).set_index("feature")


def remove_series(X: pd.DataFrame, names: Optional[Iterable[str]]) -> pd.DataFrame:
    """계열 이름(대소문자 무시) 또는 정확한 피처 이름으로 열을 제거한다. 콤마로 여러 개 가능."""
    if not names:
        return X
    targets = [t.strip() for n in names for t in str(n).split(",") if t.strip()]
    drop = []
    for t in targets:
        hit = [c for c in X.columns if c == t or feature_family(c).lower() == t.lower()]
        if not hit:
            logger.warning("--remove-series '%s' 에 해당하는 피처가 없습니다", t)
        drop.extend(hit)
    return X.drop(columns=sorted(set(drop), key=list(X.columns).index))


def expand_feature_names(columns: Sequence[str], names: Optional[Iterable[str]]) -> list:
    """피처/계열 이름 목록을 실제 피처 이름으로 펼친다 (--pin-features 용)."""
    if not names:
        return []
    out = []
    for n in names:
        for t in str(n).split(","):
            t = t.strip()
            if not t:
                continue
            hit = [c for c in columns if c == t or feature_family(c).lower() == t.lower()]
            if not hit:
                raise KeyError(f"피처 '{t}' 가 없습니다. 사용 가능한 피처: {list(columns)}")
            out.extend(h for h in hit if h not in out)
    return out


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def build_features(ret: pd.Series, feature_set: str = "paper", extra: Optional[pd.DataFrame] = None,
                   remove: Optional[Iterable[str]] = None, warmup: int = 252, return_context: bool = False):
    """수익률 → 피처 행렬.

    Args:
        ret: 신호 수익률 (ret 또는 rel_ret)
        feature_set: paper / example / extra / none
        extra: 이미 변환·정렬된 사용자 변수 (index = ret.index)
        remove: 제거할 계열 / 피처 이름
        warmup: 앞에서 버릴 행 수 (EWM·롤링 통계 안정화용)
        return_context: True 면 (X, context) 를 돌려준다. context 는 X 첫 행 직전까지 버려진 행 중
            결측 없이 이어지는 뒤쪽 부분 (Mamba 시퀀스 앞부분을 0 대신 채우는 용도, --mamba-warmup-context)
    """
    ret = ret.dropna()
    parts = []
    if feature_set != "none":
        if feature_set not in _BUILDERS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")
        parts.append(_BUILDERS[feature_set](ret))
    if extra is not None and extra.shape[1]:
        dup = set(extra.columns) & set(parts[0].columns) if parts else set()
        if dup:
            raise ValueError(f"사용자 변수 이름이 내장 피처와 겹칩니다: {sorted(dup)}")
        parts.append(extra.reindex(ret.index))
    if not parts:
        raise ValueError("피처가 없습니다: --feature-set none 이면 --extra-features 가 필요합니다")

    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    X = remove_series(X, remove)
    if X.shape[1] == 0:
        raise ValueError("--remove-series 로 모든 피처가 제거되었습니다")

    # 중간 결측은 직전 값으로 채우고(과거 값만 사용), 앞쪽 결측 행만 버린다
    filled = X.ffill()
    n_filled = int((X.isna() & filled.notna()).sum().sum())
    if n_filled:
        logger.warning("피처 중간 결측 %d개를 직전 값으로 채웠습니다", n_filled)
    X = filled.iloc[warmup:]
    n_before = len(X)
    X = X.dropna()
    if len(X) < n_before:
        logger.warning("결측 피처가 있는 앞쪽 %d행을 제외했습니다 (사용자 변수 시작일 등)", n_before - len(X))
    if len(X) == 0:
        raise ValueError("warmup 이후 남은 피처 행이 없습니다")
    if not return_context:
        return X
    context = filled.loc[filled.index < X.index[0]]
    bad = context.isna().any(axis=1).to_numpy()
    if bad.any():
        context = context.iloc[np.flatnonzero(bad)[-1] + 1:]
    return X, context
