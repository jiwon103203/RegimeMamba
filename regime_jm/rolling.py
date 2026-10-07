"""3) 6개월마다 재추정 + 재추정 사이 구간은 온라인 추론 (파이프라인의 핵심).

재추정 시점 d (1·7월 첫 영업일) 마다:
  1. d 이전 최대 ``train_window`` 행(최소 ``min_train``)만으로 DataClipperStd(3σ)·StandardScaler 를 fit
  2. ``init_model`` 로 모델을 만들고 ``fit(..., sort_by="cumret")`` → 상태 0 = bull, 마지막 상태 = bear
  3. [학습창 + 다음 재추정 전까지 구간] 을 이어 ``predict_proba_online`` 을 돌린 뒤 학습창 행을 잘라낸다.
     온라인 추론의 t 행 결과는 t 행까지의 피처만 쓰므로 해당 반기 구간이 인과적으로 추론된다.
  4. 재추정 파라미터(중심점, 연율 수익률·변동성, stay_prob)와 sjm 피처 가중치를 기록한다.

``encoder`` (예: ``mamba_encoder.MambaEncoder``) 를 주면 1 이전에 [학습창 + 구간] 피처를 그 창에서 학습한
표현(hidden 벡터)으로 바꾼 뒤 같은 절차로 Jump Model 을 적합한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD

from .sparse_pin import PinnedSparseJumpModel

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252
MODELS = ("jm", "sjm")


@dataclass
class RollingJMResult:
    """롤링 재추정 결과.

    regimes        날짜별 온라인 추론 결과: regime, prob_0.., refit_date
    params         재추정 × 상태별 파라미터: ann_ret, ann_vol, stay_prob, freq, center_<피처>
    feat_weights   (sjm) 재추정 × 피처 가중치 (jumpmodels 의 feat_weights = sqrt(w))
    insample_last  마지막 학습창의 in-sample 적합 결과: label, prob_.., signal_ret
    """
    regimes: pd.DataFrame
    params: pd.DataFrame
    feat_weights: Optional[pd.DataFrame]
    insample_last: pd.DataFrame
    n_states: int = 2
    refit_dates: List[pd.Timestamp] = field(default_factory=list)
    last_model: Any = None

    @property
    def bear_state(self) -> int:
        return self.n_states - 1


def state_label(state: int, n_states: int) -> str:
    if state == 0:
        return "bull"
    if state == n_states - 1:
        return "bear"
    return f"state{state}"


def refit_schedule(index: pd.DatetimeIndex, months: Sequence[int] = (1, 7), min_train: int = 500,
                   start=None, end=None) -> List[pd.Timestamp]:
    """각 ``months`` 의 첫 거래일 중, 이전 행이 ``min_train`` 개 이상인 날짜."""
    idx = pd.DatetimeIndex(index)
    first_days = pd.Series(idx, index=idx).groupby(idx.to_period("M")).min()
    out = []
    for period, day in first_days.items():
        if period.month not in months:
            continue
        pos = idx.get_loc(day)
        if pos < min_train:  # 데이터 첫 달(월 중간 시작일 수 있음)도 여기서 걸러진다
            continue
        if start is not None and day < pd.Timestamp(start):
            continue
        if end is not None and day > pd.Timestamp(end):
            continue
        out.append(day)
    return out


def default_max_feats(n_features: int) -> float:
    """sjm 의 유효 피처 수 기본값: 전체의 1/3 (최소 2, 최대 전체)."""
    return float(min(n_features, max(2.0, n_features / 3.0)))


def init_model(model: str = "jm", *, n_states: int = 2, jump_penalty: float = 50.0, cont: bool = True,
               n_features: Optional[int] = None, max_feats: Optional[float] = None,
               pinned: Optional[Iterable[str]] = None, grid_size: float = 0.05, n_init: int = 10,
               random_state: int = 0):
    """jm (cont=True 이면 CJM) 또는 sjm (Pinned SJM) 인스턴스."""
    if model == "jm":
        if pinned:
            raise ValueError("--pin-features 는 --model sjm 에서만 쓸 수 있습니다")
        return JumpModel(n_components=n_states, jump_penalty=jump_penalty, cont=cont, grid_size=grid_size,
                         n_init=n_init, random_state=random_state)
    if model == "sjm":
        if max_feats is None:
            if n_features is None:
                raise ValueError("sjm 기본 max_feats 계산에 n_features 가 필요합니다")
            max_feats = default_max_feats(n_features)
        return PinnedSparseJumpModel(n_components=n_states, max_feats=max_feats, jump_penalty=jump_penalty,
                                     cont=cont, grid_size=grid_size, n_init_jm=n_init, random_state=random_state,
                                     pinned=list(pinned) if pinned else None)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


class WindowPreprocessor:
    """학습창에만 fit 하는 3σ 클리핑 + 표준화."""

    def __init__(self, clip_mul: float = 3.0):
        self.clip_mul = clip_mul

    def fit(self, X: pd.DataFrame) -> "WindowPreprocessor":
        self.clipper = DataClipperStd(mul=self.clip_mul).fit(X)
        self.scaler = StandardScalerPD().fit(self.clipper.transform(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(self.clipper.transform(X))


def _inner_jm(model) -> JumpModel:
    return getattr(model, "jm_ins", model)


def _stay_probs(labels: np.ndarray, n_states: int) -> np.ndarray:
    labels = np.asarray(labels)
    out = np.full(n_states, np.nan)
    for k in range(n_states):
        cur = labels[:-1] == k
        if cur.any():
            out[k] = float(np.mean(labels[1:][cur] == k))
    return out


def fit_window(X_train: pd.DataFrame, y_train: pd.Series, *, model: str = "jm", clip_mul: float = 3.0,
               **model_kwargs) -> Tuple[Any, WindowPreprocessor]:
    """학습창 하나에 전처리 + 모델을 적합한다."""
    prep = WindowPreprocessor(clip_mul).fit(X_train)
    m = init_model(model, n_features=X_train.shape[1], **model_kwargs)
    m.fit(prep.transform(X_train), ret_ser=y_train, sort_by="cumret")
    return m, prep


def infer_online(model, prep: WindowPreprocessor, X_train: pd.DataFrame, X_seg: pd.DataFrame) -> pd.DataFrame:
    """[학습창 + 구간] 에 온라인 추론을 돌리고 구간 행만 돌려준다."""
    X_all = pd.concat([X_train, X_seg])
    proba = model.predict_proba_online(prep.transform(X_all))
    proba = pd.DataFrame(np.asarray(proba), index=X_all.index)
    return proba.iloc[len(X_train):]


def center_distances(model, prep: WindowPreprocessor, X: pd.DataFrame) -> pd.DataFrame:
    """각 행과 상태별 중심점의 유클리드 거리 ``dist_0..`` (모델이 손실을 재는 공간과 같다).

    학습창 기준 클리핑·표준화를 적용한 공간이며, sjm 이면 피처와 중심점 모두 feat_weights 가 곱해진 공간이다.
    jumpmodels 의 손실은 0.5 * dist**2 이다. t 행의 거리는 t 행 피처와 재추정 모델만 쓰므로 인과적이다.
    """
    Z = prep.transform(X).to_numpy(dtype=float)
    fw = getattr(model, "feat_weights", None)
    if fw is not None:
        Z = Z * np.asarray(fw, dtype=float)
    centers = np.asarray(model.centers_, dtype=float)  # sjm 은 이미 가중치가 곱해진 중심점
    dist = np.sqrt(((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
    return pd.DataFrame(dist, index=X.index, columns=[f"dist_{k}" for k in range(centers.shape[0])])


def window_params(model, prep: WindowPreprocessor, X_train: pd.DataFrame, n_states: int) -> pd.DataFrame:
    """상태별 파라미터: 연율 수익률·변동성, stay_prob, 비중, 표준화 공간의 중심점."""
    jm = _inner_jm(model)
    labels = np.asarray(jm.labels_)
    ret_ = np.asarray(getattr(model, "ret_", np.full(n_states, np.nan)), dtype=float)
    vol_ = np.asarray(getattr(model, "vol_", np.full(n_states, np.nan)), dtype=float)
    centers = np.asarray(model.centers_, dtype=float)
    fw = getattr(model, "feat_weights", None)
    if fw is not None:  # sjm 의 centers_ 는 가중치가 곱해져 있다
        fw = np.asarray(fw, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            centers = np.where(fw > 0, centers / fw, np.nan)
    stay = _stay_probs(labels, n_states)
    rows = []
    for k in range(n_states):
        row = {
            "state": k,
            "label": state_label(k, n_states),
            "ann_ret": ret_[k] * PERIODS_PER_YEAR,
            "ann_vol": vol_[k] * np.sqrt(PERIODS_PER_YEAR),
            "stay_prob": stay[k],
            "freq": float(np.mean(labels == k)),
        }
        row.update({f"center_{c}": centers[k, j] for j, c in enumerate(X_train.columns)})
        rows.append(row)
    return pd.DataFrame(rows)


def run_rolling_jm(features: pd.DataFrame, signal_ret: pd.Series, *, model: str = "jm", cont: bool = True,
                   n_states: int = 2, jump_penalty: float = 50.0, max_feats: Optional[float] = None,
                   pinned: Optional[Iterable[str]] = None, train_window: int = 3000, min_train: int = 500,
                   refit_months: Sequence[int] = (1, 7), start=None, end=None, clip_mul: float = 3.0,
                   grid_size: float = 0.05, n_init: int = 10, random_state: int = 0,
                   encoder: Optional[Callable[..., pd.DataFrame]] = None,
                   center_distance: bool = False) -> RollingJMResult:
    """6개월 재추정 + 온라인 추론으로 날짜별 국면을 만든다 (모듈 docstring 참고).

    Args:
        features: build_features 결과
        signal_ret: 모델이 학습할 수익률 (상태 정렬 sort_by='cumret' 에 사용)
        start: 이 날짜 이후의 재추정만 수행 (학습에는 그 이전 데이터도 사용)
        end: 이 날짜까지의 피처만 사용
        encoder: ``encoder(X, y, lo, pos, seg_end)`` → X.iloc[lo:seg_end] 행의 새 피처.
            pos 이전 행으로만 학습해야 한다 (인과성)
        center_distance: True 면 regimes 에 상태별 중심점과의 거리 ``dist_0..`` 를 붙인다 (``center_distances``)
    """
    if train_window < min_train:
        raise ValueError("train_window 는 min_train 이상이어야 합니다")
    X = features if end is None else features.loc[:pd.Timestamp(end)]
    y = signal_ret.reindex(X.index)
    if y.isna().any():
        raise ValueError("signal_ret 에 피처 날짜의 결측값이 있습니다")

    dates = refit_schedule(X.index, refit_months, min_train, start=start)
    if not dates:
        raise ValueError(f"재추정할 수 있는 날짜가 없습니다: 첫 {'/'.join(map(str, refit_months))}월 첫 영업일 이전에 "
                         f"최소 {min_train}행이 필요합니다 (피처 {len(X)}행)")

    model_kwargs = dict(model=model, n_states=n_states, jump_penalty=jump_penalty, cont=cont,
                        max_feats=max_feats, pinned=pinned, grid_size=grid_size, n_init=n_init,
                        random_state=random_state)
    regime_parts, param_parts, weight_rows = [], [], []
    m = prep = X_tr = y_tr = None
    for i, d in enumerate(dates):
        pos = X.index.get_loc(d)
        lo = max(0, pos - train_window)
        X_tr, y_tr = X.iloc[lo:pos], y.iloc[lo:pos]
        seg_end = X.index.get_loc(dates[i + 1]) if i + 1 < len(dates) else len(X)
        X_seg = X.iloc[pos:seg_end]
        if encoder is not None:
            H = encoder(X, y, lo, pos, seg_end)
            X_tr, X_seg = H.iloc[:pos - lo], H.iloc[pos - lo:]

        m, prep = fit_window(X_tr, y_tr, clip_mul=clip_mul, **model_kwargs)
        proba = infer_online(m, prep, X_tr, X_seg)
        seg = pd.DataFrame({"regime": proba.to_numpy().argmax(axis=1)}, index=proba.index)
        for k in range(n_states):
            seg[f"prob_{k}"] = proba[k].to_numpy()
        seg["refit_date"] = d
        if center_distance:
            seg = seg.join(center_distances(m, prep, X_seg))
        regime_parts.append(seg)

        params = window_params(m, prep, X_tr, n_states)
        params.insert(0, "refit_date", d)
        params.insert(1, "train_start", X_tr.index[0])
        params.insert(2, "train_end", X_tr.index[-1])
        params.insert(3, "n_train", len(X_tr))
        param_parts.append(params)

        if model == "sjm":
            weight_rows.append(pd.Series(np.asarray(m.feat_weights, dtype=float), index=X_tr.columns, name=d))
        logger.info("refit %s: train %s~%s (%d), infer %d days", d.date(), X_tr.index[0].date(),
                    X_tr.index[-1].date(), len(X_tr), len(X_seg))

    regimes = pd.concat(regime_parts)
    regimes.index.name = "date"
    feat_weights = None
    if weight_rows:
        feat_weights = pd.DataFrame(weight_rows)
        feat_weights.index.name = "refit_date"

    jm = _inner_jm(m)
    proba_in = np.asarray(jm.proba_)
    insample = pd.DataFrame({"label": np.asarray(jm.labels_)}, index=X_tr.index)
    for k in range(n_states):
        insample[f"prob_{k}"] = proba_in[:, k]
    insample["signal_ret"] = y_tr

    return RollingJMResult(regimes=regimes, params=pd.concat(param_parts, ignore_index=True),
                           feat_weights=feat_weights, insample_last=insample, n_states=n_states,
                           refit_dates=list(dates), last_model=m)
