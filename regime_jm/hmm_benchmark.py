"""5) HMM 벤치마크: 같은 수익률 시리즈에 Gaussian HMM 을 롤링으로 적합한다.

- ``refit_every`` 거래일(기본 21일)마다 직전 최대 ``train_window`` 일로 재적합한다.
- 추론은 forward filter (p(s_t | r_1..r_t)) 라서 인과적이다. hmmlearn 의 predict_proba 는
  전체 구간을 보는 smoother 이므로 쓰지 않는다.
- 평균 수익률이 높은 상태를 bull(0), 낮은 상태를 bear(마지막)로 정렬한다.
- 과거 구간만 보는 trailing median filter 로 국면을 평활화한다.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

logger = logging.getLogger(__name__)

RET_SCALE = 100.0  # 수치 안정성을 위해 % 단위로 적합


def forward_filter(x: np.ndarray, startprob: np.ndarray, transmat: np.ndarray, means: np.ndarray,
                   variances: np.ndarray) -> np.ndarray:
    """1차원 Gaussian HMM 의 filtered probability p(s_t | x_1..x_t)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    loglik = norm.logpdf(x[:, None], loc=means[None, :], scale=np.sqrt(variances)[None, :])
    log_a = np.log(np.clip(transmat, 1e-300, None))
    out = np.empty_like(loglik)
    alpha = np.log(np.clip(startprob, 1e-300, None)) + loglik[0]
    alpha -= logsumexp(alpha)
    out[0] = alpha
    for t in range(1, len(x)):
        alpha = logsumexp(alpha[:, None] + log_a, axis=0) + loglik[t]
        alpha -= logsumexp(alpha)
        out[t] = alpha
    return np.exp(out)


def fit_gaussian_hmm(x: np.ndarray, n_states: int = 2, n_init: int = 3, random_state: int = 0,
                     n_iter: int = 100):
    """여러 초기값 중 로그우도가 가장 높은 GaussianHMM. 상태는 평균 내림차순(0=bull)으로 정렬해 돌려준다."""
    from hmmlearn.hmm import GaussianHMM

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    best, best_score = None, -np.inf
    for i in range(n_init):
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter,
                            random_state=random_state + i)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X)
                score = model.score(X)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(score) and score > best_score:
            best, best_score = model, score
    if best is None:
        raise RuntimeError("HMM 적합에 실패했습니다")
    order = np.argsort(-best.means_.ravel())
    return {
        "startprob": best.startprob_[order],
        "transmat": best.transmat_[np.ix_(order, order)],
        "means": best.means_.ravel()[order],
        "variances": np.asarray(best.covars_).reshape(n_states, -1)[:, 0][order],
        "score": best_score,
    }


def trailing_median(regime: pd.Series, window: int) -> pd.Series:
    """과거 ``window`` 일만 보는 median filter (window<=1 이면 그대로)."""
    if window is None or window <= 1:
        return regime.astype(int)
    med = regime.rolling(window, min_periods=1).median()
    return np.floor(med + 0.5).astype(int)


def run_rolling_hmm(ret: pd.Series, *, n_states: int = 2, train_window: int = 3000, min_train: int = 500,
                    refit_every: int = 21, median_window: int = 5, start=None, n_init: int = 3,
                    random_state: int = 0, n_iter: int = 100) -> pd.DataFrame:
    """롤링 HMM 국면.

    Returns:
        DataFrame (index=날짜): regime(평활화), regime_raw, prob_0.., refit_date
    """
    ret = ret.dropna()
    first = min_train
    if start is not None:
        first = max(first, int(ret.index.searchsorted(pd.Timestamp(start))))
    if first >= len(ret):
        raise ValueError("HMM 을 적합할 데이터가 부족합니다")

    x_all = ret.to_numpy() * RET_SCALE
    parts = []
    positions = list(range(first, len(ret), refit_every))
    for i, pos in enumerate(positions):
        lo = max(0, pos - train_window)
        seg_end = positions[i + 1] if i + 1 < len(positions) else len(ret)
        p = fit_gaussian_hmm(x_all[lo:pos], n_states, n_init, random_state, n_iter)
        filt = forward_filter(x_all[lo:seg_end], p["startprob"], p["transmat"], p["means"], p["variances"])
        seg = filt[pos - lo:]
        df = pd.DataFrame(seg, index=ret.index[pos:seg_end], columns=[f"prob_{k}" for k in range(n_states)])
        df["refit_date"] = ret.index[pos]
        parts.append(df)
    out = pd.concat(parts)
    probs = out[[f"prob_{k}" for k in range(n_states)]].to_numpy()
    out.insert(0, "regime_raw", probs.argmax(axis=1))
    out.insert(0, "regime", trailing_median(out["regime_raw"], median_window).to_numpy())
    out.index.name = "date"
    logger.info("HMM: %d refits, %s ~ %s", len(positions), out.index[0].date(), out.index[-1].date())
    return out
