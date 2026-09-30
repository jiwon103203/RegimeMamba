"""SJM 에서 꼭 남길 피처를 lasso 단계에서 고정하는 ``PinnedSparseJumpModel``.

jumpmodels 의 SJM 은 좌표하강으로 (1) 가중치 w 고정 → JM 적합, (2) BCSS 로 lasso 를 풀어 w 갱신을
반복한다. (2) 의 soft-thresholding 은 BCSS 가 작은 피처의 가중치를 0 으로 만든다. 여기서는 고정(pinned)
피처를 thresholding 에서 제외해 항상 양의 가중치를 갖게 하고, 나머지 피처만 L1 제약
(||w||_1 <= sqrt(max_feats), ||w||_2 = 1) 에 맞춰 줄인다. 고정 피처가 없으면 원래 SJM 과 똑같다.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from numpy.linalg import norm
from jumpmodels.sparse_jump import SparseJumpModel, binary_search_decrease, compute_BCSS
from jumpmodels.utils import check_2d_array, raise_arr_to_pd_obj, weighted_mean_cluster


def solve_lasso_pinned(a: np.ndarray, norm_ub: float, pinned: np.ndarray, floor: float = 1e-3,
                       tol: float = 1e-8) -> np.ndarray:
    """고정 피처를 제외하고 soft-thresholding 하는 lasso 해.

    Args:
        a: BCSS / max(BCSS) (0 이상)
        norm_ub: L1 상한 (= sqrt(max_feats))
        pinned: 고정 피처 bool mask
        floor: 고정 피처의 최소 BCSS 값 (BCSS 가 0 이어도 가중치가 0 이 되지 않도록)
    """
    a = np.asarray(a, dtype=float)
    pinned = np.asarray(pinned, dtype=bool)
    a = np.where(pinned, np.maximum(a, floor), np.maximum(a, 0.0))

    def weights(thres: float) -> np.ndarray:
        y = np.where(pinned, a, np.maximum(0.0, a - thres))
        return y / norm(y)

    free = a[~pinned]
    if free.size == 0 or free.max() <= 0:
        return weights(0.0)
    right = float(free.max())  # 이 값에서 고정 피처만 남는다
    thres = binary_search_decrease(lambda t: weights(t).sum(), 0.0, right, norm_ub, tol_x=tol)
    return weights(thres)


class PinnedSparseJumpModel(SparseJumpModel):
    """``pinned`` 에 든 피처(이름 또는 열 위치)는 항상 선택되는 Sparse Jump Model."""

    def __init__(self, n_components: int = 2, max_feats: float = 100., jump_penalty: float = 0., cont: bool = False,
                 grid_size: float = 0.05, mode_loss: bool = True, random_state=None, max_iter: int = 30,
                 tol_w: float = 1e-4, max_iter_jm: int = 1000, tol_jm: float = 1e-8, n_init_jm: int = 10,
                 verbose: int = 0, pinned: Optional[Iterable] = None, pin_floor: float = 1e-3):
        super().__init__(n_components=n_components, max_feats=max_feats, jump_penalty=jump_penalty, cont=cont,
                         grid_size=grid_size, mode_loss=mode_loss, random_state=random_state, max_iter=max_iter,
                         tol_w=tol_w, max_iter_jm=max_iter_jm, tol_jm=tol_jm, n_init_jm=n_init_jm, verbose=verbose)
        self.pinned = pinned
        self.pin_floor = pin_floor

    def _pinned_mask(self, X) -> np.ndarray:
        n = X.shape[1]
        mask = np.zeros(n, dtype=bool)
        if not self.pinned:
            return mask
        columns = list(getattr(X, "columns", range(n)))
        for p in self.pinned:
            if p in columns:
                mask[columns.index(p)] = True
            elif isinstance(p, (int, np.integer)) and 0 <= p < n:
                mask[p] = True
            else:
                raise KeyError(f"pinned feature {p!r} not in X")
        return mask

    def fit(self, X, ret_ser=None, sort_by: Optional[str] = "cumret"):
        pinned = self._pinned_mask(X)
        self.pinned_mask_ = pinned
        if not pinned.any():
            return super().fit(X, ret_ser=ret_ser, sort_by=sort_by)

        # SparseJumpModel.fit 과 같은 좌표하강, lasso 단계만 solve_lasso_pinned 로 교체
        X_arr = check_2d_array(X)
        self.n_features_all = X_arr.shape[1]
        jm = self.init_jm()
        norm_ub = np.sqrt(self.max_feats)
        w_old = np.ones(self.n_features_all) * 2
        w = np.ones(self.n_features_all) / np.sqrt(self.n_features_all)
        centers_unweighted = None  # 두 번째 반복부터 직전 중심점으로 warm start
        n_iter = 0
        while n_iter < self.max_iter and norm(w - w_old, 1) / norm(w_old, 1) > self.tol_w:
            n_iter += 1
            w_old = w
            feat_weights = np.sqrt(w)
            if n_iter > 1:
                jm.centers_ = centers_unweighted * feat_weights
            jm.fit(X, ret_ser=ret_ser, feat_weights=feat_weights, sort_by=sort_by)
            centers_unweighted = weighted_mean_cluster(X_arr, jm.proba_)
            BCSS = compute_BCSS(X_arr, jm.proba_, centers_unweighted)
            if (BCSS <= 0).all():
                self.print_log(n_iter, BCSS, w)
                break
            w = solve_lasso_pinned(BCSS / BCSS.max(), norm_ub, pinned, floor=self.pin_floor)
            self.print_log(n_iter, BCSS, w)

        self.w = raise_arr_to_pd_obj(w, X, index_key="columns")
        self.feat_weights = raise_arr_to_pd_obj(jm.feat_weights, X, index_key="columns")
        self.centers_ = jm.centers_
        self.labels_ = jm.labels_
        self.proba_ = jm.proba_
        if ret_ser is not None:
            self.ret_ = jm.ret_
            self.vol_ = jm.vol_
        return self
