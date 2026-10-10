"""--encoder mamba: 재추정마다 Mamba 를 학습해 피처 시퀀스를 hidden 벡터로 바꾼다.

``rolling.run_rolling_jm`` 의 ``encoder`` 로 호출되며, 재추정 시점 d 마다
  1. d 이전 학습창으로만 3σ 클리핑·표준화를 fit 하고 피처에 적용
  2. t 행에서 끝나는 ``mamba_seq_len`` 시퀀스 → ``mamba_targets`` × ``mamba_horizons`` 의 t+1..t+h 일 타깃을
     MSE 로 동시에 학습 (기본: 다음 날 수익률. horizons 1,5,20 = 다음 날·1주·1달, targets 는 ``forward_targets``).
     타깃은 학습창 안의 같은 h일 값으로 맞춘 스케일로 바꾼다 (return: 표준편차로 나눔, vol · mdd: 표준화).
     타깃이 모두 d 이전인 시퀀스만 쓰고, 뒤쪽 ``mamba_valid_frac`` 를 시간 순서대로 떼어 early stopping
     (h>1 이면 학습 타깃이 검증 구간과 겹치지 않게 그 사이 max(h)-1 개 시퀀스를 버린다)
  3. [학습창 + 다음 재추정 전 구간] 각 t 행의 hidden 벡터(``h_0`` ..)를 돌려준다 → 이후는 jm / sjm 과 같다.
네트워크는 d 이전 데이터로만 학습되고 t 행의 hidden 은 t 행까지의 피처만 쓰므로 인과적이다.

mamba-ssm 의 selective scan 은 CUDA 전용이라 GPU 가 필수다. torch · mamba-ssm 은 이 인코더를 쓸 때만 import 한다.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .rolling import WindowPreprocessor

logger = logging.getLogger(__name__)

DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")
TARGETS = ("return", "vol", "mdd")


def forward_targets(y: np.ndarray, t: np.ndarray, h: int, kind: str) -> np.ndarray:
    """t 행 다음 h 일(y[t+1] .. y[t+h]) 신호 수익률로 만든 타깃.

    return  h일 수익률 합 y[t+1] + .. + y[t+h]
    vol     실현 변동성 sqrt(mean(y[t+1..t+h]²)) (평균 0 가정, 일간 단위)
    mdd     t 일 종가에서 시작한 (1+y) 복리 자산 경로의 최대 낙폭 1 - min(W / 이전 고점) ≥ 0
    """
    t = np.asarray(t)
    if kind == "return":
        csum = np.concatenate([[0.0], np.cumsum(y)])
        return csum[t + 1 + h] - csum[t + 1]
    if kind == "vol":
        csq = np.concatenate([[0.0], np.cumsum(y ** 2)])
        return np.sqrt((csq[t + 1 + h] - csq[t + 1]) / h)
    if kind == "mdd":
        logw = np.concatenate([[0.0], np.cumsum(np.log1p(y))])  # logw[i] = y[:i] 복리 로그 자산
        paths = np.lib.stride_tricks.sliding_window_view(logw, h + 1)[t + 1]  # t 종가 + 다음 h 일
        return 1.0 - np.exp((paths - np.maximum.accumulate(paths, axis=1)).min(axis=1))
    raise ValueError(f"알 수 없는 타깃 {kind!r} ({', '.join(TARGETS)})")


def n_outputs(cfg) -> int:
    """Mamba 예측 헤드의 출력 수 = 타깃 수 × horizon 수."""
    return len(cfg.mamba_targets) * len(cfg.mamba_horizons)


def target_labels(targets, horizons) -> List[str]:
    """출력 열 순서(타깃 → horizon)의 이름. 수익률만이면 h<h>, 아니면 <타깃>_h<h>."""
    only_return = tuple(targets) == ("return",)
    return [f"h{h}" if only_return else f"{k}_h{h}" for k in targets for h in horizons]


def resolve_device(spec: str = "auto") -> str:
    """``--device`` 를 CUDA 장치 이름으로 바꾼다. GPU · torch · mamba-ssm 이 없으면 ValueError."""
    spec = str(spec or "auto").strip().lower()
    if not DEVICE_RE.match(spec):
        raise ValueError(f"--device 는 auto, cuda, cuda:N 중 하나여야 합니다 (got {spec!r})")
    if spec == "cpu":
        raise ValueError("mamba-ssm 은 CUDA 전용입니다: --encoder mamba 에는 --device cpu 를 쓸 수 없습니다")
    try:
        import torch
    except ImportError as e:
        raise ValueError("--encoder mamba 에는 torch (CUDA 빌드) 가 필요합니다") from e
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU 를 찾을 수 없습니다: --encoder mamba 는 GPU 가 필요합니다")
    device = "cuda:0" if spec in ("auto", "cuda") else spec
    if int(device.split(":")[1]) >= torch.cuda.device_count():
        raise ValueError(f"{device} 가 없습니다 (사용 가능한 GPU {torch.cuda.device_count()}개)")
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as e:
        raise ValueError("--encoder mamba 에는 mamba-ssm 이 필요합니다") from e
    return device


def build_mamba_backbone(n_features: int, cfg):
    """``regime_mamba`` 의 TimeSeriesMamba: forward(x, return_hidden=True) → (예측, 마지막 hidden)."""
    from regime_mamba.models.mamba_model import TimeSeriesMamba
    return TimeSeriesMamba(input_dim=n_features, d_model=cfg.mamba_d_model, d_state=cfg.mamba_d_state,
                           d_conv=cfg.mamba_d_conv, expand=cfg.mamba_expand, n_layers=cfg.mamba_layers,
                           dropout=cfg.mamba_dropout, output_dim=n_outputs(cfg))


def _windows(Z: np.ndarray, ends: np.ndarray, seq_len: int) -> np.ndarray:
    """``Z`` 에서 각 ``ends`` 행으로 끝나는 길이 ``seq_len`` 시퀀스. 데이터 시작 이전은 0 으로 채운다."""
    pad = np.zeros((seq_len - 1, Z.shape[1]), dtype=Z.dtype)
    view = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, Z]), seq_len, axis=0)  # (n, F, L)
    return np.ascontiguousarray(view[ends].transpose(0, 2, 1))


class MambaEncoder:
    """``encoder(X, y, lo, pos, seg_end)`` → X.iloc[lo:seg_end] 행의 hidden 벡터 (학습은 pos 이전 행만).

    ``cfg`` 는 ``PipelineConfig`` (``mamba_*``, ``clip_mul``, ``seed`` 를 읽는다).
    ``backbone_factory(n_features, cfg)`` 는 ``forward(x, return_hidden=True)`` 가
    ``(pred[B, len(mamba_targets) * len(mamba_horizons)], hidden[B, d])`` 를 돌려주는 ``torch.nn.Module`` 을 만든다.
    ``context`` 는 X 첫 행 이전의 피처 행 (warmup 으로 버려진 구간). 주면 시퀀스가 X 시작 전으로 넘어갈 때
    0 대신 이 행들로 채운다 (학습창 fit 통계로 같이 변환. 모자라면 남는 앞부분만 0).
    """

    def __init__(self, cfg, device: str, backbone_factory: Optional[Callable[[int, Any], Any]] = None,
                 context: Optional[pd.DataFrame] = None):
        self.cfg = cfg
        self.device = device
        self.backbone_factory = backbone_factory or build_mamba_backbone
        self.context = context
        self.history: List[Dict[str, Any]] = []

    def __call__(self, X: pd.DataFrame, y: pd.Series, lo: int, pos: int, seg_end: int) -> pd.DataFrame:
        import torch

        c, L = self.cfg, self.cfg.mamba_seq_len
        torch.manual_seed(c.seed + len(self.history))
        prep = WindowPreprocessor(c.clip_mul).fit(X.iloc[lo:pos])
        ctx = max(0, lo - L + 1)  # 학습창 첫 시퀀스에 쓸 이전 행
        pre = self._context_rows(X, L - 1 - (lo - ctx))  # X 시작 전으로 넘어가는 부분 (context 가 있을 때)
        frame = pd.concat([pre, X.iloc[ctx:seg_end]]) if len(pre) else X.iloc[ctx:seg_end]
        Z = prep.transform(frame).to_numpy(dtype=np.float32)
        base = ctx - len(pre)  # Z 0 행에 해당하는 X 위치 (context 를 붙이면 음수)

        # 학습 샘플: t ∈ [lo, pos-max(h)) 에서 끝나는 시퀀스 → 타깃 × h 마다 y[t+1..t+h] 로 만든 값 (모두 pos 이전)
        horizons = tuple(c.mamba_horizons)
        gap = max(horizons) - 1
        y_arr = y.to_numpy(dtype=np.float64)
        t_ends = np.arange(lo, pos - gap - 1)
        if len(t_ends) < gap + 2:
            raise ValueError("Mamba 학습 샘플이 부족합니다")
        Xs = _windows(Z, t_ends - base, L)
        Ys = np.empty((len(t_ends), len(c.mamba_targets) * len(horizons)), dtype=np.float32)
        for j, (kind, h) in enumerate((k, h) for k in c.mamba_targets for h in horizons):
            # 스케일: 학습창 [lo, pos) 안의 모든 h일 구간 값 (return 은 평균을 빼지 않아 부호를 유지)
            ref = forward_targets(y_arr, np.arange(lo - 1, pos - h), h, kind)
            center = 0.0 if kind == "return" else float(ref.mean())
            Ys[:, j] = (forward_targets(y_arr, t_ends, h, kind) - center) / (float(ref.std()) or 1.0)
        n_val = min(max(int(round((len(Xs) - gap) * c.mamba_valid_frac)), 1), len(Xs) - gap - 1)
        n_tr = len(Xs) - gap - n_val  # 학습 타깃(t+h)이 검증 구간에 겹치지 않게 gap 개를 버림

        model = self.backbone_factory(Z.shape[1], c).to(self.device)
        info = {"refit_date": X.index[pos],
                **self._train(model, Xs[:n_tr], Ys[:n_tr], Xs[-n_val:], Ys[-n_val:],
                              target_labels(c.mamba_targets, horizons)),
                "n_train_seq": n_tr, "n_valid_seq": n_val,
                "n_context_rows": len(pre), "n_zero_pad": max(0, L - 1 - (lo - base))}
        self.history.append(info)
        logger.info("mamba refit %s: epochs %d (best %d), train %.4f, valid %.4f", info["refit_date"].date(),
                    info["epochs"], info["best_epoch"], info["train_loss"], info["valid_loss"])

        hidden = self._encode(model, _windows(Z, np.arange(lo - base, seg_end - base), L))
        return pd.DataFrame(hidden, index=X.index[lo:seg_end], columns=[f"h_{k}" for k in range(hidden.shape[1])])

    def _context_rows(self, X: pd.DataFrame, need: int) -> pd.DataFrame:
        """X 첫 행 직전 context 의 마지막 ``need`` 행 (없거나 필요 없으면 빈 DataFrame)."""
        if need <= 0 or self.context is None or not len(self.context):
            return X.iloc[:0]
        ctx = self.context.loc[self.context.index < X.index[0], list(X.columns)]
        return ctx.iloc[max(0, len(ctx) - need):]

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def _batches(self, n: int, rng: Optional[np.random.Generator] = None):
        order = rng.permutation(n) if rng is not None else np.arange(n)
        for i in range(0, n, self.cfg.mamba_batch_size):
            yield order[i:i + self.cfg.mamba_batch_size]

    def _tensor(self, a: np.ndarray):
        import torch
        return torch.from_numpy(a).to(self.device)

    def _valid_loss(self, model, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """출력(타깃 × horizon) 별 검증 MSE."""
        import torch
        model.eval()
        total = np.zeros(Y.shape[1])
        with torch.no_grad():
            for b in self._batches(len(X)):
                pred = model(self._tensor(X[b])).reshape(len(b), -1)
                total += ((pred - self._tensor(Y[b])) ** 2).sum(dim=0).cpu().numpy()
        return total / len(X)

    def _train(self, model, X_tr, Y_tr, X_va, Y_va, labels=("h1",)) -> Dict[str, Any]:
        import torch
        c = self.cfg
        opt = torch.optim.AdamW(model.parameters(), lr=c.mamba_lr, weight_decay=0.01)
        rng = np.random.default_rng(c.seed + len(self.history))
        best, best_state, best_epoch, bad, train_loss, epoch = np.inf, None, 0, 0, np.nan, 0
        best_by_h = np.full(len(labels), np.nan)
        for epoch in range(1, c.mamba_epochs + 1):
            model.train()
            total = 0.0
            for b in self._batches(len(X_tr), rng):
                opt.zero_grad()
                pred = model(self._tensor(X_tr[b])).reshape(len(b), -1)
                if pred.shape[1] != Y_tr.shape[1]:
                    raise ValueError(f"Mamba 출력 {pred.shape[1]}개 ≠ 타깃 {Y_tr.shape[1]}개 (n_outputs(cfg))")
                loss = torch.nn.functional.mse_loss(pred, self._tensor(Y_tr[b]))  # 출력(타깃 × horizon) 평균
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                total += loss.item() * len(b)
            train_loss = total / len(X_tr)
            val_by_h = self._valid_loss(model, X_va, Y_va)
            val = float(val_by_h.mean())
            if val < best - 1e-8:
                best, best_epoch, bad, best_state = val, epoch, 0, copy.deepcopy(model.state_dict())
                best_by_h = val_by_h
            else:
                bad += 1
                if bad >= c.mamba_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        info = {"epochs": epoch, "best_epoch": best_epoch, "train_loss": train_loss, "valid_loss": float(best)}
        if len(labels) > 1:
            info.update({f"valid_loss_{k}": float(v) for k, v in zip(labels, best_by_h)})
        return info

    def _encode(self, model, X: np.ndarray) -> np.ndarray:
        import torch
        model.eval()
        with torch.no_grad():
            out = [model(self._tensor(X[b]), return_hidden=True)[1].float().cpu().numpy() for b in self._batches(len(X))]
        return np.concatenate(out).astype(np.float64)
