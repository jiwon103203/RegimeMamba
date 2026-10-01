"""--encoder mamba: 재추정마다 Mamba 를 학습해 피처 시퀀스를 hidden 벡터로 바꾼다.

``rolling.run_rolling_jm`` 의 ``encoder`` 로 호출되며, 재추정 시점 d 마다
  1. d 이전 학습창으로만 3σ 클리핑·표준화를 fit 하고 피처에 적용
  2. t 행에서 끝나는 ``mamba_seq_len`` 시퀀스 → 다음 날 신호 수익률(학습창 표준편차로 나눔)을 MSE 로 학습.
     타깃이 d 이전인 시퀀스만 쓰고, 뒤쪽 ``mamba_valid_frac`` 를 시간 순서대로 떼어 early stopping
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
                           dropout=cfg.mamba_dropout, output_dim=1)


def _windows(Z: np.ndarray, ends: np.ndarray, seq_len: int) -> np.ndarray:
    """``Z`` 에서 각 ``ends`` 행으로 끝나는 길이 ``seq_len`` 시퀀스. 데이터 시작 이전은 0 으로 채운다."""
    pad = np.zeros((seq_len - 1, Z.shape[1]), dtype=Z.dtype)
    view = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, Z]), seq_len, axis=0)  # (n, F, L)
    return np.ascontiguousarray(view[ends].transpose(0, 2, 1))


class MambaEncoder:
    """``encoder(X, y, lo, pos, seg_end)`` → X.iloc[lo:seg_end] 행의 hidden 벡터 (학습은 pos 이전 행만).

    ``cfg`` 는 ``PipelineConfig`` (``mamba_*``, ``clip_mul``, ``seed`` 를 읽는다).
    ``backbone_factory(n_features, cfg)`` 는 ``forward(x, return_hidden=True)`` 가
    ``(pred[B, 1], hidden[B, d])`` 를 돌려주는 ``torch.nn.Module`` 을 만든다.
    """

    def __init__(self, cfg, device: str, backbone_factory: Optional[Callable[[int, Any], Any]] = None):
        self.cfg = cfg
        self.device = device
        self.backbone_factory = backbone_factory or build_mamba_backbone
        self.history: List[Dict[str, Any]] = []

    def __call__(self, X: pd.DataFrame, y: pd.Series, lo: int, pos: int, seg_end: int) -> pd.DataFrame:
        import torch

        c, L = self.cfg, self.cfg.mamba_seq_len
        torch.manual_seed(c.seed + len(self.history))
        prep = WindowPreprocessor(c.clip_mul).fit(X.iloc[lo:pos])
        ctx = max(0, lo - L + 1)  # 학습창 첫 시퀀스에 쓸 이전 행
        Z = prep.transform(X.iloc[ctx:seg_end]).to_numpy(dtype=np.float32)

        # 학습 샘플: t ∈ [lo, pos-1) 에서 끝나는 시퀀스 → y[t+1] (타깃도 pos 이전)
        y_arr = y.to_numpy(dtype=np.float64)
        t_ends = np.arange(lo, pos - 1)
        if len(t_ends) < 2:
            raise ValueError("Mamba 학습 샘플이 부족합니다")
        Xs = _windows(Z, t_ends - ctx, L)
        Ys = (y_arr[t_ends + 1] / (float(np.std(y_arr[lo:pos])) or 1.0)).astype(np.float32)
        n_val = min(max(int(round(len(Xs) * c.mamba_valid_frac)), 1), len(Xs) - 1)
        n_tr = len(Xs) - n_val

        model = self.backbone_factory(Z.shape[1], c).to(self.device)
        info = {"refit_date": X.index[pos], **self._train(model, Xs[:n_tr], Ys[:n_tr], Xs[n_tr:], Ys[n_tr:]),
                "n_train_seq": n_tr, "n_valid_seq": n_val}
        self.history.append(info)
        logger.info("mamba refit %s: epochs %d (best %d), train %.4f, valid %.4f", info["refit_date"].date(),
                    info["epochs"], info["best_epoch"], info["train_loss"], info["valid_loss"])

        hidden = self._encode(model, _windows(Z, np.arange(lo - ctx, seg_end - ctx), L))
        return pd.DataFrame(hidden, index=X.index[lo:seg_end], columns=[f"h_{k}" for k in range(hidden.shape[1])])

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def _batches(self, n: int, rng: Optional[np.random.Generator] = None):
        order = rng.permutation(n) if rng is not None else np.arange(n)
        for i in range(0, n, self.cfg.mamba_batch_size):
            yield order[i:i + self.cfg.mamba_batch_size]

    def _tensor(self, a: np.ndarray):
        import torch
        return torch.from_numpy(a).to(self.device)

    def _valid_loss(self, model, X: np.ndarray, Y: np.ndarray) -> float:
        import torch
        model.eval()
        total = 0.0
        with torch.no_grad():
            for b in self._batches(len(X)):
                pred = model(self._tensor(X[b])).reshape(-1)
                total += torch.nn.functional.mse_loss(pred, self._tensor(Y[b]), reduction="sum").item()
        return total / len(X)

    def _train(self, model, X_tr, Y_tr, X_va, Y_va) -> Dict[str, Any]:
        import torch
        c = self.cfg
        opt = torch.optim.AdamW(model.parameters(), lr=c.mamba_lr, weight_decay=0.01)
        rng = np.random.default_rng(c.seed + len(self.history))
        best, best_state, best_epoch, bad, train_loss, epoch = np.inf, None, 0, 0, np.nan, 0
        for epoch in range(1, c.mamba_epochs + 1):
            model.train()
            total = 0.0
            for b in self._batches(len(X_tr), rng):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(self._tensor(X_tr[b])).reshape(-1), self._tensor(Y_tr[b]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                total += loss.item() * len(b)
            train_loss = total / len(X_tr)
            val = self._valid_loss(model, X_va, Y_va)
            if val < best - 1e-8:
                best, best_epoch, bad, best_state = val, epoch, 0, copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= c.mamba_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return {"epochs": epoch, "best_epoch": best_epoch, "train_loss": train_loss, "valid_loss": float(best)}

    def _encode(self, model, X: np.ndarray) -> np.ndarray:
        import torch
        model.eval()
        with torch.no_grad():
            out = [model(self._tensor(X[b]), return_hidden=True)[1].float().cpu().numpy() for b in self._batches(len(X))]
        return np.concatenate(out).astype(np.float64)
