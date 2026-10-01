"""3') --model mamba: 재추정마다 Mamba 를 학습해 피처 시퀀스를 hidden 벡터로 압축한다.

재추정 시점 d 마다 (``rolling.run_rolling_jm`` 의 ``encoder`` 로 호출된다):
  1. d 이전 학습창으로만 3σ 클리핑·표준화를 fit 하고 피처 전체에 적용
  2. t 행에서 끝나는 길이 ``seq_len`` 시퀀스 → 다음 날 신호 수익률(학습창 표준편차로 나눔)을 MSE 로 학습.
     타깃이 d 이전인 시퀀스만 쓰고, 그 중 뒤쪽 ``valid_frac`` 를 시간 순서대로 떼어 early stopping 한다.
  3. 학습창 + 다음 재추정 전까지 구간의 각 t 행에 대해 t 행에서 끝나는 시퀀스의 마지막 hidden 을 뽑는다.
     네트워크는 d 이전 데이터로만 학습되고, t 행의 hidden 은 t 행까지의 피처만 쓰므로 인과적이다.
이 hidden 벡터(``h_0`` ..)가 Jump Model 의 입력이 된다.

Mamba(mamba-ssm) 의 selective scan 은 CUDA 전용이라 GPU 가 필수다. torch 와 mamba-ssm 은
``--model mamba`` 일 때만 import 하므로 jm / sjm 은 torch 없이 돌아간다.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .rolling import WindowPreprocessor

logger = logging.getLogger(__name__)

DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")


@dataclass
class MambaSettings:
    seq_len: int = 60
    d_model: int = 8
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 4
    dropout: float = 0.1
    epochs: int = 100
    patience: int = 10
    batch_size: int = 1024
    lr: float = 5e-4
    valid_frac: float = 0.2
    clip_mul: float = 3.0
    seed: int = 0


def resolve_device(spec: str = "auto") -> str:
    """``--device`` 를 실제 CUDA 장치 이름으로 바꾼다. GPU 나 mamba-ssm 이 없으면 ValueError."""
    spec = str(spec or "auto").strip().lower()
    if not DEVICE_RE.match(spec):
        raise ValueError(f"--device 는 auto, cuda, cuda:N 중 하나여야 합니다 (got {spec!r})")
    if spec == "cpu":
        raise ValueError("Mamba(mamba-ssm) 는 CUDA 전용입니다: --model mamba 에는 --device cpu 를 쓸 수 없습니다")
    try:
        import torch
    except ImportError as e:
        raise ValueError("--model mamba 에는 torch 가 필요합니다 (CUDA 빌드)") from e
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU 를 찾을 수 없습니다: --model mamba 는 GPU 가 필요합니다 "
                         "(torch.cuda.is_available() == False)")
    device = "cuda:0" if spec in ("auto", "cuda") else spec
    idx = int(device.split(":")[1])
    if idx >= torch.cuda.device_count():
        raise ValueError(f"{device} 가 없습니다 (사용 가능한 GPU {torch.cuda.device_count()}개)")
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as e:
        raise ValueError("--model mamba 에는 mamba-ssm 이 필요합니다 (setup_and_run.sh / docs/WINDOWS_SETUP.md 참고)") from e
    return device


def build_mamba_backbone(n_features: int, s: MambaSettings):
    """``regime_mamba`` 의 TimeSeriesMamba: forward(x, return_hidden=True) → (예측, 마지막 hidden)."""
    from regime_mamba.models.mamba_model import TimeSeriesMamba
    return TimeSeriesMamba(input_dim=n_features, d_model=s.d_model, d_state=s.d_state, d_conv=s.d_conv,
                           expand=s.expand, n_layers=s.n_layers, dropout=s.dropout, output_dim=1)


def _windows(Z: np.ndarray, ends: np.ndarray, seq_len: int) -> np.ndarray:
    """``Z`` 에서 각 ``ends`` 행으로 끝나는 길이 ``seq_len`` 시퀀스. 데이터 시작 이전은 0 으로 채운다."""
    pad = np.zeros((seq_len - 1, Z.shape[1]), dtype=Z.dtype)
    Zp = np.concatenate([pad, Z])
    view = np.lib.stride_tricks.sliding_window_view(Zp, seq_len, axis=0)  # (n, F, L)
    return np.ascontiguousarray(view[ends].transpose(0, 2, 1))


class MambaEncoder:
    """재추정 창마다 새 백본을 학습하고 hidden 벡터를 돌려주는 ``run_rolling_jm`` 용 encoder.

    ``backbone_factory(n_features, settings)`` 는 ``forward(x, return_hidden=True)`` 가
    ``(pred[B, 1], hidden[B, d_model])`` 를 돌려주는 ``torch.nn.Module`` 을 만든다.
    """

    def __init__(self, settings: MambaSettings, device: str,
                 backbone_factory: Optional[Callable[[int, MambaSettings], Any]] = None):
        self.s = settings
        self.device = device
        self.backbone_factory = backbone_factory or build_mamba_backbone
        self.history: List[Dict[str, Any]] = []
        self.last_model = None

    def __call__(self, X: pd.DataFrame, y: pd.Series, lo: int, pos: int, seg_end: int) -> pd.DataFrame:
        """X.iloc[lo:seg_end] 각 행의 hidden 벡터 (학습은 pos 이전 행만 사용)."""
        import torch

        s = self.s
        torch.manual_seed(s.seed + len(self.history))
        prep = WindowPreprocessor(s.clip_mul).fit(X.iloc[lo:pos])
        ctx = max(0, lo - s.seq_len + 1)  # 학습창 첫 시퀀스에 쓸 이전 행
        Z = prep.transform(X.iloc[ctx:seg_end]).to_numpy(dtype=np.float32)
        off = lo - ctx

        # 학습 샘플: t ∈ [lo, pos-1) 에서 끝나는 시퀀스 → y[t+1] (타깃도 pos 이전)
        y_arr = y.to_numpy(dtype=np.float64)
        y_scale = float(np.std(y_arr[lo:pos])) or 1.0
        t_ends = np.arange(lo, pos - 1)
        if len(t_ends) < 2:
            raise ValueError("Mamba 학습 샘플이 부족합니다")
        Xs = _windows(Z, t_ends - ctx, s.seq_len)
        Ys = (y_arr[t_ends + 1] / y_scale).astype(np.float32)
        n_val = int(round(len(Xs) * s.valid_frac))
        n_val = min(max(n_val, 1), len(Xs) - 1)
        n_tr = len(Xs) - n_val

        model = self.backbone_factory(Z.shape[1], s).to(self.device)
        info = self._train(model, Xs[:n_tr], Ys[:n_tr], Xs[n_tr:], Ys[n_tr:])
        info.update({"refit_date": X.index[pos], "n_train_seq": n_tr, "n_valid_seq": n_val})
        self.history.append(info)
        self.last_model = model
        logger.info("mamba refit %s: epochs %d (best %d), train %.4f, valid %.4f",
                    info["refit_date"], info["epochs"], info["best_epoch"], info["train_loss"], info["valid_loss"])

        hidden = self._encode(model, _windows(Z, np.arange(off, off + seg_end - lo), s.seq_len))
        cols = [f"h_{k}" for k in range(hidden.shape[1])]
        return pd.DataFrame(hidden, index=X.index[lo:seg_end], columns=cols)

    def _batches(self, n: int, shuffle: bool, gen=None):
        order = np.arange(n)
        if shuffle:
            order = gen.permutation(n)
        for i in range(0, n, self.s.batch_size):
            yield order[i:i + self.s.batch_size]

    def _loss(self, model, X: np.ndarray, Y: np.ndarray) -> float:
        import torch
        model.eval()
        total = 0.0
        with torch.no_grad():
            for b in self._batches(len(X), False):
                xb = torch.from_numpy(X[b]).to(self.device)
                yb = torch.from_numpy(Y[b]).to(self.device)
                pred = model(xb).reshape(-1)
                total += float(torch.nn.functional.mse_loss(pred, yb, reduction="sum"))
        return total / len(X)

    def _train(self, model, X_tr, Y_tr, X_va, Y_va) -> Dict[str, Any]:
        import torch
        s = self.s
        opt = torch.optim.AdamW(model.parameters(), lr=s.lr, weight_decay=0.01)
        gen = np.random.default_rng(s.seed + len(self.history))
        best, best_state, best_epoch, bad = np.inf, None, 0, 0
        train_loss = np.nan
        epoch = 0
        for epoch in range(1, s.epochs + 1):
            model.train()
            total = 0.0
            for b in self._batches(len(X_tr), True, gen):
                xb = torch.from_numpy(X_tr[b]).to(self.device)
                yb = torch.from_numpy(Y_tr[b]).to(self.device)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(xb).reshape(-1), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                total += loss.item() * len(b)
            train_loss = total / len(X_tr)
            val = self._loss(model, X_va, Y_va)
            if val < best - 1e-8:
                best, best_epoch, bad = val, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= s.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return {"epochs": epoch, "best_epoch": best_epoch, "train_loss": train_loss, "valid_loss": float(best)}

    def _encode(self, model, X: np.ndarray) -> np.ndarray:
        import torch
        model.eval()
        out = []
        with torch.no_grad():
            for b in self._batches(len(X), False):
                _, h = model(torch.from_numpy(X[b]).to(self.device), return_hidden=True)
                out.append(h.float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
