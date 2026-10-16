#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Jump Model 국면 파이프라인 진입점.

입력 파일(날짜·종가·무위험금리)
  │  data_io.py         1) 로드·정제 → 초과수익률 (+ 벤치마크 차감 rel_ret)
  │  features.py        2) EWM downside deviation · Sortino (+ extra / 커스텀 변수)
  │  rolling.py         3) 6개월마다 재추정(3000일 학습창) + 사이 구간 온라인 추론
  │  mamba_encoder.py      (--encoder mamba) 재추정마다 Mamba 학습 → hidden 벡터를 Jump Model 입력으로 (GPU)
  │  backtest.py        4) 0/1 전략 백테스트 · 성과표 · 거래 지연 로버스트니스
  │  hmm_benchmark.py   5) HMM 벤치마크와 비교 (--hmm)
  │  weights.py         6) 변수 유형별 가중 비중 (sjm)
  │  regime_episodes.py    bear 에피소드 · 유사 국면 · 종료 시나리오
  ▼  plotting.py        7) csv 와 png 저장 (out/)

사용 예:
    python run_pipeline.py data.csv
    python run_pipeline.py data.csv --model sjm --feature-set extra --pin-features sortino_20 --hmm
    python run_pipeline.py sector.csv --relative-benchmark kospi.csv --backtest-ret relative
    python run_pipeline.py data.csv --extra-features macro.csv:VIX:log macro.csv:USDKRW:logdiff
    python run_pipeline.py data.csv --feature-set extra --use-features sortino DD_20 vol-log
    python run_pipeline.py data.csv --inference
    python run_pipeline.py data.csv --encoder mamba --feature-set example --device cuda:0
    python run_pipeline.py data.csv --encoder mamba --n-seeds 5 --seed-mode both
    python run_pipeline.py data.csv --encoder mamba --n-seeds 20 --seed-workers 8 --seed-devices cuda:0,cuda:1
    python run_pipeline.py data.csv --encoder mamba --mamba-horizons 1,5,20
    python run_pipeline.py data.csv --encoder mamba --mamba-horizons 5,20 --mamba-targets vol,mdd
    python run_pipeline.py data.csv --encoder mamba --mamba-targets return,mdd --separate --separate-weights 0.3 0.5 0.7
    python run_pipeline.py data.csv --encoder mamba --feature-set example --mamba-targets return,mdd --separate \
        --separate-features mdd=DD-log,sortino_20 return=ret,sortino
    python run_pipeline.py data.csv --encoder mamba --mamba-targets return,mdd --separate \
        --separate-jump-penalty return=80 mdd=30
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import logging
import multiprocessing
import os
import sys
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from regime_jm import mamba_encoder, plotting
from regime_jm.backtest import delay_robustness_table, performance_table, regime_summary, run_0_1_strategy
from regime_jm.config import SEED_MODES, PipelineConfig
from regime_jm.data_io import (RF_UNITS, align_to_index, load_series, parse_extra_spec, prepare_inputs,
                               resolve_signal_return)
from regime_jm.features import FEATURE_SETS, apply_transform, build_features, expand_feature_names
from regime_jm.hmm_benchmark import run_rolling_hmm
from regime_jm.regime_episodes import (compare_episode_paths, episode_metrics, episode_paths, extract_episodes,
                                       length_scenarios, rank_similar_episodes)
from regime_jm.rolling import RollingJMResult, refit_schedule, run_rolling_jm, state_label
from regime_jm.weights import current_weight_summary, group_weights, weight_groups_long

logger = logging.getLogger("regime_jm")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


# ---------------------------------------------------------------------------
# 공통 단계
# ---------------------------------------------------------------------------

def _resolve_cfg(cfg: Optional[PipelineConfig], overrides: Dict[str, Any]) -> PipelineConfig:
    cfg = cfg or PipelineConfig()
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg.validate()


def load_extra_features(specs: Sequence[str], main_path: str, index: pd.DatetimeIndex,
                        main_date_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """--extra-features 를 읽어 변환(원래 주기) → 거래일 forward-fill 정렬한다."""
    if not specs:
        return None
    cache: Dict[str, pd.DataFrame] = {}
    cols: Dict[str, pd.Series] = {}
    for raw in specs:
        spec = parse_extra_spec(raw)
        path = spec.file or main_path
        date_col = main_date_col if path == main_path else None
        s = load_series(path, spec.column, date_col=date_col, _cache=cache)
        name = spec.name
        if name in cols:
            raise ValueError(f"사용자 변수 이름 '{name}' 이 중복됩니다")
        cols[name] = align_to_index(apply_transform(s, spec.transform), index)
        logger.info("extra feature %s (%s, %d obs)", name, path, len(s))
    return pd.DataFrame(cols, index=index)


def prepare_features(cfg: PipelineConfig):
    """1)~2) 단계: 입력 → (data, 신호 수익률 열 이름, 피처 행렬, warmup context).

    context 는 --encoder mamba --mamba-warmup-context 일 때만 warmup 으로 버려진 피처 행 (아니면 None).
    """
    data = prepare_inputs(cfg.input, date_col=cfg.date_col, close_col=cfg.close_col, rf_col=cfg.rf_col,
                          rf_unit=cfg.rf_unit, rf_const=cfg.rf_const, benchmark=cfg.relative_benchmark,
                          bench_date_col=cfg.bench_date_col, bench_close_col=cfg.bench_close_col,
                          start=cfg.start, end=cfg.end)
    signal_col = resolve_signal_return(data, cfg.signal_ret)
    extra = load_extra_features(cfg.extra_features, cfg.input, data.index, cfg.date_col)
    use_context = cfg.encoder == "mamba" and cfg.mamba_warmup_context
    X = build_features(data[signal_col], cfg.feature_set, extra=extra, remove=cfg.remove_series,
                       warmup=cfg.warmup, return_context=use_context, keep=cfg.use_features)
    context = None
    if use_context:
        X, context = X
        logger.info("mamba warmup context: X 이전 %d행을 시퀀스 앞부분으로 사용 (seq_len %d → 필요 최대 %d행)",
                    len(context), cfg.mamba_seq_len, cfg.mamba_seq_len - 1)
    logger.info("signal=%s, features=%d (%s), rows=%d (%s ~ %s)", signal_col, X.shape[1], cfg.feature_set,
                len(X), X.index[0].date(), X.index[-1].date())
    return data, signal_col, X, context


def _make_encoder(cfg: PipelineConfig, context: Optional[pd.DataFrame] = None
                  ) -> Optional[mamba_encoder.MambaEncoder]:
    if cfg.encoder != "mamba":
        return None
    device = mamba_encoder.resolve_device(cfg.device)
    weights = mamba_encoder.loss_weights(cfg)
    logger.info("mamba encoder on %s (seq_len=%d, d_model=%d, layers=%d, targets=%s, horizons=%s, loss=%s)", device,
                cfg.mamba_seq_len, cfg.mamba_d_model, cfg.mamba_layers, ",".join(cfg.mamba_targets),
                ",".join(map(str, cfg.mamba_horizons)),
                "mean" if weights is None else "weighted " + ",".join(f"{v:.3g}" for v in weights))
    return mamba_encoder.MambaEncoder(cfg, device, context=context)


def _fit_rolling(cfg: PipelineConfig, X: pd.DataFrame, signal: pd.Series, start=None,
                 encoder=None) -> RollingJMResult:
    pinned = expand_feature_names(X.columns, cfg.pin_features)
    return run_rolling_jm(X, signal, model=cfg.model, cont=cfg.cont, n_states=cfg.n_states,
                          jump_penalty=cfg.jump_penalty, max_feats=cfg.max_feats, pinned=pinned or None,
                          train_window=cfg.train_window, min_train=cfg.min_train, refit_months=cfg.refit_months,
                          start=start, clip_mul=cfg.clip_mul, grid_size=cfg.grid_size, n_init=cfg.n_init,
                          random_state=cfg.seed, encoder=encoder, center_distance=cfg.center_distance)


def _regime_table(res: RollingJMResult, signal: pd.Series) -> pd.DataFrame:
    reg = res.regimes.copy()
    reg.insert(1, "label", [state_label(s, res.n_states) for s in reg["regime"]])
    reg["signal_ret"] = signal.reindex(reg.index)
    return reg


def _strategy_kwargs(cfg: PipelineConfig, bear_state: int) -> Dict[str, Any]:
    return dict(bear_state=bear_state, min_cash=cfg.min_cash, max_cash=cfg.max_cash, cost_buy=cfg.cost_buy,
                cost_sell=cfg.cost_sell, mode=cfg.backtest_ret)


def _save(df: Optional[pd.DataFrame], out_dir: str, name: str, index: bool = True) -> Optional[str]:
    if df is None:
        return None
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def _current_state(reg: pd.DataFrame, n_states: int) -> Dict[str, Any]:
    last = reg.iloc[-1]
    run = reg["regime"].ne(reg["regime"].shift()).cumsum()
    return {
        "asof": reg.index[-1].date().isoformat(),
        "regime": int(last["regime"]),
        "label": state_label(int(last["regime"]), n_states),
        "prob_bear": float(last[f"prob_{n_states - 1}"]),
        "days_in_regime": int((run == run.iloc[-1]).sum()),
        "refit_date": pd.Timestamp(last["refit_date"]).date().isoformat(),
    }


def _write_json(obj, out_dir: str, name: str):
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# 전체 백테스트
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Optional[PipelineConfig] = None, **overrides) -> Dict[str, Any]:
    """전체 파이프라인을 실행하고 결과 DataFrame 들을 dict 로 돌려준다 (파일은 cfg.out_dir 에 저장)."""
    cfg = _resolve_cfg(cfg, overrides)
    if cfg.separate:
        return run_separate(cfg)
    if len(cfg.seed_list) > 1:
        return run_multi_seed(cfg)
    if cfg.inference:
        return run_inference(cfg)
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    _write_json(cfg.to_dict(), out, "run_config.json")
    encoder = _make_encoder(cfg)  # GPU 가 없으면 데이터 처리 전에 실패

    # 1)~2) 데이터 · 피처
    data, signal_col, X, context = prepare_features(cfg)
    signal = data[signal_col]
    if encoder is not None:
        encoder.context = context

    # 3) 롤링 재추정 + 온라인 추론 (mamba: 재추정마다 Mamba 학습 → hidden 벡터)
    res = _fit_rolling(cfg, X, signal, start=cfg.oos_start, encoder=encoder)
    reg = _regime_table(res, signal)

    # 4) 백테스트
    strat_kwargs = _strategy_kwargs(cfg, res.bear_state)
    strategy = run_0_1_strategy(reg["regime"], data, delay=cfg.delay, **strat_kwargs)
    performance = performance_table(strategy)
    summary = regime_summary(reg["regime"], signal, res.n_states)
    delay_table = delay_robustness_table(reg["regime"], data, cfg.delays, **strat_kwargs)

    # 5) HMM 벤치마크 (공통 구간 비교)
    hmm_regimes = comparison = hmm_strategy = None
    if cfg.hmm:
        hmm_regimes = run_rolling_hmm(signal, n_states=cfg.hmm_states, train_window=cfg.train_window,
                                      min_train=cfg.min_train, refit_every=cfg.hmm_refit,
                                      median_window=cfg.hmm_median, start=res.refit_dates[0],
                                      n_init=cfg.hmm_n_init, random_state=cfg.seed)
        hmm_kwargs = {**strat_kwargs, "bear_state": cfg.hmm_states - 1}
        hmm_strategy = run_0_1_strategy(hmm_regimes["regime"], data, delay=cfg.delay, **hmm_kwargs)
        common = strategy.index.intersection(hmm_strategy.index)
        comparison = performance_table(strategy.loc[common], {"HMM strategy": hmm_strategy.loc[common]})

    # 6) 해석: 피처 가중 비중 · 에피소드
    groups = weights_now = None
    if res.feat_weights is not None:
        groups = weight_groups_long(res.feat_weights)
        weights_now = current_weight_summary(res.feat_weights)
    path_ret = data["rel_ret"] if signal_col == "rel_ret" else data["asset_ret"]
    episodes = episode_metrics(extract_episodes(reg["regime"], n_states=res.n_states), path_ret, res.bear_state)
    similar = rank_similar_episodes(episodes, path_ret, top_k=cfg.top_k)
    path_cmp = compare_episode_paths(path_ret, episodes)
    similar_ids = list(similar["episode_id"]) if not similar.empty else None
    scenarios = length_scenarios(episodes, similar_ids=similar_ids)
    current_ep = episodes.iloc[-1]
    path_ids = [int(current_ep["episode_id"])] + (similar_ids or [])
    paths = episode_paths(path_ret, episodes, path_ids)

    # 7) 저장
    _save(X, out, "features.csv")
    _save(reg, out, "regimes.csv")
    _save(res.params, out, "refit_params.csv", index=False)
    _save(res.insample_last, out, "insample_last.csv")
    _save(strategy, out, "strategy.csv")
    _save(performance, out, "performance.csv")
    _save(summary, out, "regime_summary.csv", index=False)
    _save(delay_table, out, "delay_robustness.csv")
    _save(res.feat_weights, out, "feat_weights.csv")
    _save(groups, out, "weight_groups.csv", index=False)
    _save(weights_now, out, "weight_groups_current.csv", index=False)
    _save(episodes, out, "regime_episodes.csv", index=False)
    _save(similar, out, "similar_episodes.csv", index=False)
    _save(path_cmp, out, "episode_path_rmse.csv", index=False)
    _save(paths, out, "episode_paths.csv")
    _save(scenarios, out, "length_scenarios.csv", index=False)
    _save(hmm_regimes, out, "hmm_regimes.csv")
    _save(comparison, out, "model_comparison.csv")
    _save(encoder.history_frame() if encoder else None, out, "mamba_train_log.csv", index=False)
    current = _current_state(reg, res.n_states)
    _write_json(current, out, "current_state.json")

    if cfg.plots:
        others = {"HMM strategy": hmm_strategy} if hmm_strategy is not None else None
        plotting.plot_regimes_cumret(strategy, res.bear_state, os.path.join(out, "regimes_cumret.png"), others)
        plotting.plot_refit_params(res.params, os.path.join(out, "refit_params.png"))
        plotting.plot_weights(strategy, os.path.join(out, "weights.png"))
        plotting.plot_delay_robustness(delay_table, os.path.join(out, "delay_robustness.png"))
        if res.feat_weights is not None:
            plotting.plot_feat_weights(res.feat_weights, group_weights(res.feat_weights, "category"),
                                       group_weights(res.feat_weights, "horizon"),
                                       os.path.join(out, "feat_weights.png"))
        plotting.plot_episode_lengths(episodes, os.path.join(out, "episode_lengths.png"), current_ep,
                                      state_label=state_label(res.bear_state, res.n_states))
        plotting.plot_similar_paths(paths, int(current_ep["episode_id"]), os.path.join(out, "similar_paths.png"),
                                    label=str(current_ep["label"]))

    logger.info("current regime: %s", current)
    return {
        "config": cfg, "data": data, "signal_col": signal_col, "features": X, "result": res, "regimes": reg,
        "strategy": strategy, "performance": performance, "regime_summary": summary,
        "delay_robustness": delay_table, "hmm_regimes": hmm_regimes, "model_comparison": comparison,
        "weight_groups": groups, "episodes": episodes, "similar_episodes": similar,
        "episode_path_rmse": path_cmp, "length_scenarios": scenarios, "current_state": current,
    }


# ---------------------------------------------------------------------------
# 추론 모드
# ---------------------------------------------------------------------------

def run_inference(cfg: Optional[PipelineConfig] = None, **overrides) -> Dict[str, Any]:
    """가장 최근 반기 시작점 직전 train_window 일로 한 번만 학습하고 현재 반기만 온라인 추론한다.

    과거 전체 백테스트와 전략 성과는 계산하지 않는다.
    """
    cfg = _resolve_cfg(cfg, overrides)
    if cfg.separate:
        return run_separate(dataclasses.replace(cfg, inference=True))
    if len(cfg.seed_list) > 1:
        return run_multi_seed(dataclasses.replace(cfg, inference=True))
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    _write_json(cfg.to_dict(), out, "run_config.json")
    encoder = _make_encoder(cfg)

    data, signal_col, X, context = prepare_features(cfg)
    signal = data[signal_col]
    if encoder is not None:
        encoder.context = context
    dates = refit_schedule(X.index, cfg.refit_months, cfg.min_train)
    if not dates:
        raise ValueError("추론할 반기 시작점이 없습니다 (데이터 부족)")
    res = _fit_rolling(cfg, X, signal, start=dates[-1], encoder=encoder)
    reg = _regime_table(res, signal)

    params = res.params
    current = _current_state(reg, res.n_states)
    current.update({
        "train_start": pd.Timestamp(params["train_start"].iloc[0]).date().isoformat(),
        "train_end": pd.Timestamp(params["train_end"].iloc[0]).date().isoformat(),
        "n_train": int(params["n_train"].iloc[0]),
        "signal": signal_col,
        "model": cfg.model + ("" if cfg.cont else " (discrete)") + (" + mamba" if encoder else ""),
    })
    for _, row in params.iterrows():
        current[f"{row['label']}_ann_ret"] = float(row["ann_ret"])
        current[f"{row['label']}_ann_vol"] = float(row["ann_vol"])

    _save(reg, out, "inference_regimes.csv")
    _save(params, out, "inference_params.csv", index=False)
    _save(pd.DataFrame([current]), out, "inference_summary.csv", index=False)
    _save(encoder.history_frame() if encoder else None, out, "inference_mamba_train_log.csv", index=False)
    weights_now = None
    if res.feat_weights is not None:
        _save(res.feat_weights, out, "inference_feat_weights.csv")
        weights_now = current_weight_summary(res.feat_weights)
        _save(weights_now, out, "inference_weight_groups.csv", index=False)
    _write_json(current, out, "current_state.json")
    if cfg.plots:
        plotting.plot_inference(reg, data["close"], res.bear_state, os.path.join(out, "inference.png"))

    logger.info("current regime: %s", current)
    return {"config": cfg, "data": data, "features": X, "result": res, "regimes": reg, "params": params,
            "current_state": current, "weight_groups": weights_now}


# ---------------------------------------------------------------------------
# 여러 시드 (--n-seeds / --seeds)
# ---------------------------------------------------------------------------

def _seed_cfg(cfg: PipelineConfig, seed: int, out_dir: str, index: int = 0) -> PipelineConfig:
    extra = {"device": cfg.seed_devices[index % len(cfg.seed_devices)]} if cfg.seed_devices else {}
    return dataclasses.replace(cfg, seed=seed, seeds=(), n_seeds=1, ensemble_bear_vote=None, seed_workers=1,
                               seed_devices=(), out_dir=out_dir, **extra)


def _check_devices(cfg: PipelineConfig) -> None:
    """GPU 가 없거나 --seed-devices 의 장치가 없으면 학습 전에 실패한다."""
    if cfg.encoder == "mamba":
        for device in cfg.seed_devices or (cfg.device,):
            mamba_encoder.resolve_device(device)


def _individual_job(scfg: PipelineConfig) -> Dict[str, Any]:
    return (run_inference if scfg.inference else run_pipeline)(scfg)


def _ensemble_job(scfg: PipelineConfig, X: pd.DataFrame, signal: pd.Series, start,
                  context: Optional[pd.DataFrame] = None):
    encoder = _make_encoder(scfg, context)
    reg = _regime_table(_fit_rolling(scfg, X, signal, start=start, encoder=encoder), signal)
    return reg, (encoder.history_frame().assign(seed=scfg.seed) if encoder is not None else None)


class _SeedTag(logging.Filter):
    """병렬 실행에서 로그 줄 앞에 [seed s] 를 붙인다."""

    def __init__(self, seed: int):
        super().__init__()
        self.tag = f"[seed {seed}] "

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"{self.tag}{record.msg}"
        return True


def _init_seed_worker(level: Optional[int]) -> None:
    if level is not None:
        logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _seed_worker(seed: int, fn, args):
    for handler in logging.getLogger().handlers:
        for f in [f for f in handler.filters if isinstance(f, _SeedTag)]:
            handler.removeFilter(f)
        handler.addFilter(_SeedTag(seed))
    logger.info("start")
    return fn(*args)


_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@contextlib.contextmanager
def _worker_threads(workers: int):
    """작업 프로세스마다 BLAS/OpenMP 스레드를 CPU 코어 / workers 개로 제한한다 (이미 지정돼 있으면 그대로)."""
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    n = str(max(1, (os.cpu_count() or 1) // workers))
    for k in _THREAD_VARS:
        os.environ.setdefault(k, n)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_seed_jobs(cfg: PipelineConfig, jobs: Dict[int, tuple]) -> Dict[int, Any]:
    """시드별 작업 {seed: (함수, 인자)} 을 실행해 {seed: 결과} 로 돌려준다.

    seed_workers > 1 이면 spawn 프로세스 풀에서 시드 여러 개를 동시에 돌린다. 시드마다 별도 프로세스라
    torch / numpy 난수 상태가 섞이지 않아 결과는 순차 실행과 같다.
    """
    seeds = list(jobs)
    workers = min(cfg.seed_workers, len(seeds))
    if workers <= 1:
        results = {}
        for i, s in enumerate(seeds, start=1):
            logger.info("seed %d (%d/%d)", s, i, len(seeds))
            fn, args = jobs[s]
            results[s] = fn(*args)
        return results

    logger.info("running %d seeds on %d worker processes", len(seeds), workers)
    level = logger.getEffectiveLevel() if logging.getLogger().handlers else None
    with _worker_threads(workers), ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                                       initializer=_init_seed_worker, initargs=(level,)) as pool:
        futures = {pool.submit(_seed_worker, s, fn, args): s for s, (fn, args) in jobs.items()}
        pending, n_done = set(futures), 0
        while pending:
            done, pending = wait(pending, return_when=FIRST_EXCEPTION)
            for f in done:
                if f.exception() is not None:
                    for p in pending:
                        p.cancel()
                    raise f.exception()
                n_done += 1
                logger.info("seed %d done (%d/%d)", futures[f], n_done, len(seeds))
        return {s: f.result() for f, s in sorted(futures.items(), key=lambda kv: seeds.index(kv[1]))}


def ensemble_regimes(regimes: Dict[Any, pd.DataFrame], n_states: int, bear_vote: Optional[float] = None,
                     weights: Optional[Dict[Any, float]] = None, column: str = "regime_seed{}") -> pd.DataFrame:
    """시드별 국면표의 상태 확률을 날짜마다 평균내고 argmax 를 앙상블 국면으로 삼는다.

    상태는 시드마다 같은 규칙(상태 0 = bull, 마지막 = bear, sort_by='cumret')으로 정렬돼 있어 바로 평균낼 수 있다.
    이산형 모델이면 확률이 one-hot 이라 다수결과 같다. agreement 는 앙상블 국면과 같은 시드의 비율,
    bear_vote 는 bear 로 판정한 시드의 비율.
    bear_vote 기준을 주면 bear 표 비율이 그 값 이상인 날을 bear 로, 나머지 날은 bear 를 뺀 상태 중 평균 확률
    argmax 로 정한다 (예: 0.4 면 5개 시드 중 2개만 bear 여도 bear).
    weights 를 주면 평균 · agreement · bear_vote 를 그 비중(합으로 정규화)의 가중평균으로 계산한다 (--separate).
    column 은 모델별 국면 열 이름 형식.
    """
    seeds = list(regimes)
    idx = regimes[seeds[0]].index
    for r in regimes.values():
        idx = idx.intersection(r.index)
    w = np.ones(len(seeds)) if weights is None else np.array([weights[s] for s in seeds], dtype=float)
    w = w / w.sum()
    prob_cols = [f"prob_{k}" for k in range(n_states)]
    avg = sum(wi * regimes[s].loc[idx, prob_cols].to_numpy(dtype=float) for wi, s in zip(w, seeds))
    by_seed = pd.DataFrame({column.format(s): regimes[s].loc[idx, "regime"].astype(int) for s in seeds}, index=idx)
    bear = n_states - 1
    votes = (by_seed.to_numpy() == bear) @ w
    if bear_vote is None:
        state = avg.argmax(axis=1)
    else:
        state = np.where(votes >= bear_vote - 1e-9, bear, avg[:, :bear].argmax(axis=1))
    first = regimes[seeds[0]].loc[idx]
    ens = pd.DataFrame({"regime": state, "label": [state_label(k, n_states) for k in state]}, index=idx)
    for k, col in enumerate(prob_cols):
        ens[col] = avg[:, k]
    ens["refit_date"] = first["refit_date"]
    if "signal_ret" in first.columns:
        ens["signal_ret"] = first["signal_ret"]
    ens["agreement"] = (by_seed.to_numpy() == state[:, None]) @ w
    ens["bear_vote"] = votes
    ens.index.name = "date"
    return pd.concat([ens, by_seed], axis=1)


def seed_performance_table(seed_results: Dict[int, Dict[str, Any]],
                           ensemble_perf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """시드별 전략 성과 + 평균 · 표준편차 · 최소 · 최대 (+ 앙상블 · Buy & Hold · 벤치마크)."""
    per = pd.DataFrame({f"seed_{s}": r["performance"].loc["JM strategy"] for s, r in seed_results.items()}).T
    num = per.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    stats = pd.DataFrame({"mean": num.mean(), "std": num.std(ddof=1), "min": num.min(), "max": num.max()}).T
    ref = next(iter(seed_results.values()))["performance"]
    parts = [per, stats]
    if ensemble_perf is not None:
        parts.append(ensemble_perf.loc[["Ensemble strategy"]].rename(index={"Ensemble strategy": "ensemble"}))
    parts.append(ref.drop(index="JM strategy"))
    table = pd.concat(parts)
    table.index.name = "portfolio"
    return table


def _seed_states(seed_results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    rows = [{"seed": s, **{k: r["current_state"][k] for k in ("asof", "label", "prob_bear", "days_in_regime")}}
            for s, r in seed_results.items()]
    return pd.DataFrame(rows)


def run_multi_seed(cfg: PipelineConfig) -> Dict[str, Any]:
    """여러 시드로 같은 설정을 돌린다 (Jump Model n_init 초기값 · Mamba 가중치 초기화/배치 순서가 시드에 따라 바뀐다).

    seed_mode
      individual  시드마다 전체 파이프라인을 seed_<s>/ 에 저장하고, 전략 성과의 평균·표준편차를 seed_performance.csv 로
      ensemble    시드별 국면 확률을 평균한 앙상블 국면으로 한 번 백테스트해 ensemble/ 에 저장 (HMM · 에피소드 분석은 생략)
      both        둘 다 (앙상블은 individual 실행 결과를 재사용하므로 추가 학습이 없다)
    --inference 와 함께 쓰면 백테스트 대신 현재 반기 추론 결과를 시드별 · 앙상블로 저장한다.
    seed_workers > 1 이면 시드 여러 개를 별도 프로세스에서 동시에 돌린다 (seed_devices 가 있으면 GPU 를 돌아가며 배정).
    """
    seeds = cfg.seed_list
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    _write_json(cfg.to_dict(), out, "run_config.json")
    individual = cfg.seed_mode in ("individual", "both")
    ensemble = cfg.seed_mode in ("ensemble", "both")
    logger.info("multi-seed run: seeds=%s, mode=%s%s%s", list(seeds), cfg.seed_mode,
                "" if cfg.ensemble_bear_vote is None else f", bear vote >= {cfg.ensemble_bear_vote:.0%}",
                " (inference)" if cfg.inference else "")

    seed_results: Dict[int, Dict[str, Any]] = {}
    regimes: Dict[int, pd.DataFrame] = {}
    mamba_logs: List[pd.DataFrame] = []
    _check_devices(cfg)  # GPU 가 없으면 데이터 처리 전에 실패
    if individual:
        jobs = {s: (_individual_job, (_seed_cfg(cfg, s, os.path.join(out, f"seed_{s}"), i),))
                for i, s in enumerate(seeds)}
        seed_results = _run_seed_jobs(cfg, jobs)
        regimes = {s: r["regimes"] for s, r in seed_results.items()}
        data = seed_results[seeds[0]]["data"]
    else:
        data, signal_col, X, context = prepare_features(cfg)
        signal = data[signal_col]
        start = cfg.oos_start
        if cfg.inference:
            dates = refit_schedule(X.index, cfg.refit_months, cfg.min_train)
            if not dates:
                raise ValueError("추론할 반기 시작점이 없습니다 (데이터 부족)")
            start = dates[-1]
        fitted = _run_seed_jobs(cfg, {s: (_ensemble_job, (_seed_cfg(cfg, s, out, i), X, signal, start, context))
                                      for i, s in enumerate(seeds)})
        regimes = {s: reg for s, (reg, _) in fitted.items()}
        mamba_logs = [log for _, log in fitted.values() if log is not None]

    result: Dict[str, Any] = {"config": cfg, "seeds": list(seeds), "seed_results": seed_results or None,
                              "seed_regimes": regimes, "data": data, "seed_performance": None, "ensemble": None,
                              "current_state": None, "performance": None}
    if individual:
        states = _seed_states(seed_results)
        _save(states, out, "seed_current_state.csv", index=False)
        result["seed_current_state"] = states

    n_states, bear = cfg.n_states, cfg.n_states - 1
    ens_perf = None
    if ensemble:
        ens_dir = os.path.join(out, "ensemble")
        os.makedirs(ens_dir, exist_ok=True)
        ens = ensemble_regimes(regimes, n_states, cfg.ensemble_bear_vote)
        current = _current_state(ens, n_states)
        current.update({"seeds": list(seeds), "seed_agreement": float(ens["agreement"].iloc[-1]),
                        "bear_vote": float(ens["bear_vote"].iloc[-1]), "bear_vote_threshold": cfg.ensemble_bear_vote,
                        "seed_labels": {str(s): state_label(int(r["regime"].iloc[-1]), n_states)
                                        for s, r in regimes.items()}})
        ens_out: Dict[str, Any] = {"regimes": ens, "current_state": current}
        prefix = "inference_" if cfg.inference else ""
        _save(ens, ens_dir, f"{prefix}regimes.csv")
        _save(pd.concat(mamba_logs, ignore_index=True) if mamba_logs else None, ens_dir,
              f"{prefix}mamba_train_log.csv", index=False)
        _write_json(current, ens_dir, "current_state.json")
        if not cfg.inference:
            kw = _strategy_kwargs(cfg, bear)
            strategy = run_0_1_strategy(ens["regime"], data, delay=cfg.delay, **kw)
            ens_perf = performance_table(strategy, name="Ensemble strategy")
            summary = regime_summary(ens["regime"], ens["signal_ret"], n_states)
            delay_table = delay_robustness_table(ens["regime"], data, cfg.delays, **kw)
            _save(strategy, ens_dir, "strategy.csv")
            _save(ens_perf, ens_dir, "performance.csv")
            _save(summary, ens_dir, "regime_summary.csv", index=False)
            _save(delay_table, ens_dir, "delay_robustness.csv")
            ens_out.update(strategy=strategy, performance=ens_perf, regime_summary=summary,
                           delay_robustness=delay_table)
            if cfg.plots:
                others = None
                if seed_results:
                    others = {f"seed {s}": r["strategy"] for s, r in seed_results.items()}
                vote_note = ("" if cfg.ensemble_bear_vote is None
                             else f", bear if vote >= {cfg.ensemble_bear_vote:.0%}")
                plotting.plot_regimes_cumret(strategy, bear, os.path.join(ens_dir, "regimes_cumret.png"), others,
                                             title=f"Seed ensemble ({len(seeds)} seeds{vote_note})",
                                             label="Ensemble strategy")
                plotting.plot_delay_robustness(delay_table, os.path.join(ens_dir, "delay_robustness.png"))
        elif cfg.plots:
            plotting.plot_inference(ens, data["close"], bear, os.path.join(ens_dir, "inference.png"))
        result.update(ensemble=ens_out, current_state=current, performance=ens_perf)
        logger.info("ensemble current regime: %s", current)

    if individual and not cfg.inference:
        table = seed_performance_table(seed_results, ens_perf)
        _save(table, out, "seed_performance.csv")
        result["seed_performance"] = table
    return result


# ---------------------------------------------------------------------------
# 타깃별 분리 학습 (--separate)
# ---------------------------------------------------------------------------

def _target_cfg(cfg: PipelineConfig, index: int, out_dir: str) -> PipelineConfig:
    """--separate 에서 index 번째 타깃 하나만 예측하도록 학습하는 설정 (출력별 손실 가중치는 그 타깃 몫만,
    입력 피처는 --separate-features 에 준 그 타깃의 피처, 없으면 전부, jump penalty 는 --separate-jump-penalty 에
    준 그 타깃의 값, 없으면 --jump-penalty)."""
    n_h = len(cfg.mamba_horizons)
    lw = cfg.mamba_loss_weights
    if len(lw) != n_h:
        lw = lw[index * n_h:(index + 1) * n_h]
    vote = cfg.ensemble_bear_vote if len(cfg.seed_list) > 1 and cfg.seed_mode != "individual" else None
    target = cfg.mamba_targets[index]
    use = tuple(cfg.separate_features.get(target, cfg.use_features))
    jp = float(cfg.separate_jump_penalty.get(target, cfg.jump_penalty))
    return dataclasses.replace(cfg, mamba_targets=(target,), mamba_loss_weights=tuple(lw), use_features=use,
                               jump_penalty=jp, separate=False, separate_weights=(), separate_features={},
                               separate_jump_penalty={}, ensemble_bear_vote=vote, out_dir=out_dir)


def blend_name(targets: Sequence[str], weights: Sequence[float]) -> str:
    """비중 이름: 한 타깃만 쓰면 그 타깃 이름, 아니면 return0.7_mdd0.3 처럼 타깃과 비중."""
    used = [t for t, w in zip(targets, weights) if w > 0]
    if len(used) == 1:
        return used[0]
    return "_".join(f"{t}{w:.4g}" for t, w in zip(targets, weights))


def run_separate(cfg: PipelineConfig) -> Dict[str, Any]:
    """--separate: mamba_targets 를 타깃마다 따로 학습하고 국면 확률을 비중대로 섞는다.

    타깃마다 그 타깃 하나만 예측하도록 학습한 Mamba → Jump Model 을 target_<타깃>/ 에 따로 돌린다
    (시드가 여러 개면 run_multi_seed 그대로). --separate-features 를 주면 타깃마다 그 피처만 입력으로 쓴다
    (타깃마다 남는 날짜가 다르면 섞을 때는 공통 날짜만), --separate-jump-penalty 를 주면 타깃마다 그 jump penalty
    로 Jump Model 을 학습한다. 상태 순서는 모두 같은 규칙(0 = bull, 마지막 = bear)이라
    날짜마다 Σ 타깃 비중 × (그 타깃의 시드 평균 상태 확률) 로 섞고 argmax 를 국면으로 삼는다
    (--ensemble-bear-vote 면 비중으로 가중한 bear 표 비율). separate_blends 의 비중마다 blend_<이름>/ 에 저장하고,
    타깃 단독(비중 1)과 함께 separate_performance.csv · separate_current_state.csv 로 한 번에 비교한다.
    --inference 와 함께 쓰면 백테스트 없이 현재 반기 국면만 섞는다.
    """
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    _write_json(cfg.to_dict(), out, "run_config.json")
    _check_devices(cfg)  # GPU 가 없으면 데이터 처리 전에 실패
    targets = list(cfg.mamba_targets)
    if cfg.separate_features or cfg.use_features:  # 피처 이름 오류는 첫 타깃 학습이 끝난 뒤가 아니라 학습 전에 낸다
        columns = prepare_features(dataclasses.replace(cfg, use_features=(), mamba_warmup_context=False))[2].columns
        for t in targets:
            opt = "--separate-features" if t in cfg.separate_features else "--use-features"
            try:
                expand_feature_names(columns, cfg.separate_features.get(t, cfg.use_features))
            except KeyError as e:
                raise KeyError(f"{opt} {t}: {e.args[0]}") from None
    logger.info("separate run: targets=%s, blends=%s%s", targets,
                ["/".join(f"{v:.3g}" for v in w) for w in cfg.separate_blends], " (inference)" if cfg.inference else "")

    target_results: Dict[str, Dict[str, Any]] = {}
    members: Dict[str, pd.DataFrame] = {}  # 섞을 개별 모델 (타깃 × 시드) 의 국면표
    owner: Dict[str, int] = {}             # 모델 → 타깃 번호
    for i, t in enumerate(targets):
        tcfg = _target_cfg(cfg, i, os.path.join(out, f"target_{t}"))
        logger.info("target %s (%d/%d), jump penalty %g, features: %s", t, i + 1, len(targets), tcfg.jump_penalty,
                    ",".join(tcfg.use_features) or "all")
        res = run_pipeline(tcfg)
        target_results[t] = res
        regs = res["seed_regimes"] if len(tcfg.seed_list) > 1 else {None: res["regimes"]}
        for s, reg in regs.items():
            key = t if s is None else f"{t}_seed{s}"
            members[key], owner[key] = reg, i
    data = target_results[targets[0]]["data"]
    n_states, bear = cfg.n_states, cfg.n_states - 1
    n_own = [sum(o == i for o in owner.values()) for i in range(len(targets))]

    # 타깃 단독(one-hot) + 섞는 비중 (one-hot 비중은 타깃 단독과 같으므로 한 번만)
    entries = [(t, tuple(float(j == i) for j in range(len(targets)))) for i, t in enumerate(targets)]
    entries += [(blend_name(targets, w), w) for w in cfg.separate_blends if max(w) < 1.0]
    prefix = "inference_" if cfg.inference else ""
    kw = _strategy_kwargs(cfg, bear)
    runs: Dict[str, Dict[str, Any]] = {}
    for name, w in entries:
        reg = ensemble_regimes(members, n_states, cfg.ensemble_bear_vote, column="regime_{}",
                               weights={m: w[owner[m]] / n_own[owner[m]] for m in members})
        for i, t in enumerate(targets):  # 타깃별 (시드 평균) 상태 확률
            own = [m for m in members if owner[m] == i]
            for k in range(n_states):
                reg[f"prob_{k}_{t}"] = sum(members[m].loc[reg.index, f"prob_{k}"] for m in own) / len(own)
        current = _current_state(reg, n_states)
        current.update({"weights": dict(zip(targets, w)), "agreement": float(reg["agreement"].iloc[-1]),
                        "bear_vote": float(reg["bear_vote"].iloc[-1]), "bear_vote_threshold": cfg.ensemble_bear_vote,
                        "target_prob_bear": {t: float(reg[f"prob_{bear}_{t}"].iloc[-1]) for t in targets}})
        run: Dict[str, Any] = {"weights": w, "regimes": reg, "current_state": current}
        if not cfg.inference:
            run["strategy"] = run_0_1_strategy(reg["regime"], data, delay=cfg.delay, **kw)
            run["performance"] = performance_table(run["strategy"], name="Blend strategy")
        runs[name] = run

    blends = [name for name, _ in entries[len(targets):]]
    for name in blends:
        run, d = runs[name], os.path.join(out, f"blend_{name}")
        os.makedirs(d, exist_ok=True)
        _save(run["regimes"], d, f"{prefix}regimes.csv")
        _write_json(run["current_state"], d, "current_state.json")
        if cfg.inference:
            if cfg.plots:
                plotting.plot_inference(run["regimes"], data["close"], bear, os.path.join(d, "inference.png"))
            continue
        run["regime_summary"] = regime_summary(run["regimes"]["regime"], run["regimes"]["signal_ret"], n_states)
        run["delay_robustness"] = delay_robustness_table(run["regimes"]["regime"], data, cfg.delays, **kw)
        _save(run["strategy"], d, "strategy.csv")
        _save(run["performance"], d, "performance.csv")
        _save(run["regime_summary"], d, "regime_summary.csv", index=False)
        _save(run["delay_robustness"], d, "delay_robustness.csv")
        if cfg.plots:
            desc = ", ".join(f"{t} {v:.0%}" for t, v in zip(targets, run["weights"]))
            plotting.plot_regimes_cumret(run["strategy"], bear, os.path.join(d, "regimes_cumret.png"),
                                         {t: runs[t]["strategy"] for t in targets},
                                         title=f"Separate targets blend ({desc})", label="Blend strategy")
            plotting.plot_delay_robustness(run["delay_robustness"], os.path.join(d, "delay_robustness.png"))

    wcols = [f"w_{t}" for t in targets]
    states = pd.DataFrame([{"name": n, **dict(zip(wcols, r["weights"])),
                            **{k: r["current_state"][k] for k in ("asof", "label", "prob_bear", "days_in_regime",
                                                                  "bear_vote")}}
                           for n, r in runs.items()])
    _save(states, out, "separate_current_state.csv", index=False)
    table = None
    if not cfg.inference:
        per = pd.DataFrame({n: r["performance"].loc["Blend strategy"] for n, r in runs.items()}).T
        per = pd.concat([pd.DataFrame([r["weights"] for r in runs.values()], index=per.index, columns=wcols), per],
                        axis=1)
        table = pd.concat([per, runs[targets[0]]["performance"].drop(index="Blend strategy")])
        table.index.name = "portfolio"
        _save(table, out, "separate_performance.csv")
        if cfg.plots:
            main = blends[0] if blends else targets[0]
            plotting.plot_regimes_cumret(runs[main]["strategy"], bear, os.path.join(out, "separate_cumret.png"),
                                         {n: r["strategy"] for n, r in runs.items() if n != main},
                                         title=f"Separate targets ({', '.join(targets)})", label=main)

    first = runs[blends[0]] if blends else None
    return {"config": cfg, "targets": targets, "target_results": target_results, "data": data, "blends": runs,
            "separate_current_state": states, "separate_performance": table,
            "current_state": first["current_state"] if first else None,
            "performance": first.get("performance") if first else None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _int_list(text: str) -> List[int]:
    return [int(t) for t in str(text).replace(" ", "").split(",") if t]


def _float_list(text: str) -> List[float]:
    return [float(t) for t in str(text).replace(" ", "").split(",") if t]


def _str_list(text: str) -> List[str]:
    return [t for t in str(text).replace(" ", "").lower().split(",") if t]


def _target_features(text: str) -> tuple:
    """``타깃=피처,피처`` (또는 ``타깃:피처,..``) → (타깃, (피처, ..)). 피처 이름은 대소문자를 그대로 둔다."""
    text = str(text).strip()
    cut = min((i for i in (text.find("="), text.find(":")) if i > 0), default=-1)
    if cut < 0:
        raise argparse.ArgumentTypeError(f"타깃=피처,피처 형식이어야 합니다 (예: mdd=DD-log_10,sortino_20), got {text!r}")
    names = tuple(t.strip() for t in text[cut + 1:].split(",") if t.strip())
    return text[:cut].strip().lower(), names


def _target_value(text: str) -> tuple:
    """``타깃=값`` (또는 ``타깃:값``) → (타깃, 값)."""
    text = str(text).strip()
    cut = min((i for i in (text.find("="), text.find(":")) if i > 0), default=-1)
    try:
        if cut < 0:
            raise ValueError
        return text[:cut].strip().lower(), float(text[cut + 1:])
    except ValueError:
        raise argparse.ArgumentTypeError(f"타깃=값 형식이어야 합니다 (예: mdd=30), got {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    d = PipelineConfig()
    p = argparse.ArgumentParser(description="Jump Model 국면 파이프라인",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", nargs="?", help="입력 파일 (날짜·종가·무위험금리; csv/xlsx)")
    p.add_argument("--input", dest="input_opt", help="입력 파일 (위치 인자 대신)")
    p.add_argument("--out", "--out-dir", dest="out_dir", default=d.out_dir, help="출력 폴더")

    g = p.add_argument_group("데이터")
    g.add_argument("--date-col", help="날짜 열 (기본: 자동 인식)")
    g.add_argument("--close-col", help="종가 열 (기본: 자동 인식)")
    g.add_argument("--rf-col", help="무위험금리 열 (기본: 자동 인식)")
    g.add_argument("--rf-unit", choices=RF_UNITS, default=d.rf_unit, help="무위험금리 단위")
    g.add_argument("--rf-const", type=float, help="파일 대신 쓸 상수 무위험금리 (rf-unit 단위)")
    g.add_argument("--relative-benchmark", metavar="PATH", help="벤치마크(예: 코스피) 파일 → rel_ret 생성")
    g.add_argument("--bench-date-col", help="벤치마크 날짜 열")
    g.add_argument("--bench-close-col", help="벤치마크 종가 열")
    g.add_argument("--start", help="데이터 사용 시작일")
    g.add_argument("--end", help="데이터 사용 종료일")
    g.add_argument("--signal-ret", choices=("auto", "absolute", "relative"), default=d.signal_ret,
                   help="모델이 학습할 수익률 (auto: 벤치마크가 있으면 rel_ret)")
    g.add_argument("--backtest-ret", choices=("absolute", "relative"), default=d.backtest_ret,
                   help="absolute: bear=무위험자산 / relative: bear=벤치마크, 초과성과(IR)로 평가")

    g = p.add_argument_group("피처")
    g.add_argument("--feature-set", choices=FEATURE_SETS, default=d.feature_set)
    g.add_argument("--extra-features", nargs="+", action="extend", default=[], metavar="FILE:COL:TRANSFORM",
                   help="사용자 변수. 파일 생략 시 입력 파일의 열. 변환: none, log, diff_n, pct_n, logdiff_n, "
                        "zscore_w, ewm_hl, lag_n ('+' 로 연결)")
    g.add_argument("--remove-series", nargs="+", action="extend", default=[], metavar="NAME",
                   help="제거할 피처 계열(예: var DD-log) 또는 피처 이름")
    g.add_argument("--use-features", nargs="+", action="extend", default=[], metavar="NAME",
                   help="이 피처(예: sortino_20) / 계열(예: sortino)만 입력으로 씀 (콤마 가능, --remove-series 뒤). "
                        "기본: 전부. --separate 에서는 --separate-features 를 안 준 타깃에 적용")
    g.add_argument("--warmup", type=int, default=d.warmup, help="앞에서 버릴 행 수")

    g = p.add_argument_group("모델")
    g.add_argument("--model", choices=("jm", "sjm"), default=d.model)
    g.add_argument("--discrete", action="store_true", help="이산형 모델 (기본: 연속형 cont=True)")
    g.add_argument("--n-states", type=int, default=d.n_states)
    g.add_argument("--jump-penalty", type=float, default=d.jump_penalty)
    g.add_argument("--max-feats", type=float, help="sjm 유효 피처 수 (기본: 피처 수의 1/3, 최소 2)")
    g.add_argument("--pin-features", nargs="+", action="extend", default=[], metavar="NAME",
                   help="sjm 에서 꼭 남길 피처/계열")
    g.add_argument("--grid-size", type=float, default=d.grid_size, help="연속형 모델 확률 격자 크기")
    g.add_argument("--n-init", type=int, default=d.n_init, help="모델 초기값 개수")
    g.add_argument("--clip-mul", type=float, default=d.clip_mul, help="학습창 기준 클리핑 σ 배수")
    g.add_argument("--center-distance", action="store_true",
                   help="regimes.csv 에 날짜별 상태 중심점과의 거리 dist_0.. 를 추가 (표준화 · sjm 가중 공간)")
    g.add_argument("--seed", type=int, default=d.seed)

    g = p.add_argument_group("여러 시드")
    g.add_argument("--n-seeds", type=int, default=d.n_seeds, help="시드 개수 (--seed, --seed+1, ...)")
    g.add_argument("--seeds", type=_int_list, help="시드 목록 (콤마, --n-seeds 대신)")
    g.add_argument("--seed-mode", choices=SEED_MODES, default=d.seed_mode,
                   help="ensemble: 국면 확률 평균으로 백테스트 / individual: 시드별 백테스트 후 성과 평균 / both")
    g.add_argument("--seed-workers", type=int, default=d.seed_workers,
                   help="동시에 실행할 시드 수 (시드마다 별도 프로세스; GPU 메모리 · CPU 코어에 맞춰 조정)")
    g.add_argument("--seed-devices", type=_str_list, metavar="DEVS",
                   help="(mamba) 시드를 돌아가며 배정할 GPU (콤마, 예: cuda:0,cuda:1). 기본: --device")
    g.add_argument("--ensemble-bear-vote", type=float, metavar="X",
                   help="앙상블에서 bear 로 판정한 시드 비율이 X 이상이면 bear (0~1, 예: 0.4). 기본: 확률 평균 argmax")

    g = p.add_argument_group("Mamba 인코더 (--encoder mamba, CUDA GPU 필요)")
    g.add_argument("--encoder", choices=("none", "mamba"), default=d.encoder,
                   help="mamba: 재추정마다 Mamba 를 학습해 피처 대신 hidden 벡터를 --model 에 넣음")
    g.add_argument("--device", default=d.device, help="auto(첫 번째 GPU) | cuda | cuda:N")
    g.add_argument("--mamba-seq-len", type=int, default=d.mamba_seq_len, help="입력 시퀀스 길이 (거래일)")
    g.add_argument("--mamba-warmup-context", action="store_true",
                   help="시퀀스가 피처 시작 전으로 넘어가면 0 대신 --warmup 으로 버린 행으로 채움")
    g.add_argument("--mamba-d-model", type=int, default=d.mamba_d_model, help="hidden 벡터 차원")
    g.add_argument("--mamba-d-state", type=int, default=d.mamba_d_state)
    g.add_argument("--mamba-d-conv", type=int, default=d.mamba_d_conv)
    g.add_argument("--mamba-expand", type=int, default=d.mamba_expand)
    g.add_argument("--mamba-layers", type=int, default=d.mamba_layers)
    g.add_argument("--mamba-dropout", type=float, default=d.mamba_dropout)
    g.add_argument("--mamba-epochs", type=int, default=d.mamba_epochs, help="재추정당 최대 epoch")
    g.add_argument("--mamba-patience", type=int, default=d.mamba_patience, help="early stopping patience")
    g.add_argument("--mamba-batch-size", type=int, default=d.mamba_batch_size)
    g.add_argument("--mamba-lr", type=float, default=d.mamba_lr)
    g.add_argument("--mamba-valid-frac", type=float, default=d.mamba_valid_frac, help="학습창 뒤쪽 검증 비율")
    g.add_argument("--mamba-horizons", type=_int_list, default=list(d.mamba_horizons),
                   help="동시에 예측할 미래 기간 (거래일, 콤마). 1,5,20 = 다음 날·1주·1달 multi-output")
    g.add_argument("--mamba-targets", type=_str_list, default=list(d.mamba_targets),
                   help="예측할 값 (콤마): return(수익률 합) | vol(실현 변동성) | mdd(최대 낙폭). "
                        "여러 개면 타깃 × horizon 을 동시에 예측")
    g.add_argument("--mamba-loss-weights", type=_float_list, metavar="W",
                   help="출력별 MSE 가중치 (콤마). horizon 수만큼 주면 horizon 별(예: 0.5,0.25,0.25), "
                        "타깃 × horizon 수만큼 주면 출력별(타깃 → horizon 순). 합으로 정규화. 기본: 단순 평균")
    g.add_argument("--separate", action="store_true",
                   help="--mamba-targets 를 함께 학습하지 않고 타깃마다 따로 학습한 뒤 국면 확률을 비중대로 섞음")
    g.add_argument("--separate-weights", type=_float_list, nargs="+", action="extend", default=[], metavar="W",
                   help="(--separate) 섞을 타깃별 비중 (콤마, 타깃 순서, 합으로 정규화). 여러 개 주면 비중마다 결과. "
                        "타깃 2개면 첫 타깃 비중 하나로도 가능 (예: 0 0.3 0.5 0.7 1). 기본: 균등")
    g.add_argument("--separate-features", type=_target_features, nargs="+", action="extend", default=[],
                   metavar="TARGET=FEATS",
                   help="(--separate) 타깃별 Mamba 입력 피처/계열 (콤마; --feature-set · --extra-features 로 만든 피처 중). "
                        "예: mdd=DD-log_10,sortino_20 return=sortino,VIX[log]. 안 준 타깃은 전부 사용")
    g.add_argument("--separate-jump-penalty", type=_target_value, nargs="+", action="extend", default=[],
                   metavar="TARGET=JP",
                   help="(--separate) 타깃별 Jump Model jump penalty. 예: return=80 mdd=30. 안 준 타깃은 --jump-penalty")

    g = p.add_argument_group("롤링 재추정")
    g.add_argument("--train-window", type=int, default=d.train_window, help="최대 학습창 (거래일)")
    g.add_argument("--min-train", type=int, default=d.min_train, help="최소 학습창 (거래일)")
    g.add_argument("--refit-months", type=_int_list, default=list(d.refit_months), help="재추정 월 (콤마)")
    g.add_argument("--oos-start", help="이 날짜 이후 재추정부터 백테스트")

    g = p.add_argument_group("백테스트")
    g.add_argument("--delay", type=int, default=d.delay, help="신호 → 체결 지연 (거래일)")
    g.add_argument("--delays", type=_int_list, default=list(d.delays), help="로버스트니스 표의 지연값 (콤마)")
    g.add_argument("--min-cash", type=float, default=d.min_cash, help="bull 일 때 현금 비중")
    g.add_argument("--max-cash", type=float, default=d.max_cash, help="bear 일 때 현금 비중")
    g.add_argument("--cost-bps", type=float, default=d.cost_bps, help="편도 거래비용 (bp)")
    g.add_argument("--cost-buy-bps", type=float, help="매수 비용 (bp, 기본: --cost-bps)")
    g.add_argument("--cost-sell-bps", type=float, help="매도 비용 (bp, 기본: --cost-bps)")

    g = p.add_argument_group("HMM 벤치마크")
    g.add_argument("--hmm", action="store_true", help="Gaussian HMM 벤치마크 비교")
    g.add_argument("--hmm-states", type=int, default=d.hmm_states)
    g.add_argument("--hmm-refit", type=int, default=d.hmm_refit, help="재추정 간격 (거래일)")
    g.add_argument("--hmm-median", type=int, default=d.hmm_median, help="trailing median filter 창")
    g.add_argument("--hmm-n-init", type=int, default=d.hmm_n_init)

    g = p.add_argument_group("해석 / 실행")
    g.add_argument("--top-k", type=int, default=d.top_k, help="유사 에피소드 개수")
    g.add_argument("--inference", action="store_true", help="현재 반기만 추론 (백테스트 없음)")
    g.add_argument("--no-plots", action="store_true", help="png 저장 안 함")
    g.add_argument("--summary-file", metavar="PATH",
                   help="실행이 끝날 때마다 성과 요약을 이 txt 파일에 이어 쓴다 (여러 옵션 일괄 실행용)")
    g.add_argument("-v", "--verbose", action="store_true")
    g.add_argument("-q", "--quiet", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    values = vars(args).copy()
    input_path = values.pop("input_opt") or values.pop("input")
    values.pop("input", None)
    values["cont"] = not values.pop("discrete")
    values["plots"] = not values.pop("no_plots")
    values["refit_months"] = tuple(values["refit_months"])
    values["delays"] = tuple(values["delays"])
    values["seeds"] = tuple(values["seeds"] or ())
    values["seed_devices"] = tuple(values["seed_devices"] or ())
    values["mamba_horizons"] = tuple(values["mamba_horizons"])
    values["mamba_targets"] = tuple(values["mamba_targets"])
    values["mamba_loss_weights"] = tuple(values["mamba_loss_weights"] or ())
    values["use_features"] = tuple(values["use_features"])
    values["separate_weights"] = tuple(tuple(w) for w in values["separate_weights"])
    separate_features: Dict[str, tuple] = {}
    for target, names in values["separate_features"]:
        if target in separate_features:
            raise ValueError(f"--separate-features 에 타깃 '{target}' 이 두 번 있습니다")
        separate_features[target] = names
    values["separate_features"] = separate_features
    separate_jp: Dict[str, float] = {}
    for target, value in values["separate_jump_penalty"]:
        if target in separate_jp:
            raise ValueError(f"--separate-jump-penalty 에 타깃 '{target}' 이 두 번 있습니다")
        separate_jp[target] = value
    values["separate_jump_penalty"] = separate_jp
    for key in ("verbose", "quiet", "summary_file"):
        values.pop(key)
    return PipelineConfig(input=input_path or "", **values)


def append_summary(path: str, argv: Sequence[str], out_dir: str, table: Optional[pd.DataFrame],
                   error: Optional[str] = None) -> None:
    """성과 요약 한 블록(실행 명령 · 출력 폴더 · 성과표 또는 오류)을 path 에 이어 쓴다."""
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = ["=" * 80, f"[{stamp}] python run_pipeline.py {' '.join(argv)}", f"out: {os.path.abspath(out_dir)}"]
    if error is not None:
        lines.append(f"FAILED: {error}")
    elif table is not None:
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200,
                               "display.max_columns", None):
            lines.append(table.to_string())
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if not (args.input or args.input_opt):
        parser.error("입력 파일이 필요합니다")
    try:
        cfg = config_from_args(args)
        result = run_pipeline(cfg)
    except (ValueError, KeyError, FileNotFoundError) as e:
        logger.error("%s", e)
        if args.summary_file:
            append_summary(args.summary_file, argv, args.out_dir, None, error=str(e))
        return 1

    print()
    for _, row in result.get("seed_current_state", pd.DataFrame()).iterrows():
        print(f"[{row['asof']}] seed {row['seed']}: {row['label']} (bear 확률 {row['prob_bear']:.1%}, "
              f"{row['days_in_regime']}일째)")
    for _, row in result.get("separate_current_state", pd.DataFrame()).iterrows():
        print(f"[{row['asof']}] {row['name']}: {row['label']} (bear 확률 {row['prob_bear']:.1%}, "
              f"{row['days_in_regime']}일째)")
    cur = result["current_state"]
    if cur and "separate_current_state" not in result:
        name =f"앙상블({len(cur['seeds'])}개 시드) 국면" if "seeds" in cur else "현재 국면"
        agree = f", 시드 일치 {cur['seed_agreement']:.0%}" if "seed_agreement" in cur else ""
        if cur.get("bear_vote_threshold") is not None:
            agree += f", bear 표 {cur['bear_vote']:.0%} (기준 {cur['bear_vote_threshold']:.0%})"
        print(f"[{cur['asof']}] {name}: {cur['label']} (bear 확률 {cur['prob_bear']:.1%}, "
              f"{cur['days_in_regime']}일째, 재추정 {cur['refit_date']}{agree})")
    perf = result.get("separate_performance")
    if perf is None:
        perf = result.get("seed_performance")
    if perf is None:
        perf = result.get("performance")
    table = None
    if perf is not None:
        cols = [c for c in perf.columns if c.startswith("w_")]
        cols += [c for c in ("cagr", "ann_vol", "sharpe", "max_drawdown", "information_ratio", "n_trades")
                if c in perf.columns]
        table = perf[cols].apply(pd.to_numeric, errors="coerce")
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
            print(table.to_string())
    if args.summary_file:
        append_summary(args.summary_file, argv, cfg.out_dir, table)
    print(f"결과: {os.path.abspath(cfg.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
