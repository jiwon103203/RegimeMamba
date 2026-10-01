#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Jump Model 국면 파이프라인 진입점.

입력 파일(날짜·종가·무위험금리)
  │  data_io.py         1) 로드·정제 → 초과수익률 (+ 벤치마크 차감 rel_ret)
  │  features.py        2) EWM downside deviation · Sortino (+ extra / 커스텀 변수)
  │  rolling.py         3) 6개월마다 재추정(3000일 학습창) + 사이 구간 온라인 추론
  │  mamba_encoder.py      (--model mamba) 재추정마다 Mamba 학습 → hidden 벡터를 Jump Model 입력으로 (GPU)
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
    python run_pipeline.py data.csv --inference
    python run_pipeline.py data.csv --model mamba --feature-set example --device cuda:0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from regime_jm import mamba_encoder, plotting
from regime_jm.backtest import delay_robustness_table, performance_table, regime_summary, run_0_1_strategy
from regime_jm.config import PipelineConfig
from regime_jm.data_io import (RF_UNITS, align_to_index, load_series, parse_extra_spec, prepare_inputs,
                               resolve_signal_return)
from regime_jm.features import FEATURE_SETS, apply_transform, build_features, expand_feature_names
from regime_jm.hmm_benchmark import run_rolling_hmm
from regime_jm.regime_episodes import (compare_episode_paths, episode_metrics, episode_paths, extract_episodes,
                                       length_scenarios, rank_similar_episodes)
from regime_jm.rolling import RollingJMResult, refit_schedule, run_rolling_jm, state_label
from regime_jm.weights import current_weight_summary, group_weights, weight_groups_long

logger = logging.getLogger("regime_jm")


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
    """1)~2) 단계: 입력 → (data, 신호 수익률 열 이름, 피처 행렬)."""
    data = prepare_inputs(cfg.input, date_col=cfg.date_col, close_col=cfg.close_col, rf_col=cfg.rf_col,
                          rf_unit=cfg.rf_unit, rf_const=cfg.rf_const, benchmark=cfg.relative_benchmark,
                          bench_date_col=cfg.bench_date_col, bench_close_col=cfg.bench_close_col,
                          start=cfg.start, end=cfg.end)
    signal_col = resolve_signal_return(data, cfg.signal_ret)
    extra = load_extra_features(cfg.extra_features, cfg.input, data.index, cfg.date_col)
    X = build_features(data[signal_col], cfg.feature_set, extra=extra, remove=cfg.remove_series,
                       warmup=cfg.warmup)
    logger.info("signal=%s, features=%d (%s), rows=%d (%s ~ %s)", signal_col, X.shape[1], cfg.feature_set,
                len(X), X.index[0].date(), X.index[-1].date())
    return data, signal_col, X


def _make_encoder(cfg: PipelineConfig) -> Optional[mamba_encoder.MambaEncoder]:
    if cfg.model != "mamba":
        return None
    device = mamba_encoder.resolve_device(cfg.device)
    logger.info("mamba encoder on %s (seq_len=%d, d_model=%d, layers=%d)", device, cfg.seq_len, cfg.d_model,
                cfg.n_layers)
    return mamba_encoder.MambaEncoder(cfg.mamba_settings(), device)


def _fit_rolling(cfg: PipelineConfig, X: pd.DataFrame, signal: pd.Series, start=None,
                 encoder=None) -> RollingJMResult:
    pinned = expand_feature_names(X.columns, cfg.pin_features)
    model = "jm" if cfg.model == "mamba" else cfg.model  # mamba: hidden 벡터에 Jump Model
    return run_rolling_jm(X, signal, model=model, cont=cfg.cont, n_states=cfg.n_states,
                          jump_penalty=cfg.jump_penalty, max_feats=cfg.max_feats, pinned=pinned or None,
                          train_window=cfg.train_window, min_train=cfg.min_train, refit_months=cfg.refit_months,
                          start=start, clip_mul=cfg.clip_mul, grid_size=cfg.grid_size, n_init=cfg.n_init,
                          random_state=cfg.seed, encoder=encoder)


def _regime_table(res: RollingJMResult, signal: pd.Series) -> pd.DataFrame:
    reg = res.regimes.copy()
    reg.insert(1, "label", [state_label(s, res.n_states) for s in reg["regime"]])
    reg["signal_ret"] = signal.reindex(reg.index)
    return reg


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
    if cfg.inference:
        return run_inference(cfg)
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    _write_json(cfg.to_dict(), out, "run_config.json")
    encoder = _make_encoder(cfg)  # GPU 가 없으면 데이터 처리 전에 실패

    # 1)~2) 데이터 · 피처
    data, signal_col, X = prepare_features(cfg)
    signal = data[signal_col]

    # 3) 롤링 재추정 + 온라인 추론 (mamba: 재추정마다 Mamba 학습 → hidden 벡터)
    res = _fit_rolling(cfg, X, signal, start=cfg.oos_start, encoder=encoder)
    reg = _regime_table(res, signal)

    # 4) 백테스트
    strat_kwargs = dict(bear_state=res.bear_state, min_cash=cfg.min_cash, max_cash=cfg.max_cash,
                        cost_buy=cfg.cost_buy, cost_sell=cfg.cost_sell, mode=cfg.backtest_ret)
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
    out = cfg.out_dir
    os.makedirs(out, exist_ok=True)
    _write_json(cfg.to_dict(), out, "run_config.json")
    encoder = _make_encoder(cfg)

    data, signal_col, X = prepare_features(cfg)
    signal = data[signal_col]
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
        "model": cfg.model + ("" if cfg.cont else " (discrete)"),
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
# CLI
# ---------------------------------------------------------------------------

def _int_list(text: str) -> List[int]:
    return [int(t) for t in str(text).replace(" ", "").split(",") if t]


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
    g.add_argument("--warmup", type=int, default=d.warmup, help="앞에서 버릴 행 수")

    g = p.add_argument_group("모델")
    g.add_argument("--model", choices=("jm", "sjm", "mamba"), default=d.model,
                   help="mamba: 재추정마다 Mamba 를 학습해 hidden 벡터를 Jump Model 에 넣음 (CUDA GPU 필요)")
    g.add_argument("--discrete", action="store_true", help="이산형 모델 (기본: 연속형 cont=True)")
    g.add_argument("--n-states", type=int, default=d.n_states)
    g.add_argument("--jump-penalty", type=float, default=d.jump_penalty)
    g.add_argument("--max-feats", type=float, help="sjm 유효 피처 수 (기본: 피처 수의 1/3, 최소 2)")
    g.add_argument("--pin-features", nargs="+", action="extend", default=[], metavar="NAME",
                   help="sjm 에서 꼭 남길 피처/계열")
    g.add_argument("--grid-size", type=float, default=d.grid_size, help="연속형 모델 확률 격자 크기")
    g.add_argument("--n-init", type=int, default=d.n_init, help="모델 초기값 개수")
    g.add_argument("--clip-mul", type=float, default=d.clip_mul, help="학습창 기준 클리핑 σ 배수")
    g.add_argument("--seed", type=int, default=d.seed)

    g = p.add_argument_group("Mamba (--model mamba)")
    g.add_argument("--device", default=d.device,
                   help="GPU 장치: auto(첫 번째 GPU), cuda, cuda:N. mamba-ssm 은 CUDA 전용이라 GPU 가 없으면 에러")
    g.add_argument("--seq-len", type=int, default=d.seq_len, help="Mamba 입력 시퀀스 길이 (거래일)")
    g.add_argument("--d-model", type=int, default=d.d_model, help="hidden 벡터 차원 (= Jump Model 입력 차원)")
    g.add_argument("--d-state", type=int, default=d.d_state, help="SSM 상태 차원")
    g.add_argument("--d-conv", type=int, default=d.d_conv, help="Mamba conv 커널 크기")
    g.add_argument("--expand", type=int, default=d.expand, help="Mamba 확장 계수")
    g.add_argument("--n-layers", type=int, default=d.n_layers, help="Mamba 블록 수")
    g.add_argument("--dropout", type=float, default=d.dropout)
    g.add_argument("--epochs", type=int, default=d.epochs, help="재추정당 최대 학습 epoch")
    g.add_argument("--patience", type=int, default=d.patience, help="early stopping patience (epoch)")
    g.add_argument("--batch-size", type=int, default=d.batch_size)
    g.add_argument("--lr", type=float, default=d.lr, help="AdamW 학습률")
    g.add_argument("--valid-frac", type=float, default=d.valid_frac, help="학습창 뒤쪽 검증 비율 (early stopping)")

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
    for key in ("verbose", "quiet"):
        values.pop(key)
    return PipelineConfig(input=input_path or "", **values)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if not (args.input or args.input_opt):
        parser.error("입력 파일이 필요합니다")
    try:
        cfg = config_from_args(args)
        result = run_pipeline(cfg)
    except (ValueError, KeyError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 1

    cur = result["current_state"]
    print(f"\n[{cur['asof']}] 현재 국면: {cur['label']} (bear 확률 {cur['prob_bear']:.1%}, "
          f"{cur['days_in_regime']}일째, 재추정 {cur['refit_date']})")
    if "performance" in result:
        cols = [c for c in ("cagr", "ann_vol", "sharpe", "max_drawdown", "information_ratio", "n_trades")
                if c in result["performance"].columns]
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
            print(result["performance"][cols].to_string())
    print(f"결과: {os.path.abspath(cfg.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
