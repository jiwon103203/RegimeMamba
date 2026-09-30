"""Generic rolling-window backtest loop shared by the scripts in ``scripts/``.

A script only has to provide ``process_window(window_info, window_dir)`` which
trains a model for that window and returns the smoothing-method results of
:func:`regime_mamba.evaluate.smoothing_eval.evaluate_smoothing_methods` (or
``None`` to skip the window). Scheduling, checkpointing, result aggregation and
the final comparison plots are handled here.
"""

import copy
import json
import logging
import os
import traceback
from collections import defaultdict
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from ..utils.io import json_serializer, load_checkpoint, save_checkpoint
from .clustering import predict_regimes
from .rolling_window_w_train import identify_regimes_for_window, train_model_for_window
from .schedule import create_window_schedule
from .smoothing_eval import evaluate_smoothing_methods, visualize_final_comparison

ProcessWindowFn = Callable[[Dict[str, Any], str], Optional[Dict[str, Dict[str, Any]]]]


def new_combined_results():
    """Empty per-method accumulator ``{method_id: {window, returns, trades, sharpes, performances}}``."""
    return defaultdict(lambda: {
        'window': [],
        'returns': {},
        'trades': {},
        'sharpes': {},
        'performances': []
    })


def record_window_results(combined_results, methods_results: Dict[str, Dict[str, Any]], window_number: int):
    """Append one window's smoothing-method results to the accumulator."""
    for method_id, result in methods_results.items():
        combined_results[method_id]['window'].append(window_number)
        combined_results[method_id]['returns'][window_number] = result['cum_return']
        combined_results[method_id]['trades'][window_number] = result['n_trades']
        combined_results[method_id]['sharpes'][window_number] = result['sharpe']
        combined_results[method_id]['performances'].append(copy.deepcopy(result['performance']))


def serialize_combined_results(combined_results, include_performances: bool = True) -> Dict[str, Any]:
    """Convert the accumulator to a JSON friendly dict (window keys become strings)."""
    data = {}
    for method, result in combined_results.items():
        data[method] = {
            'window': result['window'],
            'returns': {str(k): v for k, v in result['returns'].items()},
            'trades': {str(k): v for k, v in result['trades'].items()},
            'sharpes': {str(k): v for k, v in result['sharpes'].items()}
        }
        if include_performances:
            data[method]['performances'] = result['performances']
    return data


def restore_combined_results(checkpoint: Dict[str, Any]) -> Tuple[Any, int]:
    """Rebuild the accumulator from a checkpoint; returns ``(combined_results, next_window)``."""
    combined_results = new_combined_results()
    for method, result in checkpoint['results'].items():
        combined_results[method]['window'] = result['window']
        combined_results[method]['returns'] = {int(k): v for k, v in result['returns'].items()}
        combined_results[method]['trades'] = {int(k): v for k, v in result['trades'].items()}
        combined_results[method]['sharpes'] = {int(k): v for k, v in result['sharpes'].items()}
        combined_results[method]['performances'] = result['performances']
    return combined_results, checkpoint['next_window']


def run_windowed_backtest(
    config,
    logger: logging.Logger,
    process_window: ProcessWindowFn,
    checkpoint_path: Optional[str] = None,
    use_clustering: bool = True,
    log_prefix: str = "",
    title_prefix: str = "",
    checkpoint_extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run ``process_window`` over the rolling-window schedule and compare methods

    Args:
        config: Configuration object (schedule fields, results_dir, checkpoint options)
        logger: Logger instance
        process_window: ``(window_info, window_dir) -> methods_results | None``
        checkpoint_path: Path to a checkpoint to resume from (optional)
        use_clustering: Whether the schedule contains a clustering period
        log_prefix: Prefix for log messages (e.g. ``[RL] ``)
        title_prefix: Prefix for final comparison plot titles
        checkpoint_extra: Extra fields stored in every checkpoint (e.g. ``{'mode': 'e2e'}``)

    Returns:
        Dict[str, Any]: ``{'combined_results': ..., 'summary': ...}``
    """
    combined_results = new_combined_results()

    start_from_window = 1
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            combined_results, start_from_window = restore_combined_results(load_checkpoint(checkpoint_path))
            logger.info(f"Resuming from checkpoint at window {start_from_window}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            logger.info("Starting from beginning")

    window_schedule = create_window_schedule(config, start_from_window, use_clustering=use_clustering)
    if not window_schedule:
        logger.info("No windows to process")
        return {'combined_results': combined_results, 'summary': None}

    total_windows = len(window_schedule)
    logger.info(
        f"{log_prefix}Processing {total_windows} windows from "
        f"{window_schedule[0]['window_number']} to {window_schedule[-1]['window_number']}"
    )

    enable_checkpointing = getattr(config, 'enable_checkpointing', False)
    checkpoint_interval = getattr(config, 'checkpoint_interval', 1) or 1

    for i, window_info in enumerate(window_schedule):
        window_number = window_info['window_number']
        logger.info(f"\n=== {log_prefix}Window {window_number} ({i+1}/{total_windows}) ===")

        window_dir = os.path.join(config.results_dir, f"window_{window_number}")
        os.makedirs(window_dir, exist_ok=True)

        logger.info(f"Training period: {window_info['train_period']['start']} to {window_info['train_period']['end']}")
        logger.info(f"Validation period: {window_info['valid_period']['start']} to {window_info['valid_period']['end']}")
        if 'clustering_period' in window_info:
            logger.info(f"Clustering period: {window_info['clustering_period']['start']} to {window_info['clustering_period']['end']}")
        logger.info(f"Forward period: {window_info['forward_period']['start']} to {window_info['forward_period']['end']}")

        try:
            methods_results = process_window(window_info, window_dir)
            if not methods_results:
                continue

            logger.info(f"{log_prefix}Found {len(methods_results)} valid results")
            record_window_results(combined_results, methods_results, window_number)

            if enable_checkpointing and (i + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'next_window': window_number + 1,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    **(checkpoint_extra or {}),
                    'results': serialize_combined_results(combined_results)
                }
                save_checkpoint(checkpoint_data, os.path.join(config.results_dir, 'checkpoint.json'))
                logger.info(f"Checkpoint saved after window {window_number}")

        except Exception as e:
            logger.error(f"{log_prefix}Error processing window {window_number}: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning(f"Skipping window {window_number}")

    logger.info(f"{log_prefix}Creating final comparison...")
    if not combined_results:
        logger.warning(f"{log_prefix}No results to compare")
        return {'combined_results': combined_results, 'summary': None}

    summary = visualize_final_comparison(combined_results, config.results_dir, title_prefix=title_prefix)

    result_data = serialize_combined_results(combined_results, include_performances=False)
    with open(os.path.join(config.results_dir, 'combined_results.json'), 'w') as f:
        json.dump(result_data, f, default=json_serializer, indent=4)

    logger.info(f"{log_prefix}Backtest complete with {len(result_data)} methods across {len(window_schedule)} windows")
    return {'combined_results': combined_results, 'summary': summary}


def run_two_stage_window(
    config,
    data: pd.DataFrame,
    window_info: Dict[str, Any],
    window_dir: str,
    logger: logging.Logger,
    parallel: bool = False,
    loader_workers: int = 2
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Original 2-stage flow for one window: train Mamba -> K-Means regimes -> smoothing methods.

    With ``config.jump_model`` the Mamba features are fed to ``ModifiedJumpModel``,
    which writes its own per-window outputs, so ``None`` is returned.
    """
    window_number = window_info['window_number']
    train_args = (
        config,
        window_info['train_period']['start'],
        window_info['train_period']['end'],
        window_info['valid_period']['start'],
        window_info['valid_period']['end'],
        data,
    )

    if config.jump_model:
        logger.info("Training jump model...")
        model = train_model_for_window(*train_args, window_number=window_number)
        if model is None:
            logger.warning("Jump model training failed, skipping window")
            return None
        # 미래 기간에 대한 예측 (결과는 window 디렉토리에 저장됨)
        model.predict(
            window_info['forward_period']['start'],
            window_info['forward_period']['end'],
            data,
            window_number,
            sort="cumret"
        )
        logger.info("Jump model results saved")
        return None

    logger.info("Training model...")
    model, _ = train_model_for_window(*train_args, window_number=window_number)
    if model is None:
        logger.warning("Model training failed, skipping window")
        return None

    if config.lstm:
        logger.info("Skipping regime identification for this model")
        kmeans, bull_regime = None, None
    else:
        logger.info("Identifying regimes...")
        kmeans, bull_regime = identify_regimes_for_window(
            config,
            model,
            data,
            window_info['clustering_period']['start'],
            window_info['clustering_period']['end']
        )
        if kmeans is None or bull_regime is None:
            logger.warning("Regime identification failed, skipping window")
            return None

    logger.info("Evaluating smoothing methods...")
    predict_fn = partial(predict_regimes, model, kmeans=kmeans, bull_regime=bull_regime, config=config)
    return evaluate_smoothing_methods(
        predict_fn,
        data,
        config,
        window_info['forward_period'],
        window_dir,
        parallel=parallel,
        max_workers=getattr(config, 'max_workers', None),
        loader_workers=loader_workers
    )
