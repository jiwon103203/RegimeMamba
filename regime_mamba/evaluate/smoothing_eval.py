"""Evaluate every smoothing method on one forward window and compare them.

Raw regimes come from the model through ``predict_fn``::

    predict_fn(dataloader) -> (raw_predictions, true_returns, dates)

``predict_fn`` must be picklable (a module-level function or ``functools.partial``)
when ``parallel=True`` because methods are then evaluated in worker processes.
"""

import json
import logging
import os
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import DateRangeRegimeMambaDataset
from ..utils.io import json_serializer
from .smoothing import apply_smoothing_method, get_smoothing_methods
from .strategy import evaluate_regime_strategy

PredictFn = Callable[[DataLoader], Tuple[np.ndarray, np.ndarray, Any]]


def evaluate_smoothing_method(
    predict_fn: PredictFn,
    data: pd.DataFrame,
    method_info: Tuple[str, Dict[str, Any]],
    config,
    forward_period: Dict[str, str],
    loader_workers: int = 2,
    log_prefix: str = ""
) -> Optional[Dict[str, Any]]:
    """Evaluate a single smoothing method

    Args:
        predict_fn: Callable returning ``(raw_predictions, true_returns, dates)`` for a dataloader
        data: Full dataframe
        method_info: Tuple of (method_name, parameters)
        config: Configuration object
        forward_period: Dictionary with forward start and end dates
        loader_workers: DataLoader worker count
        log_prefix: Prefix for log messages (e.g. ``[E2E]``)

    Returns:
        Dict[str, Any]: Result dictionary, or None if the window has too little data
    """
    method_name, params = method_info
    forward_start, forward_end = forward_period['start'], forward_period['end']

    try:
        forward_dataset = DateRangeRegimeMambaDataset(
            data=data,
            seq_len=config.seq_len,
            start_date=forward_start,
            end_date=forward_end,
            config=config
        )

        if len(forward_dataset) < 10:
            return None

        forward_loader = DataLoader(
            forward_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=loader_workers
        )

        raw_predictions, true_returns, dates = predict_fn(forward_loader)
        smoothed_predictions = apply_smoothing_method(raw_predictions, method_name, params)

        # Evaluate strategy with transaction costs
        results_df, performance = evaluate_regime_strategy(
            smoothed_predictions,
            true_returns,
            dates,
            transaction_cost=config.transaction_cost,
            config=config
        )

        if results_df is None or performance is None:
            return None

        results_df['smoothing_method'] = method_name
        for param_name, param_value in params.items():
            results_df[f'smoothing_{param_name}'] = param_value
        results_df['raw_regime'] = raw_predictions.flatten()

        raw_trades = (np.diff(raw_predictions.flatten()) != 0).sum() + (raw_predictions[0] == 1)
        smoothed_trades = (np.diff(smoothed_predictions.flatten()) != 0).sum() + (smoothed_predictions[0] == 1)

        performance['smoothing_method'] = method_name
        for param_name, param_value in params.items():
            performance[f'smoothing_{param_name}'] = param_value
        performance['raw_trades'] = int(raw_trades)
        performance['smoothed_trades'] = int(smoothed_trades)
        performance['trade_reduction'] = int(raw_trades - smoothed_trades)
        performance['trade_reduction_pct'] = ((raw_trades - smoothed_trades) / raw_trades * 100) if raw_trades > 0 else 0

        param_str = '_'.join([f"{k}={v}" for k, v in params.items()]) if params else "default"
        method_id = f"{method_name}_{param_str}" if params else method_name

        return {
            'method_id': method_id,
            'method_name': method_name,
            'params': params,
            'df': results_df,
            'performance': performance,
            'cum_return': performance['cumulative_returns']['strategy'],
            'n_trades': performance['trading_metrics']['number_of_trades'],
            'sharpe': performance['sharpe_ratio']['strategy']
        }

    except Exception as e:
        logging.error(f"{log_prefix}Error evaluating method {method_name}: {str(e)}")
        traceback.print_exc()
        return None


def _save_method_result(result: Dict[str, Any], window_results_dir: str):
    method_dir = os.path.join(window_results_dir, result['method_id'])
    os.makedirs(method_dir, exist_ok=True)
    result['df'].to_csv(os.path.join(method_dir, 'results.csv'), index=False)
    with open(os.path.join(method_dir, 'performance.json'), 'w') as f:
        json.dump(result['performance'], f, default=json_serializer, indent=4)


def evaluate_smoothing_methods(
    predict_fn: PredictFn,
    data: pd.DataFrame,
    config,
    forward_period: Dict[str, str],
    window_results_dir: str,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    loader_workers: int = 2,
    log_prefix: str = "",
    title_prefix: str = ""
) -> Dict[str, Dict[str, Any]]:
    """Evaluate every method of :func:`get_smoothing_methods` on one forward window

    Each result is saved to ``<window_results_dir>/<method_id>/{results.csv,performance.json}``
    and a ``methods_comparison.png`` is drawn.

    Args:
        predict_fn: See module docstring
        data: Full dataframe
        config: Configuration object
        forward_period: Dictionary with forward start and end dates
        window_results_dir: Window results directory
        parallel: Evaluate methods in a process pool (CPU only)
        max_workers: Maximum number of worker processes
        loader_workers: DataLoader worker count
        log_prefix: Prefix for log / progress messages
        title_prefix: Prefix for plot titles

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of method results keyed by method id
    """
    smoothing_methods = get_smoothing_methods()
    all_methods_results = {}
    desc = f"{log_prefix}Evaluating methods"

    if not parallel:
        for method_info in tqdm(smoothing_methods, desc=desc):
            result = evaluate_smoothing_method(
                predict_fn, data, method_info, config, forward_period, loader_workers, log_prefix
            )
            if result:
                all_methods_results[result['method_id']] = result
                _save_method_result(result, window_results_dir)
    else:
        max_workers = max_workers or min(len(smoothing_methods), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    evaluate_smoothing_method,
                    predict_fn, data, method_info, config, forward_period, loader_workers, log_prefix
                )
                for method_info in smoothing_methods
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                try:
                    result = future.result()
                    if result:
                        all_methods_results[result['method_id']] = result
                        _save_method_result(result, window_results_dir)
                except Exception as e:
                    logging.error(f"{log_prefix}Error processing future: {str(e)}")

    if all_methods_results:
        visualize_methods_comparison(
            all_methods_results,
            os.path.join(window_results_dir, 'methods_comparison.png'),
            forward_period,
            title_prefix=title_prefix
        )

    return all_methods_results


def visualize_methods_comparison(
    methods_results: Dict[str, Dict[str, Any]],
    save_path: str,
    forward_period: Dict[str, str],
    title_prefix: str = ""
):
    """Visualize comparison of smoothing methods

    Args:
        methods_results: Dictionary of method results
        save_path: Path to save visualization
        forward_period: Dictionary with forward start and end dates
        title_prefix: Prefix for the plot title
    """
    plt.figure(figsize=(15, 10))

    # Cumulative returns
    plt.subplot(2, 1, 1)
    first_method = list(methods_results.keys())[0]
    plt.plot(
        methods_results[first_method]['df']['Cum_Market'] * 100,
        label='Market',
        color='gray',
        linestyle='--'
    )
    for method_id, result in methods_results.items():
        plt.plot(result['df']['Cum_Strategy'] * 100, label=method_id)

    plt.title(f"{title_prefix}Comparison of Smoothing Methods ({forward_period['start']} to {forward_period['end']})")
    plt.ylabel('Cumulative Returns (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Returns vs trades
    plt.subplot(2, 1, 2)
    method_ids = list(methods_results.keys())
    returns = [methods_results[method_id]['cum_return'] for method_id in method_ids]
    trades = [methods_results[method_id]['n_trades'] for method_id in method_ids]

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.bar(np.arange(len(method_ids)) - 0.2, returns, width=0.4, color='blue', alpha=0.7)
    ax1.set_ylabel('Returns (%)', color='blue')
    ax1.tick_params(axis='y', colors='blue')

    ax2.bar(np.arange(len(method_ids)) + 0.2, trades, width=0.4, color='red', alpha=0.7)
    ax2.set_ylabel('Number of Trades', color='red')
    ax2.tick_params(axis='y', colors='red')

    plt.xticks(np.arange(len(method_ids)), method_ids, rotation=45, ha='right')
    plt.title('Returns vs. Number of Trades by Method')
    plt.grid(True, alpha=0.3)

    custom_lines = [
        Line2D([0], [0], color='blue', lw=4, alpha=0.7),
        Line2D([0], [0], color='red', lw=4, alpha=0.7)
    ]
    plt.legend(custom_lines, ['Returns (%)', 'Number of Trades'])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_final_comparison(
    combined_results: Dict[str, Dict[str, Any]],
    save_dir: str,
    title_prefix: str = ""
) -> Dict[str, Any]:
    """Visualize final comparison of smoothing methods across all windows

    Writes ``overall_metrics_comparison.png``, ``cumulative_returns.png``,
    ``best_method_histogram.png`` and ``methods_summary.json`` to ``save_dir``.

    Args:
        combined_results: Combined results dictionary
        save_dir: Directory to save visualizations
        title_prefix: Prefix for plot titles / log messages

    Returns:
        Dict[str, Any]: Summary dictionary
    """
    all_methods = sorted(list(combined_results.keys()))
    all_windows = sorted(list(set(combined_results[all_methods[0]]['window'])))

    method_metrics = {}
    for method in all_methods:
        returns = [combined_results[method]['returns'][window] for window in all_windows]
        trades = [combined_results[method]['trades'][window] for window in all_windows]
        sharpes = [combined_results[method]['sharpes'][window] for window in all_windows]

        method_metrics[method] = {
            'avg_return': np.mean(returns),
            'avg_trades': np.mean(trades),
            'avg_sharpe': np.mean(sharpes),
            'returns': returns,
            'cum_returns': np.cumsum(returns),
            'windows': all_windows
        }

    sorted_methods = sorted(all_methods, key=lambda x: method_metrics[x]['avg_return'], reverse=True)
    sharpe_sorted = sorted(all_methods, key=lambda x: method_metrics[x]['avg_sharpe'], reverse=True)

    # 1. Average metrics comparison
    plt.figure(figsize=(15, 12))
    for pos, (key, color, title, ylabel) in enumerate([
        ('avg_return', 'blue', 'Average Returns by Method (%)', 'Average Return (%)'),
        ('avg_trades', 'red', 'Average Trades by Method', 'Average Trades'),
        ('avg_sharpe', 'green', 'Average Sharpe Ratio by Method', 'Average Sharpe Ratio'),
    ], start=1):
        plt.subplot(2, 2, pos)
        plt.bar(
            range(len(sorted_methods)),
            [method_metrics[method][key] for method in sorted_methods],
            color=color,
            alpha=0.7
        )
        plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45, ha='right')
        plt.title(f'{title_prefix}{title}')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)

    # Returns vs. trades scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(
        [method_metrics[method]['avg_trades'] for method in all_methods],
        [method_metrics[method]['avg_return'] for method in all_methods],
        color='purple',
        alpha=0.7
    )
    for method in all_methods:
        plt.annotate(
            method,
            (method_metrics[method]['avg_trades'], method_metrics[method]['avg_return']),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontsize=8
        )
    plt.xlabel('Average Trades')
    plt.ylabel('Average Return (%)')
    plt.title(f'{title_prefix}Returns vs. Trades')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'overall_metrics_comparison.png'))
    plt.close()

    # 2. Cumulative returns comparison
    plt.figure(figsize=(12, 8))
    for method in sorted_methods:
        plt.plot(
            all_windows,
            method_metrics[method]['cum_returns'],
            marker='o',
            markersize=4,
            label=method
        )
    plt.title(f'{title_prefix}Cumulative Returns by Method')
    plt.xlabel('Window')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cumulative_returns.png'))
    plt.close()

    # 3. Best method by window
    best_methods = []
    for window in all_windows:
        window_returns = {method: combined_results[method]['returns'][window] for method in all_methods}
        best_methods.append(max(window_returns.items(), key=lambda x: x[1])[0])

    best_method_counts = Counter(best_methods)
    sorted_best_methods = sorted(best_method_counts.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(12, 6))
    plt.bar(
        range(len(sorted_best_methods)),
        [count for _, count in sorted_best_methods],
        color='orange',
        alpha=0.7
    )
    plt.xticks(
        range(len(sorted_best_methods)),
        [method for method, _ in sorted_best_methods],
        rotation=45,
        ha='right'
    )
    plt.title(f'{title_prefix}Best Method by Window')
    plt.ylabel('Number of Windows')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_method_histogram.png'))
    plt.close()

    # 4. Summary dictionary
    summary = {
        'methods': {},
        'best_methods': {
            'by_return': sorted_methods[0],
            'by_sharpe': sharpe_sorted[0],
            'by_frequency': sorted_best_methods[0][0] if sorted_best_methods else None
        },
        'windows': len(all_windows),
        'total_methods': len(all_methods)
    }
    for method in all_methods:
        summary['methods'][method] = {
            'avg_return': float(method_metrics[method]['avg_return']),
            'avg_trades': float(method_metrics[method]['avg_trades']),
            'avg_sharpe': float(method_metrics[method]['avg_sharpe']),
            'best_window_count': best_method_counts.get(method, 0)
        }

    with open(os.path.join(save_dir, 'methods_summary.json'), 'w') as f:
        json.dump(summary, f, default=json_serializer, indent=4)

    logging.info(f"\n===== {title_prefix}Smoothing Method Performance Summary =====")
    logging.info("Top methods by average return:")
    for i, method in enumerate(sorted_methods[:3]):
        logging.info(f"  {i+1}. {method}: {method_metrics[method]['avg_return']:.2f}%")

    logging.info("\nTop methods by average Sharpe ratio:")
    for i, method in enumerate(sharpe_sorted[:3]):
        logging.info(f"  {i+1}. {method}: {method_metrics[method]['avg_sharpe']:.2f}")

    logging.info("\nMost frequently best methods:")
    for i, (method, count) in enumerate(sorted_best_methods[:3]):
        logging.info(f"  {i+1}. {method}: {count} windows")

    return summary
