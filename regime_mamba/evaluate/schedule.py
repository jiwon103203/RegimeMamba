"""Rolling-window schedule shared by the backtest scripts."""

from datetime import datetime
from typing import Any, Dict, List

from dateutil.relativedelta import relativedelta


def create_window_schedule(
    config,
    start_from_window: int = 1,
    use_clustering: bool = True
) -> List[Dict[str, Any]]:
    """Create window schedule for rolling window backtest

    Each window ends at ``current_date``::

        [train_start ... train_end][valid (+ clustering) ... current_date][forward ... forward_end]

    Args:
        config: Configuration object (start_date, end_date, total_window_years,
            valid_years, clustering_years, forward_months)
        start_from_window: Window number to start from (for resuming)
        use_clustering: If False, the clustering period is omitted and
            ``clustering_years`` is ignored (E2E / RL models need no K-Means step)

    Returns:
        List[Dict[str, Any]]: List of window dictionaries
    """
    window_schedule = []

    current_date = datetime.strptime(config.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(config.end_date, '%Y-%m-%d')
    clustering_years = config.clustering_years if use_clustering else 0

    window_number = 1
    while current_date <= end_date:
        # e.g. current 2007-04-20, total 25y, valid 10y, clustering 10y -> train 1982-04-20 ~ 1987-04-20
        train_start = (current_date - relativedelta(years=config.total_window_years)).strftime('%Y-%m-%d')
        train_end = (current_date - relativedelta(years=config.valid_years + clustering_years)).strftime('%Y-%m-%d')

        valid_start = (current_date - relativedelta(years=config.valid_years)).strftime('%Y-%m-%d')
        valid_end = current_date.strftime('%Y-%m-%d')

        forward_start = current_date.strftime('%Y-%m-%d')
        forward_end = (current_date + relativedelta(months=config.forward_months)).strftime('%Y-%m-%d')

        if window_number >= start_from_window:
            window = {
                'window_number': window_number,
                'train_period': {'start': train_start, 'end': train_end},
                'valid_period': {'start': valid_start, 'end': valid_end},
                'forward_period': {'start': forward_start, 'end': forward_end},
                'current_date': current_date.strftime('%Y-%m-%d')
            }
            if use_clustering:
                window['clustering_period'] = {
                    'start': (current_date - relativedelta(years=clustering_years)).strftime('%Y-%m-%d'),
                    'end': current_date.strftime('%Y-%m-%d')
                }
            window_schedule.append(window)

        current_date += relativedelta(months=config.forward_months)
        window_number += 1

    return window_schedule
