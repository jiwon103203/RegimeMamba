"""Regime signal smoothing methods compared per window by ``smoothing_eval``."""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

def apply_regime_smoothing(regime_predictions, method="ma", window=5, threshold=0.5):
    """
    Apply various smoothing techniques to regime predictions

    Args:
        regime_predictions: Original regime predictions (1=Bull, 0=Bear)
        method: Smoothing method ('ma'=moving average, 'exp'=exponential smoothing)
        window: Smoothing window size
        threshold: Regime decision threshold

    Returns:
        smoothed_regimes: Smoothed regimes (1=Bull, 0=Bear)
    """
    regime_series = pd.Series(regime_predictions.flatten())

    # Apply smoothing based on method
    if method == "ma":
        # Apply moving average
        smoothed_probs = regime_series.rolling(window=window, center=False).mean()
        # Fill NaN values with first valid value
        smoothed_probs.fillna(regime_series.iloc[0], inplace=True)

    elif method == "exp":
        # Apply exponential moving average
        smoothed_probs = regime_series.ewm(span=window, adjust=False).mean()
    else:
        raise ValueError(f"Unsupported smoothing method: {method}")

    # Apply threshold to determine final regime
    smoothed_regimes = (smoothed_probs > threshold).astype(int)

    return smoothed_regimes.values

def apply_confirmation_rule(regime_predictions, confirmation_days=3):
    """
    Apply confirmation rule to regime changes - only change regime after N consecutive days of same signal

    Args:
        regime_predictions: Original regime predictions (1=Bull, 0=Bear)
        confirmation_days: Number of consecutive days required to confirm regime change

    Returns:
        confirmed_regimes: Regimes with confirmation rule applied (1=Bull, 0=Bear)
    """
    regimes = regime_predictions.flatten()
    confirmed_regimes = np.copy(regimes)

    # Set initial regime
    current_regime = regimes[0]
    confirmation_count = 1

    # Apply confirmation rule for each day
    for i in range(1, len(regimes)):
        if regimes[i] == current_regime:
            # If same as current regime, maintain confirmation count
            confirmed_regimes[i] = current_regime
            confirmation_count = min(confirmation_count + 1, confirmation_days)
        else:
            # Different regime signal
            if confirmation_count >= confirmation_days:
                # Previous regime sufficiently confirmed
                confirmation_count = 1
                current_regime = regimes[i]
                confirmed_regimes[i] = current_regime
            else:
                # Not confirmed yet - maintain previous regime
                confirmed_regimes[i] = current_regime
                confirmation_count += 1

    return confirmed_regimes

def apply_minimum_holding_period(regime_predictions, returns=None, min_holding_days=20):
    """
    Apply minimum holding period rule - maintain regime for at least N days after change

    Args:
        regime_predictions: Original regime predictions (1=Bull, 0=Bear)
        returns: Return data (optional, for return-based exit rules)
        min_holding_days: Minimum holding period (days)

    Returns:
        filtered_regimes: Regimes with minimum holding period applied (1=Bull, 0=Bear)
    """
    regimes = regime_predictions.flatten()
    filtered_regimes = np.copy(regimes)

    # Set initial regime
    current_regime = regimes[0]
    days_since_change = 0

    # Apply minimum holding period rule for each day
    for i in range(1, len(regimes)):
        if days_since_change < min_holding_days:
            # Minimum holding period not elapsed - maintain existing regime
            filtered_regimes[i] = current_regime
            days_since_change += 1
        else:
            # Minimum holding period elapsed - regime change allowed
            if regimes[i] != current_regime:
                # Regime change
                current_regime = regimes[i]
                days_since_change = 0
            filtered_regimes[i] = current_regime

    return filtered_regimes

def get_smoothing_methods() -> List[Tuple[str, Dict[str, Any]]]:
    """Get list of smoothing methods to evaluate

    Returns:
        List[Tuple[str, Dict[str, Any]]]: List of (method_name, parameters) tuples
    """
    return [
        ('none', {}),
        ('ma', {'window': 3}),
        ('ma', {'window': 5}),
        ('exp', {'window': 5}),
        ('confirmation', {'days': 1}),
        ('confirmation', {'days': 2}),
        ('confirmation', {'days': 3}),
        ('min_holding', {'days': 10}),
        ('min_holding', {'days': 20}),
        ('min_holding', {'days': 30}),
        ('min_holding', {'days': 60}),
    ]


def apply_smoothing_method(
    raw_predictions: np.ndarray,
    method_name: str,
    params: Dict[str, Any]
) -> np.ndarray:
    """Apply a specific smoothing method to raw predictions

    Args:
        raw_predictions: Raw regime predictions
        method_name: Smoothing method name ('none', 'ma', 'exp', 'confirmation', 'min_holding')
        params: Smoothing parameters

    Returns:
        np.ndarray: Smoothed predictions
    """
    if method_name == 'none':
        return raw_predictions
    elif method_name == 'ma':
        window = params.get('window', 10)
        return apply_regime_smoothing(
            raw_predictions, method='ma', window=window
        ).reshape(-1, 1)
    elif method_name == 'exp':
        window = params.get('window', 10)
        return apply_regime_smoothing(
            raw_predictions, method='exp', window=window
        ).reshape(-1, 1)
    elif method_name == 'confirmation':
        days = params.get('days', 3)
        return apply_confirmation_rule(
            raw_predictions, confirmation_days=days
        ).reshape(-1, 1)
    elif method_name == 'min_holding':
        days = params.get('days', 20)
        return apply_minimum_holding_period(
            raw_predictions, min_holding_days=days
        ).reshape(-1, 1)
    else:
        logging.warning(f"Unknown smoothing method: {method_name}, using raw predictions")
        return raw_predictions
