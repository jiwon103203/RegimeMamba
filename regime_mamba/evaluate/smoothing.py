import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from tqdm import tqdm
import os

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

def apply_probability_threshold(hidden_states, kmeans, bull_regime, threshold=0.6):
    """
    Determine regimes based on cluster probability threshold

    Args:
        hidden_states: Model's hidden states
        kmeans: Trained KMeans model
        bull_regime: Bull regime cluster ID
        threshold: Probability threshold (must exceed this to belong to cluster)

    Returns:
        regime_predictions: Regimes with probability threshold applied (1=Bull, 0=Bear)
    """
    # Calculate distances between each data point and cluster centers
    distances = kmeans.transform(hidden_states)

    # Convert distances to probabilities (using negative distances so closer = higher probability)
    neg_distances = -distances
    probabilities = np.exp(neg_distances) / np.sum(np.exp(neg_distances), axis=1, keepdims=True)

    # Probability of belonging to Bull regime cluster
    bull_probabilities = probabilities[:, bull_regime]

    # Apply threshold: Bull probability > threshold -> Bull(1), otherwise Bear(0)
    regime_predictions = (bull_probabilities > threshold).astype(int)

    return regime_predictions

def apply_filtering(predictions, method='minimum_holding', window=10, confirmation_days=3, min_holding_days=20, threshold=0.6):
    """
    Apply filtering to regime predictions

    Args:
        predictions: Original regime predictions (1=Bull, 0=Bear)
        method: Filtering method ('smoothing', 'confirmation', 'minimum_holding', 'none')
        window: Moving average window size
        confirmation_days: Days required for confirmation
        min_holding_days: Minimum holding period (days)
        threshold: Probability threshold

    Returns:
        filtered_predictions: Filtered regime predictions
    """
    # Convert predictions to 1D array
    preds = predictions.flatten()

    if method == 'none':
        return predictions

    elif method == 'smoothing':
        # Apply moving average
        return apply_regime_smoothing(predictions, method="ma", window=window, threshold=threshold)

    elif method == 'exp':
        # Apply exponential moving average
        return apply_regime_smoothing(predictions, method="exp", window=window, threshold=threshold)
    elif method == 'confirmation':
        # Apply regime change confirmation rule
        return apply_confirmation_rule(predictions, confirmation_days=confirmation_days).reshape(-1, 1)

    elif method == 'minimum_holding':
        # Apply minimum holding period
        return apply_minimum_holding_period(predictions, min_holding_days=min_holding_days).reshape(-1, 1)

    else:
        print(f"Unknown filtering method: {method}, no filtering applied")
        return predictions

def predict_regimes_with_filtering(model, dataloader, kmeans, bull_regime, config,
                                  filter_method='minimum_holding',
                                  window=10,
                                  confirmation_days=3,
                                  min_holding_days=20,
                                  probability_threshold=0.6):
    """
    Predict regimes with filtering applied

    Args:
        model: Model to evaluate
        dataloader: Data loader
        kmeans: Trained KMeans model
        bull_regime: Bull regime cluster ID
        config: Configuration object
        filter_method: Filtering method ('smoothing', 'confirmation', 'minimum_holding', 'probability', 'none')
        Other filtering-related parameters

    Returns:
        predictions: Filtered regime predictions (1=Bull, 0=Bear)
        true_returns: Actual returns
        dates: Date information
        raw_predictions: Original unfiltered regime predictions
        hidden_states: Model's hidden states
    """
    model.eval()
    hidden_states_list = []
    predictions = []
    true_returns = []
    dates = []

    with torch.no_grad():
        for x, y, date, r in dataloader:
            x = x.to(config.device)
            _, hidden = model(x, return_hidden=True)
            hidden_cpu = hidden.cpu().numpy()

            # Cluster assignment
            cluster = kmeans.predict(hidden_cpu)

            # Apply probability-based regime determination if selected
            if filter_method == 'probability':
                # Calculate distances to clusters
                distances = kmeans.transform(hidden_cpu)
                # Convert to probabilities
                neg_distances = -distances
                probabilities = np.exp(neg_distances) / np.sum(np.exp(neg_distances), axis=1, keepdims=True)
                # Bull regime probability
                bull_probs = probabilities[:, bull_regime]
                # Apply threshold
                regime_pred = (bull_probs > probability_threshold).astype(int)
            else:
                # Basic regime prediction (1 for Bull regime, 0 otherwise)
                regime_pred = np.where(cluster == bull_regime, 1, 0)

            hidden_states_list.append(hidden_cpu)
            predictions.extend(regime_pred)
            true_returns.extend(r.numpy())
            dates.extend(date)


    # Convert to NumPy arrays
    predictions = np.array(predictions)
    true_returns = np.array(true_returns)
    hidden_states = np.vstack(hidden_states_list)

    # Save original predictions
    raw_predictions = predictions.copy()

    # Apply selected filtering method
    if filter_method == 'smoothing':
        predictions = apply_regime_smoothing(predictions, method="ma", window=window)
    elif filter_method == 'exp':
        predictions = apply_regime_smoothing(predictions, method="exp", window=window)
    elif filter_method == 'confirmation':
        predictions = apply_confirmation_rule(predictions, confirmation_days=confirmation_days)
    elif filter_method == 'minimum_holding':
        predictions = apply_minimum_holding_period(predictions, min_holding_days=min_holding_days)
    elif filter_method == 'probability':
        # Already applied above
        pass
    elif filter_method != 'none':
        print(f"Unknown filtering method: {filter_method}, no filtering applied")

    return predictions, true_returns, dates, raw_predictions, hidden_states

def compare_filtering_strategies(original_regimes, returns, dates, transaction_cost=0.001, save_path=None):
    """
    Compare performance of various regime filtering strategies

    Args:
        original_regimes: Original regime predictions (1=Bull, 0=Bear)
        returns: Actual returns
        dates: Date information
        transaction_cost: Transaction cost
        save_path: Path to save results graph (None if not saving)

    Returns:
        results: Results for each strategy
        summary_df: Summary table
    """
    # Define filtering strategies
    strategies = {
        "Original": original_regimes,
        "MA(5)": apply_regime_smoothing(original_regimes, method="ma", window=5),
        "MA(10)": apply_regime_smoothing(original_regimes, method="ma", window=10),
        "EMA(10)": apply_regime_smoothing(original_regimes, method="exp", window=10),
        "Confirm(3)": apply_confirmation_rule(original_regimes, confirmation_days=3),
        "Confirm(5)": apply_confirmation_rule(original_regimes, confirmation_days=5),
        "MinHold(10)": apply_minimum_holding_period(original_regimes, min_holding_days=10),
        "MinHold(20)": apply_minimum_holding_period(original_regimes, min_holding_days=20),
    }

    # Storage for results
    results = {}

    # Evaluate each strategy
    for name, regimes in strategies.items():
        # Create results
        df = pd.DataFrame({
            'Date': dates,
            'Regime': regimes.flatten(),
            'Return': returns.flatten()
        })

        # Detect regime changes (when trades occur)
        df['Regime_Change'] = df['Regime'].diff().fillna(0) != 0

        # First entry also counts as a trade
        df.loc[0, 'Regime_Change'] = df.loc[0, 'Regime'] == 1

        # Calculate transaction costs (applied whenever regime changes)
        df['Transaction_Cost'] = np.where(df['Regime_Change'], transaction_cost * 100, 0)

        # Calculate strategy returns considering transaction costs
        df['Strategy_Return'] = df['Regime'] * df['Return'] - df['Transaction_Cost']

        # Calculate cumulative returns
        df['Cum_Market'] = (1 + df['Return']/100).cumprod() - 1
        df['Cum_Strategy'] = (1 + df['Strategy_Return']/100).cumprod() - 1

        # Calculate key metrics
        market_return = df['Cum_Market'].iloc[-1] * 100
        strategy_return = df['Cum_Strategy'].iloc[-1] * 100
        n_trades = df['Regime_Change'].sum()
        total_cost = df['Transaction_Cost'].sum()

        # Calculate maximum drawdown
        df['Strategy_Peak'] = df['Cum_Strategy'].cummax()
        df['Strategy_Drawdown'] = (df['Cum_Strategy'] - df['Strategy_Peak']) / (1 + df['Strategy_Peak']) * 100
        max_drawdown = df['Strategy_Drawdown'].min()

        # Store results
        results[name] = {
            'cum_return': strategy_return,
            'n_trades': n_trades,
            'total_cost': total_cost,
            'max_drawdown': max_drawdown,
            'df': df
        }

    # Create comparison chart
    plt.figure(figsize=(15, 12))

    # Compare cumulative returns
    plt.subplot(3, 1, 1)
    for name, result in results.items():
        plt.plot(result['df']['Cum_Strategy'] * 100, label=name)
    plt.plot(results['Original']['df']['Cum_Market'] * 100, label='Market', color='black', linestyle='--')
    plt.title('Cumulative Returns of Different Filtering Strategies')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)

    # Compare number of trades
    plt.subplot(3, 1, 2)
    names = list(results.keys())
    trade_counts = [results[name]['n_trades'] for name in names]
    plt.bar(names, trade_counts)
    plt.title('Number of Trades by Strategy')
    plt.ylabel('Number of Trades')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    # Compare returns vs maximum drawdowns
    plt.subplot(3, 1, 3)
    returns = [results[name]['cum_return'] for name in names]
    drawdowns = [results[name]['max_drawdown'] for name in names]

    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width/2, returns, width, label='Cumulative Return (%)')
    plt.bar(x + width/2, drawdowns, width, label='Maximum Drawdown (%)')
    plt.title('Returns vs. Maximum Drawdowns')
    plt.xticks(x, names, rotation=45)
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    # Create summary table
    summary_data = {
        'Strategy': names,
        'Cumulative Return (%)': [results[name]['cum_return'] for name in names],
        'Number of Trades': [results[name]['n_trades'] for name in names],
        'Total Cost (%)': [results[name]['total_cost'] for name in names],
        'Maximum Drawdown (%)': [results[name]['max_drawdown'] for name in names]
    }

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.sort_values('Cumulative Return (%)', ascending=False).to_string(index=False))

    return results, summary_df

def visualize_filtered_vs_original(results_df, raw_predictions, filtered_predictions, title, save_path=None):
    """
    Visualize comparison between original and filtered regimes

    Args:
        results_df: Results dataframe
        raw_predictions: Original regime predictions
        filtered_predictions: Filtered regime predictions
        title: Chart title
        save_path: Path to save (None if not saving)
    """
    # Calculate trade points for original regimes
    raw_regime = raw_predictions.flatten()
    raw_changes = np.diff(raw_regime, prepend=raw_regime[0])
    raw_trade_points = np.where(raw_changes != 0)[0]

    # Calculate trade points for filtered regimes
    filtered_regime = filtered_predictions.flatten()
    filtered_changes = np.diff(filtered_regime, prepend=filtered_regime[0])
    filtered_trade_points = np.where(filtered_changes != 0)[0]

    # Calculate cumulative returns
    raw_cum_return = (1 + results_df['Return']/100 * raw_regime).cumprod() - 1
    filtered_cum_return = results_df['Cum_Strategy']
    market_cum_return = results_df['Cum_Market']

    # Create visualization (3 subplots)
    plt.figure(figsize=(15, 12))

    # 1. Compare cumulative returns
    plt.subplot(3, 1, 1)
    plt.plot(market_cum_return * 100, label='Market', color='gray')
    plt.plot(raw_cum_return * 100, label='Original Strategy', color='red', linestyle='--')
    plt.plot(filtered_cum_return * 100, label='Filtered Strategy', color='blue')
    plt.title(f'{title} - Cumulative Returns Comparison')
    plt.legend()
    plt.ylabel('Return (%)')
    plt.grid(True)

    # 2. Compare regime signals
    plt.subplot(3, 1, 2)
    plt.plot(raw_regime, label='Original Regime', color='red', alpha=0.6)
    plt.plot(filtered_regime, label='Filtered Regime', color='blue')

    # Mark trade points
    for point in raw_trade_points:
        plt.axvline(x=point, color='red', alpha=0.3, linestyle='--')
    for point in filtered_trade_points:
        plt.axvline(x=point, color='blue', alpha=0.3, linestyle='-')

    plt.title('Regime Signals Comparison')
    plt.ylabel('Regime (1=Bull, 0=Bear)')
    plt.legend()
    plt.grid(True)

    # 3. Compare cumulative transaction costs
    plt.subplot(3, 1, 3)

    # Calculate original transaction costs
    raw_costs = np.zeros_like(raw_regime)
    raw_costs[raw_trade_points] = results_df['Transaction_Cost'].iloc[0]  # Use first transaction cost value
    if raw_regime[0] == 1:  # If first day is Bull, count as trade
        raw_costs[0] = results_df['Transaction_Cost'].iloc[0]

    # Get filtered transaction costs (already in results_df)
    filtered_costs = results_df['Transaction_Cost'].values

    # Calculate cumulative costs
    raw_cum_costs = np.cumsum(raw_costs)
    filtered_cum_costs = np.cumsum(filtered_costs)

    plt.plot(raw_cum_costs, label=f'Original Cumulative Cost ({len(raw_trade_points)} trades)', color='red', alpha=0.6)
    plt.plot(filtered_cum_costs, label=f'Filtered Cumulative Cost ({len(filtered_trade_points)} trades)', color='blue')
    plt.title('Cumulative Transaction Costs')
    plt.ylabel('Cost (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    # Print improvement statistics
    trade_reduction = len(raw_trade_points) - len(filtered_trade_points)
    trade_reduction_pct = trade_reduction / len(raw_trade_points) * 100 if len(raw_trade_points) > 0 else 0
    
    raw_final_return = raw_cum_return.iloc[-1] * 100
    filtered_final_return = filtered_cum_return.iloc[-1] * 100
    return_improvement = filtered_final_return - raw_final_return
    
    print(f"\nFiltering Improvement Statistics:")
    print(f"  Original number of trades: {len(raw_trade_points)}")
    print(f"  Filtered number of trades: {len(filtered_trade_points)}")
    print(f"  Trade reduction: {trade_reduction} ({trade_reduction_pct:.2f}%)")
    print(f"  Original final return: {raw_final_return:.2f}%")
    print(f"  Filtered final return: {filtered_final_return:.2f}%")
    print(f"  Return improvement: {return_improvement:.2f}%")
    
    return {
        'trade_reduction': trade_reduction,
        'trade_reduction_pct': trade_reduction_pct,
        'raw_final_return': raw_final_return,
        'filtered_final_return': filtered_final_return,
        'return_improvement': return_improvement
    }

def find_optimal_filtering(config, data_path, save_path=None):
    """
    Find optimal regime filtering parameters by trying various methods

    Args:
        config: Configuration object
        data_path: Data file path
        save_path: Results save path (None if not saving)

    Returns:
        optimal_params: Optimal filtering parameters
        results: Results for each strategy
    """

    from torch.utils.data import DataLoader
    from sklearn.cluster import KMeans
    
    from ..data.dataset import RegimeMambaDataset
    from ..models.mamba_model import create_model_from_config
    from .clustering import extract_hidden_states, predict_regimes
    
    # Load data
    data = pd.read_csv(data_path)
    data = data.iloc[2:]
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data['returns'] = data['returns'] * 100
    data["dd_10"] = data["dd_10"] * 100
    data["sortino_20"] = data["sortino_20"] * 100
    data["sortino_60"] = data["sortino_60"] * 100

    # Identify date column
    date_col = 'Price' if 'Price' in data.columns else 'Date'

    print("Loading model...")
    # Load model
    model = create_model_from_config(config)
    if hasattr(config, 'model_path') and config.model_path:
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint: {config.model_path}")
    
    model.to(config.device)
    model.eval()

    # Split data by period
    print("Splitting data...")
    train_end = '2009-12-31'
    test_start = '2010-01-01'
    test_end = '2015-12-31'
    
    train_data = data[(data[date_col] >= '2000-01-01') & (data[date_col] <= train_end)]
    test_data = data[(data[date_col] >= test_start) & (data[date_col] <= test_end)]

    # Create datasets and dataloaders
    print("Creating datasets...")
    train_dataset = RegimeMambaDataset(data_path, seq_len=config.seq_len, mode="valid")  # valid mode uses 2000-2009 data
    test_dataset = RegimeMambaDataset(data_path, seq_len=config.seq_len, mode="test")    # test mode uses post-2010 data

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Identify regimes using training data
    print("Identifying regimes with training data...")
    train_hidden, train_returns, _ = extract_hidden_states(model, train_loader, config)
    kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
    clusters = kmeans.fit_predict(train_hidden)

    # Identify Bull regime (cluster with highest average return)
    cluster_returns = {}
    for i in range(config.n_clusters):
        cluster_mask = (clusters == i)
        avg_return = train_returns[cluster_mask].mean()
        cluster_returns[i] = avg_return

    bull_regime = max(cluster_returns, key=cluster_returns.get)
    print(f"Bull regime cluster: {bull_regime}")

    # Apply to test data
    print("Applying regime prediction to test data...")
    test_predictions, test_returns, test_dates = predict_regimes(
        model, test_loader, kmeans, bull_regime, config
    )

    # Compare various filtering strategies
    print("Comparing filtering strategies...")
    results, summary = compare_filtering_strategies(
        test_predictions, test_returns, test_dates, 
        transaction_cost=config.transaction_cost,
        save_path=save_path
    )

    # Find optimal strategy
    optimal_strategy = summary.sort_values('Cumulative Return (%)', ascending=False).iloc[0]['Strategy']
    print(f"\nOptimal filtering strategy: {optimal_strategy}")

    # Determine parameters for optimal strategy
    if optimal_strategy == 'Original':
        optimal_params = {'filter_method': 'none'}
    elif optimal_strategy.startswith('MA'):
        window = int(optimal_strategy.split('(')[1].split(')')[0])
        optimal_params = {'filter_method': 'smoothing', 'window': window}
    elif optimal_strategy.startswith('EMA'):
        window = int(optimal_strategy.split('(')[1].split(')')[0])
        optimal_params = {'filter_method': 'exp', 'window': window}
    elif optimal_strategy.startswith('Confirm'):
        days = int(optimal_strategy.split('(')[1].split(')')[0])
        optimal_params = {'filter_method': 'confirmation', 'confirmation_days': days}
    elif optimal_strategy.startswith('MinHold'):
        days = int(optimal_strategy.split('(')[1].split(')')[0])
        optimal_params = {'filter_method': 'minimum_holding', 'min_holding_days': days}
    else:
        optimal_params = {'filter_method': 'none'}

    print("Optimal filtering parameters:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value}")

    # Apply optimal parameters and visualize
    if optimal_params['filter_method'] != 'none':
        if optimal_params['filter_method'] == 'smoothing':
            filtered_predictions = apply_regime_smoothing(
                test_predictions, 
                method="ma", 
                window=optimal_params['window']
            )
        elif optimal_params['filter_method'] == 'exp':
            filtered_predictions = apply_regime_smoothing(
                test_predictions, 
                method="exp", 
                window=optimal_params['window']
            )
        elif optimal_params['filter_method'] == 'confirmation':
            filtered_predictions = apply_confirmation_rule(
                test_predictions, 
                confirmation_days=optimal_params['confirmation_days']
            )
        elif optimal_params['filter_method'] == 'minimum_holding':
            filtered_predictions = apply_minimum_holding_period(
                test_predictions, 
                min_holding_days=optimal_params['min_holding_days']
            )
            
        # Evaluate performance using filtered regimes
        from .strategy import evaluate_regime_strategy
        filtered_results_df, _ = evaluate_regime_strategy(
            filtered_predictions.reshape(-1, 1), 
            test_returns, 
            test_dates, 
            transaction_cost=config.transaction_cost
        )
        
        # Compare original vs filtered regimes
        if save_path:
            save_dir = os.path.dirname(save_path)
            optimal_viz_path = os.path.join(save_dir, 'optimal_filtering_comparison.png')
        else:
            optimal_viz_path = None
            
        visualize_filtered_vs_original(
            filtered_results_df,
            test_predictions,
            filtered_predictions.reshape(-1, 1),
            f"Optimal Filtering Strategy: {optimal_strategy}",
            save_path=optimal_viz_path
        )

    return optimal_params, results