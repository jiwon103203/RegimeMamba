import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def extract_hidden_states(model, dataloader, config):
    """
    Extract hidden states and target returns from the model

    Args:
        model: Model to evaluate
        dataloader: Data loader
        config: Configuration object

    Returns:
        hidden_states: Extracted hidden states
        returns: Target returns
        dates: Date information
    """
    model.eval()
    hidden_states = []
    returns = []
    dates = []

    with torch.no_grad():
        for x, y, date, r in dataloader:
            x = x.to(config.device)
            _, hidden = model(x, return_hidden=True)
            hidden_states.append(hidden.cpu().numpy())
            returns.append(r.numpy().reshape(-1,1))
            dates.extend(date)

    hidden_states = np.vstack(hidden_states) # Stack by rows (n_samples, hidden_size)
    returns = np.vstack(returns) # Stack by rows (n_samples, 1)


    return hidden_states, returns, dates

def cosine_kmeans(hidden_states, n_clusters=2, random_state=42, max_iter=300):
    """
    KMeans clustering based on cosine similarity
    
    Args:
        hidden_states: Hidden states to cluster (n_samples, hidden_size)
        n_clusters: Number of clusters
        random_state: Random seed
        max_iter: Maximum number of iterations
        
    Returns:
        kmeans: Trained KMeans model
        normalized_states: Normalized hidden states
        clusters: Cluster assignment labels
    """
    # 1. Normalize data (L2 norm): Essential step for cosine similarity
    normalized_states = normalize(hidden_states, norm='l2')
    
    # 2. Apply standard KMeans (using Euclidean distance on normalized data is equivalent to cosine similarity)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, max_iter=max_iter)
    clusters = kmeans.fit_predict(normalized_states)
    
    return kmeans, normalized_states, clusters

def identify_bull_bear_regimes(hidden_states, returns, config):
    """
    Cluster hidden states and identify Bull/Bear regimes

    Args:
        hidden_states: Extracted hidden states
        returns: Target returns
        config: Configuration object

    Returns:
        kmeans: Trained KMeans model
        bull_regime: Bull regime cluster ID
    """
    if config.cluster_method == 'cosine_kmeans':
        kmeans, hidden_states, clusters = cosine_kmeans(hidden_states, n_clusters=config.n_clusters, random_state=42)
    
    elif config.cluster_method == 'kmeans':
        # K-Means clustering
        kmeans = KMeans(n_clusters=config.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(hidden_states)
    
    # Calculate silhouette score (clustering quality evaluation)
    try:
        silhouette_avg = silhouette_score(hidden_states, clusters)
        print(f"Silhouette score: {silhouette_avg:.3f}")
    except:
        print("Failed to calculate silhouette score")

    # Calculate average returns for each cluster
    cluster_returns = {}
    for i in range(config.n_clusters):
        cluster_mask = (clusters == i)
        avg_return = returns[cluster_mask].mean()
        cluster_returns[i] = avg_return

    # Choose the cluster with the highest average return as Bull regime
    bull_regime = max(cluster_returns, key=cluster_returns.get)
    if config.n_clusters == 3:
        
        bear_regime = min(cluster_returns, key=cluster_returns.get)

        print(f"Cluster average returns: {cluster_returns}")
        print(f"Bull regime cluster: {bull_regime}")
        print(f"Bear regime cluster: {bear_regime}")

        return kmeans, bull_regime, bear_regime

    print(f"Cluster average returns: {cluster_returns}")
    print(f"Bull regime cluster: {bull_regime}")

    return kmeans, bull_regime

def predict_regimes(model, dataloader, kmeans, bull_regime, config, bear_regime=None):
    """
    Predict regimes for test data

    Args:
        model: Model to evaluate
        dataloader: Data loader
        kmeans: Trained KMeans model
        bull_regime: Bull regime cluster ID
        config: Configuration object

    Returns:
        predictions: Predicted regimes (1=Bull, 0=Bear)
        true_returns: Actual returns
        dates: Date information
    """
    model.eval()
    predictions = []
    true_returns = []
    dates = []

    with torch.no_grad():
        for x, y, date, r in dataloader:
            x = x.to(config.device)
            _, hidden = model(x, return_hidden=True)
            hidden = hidden.cpu().numpy()

            # Cluster assignment
            cluster = kmeans.predict(hidden)

            if config.n_clusters == 2:
                # 1 for Bull regime, 0 otherwise
                regime_pred = np.where(cluster == bull_regime, 1, 0)
            elif config.n_clusters == 3:
                # 2 for Bull regime, 0 for Bear regime, 1 otherwise
                regime_pred = np.where(cluster == bull_regime, 2, 0)
                regime_pred = np.where(cluster == bear_regime, 0, regime_pred)

            predictions.extend(regime_pred)
            true_returns.extend(r.numpy())
            dates.extend(date)

    return np.array(predictions), np.array(true_returns), dates