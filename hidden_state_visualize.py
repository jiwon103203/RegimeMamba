from regime_mamba.evaluate.clustering import *
from regime_mamba.models.mamba_model import *
from regime_mamba.data.dataset import DateRangeRegimeMambaDataset, create_date_range_dataloader
from regime_mamba.config.config import RegimeMambaConfig
from regime_mamba.utils.utils import set_seed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any, Optional, Callable
import yaml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_args():
    """ Parse command line arguments """

    parser = argparse.ArgumentParser(description="Hidden state visualization")

    parser.add_argument("--model_path", type=str, default="model.pt", help="Model path")
    parser.add_argument("--data_path", type=str, default="data.csv", help="Data path")
    parser.add_argument("--results_dir", type=str, default="./hidden_state", help="Results directory")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Dataset Period parameters
    parser.add_argument("--start_date", type=str, default="1990-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2023-12-31", help="End date")
    parser.add_argument("--preprocessed", type=bool, default=False, help="Preprocessed data")

    # Model parameters
    parser.add_argument("--input_dim", type=int, default=4, help="Input dimension")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--d_state", type=int, default=128, help="State dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--target_type", type=str, default="average", help="Target type")
    parser.add_argument("--target_horizon", type=int, default=5, help="Target horizon")
    parser.add_argument("--cluster_method", type=str, default="cosine_kmeans", help="Clustering method")

    return parser.parse_args()

def load_config(args) -> RegimeMambaConfig:
    """ Load configuration from file or command-line arguments

    Args:
        args: Command-line arguments
    

    Returns:
        RegimeMambaConfig: Configuration object
    """

    config = RegimeMambaConfig()

    if args.config:
        with open(args.config,"r") as f:
            yaml_config = yaml.safe_load(f)

            for key, values in yaml_config.items():
                if hasattr(config, key):
                    setattr(config, key, values)
    
    # Override with command-line arguments if provided
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)

    return config

def load_model(args) -> TimeSeriesMamba:
    """ Load pre-trained model

    Args:
        args: Command-line arguments

    Returns:
        TimeSeriesMamba: Pre-trained model
    """

    model = create_model_from_config(load_config(args))

    return model

def load_dataloader(args) -> DateRangeRegimeMambaDataset:
    """ Load data
    
    Args:
        args: Command-line arguments

    Returns:
        DateRangeRegimeMambaDataset: Data loader
    """

    data = pd.read_csv(args.data_path)
    dataloader = create_date_range_dataloader(data = data, seq_len = args.seq_len, batch_size = args.batch_size, start_date=args.start_date, end_date=args.end_date, target_type=args.target_type,target_horizon=args.target_horizon, preprocessed=args.preprocessed)

    return dataloader

def visualize_hidden_states(hidden_states: np.ndarray, returns: np.ndarray, results_dir: str):
    """ Visualize hidden states with various dimensionality reduction techniques

    Args:
        hidden_states: Hidden states (n_samples, hidden_size)
        returns: Returns values for coloring
        results_dir: Results directory
    """
    import os
    from sklearn.manifold import TSNE
    try:
        import umap
        has_umap = True
    except ImportError:
        print("UMAP not installed. Install it with 'pip install umap-learn'")
        has_umap = False
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. PCA
    pca = PCA(n_components=2)
    pca_analysis = pca.fit_transform(hidden_states)
    print("PCA 설명력:", pca.explained_variance_ratio_)

    # Scatter plot with returns-based coloring - PCA
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        pca_analysis[:, 0], 
        pca_analysis[:, 1],
        c=returns,  # Color by returns
        cmap='coolwarm',  # Heat map style colormap
        s=50,  # Point size
        alpha=0.8,  # Transparency
        edgecolors='k',  # Black edge for better visibility
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Returns (%)', rotation=270, labelpad=20)
    
    # Enhance plot appearance
    ax.set_title("Hidden States - PCA Projection Colored by Returns", fontsize=14)
    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/hidden_states_pca.png", dpi=300)
    plt.close()
    
    # 2. 3D scatter plot with returns as color - PCA
    from mpl_toolkits.mplot3d import Axes3D
    
    if hidden_states.shape[1] >= 3:
        # Get 3 components if available
        pca3d = PCA(n_components=3)
        pca3d_analysis = pca3d.fit_transform(hidden_states)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter3d = ax.scatter(
            pca3d_analysis[:, 0],
            pca3d_analysis[:, 1],
            pca3d_analysis[:, 2],
            c=returns,
            cmap='coolwarm',
            s=50,
            alpha=0.8
        )
        
        cbar = plt.colorbar(scatter3d)
        cbar.set_label('Returns (%)', rotation=270, labelpad=20)
        
        ax.set_title("Hidden States - 3D PCA Projection Colored by Returns", fontsize=14)
        ax.set_xlabel("PCA Component 1", fontsize=12)
        ax.set_ylabel("PCA Component 2", fontsize=12)
        ax.set_zlabel("PCA Component 3", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/hidden_states_pca_3d.png", dpi=300)
        plt.close()
    
    # 3. Create a contour plot (heatmap) showing returns distribution - PCA
    if len(pca_analysis) > 20:  # Need enough points for a meaningful contour
        from scipy.interpolate import griddata
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create a grid for contour plotting
        x = pca_analysis[:, 0]
        y = pca_analysis[:, 1]
        
        # Create a grid
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate returns values onto the grid
        zi = griddata((x, y), returns, (xi, yi), method='cubic')
        
        # Create contour plot
        contour = ax.contourf(xi, yi, zi, 15, cmap='coolwarm', alpha=0.8)
        
        # Overlay original points
        ax.scatter(x, y, color='black', s=10, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label('Returns (%)', rotation=270, labelpad=20)
        
        # Set labels and title
        ax.set_title("Returns Heatmap in PCA Space", fontsize=14)
        ax.set_xlabel("PCA Component 1", fontsize=12)
        ax.set_ylabel("PCA Component 2", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/hidden_states_pca_heatmap.png", dpi=300)
        plt.close()
    
    # 4. t-SNE visualization
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(hidden_states) - 1))
    tsne_results = tsne.fit_transform(hidden_states)
    
    # Scatter plot with returns-based coloring - t-SNE
    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1],
        c=returns,  # Color by returns
        cmap='coolwarm',  # Heat map style colormap
        s=50,  # Point size
        alpha=0.8,  # Transparency
        edgecolors='k',  # Black edge for better visibility
        linewidths=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Returns (%)', rotation=270, labelpad=20)
    
    # Enhance plot appearance
    ax.set_title("Hidden States - t-SNE Projection Colored by Returns", fontsize=14)
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/hidden_states_tsne.png", dpi=300)
    plt.close()
    
    # 5. t-SNE heatmap
    if len(tsne_results) > 20:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create a grid for contour plotting
        x = tsne_results[:, 0]
        y = tsne_results[:, 1]
        
        # Create a grid
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate returns values onto the grid
        zi = griddata((x, y), returns, (xi, yi), method='cubic')
        
        # Create contour plot
        contour = ax.contourf(xi, yi, zi, 15, cmap='coolwarm', alpha=0.8)
        
        # Overlay original points
        ax.scatter(x, y, color='black', s=10, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour)
        cbar.set_label('Returns (%)', rotation=270, labelpad=20)
        
        # Set labels and title
        ax.set_title("Returns Heatmap in t-SNE Space", fontsize=14)
        ax.set_xlabel("t-SNE Component 1", fontsize=12)
        ax.set_ylabel("t-SNE Component 2", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/hidden_states_tsne_heatmap.png", dpi=300)
        plt.close()
    
    # 6. UMAP visualization (if available)
    if has_umap:
        print("Running UMAP dimensionality reduction...")
        umap_reducer = umap.UMAP(random_state=42)
        umap_results = umap_reducer.fit_transform(hidden_states)
        
        # Scatter plot with returns-based coloring - UMAP
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(
            umap_results[:, 0], 
            umap_results[:, 1],
            c=returns,  # Color by returns
            cmap='coolwarm',  # Heat map style colormap
            s=50,  # Point size
            alpha=0.8,  # Transparency
            edgecolors='k',  # Black edge for better visibility
            linewidths=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Returns (%)', rotation=270, labelpad=20)
        
        # Enhance plot appearance
        ax.set_title("Hidden States - UMAP Projection Colored by Returns", fontsize=14)
        ax.set_xlabel("UMAP Component 1", fontsize=12)
        ax.set_ylabel("UMAP Component 2", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/hidden_states_umap.png", dpi=300)
        plt.close()
        
        # 7. UMAP heatmap
        if len(umap_results) > 20:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Create a grid for contour plotting
            x = umap_results[:, 0]
            y = umap_results[:, 1]
            
            # Create a grid
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)
            
            # Interpolate returns values onto the grid
            zi = griddata((x, y), returns, (xi, yi), method='cubic')
            
            # Create contour plot
            contour = ax.contourf(xi, yi, zi, 15, cmap='coolwarm', alpha=0.8)
            
            # Overlay original points
            ax.scatter(x, y, color='black', s=10, alpha=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(contour)
            cbar.set_label('Returns (%)', rotation=270, labelpad=20)
            
            # Set labels and title
            ax.set_title("Returns Heatmap in UMAP Space", fontsize=14)
            ax.set_xlabel("UMAP Component 1", fontsize=12)
            ax.set_ylabel("UMAP Component 2", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f"{results_dir}/hidden_states_umap_heatmap.png", dpi=300)
            plt.close()

def main():
    args = parse_args()

    set_seed(args.seed)

    model = load_model(args)
    dataloader = load_dataloader(args)

    hidden_states, returns, dates = extract_hidden_states(model, dataloader, config)
    
    visualize_hidden_states(hidden_states, returns, args.results_dir)