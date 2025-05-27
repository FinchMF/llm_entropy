"""Plotting utilities for visualization."""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Union, Optional, Tuple

def plot_metric_distribution(
    values: List[Union[List, np.ndarray]], 
    title: str,
    filepath: str,  # Changed from output_path to filepath
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Plot distribution of metric values across layers.
    
    Args:
        values: List of metric values from different sequences
        title: Plot title
        filepath: Path to save plot
        figsize: Figure dimensions (width, height)
    """
    # Validate input values
    if not values or all(v is None for v in values):
        raise ValueError("No valid values to plot")
        
    # Filter out None values and convert to numpy arrays
    valid_values = []
    for val in values:
        if val is not None:
            if isinstance(val, (list, np.ndarray)):
                valid_values.append(np.array(val))
            elif isinstance(val, torch.Tensor):
                valid_values.append(val.detach().cpu().numpy())
                
    if not valid_values:
        raise ValueError("No valid numerical values to plot")

    plt.figure(figsize=figsize)
    
    # Plot individual sequences
    for i, seq_values in enumerate(valid_values):
        plt.plot(seq_values, alpha=0.3, label=f'Sequence {i+1}' if i < 5 else '')
    
    # Compute statistics
    try:
        values_array = np.stack(valid_values)
        mean = np.mean(values_array, axis=0)
        std = np.std(values_array, axis=0)
        x = range(len(mean))
        
        # Plot mean and std
        plt.plot(x, mean, 'k-', linewidth=2, label='Mean')
        plt.fill_between(x, mean - std, mean + std, color='k', alpha=0.2, label='±1 STD')
    except Exception as e:
        plt.close()
        raise ValueError(f"Failed to compute statistics: {str(e)}")
    
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def plot_comparative_metrics(
    standard_values: np.ndarray,
    control_values: np.ndarray,
    title: str,
    filepath: str,  # Changed from output_path
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Plot comparison between standard and control metrics.
    
    Args:
        standard_values: Array of standard metric values
        control_values: Array of controlled metric values
        title: Plot title
        filepath: Path to save plot
        figsize: Figure dimensions
    """
    plt.figure(figsize=figsize)
    
    # Compute statistics for both sets
    std_mean = np.mean(standard_values, axis=0)
    std_std = np.std(standard_values, axis=0)
    ctrl_mean = np.mean(control_values, axis=0)
    ctrl_std = np.std(control_values, axis=0)
    x = range(len(std_mean))
    
    # Plot standard metrics
    plt.plot(x, std_mean, 'b-', label='Standard', linewidth=2)
    plt.fill_between(x, std_mean - std_std, std_mean + std_std,
                    color='b', alpha=0.2, label='Standard ±1 STD')
    
    # Plot control metrics
    plt.plot(x, ctrl_mean, 'r-', label='Control', linewidth=2)
    plt.fill_between(x, ctrl_mean - ctrl_std, ctrl_mean + ctrl_std,
                    color='r', alpha=0.2, label='Control ±1 STD')
    
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
