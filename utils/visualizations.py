"""Visualization utilities for analyzing language model layer-wise behavior.

This module provides functions for visualizing various metrics across model layers,
including entropy, perplexity, and token probability trajectories. It also includes
utilities for saving prediction results to CSV files.

Typical usage example:

    results = compute_layer_metrics(model, text)
    plot_entropy_curves(results, "Layer-wise Entropy", "entropy_plot.png")
    plot_token_prob_trajectory(trajectories, "Token Probabilities", "probs_plot.png")
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_entropy_curves(results, title, output_path):
    """Plots layer-wise entropy curves for multiple sequences.
    
    Args:
        results (list): List of layer-wise metric values for each sequence.
            Each element should be a list or array of numerical values.
        title (str): Title for the plot.
        output_path (str): Path where the plot will be saved.
    
    Returns:
        None
    
    Note:
        Creates a line plot where each line represents the metric values
        across layers for a single sequence.
    """
    plt.figure(figsize=(10, 6))
    
    for i, values in enumerate(results):
        plt.plot(values)
    
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_token_prob_trajectory(trajectories, title, output_path):
    """Plots token probability trajectories across model layers.
    
    Args:
        trajectories (list): List of token probability trajectories.
            Each trajectory can be either a dict mapping layer indices to probabilities,
            or a list/array of probabilities.
        title (str): Title for the plot.
        output_path (str): Path where the plot will be saved.
    
    Returns:
        None
    
    Note:
        Handles both dictionary and list-based trajectory formats.
        For dictionary trajectories, assumes keys are layer indices
        (either as strings or integers).
    """
    plt.figure(figsize=(12, 6))
    
    for i, trajectory in enumerate(trajectories):
        if isinstance(trajectory, dict):
            # Convert all keys to strings for consistent handling
            trajectory = {str(k): v for k, v in trajectory.items()}
            # Sort by numeric value if possible, otherwise lexicographically
            try:
                layers = sorted(trajectory.keys(), key=lambda x: int(x))
            except ValueError:
                layers = sorted(trajectory.keys())
            probs = [trajectory[layer] for layer in layers]
        else:
            probs = trajectory
            
        plt.plot(probs)
    
    plt.xlabel('Layer')
    plt.ylabel('Token Probability')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_predictions_to_csv(results, output_path):
    """Saves model predictions to a CSV file with support for comparative results."""
    if not results:
        return False
        
    data = []
    for sentence, predictions in results:
        # Extract analysis type from sentence if present
        analysis_type = "standard"
        if " (base)" in sentence:
            analysis_type = "base"
            sentence = sentence.replace(" (base)", "")
        elif " (control)" in sentence:
            analysis_type = "control"
            sentence = sentence.replace(" (control)", "")
            
        # Handle predictions in standard dict format
        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, dict) and 'token' in pred and 'probability' in pred:
                    data.append({
                        "Sentence": sentence,
                        "Token": str(pred['token']),
                        "Probability": float(pred['probability']),
                        "Analysis": analysis_type
                    })
                    
    if data:
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        return True
    return False